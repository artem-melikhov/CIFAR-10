"""Controlled full-data fine-tuning comparison, separate experiment checkpoints."""
import argparse,json,sys,time,statistics
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from alexnet_bottleneck import ROOT,make_model
from alexnet_deep_sweep import OUT,configure
from alexnet_lowrank import LowRankConv
sys.path.insert(0,str(ROOT/'utils'))
from data_utils import images_to_tensor,train_dataset_split,calculate_mean_std,AutoAugumentationDataset,create_dataloader

def main():
 p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--rank',type=int,default=0);p.add_argument('--all-convs',action='store_true');p.add_argument('--epochs',type=int,default=3);a=p.parse_args()
 torch.manual_seed(52);OUT.mkdir(exist_ok=True);result={'case':a.case,'rank':a.rank,'all_convs':a.all_convs,'epochs':[],'batch':256,'workers':4,'lr':3e-4,'split_seed':52}
 model=configure(make_model(),'original' if a.case=='original_full' else 'combined' if a.case=='combined_full' else 'metal')
 if a.rank:
  for i in ([4,8,10,12] if a.all_convs else [4]):model.network[i]=LowRankConv(model.network[i],a.rank)
 result['parameters']=sum(p.numel() for p in model.parameters())
 x,y=images_to_tensor(str(ROOT/'data/train'),str(ROOT/'data/trainLabels.csv'))
 train,val=train_dataset_split(TensorDataset(x,y),.8,random_seed=52);mean,std=calculate_mean_std(train)
 vx=x[val.indices].float();vx.sub_(mean.view(1,3,1,1)).div_(std.view(1,3,1,1))
 loader=create_dataloader(AutoAugumentationDataset(train),256,shuffle=True,random_seed=52,num_workers=4)
 vl=create_dataloader(TensorDataset(vx,y[val.indices]),256,shuffle=False,random_seed=52)
 mean=mean.to('mps').view(1,3,1,1);std=std.to('mps').view(1,3,1,1)
 optimizer=torch.optim.Adam(model.parameters(),lr=3e-4,fused=True if a.case=='combined_full' else None)
 def save():(OUT/f'{a.case}.json').write_text(json.dumps(result,indent=2))
 def validation():
  model.eval();correct=0;loss=0.;start=time.perf_counter()
  with torch.no_grad():
   for bx,by in vl:
    bx,by=bx.to('mps'),by.to('mps');logits=model(bx)
    loss+=F.cross_entropy(logits,by,reduction='sum').item();correct+=(logits.argmax(1)==by).sum().item()
  torch.mps.synchronize()
  return {'accuracy':correct/len(val),'loss':loss/len(val),'seconds':time.perf_counter()-start}
 result['initial_validation']=validation();save();print('INITIAL',a.case,result['initial_validation'],flush=True)
 for epoch in range(a.epochs):
  torch.manual_seed(52+epoch);model.train();torch.mps.synchronize();started=time.perf_counter();peak=0;correct=0;loss_sum=0.
  for index,(bx,by) in enumerate(loader):
   bx,by=bx.to('mps'),by.to('mps');bx=(bx.float()-mean)/std
   optimizer.zero_grad(set_to_none=True);logits=model(bx);loss=F.cross_entropy(logits,by)
   peak=max(peak,torch.mps.driver_allocated_memory());loss.backward();optimizer.step()
   loss_sum+=loss.item()*len(by);correct+=(logits.argmax(1)==by).sum().item()
  torch.mps.synchronize();elapsed=time.perf_counter()-started
  row={'epoch':epoch+1,'train_seconds':elapsed,'images_per_second':len(train)/elapsed,'train_loss':loss_sum/len(train),'train_accuracy':correct/len(train),'peak_driver_bytes':peak,'validation':validation()}
  result['epochs'].append(row);save();print('EPOCH',a.case,row,flush=True)
 checkpoints=OUT/'checkpoints';checkpoints.mkdir(exist_ok=True)
 torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},checkpoints/f'{a.case}.pth')
 result['median_train_seconds']=statistics.median(row['train_seconds'] for row in result['epochs']);result['status']='complete';save()
 if loader._iterator:loader._iterator._shutdown_workers()

if __name__=='__main__':main()
