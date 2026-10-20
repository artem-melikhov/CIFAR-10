"""Time the real uint8/AutoAugment pipeline before/after equivalent fused LRN."""
import argparse,json,sys,time,statistics
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import TensorDataset
from alexnet_bottleneck import ROOT,OUT,make_model
from alexnet_metal_lrn import MetalLRN
sys.path.insert(0,str(ROOT/'utils'))
from data_utils import images_to_tensor,train_dataset_split,calculate_mean_std,AutoAugumentationDataset,create_dataloader

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['baseline','metal_lrn'],required=True)
    args=parser.parse_args();torch.manual_seed(52)
    x,y=images_to_tensor(str(ROOT/'data/train'),str(ROOT/'data/trainLabels.csv'))
    train,_=train_dataset_split(TensorDataset(x,y),.8,random_seed=52)
    mean,std=calculate_mean_std(train)
    mean=mean.to('mps').view(1,3,1,1);std=std.to('mps').view(1,3,1,1)
    loader=create_dataloader(AutoAugumentationDataset(train),256,shuffle=True,random_seed=52,num_workers=4)
    model=make_model().train()
    if args.case=='metal_lrn':
        for i in [2,6]:model.network[i]=MetalLRN(model.network[i])
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    times=[];waits=[]
    iterator=iter(loader)
    for index in range(len(loader)):
        start=time.perf_counter();bx,by=next(iterator);wait=time.perf_counter()-start
        bx=(bx.to('mps').float()-mean)/std;by=by.to('mps')
        optimizer.zero_grad();logits=model(bx);loss=nn.functional.cross_entropy(logits,by)
        loss.backward();optimizer.step()
        loss.item();(logits.argmax(1)==by).sum().item()
        elapsed=time.perf_counter()-start
        if index>=8:times.append(elapsed);waits.append(wait)
        if index%40==0:print(args.case,index,round(elapsed,4),flush=True)
    result={'case':args.case,'batch_size':256,'workers':4,'warmup_batches':8,
            'measured_batches':len(times),'measured_images':len(train)-8*256,
            'seconds':sum(times),'images_per_second':(len(train)-8*256)/sum(times),
            'median_step_seconds':statistics.median(times),'loader_wait_seconds':sum(waits),
            'loader_wait_fraction':sum(waits)/sum(times),'steps_seconds':times,
            'driver_bytes':torch.mps.driver_allocated_memory()}
    (OUT/f'pipeline_{args.case}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='steps_seconds'}),flush=True)
    if loader._iterator:loader._iterator._shutdown_workers()

if __name__=='__main__':main()
