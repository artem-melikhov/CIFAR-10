"""Measure the speed/quality tradeoff of an approximate separable conv2."""
import argparse,json,time,statistics,sys
import torch
from torch import nn
from torch.nn import functional as F
from alexnet_bottleneck import ROOT,make_model
from alexnet_deep_sweep import OUT,configure

class LowRankConv(nn.Module):
    def __init__(self,conv,rank):
        super().__init__()
        weight=conv.weight.detach().cpu();o,i,h,w=weight.shape
        matrix=weight.permute(0,2,1,3).reshape(o*h,i*w)
        u,s,vh=torch.linalg.svd(matrix,full_matrices=False)
        self.energy=float(s[:rank].square().sum()/s.square().sum())
        self.horizontal=nn.Conv2d(i,rank,(1,w),padding=(0,w//2),bias=False)
        self.vertical=nn.Conv2d(rank,o,(h,1),padding=(h//2,0),bias=True)
        with torch.no_grad():
            self.horizontal.weight.copy_((s[:rank].sqrt().view(-1,1)*vh[:rank]).reshape(rank,i,1,w))
            self.vertical.weight.copy_((u[:,:rank]*s[:rank].sqrt()).reshape(o,h,rank).permute(0,2,1).unsqueeze(-1))
            self.vertical.bias.copy_(conv.bias.detach().cpu())
        self.to(device=conv.weight.device,dtype=conv.weight.dtype)
    def forward(self,x):return self.vertical(self.horizontal(x))

def evaluate(model,data):
    model.eval();correct=0;loss=0.
    with torch.no_grad():
        for start in range(0,len(data['val_y']),256):
            x=data['val_x'][start:start+256].to('mps');y=data['val_y'][start:start+256].to('mps')
            logits=model(x);loss+=F.cross_entropy(logits,y,reduction='sum').item();correct+=(logits.argmax(1)==y).sum().item()
    return {'accuracy':correct/len(data['val_y']),'loss':loss/len(data['val_y'])}

def main():
    p=argparse.ArgumentParser();p.add_argument('--rank',type=int,default=0);p.add_argument('--all-convs',action='store_true');a=p.parse_args();torch.manual_seed(52)
    model=configure(make_model(),'metal');result={'rank':a.rank,'all_convs':a.all_convs}
    if a.rank:
        model.network[4]=LowRankConv(model.network[4],a.rank);result['retained_weight_energy']=model.network[4].energy
    if a.all_convs:
        result['other_energy']={}
        for i in [8,10,12]:
            model.network[i]=LowRankConv(model.network[i],a.rank);result['other_energy'][str(i)]=model.network[i].energy
    data=torch.load(ROOT/'data/alexnet_precision_inputs.pt',weights_only=True)
    result['initial_validation']=evaluate(model,data);print('initial',result,flush=True)
    mean=data['mean'].to('mps').view(1,3,1,1);std=data['std'].to('mps').view(1,3,1,1)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4);times=[];model.train()
    for step in range(40):
        torch.manual_seed(1000+step)
        x=data['train_x'][step*256:(step+1)*256].to('mps');y=data['train_y'][step*256:(step+1)*256].to('mps')
        torch.mps.synchronize();start=time.perf_counter()
        x=(x.float()-mean)/std
        optimizer.zero_grad();logits=model(x);loss=F.cross_entropy(logits,y);loss.backward();optimizer.step()
        loss.item();(logits.argmax(1)==y).sum().item();torch.mps.synchronize()
        times.append(time.perf_counter()-start)
    result.update(final_validation=evaluate(model,data),median_step_seconds=statistics.median(times[4:]),steps_seconds=times)
    OUT.mkdir(exist_ok=True);(OUT/(f'lowrank_{a.rank}'+('_all' if a.all_convs else '')+'.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
