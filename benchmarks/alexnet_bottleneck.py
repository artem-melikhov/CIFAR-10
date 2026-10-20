"""Attribute AlexNet MPS latency to layers and verify suspected bottlenecks."""
import argparse
import ast
import json
from pathlib import Path
import statistics
import time

import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks/alexnet_bottleneck'


class LRN2D(nn.Module):
    """Same channel window as LRN, expressed through 2D average pooling."""
    def __init__(self, original):
        super().__init__()
        self.size,self.alpha,self.beta,self.k=original.size,original.alpha,original.beta,original.k

    def forward(self,x):
        n,c,h,w=x.shape
        square=x.square().reshape(n,1,c,h*w)
        square=F.pad(square,(0,0,self.size//2,(self.size-1)//2))
        mean=F.avg_pool2d(square,(self.size,1),stride=1).reshape(n,c,h,w)
        return x/(mean*self.alpha+self.k).pow(self.beta)


class LRNShift(nn.Module):
    """Same channel sum, explicitly adding each shifted channel slice."""
    def __init__(self,original):
        super().__init__()
        self.size,self.alpha,self.beta,self.k=original.size,original.alpha,original.beta,original.k

    def forward(self,x):
        c=x.shape[1]
        square=F.pad(x.square(),(0,0,0,0,self.size//2,(self.size-1)//2))
        total=square[:,:c]
        for offset in range(1,self.size):
            total=total+square[:,offset:offset+c]
        return x/(total*(self.alpha/self.size)+self.k).pow(self.beta)


def make_model():
    notebook=json.loads((ROOT/'notebooks/AlexNet.ipynb').read_text())
    node=next(n for c in notebook['cells'] if c['cell_type']=='code'
              for n in ast.parse(''.join(c['source'])).body
              if isinstance(n,ast.ClassDef) and n.name=='AlexNet')
    scope={'torch':torch,'nn':nn}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'AlexNet.ipynb:model','exec'),scope)
    model=scope['AlexNet'](3,10).to('mps')
    weights=torch.load(ROOT/'notebooks/models/alexnet.pth',map_location='cpu',weights_only=True)
    model.load_state_dict(weights)
    return model


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--case',default='layers',choices=['layers','baseline','no_lrn1','no_lrn2','no_lrn','lrn2d','lrn_shift','metal_lrn','channels_last'])
    p.add_argument('--repeats',type=int,default=7)
    args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(52)
    model=make_model()
    if args.case.startswith('no_lrn'):
        for index in ([2] if args.case=='no_lrn1' else [6] if args.case=='no_lrn2' else [2,6]):
            model.network[index]=nn.Identity()
    if args.case in ['lrn2d','lrn_shift']:
        cls=LRN2D if args.case=='lrn2d' else LRNShift
        for index in [2,6]:model.network[index]=cls(model.network[index])
    if args.case=='metal_lrn':
        from alexnet_metal_lrn import MetalLRN
        for index in [2,6]:model.network[index]=MetalLRN(model.network[index])
    if args.case=='channels_last':model.to(memory_format=torch.channels_last)
    data=torch.load(ROOT/'data/alexnet_precision_inputs.pt',weights_only=True)
    mean=data['mean'].to('mps').view(1,-1,1,1)
    std=data['std'].to('mps').view(1,-1,1,1)
    raw=data['train_x'][:256].to('mps')
    x=((raw.float()-mean)/std).detach()
    if args.case=='channels_last':x=x.contiguous(memory_format=torch.channels_last)
    y=data['train_y'][:256].to('mps')
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    loss_fn=nn.CrossEntropyLoss()
    model.train()
    totals=[]
    layer_times={}
    handles=[]
    starts={}
    outputs={}
    phase_times=[]
    def sync():torch.mps.synchronize()
    def pre(key,phase):
        def hook(*unused):
            sync();starts[key,phase]=time.perf_counter()
        return hook
    def post(key,phase):
        def hook(*unused):
            sync()
            layer_times.setdefault(key,{'forward':[],'backward':[]})[phase].append(time.perf_counter()-starts[key,phase])
            if phase=='forward':outputs[key]=list(unused[-1].shape)
        return hook
    if args.case=='layers':
        # Give the first module a differentiable input so its full-backward hook
        # encloses its actual backward instead of firing at the output boundary.
        x.requires_grad_(True)
        for index,layer in enumerate(model.network):
            key=f'{index:02d}:{type(layer).__name__}'
            handles += [layer.register_forward_pre_hook(pre(key,'forward')),
                        layer.register_forward_hook(post(key,'forward')),
                        layer.register_full_backward_pre_hook(pre(key,'backward')),
                        layer.register_full_backward_hook(post(key,'backward'))]
    for step in range(3+args.repeats):
        optimizer.zero_grad()
        if x.requires_grad:x.grad=None
        sync();start=time.perf_counter()
        logits=model(x)
        loss=loss_fn(logits,y)
        sync();forward=time.perf_counter()-start
        start_backward=time.perf_counter()
        loss.backward()
        sync();backward=time.perf_counter()-start_backward
        start_adam=time.perf_counter()
        optimizer.step()
        sync();adam=time.perf_counter()-start_adam
        elapsed=time.perf_counter()-start
        if step>=3:
            totals.append(elapsed)
            phase_times.append({'forward_loss_seconds':forward,'backward_seconds':backward,'adam_seconds':adam})
        elif step==2:
            layer_times.clear()
        print(args.case,step,round(elapsed,6),flush=True)
    result={'case':args.case,'batch_size':256,'median_step_seconds':statistics.median(totals),
            'steps_seconds':totals,'phases':phase_times,
            'median_phases':{k:statistics.median(row[k] for row in phase_times) for k in phase_times[0]},
            'driver_bytes':torch.mps.driver_allocated_memory(),'tensor_bytes':torch.mps.current_allocated_memory()}
    if args.case=='layers':
        result['layers']=[{'name':key,'shape':outputs[key],
                           'forward_ms':1000*statistics.median(times['forward']),
                           'backward_ms':1000*statistics.median(times['backward'])}
                          for key,times in layer_times.items()]
    (OUT/f'{args.case}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    for handle in handles:handle.remove()


if __name__=='__main__':main()
