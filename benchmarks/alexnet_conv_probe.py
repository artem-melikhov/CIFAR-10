"""Isolate conv2: layouts, dtype, forward, input and parameter gradients."""
import argparse,json,time,statistics,contextlib
import torch
from torch import nn
from alexnet_deep_sweep import ConvRows,ConvUnfold,OUT

def main():
 p=argparse.ArgumentParser();p.add_argument('--layout',choices=['nchw','nhwc'],default='nchw');p.add_argument('--dtype',choices=['fp32','fp16','bf16'],default='fp32');p.add_argument('--impl',choices=['native','rows','unfold'],default='native');a=p.parse_args()
 torch.manual_seed(52);m=nn.Conv2d(96,256,5,padding=2).to('mps')
 if a.layout=='nhwc':m=m.to(memory_format=torch.channels_last)
 if a.impl!='native':m=(ConvRows if a.impl=='rows' else ConvUnfold)(m)
 x=torch.randn(256,96,16,16,device='mps')
 if a.layout=='nhwc':x=x.contiguous(memory_format=torch.channels_last)
 x.requires_grad_();dtype={'fp32':None,'fp16':torch.float16,'bf16':torch.bfloat16}[a.dtype]
 result={'layout':a.layout,'dtype':a.dtype,'implementation':a.impl};times={k:[] for k in ['forward','input_gradient','weight_gradient','full_backward']}
 try:
  for iteration in range(7):
   m.zero_grad(set_to_none=True);x.grad=None;torch.mps.synchronize();start=time.perf_counter()
   with torch.autocast('mps',dtype=dtype) if dtype else contextlib.nullcontext():y=m(x)
   torch.mps.synchronize();forward=time.perf_counter()-start
   g=torch.ones_like(y);torch.mps.synchronize();start=time.perf_counter()
   torch.autograd.grad(y,x,g,retain_graph=True);torch.mps.synchronize();dx=time.perf_counter()-start
   start=time.perf_counter();torch.autograd.grad(y,m.weight,g,retain_graph=True);torch.mps.synchronize();dw=time.perf_counter()-start
   start=time.perf_counter();y.backward(g);torch.mps.synchronize();back=time.perf_counter()-start
   if iteration>=2:
    for k,v in zip(times,[forward,dx,dw,back]):times[k].append(v)
  result.update(status='complete',median_seconds={k:statistics.median(v) for k,v in times.items()},samples=times)
 except Exception as exc:
  import traceback;traceback.print_exc();result.update(status='error',error=repr(exc))
 OUT.mkdir(exist_ok=True);(OUT/f'conv_{a.impl}_{a.layout}_{a.dtype}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
