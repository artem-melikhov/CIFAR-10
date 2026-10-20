"""Attribute the first stock LRN backward to individual autograd operations."""
import json,time,statistics
import torch
from alexnet_bottleneck import OUT

def main():
    torch.manual_seed(52)
    x=torch.randn(256,96,32,32,device='mps',requires_grad=True)
    g=torch.randn_like(x);lrn=torch.nn.LocalResponseNorm(5)
    times={};forward=[];backward=[]
    for step in range(8):
        x.grad=None;torch.mps.synchronize();start=time.perf_counter();y=lrn(x)
        torch.mps.synchronize();f=time.perf_counter()-start
        seen=set();handles=[];starts={}
        def visit(node):
            if node is None or node in seen:return
            seen.add(node);name=type(node).__name__
            def pre(*unused):
                torch.mps.synchronize();starts[node]=time.perf_counter()
            def post(*unused):
                torch.mps.synchronize()
                if step>=2:times.setdefault(name,[]).append(time.perf_counter()-starts[node])
            handles.extend([node.register_prehook(pre),node.register_hook(post)])
            for parent,_ in node.next_functions:visit(parent)
        visit(y.grad_fn)
        start=time.perf_counter();y.backward(g);torch.mps.synchronize();b=time.perf_counter()-start
        if step>=2:forward.append(f);backward.append(b)
        for h in handles:h.remove()
    result={'shape':list(x.shape),'forward_seconds':statistics.median(forward),
            'instrumented_backward_seconds':statistics.median(backward),
            'backward_operators_ms':{k:statistics.median(v)*1000 for k,v in times.items()}}
    (OUT/'lrn_operators.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
