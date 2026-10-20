"""Second-stage experiments: exact conv rewrites, compilation, AMP, optimizer and sync."""
import argparse,json,os,time,statistics,contextlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from alexnet_bottleneck import ROOT,make_model
from alexnet_metal_lrn import MetalLRN
OUT=ROOT/'benchmarks/alexnet_deep'

class ConvRows(nn.Module):
    """Exact 5x5 convolution as five 1x5 convolutions; shared original parameters."""
    def __init__(self,conv):
        super().__init__();self.weight=conv.weight;self.bias=conv.bias
    def forward(self,x):
        x=F.pad(x,(0,0,2,2));h=x.shape[2]-4
        result=F.conv2d(x[:,:,0:h,:],self.weight[:,:,0:1,:],None,padding=(0,2))
        for row in range(1,5):
            result=result+F.conv2d(x[:,:,row:row+h,:],self.weight[:,:,row:row+1,:],None,padding=(0,2))
        return result+self.bias.view(1,-1,1,1)

class ConvUnfold(nn.Module):
    def __init__(self,conv):
        super().__init__();self.weight=conv.weight;self.bias=conv.bias
    def forward(self,x):
        n,c,h,w=x.shape
        columns=F.unfold(x,5,padding=2)
        # Large GEMM rather than a broadcast/batched GEMM.
        columns=columns.permute(0,2,1).reshape(n*h*w,-1)
        y=F.linear(columns,self.weight.reshape(self.weight.shape[0],-1),self.bias)
        return y.reshape(n,h,w,-1).permute(0,3,1,2).contiguous()

class AmpMetalLRN(MetalLRN):
    def forward(self,x):return super().forward(x.float())

def configure(model,case):
    metal=case not in ['original','compile_original','compile_original_reduce']
    if metal:
        for index in [2,6]:model.network[index]=AmpMetalLRN(model.network[index])
    if case=='conv_rows':model.network[4]=ConvRows(model.network[4])
    if case=='conv_unfold':model.network[4]=ConvUnfold(model.network[4])
    if case in ['conv_dw_gemm','conv_batch_chunks','conv_output_chunks']:
        from alexnet_conv_alternatives import ConvWeightGEMM,ConvBatchChunks,ConvOutputChunks
        cls={'conv_dw_gemm':ConvWeightGEMM,'conv_batch_chunks':ConvBatchChunks,'conv_output_chunks':ConvOutputChunks}[case]
        model.network[4]=cls(model.network[4])
    if case in ['winograd_all','winograd_combined']:
        from alexnet_winograd_metal import ConvWinogradMetal
        from alexnet_winograd3_metal import ConvWinograd3Metal
        model.network[4]=ConvWinogradMetal(model.network[4])
        for i in [8,10,12]:model.network[i]=ConvWinograd3Metal(model.network[i])
    if case=='winograd_metal':
        from alexnet_winograd_metal import ConvWinogradMetal
        model.network[4]=ConvWinogradMetal(model.network[4])
    if case=='winograd':
        from alexnet_winograd import ConvWinograd
        model.network[4]=ConvWinograd(model.network[4])
    if case in ['relu_lrn','combined','winograd_combined']:
        from alexnet_metal_relu_lrn import MetalReLULRN
        for index in [2,6]:
            model.network[index]=MetalReLULRN(model.network[index]);model.network[index-1]=nn.Identity()
    if case=='inplace':
        for i,m in enumerate(model.network):
            if isinstance(m,nn.ReLU):model.network[i]=nn.ReLU(inplace=True)
    if case=='compile_metal_disabled':
        for index in [2,6]:model.network[index].forward=torch.compiler.disable(model.network[index].forward)
    return model

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--label');p.add_argument('--batch',type=int,default=256)
    args=p.parse_args();OUT.mkdir(exist_ok=True);label=args.label or args.case
    result={'case':args.case,'label':label,'batch':args.batch,'env':{k:os.environ.get(k) for k in ['PYTORCH_MPS_FAST_MATH','PYTORCH_MPS_PREFER_METAL']}}
    started=time.perf_counter()
    try:
        torch.manual_seed(52);model=configure(make_model(),args.case).train()
        data=torch.load(ROOT/'data/alexnet_precision_inputs.pt',weights_only=True)
        x=data['train_x'][:args.batch].to('mps').float()
        x=(x-data['mean'].to('mps').view(1,3,1,1))/data['std'].to('mps').view(1,3,1,1)
        y=data['train_y'][:args.batch].to('mps')
        kwargs={}
        if args.case=='foreach':kwargs['foreach']=True
        if args.case in ['fused_adam','combined','winograd_combined']:kwargs['fused']=True
        optimizer=torch.optim.Adam(model.parameters(),lr=3e-4,**kwargs)
        if args.case.startswith('compile'):
            model=torch.compile(model,dynamic=False,mode='reduce-overhead' if args.case.endswith('reduce') else 'default')
        dtype=torch.float16 if args.case=='amp16' else torch.bfloat16 if args.case=='ampbf16' else None
        scaler=torch.amp.GradScaler('mps',enabled=dtype==torch.float16,init_scale=1024.)
        def step():
            optimizer.zero_grad(set_to_none=True)
            ctx=torch.autocast('mps',dtype=dtype) if dtype else contextlib.nullcontext()
            with ctx:
                logits=model(x);loss=F.cross_entropy(logits,y)
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
            metric=(logits.detach().argmax(1)==y).sum()
            if args.case not in ['async','combined','winograd_combined']:loss.item();metric.item()
            return loss.detach()
        for _ in range(4):step();torch.mps.synchronize()
        result['setup_warmup_seconds']=time.perf_counter()-started
        times=[];loss=None
        for repeat in range(3):
            torch.mps.synchronize();start=time.perf_counter()
            for _ in range(8):loss=step()
            torch.mps.synchronize();times.append((time.perf_counter()-start)/8)
        result.update(status='complete',block_step_seconds=times,median_step_seconds=statistics.median(times),
                      final_loss=loss.item(),driver_bytes=torch.mps.driver_allocated_memory())
    except Exception as exc:
        import traceback
        traceback.print_exc()
        result.update(status='error',error=repr(exc))
    result['total_seconds']=time.perf_counter()-started
    (OUT/f'{label}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
