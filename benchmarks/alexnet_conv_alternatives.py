"""Exact conv2 alternatives preserving the original weight and bias tensors."""
import torch
from torch import nn
from torch.nn import functional as F

class _ConvWeightGEMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,b):
        ctx.save_for_backward(x,w)
        return F.conv2d(x,w,b,padding=2)
    @staticmethod
    def backward(ctx,g):
        x,w=ctx.saved_tensors
        dx=torch.nn.grad.conv2d_input(x.shape,w,g,padding=2) if ctx.needs_input_grad[0] else None
        dw=None
        if ctx.needs_input_grad[1]:
            columns=F.unfold(x,5,padding=2).permute(1,0,2).reshape(w.shape[1]*25,-1)
            grad=g.permute(1,0,2,3).reshape(w.shape[0],-1)
            dw=(grad@columns.t()).reshape_as(w)
        db=g.sum((0,2,3)) if ctx.needs_input_grad[2] else None
        return dx,dw,db

class ConvWeightGEMM(nn.Module):
    def __init__(self,conv):
        super().__init__();self.weight=conv.weight;self.bias=conv.bias
    def forward(self,x):return _ConvWeightGEMM.apply(x,self.weight,self.bias)

class ConvBatchChunks(nn.Module):
    def __init__(self,conv,chunk=64):
        super().__init__();self.weight=conv.weight;self.bias=conv.bias;self.chunk=chunk
    def forward(self,x):
        return torch.cat([F.conv2d(part,self.weight,self.bias,padding=2) for part in x.split(self.chunk)],dim=0)

class ConvOutputChunks(nn.Module):
    def __init__(self,conv,chunk=64):
        super().__init__();self.weight=conv.weight;self.bias=conv.bias;self.chunk=chunk
    def forward(self,x):
        return torch.cat([F.conv2d(x,w,b,padding=2) for w,b in zip(self.weight.split(self.chunk),self.bias.split(self.chunk))],dim=1)
