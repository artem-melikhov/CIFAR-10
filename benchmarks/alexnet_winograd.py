"""Experimental F(2,5) Cook-Toom convolution, same trainable weights.

Transform construction follows the Cook-Toom interpolation approach described in
https://github.com/andravin/wincnn (points 0,1,-1,2,-2 and infinity).
"""
import torch
from torch import nn
from torch.nn import functional as F

class ConvWinograd(nn.Module):
    def __init__(self,conv):
        super().__init__();self.weight=conv.weight;self.bias=conv.bias
        kwargs={'device':conv.weight.device,'dtype':conv.weight.dtype}
        self.register_buffer('bt',torch.tensor([[4,0,-5,0,1,0],[0,-4,-4,1,1,0],[0,4,-4,-1,1,0],
                                    [0,-2,-1,2,1,0],[0,2,-1,-2,1,0],[0,4,0,-5,0,1]],**kwargs))
        self.register_buffer('g',torch.tensor([[1/4,0,0,0,0],[-1/6]*5,[-1/6,1/6,-1/6,1/6,-1/6],
                                    [1/24,1/12,1/6,1/3,2/3],[1/24,-1/12,1/6,-1/3,2/3],[0,0,0,0,1]],**kwargs))
        self.register_buffer('at',torch.tensor([[1,1,1,1,1,0],[0,1,-1,2,-2,1]],**kwargs))
    def forward(self,x):
        n,c,h,w=x.shape;o=self.weight.shape[0];th,tw=h//2,w//2
        tiles=F.pad(x,(2,2,2,2)).unfold(2,6,2).unfold(3,6,2)
        v=self.bt@tiles@self.bt.t()
        v=v.permute(4,5,1,0,2,3).reshape(36,c,n*th*tw)
        u=self.g@self.weight@self.g.t()
        u=u.permute(2,3,0,1).reshape(36,o,c)
        m=torch.bmm(u,v).reshape(6,6,o,n,th,tw).permute(3,2,4,5,0,1)
        out=self.at@m@self.at.t()
        return out.permute(0,1,2,4,3,5).reshape(n,o,h,w)+self.bias.view(1,-1,1,1)
