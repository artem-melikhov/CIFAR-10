"""Experimental fused ReLU + LRN, for finite FP32 NCHW inputs."""
import torch
from torch import nn

_LIBRARY = None

def library():
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = torch.mps.compile_shader(r'''
#include <metal_stdlib>
using namespace metal;
kernel void lrn_forward(device const float* x, device float* y,
                        device float* r, device float* rs,
                        constant uint& channels, constant uint& spatial,
                        constant float& a, constant float& beta,
                        constant float& k, constant uint& radius,
                        uint i [[thread_position_in_grid]]) {
    uint c = (i / spatial) % channels;
    float sum = 0.0f;
    for (int d = -int(radius); d <= int(radius); ++d) {
        int neighbor = int(c) + d;
        if (neighbor >= 0 && neighbor < int(channels)) {
            float v = max(x[int(i) + d * int(spatial)], 0.0f);
            sum += v * v;
        }
    }
    float s = k + a * sum;
    float ri = pow(s, -beta);
    y[i] = max(x[i], 0.0f) * ri;
    r[i] = ri;
    rs[i] = ri / s;
}
kernel void lrn_backward(device const float* x, device const float* g,
                         device const float* r, device const float* rs,
                         device float* dx,
                         constant uint& channels, constant uint& spatial,
                         constant float& coefficient, constant uint& radius,
                         uint i [[thread_position_in_grid]]) {
    uint c = (i / spatial) % channels;
    float sum = 0.0f;
    for (int d = -int(radius); d <= int(radius); ++d) {
        int neighbor = int(c) + d;
        if (neighbor >= 0 && neighbor < int(channels)) {
            int j = int(i) + d * int(spatial);
            sum += g[j] * max(x[j], 0.0f) * rs[j];
        }
    }
    dx[i] = x[i] > 0.0f ? g[i] * r[i] - coefficient * x[i] * sum : 0.0f;
}
''')
    return _LIBRARY

class _FusedReLULRN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, size, alpha, beta, k):
        if x.device.type != 'mps' or x.dtype != torch.float32 or size % 2 != 1:
            raise ValueError('Experimental kernel requires MPS FP32 and odd window size')
        x = x.contiguous()
        y, r, rs = torch.empty_like(x), torch.empty_like(x), torch.empty_like(x)
        c, spatial = x.shape[1], x.shape[2] * x.shape[3]
        library().lrn_forward(x, y, r, rs, c, spatial, alpha/size, beta, k, size//2,
                              threads=x.numel())
        ctx.save_for_backward(x, r, rs)
        ctx.config = c, spatial, 2*alpha*beta/size, size//2
        return y

    @staticmethod
    def backward(ctx, grad):
        x, r, rs = ctx.saved_tensors
        dx = torch.empty_like(x)
        library().lrn_backward(x, grad.contiguous(), r, rs, dx, *ctx.config,
                               threads=x.numel())
        return dx, None, None, None, None

class MetalReLULRN(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.size,self.alpha,self.beta,self.k = original.size,original.alpha,original.beta,original.k

    def forward(self,x):
        return _FusedReLULRN.apply(x,self.size,self.alpha,self.beta,self.k)
