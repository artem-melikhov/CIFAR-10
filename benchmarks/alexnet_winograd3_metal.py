"""Experimental differentiable F(2,3) conv2: Metal transforms + batched GEMMs.

Same Cook-Toom matrices as https://github.com/andravin/wincnn. FP32 NCHW, even spatial sizes,
3x3/stride1/pad1, no groups. No changes to the original parameterization.
"""
import torch
from torch import nn

BT=[[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,-1,0,1]]
G=[[1,0,0],[.5,.5,.5],[.5,-.5,.5],[0,0,1]]
AT=[[1,1,1,0],[0,1,-1,1]]
_LIB=None

def helper(name,matrix):
    m,n=len(matrix),len(matrix[0]);lines=[]
    for row,coefficients in enumerate(matrix):
        terms=[f'({float(v):.10g}f * a[{j}])' if float(v)!=int(v) else f'({int(v)}.0f * a[{j}])' for j,v in enumerate(coefficients) if v]
        lines.append(f'b[{row}] = '+(' + '.join(terms) if terms else '0.0f')+';')
    return f'''inline void {name}1(thread const float* a, thread float* b) {{ {' '.join(lines)} }}
inline void {name}2(thread const float* a, thread float* b) {{
 float tmp[{n*m}],column[{n}],result[{m}];
 for (uint row=0;row<{n};++row) {name}1(a+row*{n},tmp+row*{m});
 for (uint col=0;col<{m};++col) {{
  for (uint row=0;row<{n};++row) column[row]=tmp[row*{m}+col];
  {name}1(column,result);
  for (uint row=0;row<{m};++row) b[row*{m}+col]=result[row];
 }}
}}
'''

def library():
    global _LIB
    if _LIB is not None:return _LIB
    code='#include <metal_stdlib>\nusing namespace metal;\n'
    for name,matrix in [('bt',BT),('g',G),('at',AT),('b',list(map(list,zip(*BT)))),('gt',list(map(list,zip(*G)))),('a',list(map(list,zip(*AT))))]:
        code+=helper(name,matrix)
    code+=r'''
kernel void input_transform(device const float* x,device float* v,
 constant uint& C,constant uint& H,constant uint& W,constant uint& T,
 uint id [[thread_position_in_grid]]) {
 uint c=id/T,t=id%T,TH=H/2,TW=W/2,n=t/(TH*TW),ty=(t/TW)%TH,tx=t%TW;
 float d[16],out[16];
 for(int r=0;r<4;++r) for(int s=0;s<4;++s) {
  int h=int(ty*2)+r-1,w=int(tx*2)+s-1;
  d[r*4+s]=(h>=0&&h<int(H)&&w>=0&&w<int(W))?x[((n*C+c)*H+h)*W+w]:0.0f;
 }
 bt2(d,out);
 for(uint k=0;k<16;++k)v[(k*C+c)*T+t]=out[k];
}
kernel void weight_transform(device const float* w,device float* u,
 constant uint& O,constant uint& C,uint id [[thread_position_in_grid]]) {
 float d[9],out[16];for(uint k=0;k<9;++k)d[k]=w[id*9+k];g2(d,out);
 for(uint k=0;k<16;++k)u[k*O*C+id]=out[k];
}
kernel void output_transform(device const float* m,device const float* bias,device float* y,
 constant uint& O,constant uint& H,constant uint& W,constant uint& T,
 uint id [[thread_position_in_grid]]) {
 uint o=id/T,t=id%T,TH=H/2,TW=W/2,n=t/(TH*TW),ty=(t/TW)%TH,tx=t%TW;
 float d[16],out[4];for(uint k=0;k<16;++k)d[k]=m[(k*O+o)*T+t];at2(d,out);
 for(uint r=0;r<2;++r)for(uint s=0;s<2;++s)y[((n*O+o)*H+ty*2+r)*W+tx*2+s]=out[r*2+s]+bias[o];
}
kernel void output_gradient(device const float* dy,device float* dm,
 constant uint& O,constant uint& H,constant uint& W,constant uint& T,
 uint id [[thread_position_in_grid]]) {
 uint o=id/T,t=id%T,TH=H/2,TW=W/2,n=t/(TH*TW),ty=(t/TW)%TH,tx=t%TW;
 float d[4],out[16];
 for(uint r=0;r<2;++r)for(uint s=0;s<2;++s)d[r*2+s]=dy[((n*O+o)*H+ty*2+r)*W+tx*2+s];
 a2(d,out);for(uint k=0;k<16;++k)dm[(k*O+o)*T+t]=out[k];
}
kernel void weight_gradient(device const float* du,device float* dw,
 constant uint& O,constant uint& C,uint id [[thread_position_in_grid]]) {
 float d[16],out[9];for(uint k=0;k<16;++k)d[k]=du[k*O*C+id];gt2(d,out);
 for(uint k=0;k<9;++k)dw[id*9+k]=out[k];
}
kernel void input_gradient_tiles(device const float* dv,device float* dt,
 constant uint& C,constant uint& T,uint id [[thread_position_in_grid]]) {
 uint c=id/T,t=id%T;float d[16],out[16];
 for(uint k=0;k<16;++k)d[k]=dv[(k*C+c)*T+t];b2(d,out);
 for(uint k=0;k<16;++k)dt[(k*C+c)*T+t]=out[k];
}
kernel void input_gradient_fold(device const float* dt,device float* dx,
 constant uint& C,constant uint& H,constant uint& W,constant uint& T,
 uint id [[thread_position_in_grid]]) {
 uint w=id%W,h=(id/W)%H,c=(id/(H*W))%C,n=id/(C*H*W),TH=H/2,TW=W/2;
 int ph=int(h)+1,pw=int(w)+1;float sum=0.0f;
 for(int ty=max(0,(ph-2)/2);ty<=min(int(TH)-1,ph/2);++ty)
  for(int tx=max(0,(pw-2)/2);tx<=min(int(TW)-1,pw/2);++tx) {
   int r=ph-ty*2,s=pw-tx*2;
   if(r>=0&&r<4&&s>=0&&s<4) {
    uint t=n*TH*TW+ty*TW+tx;sum+=dt[((r*4+s)*C+c)*T+t];
   }
  }
 dx[id]=sum;
}
'''
    _LIB=torch.mps.compile_shader(code)
    return _LIB

class _Winograd3(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,bias):
        if x.dtype!=torch.float32 or x.device.type!='mps':raise ValueError('FP32 MPS only')
        x=x.contiguous();w=w.contiguous();n,c,h,width=x.shape;o=w.shape[0]
        if h%2 or width%2:raise ValueError('Even H/W required')
        t=n*(h//2)*(width//2);u=torch.empty((16,o,c),device=x.device);v=torch.empty((16,c,t),device=x.device)
        lib=library();lib.input_transform(x,v,c,h,width,t,threads=c*t)
        lib.weight_transform(w,u,o,c,threads=o*c)
        m=torch.bmm(u,v);y=torch.empty((n,o,h,width),device=x.device)
        lib.output_transform(m,bias,y,o,h,width,t,threads=o*t)
        ctx.save_for_backward(u,v);ctx.shape=(n,c,h,width,o,t)
        return y
    @staticmethod
    def backward(ctx,dy):
        u,v=ctx.saved_tensors;n,c,h,w,o,t=ctx.shape;lib=library();dy=dy.contiguous()
        dm=torch.empty((16,o,t),device=dy.device)
        lib.output_gradient(dy,dm,o,h,w,t,threads=o*t)
        dx=dw=None
        if ctx.needs_input_grad[0]:
            dv=torch.bmm(u.transpose(1,2),dm);dt=torch.empty_like(dv)
            lib.input_gradient_tiles(dv,dt,c,t,threads=c*t)
            dx=torch.empty((n,c,h,w),device=dy.device)
            lib.input_gradient_fold(dt,dx,c,h,w,t,threads=n*c*h*w)
        if ctx.needs_input_grad[1]:
            du=torch.bmm(dm,v.transpose(1,2));dw=torch.empty((o,c,3,3),device=dy.device)
            lib.weight_gradient(du,dw,o,c,threads=o*c)
        db=dy.sum((0,2,3)) if ctx.needs_input_grad[2] else None
        return dx,dw,db

class ConvWinograd3Metal(nn.Module):
    def __init__(self,conv):
        super().__init__();self.weight=conv.weight;self.bias=conv.bias
    def forward(self,x):return _Winograd3.apply(x,self.weight,self.bias)
