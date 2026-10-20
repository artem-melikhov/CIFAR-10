"""Check equivalent LRN numerics and fixed checkpoint validation predictions."""
import json
from pathlib import Path
import torch
from torch import nn
from alexnet_bottleneck import make_model, LRN2D, OUT, ROOT
from alexnet_metal_lrn import MetalLRN

def main():
    torch.manual_seed(52)
    results={'numerics':[],'validation':{}}
    for shape in [(2,7,4,4),(2,96,32,32),(2,256,16,16)]:
        for scale in [1.,10.,100.]:
            for cls in [LRN2D,MetalLRN]:
                x=(torch.randn(shape,device='mps')*scale).requires_grad_()
                z=x.detach().clone().requires_grad_()
                g=torch.randn_like(x)
                original=nn.LocalResponseNorm(5)
                a,b=original(x),cls(original)(z)
                a.backward(g);b.backward(g)
                torch.testing.assert_close(a,b,rtol=3e-5,atol=3e-5)
                torch.testing.assert_close(x.grad,z.grad,rtol=5e-5,atol=5e-6)
                results['numerics'].append({'implementation':cls.__name__,'shape':shape,'scale':scale,
                    'forward_max_abs':(a-b).abs().max().item(),
                    'input_gradient_max_abs':(x.grad-z.grad).abs().max().item()})
    data=torch.load(ROOT/'data/alexnet_precision_inputs.pt',weights_only=True)
    predictions={}
    for name,cls in [('baseline',None),('lrn2d',LRN2D),('metal_lrn',MetalLRN)]:
        model=make_model().eval()
        if cls:
            for i in [2,6]:model.network[i]=cls(model.network[i])
        correct=0;loss_sum=0.;pred=[]
        with torch.no_grad():
            for start in range(0,len(data['val_y']),256):
                x=data['val_x'][start:start+256].to('mps');y=data['val_y'][start:start+256].to('mps')
                logits=model(x)
                loss_sum+=nn.functional.cross_entropy(logits,y,reduction='sum').item()
                p=logits.argmax(1).cpu();pred.append(p);correct+=(p==y.cpu()).sum().item()
        predictions[name]=torch.cat(pred)
        results['validation'][name]={'accuracy':correct/len(data['val_y']),
            'loss':loss_sum/len(data['val_y']),
            'changed_predictions':(predictions[name]!=predictions['baseline']).sum().item()}
        del model
        torch.mps.empty_cache()
        print(name,results['validation'][name],flush=True)
    (OUT/'equivalence.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()
