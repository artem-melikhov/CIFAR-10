"""Check experimental kernels against checkpoint logits, gradients, full validation."""
import json,argparse
import torch
from torch import nn
from torch.nn import functional as F
from alexnet_bottleneck import ROOT,make_model
from alexnet_deep_sweep import configure,OUT
from alexnet_winograd3_metal import ConvWinograd3Metal
from alexnet_lowrank import evaluate

def main():
 p=argparse.ArgumentParser();p.add_argument('--cases',nargs='+',default=['combined','winograd_metal','winograd_all','winograd_combined']);args=p.parse_args()
 torch.manual_seed(52);results={'isolated_winograd3':[],'models':{}}
 for shape,o in [((2,3,8,8),7),((2,256,8,8),384),((2,384,8,8),384)]:
  native=nn.Conv2d(shape[1],o,3,padding=1).to('mps');variant=ConvWinograd3Metal(native)
  x=torch.randn(shape,device='mps',requires_grad=True);y=native(x);z=variant(x);g=torch.randn_like(y)
  a=torch.autograd.grad(y,(x,native.weight,native.bias),g,retain_graph=True);b=torch.autograd.grad(z,(x,native.weight,native.bias),g)
  row={'shape':shape,'output_max_abs':(y-z).abs().max().item(),'gradient_max_abs':[(t-u).abs().max().item() for t,u in zip(a,b)],
       'gradient_relative_l2':[(t-u).norm().item()/max(t.norm().item(),1e-20) for t,u in zip(a,b)]}
  torch.testing.assert_close(y,z,rtol=1e-4,atol=1e-5)
  assert max(row['gradient_relative_l2'])<1e-4,row
  results['isolated_winograd3'].append(row)
 data=torch.load(ROOT/'data/alexnet_precision_inputs.pt',weights_only=True)
 x=data['train_x'][:16].to('mps').float();x=(x-data['mean'].to('mps').view(1,3,1,1))/data['std'].to('mps').view(1,3,1,1);target=data['train_y'][:16].to('mps')
 references={};reference_logits=None;reference_predictions=None
 for case in ['original',*args.cases]:
  model=configure(make_model(),case).train();torch.manual_seed(777)
  logits=model(x);F.cross_entropy(logits,target).backward()
  gradients={k:p.grad.detach().cpu() for k,p in model.named_parameters()};logits=logits.detach().cpu()
  if case=='original':references=gradients;reference_logits=logits
  relative={k:(v-references[k]).norm().item()/max(references[k].norm().item(),1e-20) for k,v in gradients.items()}
  absolute={k:(v-references[k]).abs().max().item() for k,v in gradients.items()}
  row={'logits_max_abs':(logits-reference_logits).abs().max().item(),'parameter_gradient_relative_l2':relative,'parameter_gradient_max_abs':absolute}
  torch.testing.assert_close(logits,reference_logits,rtol=2e-4,atol=2e-4)
  assert max(relative.values())<5e-4,row
  row['validation']=evaluate(model,data);model.eval();pred=[]
  with torch.no_grad():
   for start in range(0,len(data['val_y']),256):pred.append(model(data['val_x'][start:start+256].to('mps')).argmax(1).cpu())
  pred=torch.cat(pred)
  if case=='original':reference_predictions=pred
  row['changed_validation_predictions']=(pred!=reference_predictions).sum().item()
  row['status']='passed';results['models'][case]=row
  (OUT/'kernel_equivalence.json').write_text(json.dumps(results,indent=2));print(case,row['validation'],row['changed_validation_predictions'],'max gradient relative L2',max(relative.values()),flush=True)
  del model,gradients;torch.mps.empty_cache()

if __name__=='__main__':main()
