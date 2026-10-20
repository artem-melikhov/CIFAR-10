"""Fast FP32/AMP FP16/AMP BF16 comparison using the saved AlexNet weights.

Prepare identical augmented batches once, then use fresh processes per mode.
Each mode evaluates full validation and performs 40 successful Adam steps.
Existing notebook kernels and checkpoint files are never changed.
"""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'benchmarks/alexnet_precision'
CACHE = ROOT / 'data/alexnet_precision_inputs.pt'


def prepare(steps, batch):
    import torch
    sys.path.insert(0, str(ROOT/'utils'))
    from data_utils import images_to_tensor, train_dataset_split, calculate_mean_std, AutoAugumentationDataset
    from torch.utils.data import TensorDataset
    torch.manual_seed(52)
    x,y = images_to_tensor(str(ROOT/'data/train'),str(ROOT/'data/trainLabels.csv'))
    train,val = train_dataset_split(TensorDataset(x,y),.8,52)
    mean,std=calculate_mean_std(train)
    vx=x[val.indices].float()
    vx.sub_(mean.view(1,-1,1,1)).div_(std.view(1,-1,1,1))
    order=torch.randperm(len(train),generator=torch.Generator().manual_seed(52))[:steps*batch]
    ds=AutoAugumentationDataset(train)
    tx=torch.empty((len(order),3,32,32),dtype=torch.uint8)
    ty=torch.empty(len(order),dtype=torch.long)
    torch.manual_seed(52)
    for j,i in enumerate(order.tolist()):
        tx[j],ty[j]=ds[i]
    torch.save({'train_x':tx,'train_y':ty,'val_x':vx,'val_y':y[val.indices],
                'mean':mean,'std':std,'steps':steps,'batch_size':batch},CACHE)
    print('PREPARED',len(tx),'fixed augmented train images;',len(vx),'validation',flush=True)


def run(mode):
    import torch
    from torch import nn
    import psutil
    dtype={'fp32':None,'fp16':torch.float16,'bf16':torch.bfloat16}[mode]
    path=OUT/f'{mode}.json'
    result={'mode':mode,'status':'running','device':'mps','torch':torch.__version__}
    def save():path.write_text(json.dumps(result,indent=2))
    save()
    try:
        torch.manual_seed(52)
        source=json.loads((ROOT/'notebooks/AlexNet.ipynb').read_text())
        node=next(n for c in source['cells'] if c['cell_type']=='code'
                  for n in ast.parse(''.join(c['source'])).body
                  if isinstance(n,ast.ClassDef) and n.name=='AlexNet')
        ns={'torch':torch,'nn':nn}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'AlexNet.ipynb:model','exec'),ns)
        model=ns['AlexNet'](3,10).to('mps')
        checkpoint=ROOT/'notebooks/models/alexnet.pth'
        weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict(weights)
        data=torch.load(CACHE,weights_only=True)
        batch=data['batch_size']
        mean=data['mean'].to('mps').view(1,-1,1,1)
        std=data['std'].to('mps').view(1,-1,1,1)
        loss_fn=nn.CrossEntropyLoss()
        peak_driver=peak_tensors=0
        def memory():
            nonlocal peak_driver,peak_tensors
            peak_driver=max(peak_driver,torch.mps.driver_allocated_memory())
            peak_tensors=max(peak_tensors,torch.mps.current_allocated_memory())
        def context(force_fp32=False):
            return torch.autocast('mps',dtype=dtype or torch.float16,
                                  enabled=dtype is not None and not force_fp32)
        def evaluate(force_fp32=False,return_predictions=False):
            model.eval()
            loss_sum=correct=total=0
            preds=[]
            torch.mps.synchronize()
            start=time.perf_counter()
            with torch.inference_mode():
                for i in range(0,len(data['val_y']),batch):
                    x=data['val_x'][i:i+batch].to('mps')
                    y=data['val_y'][i:i+batch].to('mps')
                    with context(force_fp32):
                        logits=model(x)
                        loss=loss_fn(logits,y)
                    if not torch.isfinite(loss).item():
                        raise RuntimeError('non-finite validation loss')
                    pred=logits.argmax(1)
                    loss_sum+=loss.item()*len(y)
                    correct+=(pred==y).sum().item()
                    total+=len(y)
                    if return_predictions:preds.extend(pred.cpu().tolist())
                    memory()
            torch.mps.synchronize()
            return {'accuracy':correct/total,'loss':loss_sum/total,
                    'seconds':time.perf_counter()-start},preds
        # Full initial FP32 reference and selected-precision inference on same weights.
        initial_fp32,reference=evaluate(force_fp32=True,return_predictions=True)
        if dtype is None:
            initial_mode,predictions=initial_fp32,reference
        else:
            initial_mode,predictions=evaluate(return_predictions=True)
        result.update(initial_fp32=initial_fp32,initial_mode=initial_mode,
                      prediction_agreement=sum(a==b for a,b in zip(reference,predictions))/len(reference))
        save()
        print(mode,'INITIAL',json.dumps(result),flush=True)
        optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
        scaler=torch.amp.GradScaler('mps',enabled=mode=='fp16')
        def step(index):
            torch.manual_seed(1000+index)
            i=index*batch
            x=data['train_x'][i:i+batch].to('mps')
            y=data['train_y'][i:i+batch].to('mps')
            x=(x.float()-mean)/std
            optimizer.zero_grad()
            with context():
                logits=model(x)
                loss=loss_fn(logits,y)
            memory()
            if not torch.isfinite(loss).item():
                raise RuntimeError('non-finite training loss')
            old_scale=scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            accepted=scaler.get_scale()>=old_scale
            loss_value=loss.item()
            (logits.argmax(1)==y).sum().item()  # Match scalar reads in normal training.
            memory()
            return accepted,loss_value
        # Compile/warm the actual training graph, then restore saved model and reset Adam.
        model.train()
        for i in range(3):step(i)
        model.load_state_dict(weights)
        optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
        scaler=torch.amp.GradScaler('mps',enabled=mode=='fp16',init_scale=scaler.get_scale())
        torch.mps.synchronize()
        start=time.perf_counter()
        completed=skipped=0
        losses=[]
        while completed<data['steps']:
            accepted,value=step(completed)
            if accepted:
                losses.append(value)
                completed+=1
            else:
                skipped+=1
                if skipped>12:raise RuntimeError('too many gradient-overflow retries')
        torch.mps.synchronize()
        train_s=time.perf_counter()-start
        result.update(train_seconds=train_s,train_images_per_second=completed*batch/train_s,
                      completed_steps=completed,skipped_attempts=skipped,
                      final_grad_scale=scaler.get_scale(),mean_training_loss=sum(losses)/len(losses))
        final_fp32,_=evaluate(force_fp32=True)
        final_mode=final_fp32 if dtype is None else evaluate()[0]
        result.update(status='complete',final_fp32=final_fp32,final_mode=final_mode,
                      max_observed_driver_bytes=peak_driver,max_observed_tensor_bytes=peak_tensors,
                      checkpoint=str(checkpoint.relative_to(ROOT)),batch_size=batch,
                      validation_images=len(data['val_y']),training_images=completed*batch)
        print(mode,'COMPLETE',json.dumps(result),flush=True)
    except Exception as e:
        result.update(status='error',error_type=type(e).__name__,error=str(e))
        print(mode,'ERROR',type(e).__name__,str(e),flush=True)
    finally:save()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--mode',choices=['fp32','fp16','bf16'])
    p.add_argument('--steps',type=int,default=40)
    p.add_argument('--batch-size',type=int,default=256)
    args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.prepare:
        prepare(args.steps,args.batch_size)
        return
    if args.mode:
        run(args.mode)
        return
    subprocess.run([sys.executable,'-u',__file__,'--prepare','--steps',str(args.steps),
                    '--batch-size',str(args.batch_size)],check=True,cwd=ROOT)
    results=[]
    for mode in ['fp32','fp16','bf16']:
        print('START',mode,flush=True)
        with (OUT/f'{mode}.log').open('w') as log:
            subprocess.run([sys.executable,'-u',__file__,'--mode',mode],cwd=ROOT,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((OUT/f'{mode}.json').read_text())
        results.append(result)
        (OUT/'results.json').write_text(json.dumps({'results':results},indent=2))
        print('RESULT',json.dumps(result),flush=True)


if __name__=='__main__':main()
