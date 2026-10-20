"""Run GPU experiments sequentially, each in a fresh process, with timeouts."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks/alexnet_deep'
OUT.mkdir(exist_ok=True)
cases=[('original',{}),('metal',{}),('async',{}),('foreach',{}),('fused_adam',{}),('inplace',{}),('amp16',{}),('ampbf16',{}),('conv_rows',{}),('conv_unfold',{}),('compile_metal',{}),('metal',{'PYTORCH_MPS_FAST_MATH':'1'}),('metal',{'PYTORCH_MPS_PREFER_METAL':'1'}),('metal',{'PYTORCH_MPS_FAST_MATH':'1','PYTORCH_MPS_PREFER_METAL':'1'})]
for case,extra in cases:
 label=case+('_'+('_'.join(k.removeprefix('PYTORCH_MPS_').lower() for k in extra)) if extra else '')
 print('START',label,flush=True)
 env=dict(os.environ,**extra)
 with (OUT/(label+'.log')).open('w') as log:
  try:
   subprocess.run([sys.executable,'-u',str(ROOT/'benchmarks/alexnet_deep_sweep.py'),'--case',case,'--label',label],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=150,check=True)
  except Exception as exc:
   (OUT/(label+'.runner_error.json')).write_text(json.dumps({'error':repr(exc)}))
 path=OUT/(label+'.json')
 print('DONE',label,path.read_text() if path.exists() else 'failed',flush=True)
