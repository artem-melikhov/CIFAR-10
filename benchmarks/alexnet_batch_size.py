"""Time the notebook's AlexNet with real PNG data in isolated processes.

Run: .venv/bin/python -u benchmarks/alexnet_batch_size.py
The parent only monitors CPU/system memory. Each batch size has its own
child process and four DataLoader workers. Existing kernels are untouched.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'benchmarks/alexnet_batches'


def child(batch_size, repeats):
    import torch
    import torchvision
    from torch import nn
    from torch.utils.data import TensorDataset
    sys.path.insert(0, str(ROOT / 'utils'))
    from data_utils import (
        images_to_tensor, train_dataset_split, calculate_mean_std,
        AutoAugumentationDataset, create_dataloader,
    )

    result_path = OUTPUT / f'batch_{batch_size}.json'
    result = {'batch_size': batch_size, 'status': 'initializing', 'epochs': [],
              'torch': torch.__version__, 'torchvision': torchvision.__version__,
              'workers': 4, 'seed': 52, 'repeats': repeats}
    def save():
        result_path.write_text(json.dumps(result, indent=2))
    save()
    dl = None
    try:
        torch.manual_seed(52)
        device = torch.device('mps')
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS is required for these hardware timings')
        result['device'] = str(device)
        notebook = json.loads((ROOT / 'notebooks/AlexNet.ipynb').read_text())
        model_node = next(node for c in notebook['cells'] if c['cell_type'] == 'code'
                          for node in ast.parse(''.join(c['source'])).body
                          if isinstance(node, ast.ClassDef) and node.name == 'AlexNet')
        namespace = {'torch': torch, 'nn': nn}
        exec(compile(ast.Module(body=[model_node], type_ignores=[]),
                     'AlexNet.ipynb:model', 'exec'), namespace)
        started = time.perf_counter()
        x, y = images_to_tensor(str(ROOT / 'data/train'), str(ROOT / 'data/trainLabels.csv'))
        train, val = train_dataset_split(TensorDataset(x, y), .8, random_seed=52)
        mean, std = calculate_mean_std(train)
        vx = x[val.indices].float()
        vx.sub_(mean.view(1, -1, 1, 1)).div_(std.view(1, -1, 1, 1))
        val_ready = TensorDataset(vx, y[val.indices])
        dl = create_dataloader(AutoAugumentationDataset(train), batch_size,
                               shuffle=True, random_seed=52, num_workers=4)
        vl = create_dataloader(val_ready, batch_size, shuffle=False, random_seed=52)
        model = namespace['AlexNet'](3, 10).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        criterion = nn.CrossEntropyLoss()
        mean_device = mean.to(device).view(1, -1, 1, 1)
        std_device = std.to(device).view(1, -1, 1, 1)
        result.update(train_images=len(train), val_images=len(val),
                      setup_seconds=time.perf_counter()-started, status='running',
                      train_steps=len(dl), val_steps=len(vl))
        save()
        print('READY', batch_size, 'setup_s', result['setup_seconds'], flush=True)
        for epoch in range(repeats + 1):
            torch.manual_seed(52 + epoch)
            peak_driver = peak_tensors = 0
            def memory():
                nonlocal peak_driver, peak_tensors
                peak_driver = max(peak_driver, torch.mps.driver_allocated_memory())
                peak_tensors = max(peak_tensors, torch.mps.current_allocated_memory())
            model.train()
            train_loss_sum = train_correct = train_n = 0
            torch.mps.synchronize()
            start = time.perf_counter()
            for bx, by in dl:
                bx, by = bx.to(device), by.to(device)
                bx = (bx.float() - mean_device) / std_device
                logits = model(bx)
                loss = criterion(logits, by)
                memory()  # Observe stored forward tensors before backward frees them.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item() * len(by)
                train_correct += (logits.argmax(1) == by).sum().item()
                train_n += len(by)
                memory()
            torch.mps.synchronize()
            train_s = time.perf_counter()-start
            model.eval()
            val_loss_sum = correct = val_n = 0
            start = time.perf_counter()
            with torch.no_grad():
                for bx, by in vl:
                    bx, by = bx.to(device), by.to(device)
                    logits = model(bx)
                    loss = criterion(logits, by)
                    val_loss_sum += loss.item() * len(by)
                    correct += (logits.argmax(1) == by).sum().item()
                    val_n += len(by)
                    memory()
            torch.mps.synchronize()
            val_s = time.perf_counter()-start
            row = {'epoch': epoch, 'train_seconds': train_s, 'val_seconds': val_s,
                   'total_seconds': train_s+val_s, 'train_images_per_second': train_n/train_s,
                   'max_observed_driver_bytes': peak_driver,
                   'max_observed_tensor_bytes': peak_tensors}
            if epoch:
                result['epochs'].append(row)
            else:
                result['warmup'] = row
            save()
            print('EPOCH', batch_size, 'warmup' if epoch == 0 else epoch,
                  json.dumps(row), flush=True)
        result.update(status='complete',
                      median_train_seconds=statistics.median(e['train_seconds'] for e in result['epochs']),
                      median_val_seconds=statistics.median(e['val_seconds'] for e in result['epochs']),
                      median_total_seconds=statistics.median(e['total_seconds'] for e in result['epochs']),
                      median_train_images_per_second=statistics.median(e['train_images_per_second'] for e in result['epochs']),
                      max_observed_driver_bytes=max(e['max_observed_driver_bytes'] for e in [result['warmup'], *result['epochs']]),
                      max_observed_tensor_bytes=max(e['max_observed_tensor_bytes'] for e in [result['warmup'], *result['epochs']]))
    except Exception as exc:
        result.update(status='error', error_type=type(exc).__name__, error=str(exc))
        print('ERROR', batch_size, type(exc).__name__, str(exc), flush=True)
    finally:
        save()
        if dl is not None and dl._iterator is not None:
            dl._iterator._shutdown_workers()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--batches', type=int, nargs='+', default=[256, 128, 512, 1024])
    args = p.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.batch_size:
        child(args.batch_size, args.repeats)
        return
    import psutil
    all_results = {'method': 'separate fresh processes, one warmup and three measured full train+val epochs; actual notebook model, PNGs, Adam lr=3e-4, four persistent workers, no plots/checkpoints', 'results': []}
    for batch in args.batches:
        print('START BATCH', batch, flush=True)
        log_path = OUTPUT / f'batch_{batch}.log'
        before_swap = psutil.swap_memory().used
        monitor = []
        abort_reason = None
        with log_path.open('w') as log:
            proc = subprocess.Popen([sys.executable, '-u', __file__, '--batch-size', str(batch),
                                     '--repeats', str(args.repeats)], cwd=ROOT,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            low_memory = 0
            while proc.poll() is None:
                vm, swap = psutil.virtual_memory(), psutil.swap_memory()
                monitor.append({'time': time.time(), 'available_bytes': vm.available,
                                'swap_used_bytes': swap.used})
                low_memory = low_memory + 1 if vm.available < 256*2**20 else 0
                if low_memory >= 3 or (swap.used-before_swap > 2*2**30 and vm.available < 2**30):
                    abort_reason = 'system memory guard: available RAM low or swap increased by >2 GiB with <1 GiB available'
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=30)
                    break
                time.sleep(1)
        path = OUTPUT / f'batch_{batch}.json'
        row = json.loads(path.read_text()) if path.exists() else {'batch_size': batch}
        if abort_reason:
            row.update(status='memory_guard_abort', error=abort_reason)
        elif proc.returncode:
            row.update(status='process_error', returncode=proc.returncode)
        row['system_memory'] = {
            'available_start_bytes': monitor[0]['available_bytes'],
            'min_available_bytes': min(x['available_bytes'] for x in monitor),
            'swap_start_bytes': before_swap,
            'max_swap_bytes': max(x['swap_used_bytes'] for x in monitor),
        }
        (OUTPUT / f'batch_{batch}_memory.json').write_text(json.dumps(monitor, indent=2))
        path.write_text(json.dumps(row, indent=2))
        all_results['results'].append(row)
        (OUTPUT / 'results.json').write_text(json.dumps(all_results, indent=2))
        print('FINISHED', batch, json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
