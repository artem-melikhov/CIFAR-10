"""Reproducible LeNet pipeline timings; no changes to the training notebook.

Run with .venv/bin/python benchmarks/lenet_pipeline.py --download
or --images PATH --labels PATH for the project's PNG dataset.
"""
import argparse
import copy
import gc
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'utils'))
from data_utils import StandardizationDataset, images_to_tensor


class LegacyFloatAugmentation(Dataset):
    """Исходная float32-схема только для воспроизведения baseline в замерах."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.augment = v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image, label = self.dataset[i]
        image = (image * 255).round().to(torch.uint8)
        return self.augment(image).float() / 255, label


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 6, 5), nn.Tanh(), nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2),
            nn.Conv2d(16, 120, 5), nn.Tanh(), nn.Flatten(),
            nn.Linear(120, 84), nn.Tanh(), nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.network(x)


class Augmented(Dataset):
    def __init__(self, x, y, mean, std, normalize=False, simple=False):
        self.x, self.y = x, y
        self.mean, self.std = mean[:, None, None], std[:, None, None]
        self.normalize = normalize
        self.augment = (v2.Compose([v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip()])
                        if simple else v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = self.augment(self.x[i])
        if self.normalize:
            x = (x.float() / 255 - self.mean) / self.std
        return x, self.y[i]


class Cached(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y  # K,N,C,H,W, choose one version per original per epoch

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        k = int(torch.randint(len(self.x), ()).item()) if len(self.x) > 1 else 0
        return self.x[k, i], self.y[i]


def sync(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--download', action='store_true')
    p.add_argument('--images')
    p.add_argument('--labels')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--limit', type=int, default=50000)
    p.add_argument('--batch-size', type=int, default=2048)
    p.add_argument('--only', nargs='*')
    p.add_argument('--output', default='benchmarks/final_results.json')
    args = p.parse_args()
    torch.manual_seed(52)
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    if args.images:
        x, y = images_to_tensor(args.images, args.labels)
        source = str(Path(args.images).resolve())
    else:
        data = torchvision.datasets.CIFAR10(str(ROOT / 'data'), train=True, download=args.download)
        x = torch.from_numpy(data.data).permute(0, 3, 1, 2).contiguous()
        y = torch.tensor(data.targets)
        del data
        source = 'official torchvision CIFAR10 train, 50000 natural RGB images'
    indices = torch.randperm(len(x), generator=torch.Generator().manual_seed(52))[:args.limit]
    n = int(len(indices) * .8)
    xt, xv = x[indices[:n]], x[indices[n:]]
    yt, yv = y[indices[:n]], y[indices[n:]]
    del x, y
    xf, vf = xt.float() / 255, xv.float() / 255
    mean = xf.mean((0, 2, 3))
    std = xf.std((0, 2, 3), correction=0)
    norm = lambda t: (t.float() / 255 - mean[None, :, None, None]) / std[None, :, None, None]
    start = time.perf_counter()
    val_ready = norm(xv)
    val_setup = time.perf_counter() - start
    # Equivalence check on the actual images, including the fused affine formula.
    reference = norm(xt[:32])
    fused = xt[:32].float() * (1 / (255 * std[None, :, None, None])) - mean[None, :, None, None] / std[None, :, None, None]
    torch.testing.assert_close(reference, fused, atol=1e-6, rtol=1e-5)
    result = {'environment': {'python': sys.version, 'platform': platform.platform(),
              'torch': torch.__version__, 'torchvision': torchvision.__version__,
              'device': str(device), 'cpu_threads': torch.get_num_threads(),
              'hardware': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
                          if sys.platform == 'darwin' else platform.processor()},
              'data': {'source': source, 'train': n, 'val': len(xv), 'seed': 52,
                       'batch_size': args.batch_size, 'mean': mean.tolist(), 'std': std.tolist()},
              'method': 'one full unmeasured warmup epoch, then repeated train+val epochs; device sync at phase boundaries; Adam lr=5e-4 wd=1e-4; no plotting/checkpoint IO; same initial weights and shuffle seeds',
              'validation_precompute_seconds': val_setup,
              'normalization_max_abs_error': (reference-fused).abs().max().item(), 'results': []}
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    save()
    torch.manual_seed(52)
    initial = copy.deepcopy(LeNet().state_dict())
    configs = [
        ('baseline', 0, 'sample', True),
        ('baseline_val_precomputed', 0, 'sample', True),
        ('uint8_sample_cpu', 0, 'sample', True),
        ('uint8_batch_cpu', 0, 'cpu', True),
        ('uint8_batch_device', 0, 'device', True),
        ('uint8_workers2', 2, 'device', True),
        ('uint8_workers4', 4, 'device', True),
        ('uint8_workers2_metrics', 2, 'device', False),
        ('uint8_workers4_metrics', 4, 'device', False),
        ('uint8_device_metrics', 0, 'device', False),
        ('cache_uint8_k1', 0, 'device', True),
        ('cache_float32_k1', 0, 'ready', True),
        ('cache_uint8_k4', 0, 'device', True),
        ('cache_float32_k4', 0, 'ready', True),
        ('simple_augmentation', 0, 'device', True),
        ('no_aug_prestandardized', 0, 'ready', True),
        ('no_aug_sample_cpu', 0, 'sample', True),
    ]
    if args.only:
        configs = [c for c in configs if c[0] in args.only]
    # Fixed shuffled order to reduce systematic ordering effects.
    random.Random(52).shuffle(configs)
    for name, workers, mode, items in configs:
        print('START', name, flush=True)
        torch.manual_seed(52)
        start = time.perf_counter()
        cache_bytes = None
        if name in ('baseline', 'baseline_val_precomputed'):
            # Same two dataset wrappers and float -> uint8 -> float round trip as project.
            td = StandardizationDataset(LegacyFloatAugmentation(TensorDataset(xf, yt)), mean, std)
        elif name.startswith('cache_'):
            k = 4 if name.endswith('k4') else 1
            aug = v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10)
            cache = torch.empty((k, *xt.shape), dtype=torch.uint8)
            for j in range(k):
                for i in range(n):
                    cache[j, i] = aug(xt[i])
            if mode == 'ready':
                ready = torch.empty(cache.shape, dtype=torch.float32)
                for j in range(k):
                    for i in range(0, n, args.batch_size):
                        ready[j, i:i+args.batch_size] = norm(cache[j, i:i+args.batch_size])
                del cache
                cache = ready
            cache_bytes = cache.nelement() * cache.element_size()
            td = Cached(cache, yt)
            del cache
            if mode == 'ready':
                del ready
        elif name == 'no_aug_prestandardized':
            td = TensorDataset(norm(xt), yt)
        elif name == 'no_aug_sample_cpu':
            td = StandardizationDataset(TensorDataset(xf, yt), mean, std)
        else:
            td = Augmented(xt, yt, mean, std, normalize=mode == 'sample', simple=name == 'simple_augmentation')
        vd = (StandardizationDataset(TensorDataset(vf, yv), mean, std)
              if name == 'baseline' else TensorDataset(val_ready, yv))
        setup = time.perf_counter() - start
        options = dict(num_workers=workers)
        if workers:
            options.update(persistent_workers=True, prefetch_factor=2)
        dl = DataLoader(td, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(52), **options)
        vl = DataLoader(vd, batch_size=args.batch_size)
        model = LeNet().to(device)
        model.load_state_dict(initial)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()
        scale = (1 / (255 * std)).to(device).reshape(1, 3, 1, 1)
        bias = (-mean / std).to(device).reshape(1, 3, 1, 1)
        records = []
        for rep in range(args.repeats+1):
            torch.manual_seed(52 + rep)
            model.train()
            sync(device)
            start = time.perf_counter()
            loss_sum = torch.zeros((), device=device) if not items else 0.
            correct = torch.zeros((), device=device, dtype=torch.long) if not items else 0
            wait_time = 0.
            iterator = iter(dl)
            while True:
                t = time.perf_counter()
                try:
                    bx, by = next(iterator)
                except StopIteration:
                    break
                wait_time += time.perf_counter()-t
                if mode == 'cpu':
                    bx = norm(bx)
                bx, by = bx.to(device), by.to(device)
                if mode == 'device':
                    bx = bx.float() * scale + bias
                logits = model(bx)
                loss = loss_fn(logits, by)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Preserve the project's exact ordering of host scalar reads.
                loss_sum += (loss.item() if items else loss.detach()) * len(by)
                count = (logits.argmax(1) == by).sum()
                correct += count.item() if items else count
            if not items:
                loss_sum, correct = loss_sum.item(), correct.item()
            sync(device)
            train_s = time.perf_counter()-start
            model.eval()
            start = time.perf_counter()
            with torch.no_grad():
                for bx, by in vl:
                    bx, by = bx.to(device), by.to(device)
                    logits = model(bx)
                    loss = loss_fn(logits, by)
                    # Same synchronization behavior as existing validation loop.
                    loss.item()
                    (logits.argmax(1) == by).sum().item()
            sync(device)
            val_s = time.perf_counter()-start
            record = {'train_seconds': train_s, 'val_seconds': val_s,
                      'total_seconds': train_s+val_s, 'loader_wait_seconds': wait_time}
            print(name, 'warmup' if rep == 0 else f'repeat={rep}', json.dumps(record), flush=True)
            if rep:
                records.append(record)
            else:
                warmup = record
        row = {'name': name, 'workers': workers, 'setup_seconds': setup,
               'warmup': warmup,
               'cache_bytes': cache_bytes, 'epochs': records,
               'median_train_seconds': statistics.median(r['train_seconds'] for r in records),
               'median_val_seconds': statistics.median(r['val_seconds'] for r in records),
               'median_total_seconds': statistics.median(r['total_seconds'] for r in records)}
        result['results'].append(row)
        save()
        if workers and dl._iterator is not None:
            dl._iterator._shutdown_workers()
        del dl, vl, td, vd, model, optimizer, iterator
        gc.collect()
        if device.type == 'mps':
            torch.mps.empty_cache()
    print('SAVED', output, flush=True)


if __name__ == '__main__':
    main()
