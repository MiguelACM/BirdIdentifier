"""Find what makes the training process gain memory on Kaggle (~1.7 MB per step).

Runs short training bursts with one ingredient changed at a time and measures how fast
the process's own memory (anonymous RSS, not shared or file cache) grows per step,
after a warm-up. The ingredient whose removal stops the growth is the cause.

Usage (from the ml/ folder):
    python -m src.leak_probe                           # all configurations that fit this machine
    python -m src.leak_probe --data-dir DIR --steps 300 --tracemalloc
"""

from __future__ import annotations

import argparse
import gc
import time
import tracemalloc
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ML_DIR = Path(__file__).resolve().parent.parent


def anon_mb() -> float:
    """This process's private memory (RssAnon), in MB. Linux only."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("RssAnon:"):
            return int(line.split()[1]) / 1024
    return float("nan")


class Synthetic(Dataset):
    """Random images, so the DataLoader and image decoding can be left out or kept in."""

    def __init__(self, n: int, size: int, classes: int):
        self.n, self.size, self.classes = n, size, classes

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        return torch.rand(3, self.size, self.size, generator=g), i % self.classes


def run(name: str, *, model_name: str, size: int, batch: int, steps: int, warmup: int,
        data_parallel: bool, amp: bool, loader: str, workers: int, data_dir: Path | None,
        image_roots: list[Path], channels_last: bool, trace: bool) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core = timm.create_model(model_name, pretrained=False, num_classes=300).to(device)
    if channels_last:
        core = core.to(memory_format=torch.channels_last)
    model = nn.DataParallel(core) if data_parallel else core
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if loader == "none":  # one fixed batch, reused: no DataLoader at all
        fixed = (torch.rand(batch, 3, size, size), torch.randint(0, 300, (batch,)))
        batches = (fixed for _ in range(steps))
    else:
        if loader == "real":
            from .train import BirdDataset, build_transforms

            cfg = timm.data.resolve_data_config({}, model=core)
            ds = BirdDataset(data_dir / "splits" / "train.csv", [data_dir / "raw", *image_roots],
                             build_transforms(cfg, size)[0])
        else:
            ds = Synthetic(steps * batch, size, 300)
        batches = iter(DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers, drop_last=True))

    readings = []
    snapshot = None
    model.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        images, labels = next(batches)
        images = images.to(device, non_blocking=True)
        if channels_last:
            images = images.to(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=amp):
            loss = criterion(model(images), labels)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        loss.item()
        if step == warmup:
            gc.collect()
            if trace:
                tracemalloc.start(25)
                snapshot = tracemalloc.take_snapshot()
        if step >= warmup:
            readings.append((step, anon_mb()))
    gc.collect()
    x, y = np.array(readings).T
    slope = float(np.polyfit(x, y, 1)[0])
    print(f"{name:42s} {slope:+7.3f} MB/step   ({y[0]:.0f} -> {y[-1]:.0f} MB over {len(x)} steps, "
          f"{(time.time() - t0) / steps:.2f} s/step)", flush=True)
    if trace and snapshot is not None:
        diff = tracemalloc.take_snapshot().compare_to(snapshot, "traceback")
        print("  biggest Python-level growth:")
        for stat in diff[:5]:
            print(f"    {stat.size_diff / 1e6:+.2f} MB in {stat.count_diff:+d} blocks at "
                  f"{stat.traceback.format()[-1].strip()}")
        tracemalloc.stop()
    del model, core, opt
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return slope


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="convnext_nano.in12k_ft_in1k")
    p.add_argument("--img-size", type=int, default=288)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--data-dir", type=Path, default=None, help="dataset folder, for the real-photo runs")
    p.add_argument("--extra-image-roots", type=Path, nargs="*", default=[],
                   help="more raw/ folders holding photos (the Kaggle dataset parts)")
    p.add_argument("--tracemalloc", action="store_true",
                   help="also list Python-level allocations (first configuration only)")
    p.add_argument("--only", nargs="*", help="run only configurations whose name contains one of these")
    args = p.parse_args()

    gpus = torch.cuda.device_count()
    common = dict(model_name=args.model, size=args.img_size, batch=args.batch_size, steps=args.steps,
                  warmup=args.warmup, workers=args.workers, data_dir=args.data_dir,
                  image_roots=args.extra_image_roots)
    configs = []
    if args.data_dir:
        configs.append(("as in training (real photos, workers)",
                        dict(data_parallel=gpus > 1, amp=gpus > 0, loader="real", channels_last=True)))
    configs += [
        ("synthetic data through workers",
         dict(data_parallel=gpus > 1, amp=gpus > 0, loader="synthetic", channels_last=True)),
        ("one fixed batch, no DataLoader",
         dict(data_parallel=gpus > 1, amp=gpus > 0, loader="none", channels_last=True)),
        ("  ... without channels_last",
         dict(data_parallel=gpus > 1, amp=gpus > 0, loader="none", channels_last=False)),
    ]
    if gpus > 0:
        configs.append(("  ... without AMP",
                        dict(data_parallel=gpus > 1, amp=False, loader="none", channels_last=True)))
    if gpus > 1:
        configs += [
            ("  ... one GPU (no DataParallel)",
             dict(data_parallel=False, amp=True, loader="none", channels_last=True)),
            ("  ... one GPU, no AMP",
             dict(data_parallel=False, amp=False, loader="none", channels_last=True)),
        ]
    if args.only:
        configs = [c for c in configs if any(o in c[0] for o in args.only)]

    print(f"torch {torch.__version__}, {gpus} GPU(s), {args.model} at {args.img_size}px, "
          f"batch {args.batch_size}, {args.steps} steps (first {args.warmup} not measured)")
    print("Growth of this process's private memory per training step (Kaggle training: ~+1.7):")
    for n, (name, cfg) in enumerate(configs):
        run(name, **common, **cfg, trace=args.tracemalloc and n == 0)


if __name__ == "__main__":
    main()
