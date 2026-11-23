"""Step 4: fine-tune a pretrained image model on our bird photos.

Takes an ImageNet-pretrained model from timm, replaces its last layer with one
output per bird species, and trains the whole network on data/splits/train.csv.
After every epoch it measures accuracy on the validation set and keeps the best
checkpoint; at the end it evaluates that checkpoint once on the test set.

Usage (from the ml/ folder, or see notebooks/train_kaggle.ipynb):
    python -m src.train --epochs 12
    python -m src.train --limit-per-class 4 --epochs 1 --batch-size 8   # quick smoke test

With more than one GPU (e.g. Kaggle's 2x T4) it trains on all of them. Photos
can be spread over several folders (the Kaggle dataset comes in parts of < 2 GB):
pass the extra ones with --extra-image-roots.

If a run dies part-way (e.g. Kaggle runs out of memory), restart it with
--resume <its run folder> to carry on from the last finished epoch.

Outputs go to runs/<model>-<timestamp>/:
    best.pt        checkpoint (weights + everything needed to rebuild the model)
    last.pt        latest epoch with optimiser state, for --resume
    history.csv    loss / accuracy per epoch
    metrics.json   final test results, per-species accuracy, most-confused pairs
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

ML_DIR = Path(__file__).resolve().parent.parent
RESTART_EXIT_CODE = 3  # "saved, please resume me in a new process" (--restart-above-gb)


# ---------------------------------------------------------------- data

def find_image(roots: list[Path], rel_path: str) -> Path:
    """The photo's path in whichever root folder holds it."""
    for root in roots:
        path = root / rel_path
        if path.exists():
            return path
    raise FileNotFoundError(f"{rel_path} not found in any of: {', '.join(map(str, roots))}")


class BirdDataset(Dataset):
    """Reads one of the split CSVs (columns: path, label)."""

    def __init__(self, csv_path: Path, images_roots: list[Path], transform, limit_per_class: int = 0):
        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if limit_per_class:  # handy for quick tests
            seen: Counter = Counter()
            kept = []
            for r in rows:
                seen[r["label"]] += 1
                if seen[r["label"]] <= limit_per_class:
                    kept.append(r)
            rows = kept
        # Stored as two numpy arrays rather than a list of Python objects: every
        # DataLoader worker is a forked copy of this process, and merely reading
        # Python objects updates their reference counts, which makes each worker
        # slowly copy the whole list into its own memory. With 120k photos and
        # several workers that ran Kaggle out of RAM after two hours.
        self.paths = np.array([str(find_image(images_roots, r["path"])).encode() for r in rows])
        self.labels = np.array([int(r["label"]) for r in rows], dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i):
        path, label = self.paths[i].decode(), int(self.labels[i])
        with Image.open(path) as img:
            img = img.convert("RGB")
        return self.transform(img), label


def build_transforms(cfg: dict, img_size: int):
    interp = T.InterpolationMode.BICUBIC
    normalize = T.Normalize(cfg["mean"], cfg["std"])
    train = T.Compose([
        # Random crops/zoom teach the model that birds appear at many sizes and positions
        T.RandomResizedCrop(img_size, scale=(0.35, 1.0), interpolation=interp),
        T.RandomHorizontalFlip(),
        # Mild brightness/contrast changes for different lighting. No hue shift:
        # plumage colour is often what tells two species apart.
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(),
        normalize,
    ])
    resize_to = int(math.floor(img_size / cfg["crop_pct"]))
    evaluate = T.Compose([
        T.Resize(resize_to, interpolation=interp),
        T.CenterCrop(img_size),
        T.ToTensor(),
        normalize,
    ])
    return train, evaluate


# ---------------------------------------------------------------- train / eval

@torch.no_grad()
def evaluate(model, loader, device, criterion, use_amp: bool):
    model.eval()
    total, loss_sum, top1, top5 = 0, 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        k = min(5, logits.shape[1])
        topk = logits.topk(k, dim=1).indices
        top1 += (topk[:, 0] == labels).sum().item()
        top5 += (topk == labels[:, None]).any(dim=1).sum().item()
        loss_sum += loss.item() * labels.size(0)
        total += labels.size(0)
        all_preds.append(topk[:, 0].cpu())
        all_labels.append(labels.cpu())
    return {
        "loss": loss_sum / total,
        "top1": top1 / total,
        "top5": top5 / total,
        "preds": torch.cat(all_preds).tolist(),
        "labels": torch.cat(all_labels).tolist(),
    }


def error_analysis(preds: list[int], labels: list[int], names: list[str]) -> dict:
    per_class_total: Counter = Counter(labels)
    per_class_correct: Counter = Counter(y for p, y in zip(preds, labels) if p == y)
    confusions: Counter = Counter((y, p) for p, y in zip(preds, labels) if p != y)
    per_class = sorted(
        ({"species": names[c], "accuracy": per_class_correct[c] / n, "n": n}
         for c, n in per_class_total.items()),
        key=lambda d: d["accuracy"],
    )
    most_confused = [
        {"true": names[t], "predicted": names[p], "count": n}
        for (t, p), n in confusions.most_common(20)
    ]
    return {"per_class": per_class, "most_confused": most_confused}


def ram_used_gb() -> tuple[float, float, int]:
    """(main process GB, DataLoader workers GB, number of workers); zeros if psutil is missing.
    Counts USS, the memory private to each process, so shared pages aren't counted twice."""
    try:
        import psutil
    except ImportError:
        return 0.0, 0.0, 0
    me = psutil.Process()
    main, workers, n = 0, 0, 0
    for proc in [me, *me.children(recursive=True)]:
        try:
            uss = proc.memory_full_info().uss
        except psutil.Error:
            continue
        if proc.pid == me.pid:
            main = uss
        else:
            workers += uss
            n += 1
    return main / 1e9, workers / 1e9, n


def container_memory() -> dict[str, float] | None:
    """Memory of the whole container as its memory limit sees it, in GB, split into
    "anon" (process memory), "shmem" (shared memory such as /dev/shm, which DataLoader
    workers use to hand batches over) and "file" (disk cache, freed when needed).
    Reads the cgroup files (v2, then v1); None where they don't exist (e.g. macOS)."""
    def read_stat(path: Path) -> dict[str, int]:
        return {k: int(v) for k, v in (line.split() for line in path.read_text().splitlines())}
    try:
        # cgroup v2: this process's own group (inside a container that's usually "/")
        group = Path("/proc/self/cgroup").read_text().splitlines()[0].split("::", 1)[-1]
        v2 = Path("/sys/fs/cgroup") / group.lstrip("/")
        if (v2 / "memory.stat").exists():
            stat = read_stat(v2 / "memory.stat")
            max_file = v2 / "memory.max"
            limit = max_file.read_text().strip() if max_file.exists() else "max"
            anon, shmem, file = stat["anon"], stat["shmem"], stat["file"] - stat["shmem"]
        else:
            stat = read_stat(Path("/sys/fs/cgroup/memory/memory.stat"))
            limit = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes").read_text().strip()
            anon = stat["total_rss"]
            shmem = stat["total_shmem"]
            file = stat["total_cache"] - shmem
    except (OSError, KeyError, ValueError):
        return None
    total_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    limit = min(int(limit), total_ram) if limit.isdigit() else total_ram
    return {"anon": anon / 1e9, "shmem": shmem / 1e9, "file": file / 1e9, "limit": limit / 1e9}


def release_free_memory() -> None:
    """Return freed heap memory to the OS (glibc only), so the log shows real growth
    rather than memory that is free but still held by malloc."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def memory_summary() -> tuple[str, float]:
    """A short memory report for the log, and the GB that can't be freed (process +
    shared memory), which is what gets a run killed when it reaches the limit."""
    mem = container_memory()
    main, workers, n = ram_used_gb()
    procs = f"main {main:.1f} + {n} workers {workers:.1f}"
    if mem is None:
        return f"RAM {procs} GB", main + workers
    used = mem["anon"] + mem["shmem"]
    return (f"RAM {used:.1f}/{mem['limit']:.0f} GB (processes {mem['anon']:.1f}, shared {mem['shmem']:.1f}; "
            f"{procs}; disk cache {mem['file']:.1f})"), used


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", type=Path, default=ML_DIR / "data",
                   help="folder containing raw/images/ and splits/")
    p.add_argument("--out-dir", type=Path, default=ML_DIR / "runs")
    p.add_argument("--model", default="convnext_nano.in12k_ft_in1k")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 2))
    p.add_argument("--limit-per-class", type=int, default=0, help="use only N images per class")
    p.add_argument("--no-pretrained", action="store_true", help="start from random weights")
    p.add_argument("--extra-image-roots", type=Path, nargs="*", default=[],
                   help="more folders containing images/<taxon>/<photo>.jpg")
    p.add_argument("--max-hours", type=float, default=0,
                   help="stop starting new epochs once training would run past this (0 = no limit), "
                        "so a run on a time-limited service still saves its results")
    p.add_argument("--restart-above-gb", type=float, default=0,
                   help=f"after an epoch, if process + shared memory is over this, save and exit with "
                        f"code {RESTART_EXIT_CODE} so the caller can resume in a fresh process (0 = never)")
    p.add_argument("--max-ram-gb", type=float, default=0,
                   help="stop training as soon as process + shared memory passes this (0 = no limit), "
                        "so the run still tests and saves the best model instead of being killed")
    p.add_argument("--pin-memory", action="store_true",
                   help="use pinned memory for GPU copies (off by default, see the comment in main)")
    p.add_argument("--resume", type=Path, default=None,
                   help="folder of an interrupted run (with last.pt and best.pt) to continue from")
    args = p.parse_args()
    start_time = time.time()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"  # mixed precision: faster, less memory on GPU
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name()})" if use_amp else ""))

    splits_dir = args.data_dir / "splits"
    images_roots = [args.data_dir / "raw", *args.extra_image_roots]
    labels_info = json.loads((splits_dir / "labels.json").read_text(encoding="utf-8"))
    names = [x["common_name"] or x["scientific_name"] for x in labels_info]

    # ---- model
    model = timm.create_model(args.model, pretrained=not args.no_pretrained, num_classes=len(names))
    cfg = timm.data.resolve_data_config({}, model=model)
    core = model.to(device).to(memory_format=torch.channels_last)  # the unwrapped model, for saving
    n_params = sum(p.numel() for p in core.parameters()) / 1e6
    print(f"Model: {args.model} ({n_params:.1f}M params), {len(names)} classes")
    n_gpus = torch.cuda.device_count()
    # DataParallel splits each batch across the GPUs: simple, and ~1.7x faster on 2 GPUs
    model = nn.DataParallel(core) if n_gpus > 1 else core
    if n_gpus > 1:
        print(f"Training on {n_gpus} GPUs")

    # ---- data
    train_tf, eval_tf = build_transforms(cfg, args.img_size)
    lim = args.limit_per_class
    train_ds = BirdDataset(splits_dir / "train.csv", images_roots, train_tf, lim)
    val_ds = BirdDataset(splits_dir / "val.csv", images_roots, eval_tf, lim)
    test_ds = BirdDataset(splits_dir / "test.csv", images_roots, eval_tf, lim)
    # No pin_memory: on Kaggle the main process grew ~1.6 MB per step with it on
    # (about 3 GB per epoch) until the run was killed. The speed cost is negligible here.
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                     pin_memory=use_amp and args.pin_memory)
    # Workers are started fresh every epoch (no persistent_workers): costs a few
    # seconds, but anything that builds up inside them is released each time
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=len(train_ds) > args.batch_size, **loader_kw)
    val_dl = DataLoader(val_ds, **loader_kw)
    test_dl = DataLoader(test_ds, **loader_kw)
    print(f"Images: train {len(train_ds)}, val {len(val_ds)}, test {len(test_ds)}")

    # ---- optimisation
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_dl)
    warmup_steps = int(args.warmup_epochs * len(train_dl))

    def lr_factor(step: int) -> float:
        # Linear warm-up (don't wreck the pretrained weights early on), then cosine decay to 0
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    run_dir = args.out_dir / f"{args.model.split('.')[0]}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_meta = {
        "model": args.model,
        "num_classes": len(names),
        "img_size": args.img_size,
        "mean": list(cfg["mean"]),
        "std": list(cfg["std"]),
        "crop_pct": cfg["crop_pct"],
        "labels": labels_info,
        "args": {k: str(v) for k, v in vars(args).items()},
    }

    # ---- resume an interrupted run (its last.pt holds the optimiser state too)
    history = []
    best_top1 = -1.0
    first_epoch = 1
    if args.resume:
        state = torch.load(args.resume / "last.pt", map_location=device, weights_only=False)
        core.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history, best_top1 = state["history"], state["best_top1"]
        first_epoch = state["epoch"] + 1
        # Copy both, so this run's folder can be resumed from even if it stops early
        shutil.copy(args.resume / "best.pt", run_dir / "best.pt")
        shutil.copy(args.resume / "last.pt", run_dir / "last.pt")
        print(f"Resuming from {args.resume} after epoch {state['epoch']} (best val top-1 {best_top1:.1%})")

    # ---- training loop
    out_of_memory = False
    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss, seen = 0.0, 0
        for step, (images, labels) in enumerate(train_dl, 1):
            images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=use_amp):
                loss = criterion(model(images), labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
            if step % 50 == 0:
                mem_text, mem_used = memory_summary()
                print(f"  epoch {epoch} step {step}/{len(train_dl)} loss {running_loss / seen:.3f}"
                      + (f" | {mem_text}" if step % 200 == 0 else ""))
                if args.max_ram_gb and mem_used > args.max_ram_gb:
                    out_of_memory = True
                    break
        if out_of_memory:
            # Throw away this unfinished epoch; best.pt / last.pt hold the last complete one
            print(f"Stopping in epoch {epoch}: memory {mem_used:.1f} GB is over --max-ram-gb {args.max_ram_gb} "
                  f"({mem_text}). Continue with --resume {run_dir}")
            break

        release_free_memory()
        val = evaluate(model, val_dl, device, criterion, use_amp)
        row = {"epoch": epoch, "train_loss": running_loss / seen, "val_loss": val["loss"],
               "val_top1": val["top1"], "val_top5": val["top5"],
               "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - t0}
        history.append(row)
        improved = val["top1"] > best_top1
        mem_text, mem_used = memory_summary()
        print(f"Epoch {epoch}/{args.epochs}: train loss {row['train_loss']:.3f} | "
              f"val loss {val['loss']:.3f} top1 {val['top1']:.1%} top5 {val['top5']:.1%} | "
              f"{row['seconds']:.0f}s | {mem_text}" + ("  * best" if improved else ""))
        if improved:
            best_top1 = val["top1"]
            torch.save({**checkpoint_meta, "epoch": epoch, "state_dict": core.state_dict()},
                       run_dir / "best.pt")
        # Everything needed to carry on with --resume if the run dies. Written to a
        # temporary file first, so a crash mid-save can't leave a broken last.pt
        torch.save({"epoch": epoch, "state_dict": core.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "history": history, "best_top1": best_top1}, run_dir / "last.pt.tmp")
        os.replace(run_dir / "last.pt.tmp", run_dir / "last.pt")
        if args.restart_above_gb and epoch < args.epochs and mem_used > args.restart_above_gb:
            # Memory grows ~1.7 MB per step in this process on Kaggle (cause not found:
            # not the DataLoader workers, not pin_memory). A fresh process starts clean,
            # so hand over to the caller, which resumes from last.pt (see the notebook).
            print(f"Memory {mem_used:.1f} GB is over --restart-above-gb {args.restart_above_gb}: "
                  f"exiting so a fresh process can continue with --resume {run_dir}", flush=True)
            sys.exit(RESTART_EXIT_CODE)

        elapsed_h = (time.time() - start_time) / 3600
        per_epoch_h = elapsed_h / (epoch - first_epoch + 1)
        if args.max_hours and epoch < args.epochs and elapsed_h + per_epoch_h > args.max_hours:
            print(f"Stopping after epoch {epoch}: another epoch would pass --max-hours "
                  f"{args.max_hours} ({elapsed_h:.1f} h used, {per_epoch_h * 60:.0f} min per epoch)")
            break

    if not history:
        raise SystemExit("No epoch finished, nothing to test or save")
    with (run_dir / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    # ---- final test on the best checkpoint (the test set is used only here, once)
    best = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    core.load_state_dict(best["state_dict"])
    test = evaluate(model, test_dl, device, criterion, use_amp)
    analysis = error_analysis(test["preds"], test["labels"], names)
    metrics = {"best_epoch": best["epoch"], "val_top1": best_top1,
               "test_top1": test["top1"], "test_top5": test["top5"], **analysis}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    print(f"\nTest (best epoch {best['epoch']}): top1 {test['top1']:.1%}  top5 {test['top5']:.1%}")
    print("Hardest species:", ", ".join(
        f"{d['species']} {d['accuracy']:.0%}" for d in analysis["per_class"][:5]))
    print("Most confused:", "; ".join(
        f"{d['true']} -> {d['predicted']} ({d['count']})" for d in analysis["most_confused"][:5]))
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
