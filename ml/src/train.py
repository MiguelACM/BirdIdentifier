"""Step 4: fine-tune a pretrained image model on our bird photos.

Takes an ImageNet-pretrained model from timm, replaces its last layer with one
output per bird species, and trains the whole network on data/splits/train.csv.
After every epoch it measures accuracy on the validation set and keeps the best
checkpoint; at the end it evaluates that checkpoint once on the test set.

Usage (from the ml/ folder, or see notebooks/train_kaggle.ipynb):
    python -m src.train --epochs 12
    python -m src.train --limit-per-class 4 --epochs 1 --batch-size 8   # quick smoke test

Outputs go to runs/<model>-<timestamp>/:
    best.pt        checkpoint (weights + everything needed to rebuild the model)
    history.csv    loss / accuracy per epoch
    metrics.json   final test results, per-species accuracy, most-confused pairs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

ML_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- data

class BirdDataset(Dataset):
    """Reads one of the split CSVs (columns: path, label)."""

    def __init__(self, csv_path: Path, images_root: Path, transform, limit_per_class: int = 0):
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
        self.items = [(images_root / r["path"], int(r["label"])) for r in rows]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
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
    per_class_correct: Counter = Counter(l for p, l in zip(preds, labels) if p == l)
    confusions: Counter = Counter((l, p) for p, l in zip(preds, labels) if p != l)
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
    args = p.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"  # mixed precision: faster, less memory on GPU
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name()})" if use_amp else ""))

    splits_dir = args.data_dir / "splits"
    images_root = args.data_dir / "raw"
    labels_info = json.loads((splits_dir / "labels.json").read_text(encoding="utf-8"))
    names = [l["common_name"] or l["scientific_name"] for l in labels_info]

    # ---- model
    model = timm.create_model(args.model, pretrained=not args.no_pretrained, num_classes=len(names))
    cfg = timm.data.resolve_data_config({}, model=model)
    model = model.to(device).to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {args.model} ({n_params:.1f}M params), {len(names)} classes")

    # ---- data
    train_tf, eval_tf = build_transforms(cfg, args.img_size)
    lim = args.limit_per_class
    train_ds = BirdDataset(splits_dir / "train.csv", images_root, train_tf, lim)
    val_ds = BirdDataset(splits_dir / "val.csv", images_root, eval_tf, lim)
    test_ds = BirdDataset(splits_dir / "test.csv", images_root, eval_tf, lim)
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                     pin_memory=use_amp, persistent_workers=args.num_workers > 0)
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

    # ---- training loop
    history = []
    best_top1 = -1.0
    for epoch in range(1, args.epochs + 1):
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
                print(f"  epoch {epoch} step {step}/{len(train_dl)} loss {running_loss / seen:.3f}")

        val = evaluate(model, val_dl, device, criterion, use_amp)
        row = {"epoch": epoch, "train_loss": running_loss / seen, "val_loss": val["loss"],
               "val_top1": val["top1"], "val_top5": val["top5"],
               "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - t0}
        history.append(row)
        improved = val["top1"] > best_top1
        print(f"Epoch {epoch}/{args.epochs}: train loss {row['train_loss']:.3f} | "
              f"val loss {val['loss']:.3f} top1 {val['top1']:.1%} top5 {val['top5']:.1%} | "
              f"{row['seconds']:.0f}s" + ("  * best" if improved else ""))
        if improved:
            best_top1 = val["top1"]
            torch.save({**checkpoint_meta, "epoch": epoch, "state_dict": model.state_dict()},
                       run_dir / "best.pt")

    with (run_dir / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    # ---- final test on the best checkpoint (the test set is used only here, once)
    best = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["state_dict"])
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
