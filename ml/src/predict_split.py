"""Run the exported ONNX model over a split and save its probabilities.

Used to evaluate things that happen after the model, like location priors,
without re-running the model each time. Preprocessing matches the web app and
the validation transform: short side to `resize` px, centre crop `img_size` px.

Usage (from the ml/ folder):
    python -m src.predict_split val test   # -> data/eval/<split>_probs.npz (resumable)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ML_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ML_DIR.parent / "web" / "model"


def load_image(path: Path, resize: int, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = resize / min(w, h)
    img = img.resize((max(size, round(w * scale)), max(size, round(h * scale))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    return (np.asarray(img, dtype=np.float32) / 255).transpose(2, 0, 1)[None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("splits", nargs="+")
    parser.add_argument("--model", type=Path, default=MODEL_DIR / "model.int8.onnx")
    parser.add_argument("--data", type=Path, default=ML_DIR / "data")
    args = parser.parse_args()

    config = json.loads((args.model.parent / "config.json").read_text())
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    out_dir = args.data / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        with open(args.data / "splits" / f"{split}.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        probs = np.zeros((len(rows), session.get_outputs()[0].shape[1]), dtype=np.float32)
        # Progress is saved every 500 photos, so a stopped run carries on where it was
        partial = out_dir / f"{split}_probs.partial.npz"
        start = 0
        if partial.exists():
            saved = np.load(partial)
            if saved["probs"].shape == probs.shape:
                start = int(saved["n"])
                probs[:start] = saved["probs"][:start]
                print(f"  {split}: continuing after {start} photos", flush=True)
        for i in range(start, len(rows)):
            x = load_image(args.data / "raw" / rows[i]["path"], config["resize"], config["img_size"])
            probs[i] = session.run(None, {input_name: x})[0][0]
            if (i + 1) % 500 == 0:
                np.savez(partial, probs=probs, n=i + 1)
                print(f"  {split}: {i + 1}/{len(rows)}", flush=True)
        labels = np.array([int(r["label"]) for r in rows])
        photo_ids = np.array([int(Path(r["path"]).stem) for r in rows])
        np.savez_compressed(out_dir / f"{split}_probs.npz", probs=probs, labels=labels,
                            photo_ids=photo_ids)
        partial.unlink(missing_ok=True)
        top1 = (probs.argmax(1) == labels).mean()
        print(f"{split}: {len(rows)} photos, top-1 {top1:.1%}", flush=True)


if __name__ == "__main__":
    main()
