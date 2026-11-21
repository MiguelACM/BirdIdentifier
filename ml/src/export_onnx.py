"""Step 5: export a trained checkpoint to ONNX for the browser.

To keep the web app simple, normalisation and softmax are built into the
exported graph. The browser only has to:
    1. resize the photo so its short side is `resize` px, centre-crop `img_size` px
    2. pass RGB values scaled to 0..1 as a float32 tensor of shape [1, 3, img_size, img_size]
    3. read back one probability per species (same order as labels.json)

Writes into the output folder:
    model.onnx    the network
    labels.json   class index -> species
    config.json   preprocessing settings for the web app

Usage (from the ml/ folder):
    python -m src.export_onnx runs/<run>/best.pt --out ../web/public/model
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn


class BrowserModel(nn.Module):
    """Wraps the classifier: 0..1 RGB in, probabilities out."""

    def __init__(self, model: nn.Module, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std).softmax(dim=1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--out", type=Path, default=None, help="default: next to the checkpoint")
    p.add_argument("--int8", action="store_true",
                   help="also write model.int8.onnx: 4x smaller, ~0.5-1%% less accurate")
    args = p.parse_args()
    out = args.out or args.checkpoint.parent
    out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = timm.create_model(ckpt["model"], pretrained=False, num_classes=ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    wrapped = BrowserModel(model, ckpt["mean"], ckpt["std"]).eval()

    size = ckpt["img_size"]
    dummy = torch.rand(1, 3, size, size)
    onnx_path = out / "model.onnx"
    # Fixed batch size of 1: the browser classifies one photo at a time
    torch.onnx.export(
        wrapped, (dummy,), onnx_path,
        input_names=["image"], output_names=["probs"],
        external_data=False,  # one self-contained file is easiest to serve
    )

    # Sanity check: ONNX Runtime must give the same answer as PyTorch
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    test = torch.rand(1, 3, size, size)
    with torch.no_grad():
        expected = wrapped(test).numpy()
    got = session.run(None, {"image": test.numpy()})[0]
    max_diff = float(np.abs(expected - got).max())
    print(f"ONNX vs PyTorch max difference: {max_diff:.2e}")
    if max_diff > 1e-3:
        raise SystemExit("Exported model does not match PyTorch output!")

    (out / "labels.json").write_text(json.dumps(ckpt["labels"], indent=2, ensure_ascii=False))
    (out / "config.json").write_text(json.dumps({
        "model": ckpt["model"],
        "img_size": size,
        "resize": int(math.floor(size / ckpt["crop_pct"])),
        "input": "float32 RGB in [0,1], shape [1,3,img_size,img_size]",
        "output": "softmax probabilities, shape [1,num_classes]",
    }, indent=2))
    mb = onnx_path.stat().st_size / 1e6
    print(f"Wrote {onnx_path} ({mb:.1f} MB), labels.json, config.json")

    if args.int8:
        # Dynamic quantisation: weights stored as 8-bit integers instead of 32-bit floats
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8_path = out / "model.int8.onnx"
        quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QUInt8)
        print(f"Wrote {int8_path} ({int8_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
