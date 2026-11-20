"""Bundle the dataset + training code into one zip to upload to Kaggle (or Colab).

The zip mirrors the ml/data folder, plus the code:
    raw/images/...   raw/manifest.csv   splits/...   src/...

Photos are shrunk so their short side is --short-side px (default 320). The
model trains on 224 px crops, so this loses nothing useful, makes the upload
~3x smaller and speeds up image loading during training.

The zip is kept under 2 GB on purpose: beyond that Python switches to the
ZIP64 format, which many zip readers (including Kaggle's) reject. ZIP64 is
disabled, so the script fails loudly instead of producing an unreadable file.

Usage (from the ml/ folder):
    python -m src.package_dataset            # -> dist/bird-identifier-data.zip
"""

from __future__ import annotations

import argparse
import io
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from PIL import Image

ML_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ML_DIR / "data"


def shrink(path: Path, short_side: int, quality: int) -> bytes:
    """Return JPEG bytes with the short side resized to `short_side` (never upscales)."""
    with Image.open(path) as img:
        w, h = img.size
        scale = short_side / min(w, h)
        if scale >= 1:
            return path.read_bytes()  # already small enough: keep the original
        img = img.convert("RGB").resize((round(w * scale), round(h * scale)), Image.BICUBIC)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality)
        return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ML_DIR / "dist" / "bird-identifier-data.zip")
    parser.add_argument("--short-side", type=int, default=320)
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in (DATA_DIR / "raw" / "images").rglob("*.jpg"))
    other: list[tuple[Path, str]] = [
        (p, p.relative_to(DATA_DIR).as_posix()) for p in sorted((DATA_DIR / "splits").glob("*"))
    ]
    other += [
        (DATA_DIR / "raw" / "manifest.csv", "raw/manifest.csv"),
        (DATA_DIR / "species.csv", "species.csv"),
    ]
    other += [(p, f"src/{p.name}") for p in sorted((ML_DIR / "src").glob("*.py"))]

    job = partial(shrink, short_side=args.short_side, quality=args.quality)
    try:
        with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_STORED,
                             allowZip64=False) as zf, ProcessPoolExecutor(args.workers) as pool:
            for path, arcname in other:
                zf.write(path, arcname)
            for i, (path, data) in enumerate(
                    zip(images, pool.map(job, images, chunksize=64)), 1):
                zf.writestr(path.relative_to(DATA_DIR).as_posix(), data)
                if i % 5000 == 0:
                    print(f"  {i}/{len(images)} images")
    except zipfile.LargeZipFile:
        args.out.unlink(missing_ok=True)
        raise SystemExit("Archive would exceed 2 GB; use a smaller --short-side or --quality.")

    n = len(images) + len(other)
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1e9:.2f} GB, {n} files)")


if __name__ == "__main__":
    main()
