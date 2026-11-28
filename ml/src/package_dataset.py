"""Bundle the dataset + training code into zips to upload to Kaggle (or Colab).

The zips mirror the ml/data folder, plus the code:
    part 1:      raw/images/...   raw/manifest.csv   splits/...   src/...
    parts 2..N:  raw/images/...   (the rest of the photos)

Photos are shrunk so their short side is --short-side px (default 320). That is
about the size of iNaturalist's 500 px "medium" photos anyway, and enough for
288 px training crops, while it makes the upload smaller and loading faster.

Each zip is kept under 2 GB on purpose: beyond that Python switches to the
ZIP64 format, which many zip readers (including Kaggle's) reject. So the
photos are spread over as many parts as needed; upload them all to the same
Kaggle dataset.

With --add-to-existing, the parts already in dist/ are kept (they are already on
Kaggle) and only photos missing from them go into new parts, numbered after them.
The first new part also carries the new splits/manifest/species lists; the notebook
uses the splits from the highest-numbered part that has them.

Usage (from the ml/ folder):
    python -m src.package_dataset                      # -> dist/bird-identifier-data-1.zip, -2.zip, ...
    python -m src.package_dataset --add-to-existing    # only new photos -> data-5.zip, ...
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
    parser.add_argument("--out-dir", type=Path, default=ML_DIR / "dist")
    parser.add_argument("--name", default="bird-identifier-data")
    parser.add_argument("--part-gb", type=float, default=1.9, help="max size of each zip")
    parser.add_argument("--short-side", type=int, default=320)
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    parser.add_argument("--add-to-existing", action="store_true",
                        help="keep the parts in --out-dir and only package photos missing from them")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(args.out_dir.glob(f"{args.name}-*.zip"), key=lambda p: int(p.stem.rsplit("-", 1)[1]))
    already: set[str] = set()
    if args.add_to_existing:
        for part in existing:
            with zipfile.ZipFile(part) as zf:
                already.update(n for n in zf.namelist() if n.endswith(".jpg"))
        print(f"Keeping {len(existing)} existing parts ({len(already)} photos)")
    else:
        for old in existing:
            old.unlink()  # don't leave parts of an older, larger build lying around
        existing = []

    images = sorted(p for p in (DATA_DIR / "raw" / "images").rglob("*.jpg")
                    if p.relative_to(DATA_DIR).as_posix() not in already)
    other: list[tuple[Path, str]] = [
        (p, p.relative_to(DATA_DIR).as_posix()) for p in sorted((DATA_DIR / "splits").glob("*"))
    ]
    other += [
        (DATA_DIR / "raw" / "manifest.csv", "raw/manifest.csv"),
        (DATA_DIR / "species.csv", "species.csv"),
    ]
    if (DATA_DIR / "other_species.csv").exists():
        other.append((DATA_DIR / "other_species.csv", "other_species.csv"))
    other += [(p, f"src/{p.name}") for p in sorted((ML_DIR / "src").glob("*.py"))]

    job = partial(shrink, short_side=args.short_side, quality=args.quality)
    limit = int(args.part_gb * 1e9)
    parts: list[Path] = []

    def new_part() -> zipfile.ZipFile:
        path = args.out_dir / f"{args.name}-{len(existing) + len(parts) + 1}.zip"
        parts.append(path)
        return zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED, allowZip64=False)

    zf = new_part()
    with ProcessPoolExecutor(args.workers) as pool:
        for path, arcname in other:
            zf.write(path, arcname)
        for i, (path, data) in enumerate(zip(images, pool.map(job, images, chunksize=64)), 1):
            if zf.fp.tell() + len(data) + 1000 > limit:  # stored, so file size ~ data written
                zf.close()
                zf = new_part()
            zf.writestr(path.relative_to(DATA_DIR).as_posix(), data)
            if i % 5000 == 0:
                print(f"  {i}/{len(images)} images, part {len(parts)}")
    zf.close()

    for part in parts:
        print(f"Wrote {part} ({part.stat().st_size / 1e9:.2f} GB)")
    print(f"{len(images)} images in {len(parts)} parts")


if __name__ == "__main__":
    main()
