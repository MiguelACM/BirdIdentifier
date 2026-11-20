"""Step 3: split the downloaded photos into train / val / test sets.

The split is deterministic: each photo goes to a set based on a hash of its
observation ID, so rerunning (or downloading more photos later) never moves an
existing photo to a different set, and the test set stays truly unseen.

Writes data/splits/{train,val,test}.csv and data/splits/labels.json
(class index -> species), which the training code and the web app both use.

Usage (from the ml/ folder):
    python -m src.make_splits
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def bucket(observation_id: str) -> float:
    """Map an ID to a stable number in [0, 1)."""
    digest = hashlib.sha1(observation_id.encode()).hexdigest()
    return int(digest[:8], 16) / 0x100000000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    args = parser.parse_args()

    with (DATA_DIR / "species.csv").open(newline="", encoding="utf-8") as f:
        species = list(csv.DictReader(f))
    with (DATA_DIR / "raw" / "manifest.csv").open(newline="", encoding="utf-8") as f:
        photos = list(csv.DictReader(f))

    # Only keep species we actually have photos for; class index = order in species.csv
    present = {p["taxon_id"] for p in photos}
    species = [s for s in species if s["taxon_id"] in present]
    label_of = {s["taxon_id"]: i for i, s in enumerate(species)}

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for p in photos:
        b = bucket(p["observation_id"])
        name = "test" if b < args.test else "val" if b < args.test + args.val else "train"
        splits[name].append({"path": p["path"], "label": label_of[p["taxon_id"]]})

    out = DATA_DIR / "splits"
    out.mkdir(exist_ok=True)
    for name, rows in splits.items():
        with (out / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label"])
            writer.writeheader()
            writer.writerows(rows)

    labels = [
        {"index": i, "taxon_id": int(s["taxon_id"]),
         "scientific_name": s["scientific_name"], "common_name": s["common_name"]}
        for i, s in enumerate(species)
    ]
    (out / "labels.json").write_text(json.dumps(labels, indent=2, ensure_ascii=False))

    for name, rows in splits.items():
        per_class = Counter(r["label"] for r in rows)
        counts = [per_class.get(i, 0) for i in range(len(species))]
        print(f"{name:5s}: {len(rows):6d} photos, "
              f"min/max per class {min(counts)}/{max(counts)}")
    print(f"{len(species)} classes -> {out}")


if __name__ == "__main__":
    main()
