"""Step 3: split the downloaded photos into train / val / test sets.

The split is deterministic: each photo goes to a set based on a hash of its
observation ID, so rerunning (or downloading more photos later) never moves an
existing photo to a different set, and the test set stays truly unseen.

Writes data/splits/{train,val,test}.csv and data/splits/labels.json
(class index -> species), which the training code and the web app both use.

If data/other_species.csv exists (see fetch_species --others), photos of those
species form one extra class, "Other bird species", as the last label. Species
marked "test" there go only to the test set, so the test can tell how the model
handles birds it has never seen; the rest are split like any other photo.

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
# The "other bird species" class uses iNaturalist's taxon for all birds (Aves), so
# links and names still point somewhere sensible
OTHER_TAXON_ID = 3


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

    other_path = DATA_DIR / "other_species.csv"
    other_split: dict[str, str] = {}
    if other_path.exists():
        with other_path.open(newline="", encoding="utf-8") as f:
            other_split = {r["taxon_id"]: r["split"] for r in csv.DictReader(f)
                           if r["taxon_id"] in present and r["taxon_id"] not in label_of}
    other_label = len(species)

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    unseen_test = skipped = 0
    for p in photos:
        if p["taxon_id"] in label_of:
            label = label_of[p["taxon_id"]]
        elif p["taxon_id"] in other_split:
            label = other_label
            if other_split[p["taxon_id"]] == "test":  # never seen in training
                splits["test"].append({"path": p["path"], "label": label})
                unseen_test += 1
                continue
        else:  # a species dropped from the lists since its photos were downloaded
            skipped += 1
            continue
        b = bucket(p["observation_id"])
        name = "test" if b < args.test else "val" if b < args.test + args.val else "train"
        splits[name].append({"path": p["path"], "label": label})

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
    if other_split:
        labels.append({"index": other_label, "taxon_id": OTHER_TAXON_ID, "scientific_name": "Aves",
                       "common_name": "Other bird species", "other": True})
    (out / "labels.json").write_text(json.dumps(labels, indent=2, ensure_ascii=False))

    for name, rows in splits.items():
        per_class = Counter(r["label"] for r in rows)
        counts = [per_class.get(i, 0) for i in range(len(species))]
        print(f"{name:5s}: {len(rows):6d} photos, min/max per species {min(counts)}/{max(counts)}"
              + (f", other birds {per_class.get(other_label, 0)}" if other_split else ""))
    if other_split:
        print(f"Other bird species: {len(other_split)} species with photos; "
              f"{unseen_test} test photos are of species never used in training")
    if skipped:
        print(f"Skipped {skipped} photos of species no longer in the lists")
    print(f"{len(labels)} classes -> {out}")


if __name__ == "__main__":
    main()
