"""Step 1: build the list of bird species we will classify.

Asks iNaturalist which bird species have the most research-grade, photographed
observations in Europe, and writes the top N to data/species.csv.

Usage (from the ml/ folder):
    python -m src.fetch_species --top 100
    # v3: keep the current list (and its order), add species up to 400, and list the rarer
    # species beyond them for the "other bird" class
    python -m src.fetch_species --top 400 --keep data/species.csv --others data/other_species.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .inat import AVES_TAXON_ID, EUROPE_PLACE_ID, INatClient
from .make_splits import bucket

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def fetch_species(client: INatClient, top: int, place_id: int) -> list[dict]:
    species: list[dict] = []
    page = 1
    while len(species) < top:
        data = client.get(
            "observations/species_counts",
            place_id=place_id,
            taxon_id=AVES_TAXON_ID,
            quality_grade="research",
            photos="true",
            per_page=500,
            page=page,
        )
        results = data["results"]
        if not results:
            break
        for r in results:
            taxon = r["taxon"]
            # species_counts returns "leaf" taxa; skip hybrids and anything not a species
            if taxon["rank"] != "species" or " x " in taxon["name"]:
                continue
            species.append(
                {
                    "taxon_id": taxon["id"],
                    "scientific_name": taxon["name"],
                    "common_name": taxon.get("preferred_common_name", ""),
                    "observations": r["count"],
                }
            )
        page += 1
    return species[:top]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top", type=int, default=100, help="number of species to keep")
    parser.add_argument("--place-id", type=int, default=EUROPE_PLACE_ID)
    parser.add_argument("--out", type=Path, default=DATA_DIR / "species.csv")
    parser.add_argument("--keep", type=Path, default=None,
                        help="existing species list to keep first, in its order (class indices stay put)")
    parser.add_argument("--others", type=Path, default=None,
                        help="also write the species beyond --top here, for the 'other bird' class")
    parser.add_argument("--others-min-obs", type=int, default=3,
                        help="leave out species with fewer observations (too few photos)")
    parser.add_argument("--others-test", type=float, default=0.3,
                        help="share of the other species kept out of training, to test on unseen species")
    args = parser.parse_args()

    # All species (the 'other' list needs the long tail), most observed first
    everything = fetch_species(INatClient(), 10**6 if args.others else args.top, args.place_id)
    by_id = {s["taxon_id"]: s for s in everything}
    species = []
    if args.keep:
        with args.keep.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                tid = int(row["taxon_id"])
                # fresh observation counts where available; keep the species either way
                species.append(by_id.get(tid, {**row, "taxon_id": tid}))
    kept = {s["taxon_id"] for s in species}
    species += [s for s in everything if s["taxon_id"] not in kept][:args.top - len(species)]
    write_csv(args.out, species)
    print(f"Wrote {len(species)} species to {args.out}" +
          (f" ({len(kept)} kept from {args.keep}, {len(species) - len(kept)} added)" if args.keep else ""))
    print(f"Least common kept: {min(species, key=lambda s: int(s['observations']))['common_name']}")

    if args.others:
        chosen = {s["taxon_id"] for s in species}
        others = [{**s, "split": "test" if bucket(str(s["taxon_id"])) < args.others_test else "train"}
                  for s in everything
                  if s["taxon_id"] not in chosen and s["observations"] >= args.others_min_obs]
        write_csv(args.others, others)
        n_test = sum(o["split"] == "test" for o in others)
        print(f"Wrote {len(others)} other species to {args.others}: {len(others) - n_test} for training, "
              f"{n_test} kept out for testing (observations {others[0]['observations']}"
              f"-{others[-1]['observations']})")


if __name__ == "__main__":
    main()
