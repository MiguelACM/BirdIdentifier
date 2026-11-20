"""Step 1: build the list of bird species we will classify.

Asks iNaturalist which bird species have the most research-grade, photographed
observations in Europe, and writes the top N to data/species.csv.

Usage (from the ml/ folder):
    python -m src.fetch_species --top 100
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .inat import AVES_TAXON_ID, EUROPE_PLACE_ID, INatClient

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top", type=int, default=100, help="number of species to keep")
    parser.add_argument("--place-id", type=int, default=EUROPE_PLACE_ID)
    parser.add_argument("--out", type=Path, default=DATA_DIR / "species.csv")
    args = parser.parse_args()

    species = fetch_species(INatClient(), args.top, args.place_id)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(species[0].keys()))
        writer.writeheader()
        writer.writerows(species)

    print(f"Wrote {len(species)} species to {args.out}")
    print(f"Least common kept: {species[-1]['common_name']} ({species[-1]['observations']} obs)")


if __name__ == "__main__":
    main()
