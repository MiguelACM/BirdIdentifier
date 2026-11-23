"""How well can the app tell that a photo is not one of its species?

Downloads photos the model has never been trained on and should not recognise:
  - "other birds": European bird species just below the model's list (rank 301-700)
  - "not birds": mammals, insects, spiders, reptiles, amphibians, plants and fungi
and runs the exported ONNX model over them. Their best-match probability is then
compared with the validation photos of known species, to pick the threshold below
which the app says "probably not one of the species I know".

Usage (from the ml/ folder):
    python -m src.eval_unknown           # download (~25 min, resumable), predict, report
    python -m src.eval_unknown --report  # only the report, from saved probabilities
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .download_images import download_one, photo_url
from .inat import AVES_TAXON_ID, EUROPE_PLACE_ID, INatClient
from .predict_split import load_image

ML_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ML_DIR / "data"
OUT_DIR = DATA_DIR / "unknown"

NOT_BIRDS = {  # iNaturalist taxon ids
    "mammals": 40151, "insects": 47158, "spiders": 47119, "reptiles": 26036,
    "amphibians": 20978, "plants": 47126, "fungi": 47170,
}


def observations(client: INatClient, per_page: int, **params) -> list[dict]:
    return client.get("observations", place_id=EUROPE_PLACE_ID, quality_grade="research",
                      photos="true", per_page=per_page, order_by="id", order="desc",
                      **params)["results"]


def other_bird_species(client: INatClient, known: set[int], first: int, last: int) -> list[dict]:
    species = []
    for page in (1, 2):
        data = client.get("observations/species_counts", place_id=EUROPE_PLACE_ID,
                          taxon_id=AVES_TAXON_ID, quality_grade="research", photos="true",
                          per_page=500, page=page)
        species += [r["taxon"] for r in data["results"]
                    if r["taxon"]["rank"] == "species" and " x " not in r["taxon"]["name"]]
    return [t for t in species[first - 1:last] if t["id"] not in known]


def fetch(client: INatClient, per_species: int, per_group: int) -> list[dict]:
    manifest_path = OUT_DIR / "manifest.csv"
    if manifest_path.exists():
        with open(manifest_path, newline="") as f:
            return list(csv.DictReader(f))
    with open(DATA_DIR / "species.csv", newline="") as f:
        known = {int(r["taxon_id"]) for r in csv.DictReader(f)}

    (OUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    picks = []  # (group, taxon name, observation)
    for taxon in other_bird_species(client, known, 301, 700):
        for obs in observations(client, per_species, taxon_id=taxon["id"]):
            picks.append(("other birds", taxon["name"], obs))
    print(f"{len(picks)} photos of other bird species")
    for group, taxon_id in NOT_BIRDS.items():
        seen_taxa: dict[int, int] = {}
        params = {"taxon_id": taxon_id}
        n = 0
        while n < per_group:
            results = observations(client, 200, **params)
            if not results:
                break
            for obs in results:
                t = obs.get("taxon") or {}
                # at most 2 photos per species, so a group isn't one common species
                if seen_taxa.get(t.get("id"), 0) >= 2 or n >= per_group:
                    continue
                seen_taxa[t.get("id")] = seen_taxa.get(t.get("id"), 0) + 1
                picks.append(("not birds", f"{group}: {t.get('name', '')}", obs))
                n += 1
            params["id_below"] = results[-1]["id"]
        print(f"{n} photos of {group}")

    rows = []
    for i, (group, name, obs) in enumerate(picks, 1):
        photo = obs["photos"][0]
        dest = OUT_DIR / "images" / f"{photo['id']}.jpg"
        if dest.exists() or download_one(client, photo_url(photo["url"], "medium"), dest):
            rows.append({"group": group, "taxon": name, "path": dest.name,
                         "observation": obs["id"], "license": photo.get("license_code") or ""})
        if i % 200 == 0:
            print(f"  downloaded {i}/{len(picks)}", flush=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def predict(rows: list[dict], model: Path) -> np.ndarray:
    import json

    import onnxruntime as ort

    config = json.loads((model.parent / "config.json").read_text())
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    probs = np.zeros((len(rows), session.get_outputs()[0].shape[1]), dtype=np.float32)
    for i, row in enumerate(rows):
        x = load_image(OUT_DIR / "images" / row["path"], config["resize"], config["img_size"])
        probs[i] = session.run(None, {name: x})[0][0]
        if (i + 1) % 200 == 0:
            print(f"  predicted {i + 1}/{len(rows)}", flush=True)
    return probs


def report(rows: list[dict], probs: np.ndarray) -> None:
    val = np.load(DATA_DIR / "eval" / "val_probs.npz")
    known_top = val["probs"].max(1)
    known_correct = val["probs"].argmax(1) == val["labels"]
    groups = np.array([r["group"] for r in rows])
    top = probs.max(1)
    print(f"\nKnown species: {len(known_top)} validation photos; "
          f"other birds: {(groups == 'other birds').sum()}; not birds: {(groups == 'not birds').sum()}")
    print("Share flagged as 'not one of my species' by threshold on the best match:")
    print("  threshold  known (correct / wrong)  other birds  not birds")
    for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        flagged = known_top < t
        print(f"  {t:9.1f}  {flagged[known_correct].mean():7.1%} / {flagged[~known_correct].mean():5.1%}"
              f"  {(top[groups == 'other birds'] < t).mean():13.1%}"
              f"  {(top[groups == 'not birds'] < t).mean():9.1%}")
    by_taxon: dict[str, list[float]] = {}
    for r, p in zip(rows, top):
        if r["group"] == "not birds":
            by_taxon.setdefault(r["taxon"].split(":")[0], []).append(p)
    print("Median best match for non-birds: " +
          ", ".join(f"{k} {np.median(v):.2f}" for k, v in by_taxon.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-species", type=int, default=3)
    parser.add_argument("--per-group", type=int, default=150)
    parser.add_argument("--model", type=Path, default=ML_DIR.parent / "web" / "model" / "model.int8.onnx")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    probs_path = DATA_DIR / "eval" / "unknown_probs.npz"
    if args.report:
        with open(OUT_DIR / "manifest.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        report(rows, np.load(probs_path)["probs"])
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = fetch(INatClient(), args.per_species, args.per_group)
    probs = predict(rows, args.model)
    np.savez_compressed(probs_path, probs=probs)
    report(rows, probs)


if __name__ == "__main__":
    main()
