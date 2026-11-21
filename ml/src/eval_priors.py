"""Measure whether location + season priors improve the model, and pick their weight.

Combines the model's probabilities with the GBIF priors the same way the web app
does:  score = log p_model + weight * log p_prior  (country + month prior, or the
country's whole-year prior when the date is unknown). The weight is chosen on the
validation set and then reported on the test set, so the test numbers are honest.

Where and when each photo was taken comes from its iNaturalist observation;
the country is found with the same grid the web app uses.

Needs: python -m src.predict_split val test, src.fetch_gbif_priors, src.make_country_grid
Usage (from the ml/ folder):
    python -m src.eval_priors
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .inat import INatClient
from .make_country_grid import lookup

ML_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = ML_DIR.parent / "web"
EVAL_DIR = ML_DIR / "data" / "eval"
WEIGHTS = np.round(np.arange(0, 2.01, 0.1), 2)


def observation_ids() -> dict[int, int]:
    with open(ML_DIR / "data" / "raw" / "manifest.csv", newline="") as f:
        return {int(r["photo_id"]): int(r["observation_id"]) for r in csv.DictReader(f)}


def fetch_observations(obs_ids: list[int]) -> dict[str, dict]:
    """{observation_id: {"lat", "lng", "month"}}, cached in data/eval/observations.json."""
    path = EVAL_DIR / "observations.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    todo = [i for i in obs_ids if str(i) not in cache]
    client = INatClient()
    for start in range(0, len(todo), 200):
        batch = todo[start:start + 200]
        data = client.get("observations", id=",".join(map(str, batch)), per_page=200)
        for obs in data["results"]:
            lat = lng = month = None
            if obs.get("location"):
                lat, lng = map(float, obs["location"].split(","))
            month = (obs.get("observed_on_details") or {}).get("month")
            cache[str(obs["id"])] = {"lat": lat, "lng": lng, "month": month}
        for i in batch:  # deleted observations: remember them so we don't ask again
            cache.setdefault(str(i), {"lat": None, "lng": None, "month": None})
        path.write_text(json.dumps(cache))
        print(f"  observations: {min(start + 200, len(todo))}/{len(todo)}")
    return cache


def log_priors(split: str, obs_of_photo: dict, observations: dict, grid: dict,
               priors: dict, use_month: bool = True) -> tuple[np.ndarray, np.ndarray, list]:
    data = np.load(EVAL_DIR / f"{split}_probs.npz")
    n_classes = data["probs"].shape[1]
    logp = np.zeros((len(data["labels"]), n_classes))
    has = np.zeros(len(data["labels"]), dtype=bool)
    countries = []
    scale = priors["scale"]
    for i, photo_id in enumerate(data["photo_ids"]):
        obs = observations.get(str(obs_of_photo[int(photo_id)]), {})
        country = lookup(grid, obs["lat"], obs["lng"]) if obs.get("lat") is not None else ""
        countries.append(country)
        entry = priors["countries"].get(country)
        if not entry:
            continue
        month = obs.get("month")
        values = entry["months"][month - 1] if (use_month and month) else entry["year"]
        logp[i] = np.log(np.asarray(values) / scale)
        has[i] = True
    return logp, has, countries


def accuracy(probs, labels, logp, weight):
    scores = np.log(np.clip(probs, 1e-9, 1)) + weight * logp
    top5 = np.argsort(-scores, axis=1)[:, :5]
    return (top5[:, 0] == labels).mean(), (top5 == labels[:, None]).any(1).mean()


def main() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    grid = json.loads((WEB_DIR / "country-grid.json").read_text())
    priors = json.loads((WEB_DIR / "priors.json").read_text())
    obs_of_photo = observation_ids()

    splits = {s: np.load(EVAL_DIR / f"{s}_probs.npz") for s in ("val", "test")}
    wanted = sorted({obs_of_photo[int(p)] for d in splits.values() for p in d["photo_ids"]})
    observations = fetch_observations(wanted)

    results = {}
    for split, data in splits.items():
        logp, has, countries = log_priors(split, obs_of_photo, observations, grid, priors)
        logp_year, _, _ = log_priors(split, obs_of_photo, observations, grid, priors, use_month=False)
        results[split] = (data["probs"], data["labels"], logp, logp_year, has, countries)

    probs, labels, logp, logp_year, has, _ = results["val"]
    val_curve = [accuracy(probs, labels, logp, w) for w in WEIGHTS]
    best = WEIGHTS[int(np.argmax([t1 for t1, _ in val_curve]))]
    print("\nValidation top-1 by prior weight:")
    for w, (t1, t5) in zip(WEIGHTS, val_curve):
        print(f"  {w:.1f}: top-1 {t1:.1%}  top-5 {t5:.1%}" + ("  <- best" if w == best else ""))

    probs, labels, logp, logp_year, has, countries = results["test"]
    print(f"\nTest set ({len(labels)} photos, {has.mean():.0%} with a supported country), weight {best}:")
    for name, lp, w in [("photo only", logp, 0), ("+ country, whole year", logp_year, best),
                        ("+ country and month", logp, best)]:
        t1, t5 = accuracy(probs, labels, lp, w)
        print(f"  {name:24s} top-1 {t1:.1%}  top-5 {t5:.1%}")
    t1a, _ = accuracy(probs[has], labels[has], logp[has], 0)
    t1b, _ = accuracy(probs[has], labels[has], logp[has], best)
    print(f"  photos with a country only: {t1a:.1%} -> {t1b:.1%}")
    top = sorted({c for c in countries}, key=countries.count, reverse=True)[:8]
    print("  most common countries:", ", ".join(f"{c or '?'} {countries.count(c)}" for c in top))


if __name__ == "__main__":
    main()
