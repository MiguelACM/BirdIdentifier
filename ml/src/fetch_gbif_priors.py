"""Build location + season priors for the web app from GBIF sighting counts.

For every European country and month, asks GBIF how many records there are of
each of the model's species, then turns the counts into "how likely is each
species here and now, among the species the model knows". The web app multiplies
these into the model's probabilities to re-rank its guesses.

Most GBIF bird records come from eBird and iNaturalist, so counts follow where
birders go. Dividing by the country-month total removes most of that effort
bias: what remains is each species' share of sightings in that place and month.

Usage (from the ml/ folder):
    python -m src.fetch_gbif_priors            # ~540 requests, ~10 min, resumable
    python -m src.fetch_gbif_priors --offline  # rebuild web/priors.json from the cache
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from .inat import INatClient

GBIF_URL = "https://api.gbif.org/v1"
ML_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = ML_DIR.parent / "web"

# ISO 3166 codes as GBIF uses them. Turkey is left out: almost all of it is in Asia.
COUNTRIES = [
    "AD", "AL", "AT", "AX", "BA", "BE", "BG", "BY", "CH", "CY", "CZ", "DE", "DK",
    "EE", "ES", "FI", "FO", "FR", "GB", "GG", "GI", "GR", "HR", "HU", "IE", "IM",
    "IS", "IT", "JE", "LI", "LT", "LU", "LV", "MC", "MD", "ME", "MK", "MT", "NL",
    "NO", "PL", "PT", "RO", "RS", "RU", "SE", "SI", "SK", "SM", "UA", "VA", "XK",
]
# Extra GBIF filters per country. Russia: only records west of the Urals (60°E),
# so Siberia's birds don't distort the priors for European Russia.
COUNTRY_FILTERS = {"RU": {"decimalLongitude": "19,60"}}

# Smoothing, in pseudo-records. A country-month with few records leans on the
# country's whole-year shares, and a country with few records on Europe's.
K_MONTH = 300
K_COUNTRY = 1000
SCALE = 10000  # priors are stored as integers out of SCALE (min 1)


AVES_KEY = 212  # GBIF backbone key of class Aves

# iNaturalist names that GBIF's backbone doesn't know yet (newer genus moves)
GBIF_NAMES = {
    "Thinornis dubius": "Charadrius dubius",  # Little Ringed Plover
    "Astur gentilis": "Accipiter gentilis",  # Eurasian Goshawk
    "Botaurus minutus": "Ixobrychus minutus",  # Little Bittern
    # GBIF still lumps the European Red-rumped Swallow into the Red-rumped Swallow;
    # in Europe they are the same birds
    "Cecropis rufula": "Cecropis daurica",
}


def match_species(client: INatClient, labels: list[dict]) -> list[int]:
    """Find the GBIF species key for each label by scientific name."""
    keys = []
    for label in labels:
        name = GBIF_NAMES.get(label["scientific_name"], label["scientific_name"])
        # kingdom/class disambiguate names shared with plants (e.g. Chloris chloris)
        m = client.get("species/match", name=name, rank="SPECIES",
                       kingdom="Animalia", **{"class": "Aves"})
        key = m.get("acceptedUsageKey") or m.get("usageKey")
        if m.get("rank") != "SPECIES" or m.get("class") != "Aves" or not key:
            raise SystemExit(f"No GBIF species match for {label['scientific_name']}: {m}")
        if m.get("acceptedUsageKey"):
            print(f"  {label['scientific_name']} is a synonym in GBIF -> {key}")
        keys.append(key)
    return keys


def fetch_counts(client: INatClient, keys: list[int], cache_path: Path) -> dict:
    """counts[country][month-1][i] = GBIF records of species i. Cached and resumable."""
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if cache.get("keys") != keys:
        cache = {"keys": keys, "counts": {}}
    counts = cache["counts"]
    todo = [(c, m) for c in COUNTRIES for m in range(1, 13)
            if len(counts.get(c, [])) < m]
    print(f"{len(todo)} country-months to fetch")
    for n, (country, month) in enumerate(todo, 1):
        # Counts for every bird species (class Aves) at once; ours are picked out below.
        # Listing our species as taxonKey filters instead breaks past ~200 (URL too long).
        data = client.get("occurrence/search", classKey=AVES_KEY, country=country, month=month,
                          occurrenceStatus="PRESENT", limit=0, **COUNTRY_FILTERS.get(country, {}),
                          facet="speciesKey", facetLimit=5000)
        by_key = {int(f["name"]): f["count"] for f in data["facets"][0]["counts"]} \
            if data.get("facets") else {}
        counts.setdefault(country, []).append([by_key.get(k, 0) for k in keys])
        if n % 12 == 0 or n == len(todo):
            cache_path.write_text(json.dumps(cache))
            print(f"  {n}/{len(todo)} ({country})")
    return counts


def shares(counts: list[int], fallback: list[float], k: float) -> list[float]:
    total = sum(counts)
    return [(c + k * f) / (total + k) for c, f in zip(counts, fallback)]


def to_ints(p: list[float]) -> list[int]:
    return [max(1, round(x * SCALE)) for x in p]


def build_priors(counts: dict, n: int) -> dict:
    europe = [sum(counts[c][m][i] for c in counts for m in range(12)) for i in range(n)]
    europe_share = shares(europe, [1 / n] * n, 0)
    out = {}
    for country in COUNTRIES:
        months = counts[country]
        year = [sum(months[m][i] for m in range(12)) for i in range(n)]
        year_share = shares(year, europe_share, K_COUNTRY)
        out[country] = {
            "records": sum(year),
            "year": to_ints(year_share),
            "months": [to_ints(shares(months[m], year_share, K_MONTH)) for m in range(12)],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", type=Path, default=WEB_DIR / "model" / "labels.json")
    parser.add_argument("--cache", type=Path, default=ML_DIR / "data" / "gbif" / "counts.json")
    parser.add_argument("--out", type=Path, default=WEB_DIR / "priors.json")
    parser.add_argument("--offline", action="store_true", help="only rebuild from the cache")
    args = parser.parse_args()

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    if args.offline:
        cache = json.loads(args.cache.read_text())
        counts, fetched = cache["counts"], cache.get("fetched", "")
    else:
        client = INatClient(api_url=GBIF_URL, min_interval=0.5)
        keys = match_species(client, labels)
        counts = fetch_counts(client, keys, args.cache)
        fetched = dt.date.today().isoformat()
        cache = json.loads(args.cache.read_text())
        cache["fetched"] = fetched
        args.cache.write_text(json.dumps(cache))

    priors = {
        "source": f"GBIF.org occurrence counts per country and month, fetched {fetched}",
        "scale": SCALE,
        "taxa": [label["taxon_id"] for label in labels],
        "countries": build_priors(counts, len(labels)),
    }
    # One country per line keeps the file small but still diffable
    lines = [f'  "{c}": {json.dumps(v, separators=(",", ":"))}' for c, v in priors["countries"].items()]
    head = {k: v for k, v in priors.items() if k != "countries"}
    text = json.dumps(head, separators=(",", ":"))[:-1] + ',"countries":{\n' + ",\n".join(lines) + "\n}}\n"
    args.out.write_text(text, encoding="utf-8")
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
