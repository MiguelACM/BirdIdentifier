"""Fetch species names in another language for the web app.

Looks up every species in web/model/labels.json on iNaturalist in the given
locale (optionally preferring the names used in one place) and writes
web/i18n/species.<lang>.json as {taxon_id: name}.

Usage (from the ml/ folder):
    python -m src.fetch_species_names --locale pt --place 7122 --lang pt   # Portugal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .inat import INatClient

WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--locale", required=True, help="iNaturalist locale, e.g. pt, es, fr")
    parser.add_argument("--place", type=int, default=None,
                        help="prefer names used in this iNaturalist place (7122 = Portugal)")
    parser.add_argument("--lang", required=True, help="language code used by the web app")
    parser.add_argument("--labels", type=Path, default=WEB_DIR / "model" / "labels.json")
    args = parser.parse_args()

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    ids = [l["taxon_id"] for l in labels]
    client = INatClient()
    params = {"locale": args.locale}
    if args.place:
        params["preferred_place_id"] = args.place

    names: dict[str, str] = {}
    for i in range(0, len(ids), 30):  # the taxa endpoint takes up to 30 IDs per call
        batch = ",".join(map(str, ids[i:i + 30]))
        for taxon in client.get(f"taxa/{batch}", **params)["results"]:
            if taxon.get("preferred_common_name"):
                names[str(taxon["id"])] = taxon["preferred_common_name"]

    out = WEB_DIR / "i18n" / f"species.{args.lang}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = {str(t): names[str(t)] for t in ids if str(t) in names}
    out.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    missing = len(ids) - len(ordered)
    print(f"Wrote {len(ordered)} names to {out}" + (f" ({missing} missing)" if missing else ""))


if __name__ == "__main__":
    main()
