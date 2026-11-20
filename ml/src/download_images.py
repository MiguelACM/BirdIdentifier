"""Step 2: download training photos for every species in data/species.csv.

For each species we page through research-grade European observations and take
the FIRST photo of each observation (several photos of the same bird would be
near-duplicates and could leak between train and test sets).

Images are saved as   data/raw/images/<taxon_id>/<photo_id>.jpg
and every image is recorded in data/raw/manifest.csv with its license and
author, so we always know where each photo came from.

The script is resumable: rerun it and it skips photos it already has.

Usage (from the ml/ folder):
    python -m src.download_images --per-species 300
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

from PIL import Image

from .inat import EUROPE_PLACE_ID, INatClient

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
IMAGE_DIR = RAW_DIR / "images"
MANIFEST = RAW_DIR / "manifest.csv"
FIELDS = [
    "photo_id", "observation_id", "taxon_id", "path",
    "license", "attribution", "observer", "url",
]


def photo_url(square_url: str, size: str) -> str:
    # iNaturalist photo URLs end in square.jpg; swap for small/medium/large/original
    return square_url.replace("/square.", f"/{size}.")


def iter_observations(client: INatClient, taxon_id: int, place_id: int):
    """Yield observations for a taxon, newest first, using id_below pagination
    (more efficient than page numbers and has no 10k-result limit)."""
    params = dict(
        taxon_id=taxon_id,
        place_id=place_id,
        quality_grade="research",
        photos="true",
        per_page=200,
        order_by="id",
        order="desc",
    )
    while True:
        results = client.get("observations", **params)["results"]
        if not results:
            return
        yield from results
        params["id_below"] = results[-1]["id"]


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(BytesIO(path.read_bytes())) as img:
            img.verify()  # catches truncated/corrupt files
        return True
    except Exception:  # noqa: BLE001
        return False


def download_one(client: INatClient, url: str, dest: Path) -> bool:
    # Write to a temp name and rename only once the image checks out, so a
    # half-written file never sits at the final path.
    tmp = dest.with_suffix(".part")
    try:
        client.download(url, tmp)
        if not is_valid_image(tmp):
            raise ValueError("not a valid image")
        tmp.replace(dest)
        return True
    except Exception as e:  # noqa: BLE001 - one bad photo shouldn't stop the run
        print(f"  failed {url}: {e}")
        tmp.unlink(missing_ok=True)
        return False


def load_manifest(verify: bool = False) -> dict[str, dict]:
    """Read the manifest, dropping duplicate rows and rows whose image is
    missing (or, with verify=True, unreadable). Rewrites the file if anything
    was dropped, so the next run re-downloads those photos."""
    if not MANIFEST.exists():
        return {}
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    manifest: dict[str, dict] = {}
    for row in rows:
        path = RAW_DIR / row["path"]
        if row["photo_id"] in manifest or not path.exists():
            continue
        if verify and not is_valid_image(path):
            path.unlink()
            continue
        manifest[row["photo_id"]] = row

    if len(manifest) != len(rows):
        print(f"Cleaned manifest: {len(rows)} rows -> {len(manifest)} "
              f"(dropped duplicates / missing / broken images)")
        with MANIFEST.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(manifest.values())
    return manifest


def acquire_lock():
    """Stop two downloads from running at once and overwriting each other."""
    import fcntl  # Unix-only; fine for this project

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    lock = (RAW_DIR / ".download.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another download is already running; not starting a second one.")
    return lock  # keep a reference: the lock lasts as long as the file is open


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-species", type=int, default=300)
    parser.add_argument("--size", default="medium", choices=["small", "medium", "large"],
                        help="small=240px, medium=500px, large=1024px (longest side)")
    parser.add_argument("--species", type=Path, default=DATA_DIR / "species.csv")
    parser.add_argument("--place-id", type=int, default=EUROPE_PLACE_ID)
    parser.add_argument("--max-per-observer", type=int, default=3,
                        help="max photos per species from any single iNaturalist user")
    parser.add_argument("--workers", type=int, default=4, help="parallel photo downloads")
    parser.add_argument("--verify", action="store_true",
                        help="check every existing image opens before resuming (slower)")
    args = parser.parse_args()

    with args.species.open(newline="", encoding="utf-8") as f:
        species = list(csv.DictReader(f))

    _lock = acquire_lock()
    client = INatClient()
    manifest = load_manifest(verify=args.verify)

    for i, sp in enumerate(species, 1):
        taxon_id = int(sp["taxon_id"])
        sp_dir = IMAGE_DIR / str(taxon_id)
        sp_dir.mkdir(parents=True, exist_ok=True)
        have = sum(1 for r in manifest.values() if int(r["taxon_id"]) == taxon_id)
        print(f"[{i}/{len(species)}] {sp['common_name']} ({sp['scientific_name']}): have {have}")
        if have >= args.per_species:
            continue

        # Cap photos per observer so a few prolific users don't dominate a class
        per_observer = Counter(
            r["observer"] for r in manifest.values() if int(r["taxon_id"]) == taxon_id
        )

        # Collect new photo jobs until we have enough for this species
        jobs: list[dict] = []
        for obs in iter_observations(client, taxon_id, args.place_id):
            if have + len(jobs) >= args.per_species:
                break
            photo = obs["photos"][0]
            pid = str(photo["id"])
            observer = (obs.get("user") or {}).get("login", "")
            if pid in manifest or per_observer[observer] >= args.max_per_observer:
                continue
            per_observer[observer] += 1
            jobs.append({
                "photo_id": pid,
                "observation_id": obs["id"],
                "taxon_id": taxon_id,
                "path": f"images/{taxon_id}/{pid}.jpg",
                "license": photo.get("license_code") or "all-rights-reserved",
                "attribution": photo.get("attribution", ""),
                "observer": observer,
                "url": photo_url(photo["url"], args.size),
            })

        with ThreadPoolExecutor(args.workers) as pool:
            ok = list(pool.map(
                lambda j: download_one(client, j["url"], RAW_DIR / j["path"]), jobs
            ))

        new_rows = [j for j, success in zip(jobs, ok) if success]
        write_header = not MANIFEST.exists()
        with MANIFEST.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerows(new_rows)
        manifest.update({r["photo_id"]: r for r in new_rows})
        print(f"  +{len(new_rows)} photos")

    print(f"Done. {len(manifest)} photos listed in {MANIFEST}")


if __name__ == "__main__":
    main()
