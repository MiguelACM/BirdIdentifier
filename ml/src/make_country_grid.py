"""Make the small lookup grid the web app uses to turn GPS coordinates into a country.

Rasterises Natural Earth's country borders (public domain) onto a 0.1° grid
over Europe and run-length encodes each row, so a photo's GPS position can be
matched to a country entirely in the browser: the location never leaves the
device. Sea cells near a coast get the nearest country, so photos taken from a
beach or a boat still count.

Usage (from the ml/ folder):
    curl -L -o data/geo/ne_50m_admin_0_countries.geojson \\
      https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson
    python -m src.make_country_grid       # -> web/country-grid.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .fetch_gbif_priors import COUNTRIES

ML_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = ML_DIR.parent / "web"

LAT0, LAT1 = 27.0, 72.0   # Canary Islands to North Cape
LON0, LON1 = -32.0, 60.0  # Azores to the Urals (European Russia)
STEP = 0.1
COAST_CELLS = 5           # how far (in cells) sea cells take the nearest country


def polygons(geometry: dict):
    if geometry["type"] == "Polygon":
        yield geometry["coordinates"]
    elif geometry["type"] == "MultiPolygon":
        yield from geometry["coordinates"]


def rasterise(rings: list, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Even-odd scanline fill: for each grid row, find where the polygon's edges
    cross that latitude and fill the cells between pairs of crossings."""
    mask = np.zeros((len(lats), len(lons)), dtype=bool)
    edges = []
    for ring in rings:
        pts = np.asarray(ring, dtype=np.float64)[:, :2]
        edges.append(np.hstack([pts[:-1], pts[1:]]))
    e = np.vstack(edges)  # x1, y1, x2, y2
    lo, hi = np.minimum(e[:, 1], e[:, 3]), np.maximum(e[:, 1], e[:, 3])
    rows = np.nonzero((lats >= lo.min()) & (lats <= hi.max()))[0]
    for r in rows:
        y = lats[r]
        hit = (lo <= y) & (hi > y)  # half-open so vertices are not counted twice
        if not hit.any():
            continue
        x1, y1, x2, y2 = e[hit].T
        xs = np.sort(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
        for a, b in zip(xs[0::2], xs[1::2]):
            mask[r, np.searchsorted(lons, a):np.searchsorted(lons, b)] = True
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--borders", type=Path,
                        default=ML_DIR / "data" / "geo" / "ne_50m_admin_0_countries.geojson")
    parser.add_argument("--out", type=Path, default=WEB_DIR / "country-grid.json")
    args = parser.parse_args()

    lats = LAT0 + STEP * (np.arange(round((LAT1 - LAT0) / STEP)) + 0.5)  # cell centres
    lons = LON0 + STEP * (np.arange(round((LON1 - LON0) / STEP)) + 0.5)
    # 0 = sea / unknown, 1 = a country outside the supported list, 2+ = COUNTRIES
    codes = ["", "-"] + COUNTRIES
    grid = np.zeros((len(lats), len(lons)), dtype=np.uint8)

    features = json.loads(args.borders.read_text())["features"]
    for f in features:
        iso = f["properties"]["ISO_A2_EH"]
        value = codes.index(iso) if iso in COUNTRIES else 1
        for poly in polygons(f["geometry"]):
            ring = np.asarray(poly[0])
            if (ring[:, 1].max() < LAT0 or ring[:, 1].min() > LAT1
                    or ring[:, 0].max() < LON0 or ring[:, 0].min() > LON1):
                continue
            grid[rasterise(poly, lats, lons)] = value

    # Grow countries into nearby sea, one cell per pass (4-neighbours)
    for _ in range(COAST_CELLS):
        padded = np.pad(grid, 1)
        neighbours = (padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:])
        grown = grid.copy()
        for shifted in neighbours:
            take = (grown == 0) & (shifted > 0)
            grown[take] = shifted[take]
        grid = grown

    # Each row: [run length, code index, run length, code index, ...]
    rle = []
    for row in grid:
        change = np.flatnonzero(np.diff(row)) + 1
        starts = np.concatenate([[0], change])
        lengths = np.diff(np.concatenate([starts, [len(row)]]))
        rle.append([int(v) for pair in zip(lengths, row[starts]) for v in pair])

    out = {"lat0": LAT0, "lon0": LON0, "step": STEP, "rows": len(lats), "cols": len(lons),
           "codes": codes, "source": "Natural Earth 1:50m countries (public domain)",
           "rle": rle}
    text = json.dumps({k: v for k, v in out.items() if k != "rle"}, separators=(",", ":"))[:-1]
    text += ',"rle":[\n' + ",\n".join(json.dumps(r, separators=(",", ":")) for r in rle) + "\n]}\n"
    args.out.write_text(text)
    covered = {codes[v] for v in np.unique(grid) if v > 1}
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB); "
          f"countries on the grid: {len(covered)}; missing: {sorted(set(COUNTRIES) - covered)}")


def lookup(grid_json: dict, lat: float, lon: float) -> str:
    """Country code at a position ("" = sea/unknown, "-" = outside supported countries).
    The web app has the same function in JavaScript."""
    r = int((lat - grid_json["lat0"]) // grid_json["step"])
    c = int((lon - grid_json["lon0"]) // grid_json["step"])
    if not (0 <= r < grid_json["rows"] and 0 <= c < grid_json["cols"]):
        return "-"
    row = grid_json["rle"][r]
    for i in range(0, len(row), 2):
        c -= row[i]
        if c < 0:
            return grid_json["codes"][row[i + 1]]
    return ""


if __name__ == "__main__":
    main()
