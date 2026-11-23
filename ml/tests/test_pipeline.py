"""Unit tests for the small pure functions in the data pipeline."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.download_images import photo_url
from src.fetch_gbif_priors import COUNTRIES, SCALE, build_priors, shares, to_ints
from src.make_country_grid import lookup
from src.make_splits import bucket
from src.predict_split import load_image

WEB = Path(__file__).resolve().parents[2] / "web"


def test_bucket_is_stable_and_in_range():
    values = [bucket(str(i)) for i in range(1000)]
    assert all(0 <= v < 1 for v in values)
    assert bucket("12345") == bucket("12345")
    # roughly uniform, so a 10% split gets about 10% of photos
    assert 0.07 < np.mean(np.array(values) < 0.1) < 0.13


def test_photo_url_swaps_size():
    url = "https://inaturalist-open-data.s3.amazonaws.com/photos/1/square.jpg"
    assert photo_url(url, "medium").endswith("/1/medium.jpg")


def test_shares_smooth_towards_fallback():
    assert shares([3, 1], [0.5, 0.5], 0) == [0.75, 0.25]
    # no records at all: the fallback is used as is
    assert shares([0, 0], [0.2, 0.8], 10) == pytest.approx([0.2, 0.8])
    # a little data moves the shares only part of the way
    smoothed = shares([4, 0], [0.5, 0.5], 4)
    assert smoothed == pytest.approx([0.75, 0.25])
    assert sum(smoothed) == pytest.approx(1)


def test_to_ints_keeps_every_species_possible():
    assert to_ints([0.5, 0.0, 0.00001]) == [SCALE // 2, 1, 1]


def test_build_priors_shapes_and_fallbacks():
    n = 3
    counts = {c: [[0] * n for _ in range(12)] for c in COUNTRIES}
    counts["PT"][5] = [100, 0, 0]  # only species 0 seen, in June
    counts["ES"][0] = [0, 50, 50]
    priors = build_priors(counts, n)
    assert set(priors) == set(COUNTRIES)
    pt = priors["PT"]
    assert pt["records"] == 100
    assert len(pt["months"]) == 12 and all(len(m) == n for m in pt["months"])
    # June in Portugal favours species 0 more than the rest of the year does
    assert pt["months"][5][0] > pt["months"][0][0]
    # a country with no records falls back to Europe's shares (species 0 = 50%)
    assert priors["AD"]["year"][0] == pytest.approx(SCALE / 2, abs=1)


@pytest.mark.parametrize("lat, lon, expected", [
    (38.72, -9.14, "PT"),   # Lisbon
    (52.52, 13.40, "DE"),   # Berlin
    (64.15, -21.94, "IS"),  # Reykjavik
    (45.0, -30.0, ""),      # mid-Atlantic
    (40.71, -74.0, "-"),    # New York, outside the grid
])
def test_country_lookup(lat, lon, expected):
    grid = json.loads((WEB / "country-grid.json").read_text())
    assert lookup(grid, lat, lon) == expected


def test_load_image_resizes_and_centre_crops(tmp_path):
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[:, 50:150] = 255  # white centre square, black sides
    Image.fromarray(img).save(tmp_path / "x.png")
    x = load_image(tmp_path / "x.png", resize=60, size=56)
    assert x.shape == (1, 3, 56, 56)
    assert x.dtype == np.float32
    # the centre crop of a 120x60 resize lands inside the white square
    assert x.min() > 0.9
