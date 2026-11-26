"""Checks that the files the web app loads agree with each other.

These catch the easy-to-miss mistakes when a new model is dropped into web/:
labels, priors and Portuguese names from a different species list, a config
that doesn't match the model, or an example photo of a species the model lacks.
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

WEB = Path(__file__).resolve().parents[2] / "web"


def load(name):
    return json.loads((WEB / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def labels():
    return load("model/labels.json")


@pytest.fixture(scope="module")
def session():
    return ort.InferenceSession(str(WEB / "model" / "model.int8.onnx"),
                                providers=["CPUExecutionProvider"])


def test_labels_are_indexed_in_order(labels):
    assert [label["index"] for label in labels] == list(range(len(labels)))
    assert len({label["taxon_id"] for label in labels}) == len(labels)


def test_model_outputs_probabilities_for_every_label(labels, session):
    config = load("model/config.json")
    size = config["img_size"]
    assert config["resize"] >= size
    x = np.random.default_rng(0).random((1, 3, size, size), dtype=np.float32)
    probs = session.run(None, {session.get_inputs()[0].name: x})[0][0]
    assert probs.shape == (len(labels),)
    assert probs.min() >= 0
    assert probs.sum() == pytest.approx(1, abs=1e-3)


def test_priors_line_up_with_labels(labels):
    priors = load("priors.json")
    assert priors["taxa"] == [label["taxon_id"] for label in labels]
    for country, entry in priors["countries"].items():
        assert len(entry["year"]) == len(labels), country
        assert len(entry["months"]) == 12, country
        for month in entry["months"]:
            assert len(month) == len(labels), country
            assert min(month) >= 1  # every species keeps a small chance
            # shares are rounded to integers out of `scale`, so they sum to about scale
            assert sum(month) == pytest.approx(priors["scale"], rel=0.05), country


def test_every_species_has_a_portuguese_name(labels):
    names = load("i18n/species.pt.json")
    assert set(names) == {str(label["taxon_id"]) for label in labels}
    assert all(name.strip() for name in names.values())


def test_examples_are_known_species_in_supported_countries(labels):
    by_taxon = {label["taxon_id"]: label["common_name"] for label in labels}
    countries = load("priors.json")["countries"]
    for example in load("examples.json"):
        assert by_taxon.get(example["taxon_id"]) == example["species"]
        assert example["country"] in countries
        assert 1 <= example["month"] <= 12


def test_offline_files_exist():
    """sw.js saves these for offline use; a renamed or missing file would break the install."""
    import re

    sw = (WEB / "sw.js").read_text(encoding="utf-8")
    ort = re.search(r'const ORT = "([^"]+)"', sw).group(1)
    # the service worker must cache the same ONNX Runtime version the app loads
    assert ort in (WEB / "app.js").read_text(encoding="utf-8")
    block = re.search(r"const PRECACHE = \[(.*?)\];", sw, re.S).group(1)
    local = [f for f in re.findall(r'"([^"]+)"', block) if f != "./"]
    assert local, "no files found in PRECACHE"
    for name in local:
        assert (WEB / name).is_file(), name
    for icon in load("manifest.webmanifest")["icons"]:
        assert (WEB / icon["src"]).is_file(), icon["src"]


def test_bird_check_output(session):
    """The "is this a bird?" check (src.bird_check) is a second output, read by app.js."""
    assert [o.name for o in session.get_outputs()] == ["probs", "bird"]
    config = load("model/config.json")
    assert 0 < config["bird_cutoff"] < 1
    size = config["img_size"]
    x = np.random.default_rng(1).random((1, 3, size, size), dtype=np.float32)
    bird = session.run(["bird"], {session.get_inputs()[0].name: x})[0]
    assert bird.shape == (1, 1)
    assert 0 <= bird[0, 0] <= 1
