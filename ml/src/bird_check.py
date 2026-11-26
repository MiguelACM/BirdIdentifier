"""A small "is this a bird?" check on top of the species model.

The species model always names one of its 300 species, even for a fox. This trains a
logistic regression on the model's own 640-number feature vector (the input of its last
layer) to tell birds from everything else, and adds it to the ONNX model as a second output.
The species output is left untouched.

Steps (from the ml/ folder), each resumable:
    python -m src.bird_check fetch      # 1. download non-bird and other-bird photos (~15 min)
    python -m src.bird_check features   # 2. run the model over them and over known birds
    python -m src.bird_check train      # 3. fit and measure the check
    python -m src.bird_check add        # 4. add it to web/model/model.int8.onnx

Photos the check is trained on never overlap the test photos of src.eval_unknown.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .download_images import download_one, photo_url
from .inat import AVES_TAXON_ID, EUROPE_PLACE_ID, INatClient

ML_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ML_DIR / "data"
OUT_DIR = DATA_DIR / "birdcheck"

# iNaturalist taxon id -> photos to collect. Broad groups, so "not a bird" covers what people
# point a camera at outdoors (as far as iNaturalist has it: no people, buildings or sky).
NOT_BIRDS = {
    "mammals": (40151, 450), "insects": (47158, 450), "spiders": (47119, 300),
    "reptiles": (26036, 300), "amphibians": (20978, 300), "plants": (47126, 450),
    "fungi": (47170, 300), "molluscs": (47115, 250), "fish": (47178, 200),
}
OTHER_BIRDS = 600  # birds of species the model doesn't know: still birds
PER_SPECIES = 2  # at most this many photos per species, so no group is one common species
# Start below this observation id: older than the newest observations, which the
# eval_unknown test photos were taken from
START_BELOW_ID = 380_000_000


def collect(client: INatClient, taxon_id: int, target: int, skip_obs: set[int],
            skip_taxa: set[int] = frozenset()) -> list[dict]:
    picks, per_taxon = [], {}
    params = dict(taxon_id=taxon_id, place_id=EUROPE_PLACE_ID, quality_grade="research",
                  photos="true", per_page=200, order_by="id", order="desc", id_below=START_BELOW_ID)
    while len(picks) < target:
        results = client.get("observations", **params)["results"]
        if not results:
            break
        for obs in results:
            taxon = obs.get("taxon") or {}
            tid = taxon.get("id")
            if (obs["id"] in skip_obs or tid in skip_taxa or not obs.get("photos")
                    or per_taxon.get(tid, 0) >= PER_SPECIES or len(picks) >= target):
                continue
            per_taxon[tid] = per_taxon.get(tid, 0) + 1
            photo = obs["photos"][0]
            picks.append({"observation": obs["id"], "taxon": taxon.get("name", ""),
                          "photo": photo["id"], "url": photo_url(photo["url"], "medium")})
        params["id_below"] = results[-1]["id"]
    return picks


def fetch(workers: int) -> None:
    (OUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    picks_path = OUT_DIR / "picks.json"
    if picks_path.exists():
        picks = json.loads(picks_path.read_text())
    else:
        client = INatClient()
        with open(DATA_DIR / "unknown" / "manifest.csv", newline="") as f:
            test_obs = {int(r["observation"]) for r in csv.DictReader(f)}
        with open(DATA_DIR / "species.csv", newline="") as f:
            known = {int(r["taxon_id"]) for r in csv.DictReader(f)}
        # Each finished group is saved, so a stopped run continues with the next group
        partial_path = OUT_DIR / "picks_partial.json"
        done = json.loads(partial_path.read_text()) if partial_path.exists() else {}
        jobs = [(g, t, n, "not_bird", frozenset()) for g, (t, n) in NOT_BIRDS.items()]
        jobs.append(("other birds", AVES_TAXON_ID, OTHER_BIRDS, "bird", known))
        for group, taxon_id, target, label, skip_taxa in jobs:
            if group not in done:
                found = collect(client, taxon_id, target, test_obs, skip_taxa)
                done[group] = [{**p, "label": label, "group": group} for p in found]
                partial_path.write_text(json.dumps(done))
            print(f"{group}: {len(done[group])} photos", flush=True)
        picks = [p for group in done.values() for p in group]
        picks_path.write_text(json.dumps(picks))
        partial_path.unlink()

    todo = [p for p in picks if not (OUT_DIR / "images" / f"{p['photo']}.jpg").exists()]
    print(f"{len(picks)} photos picked, {len(todo)} still to download", flush=True)
    client = INatClient()

    def get(p: dict) -> None:
        download_one(client, p["url"], OUT_DIR / "images" / f"{p['photo']}.jpg")

    with ThreadPoolExecutor(workers) as pool:
        for n, _ in enumerate(pool.map(get, todo), 1):
            if n % 250 == 0:
                print(f"  downloaded {n}/{len(todo)}", flush=True)

    rows = [{"label": p["label"], "group": p["group"], "taxon": p["taxon"],
             "path": f"{p['photo']}.jpg", "observation": p["observation"]}
            for p in picks if (OUT_DIR / "images" / f"{p['photo']}.jpg").exists()]
    with open(OUT_DIR / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Done: {len(rows)} photos in {OUT_DIR / 'manifest.csv'} "
          f"({sum(r['label'] == 'not_bird' for r in rows)} not birds, "
          f"{sum(r['label'] == 'bird' for r in rows)} other birds)", flush=True)


MODEL = ML_DIR.parent / "web" / "model" / "model.int8.onnx"
FEATURES = "view_14"  # name of the 640-number input of the species layer in the ONNX graph
FEATURE_DIR = OUT_DIR / "features"


def sample_split(split: str, per_species: int) -> list[tuple[Path, int, str]]:
    """A few photos per known species from a train/val split (seeded, so reruns match)."""
    import random

    with open(DATA_DIR / "splits" / f"{split}.csv", newline="") as f:
        by_label: dict[str, list[str]] = {}
        for r in csv.DictReader(f):
            by_label.setdefault(r["label"], []).append(r["path"])
    rng = random.Random(0)
    return [(DATA_DIR / "raw" / p, 1, "known birds")
            for label in sorted(by_label, key=int)
            for p in rng.sample(by_label[label], min(per_species, len(by_label[label])))]


def feature_sets() -> dict[str, list[tuple[Path, int, str]]]:
    """(photo, is_bird, group) for each set. Training sets first, then the test sets."""
    def manifest(folder: Path) -> list[tuple[Path, int, str]]:
        # both manifests call birds of species outside the model's list "other birds"
        with open(folder / "manifest.csv", newline="") as f:
            return [(folder / "images" / r["path"], int(r["group"] == "other birds"),
                     # eval_unknown stores "not birds" with the group in the taxon ("mammals: ...")
                     r["taxon"].split(":")[0] if r["group"] == "not birds" else r["group"])
                    for r in csv.DictReader(f)]

    return {
        "train_known": sample_split("train", 5),
        "train_other": manifest(OUT_DIR),
        "test_known": sample_split("val", 3),
        "test_other": manifest(DATA_DIR / "unknown"),
    }


def features() -> None:
    import numpy as np
    import onnx
    import onnxruntime as ort

    from .predict_split import load_image

    model = onnx.load(MODEL)
    model.graph.output.append(onnx.helper.make_tensor_value_info(FEATURES, onnx.TensorProto.FLOAT, [1, 640]))
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    config = json.loads((MODEL.parent / "config.json").read_text())
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    sets = feature_sets()
    total = sum(len(v) for v in sets.values())
    done_before = 0
    for name, items in sets.items():
        out, partial = FEATURE_DIR / f"{name}.npz", FEATURE_DIR / f"{name}.partial.npz"
        if out.exists():
            done_before += len(items)
            continue
        feats = np.zeros((len(items), 640), np.float32)
        top = np.zeros(len(items), np.float32)
        start = 0
        if partial.exists():  # continue a stopped run
            saved = np.load(partial)
            start = int(saved["n"])
            feats[:start], top[:start] = saved["feats"][:start], saved["top"][:start]
        for i in range(start, len(items)):
            x = load_image(items[i][0], config["resize"], config["img_size"])
            probs, f = session.run(None, {"image": x})
            feats[i], top[i] = f[0], probs.max()
            if (i + 1) % 250 == 0 or i + 1 == len(items):
                np.savez(partial, feats=feats, top=top, n=i + 1)
                print(f"  {done_before + i + 1}/{total} photos ({name})", flush=True)
        np.savez_compressed(out, feats=feats, top=top, is_bird=np.array([b for _, b, _ in items]),
                            group=np.array([g for _, _, g in items]),
                            path=np.array([str(p.relative_to(DATA_DIR)) for p, _, _ in items]))
        partial.unlink()
        done_before += len(items)
    print(f"Done: features for {total} photos in {FEATURE_DIR}", flush=True)


HEAD_PATH = OUT_DIR / "head.json"


def fit_logistic(x, y, sample_weight, l2: float, iters: int = 25):
    """L2-regularised logistic regression by Newton's method (IRLS). Returns (w, b)."""
    import numpy as np

    xb = np.hstack([x, np.ones((len(x), 1))])  # last column = bias, not regularised
    w = np.zeros(xb.shape[1])
    reg = np.full(xb.shape[1], l2)
    reg[-1] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-xb @ w))
        grad = xb.T @ (sample_weight * (p - y)) + reg * w
        hess = (xb * (sample_weight * p * (1 - p))[:, None]).T @ xb + np.diag(reg)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return w[:-1], w[-1]


def load_set(name: str) -> dict:
    import numpy as np

    d = np.load(FEATURE_DIR / f"{name}.npz")
    return {k: d[k] for k in d.files}


def train() -> None:
    import numpy as np

    tr = [load_set("train_known"), load_set("train_other")]
    x = np.vstack([s["feats"] for s in tr]).astype(np.float64)
    y = np.concatenate([s["is_bird"] for s in tr]).astype(np.float64)
    # balanced classes: birds and non-birds count the same in the loss
    weight = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))

    # pick the regularisation by 5-fold cross-validation (log loss)
    rng = np.random.default_rng(0)
    fold = rng.integers(0, 5, len(y))
    print("Regularisation (5-fold cross-validation, balanced log loss):")
    best = None
    for l2 in [1, 10, 30, 100, 300, 1000, 3000]:
        losses = []
        for k in range(5):
            w, b = fit_logistic(x[fold != k], y[fold != k], weight[fold != k], l2)
            p = np.clip(1 / (1 + np.exp(-(x[fold == k] @ w + b))), 1e-7, 1 - 1e-7)
            yk, wk = y[fold == k], weight[fold == k]
            losses.append(-(wk * (yk * np.log(p) + (1 - yk) * np.log(1 - p))).sum() / wk.sum())
        loss = float(np.mean(losses))
        print(f"  l2={l2:<6} loss {loss:.4f}")
        if best is None or loss < best[1]:
            best = (l2, loss)
    w, b = fit_logistic(x, y, weight, best[0])
    HEAD_PATH.write_text(json.dumps({"l2": best[0], "w": w.tolist(), "b": float(b)}))
    print(f"Chose l2={best[0]}; saved {HEAD_PATH}")
    report(w, b)


def report(w, b) -> None:
    import numpy as np

    known, other = load_set("test_known"), load_set("test_other")

    def p_bird(s):
        return 1 / (1 + np.exp(-(s["feats"].astype(np.float64) @ w + b)))

    pk, po = p_bird(known), p_bird(other)
    birds, groups = other["is_bird"] == 1, other["group"]
    print(f"\nTest photos (never used for training): {len(pk)} known species, "
          f"{birds.sum()} other bird species, {(~birds).sum()} non-birds")
    print("  cutoff  known birds rejected  other birds rejected  non-birds caught"
          "  | non-birds caught with the 40% cutoff too")
    for t in [0.1, 0.2, 0.3, 0.5, 0.7]:
        either = ((po < t) | (other["top"] < 0.4))[~birds].mean()
        print(f"  {t:6.1f}  {(pk < t).mean():20.1%}  {(po[birds] < t).mean():20.1%}"
              f"  {(po[~birds] < t).mean():16.1%}  | {either:.1%}")
    print("  (the 40% cutoff alone catches "
          f"{(other['top'][~birds] < 0.4).mean():.1%} of non-birds)")
    print("Non-birds caught at cutoff 0.5, by group: " + ", ".join(
        f"{g} {(po[groups == g] < 0.5).mean():.0%}" for g in sorted(set(groups[~birds]))))
    worst = np.argsort(pk)[:5]
    print("Known-bird photos the check is least sure about: " +
          ", ".join(f"{known['path'][i].split('/')[-2]}:{pk[i]:.2f}" for i in worst))


def add_to_model() -> None:
    """Append the check to the ONNX model as a second output, "bird" = P(photo shows a bird).

    Only new nodes are added after the 640-number features; the species output is untouched.
    """
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper

    from .predict_split import load_image

    head = json.loads(HEAD_PATH.read_text())
    model = onnx.load(MODEL)
    if any(o.name == "bird" for o in model.graph.output):
        raise SystemExit(f"{MODEL} already has a bird output; restore it from git first")
    w = np.array(head["w"], np.float32).reshape(640, 1)
    model.graph.initializer.extend([numpy_helper.from_array(w, "bird_check.weight"),
                                    numpy_helper.from_array(np.array([head["b"]], np.float32),
                                                            "bird_check.bias")])
    model.graph.node.extend([
        helper.make_node("MatMul", [FEATURES, "bird_check.weight"], ["bird_check.matmul"]),
        helper.make_node("Add", ["bird_check.matmul", "bird_check.bias"], ["bird_check.logit"]),
        helper.make_node("Sigmoid", ["bird_check.logit"], ["bird"]),
    ])
    model.graph.output.append(helper.make_tensor_value_info("bird", TensorProto.FLOAT, [1, 1]))
    onnx.checker.check_model(model)

    # Check before replacing the file: same species probabilities, and the bird output
    # matches the check computed from the saved features
    old = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    new = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    config = json.loads((MODEL.parent / "config.json").read_text())
    test = load_set("test_other")
    for i in range(0, len(test["path"]), len(test["path"]) // 8):
        x = load_image(DATA_DIR / str(test["path"][i]), config["resize"], config["img_size"])
        p_old = old.run(None, {"image": x})[0]
        p_new, bird = new.run(None, {"image": x})
        expected = 1 / (1 + np.exp(-(test["feats"][i] @ w[:, 0] + head["b"])))
        assert np.array_equal(p_old, p_new), "species output changed"
        assert abs(bird[0, 0] - expected) < 1e-4, (bird, expected)
    size_before = MODEL.stat().st_size
    onnx.save(model, MODEL)
    config["bird_cutoff"] = head["cutoff"]
    config["output"] = config["output"] + '; "bird": probability that the photo shows a bird, shape [1,1]'
    (MODEL.parent / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Added the bird output to {MODEL} ({size_before:,} -> {MODEL.stat().st_size:,} bytes); "
          f"bird_cutoff {head['cutoff']} in config.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["fetch", "features", "train", "add"])
    parser.add_argument("--workers", type=int, default=4, help="parallel photo downloads")
    args = parser.parse_args()
    if args.step == "fetch":
        fetch(args.workers)
    elif args.step == "features":
        features()
    elif args.step == "train":
        train()
    elif args.step == "add":
        add_to_model()


if __name__ == "__main__":
    main()
