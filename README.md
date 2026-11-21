# Bird Identifier

A learning project: identify European bird species from a photo, with the model
running **entirely in the browser**.

**Live demo:** https://miguelacm.github.io/BirdIdentifier/

- **ML:** PyTorch + [timm](https://github.com/huggingface/pytorch-image-models), fine-tuning a pretrained image model, exported to ONNX
- **Web:** ONNX Runtime Web (WebGPU / WASM), so photos never leave the device
- **Data:** research-grade bird observations from [iNaturalist](https://www.inaturalist.org/)

## Roadmap

- [x] Data pipeline: species list, photo download, train/val/test split
- [x] Training and ONNX export code (tested locally on a tiny subset)
- [x] Train on a GPU (Kaggle) and evaluate: **75.2% top-1, 90.1% top-5** on the test set
- [ ] Improve: more epochs, higher resolution (288px), more photos per species
- [x] Web app: upload a photo and see the top 5 species, running locally in the browser
- [ ] Re-rank predictions by location and season
- [ ] Scale up from 100 species to all of Europe

## Project layout

```
ml/
  src/
    inat.py             # small rate-limited iNaturalist API client
    fetch_species.py    # step 1: pick the top-N European species
    download_images.py  # step 2: download photos + license manifest
    make_splits.py      # step 3: deterministic train/val/test split
    package_dataset.py  # bundle data + code into a zip for Kaggle
    train.py            # step 4: fine-tune a timm model
    export_onnx.py      # step 5: export to ONNX for the browser
  notebooks/
    train_kaggle.ipynb  # runs train + export on a free Kaggle GPU
  data/
    species.csv         # committed: the species list
    raw/                # ignored: downloaded images + manifest.csv
    splits/             # ignored: train/val/test CSVs + labels.json
web/
  index.html, style.css, app.js   # static site, no build step
  model/                # committed: model.int8.onnx (15 MB), labels.json, config.json
```

## Building the dataset

```bash
cd ml
pip install -r requirements.txt

python -m src.fetch_species --top 100          # -> data/species.csv
python -m src.download_images --per-species 300  # -> data/raw/ (resumable)
python -m src.make_splits                       # -> data/splits/
```

The full download (100 species x 300 photos at 500px) is about 30k images and
~3.5 GB, and takes a couple of hours. You can stop and restart it at any time; only
one download can run at once. Add `--verify` to re-check every saved image first.

Design choices worth knowing about:

- **One photo per observation:** several photos of the same bird would be near-duplicates
  and could leak between the train and test sets.
- **Max 3 photos per observer per species:** keeps a few prolific users from dominating
  a class with one camera, location and style.
- **Hash-based splits:** a photo's split depends only on its observation ID, so downloading
  more data later never moves existing photos into a different set.

## Training

The model is `convnext_nano.in12k_ft_in1k` from timm (15M parameters, pretrained on
ImageNet-12k), fine-tuned for 12 epochs with AdamW, warm-up + cosine learning-rate
schedule, label smoothing and mixed precision. Training needs a GPU, so it runs on Kaggle:

1. `python -m src.package_dataset` creates `ml/dist/bird-identifier-data.zip` (data + code,
   about 1.4 GB). Photos are shrunk to a 320px short side, which is plenty for 224px training,
   so the zip stays under 2 GB. Larger zips use the ZIP64 format, which Kaggle rejects.
2. Upload the zip as a new Kaggle Dataset.
3. Import `ml/notebooks/train_kaggle.ipynb` into Kaggle, attach the dataset, and turn on
   a GPU and internet access. Then use *Save & Run All*.
4. Download `bird-model.zip` from the notebook output. It contains `model.onnx`,
   `labels.json`, `config.json` and the metrics.

For a quick local check without a GPU (very slow, but tests the whole pipeline):

```bash
python -m src.train --limit-per-class 3 --epochs 1 --batch-size 16
python -m src.export_onnx runs/<run>/best.pt
```

### Results (v1)

12 epochs on a Kaggle T4 (about 85 s per epoch):

| | top-1 | top-5 |
|---|---|---|
| validation | 75.9% | 91.2% |
| test | 75.2% | 90.1% |

Validation accuracy was still rising at the last epoch, so longer training should help.
Most errors are between species that birders also find hard: large gulls (Herring,
Yellow-legged, Lesser Black-backed), crows (Rook, Carrion Crow, Raven), kites and harriers,
and Little vs Great Egret. Distinctive birds (flamingo, heron, swan, moorhen) score above 90%.

The exported ONNX model includes normalisation and softmax. The browser passes in a
224×224 RGB image scaled to 0–1 and gets back one probability per species.

## Web app

A static page (plain HTML/CSS/JS) that runs the model with
[ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) (WebAssembly). The photo
is resized and centre-cropped in a canvas exactly like the validation transform, and
classified on the device in about 0.4 s. After a prediction, the page loads a reference
photo and Wikipedia link for each suggested species from the iNaturalist API. Only species
IDs are sent, never the photo.

By default it uses the **int8-quantised** model: 15 MB instead of 60 MB, with about
0.5–1% lower accuracy (on 600 test photos: 76.8% vs 77.5% top-1). The model is committed
in `web/model/` (non-commercial use only, see [web/model/README.md](web/model/README.md)).

Run it locally:

```bash
python3 -m http.server 8000 --directory web   # then open http://localhost:8000
```

To replace the model with a newly trained one:

```bash
cd ml && python -m src.export_onnx runs/<run>/best.pt --out ../web/model --int8
```

The site is published to GitHub Pages by `.github/workflows/pages.yml` whenever `web/`
changes on `main`.

## Data and licensing

Photos come from iNaturalist users under a mix of licenses (CC0, CC BY, CC BY-NC,
and some all-rights-reserved). They are **not** included in this repository.
The scripts download them for personal, non-commercial study, and
`data/raw/manifest.csv` records the license and author of every image.
Any trained model weights shared from this project are for non-commercial use only.

The scripts follow iNaturalist's API guidance (about 1 request per second).
Please keep it that way if you change them.
