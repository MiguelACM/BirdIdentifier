# Bird Identifier

A learning project: identify European bird species from a photo, with the model
running **entirely in the browser**.

**Live demo:** https://miguelacm.github.io/BirdIdentifier/

- **ML:** PyTorch + [timm](https://github.com/huggingface/pytorch-image-models), fine-tuning a pretrained image model, exported to ONNX
- **Web:** ONNX Runtime Web (WebAssembly), so photos never leave the device
- **Data:** research-grade bird observations from [iNaturalist](https://www.inaturalist.org/)

## Roadmap

- [x] Data pipeline: species list, photo download, train/val/test split
- [x] Training and ONNX export code (tested locally on a tiny subset)
- [x] Train on a GPU (Kaggle) and evaluate (v1, 100 species): **75.2% top-1, 90.1% top-5** on the test set
- [x] Web app: upload a photo and see the top 5 species, running locally in the browser
- [x] Re-rank predictions by location and season (GBIF sightings per country and month)
- [x] Scale up to 300 species, train longer (25 epochs) at higher resolution (288px) with
      more photos (500 per species): **76.5% top-1, 89.6% top-5** on the test set (v2)
- [ ] Scale up further: about 450 European species have at least 300 iNaturalist observations

## Project layout

```
ml/
  src/
    inat.py             # small rate-limited iNaturalist API client
    fetch_species.py    # step 1: pick the top-N European species
    download_images.py  # step 2: download photos + license manifest
    make_splits.py      # step 3: deterministic train/val/test split
    package_dataset.py  # bundle data + code into < 2 GB zip parts for Kaggle
    train.py            # step 4: fine-tune a timm model
    export_onnx.py      # step 5: export to ONNX for the browser
    fetch_species_names.py  # species names in other languages (web/i18n/)
    fetch_gbif_priors.py    # location + season priors from GBIF (web/priors.json)
    make_country_grid.py    # GPS -> country lookup grid (web/country-grid.json)
    predict_split.py    # save the model's predictions on a split, for evaluation
    eval_priors.py      # measure the priors and pick their weight
  notebooks/
    train_kaggle.ipynb  # runs train + export on a free Kaggle GPU
  data/
    species.csv         # committed: the species list
    raw/                # ignored: downloaded images + manifest.csv
    splits/             # ignored: train/val/test CSVs + labels.json
web/
  index.html, style.css, app.js   # static site, no build step
  i18n.js, i18n/        # page text (EN/PT) and Portuguese species names
  location.js, exif.js  # location + season re-ranking, EXIF GPS/date reader
  priors.json           # species share of GBIF sightings per country and month
  country-grid.json     # 0.1° grid for turning GPS coordinates into a country
  examples.json         # example photos (test set, CC0/CC BY, loaded from iNaturalist)
  model/                # committed: model.int8.onnx (15 MB), labels.json, config.json
```

## Building the dataset

```bash
cd ml
pip install -r requirements.txt

python -m src.fetch_species --top 300          # -> data/species.csv
python -m src.download_images --per-species 500  # -> data/raw/ (resumable)
python -m src.make_splits                       # -> data/splits/
```

The full download (300 species x 500 photos at 500px) is about 150k images and
~17 GB, and takes a few hours (photos already downloaded are kept). You can stop and restart it at any time; only
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
ImageNet-12k), fine-tuned at 288×288 for 25 epochs with AdamW, warm-up + cosine
learning-rate schedule, label smoothing and mixed precision. Training needs a GPU, so it
runs on Kaggle, on both of its T4s:

1. `python -m src.package_dataset` writes `ml/dist/bird-identifier-data-1.zip`, `-2.zip`, …
   (data + code, about 6.7 GB in parts of 1.9 GB). Photos are shrunk to a 320px short side,
   which is plenty for 288px training. Each part stays under 2 GB because larger zips use the
   ZIP64 format, which Kaggle rejects.
2. Upload all the parts to **one** new Kaggle Dataset.
3. Import `ml/notebooks/train_kaggle.ipynb` into Kaggle, attach the dataset, and turn on
   GPU T4 x2 and internet access. Then use *Save & Run All* (about 5 hours).
4. Download `bird-model.zip` from the notebook output. It contains `model.onnx`,
   `model.int8.onnx`, `labels.json`, `config.json` and the metrics.

Two things make long Kaggle runs survivable:

- **Resuming:** every epoch saves `last.pt` (weights, optimiser and schedule state). Attach a
  stopped run's output (or a zip of its `runs/` folder) to the notebook and it carries on
  from the last finished epoch. Newer code can be added to the dataset as a small
  `code/src/` zip, so the photos don't need uploading again.
- **A memory leak:** on Kaggle the training process gains about 1.7 MB of RAM per step
  (~3 GB per epoch) until the container is killed, after about 8 epochs. It is not the
  DataLoader workers (they are restarted every epoch and their photo list is a numpy array)
  and not `pin_memory`; the cause was never found. `train.py` logs the container's memory
  every 200 steps and exits with code 3 after any epoch that ends above 16 GB, and the
  notebook then restarts it from `last.pt` in a fresh process. As a last resort it stops
  cleanly above 24 GB, still testing and exporting the best checkpoint.

For a quick local check without a GPU (very slow, but tests the whole pipeline):

```bash
python -m src.train --limit-per-class 3 --epochs 1 --batch-size 16
python -m src.export_onnx runs/<run>/best.pt
```

### Results

**v2:** 300 species, 500 photos each, 288px, 25 epochs on 2× T4 (about 11 minutes per
epoch). Best validation accuracy was at epoch 23, and the last few epochs changed little:

| | top-1 | top-5 |
|---|---|---|
| validation | 76.6% | 89.7% |
| test | 76.5% | 89.6% |

With three times as many species, accuracy went slightly up. The remaining mistakes are
mostly between species that birders also confuse: Marsh, Blyth's Reed and Booted Warblers,
Eurasian and Short-toed Treecreepers, Chiffchaff and Willow Warbler, House and Italian
Sparrows, and the large gulls (Yellow-legged, Caspian, Herring). The hardest species are
Carrion Crow and Common Tern (about 50%); distinctive birds such as Black Swan, Black-winged
Stilt and Dipper score 94–95%.

**v1:** 100 species, 300 photos each, 224px, 12 epochs on a Kaggle T4 (about 85 s per epoch):

| | top-1 | top-5 |
|---|---|---|
| validation | 75.9% | 91.2% |
| test | 75.2% | 90.1% |

Validation accuracy was still rising at the last epoch. Most errors were between species that birders also find hard: large gulls (Herring,
Yellow-legged, Lesser Black-backed), crows (Rook, Carrion Crow, Raven), kites and harriers,
and Little vs Great Egret. Distinctive birds (flamingo, heron, swan, moorhen) score above 90%.

The exported ONNX model includes normalisation and softmax. The browser passes in a
288×288 RGB image scaled to 0–1 and gets back one probability per species.

## Web app

A static page (plain HTML/CSS/JS) that runs the model with
[ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) (WebAssembly). The photo
is first classified using the same centre square as the validation transform, on the
device, in about 1 s. A **crop box** shows exactly what the model sees. Users can drag it
onto the bird or resize it from a corner (also with the arrow keys and +/-), and the model
re-runs on the new crop when they let go. This matters for small or off-centre birds,
which the centre square often misses. After a prediction, the page loads a reference
photo and Wikipedia link for each suggested species from the iNaturalist API. Only species
IDs are sent, never the photo.

The page is available in **English and Portuguese**. It follows the browser's language, and
an EN | PT switch at the top remembers the visitor's choice. Page text lives in
`web/i18n.js`. Portuguese species names (the names used in Portugal) come from iNaturalist
via `python -m src.fetch_species_names --locale pt --place 7122 --lang pt`, which writes
`web/i18n/species.pt.json`. In Portuguese, Wikipedia links point to pt.wikipedia.org.

### Location and season

A photo alone can't separate some look-alikes, but where and when it was taken often can:
a large pale gull in Portugal is far more likely to be a Yellow-legged than a Herring Gull,
and a swallow in Germany in January is unlikely. Once a country is set, the page re-ranks
the model's guesses with how often each species is recorded there in that month:

    p(species | photo, place, month)  ∝  p_model(species | photo) × p_prior(species | place, month)^w

- **Where the location comes from:** the photo's own EXIF GPS position and date, when it has
  them (read in the browser by `web/exif.js`, so they never leave the device), or a country
  and month picked by the visitor. A hand-picked country is remembered; the month defaults to
  the current one. Hints are off until a country is known. Example photos use the country
  and month of their iNaturalist observation.
- **Priors:** `python -m src.fetch_gbif_priors` asks [GBIF](https://www.gbif.org/) for the number
  of records of each species in each of 52 European countries and each month (624 requests),
  and stores each species' share of the country-month total in `web/priors.json` (500 KB,
  ~165 KB compressed). It asks for all bird records at once and picks out the model's
  species, because listing 300 species in the URL makes it too long for GBIF. Using shares rather than raw counts removes most of the "more birders
  in Britain" effect. Months with few records lean on the country's whole year, and small
  countries on Europe as a whole (additive smoothing). Russia only counts records west of the
  Urals (60°E), so Siberian birds don't skew European Russia. Turkey is left out, as almost
  all of it is in Asia.
- **GPS to country:** `python -m src.make_country_grid` rasterises
  [Natural Earth](https://www.naturalearthdata.com/) country borders onto a 0.1° grid
  (34 KB, run-length encoded), with sea cells near a coast given to the nearest country, so
  no online reverse-geocoding service is needed.
- **Display:** species with few records there and then get a "few records" note, and if the
  location changed the top match, the page says what the photo alone would have picked.

**Does it help?** `python -m src.eval_priors` uses each test photo's real iNaturalist location
and date. It picks the weight `w` on the validation set (still best at 0.7; anything from 0.5
to 0.8 is within 0.2 points), then measures on the test set (int8 model):

| test set (14,989 photos) | top-1 | top-5 |
|---|---|---|
| photo only | 76.8% | **89.9%** |
| + country (whole year) | 77.4% | 88.8% |
| + country and month | **77.9%** | 89.3% |

99% of test photos were taken in a covered country (23% of them in European Russia). The gain
is smaller than with 100 species (+1.1 points, it was +2.1): a better model leaves fewer
mistakes that location can fix, and with 300 species more of the look-alikes live in the same
places. Top-5 drops slightly, as the prior pushes rare-but-possible species out of the
shortlist. A weight above 1 starts to hurt: the prior then overrules clear photos of birds
that are simply rare in that place.

Six **example photos** (`web/examples.json`) let visitors try it without a photo of their own.
They come from the test set, so the model never saw them during training, and are CC0 / CC BY
licensed, credited and loaded from iNaturalist. With location all six are correct; without it,
the juvenile Herring Gull is taken for a Spotless Starling, which shows why the hints help.

By default it uses the **int8-quantised** model: 16 MB instead of 60 MB, with no
measurable loss (on the full test set: 76.8% top-1 vs 76.5% for the PyTorch model). The model is committed
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
Location priors are derived from occurrence counts on [GBIF.org](https://www.gbif.org/), and
country borders come from Natural Earth (public domain).
The scripts download them for personal, non-commercial study, and
`data/raw/manifest.csv` records the license and author of every image.
Any trained model weights shared from this project are for non-commercial use only.

The scripts follow iNaturalist's API guidance (about 1 request per second).
Please keep it that way if you change them.
