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
- [x] Train on a GPU (Kaggle) and evaluate: **75.2% top-1, 90.1% top-5** on the test set
- [ ] Improve: more epochs, higher resolution (288px), more photos per species
- [x] Web app: upload a photo and see the top 5 species, running locally in the browser
- [x] Re-rank predictions by location and season (GBIF sightings per country and month)
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
is first classified using the same centre square as the validation transform, on the
device, in about 0.4 s. A **crop box** shows exactly what the model sees. Users can drag it
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
  and stores each species' share of the country-month total in `web/priors.json` (200 KB,
  ~80 KB compressed). Using shares rather than raw counts removes most of the "more birders
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
and date. It picks the weight `w` on the validation set (best at 0.7; anything from 0.4 to 0.8
is within 0.3 points), then measures on the test set (int8 model):

| test set (3049 photos) | top-1 | top-5 |
|---|---|---|
| photo only | 74.5% | 90.2% |
| + country (whole year) | 76.2% | 90.3% |
| + country and month | **76.6%** | **90.7%** |

All but 5 test photos were taken in a covered country (22% of them in European Russia, which
the first version left out: adding it took the gain from +1.6 to +2.1 points). A weight above
1 starts to hurt:
the prior then overrules clear photos of birds that are simply rare in that place.

Six **example photos** (`web/examples.json`) let visitors try it without a photo of their own.
They come from the test set, so the model never saw them during training, and are CC0 / CC BY
licensed, credited and loaded from iNaturalist. Five are correct; one (a Herring Gull taken
for a Lesser Black-backed Gull) is a known mistake, kept on purpose to show the model's limits.

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
Location priors are derived from occurrence counts on [GBIF.org](https://www.gbif.org/), and
country borders come from Natural Earth (public domain).
The scripts download them for personal, non-commercial study, and
`data/raw/manifest.csv` records the license and author of every image.
Any trained model weights shared from this project are for non-commercial use only.

The scripts follow iNaturalist's API guidance (about 1 request per second).
Please keep it that way if you change them.
