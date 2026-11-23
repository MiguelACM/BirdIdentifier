# Model files

The model the web app uses:

| File | What it is |
|---|---|
| `model.int8.onnx` | `convnext_nano.in12k_ft_in1k` fine-tuned on 300 European bird species, int8-quantised (15.6 MB). Input: 1×3×288×288 RGB in 0–1. Output: 300 softmax probabilities. Normalisation is built in. |
| `labels.json` | Class index → iNaturalist taxon ID, scientific name, common name |
| `config.json` | Preprocessing settings (resize the short side to 303, centre-crop 288) |

Test-set accuracy (full-precision model): 76.5% top-1, 89.6% top-5. The int8 version
scores slightly lower. See the main README for how it was trained and exported.

## License

The weights were trained on iNaturalist photos, most of which are licensed CC BY-NC
or all-rights-reserved. The model is therefore shared **for non-commercial use only**,
for learning and demonstration. The training photos themselves are not included in
this repository.
