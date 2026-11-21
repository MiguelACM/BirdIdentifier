// Location + season hints. priors.json holds, for each European country and
// month, how common each of the model's species is among GBIF sightings
// (built by ml/src/fetch_gbif_priors.py). The model's probabilities are
// multiplied by prior^PRIOR_WEIGHT and renormalised, which re-ranks look-alikes
// by what is actually seen there at that time of year.

// Chosen on the validation set by ml/src/eval_priors.py (see the README)
export const PRIOR_WEIGHT = 0.7;
// A species with less than this share of sightings (vs 1% for an average species
// among the 100) is flagged as "few records" there and then.
export const RARE_BELOW = 0.001;

let priors = null, grid = null;

export async function loadPriors(labels) {
  const data = await fetch("priors.json").then((r) => r.json());
  // The prior columns must line up with the model's outputs
  if (data.taxa.length !== labels.length || data.taxa.some((id, i) => id !== labels[i].taxon_id)) {
    throw new Error("priors.json does not match the model's labels");
  }
  priors = data;
  return Object.keys(data.countries);
}

// Prior shares for a country and month (1-12), or the whole year if month is
// null. Returns null when there is no prior to apply.
export function priorFor(country, month) {
  const entry = priors?.countries[country];
  if (!entry) return null;
  const values = month ? entry.months[month - 1] : entry.year;
  return values.map((v) => v / priors.scale);
}

export function applyPrior(probs, prior) {
  if (!prior) return Float32Array.from(probs);
  const out = new Float32Array(probs.length);
  let total = 0;
  for (let i = 0; i < probs.length; i++) {
    out[i] = probs[i] * prior[i] ** PRIOR_WEIGHT;
    total += out[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= total;
  return out;
}

// GPS position -> country code using the run-length encoded grid built by
// ml/src/make_country_grid.py. Returns "" for open sea and "-" for a country
// outside the supported list. Loaded only when a photo has GPS data.
export async function countryAt(lat, lon) {
  grid ??= await fetch("country-grid.json").then((r) => r.json());
  const row = Math.floor((lat - grid.lat0) / grid.step);
  let col = Math.floor((lon - grid.lon0) / grid.step);
  if (row < 0 || row >= grid.rows || col < 0 || col >= grid.cols) return "-";
  const runs = grid.rle[row];
  for (let i = 0; i < runs.length; i += 2) {
    col -= runs[i];
    if (col < 0) return grid.codes[runs[i + 1]];
  }
  return "";
}
