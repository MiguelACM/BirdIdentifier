// Bird Identifier: runs the exported ONNX model in the browser with ONNX Runtime Web.
//
// Flow: load model + labels -> user picks a photo -> classify the centre square
// (same as the validation transform in ml/src/train.py) -> show the top 5
// species. The user can then move/resize the square onto the bird, and the
// model re-runs on that crop when they let go.

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.min.mjs";
import { LANGUAGES, applyStaticText, initialLanguage, saveLanguage, translate } from "./i18n.js";
import { readExif } from "./exif.js";
import { RARE_BELOW, applyPrior, countryAt, loadPriors, priorFor } from "./location.js";

const MODEL_DIR = "model";
// 8-bit version: 4x smaller download than model.onnx and only ~0.5-1% less accurate.
// Swap in "model.onnx" to use the full-precision model.
const MODEL_FILE = "model.int8.onnx";
const TOP_K = 5;
const UNCERTAIN_BELOW = 0.4; // show a warning when the best match is less likely than this

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";

const $ = (id) => document.getElementById(id);
const els = {
  status: $("model-status"),
  statusText: $("model-status-text"),
  progress: $("model-progress"),
  dropzone: $("dropzone"),
  fileInput: $("file-input"),
  result: $("result"),
  preview: $("preview-img"),
  list: $("prediction-list"),
  uncertain: $("uncertain"),
  timing: $("timing"),
  speciesList: $("species-list"),
  exampleList: $("example-list"),
  exampleCredit: $("example-credit"),
  exampleVerdict: $("example-verdict"),
  cropBox: $("crop-box"),
  cropReset: $("crop-reset"),
  location: $("location"),
  countrySelect: $("country-select"),
  monthSelect: $("month-select"),
  locationNote: $("location-note"),
};

let session, labels, config, examples = [];

// ------------------------------------------------------------------ language

let lang = "en";
const speciesNames = {}; // lang -> {taxon_id: name}; English comes from labels.json
const t = (key, vars) => translate(lang, key, vars);
const speciesName = (taxonId, english) => speciesNames[lang]?.[taxonId] ?? english;

// The model status line is re-rendered when the language changes
let status = { key: "loading", vars: {} };
function setStatus(key, vars = {}) {
  status = { key, vars };
  els.statusText.textContent = t(key, vars);
}

async function setLanguage(newLang) {
  if (newLang !== "en" && !speciesNames[newLang]) {
    speciesNames[newLang] = await fetch(`i18n/species.${newLang}.json`)
      .then((r) => r.json()).catch(() => ({}));
  }
  lang = newLang;
  applyStaticText(lang);
  for (const btn of document.querySelectorAll(".lang-switch button")) {
    btn.setAttribute("aria-pressed", String(btn.dataset.lang === lang));
  }
  setStatus(status.key, status.vars);
  if (labels) renderSpeciesList();
  renderLocationControls();
  renderExamples();
  if (current.bitmap) showExampleCredit(current.example);
  if (lastRun) showResults(lastRun.probs, lastRun.ms); // re-render, no need to re-run the model
}

// ------------------------------------------------------------------ loading

async function fetchWithProgress(url, onProgress) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${url}: HTTP ${resp.status}`);
  const total = Number(resp.headers.get("Content-Length")) || 0;
  if (!total || !resp.body) return new Uint8Array(await resp.arrayBuffer());

  const reader = resp.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(received / total);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) { bytes.set(c, offset); offset += c.length; }
  return bytes;
}

async function loadModel() {
  [labels, config] = await Promise.all([
    fetch(`${MODEL_DIR}/labels.json`).then((r) => r.json()),
    fetch(`${MODEL_DIR}/config.json`).then((r) => r.json()),
  ]);
  renderSpeciesList();
  loadPriors(labels).then((codes) => {
    countries = codes;
    renderLocationControls();
  }).catch((err) => console.warn("Location hints unavailable:", err));

  const bytes = await fetchWithProgress(`${MODEL_DIR}/${MODEL_FILE}`, (f) => {
    els.progress.value = f;
    setStatus("loadingPct", { pct: Math.round(f * 100) });
  });
  setStatus("preparing");
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });

  els.status.classList.add("ready");
  setStatus("ready", { n: labels.length });
  els.dropzone.classList.remove("disabled");
  for (const btn of els.exampleList.querySelectorAll("button")) btn.disabled = false;
}

// ------------------------------------------------------------------ examples

// Test-set photos (never seen in training), CC0 / CC BY, loaded from iNaturalist.
// One is a known mistake on purpose, to show the model's limits honestly.
async function loadExamples() {
  examples = await fetch("examples.json").then((r) => r.json());
  renderExamples();
}

function renderExamples() {
  els.exampleList.replaceChildren(...examples.map((ex) => {
    const name = speciesName(ex.taxon_id, ex.species);
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "example";
    btn.disabled = !session;
    btn.title = t("exampleTitle", { species: name, attribution: ex.attribution });
    const img = document.createElement("img");
    img.src = ex.thumb;
    img.alt = name;
    img.loading = "lazy";
    const caption = document.createElement("span");
    caption.textContent = name;
    btn.append(img, caption);
    btn.addEventListener("click", () => classifyExample(ex).catch(showError));
    return btn;
  }));
}

async function classifyExample(ex) {
  els.timing.textContent = t("loadingExample");
  const resp = await fetch(ex.url);
  if (!resp.ok) throw new Error(t("exampleLoadError", { status: resp.status }));
  const blob = await resp.blob();
  await classify(new File([blob], "example.jpg", { type: blob.type }), ex);
}

function renderSpeciesList() {
  const named = labels.map((s) => ({ ...s, name: speciesName(s.taxon_id, s.common_name) }));
  named.sort((a, b) => a.name.localeCompare(b.name, lang));
  els.speciesList.replaceChildren(...named.map((s) => {
    const li = document.createElement("li");
    li.append(`${s.name} `);
    const i = document.createElement("i");
    i.textContent = s.scientific_name;
    li.append(i);
    return li;
  }));
}

// ------------------------------------------------------------------ preprocessing

// The default crop mirrors the eval transform in train.py: resizing the short
// side to config.resize px and centre-cropping config.img_size px is the same as
// taking a centred square whose side is img_size/resize of the short side.
function defaultCrop(width, height) {
  const size = Math.min(width, height) * (config.img_size / config.resize);
  return { x: (width - size) / 2, y: (height - size) / 2, size };
}

// Draw the crop square (in original-photo pixels) at img_size x img_size,
// scale RGB to 0..1 and lay it out as CHW. (Normalisation is built into the model.)
function cropToTensor(bitmap, crop) {
  const size = config.img_size;

  // Downscale in halving steps: one big jump (e.g. 4000px -> 224px) looks
  // jagged in some browsers, which would hurt accuracy.
  let source = bitmap, srcX = crop.x, srcY = crop.y, srcSide = crop.size;
  while (srcSide > size * 2) {
    const half = Math.round(srcSide / 2);
    const step = new OffscreenCanvas(half, half);
    const stepCtx = step.getContext("2d");
    stepCtx.imageSmoothingQuality = "high";
    stepCtx.drawImage(source, srcX, srcY, srcSide, srcSide, 0, 0, half, half);
    source = step; srcX = 0; srcY = 0; srcSide = half;
  }

  const canvas = new OffscreenCanvas(size, size);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(source, srcX, srcY, srcSide, srcSide, 0, 0, size, size);

  const { data } = ctx.getImageData(0, 0, size, size); // RGBA, row by row
  const plane = size * size;
  const chw = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    chw[i] = data[i * 4] / 255;                 // R
    chw[plane + i] = data[i * 4 + 1] / 255;     // G
    chw[2 * plane + i] = data[i * 4 + 2] / 255; // B
  }
  return new ort.Tensor("float32", chw, [1, 3, size, size]);
}

// ------------------------------------------------------------------ prediction

// The photo currently shown: decoded once, re-cropped as the user moves the box.
const current = { bitmap: null, crop: null, example: null };

async function classify(file, example = null) {
  if (!session) return;
  // createImageBitmap applies EXIF orientation, so phone photos come out upright
  const bitmap = await createImageBitmap(file);
  current.bitmap?.close();
  Object.assign(current, { bitmap, crop: defaultCrop(bitmap.width, bitmap.height), example });

  if (els.preview.src) URL.revokeObjectURL(els.preview.src);
  els.preview.src = URL.createObjectURL(file);
  els.result.hidden = false;
  showExampleCredit(example);
  Object.assign(place, await placeForPhoto(file, example));
  renderLocationControls();
  renderCropBox();
  await runModel();
}

// Run the model on the current crop. If the user moves the box while a run is
// in progress, run once more afterwards with the latest crop.
let running = false, rerun = false;
let lastRun = null; // kept so a language switch can re-render without re-running
async function runModel() {
  if (running) { rerun = true; return; }
  running = true;
  try {
    do {
      rerun = false;
      els.timing.textContent = t("identifying");
      const t0 = performance.now();
      const output = await session.run({ image: cropToTensor(current.bitmap, current.crop) });
      lastRun = { probs: output.probs.data, ms: performance.now() - t0 };
      showResults(lastRun.probs, lastRun.ms);
    } while (rerun);
  } finally {
    running = false;
  }
}

function showResults(probs, ms) {
  // Re-rank with the location + season prior (if a country is set)
  const prior = priorFor(place.country, place.month);
  const adjusted = applyPrior(probs, prior);
  const top = [...adjusted.keys()].sort((a, b) => adjusted[b] - adjusted[a]).slice(0, TOP_K);
  const results = top.map((i) => ({ ...labels[i], p: adjusted[i], prior: prior?.[i] }));
  const photoTop = probs.indexOf(Math.max(...probs));
  els.locationNote.textContent = locationNote(photoTop === top[0] ? null : labels[photoTop]);

  els.uncertain.hidden = results[0].p >= UNCERTAIN_BELOW;
  els.exampleVerdict.hidden = true;
  els.list.replaceChildren(...results.map(renderPrediction));
  els.timing.textContent = t("identified", { ms: Math.round(ms) });
  if (current.example) showExampleVerdict(current.example, results[0]);
  addReferencePhotos(results);
}

// ------------------------------------------------------------------ crop box

// Crop coordinates are kept in original-photo pixels; the box on screen is
// positioned in percentages of the displayed image, so window resizes need no code.
function renderCropBox() {
  const { width, height } = current.bitmap;
  const { x, y, size } = current.crop;
  Object.assign(els.cropBox.style, {
    left: `${(x / width) * 100}%`,
    top: `${(y / height) * 100}%`,
    width: `${(size / width) * 100}%`,
    height: `${(size / height) * 100}%`,
  });
}

const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const minCropSize = () => Math.min(current.bitmap.width, current.bitmap.height) * 0.1;

// Screen pixels -> original-photo pixels
function toPhotoCoords(e) {
  const rect = els.preview.getBoundingClientRect();
  const scale = current.bitmap.width / rect.width;
  return { x: (e.clientX - rect.left) * scale, y: (e.clientY - rect.top) * scale };
}

let drag = null;

function startDrag(e) {
  if (!current.bitmap) return;
  e.preventDefault();
  const handle = e.target.closest(".handle");
  const p = toPhotoCoords(e);
  const { x, y, size } = current.crop;
  if (handle) {
    // Resize from a corner, keeping the opposite corner fixed
    const sx = Number(handle.dataset.sx), sy = Number(handle.dataset.sy);
    drag = { mode: "resize", sx, sy, ax: sx > 0 ? x : x + size, ay: sy > 0 ? y : y + size };
  } else {
    drag = { mode: "move", start: p, crop: { ...current.crop } };
  }
  els.cropBox.setPointerCapture(e.pointerId);
}

function onDrag(e) {
  if (!drag) return;
  const p = toPhotoCoords(e);
  const { width: W, height: H } = current.bitmap;
  if (drag.mode === "move") {
    const { x, y, size } = drag.crop;
    current.crop = {
      size,
      x: clamp(x + p.x - drag.start.x, 0, W - size),
      y: clamp(y + p.y - drag.start.y, 0, H - size),
    };
  } else {
    const { sx, sy, ax, ay } = drag;
    const wanted = Math.max(sx * (p.x - ax), sy * (p.y - ay));
    const room = Math.min(sx > 0 ? W - ax : ax, sy > 0 ? H - ay : ay);
    const size = clamp(wanted, minCropSize(), room);
    current.crop = { size, x: sx > 0 ? ax : ax - size, y: sy > 0 ? ay : ay - size };
  }
  renderCropBox();
}

function endDrag() {
  if (!drag) return;
  drag = null;
  runModel().catch(showError);
}

// Keyboard: arrows move the box, +/- resize it (around its centre)
let keyTimer;
function onCropKey(e) {
  if (!current.bitmap) return;
  const { width: W, height: H } = current.bitmap;
  const step = Math.min(W, H) * (e.shiftKey ? 0.1 : 0.02);
  let { x, y, size } = current.crop;
  const moves = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
  if (moves[e.key]) {
    x += moves[e.key][0];
    y += moves[e.key][1];
  } else if (e.key === "+" || e.key === "=" || e.key === "-") {
    const newSize = clamp(size + (e.key === "-" ? -step : step), minCropSize(), Math.min(W, H));
    x += (size - newSize) / 2;
    y += (size - newSize) / 2;
    size = newSize;
  } else {
    return;
  }
  e.preventDefault();
  current.crop = { size, x: clamp(x, 0, W - size), y: clamp(y, 0, H - size) };
  renderCropBox();
  clearTimeout(keyTimer);
  keyTimer = setTimeout(() => runModel().catch(showError), 300); // wait until keys stop
}

function resetCrop() {
  if (!current.bitmap) return;
  current.crop = defaultCrop(current.bitmap.width, current.bitmap.height);
  renderCropBox();
  runModel().catch(showError);
}

function showExampleCredit(example) {
  els.exampleCredit.hidden = !example;
  if (!example) return;
  const link = document.createElement("a");
  link.href = example.observation;
  link.target = "_blank";
  link.rel = "noopener";
  link.textContent = "iNaturalist";
  const species = speciesName(example.taxon_id, example.species);
  els.exampleCredit.replaceChildren(
    t("exampleCredit", { species, attribution: example.attribution }), link, ".");
}

function showExampleVerdict(example, best) {
  const correct = best.taxon_id === example.taxon_id;
  const species = speciesName(example.taxon_id, example.species);
  els.exampleVerdict.classList.toggle("ok", correct);
  els.exampleVerdict.textContent = t(correct ? "correct" : "wrong", { species });
  els.exampleVerdict.hidden = false;
  if (!correct) els.uncertain.hidden = true; // the verdict already explains it
}

function renderPrediction(r) {
  const li = document.createElement("li");
  li.className = "prediction";
  li.dataset.taxonId = r.taxon_id;
  li.innerHTML = `
    <img class="thumb" alt="" loading="lazy">
    <div>
      <div class="names">
        <span class="common"></span>
        <span class="scientific"></span>
      </div>
      <div class="bar-row">
        <div class="bar"><span></span></div>
        <span class="pct"></span>
      </div>
      <div class="rare" hidden></div>
      <div class="links"><a target="_blank" rel="noopener">iNaturalist</a></div>
    </div>`;
  li.querySelector(".common").textContent = speciesName(r.taxon_id, r.common_name);
  li.querySelector(".scientific").textContent = r.scientific_name;
  li.querySelector(".bar > span").style.width = `${(r.p * 100).toFixed(1)}%`;
  li.querySelector(".pct").textContent = `${(r.p * 100).toFixed(r.p < 0.1 ? 1 : 0)}%`;
  li.querySelector("a").href = `https://www.inaturalist.org/taxa/${r.taxon_id}`;
  if (r.prior !== undefined && r.prior < RARE_BELOW) {
    const rare = li.querySelector(".rare");
    const country = countryName(place.country);
    rare.textContent = place.month
      ? t("rareMonth", { country, month: monthName(place.month) })
      : t("rareYear", { country });
    rare.hidden = false;
  }
  return li;
}

// Show a reference photo of each predicted species so you can compare.
// Only the species IDs are sent to iNaturalist, never your photo.
async function addReferencePhotos(results) {
  try {
    const ids = results.map((r) => r.taxon_id).join(",");
    // locale makes iNaturalist return the Wikipedia link in the page's language
    const resp = await fetch(`https://api.inaturalist.org/v1/taxa/${ids}?locale=${lang}`);
    const { results: taxa } = await resp.json();
    for (const t of taxa) {
      const li = els.list.querySelector(`[data-taxon-id="${t.id}"]`);
      const photo = t.default_photo;
      if (!li || !photo) continue;
      const img = li.querySelector(".thumb");
      img.src = photo.square_url;
      img.title = photo.attribution || "";
      if (t.wikipedia_url) {
        const wiki = document.createElement("a");
        wiki.href = t.wikipedia_url;
        wiki.target = "_blank";
        wiki.rel = "noopener";
        wiki.textContent = "Wikipedia";
        li.querySelector(".links").append(" · ", wiki);
      }
    }
  } catch {
    // Reference photos are a nice-to-have; ignore network errors
  }
}

// ------------------------------------------------------------------ location + season

// Where and when the current photo was taken. Comes from the photo's EXIF data,
// an example's iNaturalist observation, or the visitor's own choice. Off (no
// country) until one of those sets it; only a country picked by hand is remembered.
const COUNTRY_KEY = "bird-identifier-country";
let countries = []; // supported country codes, once priors.json has loaded
const currentMonth = () => new Date().getMonth() + 1;
const place = { country: savedCountry(), month: currentMonth(), source: null, what: null, outside: false };

function savedCountry() {
  try { return localStorage.getItem(COUNTRY_KEY) || ""; } catch { return ""; }
}
function saveCountry(code) {
  try {
    if (code) localStorage.setItem(COUNTRY_KEY, code);
    else localStorage.removeItem(COUNTRY_KEY);
  } catch { /* storage may be blocked */ }
}

// Russia is only covered west of the Urals, so say so in its name
const countryName = (code) => code === "RU" ? t("countryRU")
  : new Intl.DisplayNames([lang], { type: "region" }).of(code);
const monthName = (m) => new Intl.DateTimeFormat(lang, { month: "long" }).format(new Date(2000, m - 1, 15));
const capitalise = (s) => s.charAt(0).toUpperCase() + s.slice(1);

async function placeForPhoto(file, example) {
  if (example?.country) {
    const known = countries.includes(example.country);
    return { country: known ? example.country : "", month: example.month, source: "observation",
             what: null, outside: !known };
  }
  const exif = await readExif(file);
  const result = { country: savedCountry(), month: exif.month ?? currentMonth(), source: null, what: null, outside: false };
  let fromGps = false;
  if (exif.lat !== null && countries.length) {
    const code = await countryAt(exif.lat, exif.lon).catch(() => "");
    if (countries.includes(code)) {
      result.country = code;
      fromGps = true;
    } else if (code === "-") {
      result.country = ""; // outside the covered countries: priors would mislead
      result.outside = true;
    }
  }
  if (fromGps || exif.month) {
    result.source = "photo";
    result.what = fromGps && exif.month ? "fromPhotoBoth" : fromGps ? "fromPhotoGps" : "fromPhotoDate";
  }
  return result;
}

function renderLocationControls() {
  els.location.hidden = !countries.length;
  if (!countries.length) return;
  const named = countries.map((code) => ({ code, name: countryName(code) }))
    .sort((a, b) => a.name.localeCompare(b.name, lang));
  els.countrySelect.replaceChildren(
    new Option(t("countryNone"), ""), ...named.map(({ code, name }) => new Option(name, code)));
  els.countrySelect.value = place.country;
  els.monthSelect.replaceChildren(new Option(t("monthAny"), ""),
    ...Array.from({ length: 12 }, (_, i) => new Option(capitalise(monthName(i + 1)), String(i + 1))));
  els.monthSelect.value = place.month ? String(place.month) : "";
  els.monthSelect.disabled = !place.country;
}

function locationNote(photoOnlyTop) {
  if (!place.country) return t(place.outside ? "outsideArea" : "locationOff");
  const parts = [];
  if (place.source === "observation") parts.push(t("fromObservation"));
  if (place.source === "photo") parts.push(t("fromPhoto", { what: t(place.what) }));
  parts.push(t("locationOn"));
  if (photoOnlyTop) {
    parts.push(t("withoutLocation", { species: speciesName(photoOnlyTop.taxon_id, photoOnlyTop.common_name) }));
  }
  return parts.join(" ");
}

function onPlaceChange() {
  Object.assign(place, {
    country: els.countrySelect.value,
    month: Number(els.monthSelect.value) || null,
    source: null, // chosen by hand now
    outside: false,
  });
  renderLocationControls();
  if (lastRun) showResults(lastRun.probs, lastRun.ms); // re-rank, no need to re-run the model
}

// ------------------------------------------------------------------ input handling

function handleFiles(files) {
  const file = [...(files || [])].find((f) => f.type.startsWith("image/"));
  if (file) classify(file).catch(showError);
}

function showError(err) {
  console.error(err);
  els.timing.textContent = t("error", { msg: err.message });
}

els.dropzone.classList.add("disabled");
els.fileInput.addEventListener("change", () => {
  handleFiles(els.fileInput.files);
  els.fileInput.value = ""; // allow picking the same file again
});
els.dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); els.fileInput.click(); }
});
for (const type of ["dragenter", "dragover"]) {
  els.dropzone.addEventListener(type, (e) => { e.preventDefault(); els.dropzone.classList.add("dragover"); });
}
for (const type of ["dragleave", "drop"]) {
  els.dropzone.addEventListener(type, () => els.dropzone.classList.remove("dragover"));
}
els.dropzone.addEventListener("drop", (e) => { e.preventDefault(); handleFiles(e.dataTransfer.files); });
document.addEventListener("paste", (e) => handleFiles(e.clipboardData?.files));

els.cropBox.addEventListener("pointerdown", startDrag);
els.cropBox.addEventListener("pointermove", onDrag);
els.cropBox.addEventListener("pointerup", endDrag);
els.cropBox.addEventListener("pointercancel", endDrag);
els.cropBox.addEventListener("keydown", onCropKey);
els.cropReset.addEventListener("click", resetCrop);
els.countrySelect.addEventListener("change", () => {
  saveCountry(els.countrySelect.value); // remember a country picked by hand
  onPlaceChange();
});
els.monthSelect.addEventListener("change", onPlaceChange);

for (const btn of document.querySelectorAll(".lang-switch button")) {
  btn.addEventListener("click", () => {
    saveLanguage(btn.dataset.lang); // only remembered when the visitor chooses
    setLanguage(btn.dataset.lang).catch(showError);
  });
}

// Pick the language before loading anything else, so nothing flashes in English
await setLanguage(initialLanguage());
loadExamples().catch((err) => console.warn("Examples unavailable:", err));
loadModel().catch((err) => {
  console.error(err);
  els.status.classList.add("error");
  setStatus("loadError", { msg: err.message });
});
