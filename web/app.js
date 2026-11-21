// Bird Identifier: runs the exported ONNX model in the browser with ONNX Runtime Web.
//
// Flow: load model + labels -> user picks a photo -> resize & centre-crop to
// 224x224 (same as the validation transform in ml/src/train.py) -> run the
// model -> show the top 5 species.

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.min.mjs";

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
};

let session, labels, config;

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

  const bytes = await fetchWithProgress(`${MODEL_DIR}/${MODEL_FILE}`, (f) => {
    els.progress.value = f;
    els.statusText.textContent = `Loading model… ${Math.round(f * 100)}%`;
  });
  els.statusText.textContent = "Preparing model…";
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });

  els.status.classList.add("ready");
  els.statusText.textContent = `Model ready: ${labels.length} species`;
  els.dropzone.classList.remove("disabled");
  for (const btn of els.exampleList.querySelectorAll("button")) btn.disabled = false;
}

// ------------------------------------------------------------------ examples

// Test-set photos (never seen in training), CC0 / CC BY, loaded from iNaturalist.
// One is a known mistake on purpose, to show the model's limits honestly.
async function loadExamples() {
  const examples = await fetch("examples.json").then((r) => r.json());
  els.exampleList.replaceChildren(...examples.map((ex) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "example";
    btn.disabled = !session;
    btn.title = `${ex.species}. Photo ${ex.attribution}`;
    const img = document.createElement("img");
    img.src = ex.thumb;
    img.alt = ex.species;
    img.loading = "lazy";
    const caption = document.createElement("span");
    caption.textContent = ex.species;
    btn.append(img, caption);
    btn.addEventListener("click", () => classifyExample(ex).catch(showError));
    return btn;
  }));
}

async function classifyExample(ex) {
  els.timing.textContent = "Loading example…";
  const resp = await fetch(ex.url);
  if (!resp.ok) throw new Error(`could not load the example photo (HTTP ${resp.status})`);
  const blob = await resp.blob();
  await classify(new File([blob], "example.jpg", { type: blob.type }), ex);
}

function renderSpeciesList() {
  const sorted = [...labels].sort((a, b) => a.common_name.localeCompare(b.common_name));
  els.speciesList.replaceChildren(...sorted.map((s) => {
    const li = document.createElement("li");
    li.append(`${s.common_name} `);
    const i = document.createElement("i");
    i.textContent = s.scientific_name;
    li.append(i);
    return li;
  }));
}

// ------------------------------------------------------------------ preprocessing

// Mirror of the eval transform in train.py: resize so the short side is
// config.resize px, centre-crop config.img_size px, scale RGB to 0..1, CHW layout.
// (Normalisation is built into the ONNX model.)
async function imageToTensor(file) {
  // createImageBitmap applies EXIF orientation, so phone photos come out upright
  const bitmap = await createImageBitmap(file);
  const size = config.img_size;
  const scale = config.resize / Math.min(bitmap.width, bitmap.height);
  const cropInSource = size / scale; // side of the centre square, in original pixels
  const sx = (bitmap.width - cropInSource) / 2;
  const sy = (bitmap.height - cropInSource) / 2;

  // Downscale in halving steps: one big jump (e.g. 4000px -> 224px) looks
  // jagged in some browsers, which would hurt accuracy.
  let source = bitmap, srcX = sx, srcY = sy, srcSide = cropInSource;
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
  bitmap.close();

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

async function classify(file, example = null) {
  if (!session) return;
  els.preview.src = URL.createObjectURL(file);
  els.result.hidden = false;
  els.list.replaceChildren();
  els.uncertain.hidden = true;
  els.exampleVerdict.hidden = true;
  els.timing.textContent = "Identifying…";
  showExampleCredit(example);

  const t0 = performance.now();
  const input = await imageToTensor(file);
  const output = await session.run({ image: input });
  const probs = output.probs.data;
  const ms = performance.now() - t0;

  const top = [...probs.keys()].sort((a, b) => probs[b] - probs[a]).slice(0, TOP_K);
  const results = top.map((i) => ({ ...labels[i], p: probs[i] }));

  els.uncertain.hidden = results[0].p >= UNCERTAIN_BELOW;
  els.list.replaceChildren(...results.map(renderPrediction));
  els.timing.textContent = `Identified in ${Math.round(ms)} ms on your device.`;
  if (example) showExampleVerdict(example, results[0]);
  addReferencePhotos(results);
}

function showExampleCredit(example) {
  els.exampleCredit.hidden = !example;
  if (!example) return;
  const link = document.createElement("a");
  link.href = example.observation;
  link.target = "_blank";
  link.rel = "noopener";
  link.textContent = "iNaturalist";
  els.exampleCredit.replaceChildren(
    `Example photo: ${example.species}. ${example.attribution}, via `, link, ".");
}

function showExampleVerdict(example, best) {
  const correct = best.common_name === example.species;
  els.exampleVerdict.classList.toggle("ok", correct);
  els.exampleVerdict.textContent = correct
    ? `✓ Correct: this photo is a ${example.species}.`
    : `✗ The model got this one wrong: it's actually a ${example.species}. ` +
      `Similar-looking species like this are where it makes most of its mistakes.`;
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
      <div class="links"><a target="_blank" rel="noopener">iNaturalist</a></div>
    </div>`;
  li.querySelector(".common").textContent = r.common_name;
  li.querySelector(".scientific").textContent = r.scientific_name;
  li.querySelector(".bar > span").style.width = `${(r.p * 100).toFixed(1)}%`;
  li.querySelector(".pct").textContent = `${(r.p * 100).toFixed(r.p < 0.1 ? 1 : 0)}%`;
  li.querySelector("a").href = `https://www.inaturalist.org/taxa/${r.taxon_id}`;
  return li;
}

// Show a reference photo of each predicted species so you can compare.
// Only the species IDs are sent to iNaturalist, never your photo.
async function addReferencePhotos(results) {
  try {
    const ids = results.map((r) => r.taxon_id).join(",");
    const resp = await fetch(`https://api.inaturalist.org/v1/taxa/${ids}`);
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

// ------------------------------------------------------------------ input handling

function handleFiles(files) {
  const file = [...(files || [])].find((f) => f.type.startsWith("image/"));
  if (file) classify(file).catch(showError);
}

function showError(err) {
  console.error(err);
  els.timing.textContent = `Something went wrong: ${err.message}`;
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

loadExamples().catch((err) => console.warn("Examples unavailable:", err));
loadModel().catch((err) => {
  console.error(err);
  els.status.classList.add("error");
  els.statusText.textContent =
    `Could not load the model (${err.message}). See the README for how to put it in web/model/.`;
});
