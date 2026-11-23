// Service worker: keeps the app working offline after the first visit.
//
// - The site's own files (page, code, model, priors) are fetched from the network first, so a
//   new deploy is picked up as soon as you're online, and every response is saved. Offline, or
//   when the network takes longer than NETWORK_TIMEOUT_MS to answer, the saved copy is used.
// - ONNX Runtime from the CDN has the version in its URL and never changes: cache first.
// - Everything else (iNaturalist photos and names) goes straight to the network.
//
// Bump CACHE when the list below changes; old caches are deleted on activate.

const CACHE = "bird-identifier-v1";
const NETWORK_TIMEOUT_MS = 4000;
const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";

// Saved on install, so the app works offline even if the visitor never used a feature (such as
// the Portuguese names or GPS lookup) while online. The model itself is saved when the page
// downloads it, so it isn't downloaded twice on the first visit.
const PRECACHE = [
  "./", "index.html", "style.css", "app.js", "i18n.js", "exif.js", "location.js",
  "priors.json", "country-grid.json", "examples.json", "i18n/species.pt.json",
  "model/labels.json", "model/config.json", "manifest.webmanifest",
  "icons/icon-192.png", "icons/favicon-32.png",
  `${ORT}ort.wasm.min.mjs`, `${ORT}ort-wasm-simd-threaded.mjs`, `${ORT}ort-wasm-simd-threaded.wasm`,
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then((cache) => cache.addAll(PRECACHE)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    for (const key of await caches.keys()) if (key !== CACHE) await caches.delete(key);
    await self.clients.claim();
  })());
});

// The page asks for files it loaded before the worker was running (the model, on a first visit)
self.addEventListener("message", (event) => {
  const urls = event.data?.save;
  if (!Array.isArray(urls)) return;
  event.waitUntil((async () => {
    const cache = await caches.open(CACHE);
    for (const url of urls) {
      const full = new URL(url, self.location).href;
      if (full.startsWith(self.registration.scope) && !(await cache.match(full))) await cache.add(full);
    }
  })());
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.origin === self.location.origin) event.respondWith(networkFirst(request));
  else if (url.href.startsWith(ORT)) event.respondWith(cacheFirst(request));
});

async function networkFirst(request) {
  const cache = await caches.open(CACHE);
  const network = fetch(request).then((response) => {
    if (response.ok) cache.put(request, response.clone());
    return response;
  });
  const cached = await cache.match(request, { ignoreSearch: true });
  if (!cached) return network;
  // Use the network if it answers in time; otherwise (offline, very slow signal) the saved copy
  const timeout = new Promise((resolve) => setTimeout(resolve, NETWORK_TIMEOUT_MS, cached));
  return Promise.race([network.catch(() => cached), timeout]);
}

async function cacheFirst(request) {
  const cache = await caches.open(CACHE);
  const cached = await cache.match(request);
  if (cached) return cached;
  const response = await fetch(request);
  if (response.ok) cache.put(request, response.clone());
  return response;
}
