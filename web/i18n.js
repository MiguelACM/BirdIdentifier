// Page text in each language. Keys used by data-i18n attributes in index.html
// and by t() in app.js. {name} placeholders are filled in by t().
// Species names come from i18n/species.<lang>.json (fetched from iNaturalist).

export const LANGUAGES = ["en", "pt"];

const STRINGS = {
  en: {
    pageTitle: "Bird Identifier",
    metaDescription: "Identify European birds from a photo. The model runs in your browser, so your photo never leaves your device.",
    langLabel: "Language",
    title: "Bird Identifier",
    tagline: "Identify 100 common European birds from a photo. It runs entirely in your browser, so your photo never leaves your device.",
    loading: "Loading model…",
    loadingPct: "Loading model… {pct}%",
    preparing: "Preparing model…",
    ready: "Model ready: {n} species",
    loadError: "Could not load the model ({msg}). See the README for how to put it in web/model/.",
    dropTitle: "Choose or take a photo",
    dropHint: "or drag and drop / paste an image here",
    examplesTitle: "No photo handy? Try one of these",
    exampleTitle: "{species}. Photo {attribution}",
    previewAlt: "Your photo",
    cropAria: "Crop area: drag to move, arrow keys to move, plus and minus to resize",
    cropHint: "The model only sees what's inside the square. Drag it onto the bird, or drag a corner to resize; results update when you let go.",
    cropReset: "Reset crop",
    topMatches: "Top matches",
    uncertain: "Not very confident. The bird may be small, far away or partly hidden, or it may not be one of the 100 species this model knows.",
    loadingExample: "Loading example…",
    identifying: "Identifying…",
    identified: "Identified in {ms} ms on your device.",
    error: "Something went wrong: {msg}",
    exampleLoadError: "could not load the example photo (HTTP {status})",
    exampleCredit: "Example photo: {species}. {attribution}, via ",
    correct: "✓ Correct: this photo is a {species}.",
    wrong: "✗ The model got this one wrong: it's actually a {species}. Similar-looking species like this are where it makes most of its mistakes.",
    countryLabel: "Country",
    countryNone: "Not set",
    countryRU: "Russia (European part)",
    monthLabel: "Month",
    monthAny: "Any time of year",
    locationOff: "Set the country to also use where and when the photo was taken.",
    locationOn: "Results also use how often each species is recorded there at that time of year (GBIF sightings).",
    fromPhoto: "Read from the photo's {what} (stays on your device).",
    fromPhotoGps: "location",
    fromPhotoDate: "date",
    fromPhotoBoth: "location and date",
    fromObservation: "Country and month of the iNaturalist observation.",
    outsideArea: "The photo was taken outside the countries covered, so location is not used.",
    withoutLocation: "Without location, the top match would be {species}.",
    rareMonth: "Few records in {country} in {month}",
    rareYear: "Few records in {country}",
    speciesSummary: "Which birds can it recognise?",
    footer: 'A learning project: a <code>convnext_nano</code> model fine-tuned on research-grade photos from <a href="https://www.inaturalist.org/">iNaturalist</a>, running with <a href="https://onnxruntime.ai/">ONNX Runtime Web</a>. Reference photos come from iNaturalist; location hints use sightings from <a href="https://www.gbif.org/">GBIF</a>. <a href="https://github.com/MiguelACM/BirdIdentifier">Source code</a>.',
  },
  pt: {
    pageTitle: "Identificador de Aves",
    metaDescription: "Identifique aves da Europa a partir de uma fotografia. O modelo corre no seu navegador, por isso a fotografia nunca sai do seu dispositivo.",
    langLabel: "Idioma",
    title: "Identificador de Aves",
    tagline: "Identifique 100 aves comuns da Europa a partir de uma fotografia. Tudo corre no seu navegador, por isso a sua fotografia nunca sai do seu dispositivo.",
    loading: "A carregar o modelo…",
    loadingPct: "A carregar o modelo… {pct}%",
    preparing: "A preparar o modelo…",
    ready: "Modelo pronto: {n} espécies",
    loadError: "Não foi possível carregar o modelo ({msg}). Consulte o README para saber como o colocar em web/model/.",
    dropTitle: "Escolher ou tirar uma fotografia",
    dropHint: "ou arraste e largue / cole uma imagem aqui",
    examplesTitle: "Não tem uma fotografia à mão? Experimente uma destas",
    exampleTitle: "{species}. Fotografia {attribution}",
    previewAlt: "A sua fotografia",
    cropAria: "Área de recorte: arraste ou use as setas para mover, mais e menos para redimensionar",
    cropHint: "O modelo só vê o que está dentro do quadrado. Arraste-o para cima da ave, ou arraste um canto para o redimensionar; os resultados atualizam quando largar.",
    cropReset: "Repor recorte",
    topMatches: "Espécies mais prováveis",
    uncertain: "Pouca confiança. A ave pode ser pequena, estar longe ou parcialmente escondida, ou pode não ser uma das 100 espécies que este modelo conhece.",
    loadingExample: "A carregar o exemplo…",
    identifying: "A identificar…",
    identified: "Identificação feita em {ms} ms no seu dispositivo.",
    error: "Algo correu mal: {msg}",
    exampleLoadError: "não foi possível carregar a fotografia de exemplo (HTTP {status})",
    exampleCredit: "Fotografia de exemplo: {species}. {attribution}, via ",
    correct: "✓ Certo: a espécie é {species}.",
    wrong: "✗ O modelo errou esta: na verdade, a espécie é {species}. É em espécies parecidas como esta que comete a maioria dos erros.",
    countryLabel: "País",
    countryNone: "Não definido",
    countryRU: "Rússia (parte europeia)",
    monthLabel: "Mês",
    monthAny: "Qualquer altura do ano",
    locationOff: "Defina o país para usar também onde e quando a fotografia foi tirada.",
    locationOn: "Os resultados também têm em conta a frequência com que cada espécie é registada nesse local nessa altura do ano (observações do GBIF).",
    fromPhoto: "Lido a partir {what} da fotografia (não sai do seu dispositivo).",
    fromPhotoGps: "da localização",
    fromPhotoDate: "da data",
    fromPhotoBoth: "da localização e da data",
    fromObservation: "País e mês da observação no iNaturalist.",
    outsideArea: "A fotografia foi tirada fora dos países abrangidos, por isso a localização não é usada.",
    withoutLocation: "Sem a localização, a espécie mais provável seria {species}.",
    rareMonth: "Poucos registos: {country}, {month}", // avoids the country article (na/no/em)
    rareYear: "Poucos registos: {country}",
    speciesSummary: "Que aves consegue reconhecer?",
    footer: 'Um projeto de aprendizagem: um modelo <code>convnext_nano</code> afinado com fotografias de grau de investigação do <a href="https://www.inaturalist.org/">iNaturalist</a>, a correr com <a href="https://onnxruntime.ai/">ONNX Runtime Web</a>. As fotografias de referência vêm do iNaturalist; as dicas de localização usam observações do <a href="https://www.gbif.org/">GBIF</a>. <a href="https://github.com/MiguelACM/BirdIdentifier">Código-fonte</a>.',
  },
};

const STORAGE_KEY = "bird-identifier-lang";

// Saved choice first, then the browser's preferred languages, then English.
export function initialLanguage() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (LANGUAGES.includes(saved)) return saved;
  } catch { /* storage may be blocked */ }
  for (const pref of navigator.languages || [navigator.language]) {
    const code = (pref || "").slice(0, 2).toLowerCase();
    if (LANGUAGES.includes(code)) return code;
  }
  return "en";
}

export function saveLanguage(lang) {
  try { localStorage.setItem(STORAGE_KEY, lang); } catch { /* ignore */ }
}

export function translate(lang, key, vars = {}) {
  const text = STRINGS[lang]?.[key] ?? STRINGS.en[key] ?? key;
  return text.replace(/\{(\w+)\}/g, (_, name) => vars[name] ?? "");
}

// Fill every element marked with data-i18n (text), data-i18n-html (trusted
// markup from this file), data-i18n-aria (aria-label) or data-i18n-alt (alt).
export function applyStaticText(lang) {
  document.documentElement.lang = lang;
  document.title = translate(lang, "pageTitle");
  document.querySelector('meta[name="description"]')
    ?.setAttribute("content", translate(lang, "metaDescription"));
  for (const el of document.querySelectorAll("[data-i18n]")) el.textContent = translate(lang, el.dataset.i18n);
  for (const el of document.querySelectorAll("[data-i18n-html]")) el.innerHTML = translate(lang, el.dataset.i18nHtml);
  for (const el of document.querySelectorAll("[data-i18n-aria]")) el.setAttribute("aria-label", translate(lang, el.dataset.i18nAria));
  for (const el of document.querySelectorAll("[data-i18n-alt]")) el.alt = translate(lang, el.dataset.i18nAlt);
}
