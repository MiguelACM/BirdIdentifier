// Tests for the web app's plain-JavaScript helpers. Run with: node --test "tests/web/*.test.mjs"
import assert from "node:assert/strict";
import { test } from "node:test";

import { STRINGS, translate } from "../../web/i18n.js";
import { PRIOR_WEIGHT, applyPrior } from "../../web/location.js";

test("every language has the same text keys and placeholders", () => {
  const placeholders = (s) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort().join(",");
  for (const [lang, strings] of Object.entries(STRINGS)) {
    assert.deepEqual(Object.keys(strings).sort(), Object.keys(STRINGS.en).sort(), lang);
    for (const [key, text] of Object.entries(strings)) {
      assert.equal(placeholders(text), placeholders(STRINGS.en[key]), `${lang}.${key}`);
    }
  }
});

test("translate fills placeholders and falls back to English", () => {
  assert.match(translate("pt", "tagline", { n: 300 }), /300/);
  assert.equal(translate("xx", "pageTitle"), STRINGS.en.pageTitle);
  assert.equal(translate("en", "no-such-key"), "no-such-key");
});

test("applyPrior without a prior keeps the probabilities", () => {
  assert.deepEqual([...applyPrior([0.5, 0.3, 0.2], null)].map((p) => +p.toFixed(6)), [0.5, 0.3, 0.2]);
});

test("applyPrior re-ranks look-alikes by the prior and renormalises", () => {
  const out = applyPrior([0.5, 0.5], [0.9, 0.1]);
  assert.ok(Math.abs(out[0] + out[1] - 1) < 1e-6);
  assert.ok(out[0] > out[1]);
  const expected = 0.9 ** PRIOR_WEIGHT / (0.9 ** PRIOR_WEIGHT + 0.1 ** PRIOR_WEIGHT);
  assert.ok(Math.abs(out[0] - expected) < 1e-6);
});
