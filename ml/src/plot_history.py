"""Draw validation accuracy per epoch as an SVG chart for the README.

Plain SVG with no plotting library, and colours that follow GitHub's light or dark theme.

Usage (from the ml/ folder):
    python -m src.plot_history runs/<run>/history.csv --sessions 8 13 19 --out ../docs/training-curve.svg
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

W, H = 720, 360
LEFT, RIGHT, TOP, BOTTOM = 56, 120, 44, 48  # plot margins
Y_MIN, Y_MAX = 0.5, 0.95

STYLE = """
  .bg { fill: #fcfcfb; }
  .grid { stroke: #e4e3df; stroke-width: 1; }
  .axis { fill: #52514e; font: 12px -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
  .title { fill: #0b0b0b; font: 600 15px -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
  .label { fill: #0b0b0b; font: 600 12px -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
  .note { fill: #52514e; font: 11px -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
  .session { stroke: #b5b4ae; stroke-width: 1; stroke-dasharray: 3 4; }
  .s1 { stroke: #2a78d6; } .d1 { fill: #2a78d6; }
  .s2 { stroke: #eb6834; } .d2 { fill: #eb6834; }
  .ring { stroke: #fcfcfb; stroke-width: 2; }
  @media (prefers-color-scheme: dark) {
    .bg { fill: #1a1a19; }
    .grid { stroke: #2f2f2d; }
    .axis, .note { fill: #c3c2b7; }
    .title, .label { fill: #f0efec; }
    .session { stroke: #5a5955; }
    .s1 { stroke: #3987e5; } .d1 { fill: #3987e5; }
    .s2 { stroke: #d95926; } .d2 { fill: #d95926; }
    .ring { stroke: #1a1a19; }
  }
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("history", type=Path)
    parser.add_argument("--sessions", type=int, nargs="*", default=[],
                        help="epochs where a new Kaggle session started (marked with a dashed line)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with open(args.history, newline="") as f:
        rows = list(csv.DictReader(f))
    epochs = [int(r["epoch"]) for r in rows]
    series = [("top-5", [float(r["val_top5"]) for r in rows], 2),
              ("top-1", [float(r["val_top1"]) for r in rows], 1)]
    pw, ph = W - LEFT - RIGHT, H - TOP - BOTTOM

    def x(e: float) -> float:
        return LEFT + (e - 1) / (max(epochs) - 1) * pw

    def y(v: float) -> float:
        return TOP + (Y_MAX - v) / (Y_MAX - Y_MIN) * ph

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'role="img" aria-label="Validation accuracy per epoch">',
           f"<style>{STYLE}</style>",
           f'<rect class="bg" width="{W}" height="{H}" rx="12"/>',
           f'<text class="title" x="{LEFT}" y="26">Validation accuracy per epoch (300 species)</text>']
    for v in [0.5, 0.6, 0.7, 0.8, 0.9]:
        out.append(f'<line class="grid" x1="{LEFT}" x2="{LEFT + pw}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
        out.append(f'<text class="axis" x="{LEFT - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.0%}</text>')
    for e in [1, 5, 10, 15, 20, 25]:
        if e <= max(epochs):
            out.append(f'<text class="axis" x="{x(e):.1f}" y="{TOP + ph + 20}" text-anchor="middle">{e}</text>')
    out.append(f'<text class="axis" x="{LEFT + pw / 2}" y="{H - 10}" text-anchor="middle">epoch</text>')
    for n, e in enumerate(args.sessions, 2):
        out.append(f'<line class="session" x1="{x(e):.1f}" x2="{x(e):.1f}" y1="{TOP}" y2="{TOP + ph}"/>')
        out.append(f'<text class="note" x="{x(e) + 4:.1f}" y="{TOP + ph - 6}">session {n}</text>')
    for name, values, k in series:
        points = " ".join(f"{x(e):.1f},{y(v):.1f}" for e, v in zip(epochs, values))
        out.append(f'<polyline class="s{k}" fill="none" stroke-width="2" stroke-linejoin="round" '
                   f'points="{points}"/>')
        out.append(f'<text class="label" x="{x(epochs[-1]) + 10:.1f}" y="{y(values[-1]) + 4:.1f}">'
                   f'{name}</text>')
    # mark the best epoch (highest top-1), the checkpoint that was exported
    top1 = series[1][1]
    best = max(range(len(top1)), key=top1.__getitem__)
    bx, by = x(epochs[best]), y(top1[best])
    out.append(f'<circle class="d1 ring" cx="{bx:.1f}" cy="{by:.1f}" r="5"/>')
    out.append(f'<text class="note" x="{bx:.1f}" y="{by + 20:.1f}" text-anchor="middle">'
               f'best: {top1[best]:.1%} (epoch {epochs[best]})</text>')
    out.append("</svg>")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
