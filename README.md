# png2svg

Flat-colour PNG to SVG converter, with a small Tkinter front end.

## Install

    pip install -r requirements.txt
    scoop install potrace     # optional C binary: ~15x faster, used if present

## Use

    python png2svg.py input.png [output.svg] [--exact|--vtracer]
    python png2svg.py input.svg output.svg|output.svgz --compress [--round[=N]]
    python png2svg_app.py     # GUI

## Modes

| Mode | Flag | Result |
|---|---|---|
| `trace()` | default | One smooth vector layer per flat colour (potrace). Best fidelity. |
| `vtrace()` | `--vtracer` | ~25x faster, more paths, more colour noise. |
| `pixel_copy()` | `--exact` | One rectangle per run of equal pixels. Exact copy, no curves. |
| `compress()` | `--compress` | Lossless: scour removes metadata, comments and whitespace but keeps every coordinate digit, id and title. 2-34% smaller, ~60% as `.svgz`. |
| | `--compress --round[=N]` | Also rounds numbers to N significant digits (default 4) and drops ids, title and desc. A few points smaller again. |

## Benchmark

    python bench.py           # score bench/*.png, diff against bench/baseline.json
    python bench.py --save    # accept the new numbers as the baseline

Scores `trace` and `vtrace` against each source PNG, plus an `rt` (round-trip)
mode that re-traces the `trace` render instead of the source, isolating error
the tracer itself adds. Every mode also gets `err_kb` (mean error times file
size, in KB) as a single size-aware score — lower is better.

A change to `png2svg.py` must move this table, or it does not go in.
