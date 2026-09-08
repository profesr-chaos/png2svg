# png2svg

Flat-colour PNG to SVG converter, with a small Tkinter front end.

## Install

    pip install -r requirements.txt
    scoop install potrace     # optional C binary: ~15x faster, used if present

## Use

    python png2svg.py input.png [output.svg] [--exact|--vtracer]
    python png2svg_app.py     # GUI

## Modes

| Mode | Flag | Result |
|---|---|---|
| `trace()` | default | One smooth vector layer per flat colour (potrace). Best fidelity. |
| `vtrace()` | `--vtracer` | ~25x faster, more paths, more colour noise. |
| `pixel_copy()` | `--exact` | One rectangle per run of equal pixels. Exact copy, no curves. |
