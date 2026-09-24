# png2svg

Flat-colour PNG to SVG converter, with a small Tkinter front end.

## Install

    pip install -r requirements.txt
    scoop install potrace     # optional C binary: ~15x faster, used if present

## Use

    python png2svg.py input.png [output.svg] [--exact|--vtracer]
    python png2svg.py input.svg output.svg|output.svgz --compress [--round[=N]]
    python png2svg_app.py     # GUI
    python png2svg_mcp.py     # MCP server for an AI agent (see below)

## Modes

| Mode | Flag | Result |
|---|---|---|
| `trace()` | default | One smooth vector layer per flat colour (potrace). Best fidelity. |
| `vtrace()` | `--vtracer` | ~25x faster, more paths, more colour noise. |
| `pixel_copy()` | `--exact` | One rectangle per run of equal pixels. Exact copy, no curves. |
| `compress()` | `--compress` | Lossless: scour removes metadata, comments and whitespace but keeps every coordinate digit, id and title. 2-34% smaller, ~60% as `.svgz`. |
| | `--compress --round[=N]` | Also rounds numbers to N significant digits (default 4) and drops ids, title and desc. A few points smaller again. |

## MCP server

`png2svg_mcp.py` lets an AI agent (Claude Code, Claude Desktop) use the app
directly. It has two tools:

| Tool | Does |
|---|---|
| `png_to_svg(png_path, svg_path?, mode?, preview?)` | Runs `trace`, `vtrace` or `exact`, and returns the SVG path, its size, the error scores and an 800 px render, so the agent can look at the result. |
| `compress_svg(svg_path, out_path?, round_digits?)` | Runs `compress()` on any SVG. Lossless unless you give `round_digits`; an `out_path` that ends in `.svgz` writes gzip. Returns the bytes saved and the render difference. |

Register it with Claude Code:

    pip install "mcp>=2"
    claude mcp add png2svg -- python C:/path/to/png2svg/png2svg_mcp.py

For Claude Desktop, add it to `claude_desktop_config.json` next to any other
server. Do not put it behind the Docker MCP gateway: a container cannot see
your Windows paths.

    "mcpServers": {
      "png2svg": {"command": "python", "args": ["C:/path/to/png2svg/png2svg_mcp.py"]}
    }

If `python` on the PATH is not the one with `mcp` installed, give its full path.

The tools read and write files, so give the agent absolute paths. If you paste
an image into the chat, save it to disk first. A `trace` of a large image takes
about 30 s; the server sends progress while it works.

    python test_mcp.py        # start the server over stdio and call both tools

## Benchmark

    python bench.py           # score bench/*.png, diff against bench/baseline.json
    python bench.py --save    # accept the new numbers as the baseline

Scores `trace` and `vtrace` against each source PNG, plus an `rt` (round-trip)
mode that re-traces the `trace` render instead of the source, isolating error
the tracer itself adds. Every mode also gets `err_kb` (mean error times file
size, in KB) as a single size-aware score — lower is better.

A change to `png2svg.py` must move this table, or it does not go in.
