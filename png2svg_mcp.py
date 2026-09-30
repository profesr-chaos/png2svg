"""MCP server: lets an AI agent convert PNGs to SVG and compress SVGs.

    claude mcp add png2svg -- python C:\\path\\to\\png2svg_mcp.py

Stdio carries the protocol, so nothing in this process may print to stdout.
The library only prints in demo() and its __main__ block, and captures the
potrace binary's output.
"""
import functools
import io
import os
from typing import Literal

import anyio.from_thread
from mcp.server.mcpserver import Context, Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from PIL import Image as PILImage
from PIL import ImageChops

import png2svg

mcp = MCPServer("png2svg", instructions=(
    "Convert flat-colour PNGs (logos, icons, flat illustrations) to SVG, and "
    "shrink any SVG. Every path must be absolute. Both tools return sizes, "
    "error scores and one image with three panels: the input, the render of "
    "the output, and their difference. Look at it before you keep the file."))

MODES = {"trace": png2svg.trace, "vtrace": png2svg.vtrace, "exact": png2svg.pixel_copy}
PANEL_PX = 512      # longest side of one panel; three in a row fit the 1568 px a model keeps
GAP = 8
DIFF_GAIN = 4       # difference panel: white is an error of 64 or more


def _say_why(fn):
    """The SDK hides the text of any exception but ToolError. This server only
    runs locally for its owner, so pass the real reason to the model: a missing
    file or an uninstalled optional dependency is something it can act on."""
    @functools.wraps(fn)
    def run(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:
            raise ToolError("%s: %s" % (type(e).__name__, e)) from e
    return run


def _abs(p, name):
    p = os.path.expanduser(p)
    if not os.path.isabs(p):                    # the server's cwd means nothing to the caller
        raise ValueError("%s must be an absolute path, got %r" % (name, p))
    return p


def _on_white(path):
    im = PILImage.open(path).convert("RGBA")
    return PILImage.alpha_composite(PILImage.new("RGBA", im.size, "white"), im).convert("RGB")


def _panels(ref, render):
    """One image, left to right: the reference, the render, and their difference
    (black = equal, brighter = larger error). An agent that got the file by path
    never saw the reference, and two panels alone hide a lost thin line."""
    a, b = _on_white(ref), _on_white(render)    # compare() renders at the reference's size
    d = functools.reduce(ImageChops.lighter, ImageChops.difference(a, b).split())
    d = d.point(lambda v: min(255, v * DIFF_GAIN)).convert("RGB")
    k = PANEL_PX / max(a.size)
    w, h = (max(1, round(s * k)) for s in a.size)
    how = PILImage.NEAREST if k > 1 else PILImage.LANCZOS   # keep an icon's pixels sharp
    # ponytail: a downscale dims 1 px errors; MaxFilter on d before the resize if they vanish
    strip = PILImage.new("RGB", (3 * w + 2 * GAP, h), "gray")
    for i, im in enumerate((a, b, d)):
        strip.paste(im.resize((w, h), how), (i * (w + GAP), 0))
    buf = io.BytesIO()
    strip.save(buf, "PNG")
    return Image(data=buf.getvalue(), format="png")


def _scored(res, compare, ref, out, preview):
    """Add the error scores of `out` against `ref` to `res`, and the panels."""
    try:
        score = compare(ref, out)
    except ImportError:
        return [res, "install cairosvg to get the error scores and the preview"]
    ref, render = score.pop("ref"), score.pop("render")
    res.update({k: round(float(v), 3) for k, v in score.items()})
    return [res, _panels(ref, render)] if preview else res


@mcp.tool()
@_say_why
def png_to_svg(ctx: Context, png_path: str, svg_path: str = "",
               mode: Literal["trace", "vtrace", "exact"] = "trace", preview: bool = True):
    """Convert a PNG to SVG and score the SVG against the PNG.

    mode: "trace" (default) gives one smooth layer per flat colour, the best
    result, but a large image can take 30 s or more. "vtrace" is ~25x faster,
    with more paths and colour noise. "exact" copies every pixel as rectangles:
    no error, no curves, large files.

    svg_path defaults to the PNG's path with a .svg extension; it overwrites.
    Returns png_bytes and svg_bytes, and the per-pixel colour error on a 0-255
    scale: mean and max (lower is better), over40 (% of pixels off by more than
    40) and within8 (% within 8).
    preview=True also returns one image with three panels, left to right: the
    PNG, the render of the SVG, and their difference (black = equal, white = an
    error of 64 or more).
    """
    src = _abs(png_path, "png_path")
    out = _abs(svg_path, "svg_path") if svg_path else os.path.splitext(src)[0] + ".svg"
    step = 0

    def say(msg):                               # runs on the tool's worker thread
        nonlocal step
        step += 1
        anyio.from_thread.run(ctx.report_progress, step, None, msg)

    MODES[mode](src, out, say)
    res = {"svg_path": out, "png_bytes": os.path.getsize(src), "svg_bytes": os.path.getsize(out)}
    return _scored(res, png2svg.compare, src, out, preview)


@mcp.tool()
@_say_why
def compress_svg(svg_path: str, out_path: str = "", round_digits: int | None = None,
                 preview: bool = True):
    """Shrink any SVG (or .svgz) with scour.

    Lossless by default: it removes metadata, comments and whitespace, and
    keeps every coordinate, id, <title> and <desc>. round_digits=N also rounds
    numbers to N significant digits (4 is a good start) and drops ids, title
    and desc, a few points smaller again.

    out_path defaults to <name>.min.svg. An out_path that ends in .svgz writes
    gzipped bytes, ~60% smaller. Returns bytes_before, bytes_after, saved_pct
    and the render difference on a 0-255 scale: mean and max (0 means no
    visible change), over40 (% of pixels off by more than 40) and within8 (%
    within 8).
    preview=True also returns one image with three panels, left to right: the
    render before, the render after, and their difference (black = equal,
    white = an error of 64 or more).
    """
    src = _abs(svg_path, "svg_path")
    out = _abs(out_path, "out_path") if out_path else os.path.splitext(src)[0] + ".min.svg"
    before, after = png2svg.compress(src, out, round_digits, gz=out.lower().endswith(".svgz"))
    res = {"out_path": out, "bytes_before": before, "bytes_after": after,
           "saved_pct": round(100 * (1 - after / before), 1)}
    return _scored(res, png2svg.compare_svg, src, out, preview)


if __name__ == "__main__":
    mcp.run()
