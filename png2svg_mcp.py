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

import png2svg

mcp = MCPServer("png2svg", instructions=(
    "Convert flat-colour PNGs (logos, icons, flat illustrations) to SVG, and "
    "shrink any SVG. Every path must be absolute. png_to_svg returns an error "
    "score and a render of the SVG: look at the render before you keep the file."))

MODES = {"trace": png2svg.trace, "vtrace": png2svg.vtrace, "exact": png2svg.pixel_copy}
PREVIEW_PX = 800    # longest side of the preview; more pixels cost tokens, not insight


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
    Scores are per-pixel colour error on a 0-255 scale: mean and max (lower is
    better), over40 (% of pixels off by more than 40) and within8 (% within 8).
    preview=True also returns a PNG render of the SVG.
    """
    src = _abs(png_path, "png_path")
    out = _abs(svg_path, "svg_path") if svg_path else os.path.splitext(src)[0] + ".svg"
    step = 0

    def say(msg):                               # runs on the tool's worker thread
        nonlocal step
        step += 1
        anyio.from_thread.run(ctx.report_progress, step, None, msg)

    MODES[mode](src, out, say)
    res = {"svg_path": out, "bytes": os.path.getsize(out)}
    try:
        score = png2svg.compare(src, out)
    except ImportError:
        return [res, "install cairosvg to get an error score and a preview"]
    render = score.pop("render")
    res.update({k: round(float(v), 3) for k, v in score.items()})
    if not preview:
        return res
    im = PILImage.open(render)
    im.thumbnail((PREVIEW_PX, PREVIEW_PX))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return [res, Image(data=buf.getvalue(), format="png")]


@mcp.tool()
@_say_why
def compress_svg(svg_path: str, out_path: str = "", round_digits: int | None = None):
    """Shrink any SVG (or .svgz) with scour.

    Lossless by default: it removes metadata, comments and whitespace, and
    keeps every coordinate, id, <title> and <desc>. round_digits=N also rounds
    numbers to N significant digits (4 is a good start) and drops ids, title
    and desc, a few points smaller again.

    out_path defaults to <name>.min.svg. An out_path that ends in .svgz writes
    gzipped bytes, ~60% smaller. mean_error is the render difference on a 0-255
    scale: 0 means no visible change.
    """
    src = _abs(svg_path, "svg_path")
    out = _abs(out_path, "out_path") if out_path else os.path.splitext(src)[0] + ".min.svg"
    before, after = png2svg.compress(src, out, round_digits, gz=out.lower().endswith(".svgz"))
    res = {"out_path": out, "bytes_before": before, "bytes_after": after,
           "saved_pct": round(100 * (1 - after / before), 1)}
    try:
        res["mean_error"] = round(float(png2svg.compare_svg(src, out)["mean"]), 4)
    except ImportError:
        pass                                    # ponytail: no cairosvg, no score
    return res


if __name__ == "__main__":
    mcp.run()
