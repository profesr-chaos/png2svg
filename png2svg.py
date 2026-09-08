"""Flat-colour PNG -> SVG converter.

Three modes:
  trace()      one smooth vector layer per flat colour (potrace). Best fidelity.
  vtrace()     VTracer (Rust, MIT): ~25x faster, more paths, more colour noise.
  pixel_copy() one rectangle per run of equal pixels: an exact copy, no curves.

    pip install potracer pillow numpy cairosvg vtracer
    scoop install potrace          # optional C binary: ~15x faster, used if present
    python png2svg.py input.png [output.svg] [--exact|--vtracer]

trace() runs one layer per core. The C potrace runs in threads, the Python port
in processes, so a caller needs the usual `if __name__ == "__main__"` guard.
"""
import concurrent.futures as cf
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
import potrace
from PIL import Image

DEFAULTS = dict(
    scale=4,             # trace grid: 4x gives quarter-pixel edge placement
    min_share=0.01,      # drop colours below this share of the artwork
    merge_dist=14,       # max-channel distance that still counts as one colour
    smooth_passes=2,     # 3x3 majority votes over the labels
    min_width=1.0,       # px: a band of a blended colour up to this wide is an
                         # anti-aliasing transition, not a shape
    erode=1,             # px of the outer rim that stays single-layer
    alphamax=1.0,        # potrace corner threshold (1.0 = smooth curves)
    opttolerance=0.6,
    turdsize=6,          # px^2: drop traced specks under this area
    straight_tol=0.1,    # px a node may move when a run collapses to one line
    backend="auto",      # "auto" | "potrace" (the C binary) | "python" (potracer)
    workers=0,           # 0 = one per core
    max_grid=20_000_000, # cap on pixels of the trace grid; scale drops to fit
)

NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))
CHUNK = 500_000


def _cfg(opts):
    cfg = dict(DEFAULTS)
    cfg.update({k: v for k, v in opts.items() if v is not None})
    exe = find_potrace()
    if cfg["backend"] == "auto":
        cfg["backend"] = "potrace" if exe else "python"
    if cfg["backend"] == "potrace" and not exe:
        raise RuntimeError("potrace is not on PATH; run: scoop install potrace")
    cfg["potrace_exe"] = exe
    return cfg


def palette(rgb, solid, cfg):
    """The flat colours, taken from the fully opaque pixels only."""
    from collections import Counter
    px = rgb[solid]
    pal = []
    for col, _ in Counter(map(tuple, px.tolist())).most_common():
        c = np.array(col)
        if all(np.abs(c - p).max() > cfg["merge_dist"] for p in pal):
            pal.append(c)
    pal = np.array(pal)
    lab = np.abs(px[:, None, :] - pal[None]).sum(2).argmin(1)
    keep = [i for i in range(len(pal)) if (lab == i).sum() / len(px) >= cfg["min_share"]]
    if not keep:
        raise ValueError("no colour covers min_share of the image; lower it")
    return pal[keep]


def label_all(rgb, solid, inside, pal):
    """Label the solid pixels, then flood the anti-aliased rim from its neighbours."""
    lab = np.full(solid.shape, -1, np.int16)
    px = rgb[solid].astype(np.int32)
    near = np.empty(len(px), np.int16)
    for i in range(0, len(px), CHUNK):          # chunked: the 4x grid is big
        c = px[i:i + CHUNK]
        near[i:i + CHUNK] = np.abs(c[:, None, :] - pal[None]).sum(2).argmin(1)
    lab[solid] = near
    lab[~inside] = -1
    return _flood(lab, inside)


def _flood(lab, inside):
    """Give every unlabelled pixel inside the shape the label of a neighbour."""
    while (todo := inside & (lab < 0)).any():
        grown = lab.copy()
        for dy, dx in NEIGHBOURS:
            src = np.roll(lab, (dy, dx), (0, 1))
            take = todo & (grown < 0) & (src >= 0)
            grown[take] = src[take]
        if (grown == lab).all():
            lab[todo] = 0                       # unreachable rim: park it somewhere
            break
        lab = grown
    return lab


def _runs(lab):
    """Per pixel: the shorter of its horizontal and vertical run of equal labels."""
    out = np.full(lab.shape, np.iinfo(np.int32).max, np.int32)
    for axis in (0, 1):
        a = lab if axis == 1 else lab.T
        cut = np.ones(a.shape, bool)
        cut[:, 1:] = a[:, 1:] != a[:, :-1]           # a run starts at every cut
        rid = np.cumsum(cut.ravel())                  # run id, unique across rows
        length = np.bincount(rid)[rid].reshape(a.shape)
        out = np.minimum(out, length if axis == 1 else length.T)
    return out


def thin(lab, rgb, inside, pal, width, tol):
    """Relabel the anti-aliasing bands that landed on a third colour.

    Two flat fills that meet get a one-pixel band of blended colour between
    them. When that blend matches a real palette colour, the band is labelled
    as one and traces as hundreds of slivers. A pixel is a band pixel if its
    run of equal labels is short in both directions AND its own colour sits on
    the line between the two colours that surround it. A real thin stroke
    fails the second test, so it survives.
    """
    w = int(round(width))
    cand = inside & (_runs(lab) <= w)
    if not cand.any():
        return lab
    surv = lab.copy()
    surv[cand] = -1
    ys, xs = np.nonzero(cand)
    h, wd = lab.shape
    n = len(pal)
    count = np.zeros((len(ys), n), np.int16)      # labels seen around each candidate
    for r in range(1, w + 2):
        for dy, dx in ((r, 0), (-r, 0), (0, r), (0, -r), (r, r), (r, -r), (-r, r), (-r, -r)):
            l = surv[np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, wd - 1)]
            ok = l >= 0
            count[np.nonzero(ok)[0], l[ok]] += 1
    top = np.argsort(-count, axis=1)[:, :2]
    a, b = top[:, 0], top[:, 1]
    has2 = (count[np.arange(len(ys)), b] > 0) & (a != b)
    p = rgb[ys, xs].astype(np.float32)
    pa, pb = pal[a].astype(np.float32), pal[b].astype(np.float32)
    v = pb - pa
    t = ((p - pa) * v).sum(1) / np.maximum((v * v).sum(1), 1e-6)
    resid = np.abs(p - (pa + t[:, None] * v)).max(1)
    blend = has2 & (t > 0.05) & (t < 0.95) & (resid <= tol)
    new = lab.copy()
    new[ys[blend], xs[blend]] = np.where(t[blend] < 0.5, a[blend], b[blend])
    return new


def smooth(lab, inside, n_colours, passes):
    """3x3 majority vote: kills compression speckle so the traced edges stay clean.

    Only a pixel with a mixed neighbourhood can change, and that is a few percent
    of them. Vote on those alone: the cost drops with the colour count, not with
    the pixel count.
    """
    for _ in range(passes):
        shifts = np.stack([np.roll(lab, (dy, dx), (0, 1))
                           for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
        mixed = (shifts != lab).any(0)
        vals = shifts[:, mixed].T               # (m, 9) neighbourhoods to settle
        if not len(vals):
            break
        best = np.zeros(len(vals), np.int8)
        pick = np.zeros(len(vals), np.int16)
        for c in range(n_colours):              # lowest index wins a tie, as before
            v = (vals == c).sum(1).astype(np.int8)
            take = v > best
            best[take] = v[take]
            pick[take] = c
        new = lab.copy()
        new[mixed] = pick
        new[~inside] = -1
        if (new == lab).all():
            break
        lab = new
    return lab


def to_path(mask, cfg):
    """Trace one layer. The C potrace runs ~50x faster than the Python port."""
    if cfg["backend"] == "potrace":
        return _trace_exe(mask, cfg)
    bmp = potrace.Bitmap(~mask)                 # potrace.Bitmap always inverts
    s = cfg["scale"]
    out = []
    f = lambda p: "%.2f %.2f" % (p.x / s, p.y / s)
    for curve in bmp.trace(turdsize=cfg["turdsize"] * s * s, alphamax=cfg["alphamax"],
                           opticurve=True, opttolerance=cfg["opttolerance"]):
        out.append("M" + f(curve.start_point))
        for seg in curve:
            if seg.is_corner:
                out.append("L%s L%s" % (f(seg.c), f(seg.end_point)))
            else:
                out.append("C%s %s %s" % (f(seg.c1), f(seg.c2), f(seg.end_point)))
        out.append("Z")
    return " ".join(out)


def find_potrace():
    """Path of the C potrace, or None. Install it with: scoop install potrace"""
    return shutil.which("potrace")


_ARGC = {"M": 2, "L": 2, "C": 6, "Z": 0}


def _trace_exe(mask, cfg):
    """Run the C potrace on one mask and return the path in our own units."""
    h, w = mask.shape
    pbm = b"P4\n%d %d\n" % (w, h) + np.packbits(mask, axis=1).tobytes()
    r = subprocess.run(
        [cfg["potrace_exe"], "-s", "-o", "-",
         "-a", str(cfg["alphamax"]), "-O", str(cfg["opttolerance"]),
         "-t", str(cfg["turdsize"] * cfg["scale"] * cfg["scale"])],
        input=pbm, capture_output=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode:
        raise RuntimeError("potrace failed: %s" % r.stderr.decode()[:200])
    return _from_potrace(r.stdout.decode(), h, cfg["scale"])


def _from_potrace(text, h_px, scale):
    """potrace writes 1/10 px units with the y axis flipped. Undo both."""
    out = []
    pt = lambda x, y: ((x / 10.0) / scale, (h_px - y / 10.0) / scale)
    f = lambda p: "%.2f %.2f" % p
    for d in re.findall(r'<path d="([^"]*)"', text, re.S):
        toks = re.findall(r"[A-Za-z]|-?\d+\.?\d*", d)
        cmd, cx, cy, i = "M", 0.0, 0.0, 0
        while i < len(toks):
            if toks[i].isalpha():
                cmd = toks[i]
                i += 1
                if cmd.upper() == "Z":
                    out.append("Z")
                    continue
            up = cmd.upper()
            rel = cmd.islower()
            n = _ARGC[up]
            vals = [float(v) for v in toks[i:i + n]]
            i += n
            abs_pts = []
            for k in range(0, n, 2):                # relative values chain from cx, cy
                x = vals[k] + (cx if rel else 0)
                y = vals[k + 1] + (cy if rel else 0)
                abs_pts.append((x, y))
            cx, cy = abs_pts[-1]
            if up == "M":
                out.append("M" + f(pt(*abs_pts[0])))
                cmd = "l" if rel else "L"           # a repeat after M means lineto
            elif up == "L":
                out.append("L" + f(pt(*abs_pts[0])))
            else:
                out.append("C%s %s %s" % tuple(f(pt(*p)) for p in abs_pts))
    return " ".join(out)


def _layer_job(packed, shape, cfg):
    mask = np.unpackbits(packed, axis=1, count=shape[1]).astype(bool)
    return to_path(mask, cfg)


def _trace_layers(masks, cfg, say):
    """Trace every layer at once. Threads suit the binary, processes the port."""
    n = len(masks)
    workers = cfg["workers"] or min(os.cpu_count() or 1, n)
    if workers < 2 or n < 2:
        return [to_path(m, cfg) for m in masks]
    if cfg["backend"] == "potrace":                 # the binary waits outside the GIL
        pool = cf.ThreadPoolExecutor(workers)
        jobs = {pool.submit(to_path, m, cfg): k for k, m in enumerate(masks)}
    else:                                           # the Python port needs processes
        pool = cf.ProcessPoolExecutor(workers)
        jobs = {pool.submit(_layer_job, np.packbits(m, axis=1), m.shape, cfg): k
                for k, m in enumerate(masks)}
    done, out = 0, [None] * n
    try:
        for fut in cf.as_completed(jobs):
            out[jobs[fut]] = fut.result()
            done += 1
            say("trace layer %d/%d" % (done, n))
    except cf.process.BrokenProcessPool:
        # a caller without an `if __name__ == "__main__"` guard cannot spawn workers
        say("no worker processes; run the layers one by one")
        return [to_path(m, cfg) for m in masks]
    finally:
        pool.shutdown()
    return out


# --- straighten: potrace fits curves, but flat artwork is mostly straight edges ---

def _parse(d):
    toks = re.findall(r"[MLCZ]|-?\d+\.?\d*", d)
    subs, cur, i = [], None, 0
    while i < len(toks):
        t = toks[i]
        if t == "M":
            cur = {"start": (float(toks[i + 1]), float(toks[i + 2])), "segs": []}
            subs.append(cur)
            i += 3
        elif t == "L":
            cur["segs"].append(("L", (float(toks[i + 1]), float(toks[i + 2]))))
            i += 3
        elif t == "C":
            cur["segs"].append(("C", *[(float(toks[i + 1 + 2 * k]),
                                        float(toks[i + 2 + 2 * k])) for k in range(3)]))
            i += 7
        else:
            i += 1
    return subs


def _dist(p, a, b):
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    n = (dx * dx + dy * dy) ** 0.5
    if n < 1e-9:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    return abs(dx * (ay - py) - dy * (ax - px)) / n


def straighten(d, tol=DEFAULTS["straight_tol"]):
    out = []
    for sub in _parse(d):
        p0, cur, flat = sub["start"], sub["start"], []
        for s in sub["segs"]:                   # a cubic hugging its chord is a line
            if s[0] == "C" and max(_dist(s[1], cur, s[3]), _dist(s[2], cur, s[3])) <= tol:
                flat.append(("L", s[3]))
            else:
                flat.append(s)
            cur = s[-1]
        merged, cur = [], p0
        for s in flat:                          # drop the joints inside a straight run
            if (s[0] == "L" and merged and merged[-1][0] == "L"
                    and _dist(merged[-1][1], cur, s[1]) <= tol):
                merged[-1] = ("L", s[1])
            else:
                if merged:
                    cur = merged[-1][-1]
                merged.append(s)
        f = lambda p: "%.2f %.2f" % p
        out.append("M" + f(p0) + " " + " ".join(
            "L" + f(s[1]) if s[0] == "L" else "C%s %s %s" % (f(s[1]), f(s[2]), f(s[3]))
            for s in merged) + " Z")
    return " ".join(out)


def _svg(w, h, body, extra=""):
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
            'width="%d" height="%d"%s>\n%s\n</svg>\n' % (w, h, w, h, extra, body))


def trace(src, out, progress=None, **opts):
    """One smooth vector layer per flat colour. Returns the colours it used."""
    cfg = _cfg(opts)
    say = progress or (lambda *_: None)
    say("read image")
    im = Image.open(src).convert("RGBA")
    w, h = im.size
    a1 = np.array(im)
    pal = palette(a1[:, :, :3].astype(int), a1[:, :, 3] >= 200, cfg)
    say("%d colours found" % len(pal))
    while cfg["scale"] > 1 and w * h * cfg["scale"] ** 2 > cfg["max_grid"]:
        cfg["scale"] -= 1                       # a big image cannot afford a 4x grid
        say("trace grid down to %dx for this size" % cfg["scale"])
    if cfg["scale"] > 1:
        im = im.resize((w * cfg["scale"], h * cfg["scale"]), Image.BILINEAR)
    a = np.array(im).astype(np.int16)
    alpha, rgb = a[:, :, 3], a[:, :, :3]
    solid, inside = alpha >= 200, alpha >= 128

    say("label pixels")
    lab = smooth(label_all(rgb, solid, inside, pal), inside, len(pal), cfg["smooth_passes"])
    lab = thin(lab, rgb, inside, pal, cfg["min_width"] * cfg["scale"], cfg["merge_dist"])
    lab = smooth(lab, inside, len(pal), cfg["smooth_passes"])

    core = lab >= 0                             # silhouette minus its outer rim
    for _ in range(cfg["erode"] * cfg["scale"]):
        e = core.copy()
        for dy, dx in NEIGHBOURS:
            e &= np.roll(core, (dy, dx), (0, 1))
        core = e

    order = sorted(range(len(pal)), key=lambda i: -(lab == i).sum())
    # Each layer sits on the larger ones, so a shared border cannot leak. It stops
    # short of the outer rim, where a second edge would double the alpha. Build the
    # union from the smallest layer up: one pass, not one scan per layer.
    masks, acc = [], np.zeros(core.shape, bool)
    for i in reversed(order):
        own = lab == i
        masks.append(own | (acc & core))
        acc |= own
    masks.reverse()

    say("trace %d layers on %s" % (len(masks), cfg["backend"]))
    parts = []
    for i, d in zip(order, _trace_layers(masks, cfg, say)):
        if d:
            parts.append('<path fill="#%02X%02X%02X" d="%s"/>'
                         % (*pal[i], straighten(d, cfg["straight_tol"])))

    open(out, "w").write(_svg(w, h, "\n".join(parts)))
    say("done")
    return ["#%02X%02X%02X" % tuple(pal[i]) for i in order]


# VTracer defaults, in the order its binding takes them. Order matters: see vtrace().
VTRACER = dict(colormode="color", hierarchical="stacked", mode="spline",
               filter_speckle=4, color_precision=6, layer_difference=16,
               corner_threshold=60, length_threshold=4.0, max_iterations=10,
               splice_threshold=45, path_precision=8)


def vtrace(src, out, progress=None, **opts):
    """Trace with VTracer. It keeps every colour, so it needs no palette step."""
    import vtracer
    say = progress or (lambda *_: None)
    cfg = dict(VTRACER)
    cfg.update({k: v for k, v in opts.items() if k in cfg and v is not None})
    say("trace with vtracer")
    # vtracer 0.6.15 segfaults on keyword arguments, so pass all 11 positionally.
    vtracer.convert_image_to_svg_py(str(src), str(out), *cfg.values())
    say("done")
    return cfg


def pixel_copy(src, out, progress=None):
    """One rectangle per run of equal pixels: exact, but it does not scale."""
    say = progress or (lambda *_: None)
    say("read image")
    a = np.array(Image.open(src).convert("RGBA"))
    h, w, _ = a.shape
    vis = a.copy()
    vis[a[:, :, 3] == 0] = 0                    # transparent pixels carry no colour

    runs = {}                                   # rgba -> list of (x, y, width)
    for y in range(h):
        if y % 256 == 0:
            say("scan row %d/%d" % (y, h))
        row = vis[y]
        cut = np.flatnonzero(np.any(row[1:] != row[:-1], 1))
        for s, e in zip(np.r_[0, cut + 1], np.r_[cut + 1, w]):
            if row[s, 3]:
                runs.setdefault(tuple(int(v) for v in row[s]), []).append((s, y, e - s))

    parts = []
    for (r, g, b, al), rects in runs.items():
        style = 'fill="#%02X%02X%02X"' % (r, g, b)
        if al != 255:
            style += ' fill-opacity="%.4f"' % (al / 255)
        parts.append("<g %s>%s</g>" % (style, "".join(
            '<rect x="%d" y="%d" width="%d" height="1"/>' % xyw for xyw in rects)))

    open(out, "w").write(_svg(w, h, "".join(parts), ' shape-rendering="crispEdges"'))
    say("done")
    return sum(len(v) for v in runs.values())


def compare(src, svg_path):
    """Render the SVG back and score it against the source. Needs cairosvg."""
    import cairosvg
    a = np.array(Image.open(src).convert("RGBA")).astype(float)
    h, w = a.shape[:2]
    png = os.path.join(tempfile.gettempdir(),
                       os.path.basename(svg_path) + ".check.png")
    cairosvg.svg2png(url=svg_path, write_to=png, output_width=w, output_height=h)
    b = np.array(Image.open(png).convert("RGBA")).astype(float)
    flat = lambda x: x[:, :, :3] * (x[:, :, 3:4] / 255.0) + 255 * (1 - x[:, :, 3:4] / 255.0)
    d = np.abs(flat(a) - flat(b)).max(2)
    return dict(mean=d.mean(), max=d.max(), over40=100 * (d > 40).mean(),
                within8=100 * (d <= 8).mean(), render=png)


def demo():
    assert straighten("M0 0 L5 0 L10 0 Z").count("L") == 1        # straight run collapses
    assert straighten("M0 0 L10 0 L10 10 L0 10 Z").count("L") == 3  # corners survive
    im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))                # red square on nothing
    for y in range(16, 48):
        for x in range(16, 48):
            im.putpixel((x, y), (200, 30, 30, 255))
    im.save("_demo.png")
    trace("_demo.png", "_demo.svg", scale=2, min_share=0.05)
    d = open("_demo.svg").read()
    assert d.count("<path") == 1 and "#C81E1E" in d, d[:200]
    n = pixel_copy("_demo.png", "_demo_exact.svg")
    assert n == 32, n                                             # 32 rows, one run each
    try:
        import vtracer                                            # noqa: F401
    except ImportError:
        print("vtracer not installed; skipped that mode")
    else:
        vtrace("_demo.png", "_demo_vt.svg", filter_speckle=2)
        assert "<path" in open("_demo_vt.svg").read()
    print("self-check ok")


if __name__ == "__main__":
    import sys
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    if not args:
        demo()
    else:
        src = args[0]
        out = args[1] if len(args) > 1 else re.sub(r"\.\w+$", "", src) + ".svg"
        if "--exact" in sys.argv:
            print("runs:", pixel_copy(src, out, print))
        elif "--vtracer" in sys.argv:
            print("vtracer:", vtrace(src, out, print))
        else:
            print("colours:", trace(src, out, print))
        try:
            r = compare(src, out)
            print("mean %.4f/255  max %.2f  off by >40: %.3f%%"
                  % (r["mean"], r["max"], r["over40"]))
        except ImportError:
            print("install cairosvg to score the result")
