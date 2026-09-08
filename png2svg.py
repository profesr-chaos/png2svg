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
    min_share=0.001,     # drop colours below this share of the artwork
    merge_dist=10,       # Lab distance (delta E) that still counts as one colour
    overlap=0.5,         # share of a colour's pixels that sit as close to another
                         # colour, above which the two are one textured colour
    smooth_passes=1,     # majority votes over the labels
    smooth_radius=1.0,   # px: half-width of the vote window; a label that flips
                         # along a soft gradient settles to the local majority
    max_island=60,       # px^2: a patch of one colour up to this size, fully
                         # inside a near colour, is shading noise and takes it
    min_width=2.0,       # px: a band of a blended colour up to this wide is an
                         # edge transition, not a shape (soft edges need more)
    edge_share=0.5,      # a colour whose pixels blend more often than this is a
                         # transition, not a fill, and goes
    band_tol=20,         # delta E a band pixel may sit off the line between the
                         # two colours around it and still count as their blend
    ink_min=0.35,        # px of colour a lost stroke must hold to be put back;
                         # 0 turns the coverage recovery off
    ink_tol=0.25,        # coverage under this is a soft edge, not held ink
    ink_share=0.2,       # only colours below this share of the art can be ink
    ink_de=80,           # delta E a recovered colour must stand off the colour
                         # it is recovered from: shading between near colours
                         # holds coverage too, but it is not a lost stroke
    erode=1,             # px of the outer rim that stays single-layer
    alphamax=0.5,        # potrace corner threshold: 1.0 rounds every corner,
                         # 0.5 keeps the corners of flat artwork sharp
    opttolerance=0.6,
    turdsize=6,          # px^2: drop traced specks under this area
    straight_tol=0.1,    # px a node may move when a run collapses to one line
    backend="auto",      # "auto" | "potrace" (the C binary) | "python" (potracer)
    workers=0,           # 0 = one per core
    max_grid=20_000_000, # cap on pixels of the trace grid; scale drops to fit
    grad_gain=0.3,       # min drop in a region's mean max-channel abs error a
                         # linear gradient must buy over its flat fill to be used;
                         # None or 0 disables gradients
    repair_passes=1,     # rounds of residual repair after the layers are traced;
                         # 0 disables (and skips the render it needs)
    repair_thr=16,       # max-channel error at which a pixel joins a repair cluster
    repair_min=16,       # px^2: smallest error cluster worth a patch of its own
)

NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))
EIGHT = np.ones((3, 3), bool)
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


_M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
_WHITE = _M.sum(1)


def to_lab(rgb):
    """sRGB 0-255 -> CIE Lab. A unit step here matches what the eye calls one."""
    c = np.asarray(rgb, np.float32) / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ _M.T.astype(np.float32) / _WHITE.astype(np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def nearest(lab_px, lab_pal, margin=None, rgb_px=None, rgb_pal=None):
    """Index of the nearest palette colour for each pixel, in Lab.

    With `margin`, also return which pixels are sure: the runner-up colour is
    more than `margin` delta E further away. Everything else sits between two
    colours and may take the local majority later.

    With `rgb_px`/`rgb_pal`, the choice between the two nearest colours A and B
    is made at 50% coverage instead of at equal Lab distance. Anti-aliasing
    mixes A and B in sRGB, so the pixel that is half A sits halfway along the
    line A->B there -- but Lab is not linear in coverage, and its midpoint for
    black/white sits at about 55% grey. Labelling by Lab alone therefore eats a
    fraction of a pixel off every dark shape. Project the pixel onto A->B in
    sRGB and split at t = 0.5. Which two colours are in play, and the sure
    mask, are unchanged.
    """
    out = np.empty(len(lab_px), np.int16)
    sure = np.empty(len(lab_px), bool)
    for i in range(0, len(lab_px), CHUNK):      # chunked: the 4x grid is big
        d = lab_px[i:i + CHUNK, None, :] - lab_pal[None]
        d = np.sqrt((d * d).sum(2))
        out[i:i + CHUNK] = d.argmin(1)
        if d.shape[1] > 1 and (margin is not None or rgb_px is not None):
            two = np.argpartition(d, 1, axis=1)[:, :2]
            r = np.arange(len(d))
            a, b = two[:, 0], two[:, 1]
            flip = d[r, a] > d[r, b]             # argpartition does not order the two
            a, b = np.where(flip, b, a), np.where(flip, a, b)
            if margin is not None:
                sure[i:i + CHUNK] = d[r, b] - d[r, a] > margin
            if rgb_px is not None:
                pa, pb = rgb_pal[a], rgb_pal[b]
                v = pb - pa
                t = ((rgb_px[i:i + CHUNK] - pa) * v).sum(1) / np.maximum((v * v).sum(1), 1e-6)
                out[i:i + CHUNK] = np.where(t.clip(0, 1) < 0.5, a, b)
    return (out, sure) if margin is not None else out


def palette(rgb, solid, cfg):
    """The flat colours, taken from the fully opaque pixels only."""
    from collections import Counter
    px = rgb[solid]
    counts = Counter(map(tuple, px.tolist())).most_common()
    cols = np.array([c for c, _ in counts])
    n = np.array([k for _, k in counts])
    lab = to_lab(cols)
    pal = []
    for i in range(len(cols)):                  # greedy by frequency, merge in Lab
        if not pal or ((lab[i] - lab[pal]) ** 2).sum(1).min() > cfg["merge_dist"] ** 2:
            pal.append(i)
    pal = _merge_overlap(cols, n, lab, pal, 2 * cfg["merge_dist"], cfg["overlap"])
    near = nearest(lab, to_lab(pal))
    share = np.bincount(near, n, len(pal)) / n.sum()
    keep = share >= cfg["min_share"]
    if not keep.any():
        raise ValueError("no colour covers min_share of the image; lower it")
    return pal[keep]


def _merge_overlap(cols, n, lab, pal, max_de, ratio, margin=3.0):
    """Join two palette entries that split one textured colour.

    A grainy fill lands on two entries a little over merge_dist apart, and its
    pixels sit about as close to one as to the other. Count, per pair, the
    pixels within `margin` delta E of both. Two flat fills share none; one
    texture shares most. Above `ratio` of the smaller entry, the pair becomes
    one entry at the pixel-weighted mean colour. Repeat until no pair is left.
    """
    pal = cols[pal].astype(float)
    while len(pal) > 1:
        lp = to_lab(pal)
        d = np.sqrt(((lab[:, None] - lp[None]) ** 2).sum(2))
        o = np.argsort(d, axis=1)[:, :2]
        rows = np.arange(len(d))
        close = d[rows, o[:, 1]] - d[rows, o[:, 0]] < margin
        m = len(pal)
        total = np.bincount(o[:, 0], n, m)
        both = np.bincount((o[:, 0] * m + o[:, 1])[close], n[close], m * m).reshape(m, m)
        both = both + both.T
        score = both / np.maximum(np.minimum(total[:, None], total[None]), 1)
        score[np.sqrt(((lp[:, None] - lp[None]) ** 2).sum(2)) > max_de] = 0
        np.fill_diagonal(score, 0)
        c, e = np.unravel_index(score.argmax(), score.shape)
        if score[c, e] < ratio:
            break
        mean = (pal[c] * total[c] + pal[e] * total[e]) / max(total[c] + total[e], 1)
        pal = np.vstack([np.delete(pal, [c, e], 0), mean])
    return pal.round().astype(int)


def label_all(rgb, solid, inside, pal, margin):
    """Label the solid pixels, then flood the anti-aliased rim from its neighbours.

    Also returns the sure mask: pixels clearly of one colour, which no vote
    may change. That keeps one-pixel text and lines through a wide vote.
    """
    lab = np.full(solid.shape, -1, np.int16)
    sure = np.zeros(solid.shape, bool)
    px = rgb[solid].astype(np.float32)
    lab[solid], sure[solid] = nearest(to_lab(px), to_lab(pal), margin,
                                      px, np.asarray(pal, np.float32))
    lab[~inside] = -1
    return _flood(lab, inside), sure


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


def _rows(t, lb, c, thr, need, wide):
    """One axis of ink(): mark the densest pixels of every run that holds ink.

    A run is a maximal stretch of one row with coverage over `thr`, bounded by
    coverage at or under it. It counts only if it is at most `wide` across --
    a row crossing a stroke, not a row running along one or down a gradient --
    and if the pixels on both sides of it carry the same colour, and not c
    already. A stroke crosses one fill and comes out on the same fill. The
    anti-aliased edge between two fills also holds coverage of a third colour,
    but it has a different colour on each side, and it is not a lost stroke:
    recovering it lays a hairline along every contour in the artwork.
    """
    h, w = t.shape
    z = np.zeros((h, 1), t.dtype)
    tp = np.hstack([z, t, z]).ravel()             # pad: no run crosses a row end
    lp = np.hstack([z - 1, lb, z - 1]).astype(np.int16).ravel()
    d = np.diff((tp > thr).astype(np.int8))
    beg, end = np.flatnonzero(d > 0) + 1, np.flatnonzero(d < 0) + 1
    hit = np.zeros(len(tp), bool)
    if not len(beg):
        return hit.reshape(h, w + 2)[:, 1:-1]
    tot = np.add.reduceat(tp, np.stack([beg, end], 1).ravel())[::2]
    take = ((tot >= need) & (end - beg <= wide)
            & (lp[beg - 1] == lp[end]) & (lp[beg - 1] != c))
    beg, end, tot = beg[take], end[take], tot[take]
    if not len(beg):
        return hit.reshape(h, w + 2)[:, 1:-1]
    n = end - beg
    idx = np.repeat(beg - np.concatenate([[0], np.cumsum(n)])[:-1], n) + np.arange(n.sum())
    mid = np.add.reduceat(tp[idx] * idx, np.concatenate([[0], np.cumsum(n)])[:-1]) / tot
    k = np.clip(np.rint(tot).astype(int), 1, n)   # how many pixels of ink to place
    beg = np.clip(np.rint(mid - (k - 1) / 2).astype(int), beg, end - k)
    hit[np.repeat(beg - np.concatenate([[0], np.cumsum(k)])[:-1], k) + np.arange(k.sum())] = True
    return hit.reshape(h, w + 2)[:, 1:-1]


def ink(lab, rgb, inside, pal, sure, s, thr=0.25, need=0.35, share=0.2, wide=2.0,
        min_de=80):
    """Put back the strokes and text that are thinner than a source pixel.

    A 0.6 px stem that straddles a pixel boundary leaves two pixels at 30% ink
    each. No threshold on colour calls either one ink, so the stroke vanishes.
    Coverage, though, is conserved: for each minority colour c, measure how
    much of it a pixel holds (t: where the pixel sits on the line from its own
    colour to c, in sRGB), then scan rows and columns for a run of held ink
    that never reaches the label. A run holding `need` of a source pixel of c
    is a lost stroke: give its round(sum t) pixels around the t-weighted
    centroid to c, and freeze them so no later vote takes them back.

    Only a colour that stands `min_de` off the one that took its place counts.
    Shading between two near colours holds coverage in exactly the same way,
    and putting that back sprinkles a soft gradient with specks.
    """
    P = np.asarray(pal, np.float32)
    lp = to_lab(P)
    de = np.sqrt(((lp[:, None] - lp[None]) ** 2).sum(2))
    px, ins = rgb.reshape(-1, 3), inside.ravel()
    cnt = np.bincount(np.maximum(lab[inside], 0), minlength=len(P))
    lab, sure = lab.copy(), sure.copy()
    for c in np.nonzero(cnt < share * cnt.sum())[0]:
        is_c = lab == c
        labi = np.maximum(lab, 0).ravel()           # what each pixel became
        j = np.flatnonzero(ins & ~is_c.ravel() & (de[c][labi] >= min_de))
        u = np.empty(len(j), np.float32)
        for i in range(0, len(j), CHUNK):           # chunked: the 4x grid is big
            k = j[i:i + CHUNK]
            a = P[labi[k]]                          # the colour that took c's place
            v = P[c] - a
            u[i:i + CHUNK] = ((px[k] - a) * v).sum(1) / np.maximum((v * v).sum(1), 1e-6)
        if not (u > thr).any():
            continue
        t = np.zeros(lab.size, np.float32)
        t[j] = u.clip(0, 1)
        t = t.reshape(lab.shape)
        hot = t > thr
        r, k = np.flatnonzero(hot.any(1)), np.flatnonzero(hot.any(0))
        hit = np.zeros(lab.shape, bool)             # only the rows and columns
        hit[r] = _rows(t[r], lab[r], c, thr, need * s, wide * s)  # that hold any ink
        hit[:, k] |= _rows(t[:, k].T, lab[:, k].T, c, thr, need * s, wide * s).T
        lab[hit], sure[hit] = c, True
    return lab, sure


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


def thin(lab, rgb, inside, pal, width, tol, edge_share):
    """Relabel the edge bands that landed on a third colour.

    Two fills that meet get a band of blended colour between them, one pixel
    for anti-aliasing, several for a soft edge. When that blend matches a
    palette colour, the band is labelled as one and traces as slivers or a
    halo. A pixel is a band pixel if its run of equal labels is short in both
    directions AND its own colour sits on the line (in Lab) between the two
    colours that surround it. A real thin stroke fails the second test, so it
    survives. A colour whose pixels are mostly band pixels only exists on
    edges: it is a transition, not a fill, and all its pixels go.
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
    for r in range(1, w + 2, max(1, w // 3)):    # a few rings out to the band edge
        for dy, dx in ((r, 0), (-r, 0), (0, r), (0, -r), (r, r), (r, -r), (-r, r), (-r, -r)):
            l = surv[np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, wd - 1)]
            ok = l >= 0
            count[np.nonzero(ok)[0], l[ok]] += 1
    top = np.argsort(-count, axis=1)[:, :2]
    a, b = top[:, 0], top[:, 1]
    has2 = (count[np.arange(len(ys)), b] > 0) & (a != b)
    p = to_lab(rgb[ys, xs])
    lab_pal = to_lab(pal)
    pa, pb = lab_pal[a], lab_pal[b]
    v = pb - pa
    t = ((p - pa) * v).sum(1) / np.maximum((v * v).sum(1), 1e-6)
    resid = np.sqrt(((p - (pa + t[:, None] * v)) ** 2).sum(1))
    blend = has2 & (t > 0.05) & (t < 0.95) & (resid <= tol)
    new = lab.copy()
    new[ys[blend], xs[blend]] = np.where(t[blend] < 0.5, a[blend], b[blend])
    total = np.bincount(lab[inside], minlength=n)
    blended = np.bincount(lab[ys[blend], xs[blend]], minlength=n)
    gone = blended > edge_share * np.maximum(total, 1)   # colours that only ever blend
    if gone.any() and not gone.all():
        keep = np.nonzero(~gone)[0]
        m = inside & gone[np.maximum(new, 0)] & (new >= 0)
        new[m] = keep[nearest(to_lab(rgb[m]), lab_pal[keep])]   # by colour, not position
    return new


def smooth(lab, inside, n_colours, radius, passes, sure=None):
    """Majority vote in a (2r+1)^2 window: settles labels that flip along a
    soft gradient, and kills compression speckle. One box filter per colour
    (PIL, C speed) counts the votes; the highest count wins, lowest index on
    a tie. A sure pixel votes but never changes.
    """
    from PIL import ImageFilter
    r = max(1, int(round(radius)))
    for _ in range(passes):
        best = np.zeros(lab.shape, np.uint8)
        pick = lab.copy()
        for c in range(n_colours):
            m = Image.fromarray((lab == c).astype(np.uint8) * 255)
            cnt = np.asarray(m.filter(ImageFilter.BoxBlur(r)))
            take = cnt > best
            best[take] = cnt[take]
            pick[take] = c
        pick[~inside] = -1
        if sure is not None:
            pick[sure] = lab[sure]
        if (pick == lab).all():
            break
        lab = pick
    return lab


def islands(lab, pal, inside, max_area, max_de, purity=0.9):
    """Give a small patch, fully inside one near colour, that colour.

    Soft shading pushes part of a fill past the palette midpoint, so a shaded
    stripe grows a patch of the outline brown. The patch is small, one colour
    surrounds it, and the two colours are near. Such a patch is noise. A real
    detail on a fill, an eye highlight on orange, sits far from its neighbour
    in colour and stays. Needs scipy; without it this step is skipped.
    """
    try:
        from scipy import ndimage
    except ImportError:
        return lab
    n = len(pal)
    lp = to_lab(pal)
    de = np.sqrt(((lp[:, None] - lp[None]) ** 2).sum(2))
    out = lab.copy()
    for c in range(n):
        comp, k = ndimage.label(lab == c)
        if k == 0:
            continue
        size = np.bincount(comp.ravel(), minlength=k + 1)
        small = size <= max_area
        small[0] = False
        if not small.any():
            continue
        ring = np.zeros((k + 1) * n)              # neighbour labels per patch
        for a, b in ((comp[:, :-1], lab[:, 1:]), (comp[:, 1:], lab[:, :-1]),
                     (comp[:-1], lab[1:]), (comp[1:], lab[:-1])):
            m = (a > 0) & (b != c) & (b >= 0) & small[a]
            ring += np.bincount(a[m] * n + b[m], minlength=(k + 1) * n)
        ring = ring.reshape(k + 1, n)
        tot = ring.sum(1)
        e = ring.argmax(1)
        take = small & (tot > 0) & (ring[np.arange(k + 1), e] >= purity * tot) & (de[c, e] <= max_de)
        if take.any():
            m = take[comp]
            out[m] = e[comp[m]]
    return out


_LUM = np.array([0.2126, 0.7152, 0.0722])


def _gradients(lab, src_rgb, pal, w, h, s, gain):
    """Fit colour = c0 + cx*x/w + cy*y/h per channel over each region's source
    pixels, and keep the plane where it beats the flat fill by more than `gain`
    (mean max-channel abs error) on at least 64 pixels.

    `src_rgb` must already be the source blended over white (compare() does the
    same before scoring): a soft alpha shading reads as an RGB gradient only
    once it is composited, and a flat opaque fill is what a path actually renders.

    Returns the <linearGradient> defs and, per accepted colour index, the
    gradient id and the fitted region's mean colour (for the returned palette).
    """
    lab_s = lab[::s, ::s]                        # back to source resolution
    defs, fills = [], {}
    for i in range(len(pal)):
        own = lab_s == i
        core = own.copy()
        for dy, dx in NEIGHBOURS:                # drop pixels within 1px of a boundary
            core &= np.roll(own, (dy, dx), (0, 1))
        # a small, mostly-boundary region (a thin stroke, a tiny icon glyph) is
        # dominated by source antialiasing right where a plane is judged, so its
        # measured gain is unreliable; give it 4x the floor to average that out
        if core.sum() < 64 or own.sum() < 256:
            continue
        ys, xs = np.nonzero(core)
        X = np.stack([np.ones(len(xs)), xs / w, ys / h], 1)
        Y = src_rgb[core].astype(float)
        # a region with little spread on one axis makes x/w, y/h collinear; a loose
        # rcond lets lstsq answer with huge, near-cancelling (and so unstable to
        # extrapolate) coefficients, so drop that ill-conditioned direction instead
        coef, *_ = np.linalg.lstsq(X, Y, rcond=1e-2)   # rows: c0, cx, cy; cols: R,G,B
        # judge the gain over the *whole* region, not just its fitted interior: a
        # thin or intricate shape is mostly boundary, and a plane that only reads
        # well on its calm interior can still lose badly once it covers the rest
        oys, oxs = np.nonzero(own)
        oX = np.stack([np.ones(len(oxs)), oxs / w, oys / h], 1)
        oY = src_rgb[own].astype(float)
        ofit = oX @ coef
        err_flat = np.abs(oY - pal[i]).max(1).mean()
        err_fit = np.abs(oY - ofit).max(1).mean()
        if err_flat - err_fit <= gain:
            continue
        gx, gy = coef[1:] @ _LUM                 # luminance-weighted (cx, cy)
        n = (gx * gx + gy * gy) ** 0.5
        if n < 1e-6:
            continue
        dx, dy = gx / n, gy / n
        # bounding box of the *fitted* pixels only: a stop beyond what the plane was
        # fit on would extrapolate, and a thin or curved region can extrapolate wildly
        bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max()
        ccx, ccy = (bx0 + bx1) / 2, (by0 + by1) / 2   # box centre: the line's fixed point
        r = max(abs((cx_ - ccx) * dx + (cy_ - ccy) * dy)
                for cx_, cy_ in ((bx0, by0), (bx1, by0), (bx0, by1), (bx1, by1)))
        p1, p2 = (ccx - r * dx, ccy - r * dy), (ccx + r * dx, ccy + r * dy)
        # a non-convex/thin region's bbox corners can still fall outside its own
        # pixels (an L-shape, a diagonal stroke), so the "fitted" corner above can
        # still extrapolate; clamp each stop to the colour the region actually has
        lo, hi = Y.min(0), Y.max(0)
        ends = [np.clip(coef[0] + coef[1] * (p[0] / w) + coef[2] * (p[1] / h), lo, hi)
                .round().astype(int) for p in (p1, p2)]
        gid = "g%d" % len(defs)
        defs.append('<linearGradient id="%s" gradientUnits="userSpaceOnUse" '
                    'x1="%.0f" y1="%.0f" x2="%.0f" y2="%.0f">'   # a px of drift is invisible
                    '<stop offset="0" stop-color="#%02X%02X%02X"/>'
                    '<stop offset="1" stop-color="#%02X%02X%02X"/></linearGradient>'
                    % (gid, p1[0], p1[1], p2[0], p2[1], *ends[0], *ends[1]))
        fills[i] = (gid, tuple(oY.mean(0).round().astype(int)))
    return defs, fills


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
        merged, cur, dropped = [], p0, []
        for s in flat:                          # drop the joints inside a straight run
            if (s[0] == "L" and merged and merged[-1][0] == "L"
                    and all(_dist(q, cur, s[1]) <= tol for q in dropped + [merged[-1][1]])):
                dropped.append(merged[-1][1])   # every dropped joint must stay near the chord
                merged[-1] = ("L", s[1])
            else:
                if merged:
                    cur = merged[-1][-1]
                merged.append(s)
                dropped = []
        f = lambda p: " ".join(("%.1f" % v).rstrip("0").rstrip(".") for v in p)
        out.append("M" + f(p0) + " " + " ".join(
            "L" + f(s[1]) if s[0] == "L" else "C%s %s %s" % (f(s[1]), f(s[2]), f(s[3]))
            for s in merged) + " Z")
    return " ".join(out)


def _offset(d, dx, dy):
    """Shift a path traced from a crop back to its place in the image.

    Every command we emit (M, L, C) takes whole x,y pairs and Z takes none, so
    the numbers alternate x, y all the way through the string.
    """
    i = [0]

    def move(m):
        v = float(m.group()) + (dx if i[0] % 2 == 0 else dy)
        i[0] += 1
        return ("%.1f" % v).rstrip("0").rstrip(".")
    return re.sub(r"-?\d+\.?\d*", move, d)


def _svg(w, h, body, extra=""):
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
            'width="%d" height="%d"%s>\n%s\n</svg>\n' % (w, h, w, h, extra, body))


def _flat(x):
    """RGBA -> RGB over white: what the pixel actually looks like on a page."""
    a = x[:, :, 3:4] / 255.0
    return x[:, :, :3] * a + 255 * (1 - a)


def repair(src_rgba, parts, defs, pal, w, h, cfg, say):
    """Patch what the layers got wrong: trace the largest error clusters.

    The palette drops a colour that covers little of the artwork, and a vote
    or an island merge moves a small shape into its neighbour; whatever the
    cause, the result is a compact blob where the render is plainly the wrong
    colour. Render the SVG as it stands, cluster the pixels that are furthest
    off, and lay one flat path per blob on top of everything else.

    Paths are not free, so a patch is kept only when the share of the total
    error it covers beats the share of the file it costs, and the pass as a
    whole is kept only when mean * bytes went down. Needs cairosvg and scipy;
    without either the pass is skipped and `parts` comes back unchanged.

    A patch takes a palette colour whenever its own mean is within merge_dist of
    one. A patch colour off the palette is a colour no layer has, and tracing the
    result again has to invent it: ten such patches on cat_mascot bought 0.08 of
    trace mean and cost 0.36 of round-trip mean. Only a blob whose colour really
    is absent -- one the palette dropped -- gets a fill of its own.
    """
    try:
        import cairosvg
        from scipy import ndimage
    except ImportError:
        say("repair needs cairosvg and scipy; skipped")
        return parts
    s = cfg["scale"]
    src = _flat(src_rgba)
    alpha = src_rgba[:, :, 3]
    lab_pal = to_lab(pal)
    inside = alpha >= 128                       # never paint outside the artwork
    tmp = os.path.join(tempfile.gettempdir(), "_repair%d.svg" % os.getpid())
    png = tmp + ".png"

    def shot(ps):
        """Write, render and score one candidate SVG. Returns (error map, bytes)."""
        text = _svg(w, h, defs + "\n".join(ps))
        open(tmp, "w").write(text)
        cairosvg.svg2png(url=tmp, write_to=png, output_width=w, output_height=h)
        got = np.array(Image.open(png).convert("RGBA")).astype(float)
        return np.abs(src - _flat(got)).max(2), len(text)

    for p in range(cfg["repair_passes"]):
        d, size = shot(parts)
        was = d.mean() * size                   # err_kb, up to a constant
        hot = (d > cfg["repair_thr"]) & inside
        # Most of the residual is a one-pixel thread along every edge: the vector
        # boundary sits a fraction of a pixel off the source antialiasing. A patch
        # cannot fix that -- its own edge lands in the same wrong place -- and those
        # threads chain into one snake that spans the artwork. An opening keeps only
        # error with a body; dilating that core back inside `hot` returns each
        # blob's own rim without letting the threads back in.
        core = ndimage.binary_opening(hot, EIGHT)
        comp, k = ndimage.label(ndimage.binary_dilation(core, EIGHT) & hot, EIGHT)
        if not k:
            break
        area = np.bincount(comp.ravel(), minlength=k + 1)[1:]
        mass = np.bincount(comp.ravel(), d.ravel(), k + 1)[1:]
        total = d.sum()
        boxes = ndimage.find_objects(comp)
        masks, cand = [], []
        # by error mass, biggest first; stop once even a bare path (~60 bytes) could
        # not earn its keep on the whole cluster -- the rest are smaller still
        for c in np.argsort(-mass):
            if mass[c] / total * size <= 60:
                break
            if area[c] < cfg["repair_min"]:
                continue
            sy, sx = boxes[c]
            y0, x0 = max(sy.start - 1, 0), max(sx.start - 1, 0)
            m = comp[y0:sy.stop + 1, x0:sx.stop + 1] == c + 1
            px = src[y0:y0 + m.shape[0], x0:x0 + m.shape[1]][m]
            col = px.mean(0)
            op = alpha[y0:y0 + m.shape[0], x0:x0 + m.shape[1]][m].mean() / 255.0
            # a soft blob: paint the colour it has under its own alpha, not over white
            fill = col if op >= 250 / 255 else np.clip((col - 255 * (1 - op)) / op, 0, 255)
            j = int(nearest(to_lab(fill[None]), lab_pal)[0])
            if ((to_lab(fill) - lab_pal[j]) ** 2).sum() <= cfg["merge_dist"] ** 2:
                fill = pal[j].astype(float)     # near enough to be the same colour
                col = fill * op + 255 * (1 - op)
            # what one flat colour will still be wrong by. A blob that is really one
            # fill leaves almost nothing; a cluster over a glyph or a busy detail is
            # black and white at once, and its mean grey is worse than what is there.
            gain = (d[y0:y0 + m.shape[0], x0:x0 + m.shape[1]][m]
                    - np.abs(px - col).max(1)).sum()
            if gain / total * size <= 60:
                continue
            masks.append(np.repeat(np.repeat(m, s, 0), s, 1))
            cand.append((gain, fill, op, y0, x0))
        if not masks:
            break
        new = []
        for (gain, fill, op, y0, x0), dpath in zip(
                cand, _trace_layers(masks, cfg, lambda *_: None)):
            if not dpath:
                continue
            dpath = _offset(straighten(dpath, cfg["straight_tol"]), x0, y0)
            fo = ' fill-opacity="%.3f"' % op if op < 250 / 255 else ""
            el = ('<path fill="#%02X%02X%02X"%s d="%s"/>'
                  % (*fill.round().astype(int), fo, dpath))
            if gain / total > len(el) / size:   # error it buys beats the file it costs
                new.append(el)
        if not new:
            break
        d2, size2 = shot(parts + new)
        if d2.mean() * size2 >= was:
            say("repair pass %d: %d patches did not pay (mean %.3f->%.3f, "
                "bytes %d->%d); dropped" % (p + 1, len(new), d.mean(), d2.mean(),
                                            size, size2))
            break
        say("repair pass %d: %d patches, mean %.3f -> %.3f, bytes %d -> %d"
            % (p + 1, len(new), d.mean(), d2.mean(), size, size2))
        parts = parts + new
    for f in (tmp, png):
        if os.path.exists(f):
            os.remove(f)
    return parts


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
    s = cfg["scale"]
    lab, sure = label_all(rgb, solid, inside, pal, cfg["merge_dist"] / 2)
    if cfg["ink_min"]:
        lab, sure = ink(lab, rgb, inside, pal, sure, s, cfg["ink_tol"],
                        cfg["ink_min"], cfg["ink_share"], cfg["min_width"],
                        cfg["ink_de"])
    lab = smooth(lab, inside, len(pal), cfg["smooth_radius"] * s, cfg["smooth_passes"], sure)
    lab = thin(lab, rgb, inside, pal, cfg["min_width"] * cfg["scale"], cfg["band_tol"],
               cfg["edge_share"])
    lab = smooth(lab, inside, len(pal), cfg["smooth_radius"] * s, cfg["smooth_passes"], sure)
    lab = islands(lab, pal, inside, cfg["max_island"] * s * s, 2 * cfg["merge_dist"])

    grad_defs, grad_fills = [], {}
    if cfg["grad_gain"]:
        a1o = a1[:, :, 3:4] / 255.0                 # blend over white: what a path renders as
        blended = a1[:, :, :3] * a1o + 255 * (1 - a1o)
        grad_defs, grad_fills = _gradients(lab, blended, pal, w, h, s, cfg["grad_gain"])

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
            fill = "url(#%s)" % grad_fills[i][0] if i in grad_fills else "#%02X%02X%02X" % tuple(pal[i])
            parts.append('<path fill="%s" d="%s"/>' % (fill, straighten(d, cfg["straight_tol"])))

    defs = "<defs>\n%s\n</defs>\n" % "\n".join(grad_defs) if grad_defs else ""
    if cfg["repair_passes"]:
        say("repair residuals")
        parts = repair(a1.astype(float), parts, defs, pal, w, h, cfg, say)
    open(out, "w").write(_svg(w, h, defs + "\n".join(parts)))
    say("done")
    return ["#%02X%02X%02X" % (grad_fills[i][1] if i in grad_fills else tuple(pal[i]))
            for i in order]


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
    d = np.abs(_flat(a) - _flat(b)).max(2)
    return dict(mean=d.mean(), max=d.max(), over40=100 * (d > 40).mean(),
                within8=100 * (d <= 8).mean(), render=png)


def demo():
    # the black/white boundary sits at 50% coverage (127), not at equal Lab (119)
    bw = np.array([[0, 0, 0], [255, 255, 255]], float)
    grey = np.array([[122, 122, 122], [133, 133, 133]], float)
    assert list(nearest(to_lab(grey), to_lab(bw))) == [1, 1]
    assert list(nearest(to_lab(grey), to_lab(bw), rgb_px=grey, rgb_pal=bw)) == [0, 1]
    # a textured fill spread over two near entries merges; two flat fills stay
    rng = np.random.default_rng(0)
    grain = np.clip(rng.normal([90, 60, 30], 12, (4000, 3)), 0, 255).astype(int)
    flat = np.array([[240, 240, 240]] * 3000 + [[200, 200, 200]] * 3000)
    pal = palette(np.vstack([grain, flat])[None], np.ones((1, 10000), bool), _cfg(dict(overlap=0.3)))
    assert len(pal) == 3, pal
    # a 3px band of blend colour between two fills is relabelled; a stroke is not
    pal = np.array([[0, 0, 0], [200, 200, 200], [100, 100, 100], [255, 0, 0]])
    lab = np.zeros((20, 40), np.int16)
    lab[:, 20:] = 1
    lab[:, 19:22] = 2                                             # the band
    lab[:, 5:8] = 3                                               # a red stroke on black
    rgb = pal[lab]
    out = thin(lab, rgb, np.ones(lab.shape, bool), pal, 4, 20, 0.7)
    assert not (out == 2).any() and (out == 3).sum() == 60, out[10]
    lab[:, 30:36] = 2                                             # now colour 2 has a body too
    lab[:, 19:22] = 2
    out = thin(lab, pal[lab], np.ones(lab.shape, bool), pal, 4, 20, 0.7)
    assert not (out[:, 19:22] == 2).any() and (out[:, 30:36] == 2).all(), out[10]
    # a small near-colour patch inside a fill goes; a far-colour patch stays
    pal = np.array([[200, 120, 60], [190, 110, 55], [255, 255, 255]])
    lab = np.zeros((40, 40), np.int16)
    lab[10:14, 10:14] = 1
    lab[25:29, 25:29] = 2
    out = islands(lab, pal, np.ones(lab.shape, bool), 50, 20)
    assert (out[10:14, 10:14] == 0).all() and (out[25:29, 25:29] == 2).all(), out
    # the vote settles a flip along a gradient but keeps a 4px line
    lab = np.zeros((30, 60), np.int16)
    lab[:, 30:] = 1
    lab[:, 28:32] = np.arange(30)[:, None] % 2                    # flicker on the border
    lab[:, 10:14] = 1                                             # a line, 4 grid px
    out = smooth(lab, np.ones(lab.shape, bool), 2, 2, 2)
    assert (out[:, 10:14] == 1).all() and (out[2:-2, 28:30] == 0).all() and (out[2:-2, 30:32] == 1).all(), out[:4]
    assert straighten("M0 0 L5 0 L10 0 Z").count("L") == 1        # straight run collapses
    assert straighten("M0 0 L10 0 L10 10 L0 10 Z").count("L") == 3  # corners survive
    # a 0.6 px line on a pixel edge leaves two columns holding a third of a
    # pixel of ink each: no threshold calls either one ink, coverage does
    from PIL import ImageDraw
    big = Image.new("L", (512, 512), 255)
    ImageDraw.Draw(big).rectangle([16, 16, 175, 175], fill=0)      # black, for the palette
    ImageDraw.Draw(big).line([(304, 0), (304, 511)], fill=0, width=5)
    p = os.path.join(tempfile.gettempdir(), "_thin_line.png")
    big.resize((64, 64), Image.BOX).convert("RGBA").save(p)        # real coverage
    trace(p, p + ".svg", min_share=0.05)
    try:
        r = compare(p, p + ".svg")
    except ImportError:
        print("cairosvg not installed; skipped the thin-line check")
    else:
        g = lambda f: 255 - np.asarray(Image.open(f).convert("L"), float)[:, 30:46]
        a, b = g(p).sum(), g(r["render"]).sum()
        assert abs(b - a) <= 0.25 * a, (a, b)                      # ink is conserved
    im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))                # red square on nothing
    for y in range(16, 48):
        for x in range(16, 48):
            im.putpixel((x, y), (200, 30, 30, 255))
    im.save("_demo.png")
    trace("_demo.png", "_demo.svg", scale=2)
    d = open("_demo.svg").read()
    assert d.count("<path") == 1 and "#C81E1E" in d, d[:200]
    # a shape too small for the palette comes back as a repair patch
    miss = Image.new("RGBA", (64, 64), (200, 30, 30, 255))
    for y in range(26, 38):
        for x in range(26, 38):
            miss.putpixel((x, y), (30, 30, 200, 255))       # 144px = 3.5%, under min_share
    miss_png = os.path.join(tempfile.gettempdir(), "_demo_miss.png")
    miss_svg = os.path.join(tempfile.gettempdir(), "_demo_miss.svg")
    miss.save(miss_png)
    trace(miss_png, miss_svg, scale=2, min_share=0.05)
    md = open(miss_svg).read()
    assert md.count("<path") == 2, md[:300]
    assert compare(miss_png, miss_svg)["mean"] < 3, compare(miss_png, miss_svg)
    # a left-to-right ramp in one flat-merged region beats its flat fill with a gradient
    ramp = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    for y in range(64):
        for x in range(64):
            ramp.putpixel((x, y), (200 + round(55 * x / 63), 0, 0, 255))
    ramp_png = os.path.join(tempfile.gettempdir(), "_demo_ramp.png")
    ramp_svg = os.path.join(tempfile.gettempdir(), "_demo_ramp.svg")
    ramp.save(ramp_png)
    trace(ramp_png, ramp_svg, scale=2, merge_dist=100)             # force one region
    gd = open(ramp_svg).read()
    assert "linearGradient" in gd, gd[:300]
    r = compare(ramp_png, ramp_svg)
    assert r["mean"] < 2, r
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
