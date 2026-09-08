"""Score every PNG in bench/ and diff against the saved baseline.

    python bench.py            # run, print table, diff vs bench/baseline.json
    python bench.py --save     # also overwrite the baseline

A change to png2svg.py must move this table, or it does not go in.
mean/over40 come from compare(): pixel error after a render back to PNG.
"rt" (round-trip, trace mode only) re-traces the trace-mode render instead of
the source PNG, so its mean/over40 isolate error the tracer itself adds, with
source noise already removed by the first round-trip. err_kb = mean * bytes /
1024 for every mode: a cheap size-aware score where lower is better, since a
smaller mean at a much larger file size is not obviously an improvement.
"""
import glob
import json
import os
import re
import shutil
import sys
import time

import png2svg

BASE = "bench/baseline.json"
MODES = {"trace": png2svg.trace, "vtrace": png2svg.vtrace}


def _score_svg(svg_path, cmp_src, secs):
    svg = open(svg_path).read()
    r = png2svg.compare(cmp_src, svg_path)
    d = dict(mean=round(r["mean"], 3), over40=round(r["over40"], 3),
             paths=svg.count("<path"), nodes=len(re.findall(r"[MLCQ]", svg)),
             bytes=len(svg), secs=round(secs, 2))
    d["err_kb"] = round(d["mean"] * d["bytes"] / 1024, 2)
    return d, r["render"]


def score(mode, src):
    out = "bench/out/%s_%s.svg" % (mode, os.path.basename(src)[:-4])
    t = time.time()
    MODES[mode](src, out)
    secs = time.time() - t
    return _score_svg(out, src, secs)


def score_rt(src, trace_render):
    name = os.path.basename(src)[:-4]
    rt_png = "bench/out/rt_%s.png" % name
    shutil.copyfile(trace_render, rt_png)
    rt_svg = "bench/out/rt_%s.svg" % name
    t = time.time()
    png2svg.trace(rt_png, rt_svg)
    secs = time.time() - t
    d, _ = _score_svg(rt_svg, rt_png, secs)
    return d


def main():
    os.makedirs("bench/out", exist_ok=True)
    base = json.load(open(BASE)) if os.path.exists(BASE) else {}
    res = {}
    keys = ("mean", "over40", "err_kb", "paths", "nodes", "bytes", "secs")
    print("%-28s %-6s " % ("file", "mode") + " ".join("%10s" % k for k in keys))
    for src in sorted(glob.glob("bench/*.png")):
        trace_render = None
        for mode in ("trace", "vtrace", "rt"):
            name = "%s:%s" % (os.path.basename(src), mode)
            if mode == "rt":
                r = res[name] = score_rt(src, trace_render)
            else:
                r, render = score(mode, src)
                res[name] = r
                if mode == "trace":
                    trace_render = render
            cells = []
            for k in keys:
                s = "%10g" % r[k]
                if name in base and k in base[name] and base[name][k] != r[k]:
                    d = r[k] - base[name][k]
                    s = "%10s" % ("%g(%+g)" % (r[k], round(d, 3)))
                cells.append(s)
            print("%-28s %-6s " % (os.path.basename(src), mode) + " ".join(cells))
    tot = {k: sum(r[k] for r in res.values()) for k in keys}
    line = "%-35s " % "TOTAL" + " ".join("%10g" % round(tot[k], 2) for k in keys)
    if base:
        bt = {k: sum(base[n][k] for n in res if n in base and k in base[n]) for k in keys}
        line += "\n%-35s " % "vs base" + " ".join(
            "%10s" % ("%+g" % round(tot[k] - bt[k], 2)) for k in keys)
    print(line)
    if "--save" in sys.argv:
        json.dump(res, open(BASE, "w"), indent=1)
        print("baseline saved")


if __name__ == "__main__":
    main()
