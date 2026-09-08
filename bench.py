"""Score every PNG in bench/ and diff against the saved baseline.

    python bench.py            # run, print table, diff vs bench/baseline.json
    python bench.py --save     # also overwrite the baseline

A change to png2svg.py must move this table, or it does not go in.
mean/over40 come from compare(): pixel error after a render back to PNG.
"""
import glob
import json
import os
import re
import sys
import time

import png2svg

BASE = "bench/baseline.json"
MODES = {"trace": png2svg.trace, "vtrace": png2svg.vtrace}


def score(mode, src):
    out = "bench/out/%s_%s.svg" % (mode, os.path.basename(src)[:-4])
    t = time.time()
    MODES[mode](src, out)
    secs = time.time() - t
    svg = open(out).read()
    r = png2svg.compare(src, out)
    return dict(mean=round(r["mean"], 3), over40=round(r["over40"], 3),
                paths=svg.count("<path"), nodes=len(re.findall(r"[MLCQ]", svg)),
                bytes=len(svg), secs=round(secs, 2))


def main():
    os.makedirs("bench/out", exist_ok=True)
    base = json.load(open(BASE)) if os.path.exists(BASE) else {}
    res = {}
    keys = ("mean", "over40", "paths", "nodes", "bytes", "secs")
    print("%-28s %-6s " % ("file", "mode") + " ".join("%10s" % k for k in keys))
    for src in sorted(glob.glob("bench/*.png")):
        for mode in MODES:
            name = "%s:%s" % (os.path.basename(src), mode)
            r = res[name] = score(mode, src)
            cells = []
            for k in keys:
                s = "%10g" % r[k]
                if name in base and base[name][k] != r[k]:
                    d = r[k] - base[name][k]
                    s = "%10s" % ("%g(%+g)" % (r[k], round(d, 3)))
                cells.append(s)
            print("%-28s %-6s " % (os.path.basename(src), mode) + " ".join(cells))
    tot = {k: sum(r[k] for r in res.values()) for k in keys}
    line = "%-35s " % "TOTAL" + " ".join("%10g" % round(tot[k], 2) for k in keys)
    if base:
        bt = {k: sum(base[n][k] for n in res if n in base) for k in keys}
        line += "\n%-35s " % "vs base" + " ".join(
            "%10s" % ("%+g" % round(tot[k] - bt[k], 2)) for k in keys)
    print(line)
    if "--save" in sys.argv:
        json.dump(res, open(BASE, "w"), indent=1)
        print("baseline saved")


if __name__ == "__main__":
    main()
