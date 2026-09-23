"""Tkinter front end for png2svg.py.

    pip install potracer pillow numpy cairosvg vtracer
    scoop install potrace          # optional C binary: ~15x faster, used if present
    python png2svg_app.py
"""
import io
import os
import queue
import re
import threading
import tkinter as tk
from tkinter import filedialog, ttk

from PIL import Image, ImageTk

import png2svg

THUMB = 220


class App(ttk.Frame):
    def __init__(self, root):
        super().__init__(root, padding=12)
        self.grid(sticky="nsew")
        root.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.msgs = queue.Queue()
        self.busy = False
        self.thumbs = [None, None]              # keep refs, else Tk drops the images

        self.src = tk.StringVar()
        self.out = tk.StringVar()
        self.mode = tk.StringVar(value="trace")
        self.merge = tk.IntVar(value=png2svg.DEFAULTS["merge_dist"])
        self.share = tk.DoubleVar(value=png2svg.DEFAULTS["min_share"] * 100)
        self.soft = tk.DoubleVar(value=png2svg.DEFAULTS["min_width"])
        self.scale = tk.IntVar(value=png2svg.DEFAULTS["scale"])
        self.passes = tk.IntVar(value=png2svg.DEFAULTS["smooth_passes"])
        self.ld = tk.IntVar(value=png2svg.VTRACER["layer_difference"])
        self.fs = tk.IntVar(value=png2svg.VTRACER["filter_speckle"])
        self.pp = tk.IntVar(value=png2svg.VTRACER["path_precision"])
        self.lossless = tk.BooleanVar(value=True)
        self.prec = tk.IntVar(value=4)
        self.gz = tk.BooleanVar(value=False)

        r = 0
        ttk.Label(self, text="Input").grid(row=r, column=0, sticky="w")
        ttk.Entry(self, textvariable=self.src).grid(row=r, column=1, sticky="ew", padx=6)
        ttk.Button(self, text="Browse", command=self.pick_src).grid(row=r, column=2)

        r += 1
        ttk.Label(self, text="SVG").grid(row=r, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(self, textvariable=self.out).grid(row=r, column=1, sticky="ew", padx=6,
                                                    pady=(6, 0))
        ttk.Button(self, text="Browse", command=self.pick_out).grid(row=r, column=2,
                                                                    pady=(6, 0))

        r += 1
        box = ttk.Frame(self)
        box.grid(row=r, column=0, columnspan=3, sticky="w", pady=(10, 0))
        ttk.Radiobutton(box, text="Smooth vector", value="trace", variable=self.mode,
                        command=self.toggle).pack(side="left")
        ttk.Radiobutton(box, text="VTracer (fast)", value="vtracer", variable=self.mode,
                        command=self.toggle).pack(side="left", padx=(12, 0))
        ttk.Radiobutton(box, text="Exact pixel copy", value="exact", variable=self.mode,
                        command=self.toggle).pack(side="left", padx=(12, 0))
        ttk.Radiobutton(box, text="Compress SVG", value="compress", variable=self.mode,
                        command=self.toggle).pack(side="left", padx=(12, 0))

        r += 1
        self.opts = ttk.LabelFrame(self, text="Vector settings", padding=8)
        self.opts.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        self.opts.columnconfigure(1, weight=1)
        self._spin("Colour distance", self.merge, 2, 60, 0,
                   "how far apart two colours must be to stay separate")
        self._spin("Smallest region %", self.share, 0.01, 20, 1,
                   "drop colours below this share of the artwork", inc=0.05)
        self._spin("Edge softness px", self.soft, 0, 6, 4,
                   "width of the blended band along a soft edge", inc=0.5)
        self._spin("Trace grid", self.scale, 1, 8, 2,
                   "4 puts edges on a quarter pixel; 1 is fastest")
        self._spin("Speckle passes", self.passes, 0, 6, 3,
                   "majority votes that clean compression noise")

        self.vopts = ttk.LabelFrame(self, text="VTracer settings", padding=8)
        self.vopts.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        self.vopts.columnconfigure(1, weight=1)
        self._spin("Colour distance", self.ld, 2, 128, 0,
                   "how far apart two colours must be to stay separate", box=self.vopts)
        self._spin("Speckle size", self.fs, 0, 64, 1,
                   "drop traced specks under this many pixels", box=self.vopts)
        self._spin("Path precision", self.pp, 1, 8, 2,
                   "decimal places in the path data; 2 makes a smaller file",
                   box=self.vopts)
        self.vopts.grid_remove()

        self.copts = ttk.LabelFrame(self, text="Compress settings", padding=8)
        self.copts.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        self.copts.columnconfigure(1, weight=1)
        ttk.Radiobutton(self.copts, text="Lossless", value=True, variable=self.lossless,
                        command=self.toggle).grid(row=0, column=0, sticky="w")
        ttk.Label(self.copts, text="keeps every digit, id and title",
                  foreground="#777").grid(row=0, column=2, sticky="w")
        ttk.Radiobutton(self.copts, text="Round to", value=False, variable=self.lossless,
                        command=self.toggle).grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.prec_box = ttk.Spinbox(self.copts, from_=1, to=8, textvariable=self.prec, width=6)
        self.prec_box.grid(row=2, column=1, sticky="w", padx=8, pady=(4, 0))
        ttk.Label(self.copts, text="significant digits; 3 is smaller but can shift edges",
                  foreground="#777").grid(row=2, column=2, sticky="w", pady=(4, 0))
        ttk.Checkbutton(self.copts, text="gzip (.svgz)", variable=self.gz,
                        command=self.swap_ext).grid(row=3, column=0, columnspan=3,
                                                    sticky="w", pady=(4, 0))
        self.copts.grid_remove()

        r += 1
        exe = png2svg.find_potrace()
        engine = ("engine: C potrace, %d workers" % (os.cpu_count() or 1) if exe else
                  "engine: Python port (slow) - for ~15x, run: scoop install potrace")
        try:
            import vtracer                             # noqa: F401
        except ImportError:
            engine += "  |  no vtracer - run: pip install vtracer"
        try:
            import scour                               # noqa: F401
        except ImportError:
            engine += "  |  no scour - run: pip install scour"
        ttk.Label(self, text=engine, foreground="#777" if exe else "#a33").grid(
            row=r, column=0, columnspan=3, sticky="w", pady=(8, 0))

        r += 1
        self.go = ttk.Button(self, text="Convert", command=self.start)
        self.go.grid(row=r, column=0, sticky="w", pady=(12, 0))
        self.bar = ttk.Progressbar(self, mode="indeterminate")
        self.bar.grid(row=r, column=1, columnspan=2, sticky="ew", padx=6, pady=(12, 0))

        r += 1
        self.status = ttk.Label(self, text="Pick a PNG or an SVG.", foreground="#555")
        self.status.grid(row=r, column=0, columnspan=3, sticky="w", pady=(8, 0))

        r += 1
        self.view = ttk.Frame(self)
        self.view.grid(row=r, column=0, columnspan=3, pady=(10, 0))
        self.src_view = ttk.Label(self.view)
        self.src_view.grid(row=0, column=0, padx=6)
        self.out_view = ttk.Label(self.view)
        self.out_view.grid(row=0, column=1, padx=6)

    def _spin(self, text, var, lo, hi, row, hint, inc=1, box=None):
        box = box or self.opts
        ttk.Label(box, text=text).grid(row=row, column=0, sticky="w")
        ttk.Spinbox(box, from_=lo, to=hi, increment=inc, textvariable=var,
                    width=6).grid(row=row, column=1, sticky="w", padx=8)
        ttk.Label(box, text=hint, foreground="#777").grid(row=row, column=2, sticky="w")

    def toggle(self):
        """Only the panel of the chosen mode stays on screen. Both share one row."""
        mode = self.mode.get()
        (self.opts.grid if mode == "trace" else self.opts.grid_remove)()
        (self.vopts.grid if mode == "vtracer" else self.vopts.grid_remove)()
        (self.copts.grid if mode == "compress" else self.copts.grid_remove)()
        self.prec_box.configure(state="disabled" if self.lossless.get() else "normal")

    def swap_ext(self):
        """The .svgz box and the output name must agree: compress() gzips by flag."""
        out = self.out.get()
        if out:
            self.out.set(re.sub(r"\.svgz?$", "", out) + (".svgz" if self.gz.get() else ".svg"))

    def pick_src(self):
        p = filedialog.askopenfilename(filetypes=[("PNG or SVG", "*.png *.svg *.svgz"),
                                                  ("All", "*.*")])
        if p:
            self.src.set(p)
            base = re.sub(r"\.\w+$", "", p)
            if p.lower().endswith((".svg", ".svgz")):   # an SVG can only be compressed
                self.mode.set("compress")
                self.toggle()
                base += ".min"                          # never write over the source
            self.out.set(base + (".svgz" if self.mode.get() == "compress" and self.gz.get()
                                 else ".svg"))
            self.show(self.src_view, p)

    def pick_out(self):
        p = filedialog.asksaveasfilename(defaultextension=".svg",
                                         filetypes=[("SVG", "*.svg"), ("SVGZ", "*.svgz")])
        if p:
            self.out.set(p)

    def show(self, label, path, i=0):
        try:
            if path.lower().endswith((".svg", ".svgz")):
                import cairosvg
                path = io.BytesIO(cairosvg.svg2png(url=path))
            im = Image.open(path).convert("RGBA")
            im.thumbnail((THUMB, THUMB))
            flat = Image.new("RGBA", im.size, (255, 255, 255, 255))
            flat.alpha_composite(im)
            self.thumbs[i] = ImageTk.PhotoImage(flat)
            label.configure(image=self.thumbs[i])
        except Exception:
            label.configure(image="")

    # --- run ---

    def start(self):
        src, out = self.src.get().strip(), self.out.get().strip()
        if not src or not os.path.exists(src):
            return self.say("Pick a PNG that exists.")
        if not out:
            return self.say("Pick an output path.")
        if os.path.abspath(out) == os.path.abspath(src):
            return self.say("Pick an output path that is not the input.")
        # read every Tk variable here: a worker thread must not touch Tk
        mode = self.mode.get()
        if mode == "compress":
            job = dict(mode=mode, gz=self.gz.get(),
                       precision=None if self.lossless.get() else self.prec.get())
        elif mode == "vtracer":
            job = dict(mode=mode, layer_difference=self.ld.get(),
                       filter_speckle=self.fs.get(), path_precision=self.pp.get())
        else:
            job = dict(mode=mode, merge_dist=self.merge.get(),
                       min_share=self.share.get() / 100, min_width=self.soft.get(),
                       scale=self.scale.get(), smooth_passes=self.passes.get())
        self.busy = True
        self.go.configure(state="disabled")
        self.bar.start(12)
        self.after_idle(self.poll)
        threading.Thread(target=self.work, args=(src, out, job), daemon=True).start()

    def work(self, src, out, job):
        put = lambda kind, text: self.msgs.put((kind, text))
        try:
            mode = job.pop("mode")
            if mode == "compress":
                put("step", "compress")
                a, b = png2svg.compress(src, out, **job)
                head = "%.1f KB -> %.1f KB (-%.0f%%)" % (a / 1024, b / 1024, 100 * (1 - b / a))
                try:
                    put("step", "score the result")
                    r = png2svg.compare_svg(src, out)
                    put("preview", r["render"])
                    head += ", mean %.3f/255 from the original" % r["mean"]
                except ImportError:
                    head += " (install cairosvg to score it)"
                return put("done", head)
            if mode == "exact":
                n = png2svg.pixel_copy(src, out, lambda t: put("step", t))
                head = "rectangles: %d" % n
            elif mode == "vtracer":
                png2svg.vtrace(src, out, lambda t: put("step", t), **job)
                head = "paths: %d" % open(out).read().count("<path")
            else:
                cols = png2svg.trace(src, out, lambda t: put("step", t), **job)
                head = "colours: %d" % len(cols)
            kb = os.path.getsize(out) / 1024
            try:
                put("step", "score the result")
                r = png2svg.compare(src, out)
                put("preview", r["render"])
                head += ", %.0f KB, mean %.3f/255, off by >40: %.3f%%" % (
                    kb, r["mean"], r["over40"])
            except ImportError:
                head += ", %.0f KB (install cairosvg to score it)" % kb
            put("done", head)
        except Exception as e:                  # show the failure, keep the app alive
            put("fail", "%s: %s" % (type(e).__name__, e))

    def poll(self):
        try:
            while True:
                kind, text = self.msgs.get_nowait()
                if kind == "preview":
                    self.show(self.out_view, text, 1)
                elif kind == "done":
                    self.finish("Wrote %s - %s" % (os.path.basename(self.out.get()), text))
                elif kind == "fail":
                    self.finish(text)
                else:
                    self.say(text)
        except queue.Empty:
            pass
        if self.busy:
            self.after(80, self.poll)

    def finish(self, text):
        self.busy = False
        self.bar.stop()
        self.go.configure(state="normal")
        self.say(text)

    def say(self, text):
        self.status.configure(text=text)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("PNG to SVG")
    root.minsize(680, 420)
    App(root)
    root.mainloop()
