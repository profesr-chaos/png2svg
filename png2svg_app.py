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
from tkinter import colorchooser, filedialog, ttk

from PIL import Image, ImageTk

import png2svg
import svg_edit

CMP_VIEW = 260                          # pixel size of each compare canvas


def _size(n):
    """Bytes in the unit that fits: B under 1 KB, KB under 1 MB, else MB."""
    if n < 1024:
        return "%d B" % n
    if n < 1024 * 1024:
        return "%.1f KB" % (n / 1024)
    return "%.1f MB" % (n / 1024 / 1024)


def _saving(before, after):
    """'Saved 1.3 KB (8%): 16.3 KB -> 15.0 KB'. A tiny .svgz can grow: the
    gzip header costs bytes, so say 'Grew', not a negative saving."""
    d = before - after
    return "%s %s (%.0f%%): %s -> %s" % ("Saved" if d >= 0 else "Grew", _size(abs(d)),
                                         100 * abs(d) / before, _size(before), _size(after))


class App(ttk.Frame):
    def __init__(self, root):
        super().__init__(root, padding=12)
        self.grid(sticky="nsew")
        root.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.msgs = queue.Queue()
        self.busy = False

        # palette panel + compare view state
        self.edited = None              # svg_edit.Edited over the traced SVG's text
        self.palette = {}                # fill key -> swatch hex, from svg_edit.list_colours
        self.swatch_widgets = {}         # fill key -> its swatch Label, for highlighting
        self.swatch_rows = {}            # fill key -> its row Frame, to scroll to it
        self.hidden = set()              # fill keys hidden in the preview only
        # the selected swatch: sunken in the list, isolated (svg_edit.highlight) in the
        # preview, and the target of Merge into.../Recolour... once one is picked
        self.selected = None
        self.merge_armed = False
        self.merge_source = None
        self.src_img = None              # PIL image, source flattened over white
        self.render_img = None           # PIL image, last rendered preview
        self.center = None               # (x, y) in source-image pixels the compare view centres on
        self.cmp_zoom = tk.StringVar(value="Fit")   # "Fit" or "1".."8"

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
        ttk.Label(self, text="Output").grid(row=r, column=0, sticky="w", pady=(6, 0))
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
        work = ttk.Frame(self)
        work.grid(row=r, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
        self._build_palette(work)
        self._build_compare(work)

    def _build_palette(self, parent):
        """The swatch list: click to isolate, double-click (or Recolour...) to pick a colour."""
        pal = ttk.LabelFrame(parent, text="Palette", padding=6)
        pal.grid(row=0, column=0, sticky="ns")

        bar = ttk.Frame(pal)
        bar.pack(fill="x")
        ttk.Button(bar, text="Recolour...", command=self.start_recolour).pack(side="left")
        ttk.Button(bar, text="Merge into...", command=self.start_merge).pack(side="left", padx=(4, 0))
        ttk.Button(bar, text="All", command=self.show_all).pack(side="left", padx=(4, 0))
        bar2 = ttk.Frame(pal)
        bar2.pack(fill="x", pady=(2, 0))
        ttk.Button(bar2, text="Undo", command=self.do_undo).pack(side="left")
        ttk.Button(bar2, text="Save", command=self.do_save).pack(side="left", padx=(4, 0))
        ttk.Button(bar2, text="Save as...", command=self.do_save_as).pack(side="left", padx=(4, 0))

        canvas = tk.Canvas(pal, width=230, height=CMP_VIEW, highlightthickness=0)
        vsb = ttk.Scrollbar(pal, orient="vertical", command=canvas.yview)
        self.palette_canvas = canvas
        self.palette_inner = ttk.Frame(canvas)
        self.palette_inner.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.palette_inner, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill="both", expand=True, pady=(6, 0))
        vsb.pack(side="left", fill="y", pady=(6, 0))

    def _build_compare(self, parent):
        """Source and result, zoomable, click either to recentre both (or read a colour)."""
        box = ttk.LabelFrame(parent, text="Compare (source | result)", padding=6)
        box.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        zrow = ttk.Frame(box)
        zrow.pack(fill="x")
        ttk.Label(zrow, text="Zoom").pack(side="left")
        ttk.Spinbox(zrow, values=("Fit", "1", "2", "3", "4", "5", "6", "7", "8"),
                   textvariable=self.cmp_zoom, width=5, state="readonly").pack(
            side="left", padx=6)

        imgs = ttk.Frame(box)
        imgs.pack(pady=(6, 0))
        self.src_canvas = tk.Canvas(imgs, width=CMP_VIEW, height=CMP_VIEW,
                                    background="#ccc", highlightthickness=1)
        self.src_canvas.grid(row=0, column=0, padx=4)
        self.out_canvas = tk.Canvas(imgs, width=CMP_VIEW, height=CMP_VIEW,
                                    background="#ccc", highlightthickness=1)
        self.out_canvas.grid(row=0, column=1, padx=4)
        self.src_canvas.bind("<Button-1>", lambda e: self.on_compare_click(e, "src"))
        self.out_canvas.bind("<Button-1>", lambda e: self.on_compare_click(e, "out"))
        self.cmp_zoom.trace_add("write", lambda *a: self.redraw_compare())

        ttk.Label(box, foreground="#777", text="Click centres the view; at Fit, or with "
                 "Ctrl held, click reads a colour into the palette instead.").pack(
            anchor="w", pady=(4, 0))

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
            self.load_source(p)

    def pick_out(self):
        cur = self.out.get()                    # open at the current output, not a blank name
        p = filedialog.asksaveasfilename(defaultextension=".svg",
                                         filetypes=[("SVG", "*.svg"), ("SVGZ", "*.svgz")],
                                         initialdir=os.path.dirname(cur) or None,
                                         initialfile=os.path.basename(cur))
        if p:
            self.out.set(p)

    def load_source(self, path):
        """Flatten the PNG over white, same as a path fill renders, for the compare view."""
        try:
            if path.lower().endswith((".svg", ".svgz")):
                import cairosvg
                path = io.BytesIO(cairosvg.svg2png(url=path))
            im = Image.open(path).convert("RGBA")
        except Exception:
            self.src_img = None
            return
        flat = Image.new("RGBA", im.size, (255, 255, 255, 255))
        flat.alpha_composite(im)
        self.src_img = flat.convert("RGB")
        self.center = None
        self.render_img = None
        self.redraw_compare()

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
                head = _saving(a, b)
                try:
                    put("step", "score the result")
                    r = png2svg.compare_svg(src, out)
                    put("preview", r["render"])
                    head += ", mean %.3f/255 from the original" % r["mean"]
                except ImportError:
                    head += " (install cairosvg to score it)"
                except Exception as e:          # the file is written; keep the saving
                    head += " (no score: %s)" % e
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
            put("svg", (out, open(out, encoding="utf-8").read()))
            try:
                put("step", "score the result")
                r = png2svg.compare(src, out)
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
                kind, payload = self.msgs.get_nowait()
                if kind == "svg":
                    path, text = payload
                    self.edited = svg_edit.Edited(text)
                    self.hidden, self.selected, self.merge_armed = set(), None, False
                    self.center = None
                    self.cmp_zoom.set("Fit")
                    self.rebuild_palette()
                    self.render_current()
                elif kind == "preview":         # compress: path to the rendered PNG
                    im = Image.open(payload).convert("RGBA")
                    self.render_img = Image.alpha_composite(
                        Image.new("RGBA", im.size, "white"), im).convert("RGB")
                    self.redraw_compare()
                elif kind == "done":
                    # result first: a narrow window cuts the end of the line
                    self.finish("%s - wrote %s" % (payload, os.path.basename(self.out.get())))
                elif kind == "fail":
                    self.finish(payload)
                else:
                    self.say(payload)
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

    # --- palette panel ---

    def rebuild_palette(self):
        for w in self.palette_inner.winfo_children():
            w.destroy()
        self.palette, self.swatch_widgets, self.swatch_rows = {}, {}, {}
        if not self.edited:
            return
        for i, (key, hexval) in enumerate(svg_edit.list_colours(self.edited.text)):
            self.palette[key] = hexval
            row = ttk.Frame(self.palette_inner)
            row.grid(row=i, column=0, sticky="w", pady=1)
            swatch = tk.Label(row, width=3, background=hexval, relief="raised",
                              borderwidth=2, cursor="hand2")
            swatch.grid(row=0, column=0)
            swatch.bind("<Button-1>", lambda e, k=key: self.swatch_click(k))
            swatch.bind("<Double-Button-1>", lambda e, k=key: self.swatch_recolour(k))
            ttk.Label(row, text=key if key.startswith("url(") else hexval,
                     width=12).grid(row=0, column=1, padx=4)
            hidden_var = tk.BooleanVar(value=key in self.hidden)
            ttk.Checkbutton(row, text="hide", variable=hidden_var,
                           command=lambda k=key, v=hidden_var: self.toggle_hidden(k, v)
                           ).grid(row=0, column=2)
            self.swatch_widgets[key] = swatch
            self.swatch_rows[key] = row
        self.highlight_rows()

    def highlight_rows(self):
        for key, w in self.swatch_widgets.items():
            w.configure(relief="sunken" if key == self.selected else "raised")

    def swatch_click(self, key):
        """A single click selects/deselects a swatch, isolating it in the preview.
        Recolouring moved to a double-click or the Recolour... button."""
        if self.merge_armed:
            self.merge_armed = False
            if key != self.merge_source:
                self.edited.apply(svg_edit.merge, self.merge_source, key)
                self.say("Merged %s into %s." % (self.merge_source, key))
                self.selected = None
                self.rebuild_palette()
                self.render_current()
            return
        self.selected = None if key == self.selected else key
        self.highlight_rows()
        self.render_current()

    def swatch_recolour(self, key):
        seed = self.palette.get(key, "#ffffff")
        _, hexval = colorchooser.askcolor(color=seed, title="Recolour")
        if not hexval:
            return
        self.edited.apply(svg_edit.recolour, key, hexval.upper())
        self.selected = None
        self.say("Recoloured %s to %s." % (key, hexval.upper()))
        self.rebuild_palette()
        self.render_current()

    def start_recolour(self):
        if not self.selected:
            return self.say("Click a swatch first, then Recolour...")
        self.swatch_recolour(self.selected)

    def start_merge(self):
        if not self.selected:
            return self.say("Click a swatch first, then Merge into...")
        self.merge_armed = True
        self.merge_source = self.selected
        self.say("Click the colour to merge %s into." % self.selected)

    def show_all(self):
        """Drop the isolated-swatch preview and show every layer again."""
        self.selected = None
        self.highlight_rows()
        self.render_current()

    def toggle_hidden(self, key, var):
        (self.hidden.add if var.get() else self.hidden.discard)(key)
        self.render_current()

    def do_undo(self):
        if not self.edited:
            return
        self.edited.undo()
        self.selected = None                      # the undone keys may no longer exist
        self.rebuild_palette()
        self.render_current()
        self.say("Undid last edit.")

    def do_save(self):
        self._write_svg(self.out.get().strip())

    def do_save_as(self):
        p = filedialog.asksaveasfilename(defaultextension=".svg",
                                         filetypes=[("SVG", "*.svg")])
        if p:
            self._write_svg(p)

    def _write_svg(self, path):
        if not self.edited or not path:
            return self.say("Nothing to save yet.")
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.edited.text)             # hidden layers never touched this text
        self.out.set(path)
        self.say("Saved %s" % os.path.basename(path))

    # --- compare view ---

    def render_current(self):
        """Re-render the SVG for the preview only: the selected swatch isolated
        (svg_edit.highlight), or otherwise the normal layers minus any hidden ones."""
        if not self.edited:
            return
        if self.selected:
            text = svg_edit.highlight(self.edited.text, self.selected)
        else:
            text = self.edited.text
            if self.hidden:
                text = svg_edit.hide_set(text, self.hidden)
        try:
            import cairosvg
        except ImportError:
            self.render_img = None
            self.redraw_compare()
            return
        if self.src_img is None:
            return
        w, h = self.src_img.size
        png_bytes = cairosvg.svg2png(bytestring=text.encode("utf-8"),
                                     output_width=w, output_height=h)
        im = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        flat = Image.new("RGBA", im.size, (255, 255, 255, 255))
        flat.alpha_composite(im)
        self.render_img = flat.convert("RGB")
        self.redraw_compare()

    def _zoom_level(self):
        """0 for Fit, else the spinbox's 1..8."""
        v = self.cmp_zoom.get()
        return 0 if v == "Fit" else int(v)

    def _view_geometry(self, size):
        """The (x0, y0, x1, y1) source-pixel box the compare view shows, and the
        scale (displayed px per source px) it's drawn at. At Fit that's the whole
        image, scaled to fit the canvas; otherwise a CMP_VIEW/zoom window centred
        on self.center."""
        w, h = size
        z = self._zoom_level()
        if z == 0:
            return 0, 0, w, h, min(CMP_VIEW / w, CMP_VIEW / h)
        if self.center is None:
            self.center = (w / 2, h / 2)
        view = CMP_VIEW / z
        x0 = min(max(self.center[0] - view / 2, 0), max(0, w - view))
        y0 = min(max(self.center[1] - view / 2, 0), max(0, h - view))
        return x0, y0, min(w, x0 + view), min(h, y0 + view), z

    def on_compare_click(self, event, which):
        """A plain click centres both views; at Fit (nothing to centre) or with
        Ctrl held, it is an eyedropper instead: pick the swatch nearest the
        clicked pixel."""
        if self.src_img is None:
            return
        x0, y0, _, _, scale = self._view_geometry(self.src_img.size)
        px, py = x0 + event.x / scale, y0 + event.y / scale
        eyedrop = self._zoom_level() == 0 or bool(event.state & 0x4)   # 0x4 = Control
        if eyedrop:
            img = self.src_img if which == "src" else self.render_img
            if img is None:
                return
            x = min(max(int(px), 0), img.width - 1)
            y = min(max(int(py), 0), img.height - 1)
            self.pick_swatch_near(img.getpixel((x, y)))
            return
        self.center = (px, py)
        self.redraw_compare()

    def pick_swatch_near(self, rgb):
        """Select the palette swatch closest to `rgb`: an exact match first, else
        the nearest by max-channel distance. A gradient swatch only counts if its
        first stop is within 8 of `rgb` - it stands for a whole gradient, not one
        colour, so a loose match to it would be misleading."""
        best_key, best_d = None, None
        for key, hexval in self.palette.items():
            c = tuple(int(hexval[i:i + 2], 16) for i in (1, 3, 5))
            d = max(abs(a - b) for a, b in zip(rgb[:3], c))
            if key.startswith("url(#") and d > 8:
                continue
            if d == 0:
                best_key, best_d = key, 0
                break
            if best_d is None or d < best_d:
                best_key, best_d = key, d
        if best_key is None:
            return
        self.selected = best_key
        self.highlight_rows()
        self.scroll_to(best_key)
        self.render_current()

    def scroll_to(self, key):
        row = self.swatch_rows.get(key)
        if not row:
            return
        self.palette_canvas.update_idletasks()
        bbox = self.palette_canvas.bbox("all")
        if not bbox or bbox[3] <= bbox[1]:
            return
        self.palette_canvas.yview_moveto((row.winfo_y() - bbox[1]) / (bbox[3] - bbox[1]))

    def redraw_compare(self):
        if self.src_img is None:
            self.src_canvas.delete("all")
            self.out_canvas.delete("all")
            return
        x0, y0, x1, y1, scale = self._view_geometry(self.src_img.size)
        box = (int(x0), int(y0), max(int(x0) + 1, round(x1)), max(int(y0) + 1, round(y1)))
        resample = Image.NEAREST if scale >= 1 else Image.LANCZOS
        for img, canvas, attr in ((self.src_img, self.src_canvas, "_src_photo"),
                                  (self.render_img, self.out_canvas, "_out_photo")):
            canvas.delete("all")
            if img is None:
                canvas.create_text(CMP_VIEW // 2, CMP_VIEW // 2, text="(needs cairosvg)")
                continue
            crop = img.crop(box)
            disp = crop.resize((max(1, round(crop.width * scale)),
                                max(1, round(crop.height * scale))), resample)
            photo = ImageTk.PhotoImage(disp)
            setattr(self, attr, photo)            # keep a ref, else Tk drops the image
            canvas.create_image(0, 0, anchor="nw", image=photo)


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        svg_edit._selfcheck()
    else:
        root = tk.Tk()
        root.title("PNG to SVG")
        root.minsize(1000, 700)
        root.geometry("1000x700")
        App(root)
        root.mainloop()
