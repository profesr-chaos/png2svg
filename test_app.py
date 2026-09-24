"""python test_app.py -- the saving text the app shows after a compression."""
from png2svg_app import _saving, _size

assert _size(0) == "0 B"
assert _size(1023) == "1023 B"
assert _size(1024) == "1.0 KB"
assert _size(int(1.5 * 1024 * 1024)) == "1.5 MB"
assert _saving(16691, 15360) == "Saved 1.3 KB (8%): 16.3 KB -> 15.0 KB"
assert _saving(1200, 1240) == "Grew 40 B (3%): 1.2 KB -> 1.2 KB"      # tiny .svgz
assert _saving(500, 500) == "Saved 0 B (0%): 500 B -> 500 B"

# Browse an SVG, then Compress, through the real Tk callbacks (needs cairosvg + scour).
import os, tempfile, time, tkinter as tk
from tkinter import filedialog
import png2svg_app

d = tempfile.mkdtemp()
svg = os.path.join(d, "a.svg")
open(svg, "w").write('<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">'
                     '<!-- note --><rect width="20" height="10" fill="#f00"/></svg>')
root = tk.Tk()
root.withdraw()
errors = []
root.report_callback_exception = lambda *e: errors.append(e[1])   # Tk hides these by default
app = png2svg_app.App(root)

filedialog.askopenfilename = lambda **kw: svg
app.pick_src()                                  # crashed: called a removed self.show()
assert app.mode.get() == "compress" and app.out.get().endswith("a.min.svg")
assert app.src_img is not None, "source preview not loaded"

seen = {}
filedialog.asksaveasfilename = lambda **kw: seen.update(kw) or ""
app.pick_out()                                  # opens at the current output
assert seen["initialfile"] == "a.min.svg" and seen["initialdir"] == d

app.start()
t = time.time()
while app.busy and time.time() - t < 30:
    root.update()
    time.sleep(0.02)
assert not errors, errors                       # poll() read an unbound `text` on "done"
assert not app.busy, "compress never finished"
status = app.status.cget("text")
assert status.startswith("Saved") and status.endswith("wrote a.min.svg"), status
assert app.render_img is not None, "preview never reached the compare view"
root.destroy()
print("app check ok")
