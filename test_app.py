"""python test_app.py -- the saving text the app shows after a compression."""
from png2svg_app import _saving, _size

assert _size(0) == "0 B"
assert _size(1023) == "1023 B"
assert _size(1024) == "1.0 KB"
assert _size(int(1.5 * 1024 * 1024)) == "1.5 MB"
assert _saving(16691, 15360) == "Saved 1.3 KB (8%): 16.3 KB -> 15.0 KB"
assert _saving(1200, 1240) == "Grew 40 B (3%): 1.2 KB -> 1.2 KB"      # tiny .svgz
assert _saving(500, 500) == "Saved 0 B (0%): 500 B -> 500 B"
print("app check ok")
