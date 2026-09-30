"""python build.py -- make dist/png2svg.exe: the GUI as one file, no console.

    pip install pyinstaller
    scoop install potrace          # bundled: ~15x faster traces

cairo also comes from the PATH: cairocffi's PyInstaller hook bundles the
libcairo-2.dll it finds there (Tesseract and GTK installs have one).
"""
import io
import os
import re
import shutil

import cairosvg
import PyInstaller.__main__
from PIL import Image

# icon.ico holds every size Windows asks for, from the one logo
png = cairosvg.svg2png(url="icon.svg", output_width=256, output_height=256)
Image.open(io.BytesIO(png)).save("icon.ico", sizes=[(n, n) for n in (16, 24, 32, 48, 64, 128, 256)])

potrace = shutil.which("potrace")
assert potrace, "potrace not on the PATH: scoop install potrace"
shim = os.path.splitext(potrace)[0] + ".shim"     # scoop puts a stub exe on the PATH
if os.path.exists(shim):
    potrace = re.search(r'path = "(.+)"', open(shim).read()).group(1)

PyInstaller.__main__.run([
    "png2svg_app.py", "--name", "png2svg", "--onefile", "--windowed", "--noconfirm",
    "--icon", "icon.ico",
    "--add-data", "icon.ico" + os.pathsep + ".",
    "--add-binary", potrace + os.pathsep + ".",
])
