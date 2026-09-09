"""Plain-text edits to a traced SVG's <path> elements.

png2svg.py's trace() and vtracer emit one <path .../> element per shape, a
`fill` attribute somewhere in the tag holding either a hex colour or a
`url(#gN)` reference into a <linearGradient> in <defs> whose first <stop>
carries the swatch colour png2svg_app.py shows. That format is regular
enough that every edit here is a plain string/regex operation - no XML
library needed, and none of this cares whether `fill` comes before or after
a path's other attributes (VTracer's own output puts `d` first).
"""
import re

_PATH = re.compile(r"<path\b[^>]*/>")
_FILL = re.compile(r'fill="([^"]*)"')
_VIS = re.compile(r'\s*visibility="hidden"')
_GRADIENT = re.compile(r'<linearGradient\s+id="([^"]+)"[^>]*>.*?</linearGradient>', re.S)
_STOP0 = re.compile(r'<stop\s+offset="0"[^>]*?stop-color="([^"]+)"')


def _gradient_first_stop(svg, gid):
    """The stop-color of <linearGradient id="gid">'s first <stop>, or None."""
    for m in _GRADIENT.finditer(svg):
        if m.group(1) == gid:
            s = _STOP0.search(m.group(0))
            return s.group(1) if s else None
    return None


def list_colours(svg):
    """[(key, swatch_hex), ...], one per distinct fill, in file order.

    `key` is exactly what sits in a path's fill="..." (a hex colour, or
    "url(#gN)"); `swatch_hex` is always a plain hex colour to paint a
    swatch with, resolved from the gradient's first stop when `key` is a
    gradient reference.
    """
    seen = {}
    for m in _PATH.finditer(svg):
        fm = _FILL.search(m.group(0))
        if not fm:
            continue
        key = fm.group(1)
        if key in seen:
            continue
        if key.startswith("url(#"):
            seen[key] = _gradient_first_stop(svg, key[5:-1]) or "#000000"
        else:
            seen[key] = key
    return list(seen.items())


def recolour(svg, key, new_hex):
    """Point every path filled with `key` at `new_hex`.

    A gradient reference is recoloured by retargeting its first stop (the
    one the swatch is drawn from) so every path using it moves together;
    the second stop is left alone.
    """
    if key.startswith("url(#"):
        gid = key[5:-1]

        def sub(m):
            if m.group(1) != gid:
                return m.group(0)
            return _STOP0.sub(lambda s: s.group(0).replace(s.group(1), new_hex),
                               m.group(0), count=1)
        return _GRADIENT.sub(sub, svg)
    return svg.replace('fill="%s"' % key, 'fill="%s"' % new_hex)


def merge(svg, src_key, dst_key):
    """Every path filled with `src_key` takes `dst_key`'s fill instead.

    Paths are kept, never dropped: a lower layer reaches half a pixel under
    its neighbours (see trace()'s layer stacking), so deleting one would
    open a seam along that border.
    """
    return svg.replace('fill="%s"' % src_key, 'fill="%s"' % dst_key)


def set_hidden(svg, key, hidden):
    """Add or remove visibility="hidden" on every path filled with `key`.

    Meant for a preview-only copy of the text - never call this on the text
    that gets saved.
    """
    def sub(m):
        tag = m.group(0)
        fm = _FILL.search(tag)
        if not fm or fm.group(1) != key:
            return tag
        tag = _VIS.sub("", tag)
        if hidden:
            tag = tag[:-2] + ' visibility="hidden"/>'
        return tag
    return _PATH.sub(sub, svg)


def hide_set(svg, keys):
    """set_hidden(), applied for every key in `keys`."""
    for key in keys:
        svg = set_hidden(svg, key, True)
    return svg


class Edited:
    """An SVG's text plus one level of undo over edits made to it."""

    def __init__(self, text):
        self.text = text
        self._prev = None

    def apply(self, fn, *args):
        """Run an edit (recolour/merge) and remember what to undo to."""
        self._prev = self.text
        self.text = fn(self.text, *args)
        return self.text

    def undo(self):
        """Restore the text from before the last apply(); a no-op twice in a row."""
        if self._prev is not None:
            self.text, self._prev = self._prev, None
        return self.text


def _selfcheck():
    svg = ('<svg><path fill="#FF0000" d="M0 0Z"/>'
           '<path fill="#00FF00" d="M1 1Z"/>'
           '<path fill="#FF0000" d="M2 2Z"/>'
           '<defs><linearGradient id="g0" x1="0" y1="0" x2="1" y2="1">'
           '<stop offset="0" stop-color="#0000FF"/>'
           '<stop offset="1" stop-color="#000099"/></linearGradient></defs>'
           '<path fill="url(#g0)" d="M3 3Z"/>'
           '<path d="M4 4Z" fill="#00FF00" transform="translate(1,1)"/></svg>')

    cols = list_colours(svg)
    assert cols == [("#FF0000", "#FF0000"), ("#00FF00", "#00FF00"),
                     ("url(#g0)", "#0000FF")], cols

    # recolour: exactly the target fills change, no path gained or lost
    out = recolour(svg, "#FF0000", "#ABCDEF")
    assert out.count('fill="#ABCDEF"') == 2 and 'fill="#FF0000"' not in out
    assert out.count("<path") == svg.count("<path") == 5

    outg = recolour(svg, "url(#g0)", "#123456")
    assert '<stop offset="0" stop-color="#123456"/>' in outg
    assert '<stop offset="1" stop-color="#000099"/>' in outg  # other stop untouched
    assert outg.count("<path") == 5

    # merge: attribute-order-agnostic, and every path survives
    merged = merge(svg, "#00FF00", "#FF0000")
    assert merged.count("<path") == svg.count("<path")
    assert 'fill="#00FF00"' not in merged and merged.count('fill="#FF0000"') == 4

    # hide adds visibility="hidden" to the right paths only; show strips it
    hidden = set_hidden(svg, "#FF0000", True)
    assert hidden.count('visibility="hidden"') == 2
    assert '<path fill="#00FF00" d="M1 1Z" visibility="hidden"/>' not in hidden
    shown = set_hidden(hidden, "#FF0000", False)
    assert 'visibility="hidden"' not in shown and shown == svg

    # undo restores the text from before the last edit, one level
    ed = Edited(svg)
    ed.apply(recolour, "#FF0000", "#ABCDEF")
    assert ed.text == out
    assert ed.undo() == svg and ed.text == svg
    assert ed.undo() == svg                      # a second undo is a no-op

    print("svg_edit selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
