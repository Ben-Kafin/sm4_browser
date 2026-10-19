"""
qtplot: the drawing and window layer of the package on PySide6 + pyqtgraph.

The package's windows were written against matplotlib's pyplot: figures laid
out in figure fractions, axes, artists returned by plot / imshow / axhline,
mpl-style mouse events (button_press_event, motion_notify_event, ...,
pick_event) and colour maps by name.  This module provides that interface
(the subset the package uses) on Qt, so every window keeps its layout, its
math, and its interaction while all drawing, events, and widgets are
PySide6 / pyqtgraph:

  figure, subplots, show, close, fignum_exists, gcf, pause, rcParams,
  get_cmap                                   the pyplot functions
  Figure, FigureCanvas, Axes                 a figure is a Qt window with one
                                             pyqtgraph scene; every axes is a
                                             bare ViewBox placed exactly at its
                                             figure-fraction rectangle, its
                                             tick axes outside it (as in
                                             matplotlib)
  gridspec.GridSpec, GridSpecFromSubplotSpec the subplot grids
  Normalize, PowerNorm, TwoSlopeNorm         colour normalizations
  Colormap, LinearSegmentedColormap,         colour maps; to_cmap(c) turns any
  ListedColormap, to_cmap                    matplotlib colour-map name, any
                                             matplotlib Colormap object, or a
                                             pyqtgraph ColorMap into one
  LineCollection, PolyCollection,            collections and transforms used
  Affine2D, ScalarMappable                   by plots.py

Speed: images are pyqtgraph ImageItems coloured through a lookup table (a
new gamma or colour limit only rebuilds the table), quad meshes are filled
as one path per colour and cached as a pixmap, many curves of one colour are
one item, and mouse motion is coalesced (a slow redraw never queues up a
backlog of motion events).

Colour maps: wherever the package takes a colour map (the if-main inputs
cmap / cmap_map, line_window_kwargs['cmap'], ...), a matplotlib name
('jet', 'viridis', 'RdBu_r', ...), a matplotlib Colormap object, or a
pyqtgraph ColorMap may be given; to_cmap converts it once (cached) into the
lookup tables pyqtgraph draws with.  matplotlib, when installed, is used
only to read the colour values of a map given by name or object; it never
draws.
"""

import os
import sys
import math
import html
import itertools
import traceback
import types

import numpy as np



def _choose_binding():
    """
    The Qt binding to draw with: the one already imported in this Python
    session when there is one (Spyder's IPython console runs its Qt event
    loop with PyQt5 or PyQt6, and IPython refuses a second binding), else
    the one named by PYQTGRAPH_QT_LIB, else PySide6 (then PyQt6, PyQt5,
    PySide2, whichever is installed).
    """
    for lib in ('PySide6', 'PyQt6', 'PyQt5', 'PySide2'):
        if lib in sys.modules:
            return lib
    if os.environ.get('PYQTGRAPH_QT_LIB'):
        return os.environ['PYQTGRAPH_QT_LIB']
    import importlib.util
    for lib in ('PySide6', 'PyQt6', 'PyQt5', 'PySide2'):
        if importlib.util.find_spec(lib) is not None:
            return lib
    return 'PySide6'


os.environ['PYQTGRAPH_QT_LIB'] = _choose_binding()
import pyqtgraph as pg                                          # noqa: E402
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets               # noqa: E402

Qt = QtCore.Qt

# A native crash or a hang inside Qt kills the kernel without a Python message; faulthandler
# writes the Python stack of such a crash to this file (and to the console when it can).
CRASH_LOG = os.path.join(os.path.expanduser('~'), 'sm4_browser_crash.log')
try:
    import faulthandler
    _crash_file = open(CRASH_LOG, 'w')
    faulthandler.enable(file=_crash_file, all_threads=True)
except Exception:
    _crash_file = None

def log_note(text):
    """A line in the crash log (the step the program was at when Qt failed)."""
    if _crash_file is not None:
        try:
            _crash_file.write(str(text) + "\n")
            _crash_file.flush()
        except Exception:
            pass


class watchdog:
    """
    with qtplot.watchdog('what'): ...  -- while the block runs longer than `after` seconds, the
    Python stack of every thread is written every `after` seconds to the console and to CRASH_LOG,
    so a step that hangs on a file names itself.
    """

    def __init__(self, what, after=15.0):
        self.what, self.after = what, float(after)

    _stack = []

    @staticmethod
    def _arm(after):
        try:
            import faulthandler
            out = sys.__stderr__ if sys.__stderr__ is not None else _crash_file
            if out is not None:
                faulthandler.dump_traceback_later(after, repeat=True, file=out)
        except Exception:
            pass

    def __enter__(self):
        log_note(self.what)
        watchdog._stack.append(self)
        self._arm(self.after)
        return self

    def __exit__(self, *exc):
        try:
            import faulthandler
            faulthandler.cancel_dump_traceback_later()
        except Exception:
            pass
        if watchdog._stack and watchdog._stack[-1] is self:
            watchdog._stack.pop()
        if watchdog._stack:                         # the enclosing block is watched again
            log_note(watchdog._stack[-1].what)
            self._arm(watchdog._stack[-1].after)
        return False


_LIMIT = 1e15            # coordinates beyond this (or not finite) are never handed to Qt's painter


def _finite_range(a, b):
    """A drawable range from (a, b): None when not finite; widened when (nearly) empty."""
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(a) and np.isfinite(b)) or max(abs(a), abs(b)) > _LIMIT:
        return None
    if b < a:
        a, b = b, a
    c = 0.5 * (a + b)
    min_span = max(abs(c) * 1e-9, 1e-12)
    if b - a < min_span:
        a, b = c - min_span, c + min_span
    return a, b


def _evpos(ev):
    """A mouse or wheel event's position in widget pixels (Qt 6: position(); Qt 5: pos() / posF())."""
    if hasattr(ev, 'position'):
        return QtCore.QPointF(ev.position())
    if hasattr(ev, 'posF'):
        return QtCore.QPointF(ev.posF())
    return QtCore.QPointF(ev.pos())


def _check_qt_binding():
    """
    PySide6 6.12.0 releases one reference to None on every Qt call that
    returns nothing; before Python 3.12 (None not yet immortal) a painting
    window then aborts with 'Fatal Python error: none_dealloc' after a few
    thousand calls.  Measured here once at import; a warning names the cure.
    """
    if sys.version_info >= (3, 12) or pg.Qt.QT_LIB != 'PySide6':
        return
    try:
        p = QtGui.QPainterPath()
        before = sys.getrefcount(None)
        for _ in range(64):
            p.setFillRule(Qt.FillRule.WindingFill)
        lost = before - sys.getrefcount(None)
    except Exception:
        return
    if lost >= 32:
        import PySide6
        print(f"qtplot WARNING: this PySide6 ({PySide6.__version__}) loses a reference to None on every Qt call "
              f"under Python {sys.version_info.major}.{sys.version_info.minor}; the windows will crash with "
              "'Fatal Python error: none_dealloc' after a while.  Use Python 3.12 or newer, or "
              "pip install \"PySide6<6.12\".")


_check_qt_binding()

# ---------------------------------------------------------------------- #
#  Configuration                                                           #
# ---------------------------------------------------------------------- #

WINDOW_BG = '#f0f0f0'     # the window colour around the white panels
FONT_SCALE = 1.25         # every text (plots, legends, controls) this much larger than matplotlib's points
DPI = 100.0               # logical pixels per inch: figure sizes, widget sizes, and font points as matplotlib's
ANTIALIAS = True          # antialiased curves (False draws faster, with jagged lines)
pg.setConfigOptions(imageAxisOrder='row-major', background='w', foreground='k', antialias=ANTIALIAS)


def screen_scale():
    """
    How many times larger than 96 dpi the screen's logical resolution is (1 under Qt 6, whose
    high-DPI scaling already works in logical pixels; e.g. 1.5 for a 4K screen at 150 % under Qt 5
    without scaling): figure windows open this much larger, and their text follows (font_zoom).
    """
    try:
        scr = QtGui.QGuiApplication.primaryScreen()
        return max(1.0, float(scr.logicalDotsPerInch()) / 96.0) if scr is not None else 1.0
    except Exception:
        return 1.0


def pt2px(pt):
    """Points to logical pixels at DPI (matplotlib's point: 1/72 inch)."""
    return float(pt) * DPI / 72.0


_FONT_SIZES = {'xx-small': 5.79, 'x-small': 6.94, 'small': 8.33, 'medium': 10.0, 'large': 12.0,
               'x-large': 14.4, 'xx-large': 17.28, 'larger': 12.0, 'smaller': 8.33}


def font_pt(size, default=10.0):
    if size is None:
        return float(default)
    if isinstance(size, str):
        return _FONT_SIZES.get(size, float(default))
    return float(size)


_font_cache = {}


def qfont(size=10.0, family=None, weight=None, zoom=1.0):
    """A QFont of `size` points at DPI (cached), times FONT_SCALE and a figure's zoom (Figure.font_zoom)."""
    px = max(1, int(round(pt2px(font_pt(size)) * FONT_SCALE * float(zoom))))
    key = (px, family, weight)
    f = _font_cache.get(key)
    if f is None:
        if family in ('monospace', 'mono', 'courier'):
            f = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        else:
            f = QtGui.QFont()
            if family and family not in ('sans-serif', 'serif'):
                f.setFamily(family)
        f.setPixelSize(px)
        if weight in ('bold', 'heavy', 'semibold', 'demibold') or (isinstance(weight, (int, float)) and weight >= 600):
            f.setBold(True)
        _font_cache[key] = f
    return QtGui.QFont(f)


# ---------------------------------------------------------------------- #
#  Colours                                                                 #
# ---------------------------------------------------------------------- #

TAB10 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
         '#bcbd22', '#17becf']
_BASE = {'b': (0.0, 0.0, 1.0), 'g': (0.0, 0.5, 0.0), 'r': (1.0, 0.0, 0.0), 'c': (0.0, 0.75, 0.75),
         'm': (0.75, 0.0, 0.75), 'y': (0.75, 0.75, 0.0), 'k': (0.0, 0.0, 0.0), 'w': (1.0, 1.0, 1.0)}
_TAB_NAMES = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
_rgba_cache = {}


def to_rgba(c, alpha=None):
    """A matplotlib colour specification as an (r, g, b, a) tuple of floats (alpha overrides when given)."""
    if isinstance(c, str):
        key = (c, alpha)
        hit = _rgba_cache.get(key)
        if hit is not None:
            return hit
        s = c.strip().lower()
        if s in ('none', ''):
            rgba = (0.0, 0.0, 0.0, 0.0)
        elif s in _BASE:
            rgba = _BASE[s] + (1.0,)
        elif s[:1] == 'c' and s[1:].isdigit():
            rgba = to_rgba(TAB10[int(s[1:]) % len(TAB10)])
        elif s.startswith('tab:') and s[4:] in _TAB_NAMES:
            rgba = to_rgba(TAB10[_TAB_NAMES.index(s[4:])])
        elif s.startswith('#'):
            h = s[1:]
            if len(h) in (3, 4):
                h = ''.join(ch * 2 for ch in h)
            vals = [int(h[i:i + 2], 16) / 255.0 for i in range(0, len(h), 2)]
            rgba = tuple(vals) + ((1.0,) if len(vals) == 3 else ())
        else:
            try:
                v = float(s)
                if not 0.0 <= v <= 1.0:
                    raise ValueError
                rgba = (v, v, v, 1.0)
            except ValueError:
                q = QtGui.QColor(s.replace(' ', ''))
                if not q.isValid():
                    raise ValueError(f"unknown colour {c!r}")
                rgba = q.getRgbF()
        if alpha is not None:
            rgba = tuple(rgba[:3]) + (float(alpha),)
        rgba = tuple(float(v) for v in rgba)
        _rgba_cache[key] = rgba
        return rgba
    if isinstance(c, QtGui.QColor):
        rgba = c.getRgbF()
    else:
        a = [float(v) for v in np.asarray(c, dtype=np.float64).ravel()]
        if max(a[:3]) > 1.0:
            a = [v / 255.0 for v in a]
        rgba = tuple(a[:3]) + ((a[3],) if len(a) > 3 else (1.0,))
    if alpha is not None:
        rgba = tuple(rgba[:3]) + (float(alpha),)
    return tuple(float(v) for v in rgba)


def qcolor(c, alpha=None):
    r, g, b, a = to_rgba(c, alpha)
    return QtGui.QColor.fromRgbF(min(max(r, 0.0), 1.0), min(max(g, 0.0), 1.0), min(max(b, 0.0), 1.0),
                                 min(max(a, 0.0), 1.0))


_LINESTYLES = {'-': Qt.PenStyle.SolidLine, 'solid': Qt.PenStyle.SolidLine,
               '--': Qt.PenStyle.DashLine, 'dashed': Qt.PenStyle.DashLine,
               ':': Qt.PenStyle.DotLine, 'dotted': Qt.PenStyle.DotLine,
               '-.': Qt.PenStyle.DashDotLine, 'dashdot': Qt.PenStyle.DashDotLine}


def mkpen(color, lw=1.5, ls='-', alpha=None):
    """A cosmetic QPen of `lw` points, matplotlib line style `ls`; None for no line."""
    if ls in (None, 'none', 'None', ' ', '') or color is None:
        return None
    if isinstance(color, str) and color.strip().lower() == 'none':
        return None
    pen = QtGui.QPen(qcolor(color, alpha))
    pen.setWidthF(max(pt2px(lw), 0.5))
    pen.setCosmetic(True)
    pen.setStyle(_LINESTYLES.get(ls, Qt.PenStyle.SolidLine))
    if pen.style() != Qt.PenStyle.SolidLine:
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
    return pen


def mkbrush(color, alpha=None):
    if color is None or (isinstance(color, str) and color.strip().lower() == 'none'):
        return None
    return QtGui.QBrush(qcolor(color, alpha))


# ---------------------------------------------------------------------- #
#  Normalizations                                                          #
# ---------------------------------------------------------------------- #

class Normalize:
    """Linear map of [vmin, vmax] onto [0, 1] (matplotlib.colors.Normalize); NaN stays NaN."""

    def __init__(self, vmin=None, vmax=None, clip=False):
        self.vmin = None if vmin is None else float(vmin)
        self.vmax = None if vmax is None else float(vmax)
        self.clip = bool(clip)

    def autoscale_None(self, A):
        a = np.asarray(np.ma.filled(np.ma.asarray(A, dtype=np.float64), np.nan), dtype=np.float64)
        ok = np.isfinite(a)
        if self.vmin is None:
            self.vmin = float(np.min(a[ok])) if ok.any() else 0.0
        if self.vmax is None:
            self.vmax = float(np.max(a[ok])) if ok.any() else 1.0

    def scaled(self):
        return self.vmin is not None and self.vmax is not None

    def _lin(self, value):
        a = np.asarray(np.ma.filled(np.ma.asarray(value, dtype=np.float64), np.nan), dtype=np.float64)
        if not self.scaled():
            self.autoscale_None(a)
        d = self.vmax - self.vmin
        out = (a - self.vmin) / (d if d != 0 else 1.0)
        if self.clip:
            out = np.clip(out, 0.0, 1.0)
        return out

    def __call__(self, value):
        return self._lin(value)

    def inverse(self, t):
        t = np.asarray(t, dtype=np.float64)
        return self.vmin + t * (self.vmax - self.vmin)


class PowerNorm(Normalize):
    """((x - vmin) / (vmax - vmin)) ** gamma, values below vmin at 0 (matplotlib.colors.PowerNorm)."""

    def __init__(self, gamma, vmin=None, vmax=None, clip=False):
        super().__init__(vmin, vmax, clip)
        self.gamma = float(gamma)

    def __call__(self, value):
        t = self._lin(value)
        with np.errstate(invalid='ignore'):
            t = np.where(t < 0, 0.0, t)
            return np.power(t, self.gamma)

    def inverse(self, t):
        t = np.asarray(t, dtype=np.float64)
        return self.vmin + np.power(np.clip(t, 0.0, None), 1.0 / self.gamma) * (self.vmax - self.vmin)


class TwoSlopeNorm(Normalize):
    """Two linear pieces: vmin -> 0, vcenter -> 0.5, vmax -> 1 (matplotlib.colors.TwoSlopeNorm)."""

    def __init__(self, vcenter, vmin=None, vmax=None):
        super().__init__(vmin, vmax)
        self.vcenter = float(vcenter)

    def __call__(self, value):
        a = np.asarray(np.ma.filled(np.ma.asarray(value, dtype=np.float64), np.nan), dtype=np.float64)
        if not self.scaled():
            self.autoscale_None(a)
        lo, c, hi = self.vmin, self.vcenter, self.vmax
        out = np.where(a < c, 0.5 * (a - lo) / ((c - lo) or 1.0), 0.5 + 0.5 * (a - c) / ((hi - c) or 1.0))
        return np.where(np.isfinite(a), out, np.nan)

    def inverse(self, t):
        t = np.asarray(t, dtype=np.float64)
        lo, c, hi = self.vmin, self.vcenter, self.vmax
        return np.where(t < 0.5, lo + 2 * t * (c - lo), c + 2 * (t - 0.5) * (hi - c))


# ---------------------------------------------------------------------- #
#  Colour maps                                                             #
# ---------------------------------------------------------------------- #

class Colormap:
    """
    A colour map as matplotlib uses one: called with values in [0, 1] it
    gives RGBA floats (bytes=True: uint8), NaN giving the transparent 'bad'
    colour; lut(n) is the uint8 lookup table pyqtgraph draws with.
    """

    def __init__(self, name, table):
        self.name = str(name)
        self._table = np.clip(np.asarray(table, dtype=np.float64).reshape(-1, 4), 0.0, 1.0)
        self.N = len(self._table)
        self._lut_cache = {}

    def __call__(self, X, alpha=None, bytes=False):
        scalar = np.ndim(X) == 0
        x = np.asarray(np.ma.filled(np.ma.asarray(X, dtype=np.float64), np.nan), dtype=np.float64)
        bad = ~np.isfinite(x)
        idx = np.clip(np.floor(np.where(bad, 0.0, x) * self.N), 0, self.N - 1).astype(np.intp)
        out = self._table[idx]
        if alpha is not None:
            out = out.copy()
            out[..., 3] = float(alpha)
        if bad.any():
            out = out.copy()
            out[bad] = 0.0
        if bytes:
            out = (out * 255.0 + 0.5).astype(np.uint8)
        if scalar:
            return tuple(out.reshape(4))
        return out

    def lut(self, n=256):
        """The (n, 4) uint8 lookup table sampled uniformly over [0, 1]."""
        t = self._lut_cache.get(n)
        if t is None:
            t = self((np.arange(n) + 0.5) / n, bytes=True)
            self._lut_cache[n] = t
        return t

    def resampled(self, n):
        return Colormap(self.name, self((np.arange(n) + 0.5) / n))

    def reversed(self):
        return Colormap(self.name + '_r', self._table[::-1])

    def to_pg(self):
        """The same map as a pyqtgraph ColorMap."""
        return pg.ColorMap(np.linspace(0.0, 1.0, self.N), (self._table * 255).astype(np.uint8))

    def __repr__(self):
        return f"<qtplot.Colormap {self.name} ({self.N} colours)>"


class LinearSegmentedColormap(Colormap):
    @staticmethod
    def from_list(name, colors, N=256):
        """Linear interpolation between colours (or (position, colour) pairs), matplotlib's from_list."""
        cols = list(colors)

        def is_pair(c):
            return (isinstance(c, (tuple, list)) and len(c) == 2 and isinstance(c[0], (int, float, np.floating))
                    and not isinstance(c[1], (int, float, np.floating)))
        if cols and all(is_pair(c) for c in cols):
            pos = np.array([float(c[0]) for c in cols], dtype=np.float64)
            rgba = np.array([to_rgba(c[1]) for c in cols])
        else:
            pos = np.linspace(0.0, 1.0, len(cols))
            rgba = np.array([to_rgba(c) for c in cols])
        x = np.linspace(0.0, 1.0, int(N))
        table = np.column_stack([np.interp(x, pos, rgba[:, k]) for k in range(4)])
        return LinearSegmentedColormap(name, table)


class ListedColormap(Colormap):
    def __init__(self, colors, name='from_list'):
        super().__init__(name, np.array([to_rgba(c) for c in colors]))


def _segment_table(data, N=256):
    x = np.linspace(0.0, 1.0, N)
    cols = []
    for key in ('red', 'green', 'blue'):
        seg = np.asarray(data[key], dtype=np.float64)
        cols.append(np.interp(x, seg[:, 0], seg[:, 1]))
    return np.column_stack(cols + [np.ones(N)])


_JET = {'red': ((0., 0, 0), (0.35, 0, 0), (0.66, 1, 1), (0.89, 1, 1), (1, 0.5, 0.5)),
        'green': ((0., 0, 0), (0.125, 0, 0), (0.375, 1, 1), (0.64, 1, 1), (0.91, 0, 0), (1, 0, 0)),
        'blue': ((0., 0.5, 0.5), (0.11, 1, 1), (0.34, 1, 1), (0.65, 0, 0), (1, 0, 0))}
_cmap_cache = {}


def _builtin_cmap(name):
    """The maps known without matplotlib: jet, gray / grey, and pyqtgraph's own (viridis, magma, ...)."""
    base, rev = (name[:-2], True) if name.endswith('_r') else (name, False)
    if base == 'jet':
        cm = Colormap('jet', _segment_table(_JET))
    elif base in ('gray', 'grey', 'Greys_r'):
        x = np.linspace(0, 1, 256)
        cm = Colormap(base, np.column_stack([x, x, x, np.ones(256)]))
    else:
        pgm = pg.colormap.get(base)
        cm = Colormap(base, pgm.getLookupTable(0.0, 1.0, 256, alpha=True) / 255.0)
    return cm.reversed() if rev else cm


def to_cmap(c):
    """
    Any colour-map specification as a Colormap: a qtplot Colormap as it is;
    a matplotlib colour-map name or Colormap object (its colours read through
    matplotlib when installed); a pyqtgraph ColorMap; a list of colours.  None
    gives matplotlib's default, viridis.  Names are converted once (cached).
    """
    if c is None:
        c = 'viridis'
    if isinstance(c, Colormap):
        return c
    if isinstance(c, pg.ColorMap):
        return Colormap(getattr(c, 'name', 'pyqtgraph') or 'pyqtgraph', c.getLookupTable(0.0, 1.0, 256, alpha=True) / 255.0)
    if isinstance(c, str):
        hit = _cmap_cache.get(c)
        if hit is not None:
            return hit
        cm = None
        try:
            import matplotlib
            try:
                mcm = matplotlib.colormaps[c]
            except AttributeError:                      # matplotlib < 3.5
                from matplotlib import cm as _cm
                mcm = _cm.get_cmap(c)
            cm = Colormap(c, mcm(np.linspace(0.0, 1.0, max(int(getattr(mcm, 'N', 256)), 2))))
        except ImportError:
            pass
        except (KeyError, ValueError):
            pass
        if cm is None:
            try:
                cm = _builtin_cmap(c)
            except Exception as exc:
                raise ValueError(f"unknown colour map {c!r} (install matplotlib to use every matplotlib name)") from exc
        _cmap_cache[c] = cm
        return cm
    if callable(c) and hasattr(c, 'N'):               # a matplotlib Colormap object
        return Colormap(getattr(c, 'name', 'matplotlib'), c(np.linspace(0.0, 1.0, max(int(c.N), 2))))
    if isinstance(c, (list, tuple)):
        return ListedColormap(c)
    raise ValueError(f"not a colour map: {c!r}")


get_cmap = to_cmap


def norm_lut(cmap, norm, vmin, vmax, n=1024):
    """
    The lookup table that colours data on [vmin, vmax] (pyqtgraph levels)
    through a normalization and a colour map: bin i (data values from
    vmin + i d to vmin + (i + 1) d) takes cmap(norm(its centre)).  A gamma
    or a new norm only rebuilds this table; the image data stay as they are.
    """
    cmap = to_cmap(cmap)
    vmin, vmax = float(vmin), float(vmax)
    if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmax <= vmin:
        vmax = vmin + 1.0
    if type(norm) is Normalize or norm is None:
        return cmap.lut(n)
    centres = vmin + (np.arange(n) + 0.5) * (vmax - vmin) / n
    return cmap(np.asarray(norm(centres), dtype=np.float64), bytes=True)


class ScalarMappable:
    """A norm and a colour map with an array (the colour bar's source), matplotlib's ScalarMappable."""

    def __init__(self, norm=None, cmap=None):
        self.norm = norm if norm is not None else Normalize()
        self.cmap = to_cmap(cmap)
        self._A = None
        self.colorbar = None

    def set_array(self, A):
        self._A = np.asarray(np.ma.filled(np.ma.asarray(A, dtype=np.float64), np.nan), dtype=np.float64)
        if not self.norm.scaled():
            self.norm.autoscale_None(self._A)

    def get_array(self):
        return self._A

    def set_norm(self, norm):
        self.norm = norm
        self.changed()

    def set_cmap(self, cmap):
        self.cmap = to_cmap(cmap)
        self.changed()

    def get_clim(self):
        return self.norm.vmin, self.norm.vmax

    def changed(self):
        if self.colorbar is not None:
            try:
                self.colorbar.update_normal(self)
            except Exception:
                pass


# ---------------------------------------------------------------------- #
#  Geometry helpers                                                        #
# ---------------------------------------------------------------------- #

class Bbox:
    """A rectangle (x0, y0, width, height), matplotlib's Bbox as the package reads it."""

    def __init__(self, x0, y0, width, height):
        self.x0, self.y0, self.width, self.height = float(x0), float(y0), float(width), float(height)

    @property
    def bounds(self):
        return (self.x0, self.y0, self.width, self.height)

    @property
    def x1(self):
        return self.x0 + self.width

    @property
    def y1(self):
        return self.y0 + self.height

    @property
    def xmin(self):
        return min(self.x0, self.x1)

    @property
    def ymin(self):
        return min(self.y0, self.y1)

    @property
    def xmax(self):
        return max(self.x0, self.x1)

    @property
    def ymax(self):
        return max(self.y0, self.y1)

    def contains(self, x, y):
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

    @classmethod
    def from_bounds(cls, x0, y0, w, h):
        return cls(x0, y0, w, h)

    def __repr__(self):
        return f"Bbox({self.x0:.4g}, {self.y0:.4g}, {self.width:.4g}, {self.height:.4g})"


class Affine2D:
    """A 2D affine transform built as matplotlib's (rotate_deg, translate, scale), added to transData."""

    def __init__(self, matrix=None):
        self._mtx = np.eye(3) if matrix is None else np.array(matrix, dtype=np.float64)

    def rotate_deg(self, deg):
        a = math.radians(float(deg))
        c, s = math.cos(a), math.sin(a)
        self._mtx = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]) @ self._mtx
        return self

    def rotate(self, rad):
        return self.rotate_deg(math.degrees(rad))

    def translate(self, tx, ty):
        self._mtx = np.array([[1, 0, float(tx)], [0, 1, float(ty)], [0, 0, 1.0]]) @ self._mtx
        return self

    def scale(self, sx, sy=None):
        sy = sx if sy is None else sy
        self._mtx = np.array([[float(sx), 0, 0], [0, float(sy), 0], [0, 0, 1.0]]) @ self._mtx
        return self

    def get_matrix(self):
        return self._mtx.copy()

    def transform(self, pts):
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        return p @ self._mtx[:2, :2].T + self._mtx[:2, 2]

    def to_qt(self):
        m = self._mtx
        return QtGui.QTransform(m[0, 0], m[1, 0], m[0, 1], m[1, 1], m[0, 2], m[1, 2])

    def __add__(self, other):
        return _Composite(self, other)


class _Composite:
    def __init__(self, first, then):
        self.first, self.then = first, then

    def __add__(self, other):
        return _Composite(self, other)


class _Transform:
    """transData / transAxes / transFigure: transform(points) gives display pixels (origin bottom left)."""

    def __init__(self, kind, owner):
        self.kind = kind
        self.owner = owner

    def transform(self, pts):
        p = np.asarray(pts, dtype=np.float64)
        shape = p.shape
        p = p.reshape(-1, 2)
        if self.kind == 'figure':
            W, H = self.owner._canvas_size()
            out = np.column_stack([p[:, 0] * W, p[:, 1] * H])
        else:
            ax = self.owner
            x0, y0, w, h = ax._px_rect()
            W, H = ax.figure._canvas_size()
            if self.kind == 'axes':
                fx, fy = p[:, 0], p[:, 1]
            else:
                (xa, xb), (ya, yb) = ax._view_range()
                fx = (p[:, 0] - xa) / ((xb - xa) or 1.0)
                fy = (p[:, 1] - ya) / ((yb - ya) or 1.0)
                if ax.vb.state['xInverted']:
                    fx = 1.0 - fx
                if ax.vb.state['yInverted']:
                    fy = 1.0 - fy
            out = np.column_stack([x0 + fx * w, H - (y0 + h) + fy * h])
        return out.reshape(shape)

    def inverted(self):
        return _Inverse(self)

    def __add__(self, other):
        return _Composite(self, other)


class _Inverse:
    def __init__(self, t):
        self.t = t

    def transform(self, pts):
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        ax = self.t.owner
        x0, y0, w, h = ax._px_rect()
        W, H = ax.figure._canvas_size()
        fx = (p[:, 0] - x0) / (w or 1.0)
        fy = (p[:, 1] - (H - (y0 + h))) / (h or 1.0)
        if self.t.kind == 'axes':
            return np.column_stack([fx, fy])
        (xa, xb), (ya, yb) = ax._view_range()
        return np.column_stack([xa + fx * (xb - xa), ya + fy * (yb - ya)])


# ---------------------------------------------------------------------- #
#  Fast graphics items                                                     #
# ---------------------------------------------------------------------- #

class QuadItem(pg.GraphicsObject):
    """
    Filled quadrilaterals, each one colour: quads (n, 4, 2) in data units and
    either colour indices into a lookup table or RGBA colours per quad.
    Every colour is one QPainterPath filled at once (no per-quad Python
    loop when painting), the result recorded in a QPicture and the item
    cached as a pixmap, so redraws of the view (a marker dragged over a
    heatmap) blit instead of refilling.
    """

    def __init__(self, quads=None, colors=None):
        super().__init__()
        self._quads = np.zeros((0, 4, 2))
        self._groups = []          # (QColor, QPainterPath)
        self._picture = None
        self._bounds = QtCore.QRectF()
        self.setCacheMode(QtWidgets.QGraphicsItem.CacheMode.DeviceCoordinateCache)
        if quads is not None:
            self.setData(quads, colors)

    def setData(self, quads, colors):
        """quads (n, 4, 2); colors (n, 4) uint8 RGBA (one per quad; alpha 0 = not drawn)."""
        q = np.asarray(quads, dtype=np.float64).reshape(-1, 4, 2)
        c = np.asarray(colors, dtype=np.uint8).reshape(-1, 4)
        with np.errstate(invalid='ignore'):
            ok = np.all(np.isfinite(q) & (np.abs(q) < _LIMIT), axis=(1, 2)) & (c[:, 3] > 0)
        q, c = q[ok], c[ok]
        self.prepareGeometryChange()
        self._quads = q
        self._colors = c                    # kept for the publication export (mplexport)
        if len(q):
            xs, ys = q[..., 0], q[..., 1]
            self._bounds = QtCore.QRectF(float(xs.min()), float(ys.min()), float(xs.max() - xs.min()),
                                         float(ys.max() - ys.min()))
        else:
            self._bounds = QtCore.QRectF()
        groups = []
        if len(q):
            key = c.view(np.uint32).ravel()
            order = np.argsort(key, kind='stable')
            ks = key[order]
            starts = np.flatnonzero(np.concatenate([[True], ks[1:] != ks[:-1]]))
            ends = np.concatenate([starts[1:], [len(ks)]])
            for s, e in zip(starts, ends):
                idx = order[s:e]
                pts = q[idx]                                         # (m, 4, 2)
                closed = np.concatenate([pts, pts[:, :1, :]], axis=1)  # (m, 5, 2)
                x = closed[..., 0].ravel()
                y = closed[..., 1].ravel()
                conn = np.ones(x.shape, dtype=np.int32)
                conn[4::5] = 0
                path = pg.functions.arrayToQPath(x, y, connect=conn, finiteCheck=False)
                path.setFillRule(Qt.FillRule.WindingFill)
                r, g, b, a = (int(v) for v in c[idx[0]])
                groups.append((QtGui.QColor(r, g, b, a), path))
        self._groups = groups
        self._picture = None
        self.update()

    def _render(self):
        pic = QtGui.QPicture()
        p = QtGui.QPainter(pic)
        p.setPen(Qt.PenStyle.NoPen)
        for color, path in self._groups:
            p.fillPath(path, QtGui.QBrush(color))
        p.end()
        self._picture = pic

    EXPORTING = False        # set while a vector file is written: cells drawn with a hairline of their colour

    def paint(self, p, *args):
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        if QuadItem.EXPORTING:
            # In a PDF / SVG viewer adjacent fills are antialiased separately and show seams; a
            # hairline of each cell's own colour closes them.
            for color, path in self._groups:
                pen = QtGui.QPen(color)
                pen.setWidthF(0.6)
                pen.setCosmetic(True)
                p.setPen(pen)
                p.setBrush(QtGui.QBrush(color))
                p.drawPath(path)
            return
        if self._picture is None:
            self._render()
        self._picture.play(p)

    def boundingRect(self):
        return QtCore.QRectF(self._bounds)

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        if not len(self._quads):
            return (None, None)
        v = self._quads[..., ax]
        return (float(v.min()), float(v.max()))


class _TextItem(pg.TextItem):
    """pyqtgraph's TextItem with an extra pixel offset (matplotlib's 'offset points')."""

    def __init__(self, *args, offset=(0.0, 0.0), **kw):
        self._offset = QtCore.QPointF(float(offset[0]), -float(offset[1]))
        super().__init__(*args, **kw)
        self.textItem.document().setDocumentMargin(1)

    def updateTextPos(self):
        super().updateTextPos()
        self.textItem.setPos(self.textItem.pos() + self._offset)


class _LegendSample(pg.GraphicsWidget):
    """One legend sample: a short line in the artist's pen and its marker (no click toggling)."""

    def __init__(self, pen=None, symbol=None, size=6, spen=None, sbrush=None, fill=None):
        super().__init__()
        self.pen, self.symbol, self.size, self.spen, self.sbrush, self.fill = pen, symbol, size, spen, sbrush, fill
        self.setFixedWidth(26)
        self.setFixedHeight(14)

    def boundingRect(self):
        return QtCore.QRectF(0, 0, 26, 14)

    def paint(self, p, *args):
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        if self.fill is not None:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(self.fill)
            p.drawRect(QtCore.QRectF(3, 2, 20, 10))
        if self.pen is not None:
            p.setPen(self.pen)
            p.drawLine(QtCore.QPointF(1, 7), QtCore.QPointF(25, 7))
        if self.symbol is not None:
            p.translate(13, 7)
            pg.graphicsItems.ScatterPlotItem.drawSymbol(p, self.symbol, self.size,
                                                        self.spen if self.spen is not None else QtGui.QPen(Qt.PenStyle.NoPen),
                                                        self.sbrush if self.sbrush is not None else QtGui.QBrush(Qt.BrushStyle.NoBrush))


def _tick_strings(vals, spacing):
    """Tick labels with the decimals the spacing needs (0.25 -> 2), the same for every label (matplotlib)."""
    spacing = abs(float(spacing)) or 1.0
    d = max(0, int(math.ceil(-math.log10(spacing) - 1e-9)))
    while d < 8 and abs(spacing * 10 ** d - round(spacing * 10 ** d)) > 1e-6 * max(1.0, spacing * 10 ** d):
        d += 1
    out = []
    for v in vals:
        t = f"{v:.{d}f}"
        if float(t) == 0.0:
            t = f"{0.0:.{d}f}"
        out.append(t.replace('-', '\u2212'))
    return out


def short_label(text):
    """
    An axis label as simple as the title: the quantity (before the first ', ') and its units
    (the last parentheses), e.g. '(dI/dV)/(I/V), uncalibrated, not normalized (arb. units)'
    -> '(dI/dV)/(I/V) (arb. units)'.
    """
    t = str(text or '').strip()
    depth, cut = 0, None
    for i, ch in enumerate(t):                 # the first ', ' outside parentheses
        depth += (ch == '(') - (ch == ')')
        if depth == 0 and t.startswith(', ', i):
            cut = i
            break
    if cut is None:
        return t
    head = t[:cut]
    tail = ''
    if t.endswith(')'):
        k = t.rfind(' (')
        if k > len(head):
            tail = t[k:]
    return (head + tail).strip()


def short_title(text):
    """
    The essential part of a title: its first line, cut at the first ': ', ', ', double space,
    or ' ['.  The rest (counts, states, file details) goes to the window's info panel.
    """
    t = str(text or '').strip()
    t = t.split('\n')[0]
    for sep in (': ', ', ', '  ', ' ['):
        k = t.find(sep)
        if k > 0:
            t = t[:k]
    t = t.strip().rstrip(':,;')
    if len(t) > 1 and t[0].islower() and t[1].islower():
        t = t[0].upper() + t[1:]              # 'raw dI/dV' -> 'Raw dI/dV' (never 'dI/dV' -> 'DI/dV')
    return t


class _Axis(pg.AxisItem):
    """
    A tick axis outside an axes' box.  Its size follows its tick labels
    (pyqtgraph sets it); a change of size re-places the owner.  mode
    'func' shows the values of a function of the view coordinate (the
    secondary y axis), mode 'norm' the data values of a colour bar whose
    view runs over the normalized [0, 1].
    """

    def __init__(self, orientation, owner):
        super().__init__(orientation)
        self._owner = None
        self._sci = None
        self._mode = None          # None, ('func', f, finv), ('norm', norm)
        self.setPen(pg.mkPen('k', width=max(pt2px(0.8), 1)))
        self.setTextPen(pg.mkPen('k'))
        self.setStyle(tickLength=int(round(pt2px(3.5))), tickTextOffset=int(round(pt2px(3.5))),
                      autoExpandTextSpace=True, autoReduceTextSpace=True, hideOverlappingLabels=False)
        self.enableAutoSIPrefix(False)
        self._tick_pt = 10.0
        self._label_pt = 10.0
        self._zoom = 1.0
        self.setTickFont(qfont(10))
        self.label.setFont(qfont(10))
        self.setZValue(5)
        self._owner = owner

    def setMaximumWidth(self, w):
        old = self.maximumWidth()
        super().setMaximumWidth(w)
        owner = getattr(self, '_owner', None)
        if abs(old - w) > 0.5 and owner is not None:
            owner._request_place()

    def setMaximumHeight(self, h):
        old = self.maximumHeight()
        super().setMaximumHeight(h)
        owner = getattr(self, '_owner', None)
        if abs(old - h) > 0.5 and owner is not None:
            owner._request_place()

    _STEPS = (1.0, 2.0, 2.5, 5.0, 10.0)

    def _font_px(self):
        f = self.style.get('tickFont')
        return float(f.pixelSize()) if f is not None and f.pixelSize() > 0 else pt2px(10)

    def _mpl_ticks(self, lo, hi, size):
        """matplotlib's MaxNLocator ('auto' bins, steps 1, 2, 2.5, 5, 10): one level of ticks."""
        lo, hi = (lo, hi) if lo <= hi else (hi, lo)
        span = hi - lo
        if not np.isfinite(span) or span <= 0:
            return []
        horizontal = self.orientation in ('top', 'bottom')
        nbins = int(size / (self._font_px() * (3.0 if horizontal else 2.0)))
        nbins = max(min(nbins, 9), 1)
        raw = span / nbins
        k = 10.0 ** math.floor(math.log10(raw))
        step = next((s * k for s in self._STEPS if s * k >= raw * (1 - 1e-9)), 10 * k)
        first = math.ceil(lo / step - 1e-9) * step
        n = int(math.floor((hi - first) / step + 1e-9)) + 1
        if not 0 <= n <= 200:
            return []
        vals = [first + i * step for i in range(max(n, 0))]
        vals = [0.0 if abs(v) < step * 1e-9 else v for v in vals]
        return [(step, vals)]

    def generateDrawSpecs(self, p):
        self._sci = None
        specs = super().generateDrawSpecs(p)
        self._note_sci()
        if specs is None or self._sci is None or not self.style['showValues']:
            return specs
        axisSpec, tickSpecs, textSpecs = specs
        b = self.mapRectFromParent(self.geometry())
        txt = f"1e{self._sci}"
        if self.orientation in ('left', 'right'):
            rect = QtCore.QRectF(b.right() - 12 if self.orientation == 'left' else b.left() + 2, b.top() - 15, 90, 14)
            flags = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom
        else:
            rect = QtCore.QRectF(b.right() - 60, b.bottom() - 14, 75, 14)
            flags = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom
        textSpecs = list(textSpecs) + [(rect, flags | Qt.TextFlag.TextDontClip, txt)]
        return axisSpec, tickSpecs, textSpecs

    def set_tick_pt(self, pt):
        self._tick_pt = font_pt(pt)
        self.setTickFont(qfont(self._tick_pt, zoom=self._zoom))

    def set_label_pt(self, pt):
        self._label_pt = font_pt(pt)
        self.label.setFont(qfont(self._label_pt, zoom=self._zoom))

    def apply_zoom(self, zoom):
        """The axis' fonts at the figure's zoom; its size follows (pyqtgraph re-measures)."""
        if abs(zoom - self._zoom) < 1e-6:
            return
        self._zoom = zoom
        self.setTickFont(qfont(self._tick_pt, zoom=zoom))
        self.label.setFont(qfont(self._label_pt, zoom=zoom))
        self.setStyle(tickLength=int(round(pt2px(3.5) * zoom)) if self.style['tickLength'] else 0,
                      tickTextOffset=int(round(pt2px(3.5) * zoom)))
        if self.label.isVisible():
            self.label.setHtml(self.labelString())
            self.label.setFont(qfont(self._label_pt, zoom=zoom))
        self.textWidth = self.textHeight = 0
        self._adjustSize()
        self.picture = None
        self.update()

    def _note_sci(self):
        """The axes' title moves up over a power-of-ten label at the top of a vertical axis."""
        flag = self._sci is not None and self.orientation in ('left', 'right')
        if flag != getattr(self, '_had_sci', False):
            self.prepareGeometryChange()
            self._had_sci = flag
            if self._owner is not None:
                self._owner._request_place()

    def boundingRect(self):
        br = super().boundingRect()
        if getattr(self, '_had_sci', False):
            br = br.united(QtCore.QRectF(br.left(), br.top() - 16, br.width() + 80, 16))
        return br

    def tickValues(self, minVal, maxVal, size):
        if self._mode is None:
            return self._mpl_ticks(minVal, maxVal, size)
        if self._mode[0] == 'func':
            f, finv = self._mode[1], self._mode[2]
            try:
                a, b = float(f(minVal)), float(f(maxVal))
            except Exception:
                return super().tickValues(minVal, maxVal, size)
            lo, hi = min(a, b), max(a, b)
            levels = self._mpl_ticks(lo, hi, size)
            return [(sp, [float(finv(v)) for v in vals]) for sp, vals in levels]
        norm = self._mode[1]
        vmin, vmax = norm.vmin, norm.vmax
        if vmin is None or vmax is None or not vmax > vmin:
            return []
        levels = self._mpl_ticks(vmin, vmax, size)
        out = []
        for sp, vals in levels[:2]:
            v = np.asarray([x for x in vals if vmin - 1e-12 * abs(vmax - vmin) <= x <= vmax + 1e-12 * abs(vmax - vmin)])
            pos = np.asarray(norm(v), dtype=np.float64) if len(v) else np.zeros(0)
            out.append((sp, [float(p) for p in pos]))
        return out

    def tickStrings(self, values, scale, spacing):
        if self._mode is None:
            # matplotlib's scientific offset: very small or large ticks as mantissas, the power of ten
            # written once at the end of the axis.
            vals = [float(v) for v in values]
            big = max((abs(v) for v in vals), default=0.0)
            if big and (big < 1e-3 or big >= 1e5):
                e = int(math.floor(math.log10(big)))
                self._sci = e
                k = 10.0 ** e
                return _tick_strings([v / k for v in vals], spacing / k)
            self._sci = None
            return _tick_strings(vals, spacing)
        if self._mode[0] == 'func':
            vals = [float(self._mode[1](v)) for v in values]
        else:
            norm = self._mode[1]
            vals = [float(norm.inverse(v)) for v in values]
        big = max((abs(v) for v in vals), default=0.0)
        if big and (big < 1e-3 or big >= 1e5):
            e = int(math.floor(math.log10(big)))
            self._sci = e
            return _tick_strings([v / 10.0 ** e for v in vals], spacing / 10.0 ** e)
        return _tick_strings(vals, spacing)


class _ViewBox(pg.ViewBox):
    """A ViewBox without its context menu; navigation only while the toolbar's pan or zoom mode is on."""

    def __init__(self, axes):
        super().__init__(enableMenu=False, defaultPadding=0.05)
        self._axes = axes
        self.setMouseEnabled(False, False)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)


# ---------------------------------------------------------------------- #
#  Artists                                                                 #
# ---------------------------------------------------------------------- #

_child_count = itertools.count()


class Artist:
    """The part of matplotlib's Artist the package uses: label, visibility, zorder, picker, remove."""

    zorder_default = 2.0

    def __init__(self, axes=None, label=None, zorder=None, picker=None, visible=True):
        self.axes = axes
        self.figure = axes.figure if axes is not None else None
        self._label = label if label is not None else f"_child{next(_child_count)}"
        self.zorder = self.zorder_default if zorder is None else float(zorder)
        self._picker = picker
        self._visible = bool(visible)
        self._items = []
        self.colorbar = None

    # items
    def _add_item(self, item, ignore_bounds=False):
        item.setZValue(self.zorder)
        if not self._visible:
            item.setVisible(False)
        self.axes.vb.addItem(item, ignoreBounds=ignore_bounds)
        self._items.append(item)
        return item

    def get_label(self):
        return self._label

    def set_label(self, s):
        self._label = '' if s is None else str(s)

    def get_visible(self):
        return self._visible

    def set_visible(self, flag):
        self._visible = bool(flag)
        for it in self._items:
            it.setVisible(self._visible)

    def set_zorder(self, z):
        self.zorder = float(z)
        for it in self._items:
            it.setZValue(self.zorder)

    def get_zorder(self):
        return self.zorder

    def set_picker(self, p):
        self._picker = p

    def get_picker(self):
        return self._picker

    def pickable(self):
        return self._picker is not None and self._visible

    def contains_px(self, x, y):
        return False

    def remove(self):
        ax = self.axes
        for it in self._items:
            try:
                if ax is not None:
                    ax.vb.removeItem(it)
                if it.scene() is not None:
                    it.scene().removeItem(it)
            except Exception:
                pass
        self._items = []
        if ax is not None:
            ax._forget(self)

    def _legend_sample(self):
        return None

    def set(self, **kw):
        for k, v in kw.items():
            f = getattr(self, 'set_' + k, None)
            if f is not None:
                f(v)


_MARKERS = {'o': 'o', 's': 's', 'D': 'd', 'd': 'd', '^': 't1', 'v': 't', '<': 't3', '>': 't2',
            'x': 'x', 'X': 'x', '+': '+', 'P': '+', '*': 'star', 'p': 'p', 'h': 'h', 'H': 'h', '.': 'o', ',': 's',
            '|': '|', '_': '_'}


def _marker_symbol(m):
    if m in (None, '', 'none', 'None', ' '):
        return None
    return _MARKERS.get(m, 'o')


class Line2D(Artist):
    """A curve and/or markers (pyqtgraph PlotDataItem): get_xdata, get_ydata, set_data, set_color, ..."""

    def __init__(self, axes, x, y, color='C0', lw=1.5, ls='-', marker=None, ms=6.0, mec=None, mfc=None,
                 mew=1.0, alpha=None, label=None, zorder=None, picker=None, visible=True, fill=None):
        super().__init__(axes, label, zorder, picker, visible)
        self._x = np.asarray(x, dtype=np.float64).ravel()
        self._y = np.asarray(y, dtype=np.float64).ravel()
        self._style = dict(color=color, lw=lw, ls=ls, marker=marker, ms=ms, mec=mec, mfc=mfc, mew=mew, alpha=alpha)
        self.item = None
        self._build()

    def _style_args(self):
        s = self._style
        color = s['color']
        pen = mkpen(color, s['lw'], s['ls'], s['alpha'])
        sym = _marker_symbol(s['marker'])
        kw = {'pen': pen}
        if sym is not None:
            ms = s['ms'] if s['marker'] != '.' else s['ms'] * 0.5
            mfc = color if s['mfc'] is None else s['mfc']
            mec = color if s['mec'] is None else s['mec']
            kw.update(symbol=sym, symbolSize=max(pt2px(ms), 1.0),
                      symbolBrush=mkbrush(mfc, s['alpha']) if mkbrush(mfc) is not None else None,
                      symbolPen=mkpen(mec, s['mew'], '-', s['alpha']) if s['mew'] and s['mew'] > 0 else None)
            if kw['symbolBrush'] is None:
                kw['symbolBrush'] = QtGui.QBrush(Qt.BrushStyle.NoBrush)
            if kw['symbolPen'] is None:
                kw['symbolPen'] = QtGui.QPen(Qt.PenStyle.NoPen)
        return kw

    def _build(self):
        kw = self._style_args()
        x, y = self._x, self._y
        if len(x) == len(y):
            # A point with a NaN coordinate is not drawn and takes no part in the autoscale (matplotlib).
            with np.errstate(invalid='ignore'):
                bad = ~(np.isfinite(x) & np.isfinite(y)) | (np.abs(x) > _LIMIT) | (np.abs(y) > _LIMIT)
            if bad.any():
                x = np.where(bad, np.nan, x)
                y = np.where(bad, np.nan, y)
        if kw.get('pen') is None and kw.get('symbol') is not None:
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
        if kw.get('symbol') is None:
            # A plain curve: the lighter PlotCurveItem (no scatter child).
            if self.item is None:
                self.item = pg.PlotCurveItem(x, y, pen=kw.get('pen') or QtGui.QPen(Qt.PenStyle.NoPen),
                                             connect='finite', antialias=ANTIALIAS, skipFiniteCheck=False)
                self._add_item(self.item)
            else:
                self.item.setData(x, y, pen=kw.get('pen') or QtGui.QPen(Qt.PenStyle.NoPen), connect='finite')
            return
        if self.item is None:
            self.item = pg.PlotDataItem(x, y, connect='finite', antialias=ANTIALIAS, **kw)
            self._add_item(self.item)
        else:
            self.item.setData(x, y, connect='finite', **kw)

    def get_xdata(self, orig=True):
        return self._x

    def get_ydata(self, orig=True):
        return self._y

    def get_data(self, orig=True):
        return self._x, self._y

    def set_data(self, *args):
        if len(args) == 1:
            x, y = args[0]
        else:
            x, y = args
        self._x = np.asarray(x, dtype=np.float64).ravel()
        self._y = np.asarray(y, dtype=np.float64).ravel()
        self._build()

    def set_xdata(self, x):
        self.set_data(x, self._y)

    def set_ydata(self, y):
        self.set_data(self._x, y)

    def get_color(self):
        return self._style['color']

    def set_color(self, c):
        self._style['color'] = c
        self._build()

    def set_linewidth(self, lw):
        self._style['lw'] = lw
        self._build()

    def set_linestyle(self, ls):
        self._style['ls'] = ls
        self._build()

    def set_alpha(self, a):
        self._style['alpha'] = a
        self._build()

    def set_markerfacecolor(self, c):
        self._style['mfc'] = c
        self._build()

    def contains_px(self, x, y):
        """Display-pixel distance test (matplotlib's pick radius = the picker value)."""
        tol = float(self._picker) if isinstance(self._picker, (int, float)) and not isinstance(self._picker, bool) else 5.0
        if len(self._x) == 0:
            return False
        xy = self.axes.transData.transform(np.column_stack([self._x, self._y]))
        ok = np.all(np.isfinite(xy), axis=1)
        if self._style['ls'] in (None, 'none', 'None', '', ' ') or ok.sum() < 2:
            d = np.hypot(xy[ok, 0] - x, xy[ok, 1] - y)
            return bool(len(d)) and float(d.min()) <= tol
        return _polyline_distance(x, y, xy) <= tol

    def _legend_sample(self):
        kw = self._style_args()
        return _LegendSample(kw.get('pen'), kw.get('symbol'), kw.get('symbolSize', 6), kw.get('symbolPen'),
                             kw.get('symbolBrush'))


def _polyline_distance(x, y, xy):
    """Distance from (x, y) to a polyline (n, 2), NaN vertices breaking it."""
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) == 0:
        return np.inf
    p, q = xy[:-1], xy[1:]
    ok = np.all(np.isfinite(p), axis=1) & np.all(np.isfinite(q), axis=1)
    best = np.inf
    fin = np.all(np.isfinite(xy), axis=1)
    if fin.any():
        best = float(np.min(np.hypot(xy[fin, 0] - x, xy[fin, 1] - y)))
    if ok.any():
        p, d = p[ok], (q - p)[ok]
        L2 = np.sum(d * d, axis=1)
        t = np.clip(((x - p[:, 0]) * d[:, 0] + (y - p[:, 1]) * d[:, 1]) / np.where(L2 > 0, L2, 1.0), 0.0, 1.0)
        best = min(best, float(np.sqrt(np.min((p[:, 0] + t * d[:, 0] - x) ** 2 + (p[:, 1] + t * d[:, 1] - y) ** 2))))
    return best


class _RefLine(Artist):
    """axhline / axvline: a pyqtgraph InfiniteLine across the axes."""

    def __init__(self, axes, value, horizontal, color='k', lw=1.5, ls='-', alpha=None, label=None, zorder=None,
                 picker=None, visible=True):
        super().__init__(axes, label, zorder, picker, visible)
        self._value = float(value)
        self._horizontal = bool(horizontal)
        self._style = dict(color=color, lw=lw, ls=ls, alpha=alpha)
        pen = mkpen(color, lw, ls, alpha) or QtGui.QPen(Qt.PenStyle.NoPen)
        bad = not np.isfinite(self._value) or abs(self._value) > _LIMIT
        self.item = pg.InfiniteLine(pos=0.0 if bad else self._value, angle=0 if horizontal else 90, pen=pen,
                                    movable=False)
        self._add_item(self.item, ignore_bounds=True)
        if bad:
            self.item.setVisible(False)

    def get_xdata(self, orig=True):
        return np.array([0.0, 1.0]) if self._horizontal else np.array([self._value] * 2)

    def get_ydata(self, orig=True):
        return np.array([self._value] * 2) if self._horizontal else np.array([0.0, 1.0])

    def set_ydata(self, y):
        if self._horizontal:
            self._value = float(np.ravel(y)[0])
            self.item.setValue(self._value)

    def set_xdata(self, x):
        if not self._horizontal:
            self._value = float(np.ravel(x)[0])
            self.item.setValue(self._value)

    def set_color(self, c):
        self._style['color'] = c
        self.item.setPen(mkpen(c, self._style['lw'], self._style['ls'], self._style['alpha']))

    def contains_px(self, x, y):
        tol = float(self._picker) if isinstance(self._picker, (int, float)) and not isinstance(self._picker, bool) else 5.0
        p = self.axes.transData.transform([[self._value, self._value]])[0]
        return abs((y - p[1]) if self._horizontal else (x - p[0])) <= tol

    def _legend_sample(self):
        s = self._style
        return _LegendSample(mkpen(s['color'], s['lw'], s['ls'], s['alpha']))


class _Span(Artist):
    zorder_default = 1.0

    def __init__(self, axes, lo, hi, vertical, color='C0', alpha=None, label=None, zorder=None, visible=True):
        super().__init__(axes, label, zorder, None, visible)
        self._span = (float(lo), float(hi), bool(vertical), color, 0.25 if alpha is None else alpha)
        self._brush = mkbrush(color, 0.25 if alpha is None else alpha)
        self.item = pg.LinearRegionItem(values=(float(lo), float(hi)),
                                        orientation='vertical' if vertical else 'horizontal',
                                        brush=self._brush, pen=QtGui.QPen(Qt.PenStyle.NoPen), movable=False)
        for ln in self.item.lines:
            ln.setPen(QtGui.QPen(Qt.PenStyle.NoPen))
        self._add_item(self.item, ignore_bounds=True)

    def _legend_sample(self):
        return _LegendSample(fill=self._brush)


class AxesImage(Artist):
    """
    imshow: a pyqtgraph ImageItem coloured through a lookup table from the
    norm and the colour map at levels (vmin, vmax), placed by its extent
    (and an optional Affine2D, set_transform).  NaN pixels are transparent.
    """

    zorder_default = 0.0

    def __init__(self, axes, data, cmap=None, norm=None, extent=None, origin='upper', alpha=None, zorder=None,
                 label=None, vmin=None, vmax=None):
        super().__init__(axes, label, zorder, None, True)
        self.cmap = to_cmap(cmap)
        self.origin = origin or 'upper'
        self._alpha = alpha
        self._affine = None
        self.item = pg.ImageItem(axisOrder='row-major')
        self.item.setAutoDownsample(False)
        self._A = None
        self.norm = norm if norm is not None else Normalize(vmin, vmax)
        self._extent = None
        self._set_array(data)
        self.set_extent(extent)
        if alpha is not None:
            self.item.setOpacity(float(alpha))
        self._add_item(self.item)

    def _set_array(self, data):
        A = np.ma.asarray(data)
        if A.dtype.kind in 'fiub' and A.ndim == 2:
            self._A = np.asarray(np.ma.filled(A.astype(np.float64), np.nan), dtype=np.float64)
            if not self.norm.scaled():
                self.norm.autoscale_None(self._A)
            self._rgb = False
        else:
            self._A = np.asarray(np.ma.filled(A, 0))
            self._rgb = True

    def _refresh(self):
        img = self._A
        if self.origin == 'upper':
            img = img[::-1]
        if self._rgb:
            a = np.asarray(img)
            if a.dtype.kind == 'f':
                a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
            self.item.setImage(a, autoLevels=False, levels=(0, 255))
            return
        vmin, vmax = self.norm.vmin, self.norm.vmax
        if vmin is None or vmax is None or not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = 0.0, 1.0
        if vmax <= vmin:
            vmax = vmin + 1e-12 if vmin == vmax else vmin
        lut = norm_lut(self.cmap, self.norm, vmin, vmax)
        self.item.setImage(img, autoLevels=False, levels=(vmin, vmax), lut=lut)

    def _place_transform(self):
        ny, nx = self._A.shape[:2]
        l, r, b, t = self._extent
        ok = (nx > 0 and ny > 0 and all(np.isfinite(v) and abs(v) < _LIMIT for v in (l, r, b, t))
              and abs(r - l) > 0 and abs(t - b) > 0)
        if ok and self._affine is not None:
            m = self._affine.get_matrix()
            ok = bool(np.all(np.isfinite(m)) and abs(np.linalg.det(m[:2, :2])) > 1e-300)
        if not ok:
            # A page without a size (a scan stopped before it began, a zero pixel scale): not drawn.
            self.item.setVisible(False)
            self.item.setTransform(QtGui.QTransform())
            return
        self.item.setVisible(self._visible)
        # Pixel (row, col) of the drawn array (origin flips applied) covers [l + col dx, ...] from (l, b).
        T = QtGui.QTransform()
        T.translate(l, b)
        T.scale((r - l) / nx, (t - b) / ny)
        if self._affine is not None:
            T = T * self._affine.to_qt()
        self.item.setTransform(T)

    def set_extent(self, extent):
        ny, nx = self._A.shape[:2]
        if extent is None:
            extent = (-0.5, nx - 0.5, ny - 0.5, -0.5) if self.origin == 'upper' else (-0.5, nx - 0.5, -0.5, ny - 0.5)
        l, r, b, t = [float(v) for v in extent]
        if self.origin == 'upper':
            # matplotlib: row 0 at the top edge t; the drawn array is flipped, so its row 0 sits at b.
            pass
        self._extent = (l, r, b, t)
        self._refresh()
        self._place_transform()

    def get_extent(self):
        return list(self._extent)

    def set_transform(self, t):
        """A transform added to transData (matplotlib's Affine2D() ... + ax.transData)."""
        aff = None
        while isinstance(t, _Composite):
            if isinstance(t.first, Affine2D):
                aff = t.first
            t = t.then
        if isinstance(t, Affine2D):
            aff = t
        self._affine = aff
        self._place_transform()

    def set_data(self, data):
        self._set_array(data)
        self._refresh()
        self._place_transform()

    def get_array(self):
        return self._A

    def set_norm(self, norm):
        self.norm = norm
        if not self.norm.scaled() and self._A is not None and not self._rgb:
            self.norm.autoscale_None(self._A)
        self._refresh()
        if self.colorbar is not None:
            self.colorbar.update_normal(self)

    def set_cmap(self, cmap):
        self.cmap = to_cmap(cmap)
        self._refresh()
        if self.colorbar is not None:
            self.colorbar.update_normal(self)

    def set_clim(self, vmin=None, vmax=None):
        if isinstance(vmin, (tuple, list)):
            vmin, vmax = vmin
        if vmin is not None:
            self.norm.vmin = float(vmin)
        if vmax is not None:
            self.norm.vmax = float(vmax)
        self.set_norm(self.norm)

    def get_clim(self):
        return self.norm.vmin, self.norm.vmax


class QuadMesh(Artist):
    """pcolormesh: filled quads (QuadItem) coloured from the norm and the colour map."""

    zorder_default = 1.0

    def __init__(self, axes, X, Y, C, cmap=None, norm=None, vmin=None, vmax=None, zorder=None, label=None,
                 alpha=None):
        super().__init__(axes, label, zorder, None, True)
        self.cmap = to_cmap(cmap)
        self.norm = norm if norm is not None else Normalize(vmin, vmax)
        self._alpha = alpha
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        C = np.asarray(np.ma.filled(np.ma.asarray(C, dtype=np.float64), np.nan), dtype=np.float64)
        nr, nc = C.shape
        if X.ndim == 1:
            X = np.broadcast_to(X[None, :], (nr + 1, nc + 1))
        if Y.ndim == 1:
            Y = np.broadcast_to(Y[:, None], (nr + 1, nc + 1))
        self._X, self._Y, self._C = X, Y, C
        if not self.norm.scaled():
            self.norm.autoscale_None(C)
        q = np.stack([np.stack([X[:-1, :-1], Y[:-1, :-1]], -1), np.stack([X[:-1, 1:], Y[:-1, 1:]], -1),
                      np.stack([X[1:, 1:], Y[1:, 1:]], -1), np.stack([X[1:, :-1], Y[1:, :-1]], -1)], axis=2)
        self._quads = q.reshape(-1, 4, 2)
        self.item = QuadItem()
        self._paint()
        self._add_item(self.item)

    def _paint(self):
        colors = self.cmap(np.asarray(self.norm(self._C.ravel()), dtype=np.float64), alpha=self._alpha, bytes=True)
        colors[~np.isfinite(self._C.ravel())] = 0
        self.item.setData(self._quads, colors)

    def set_array(self, C):
        self._C = np.asarray(np.ma.filled(np.ma.asarray(C, dtype=np.float64), np.nan), dtype=np.float64).reshape(self._C.shape)
        self._paint()

    def get_array(self):
        return self._C

    def set_norm(self, norm):
        self.norm = norm
        self._paint()

    def set_cmap(self, cmap):
        self.cmap = to_cmap(cmap)
        self._paint()


class ContourSet(Artist):
    """contourf on a regular grid: the bands drawn as an image (one colour per band between levels)."""

    zorder_default = 1.0

    def __init__(self, axes, X, Y, Z, levels=10, cmap=None, zorder=None):
        super().__init__(axes, None, zorder, None, True)
        self._args = (np.asarray(X), np.asarray(Y), np.asarray(np.ma.filled(np.ma.asarray(Z, dtype=float), np.nan)),
                      levels, cmap)
        Z = np.asarray(np.ma.filled(np.ma.asarray(Z, dtype=np.float64), np.nan), dtype=np.float64)
        ok = np.isfinite(Z)
        lo, hi = (float(Z[ok].min()), float(Z[ok].max())) if ok.any() else (0.0, 1.0)
        if np.ndim(levels) == 0:
            lev = np.linspace(lo, hi, int(levels) + 1)
        else:
            lev = np.asarray(levels, dtype=np.float64)
        self.levels = lev
        base = to_cmap(cmap)
        mids = 0.5 * (lev[:-1] + lev[1:])
        band_colors = base((mids - lev[0]) / ((lev[-1] - lev[0]) or 1.0))
        self.cmap = Colormap(base.name + '_bands', band_colors)
        self.norm = Normalize(lev[0], lev[-1])
        band = np.clip(np.digitize(Z, lev[1:-1]), 0, len(mids) - 1).astype(np.float64)
        band[~ok] = np.nan
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        x = X[0] if X.ndim == 2 else X
        y = Y[:, 0] if Y.ndim == 2 else Y
        self.item = pg.ImageItem(axisOrder='row-major')
        self.item.setImage(band, autoLevels=False, levels=(0, len(mids)), lut=self.cmap.lut(len(mids)))
        T = QtGui.QTransform()
        T.translate(float(x[0]), float(y[0]))
        T.scale((float(x[-1]) - float(x[0])) / band.shape[1], (float(y[-1]) - float(y[0])) / band.shape[0])
        self.item.setTransform(T)
        self._add_item(self.item)


class LineCollection(Artist):
    """Line segments / polylines of one colour, one PlotCurveItem with breaks (matplotlib's LineCollection)."""

    zorder_default = 2.0

    def __init__(self, segments, colors='k', linewidths=1.5, linestyles='-', zorder=None, alpha=None, label=None,
                 **kw):
        super().__init__(None, label, zorder, None, True)
        self._segments = [np.asarray(s, dtype=np.float64).reshape(-1, 2) for s in segments]
        if isinstance(colors, (list, tuple)) and len(colors) and not isinstance(colors[0], (int, float)):
            colors = colors[0]
        self._color = colors
        self._lw = float(np.ravel(linewidths)[0]) if np.ndim(linewidths) else float(linewidths)
        self._ls = linestyles if isinstance(linestyles, str) else '-'
        self._alpha = alpha
        self.item = None

    def _attach(self, axes):
        self.axes = axes
        self.figure = axes.figure
        parts = []
        for s in self._segments:
            with np.errstate(invalid='ignore'):
                s = np.where(np.isfinite(s) & (np.abs(s) < _LIMIT), s, np.nan)
            parts.append(s)
            parts.append(np.full((1, 2), np.nan))
        xy = np.vstack(parts) if parts else np.zeros((0, 2))
        self.item = pg.PlotCurveItem(xy[:, 0], xy[:, 1], pen=mkpen(self._color, self._lw, self._ls, self._alpha),
                                     connect='finite', antialias=ANTIALIAS)
        self._add_item(self.item)

    def get_segments(self):
        return list(self._segments)


class PolyCollection(Artist):
    """Filled polygons (quads here) with colours from an array and a norm, or face colours (QuadItem)."""

    zorder_default = 1.0

    def __init__(self, verts, array=None, cmap=None, norm=None, facecolors=None, edgecolors=None,
                 antialiased=False, zorder=None, alpha=None, label=None, **kw):
        super().__init__(None, label, zorder, None, True)
        v = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in verts] if not isinstance(verts, np.ndarray) \
            else np.asarray(verts, dtype=np.float64)
        if isinstance(v, list):
            v = np.array([p if len(p) == 4 else np.vstack([p[:3], p[2:3]]) for p in v]) if v else np.zeros((0, 4, 2))
        self._verts = np.asarray(v, dtype=np.float64).reshape(-1, 4, 2)
        self._array = None if array is None else np.asarray(array, dtype=np.float64).ravel()
        self.cmap = to_cmap(cmap) if (cmap is not None or array is not None) else None
        self.norm = norm if norm is not None else (Normalize() if array is not None else None)
        if self.norm is not None and self._array is not None and not self.norm.scaled():
            self.norm.autoscale_None(self._array)
        self._facecolors = facecolors
        self._alpha = alpha
        self.item = None

    def _colors(self):
        if self._array is not None:
            c = self.cmap(np.asarray(self.norm(self._array), dtype=np.float64), alpha=self._alpha, bytes=True)
            c[~np.isfinite(self._array)] = 0
            return c
        fc = self._facecolors
        if fc is None:
            fc = 'C0'
        if isinstance(fc, str) or (np.ndim(fc) == 1 and len(np.ravel(fc)) in (3, 4) and not isinstance(fc[0], str)):
            c = np.array([to_rgba(fc, self._alpha)] * len(self._verts))
        else:
            c = np.array([to_rgba(f, self._alpha) for f in fc]) if len(fc) and isinstance(fc[0], str) \
                else np.asarray(fc, dtype=np.float64).reshape(-1, 4)
        return (np.clip(c, 0, 1) * 255 + 0.5).astype(np.uint8)

    def _attach(self, axes):
        self.axes = axes
        self.figure = axes.figure
        self.item = QuadItem(self._verts, self._colors())
        self._add_item(self.item)

    def set_array(self, a):
        self._array = np.asarray(a, dtype=np.float64).ravel()
        if self.item is not None:
            self.item.setData(self._verts, self._colors())

    def get_array(self):
        return self._array

    def set_norm(self, norm):
        self.norm = norm
        if self.item is not None:
            self.item.setData(self._verts, self._colors())

    def set_cmap(self, cmap):
        self.cmap = to_cmap(cmap)
        if self.item is not None:
            self.item.setData(self._verts, self._colors())


_HA = {'left': 0.0, 'center': 0.5, 'centre': 0.5, 'right': 1.0}
_VA = {'top': 0.0, 'center': 0.5, 'centre': 0.5, 'center_baseline': 0.55, 'baseline': 0.8, 'bottom': 1.0}


class Text(Artist):
    """
    A text in data, axes, or figure coordinates (pyqtgraph TextItem):
    set_text, get_text, set_color, set_fontsize, set_visible, set_position.
    """

    zorder_default = 3.0

    def __init__(self, owner, x, y, s, coords='data', fontsize=None, ha='left', va='baseline', color='k',
                 family=None, fontweight=None, weight=None, bbox=None, visible=True, zorder=None, offset=(0, 0),
                 linespacing=None, clip_on=False, wrap=False, rotation=0, fontfamily=None, size=None,
                 horizontalalignment=None, verticalalignment=None, label=None, **kw):
        axes = owner if isinstance(owner, Axes) else None
        super().__init__(axes, label, zorder, None, visible)
        self.figure = owner if isinstance(owner, Figure) else owner.figure
        self._owner = owner
        self._coords = coords
        self._x, self._y = float(x), float(y)
        self._text = '' if s is None else str(s)
        self._size = font_pt(size if fontsize is None else fontsize)
        self._family = family or fontfamily
        self._weight = fontweight or weight
        self._color = color
        ha = horizontalalignment or ha
        va = verticalalignment or va
        self._ha, self._va = ha, va
        self._bbox = dict(bbox) if bbox else None
        self._rotation = float(rotation or 0)
        self._offset_px = (float(offset[0]), float(offset[1]))
        fill = None
        border = None
        if bbox:
            fc = bbox.get('facecolor', bbox.get('fc', 'white'))
            fill = mkbrush(fc, bbox.get('alpha', 1.0))
            ec = bbox.get('edgecolor', bbox.get('ec'))
            lw = bbox.get('lw', bbox.get('linewidth', 1.0))
            if ec is not None and lw:
                border = mkpen(ec, lw)
        self.item = _TextItem('', anchor=(_HA.get(ha, 0.0), _VA.get(va, 0.8)), fill=fill, border=border,
                              angle=float(rotation or 0), offset=offset)
        self.item.setZValue(self.zorder + 10 if coords != 'data' else self.zorder)
        self._apply()
        if not self._visible:
            self.item.setVisible(False)
        if coords == 'data':
            self.axes.vb.addItem(self.item, ignoreBounds=True)
            if np.isfinite(self._x) and np.isfinite(self._y) and max(abs(self._x), abs(self._y)) < _LIMIT:
                self.item.setPos(self._x, self._y)
            else:
                self.item.setVisible(False)
        else:
            self.figure.canvas.scene().addItem(self.item)
            self._place()
        self._items.append(self.item)

    def _apply(self):
        ti = self.item
        ti.setFont(qfont(self._size, self._family, self._weight, zoom=getattr(self.figure, 'font_zoom', 1.0)))
        ti.setColor(qcolor(self._color))
        ti.setText(self._text)
        doc = ti.textItem.document()
        opt = doc.defaultTextOption()
        opt.setAlignment({'center': Qt.AlignmentFlag.AlignHCenter, 'right': Qt.AlignmentFlag.AlignRight}.get(
            self._ha, Qt.AlignmentFlag.AlignLeft))
        doc.setDefaultTextOption(opt)
        ti.textItem.setTextWidth(-1)
        if '\n' in self._text and self._ha in ('center', 'right'):
            ti.textItem.setTextWidth(doc.idealWidth() + 2)
        ti.updateTextPos()

    def _place(self):
        if self._coords == 'data':
            return
        if self._coords == 'axes':
            x0, y0, w, h = self.axes._px_rect()
            self.item.setPos(x0 + self._x * w, y0 + (1.0 - self._y) * h)
        else:
            W, H = self.figure._canvas_size()
            self.item.setPos(self._x * W, (1.0 - self._y) * H)

    def set_text(self, s):
        s = '' if s is None else str(s)
        if s != self._text:
            self._text = s
            self._apply()

    def get_text(self):
        return self._text

    def set_color(self, c):
        self._color = c
        self.item.setColor(qcolor(c))

    def get_color(self):
        return self._color

    def set_fontsize(self, s):
        self._size = font_pt(s)
        self._apply()

    set_size = set_fontsize

    def get_fontsize(self):
        return self._size

    def set_position(self, xy):
        self._x, self._y = float(xy[0]), float(xy[1])
        if self._coords == 'data':
            self.item.setPos(self._x, self._y)
        else:
            self._place()

    def get_position(self):
        return (self._x, self._y)

    def set_fontweight(self, w):
        self._weight = w
        self._apply()

    def set_visible(self, flag):
        self._visible = bool(flag)
        self.item.setVisible(self._visible)

    def remove(self):
        try:
            if self._coords == 'data' and self.axes is not None:
                self.axes.vb.removeItem(self.item)
            elif self.item.scene() is not None:
                self.item.scene().removeItem(self.item)
        except Exception:
            pass
        self._items = []
        if isinstance(self._owner, Axes):
            self._owner._forget(self)
        else:
            self._owner._forget_text(self)


class Annotation(Artist):
    """annotate: a text with an offset and/or an arrow from xytext to xy (data coordinates)."""

    zorder_default = 3.0

    def __init__(self, axes, s, xy, xytext=None, textcoords=None, arrowprops=None, color='k', fontsize=None,
                 fontweight=None, zorder=None, ha='left', va='baseline', **kw):
        super().__init__(axes, None, zorder, None, True)
        self.text = None
        self._ann = dict(s=s, xy=tuple(xy), xytext=None if xytext is None else tuple(xytext), textcoords=textcoords,
                         arrowprops=dict(arrowprops) if arrowprops else None, color=color, fontsize=fontsize,
                         fontweight=fontweight, ha=ha, va=va)
        xy = (float(xy[0]), float(xy[1]))
        if arrowprops is not None and xytext is not None:
            a = (float(xytext[0]), float(xytext[1]))
            col = arrowprops.get('color', color)
            lw = arrowprops.get('lw', arrowprops.get('linewidth', 1.0))
            line = pg.PlotCurveItem([a[0], xy[0]], [a[1], xy[1]], pen=mkpen(col, lw), antialias=True)
            self._add_item(line, ignore_bounds=True)
            ang = self._screen_angle(a, xy)
            style = arrowprops.get('arrowstyle', '->')
            arrow = pg.ArrowItem(angle=ang, tipAngle=40, headLen=pt2px(6 + lw * 2), tailLen=0, tailWidth=0,
                                 pen=mkpen(col, lw), brush=mkbrush(col) if style in ('-|>', 'simple', 'fancy', 'wedge')
                                 else QtGui.QBrush(Qt.BrushStyle.NoBrush))
            arrow.setPos(xy[0], xy[1])
            self._add_item(arrow, ignore_bounds=True)
        if s:
            off = (0.0, 0.0)
            if xytext is not None and textcoords in ('offset points', 'offset pixels'):
                k = pt2px(1) if textcoords == 'offset points' else 1.0
                off = (float(xytext[0]) * k, float(xytext[1]) * k)
                pos = xy
            elif xytext is not None:
                pos = (float(xytext[0]), float(xytext[1]))
            else:
                pos = xy
            self.text = Text(axes, pos[0], pos[1], s, 'data', fontsize=fontsize, color=color, fontweight=fontweight,
                             ha=ha, va=va, offset=off, zorder=self.zorder)
            self._items.append(self.text.item)

    def _screen_angle(self, a, b):
        ax = self.axes
        try:
            pa, pb = ax.transData.transform([a, b])
            dx, dy = pb[0] - pa[0], pb[1] - pa[1]
        except Exception:
            dx, dy = b[0] - a[0], b[1] - a[1]
        if dx == 0 and dy == 0:
            dx, dy = b[0] - a[0], b[1] - a[1]
        # pyqtgraph's arrow at angle 0 points to the left (-x); screen y runs down.
        return math.degrees(math.atan2(dy, -dx))

    def set_visible(self, flag):
        super().set_visible(flag)


# ---------------------------------------------------------------------- #
#  Colour bar                                                              #
# ---------------------------------------------------------------------- #

class Colorbar:
    """
    A colour bar in its own axes: the colour map evenly along the bar, the
    tick labels at their data values through the norm (so a gamma spaces
    them as matplotlib does), the label on the right.
    """

    def __init__(self, figure, mappable, cax, label=None, parent_axes=None, parent_rect=None):
        self.figure = figure
        self.ax = cax
        self.mappable = mappable
        self._parent = parent_axes
        self._parent_rect = parent_rect
        self._label = label or ''
        cax.clear()
        mappable.colorbar = self
        cax._colorbar = self
        cax._is_colorbar = True
        cax._apply_toolbar_mode('')
        cax.xaxis_item.setStyle(showValues=False, tickLength=0)
        cax.xaxis_item.setTicks([[]])
        cax.xaxis_item._dirty = cax.yaxis_item._dirty = True
        cax.yaxis_item.setVisible(False)
        self._axis = cax._make_side_axis('right')
        self.image = pg.ImageItem(axisOrder='row-major')
        self.image.setImage(((np.arange(512) + 0.5) / 512.0)[:, None], autoLevels=False, levels=(0.0, 1.0))
        self.image.setRect(QtCore.QRectF(0.0, 0.0, 1.0, 1.0))
        cax.vb.addItem(self.image)
        cax.vb.disableAutoRange()
        cax.vb.setRange(xRange=(0, 1), yRange=(0, 1), padding=0)
        self.update_normal(mappable)
        if label:
            self.set_label(label)
        cax._place()

    def update_normal(self, mappable):
        self.mappable = mappable
        cmap = to_cmap(getattr(mappable, 'cmap', None))
        norm = getattr(mappable, 'norm', None) or Normalize(0, 1)
        self.image.setLookupTable(cmap.lut(512))
        self._axis._mode = ('norm', norm)
        self._axis.picture = None
        self._axis.update()

    def set_label(self, label, **kw):
        self._label = '' if label is None else str(label)
        self._label_kw = dict(kw)
        self.ax._side_label(self._axis, self._label, kw.get('fontsize', kw.get('size')))

    def set_ticks(self, *a, **k):
        pass

    def remove(self):
        try:
            if getattr(self.mappable, 'colorbar', None) is self:
                self.mappable.colorbar = None
        except Exception:
            pass
        self.ax.remove()
        if self._parent is not None and self._parent_rect is not None:
            self._parent.set_position(self._parent_rect)


# ---------------------------------------------------------------------- #
#  Axes                                                                    #
# ---------------------------------------------------------------------- #

class _Patch:
    def __init__(self, ax):
        self._ax = ax

    def set_facecolor(self, c):
        self._ax.vb.setBackgroundColor(qcolor(c))

    def set_alpha(self, a):
        pass

    def set_visible(self, f):
        pass


class _TitleProxy:
    """ax.title: set_fontsize / set_text / get_text on the axes' title."""

    def __init__(self, ax):
        self._ax = ax

    def set_fontsize(self, s):
        self._ax._title_size = font_pt(s)
        if self._ax._title is not None:
            self._ax._title.set_fontsize(s)

    def set_text(self, s):
        self._ax.set_title(s)

    def get_text(self):
        return self._ax._title.get_text() if self._ax._title is not None else ''

    def set_color(self, c):
        if self._ax._title is not None:
            self._ax._title.set_color(c)


class SecondaryAxis:
    """secondary_yaxis: tick labels of f(y) on the right of the axes."""

    def __init__(self, ax, axis_item):
        self._ax = ax
        self._axis = axis_item

    def set_ylabel(self, label, **kw):
        self._ax._side_label(self._axis, label, kw.get('fontsize'))

    def tick_params(self, labelsize=None, **kw):
        if labelsize is not None:
            self._axis.set_tick_pt(labelsize)


class Axes:
    """
    One axes: a pyqtgraph ViewBox placed exactly at its figure-fraction
    rectangle (matplotlib's axes box, framed), its x and y tick axes outside
    the box, the title above it.  The plotting methods return artists with
    matplotlib's interface (remove, set_visible, get_label, ...).
    """

    def __init__(self, fig, rect=None, spec=None, sharex=None, sharey=None, inset_parent=None, label=''):
        self.figure = fig
        self._rect = list(rect) if rect is not None else None
        self._spec = spec
        self._inset_parent = inset_parent
        self._insets = []
        scene = fig.canvas.scene()
        self.vb = _ViewBox(self)
        self.vb.setBorder(pg.mkPen('k', width=max(pt2px(0.8), 1)))
        self.vb.setBackgroundColor('w')
        scene.addItem(self.vb)
        self.xaxis_item = _Axis('bottom', self)
        self.yaxis_item = _Axis('left', self)
        for a in (self.xaxis_item, self.yaxis_item):
            a.linkToView(self.vb)
            scene.addItem(a)
            a.apply_zoom(fig.font_zoom)
        self._side_axes = []           # right / secondary axes
        self._title = None
        self._title_size = 12.0
        self.title = _TitleProxy(self)
        self.transData = _Transform('data', self)
        self.transAxes = _Transform('axes', self)
        self.patch = _Patch(self)
        self._children = []            # artists in the order added
        self._axes_texts = []          # Text in axes coordinates (placed with the axes)
        self._legend = None
        self._visible = True
        self._axison = True
        self._frame = True
        self._cycle = 0
        self._tight = False
        self._xlabel = ''
        self._ylabel = ''
        self._colorbar = None
        self._is_colorbar = False
        self._home = None
        self._place_pending = False
        self._aspect = 'auto'
        self._shared_x = sharex
        self._shared_y = sharey
        if sharex is not None:
            self.vb.setXLink(sharex.vb)
        if sharey is not None:
            self.vb.setYLink(sharey.vb)
            # The shared y values are labelled once, on the axes they were shared from.
            self.yaxis_item.setStyle(showValues=False)
            self.yaxis_item._dirty = True
        self._apply_toolbar_mode(fig.canvas.toolbar.mode if fig.canvas.toolbar is not None else '')
        if inset_parent is not None:
            inset_parent._insets.append(self)
        fig._axes.append(self)
        self._place()

    # --- artists bookkeeping ---

    @property
    def images(self):
        return [a for a in self._children if isinstance(a, AxesImage)]

    @property
    def lines(self):
        return [a for a in self._children if isinstance(a, (Line2D, _RefLine))]

    @property
    def collections(self):
        return [a for a in self._children if isinstance(a, (LineCollection, PolyCollection, QuadMesh))]

    @property
    def texts(self):
        return [a for a in self._children if isinstance(a, Text)]

    def get_children(self):
        return list(self._children)

    def _register(self, artist):
        self._children.append(artist)
        if isinstance(artist, (AxesImage, QuadMesh, ContourSet)) and not self._tight:
            self._tight = True
            self.vb.setDefaultPadding(0.0)
        return artist

    def _forget(self, artist):
        try:
            self._children.remove(artist)
        except ValueError:
            pass
        try:
            self._axes_texts.remove(artist)
        except ValueError:
            pass

    def _next_color(self):
        c = TAB10[self._cycle % len(TAB10)]
        self._cycle += 1
        return c

    # --- geometry ---

    def get_position(self, original=False):
        if self._inset_parent is not None:
            px0, py0, pw, ph = self._inset_parent.get_position().bounds
            x, y, w, h = self._rect
            return Bbox(px0 + x * pw, py0 + y * ph, w * pw, h * ph)
        if self._rect is not None:
            return Bbox(*self._rect)
        if self._spec is not None:
            return self._spec.get_position(self.figure)
        return Bbox(0.125, 0.11, 0.775, 0.77)

    def set_position(self, pos, which='both'):
        b = pos.bounds if hasattr(pos, 'bounds') else tuple(pos)
        if self._inset_parent is not None:
            self._inset_parent = None
        self._rect = [float(v) for v in b]
        self._place()

    def get_subplotspec(self):
        return self._spec

    def _px_rect(self):
        W, H = self.figure._canvas_size()
        x, y, w, h = self.get_position().bounds
        r = (x * W, (1.0 - y - h) * H, w * W, h * H)
        if getattr(self, '_adjust_box', False) and self.vb.state['aspectLocked']:
            # matplotlib's adjustable='box': the box shrinks (about its centre) to the data's aspect.
            tx, ty = getattr(self, '_target_x', None), getattr(self, '_target_y', None)
            if tx is not None and ty is not None and tx[1] > tx[0] and ty[1] > ty[0] and r[2] > 0 and r[3] > 0:
                want = (ty[1] - ty[0]) / (tx[1] - tx[0]) * float(self.vb.state['aspectLocked'])
                bw, bh = r[2], r[3]
                if bh / bw > want:
                    nh = bw * want
                    r = (r[0], r[1] + 0.5 * (bh - nh), bw, nh)
                else:
                    nw = bh / want
                    r = (r[0] + 0.5 * (bw - nw), r[1], nw, bh)
        other = getattr(self, '_clear_of', None)
        if other is not None and other._visible:
            # This axes' left tick labels start right of the other axes' right-hand labels.
            ox, oy, ow, oh = other._px_rect()
            right = ox + ow + sum(a.maximumWidth() for a in other._side_axes if a.isVisible())
            need = right + self._clear_pad + (self.yaxis_item.maximumWidth() if self.yaxis_item.isVisible() else 0)
            if r[0] < need:
                r = (need, r[1], max(r[2] - (need - r[0]), 10.0), r[3])
        return r

    def _refont(self, zoom):
        """Every text of the axes at the figure's new zoom."""
        for a in [self.xaxis_item, self.yaxis_item] + self._side_axes:
            a.apply_zoom(zoom)
        if self._title is not None:
            self._title._apply()
        for t in self._axes_texts:
            t._apply()
        for a in self._children:
            if isinstance(a, Text):
                a._apply()
            elif isinstance(a, Annotation) and a.text is not None:
                a.text._apply()
        if self._legend is not None and getattr(self, '_legend_args', None) is not None:
            args, kw = self._legend_args
            self.legend(*args, **kw)
            if getattr(self, '_legend_clear', False):
                self.keep_legend_clear()

    def _refit_legend(self):
        if self._legend is None or getattr(self, '_legend_args', None) is None:
            return
        args, kw = self._legend_args
        self.legend(*args, **kw)
        if getattr(self, '_legend_clear', False):
            self.keep_legend_clear()

    def keep_clear_of(self, other, pad_pt=6.0):
        """Keep this axes (with its y tick labels and label) right of `other` and its colour-bar labels."""
        self._clear_of = other
        self._clear_pad = pt2px(pad_pt)
        other._dependents = getattr(other, '_dependents', []) + [self]
        self._place()

    def keep_legend_clear(self, pad_px=4.0):
        """
        The y limits raised so that no curve reaches the legend (an upper legend): the data's
        range with matplotlib's 5 % margin below, and above it room for the legend's height.
        """
        leg = self._legend
        ys = []
        for a in self._children:
            if isinstance(a, Line2D) and a.get_visible():
                y = a.get_ydata()
                y = y[np.isfinite(y)]
                if len(y):
                    ys.append((float(y.min()), float(y.max())))
        if leg is None or not ys:
            return
        lo, hi = min(v[0] for v in ys), max(v[1] for v in ys)
        span = (hi - lo) if hi > lo else (abs(hi) or 1.0)
        lo -= 0.05 * span
        hi += 0.02 * span
        x0, y0, w, h = self._px_rect()
        lh = float(leg.geometry().height()) + 4.0 + pad_px
        f = min(lh / max(h, 1.0), 0.8)
        top = (hi - f * lo) / (1.0 - f)
        self.vb.disableAutoRange(axis=pg.ViewBox.YAxis)
        self._target_y = (lo, top)
        self.vb.setYRange(lo, top, padding=0)
        self._store_home()
        self._legend_clear = True

    def get_window_extent(self, renderer=None):
        x0, y0, w, h = self._px_rect()
        W, H = self.figure._canvas_size()
        return Bbox(x0, H - (y0 + h), w, h)

    def _request_place(self):
        if not self._place_pending:
            self._place_pending = True
            QtCore.QTimer.singleShot(0, self._place)

    def _place(self):
        self._place_pending = False
        x0, y0, w, h = self._px_rect()
        w, h = max(w, 1.0), max(h, 1.0)
        self.vb.setGeometry(QtCore.QRectF(x0, y0, w, h))
        if self.vb.state['aspectLocked'] and getattr(self, '_target_x', None) is not None \
                and getattr(self, '_target_y', None) is not None and not self.figure.canvas.toolbar.mode:
            self._apply_targets()
        xa, ya = self.xaxis_item, self.yaxis_item
        for a in [xa, ya] + self._side_axes:
            a.picture = None
            a.update()
        aw = ya.maximumWidth() if ya.isVisible() else 0
        ya.setGeometry(QtCore.QRectF(x0 - aw, y0, aw, h))
        ah = xa.maximumHeight() if xa.isVisible() else 0
        xa.setGeometry(QtCore.QRectF(x0, y0 + h, w, ah))
        xr = x0 + w
        for a in self._side_axes:
            sw = a.maximumWidth() if a.isVisible() else 0
            a.setGeometry(QtCore.QRectF(xr, y0, sw, h))
            xr += sw
        if self._title is not None:
            lift = 15.0 if any(getattr(a, '_had_sci', False) for a in [self.yaxis_item] + self._side_axes) else 0.0
            base = getattr(self._title, '_base_offset', None)
            if base is None:
                base = self._title._base_offset = self._title.item._offset.y()
            self._title.item._offset = QtCore.QPointF(self._title.item._offset.x(), base - lift)
            self._title.item.updateTextPos()
            self._title._place()
        for t in self._axes_texts:
            t._place()
        for ins in self._insets:
            ins._place()
        for d in getattr(self, '_dependents', []):
            d._request_place()
        # A legend is refitted to the panel's new width (it never sticks out of the panel).
        if self._legend is not None and getattr(self, '_legend_args', None) is not None \
                and abs(getattr(self, '_legend_w', w) - w) > 2:
            QtCore.QTimer.singleShot(0, self._refit_legend)
        self._legend_w = w
        if getattr(self, '_legend_clear', False) and getattr(self, '_last_h', None) != round(h):
            self._last_h = round(h)
            QtCore.QTimer.singleShot(0, self.keep_legend_clear)

    def _view_range(self):
        if self.vb.state['autoRange'][0] or self.vb.state['autoRange'][1]:
            try:
                self.vb.updateAutoRange()
            except Exception:
                pass
        return self.vb.viewRange()

    # --- visibility, frame ---

    def _all_items(self):
        items = [self.vb, self.xaxis_item, self.yaxis_item] + list(self._side_axes)
        if self._title is not None:
            items.append(self._title.item)
        items.extend(t.item for t in self._axes_texts)
        return items

    def set_visible(self, flag):
        self._visible = bool(flag)
        self.vb.setVisible(self._visible)
        self.xaxis_item.setVisible(self._visible and self._axison)
        self.yaxis_item.setVisible(self._visible and self._axison and not self._is_colorbar)
        for a in self._side_axes:
            a.setVisible(self._visible)
        if self._title is not None:
            self._title.item.setVisible(self._visible and self._title.get_visible())
        for t in self._axes_texts:
            t.item.setVisible(self._visible and t.get_visible())
        for ins in self._insets:
            ins.set_visible(flag)
        self._place()

    def get_visible(self):
        return self._visible

    def set_frame_on(self, b):
        self._frame = bool(b)
        self.vb.setBorder(pg.mkPen('k', width=max(pt2px(0.8), 1)) if self._frame and self._axison else None)

    def axis(self, arg=None, **kw):
        if arg in ('off', False):
            self._axison = False
            self.xaxis_item.setVisible(False)
            self.yaxis_item.setVisible(False)
            self.vb.setBorder(None)
            self.vb.setBackgroundColor(None)
        elif arg in ('on', True):
            self._axison = True
            self.xaxis_item.setVisible(self._visible)
            self.yaxis_item.setVisible(self._visible)
            self.set_frame_on(self._frame)
            self.vb.setBackgroundColor('w')
        elif arg in ('equal', 'scaled', 'image'):
            self.set_aspect('equal')
        self._place()

    def set_axis_off(self):
        self.axis('off')

    def set_axis_on(self):
        self.axis('on')

    def set_navigate(self, b):
        pass

    def get_navigate(self):
        return True

    def remove(self):
        fig = self.figure
        sc = fig.canvas.scene()
        for ins in list(self._insets):
            ins.remove()
        if self._legend is not None:
            self._remove_legend()
        for it in self._all_items():
            try:
                sc.removeItem(it)
            except Exception:
                pass
        if self._inset_parent is not None:
            try:
                self._inset_parent._insets.remove(self)
            except ValueError:
                pass
        try:
            fig._axes.remove(self)
        except ValueError:
            pass
        fig._set_info(self, '')

    # --- clearing ---

    def clear(self):
        """Every artist, the title, the labels, the legend, the limits (autoscale on), and the aspect reset."""
        if self._legend is not None:
            self._remove_legend()
        for a in list(self._children):
            try:
                a.remove()
            except Exception:
                pass
        self._children = []
        for t in list(self._axes_texts):
            try:
                t.remove()
            except Exception:
                pass
        self._axes_texts = []
        self.vb.clear()
        if self._title is not None:
            try:
                self.figure.canvas.scene().removeItem(self._title.item)
            except Exception:
                pass
            self._title = None
            self.figure._set_info(self, '')
        sc = self.figure.canvas.scene()
        for a in self._side_axes:
            try:
                sc.removeItem(a)
            except Exception:
                pass
        self._side_axes = []
        self._cycle = 0
        self._tight = False
        self.vb.setDefaultPadding(0.05)
        self._is_colorbar = False
        self._colorbar = None
        for axis in (self.xaxis_item, self.yaxis_item):
            if getattr(axis, '_dirty', True):
                axis.setTicks(None)
                axis._mode = None
                axis.setStyle(showValues=not (axis is self.yaxis_item and self._shared_y is not None),
                              tickLength=int(round(pt2px(3.5))))
                axis.set_tick_pt(10)
                axis.setGrid(False)
                axis._dirty = False
            if axis.labelText:
                axis.setLabel('')
                axis.showLabel(False)
        self.xaxis_item.setVisible(self._visible and self._axison)
        self.yaxis_item.setVisible(self._visible and self._axison)
        self._xlabel = self._ylabel = ''
        self.vb.setAspectLocked(False)
        self._aspect = 'auto'
        self._adjust_box = False
        if self.vb.state['xInverted']:
            self.vb.invertX(False)
        if self.vb.state['yInverted']:
            self.vb.invertY(False)
        self.vb.enableAutoRange(x=True, y=True)
        self._home = None
        self._target_x = self._target_y = None
        self._legend_clear = False
        self._place()

    cla = clear

    # --- labels and title ---

    def set_title(self, label, fontsize=None, loc='center', pad=None, fontdict=None, color=None, y=None, **kw):
        if fontdict:
            fontsize = fontdict.get('fontsize', fontsize)
        if self._title is not None:
            self._title.remove()
            self._title = None
        full = '' if label is None else str(label)
        label = short_title(full)
        self.figure._set_info(self, full if full.strip() != label else '')
        size = font_pt(fontsize, self._title_size) if fontsize is not None else 12.0
        self._title_size = size
        ha = {'left': 'left', 'right': 'right'}.get(loc, 'center')
        x = {'left': 0.0, 'right': 1.0}.get(loc, 0.5)
        pad_px = pt2px(6.0 if pad is None else pad)
        t = Text(self, x, 1.0 if y is None else y, label, 'axes', fontsize=size, ha=ha, va='bottom',
                 color=color or kw.get('c', 'k'), offset=(0, pad_px), fontweight=kw.get('fontweight'))
        try:
            self._axes_texts.remove(t)
        except ValueError:
            pass
        try:
            self._children.remove(t)
        except ValueError:
            pass
        t.item.setVisible(self._visible)
        self._title = t
        return t

    def get_title(self, loc='center'):
        return self._title.get_text() if self._title is not None else ''

    def _axis_label(self, axis, text, fontsize=None, color=None):
        text = '' if text is None else str(text)
        if not text:
            axis.setLabel('')
            axis.showLabel(False)
            return
        axis.setLabel(html.escape(short_label(text)).replace('\n', '<br>'))
        axis.set_label_pt(10 if fontsize is None else fontsize)
        if color is not None:
            axis.label.setDefaultTextColor(qcolor(color))
        axis.showLabel(True)
        axis._adjustSize()

    def set_xlabel(self, label, fontsize=None, labelpad=None, color=None, **kw):
        self._xlabel = '' if label is None else str(label)
        self._axis_label(self.xaxis_item, label, fontsize or kw.get('size'), color)

    def set_ylabel(self, label, fontsize=None, labelpad=None, color=None, **kw):
        self._ylabel = '' if label is None else str(label)
        self._axis_label(self.yaxis_item, label, fontsize or kw.get('size'), color)

    def get_xlabel(self):
        return self._xlabel

    def get_ylabel(self):
        return self._ylabel

    def _make_side_axis(self, side='right'):
        a = _Axis(side, self)
        a.apply_zoom(self.figure.font_zoom)
        a.linkToView(self.vb)
        self.figure.canvas.scene().addItem(a)
        self._side_axes.append(a)
        a.setVisible(self._visible)
        self._place()
        return a

    def _side_label(self, axis, text, fontsize=None):
        self._axis_label(axis, text, fontsize)
        self._place()

    # --- ticks ---

    def set_xticks(self, ticks, labels=None, **kw):
        self._set_ticks(self.xaxis_item, ticks, labels)

    def set_yticks(self, ticks, labels=None, **kw):
        self._set_ticks(self.yaxis_item, ticks, labels)

    def _set_ticks(self, axis, ticks, labels):
        axis._dirty = True
        ticks = list(np.ravel(ticks)) if ticks is not None else []
        if len(ticks) == 0:
            axis.setTicks([[]])
            axis.setStyle(showValues=False, tickLength=0)
        else:
            labs = labels if labels is not None else [axis.tickStrings([t], 1.0, abs(ticks[-1] - ticks[0]) / max(len(ticks) - 1, 1) or 1.0)[0]
                                                      for t in ticks]
            axis.setTicks([[(float(t), str(s)) for t, s in zip(ticks, labs)]])
        axis._adjustSize()
        self._place()

    def set_xticklabels(self, labels, **kw):
        self.xaxis_item._dirty = True
        if len(list(labels)) == 0:
            self.xaxis_item.setStyle(showValues=False)
            self.xaxis_item._adjustSize()
            self._place()

    def set_yticklabels(self, labels, **kw):
        self.yaxis_item._dirty = True
        if len(list(labels)) == 0:
            self.yaxis_item.setStyle(showValues=False)
            self.yaxis_item._adjustSize()
            self._place()

    def tick_params(self, axis='both', labelsize=None, labelleft=None, labelbottom=None, labelright=None,
                    left=None, bottom=None, length=None, **kw):
        self.xaxis_item._dirty = self.yaxis_item._dirty = True
        targets = []
        if axis in ('both', 'x'):
            targets.append(self.xaxis_item)
        if axis in ('both', 'y'):
            targets.append(self.yaxis_item)
            targets.extend(self._side_axes)
        if labelsize is not None:
            for a in targets:
                a.set_tick_pt(labelsize)
                a._adjustSize()
        if labelleft is not None:
            self.yaxis_item.setStyle(showValues=bool(labelleft))
            self.yaxis_item._adjustSize()
        if labelbottom is not None:
            self.xaxis_item.setStyle(showValues=bool(labelbottom))
            self.xaxis_item._adjustSize()
        if length is not None:
            for a in targets:
                a.setStyle(tickLength=int(round(pt2px(length))))
        if left is False:
            self.yaxis_item.setStyle(tickLength=0)
            self.yaxis_item._adjustSize()
        if bottom is False:
            self.xaxis_item.setStyle(tickLength=0)
            self.xaxis_item._adjustSize()
        self._place()

    def ticklabel_format(self, **kw):
        pass

    def grid(self, visible=True, which='major', axis='both', alpha=None, **kw):
        if visible is None:
            visible = True
        # matplotlib's grid is light grey (#b0b0b0) at the given alpha over white
        val = int(255 * 0.45 * (1.0 if alpha is None else float(alpha))) if visible else False
        self.xaxis_item._dirty = self.yaxis_item._dirty = True
        if axis in ('both', 'x'):
            self.xaxis_item.setGrid(val)
        if axis in ('both', 'y'):
            self.yaxis_item.setGrid(val)

    def minorticks_off(self):
        pass

    # --- limits and aspect ---

    def _range_of(self, args, kw, lo_key, hi_key):
        if len(args) == 1 and np.ndim(args[0]) == 1:
            lo, hi = args[0]
        elif len(args) >= 2:
            lo, hi = args[0], args[1]
        elif len(args) == 1:
            lo, hi = args[0], None
        else:
            lo, hi = None, None
        lo = kw.get(lo_key, kw.get('left' if lo_key == 'xmin' else 'bottom', lo))
        hi = kw.get(hi_key, kw.get('right' if hi_key == 'xmax' else 'top', hi))
        return lo, hi

    def set_xlim(self, *args, **kw):
        lo, hi = self._range_of(args, kw, 'xmin', 'xmax')
        cur = self._view_range()[0]
        lo = cur[0] if lo is None else float(lo)
        hi = cur[1] if hi is None else float(hi)
        inv = hi < lo
        if inv != bool(self.vb.state['xInverted']):
            self.vb.invertX(inv)
        a, b = (hi, lo) if inv else (lo, hi)
        if b == a:
            a, b = a - 0.5, b + 0.5
        r = _finite_range(a, b)
        if r is None:
            print(f"qtplot: x limits ({lo}, {hi}) are not drawable; kept as they were.")
            return self.get_xlim()
        a, b = r
        self.vb.disableAutoRange(axis=pg.ViewBox.XAxis)
        self._target_x = (a, b)
        self._apply_targets()
        return (lo, hi)

    def set_ylim(self, *args, **kw):
        lo, hi = self._range_of(args, kw, 'ymin', 'ymax')
        cur = self._view_range()[1]
        lo = cur[0] if lo is None else float(lo)
        hi = cur[1] if hi is None else float(hi)
        inv = hi < lo
        if inv != bool(self.vb.state['yInverted']):
            self.vb.invertY(inv)
        a, b = (hi, lo) if inv else (lo, hi)
        if b == a:
            a, b = a - 0.5, b + 0.5
        r = _finite_range(a, b)
        if r is None:
            print(f"qtplot: y limits ({lo}, {hi}) are not drawable; kept as they were.")
            return self.get_ylim()
        a, b = r
        self.vb.disableAutoRange(axis=pg.ViewBox.YAxis)
        self._target_y = (a, b)
        self._apply_targets()
        return (lo, hi)

    def _apply_targets(self):
        """
        The limits set: with an equal aspect both ranges at once, the one too
        short for the box widened about its centre (matplotlib's 'datalim'),
        so neither requested range is cut.
        """
        tx = getattr(self, '_target_x', None)
        ty = getattr(self, '_target_y', None)
        tx = None if tx is None else _finite_range(*tx)
        ty = None if ty is None else _finite_range(*ty)
        if self.vb.state['aspectLocked'] and tx is not None and ty is not None:
            x0, y0, w, h = self._px_rect()
            ratio = float(self.vb.state['aspectLocked'])
            dx, dy = tx[1] - tx[0], ty[1] - ty[0]
            if w > 0 and h > 0 and dx > 0 and dy > 0:
                # data units per pixel needed on each axis
                ux, uy = dx / w, dy / (h / ratio) if False else dy / h
                u = max(ux, uy)
                cx, cy = 0.5 * (tx[0] + tx[1]), 0.5 * (ty[0] + ty[1])
                tx = (cx - 0.5 * u * w, cx + 0.5 * u * w)
                ty = (cy - 0.5 * u * h, cy + 0.5 * u * h)
            self.vb.setRange(xRange=tx, yRange=ty, padding=0)
            if getattr(self, '_adjust_box', False):
                self._place_box_only()
        elif tx is not None and ty is not None:
            self.vb.setRange(xRange=tx, yRange=ty, padding=0)
        elif tx is not None:
            self.vb.setXRange(*tx, padding=0)
        elif ty is not None:
            self.vb.setYRange(*ty, padding=0)
        self._store_home()

    def _store_home(self):
        (xa, xb), (ya, yb) = self.vb.viewRange()
        self._home = (xa, xb, ya, yb)

    def get_xlim(self):
        a, b = self._view_range()[0]
        return (b, a) if self.vb.state['xInverted'] else (a, b)

    def get_ylim(self):
        a, b = self._view_range()[1]
        return (b, a) if self.vb.state['yInverted'] else (a, b)

    def set_aspect(self, aspect, adjustable=None, anchor=None, share=False):
        self._aspect = aspect
        was = getattr(self, '_adjust_box', False)
        self._adjust_box = adjustable in (None, 'box')
        if aspect == 'equal' or (isinstance(aspect, (int, float)) and not isinstance(aspect, bool)):
            self.vb.setAspectLocked(True, 1.0 if aspect == 'equal' else float(aspect))
        else:
            self.vb.setAspectLocked(False)

    def _place_box_only(self):
        x0, y0, w, h = self._px_rect()
        if abs(self.vb.geometry().width() - w) > 0.5 or abs(self.vb.geometry().height() - h) > 0.5 \
                or abs(self.vb.geometry().x() - x0) > 0.5 or abs(self.vb.geometry().y() - y0) > 0.5:
            self._request_place()

    def get_aspect(self):
        return self._aspect

    def autoscale(self, enable=True, axis='both', tight=None):
        if enable:
            self.vb.enableAutoRange(x=axis in ('both', 'x'), y=axis in ('both', 'y'))

    def autoscale_view(self, *a, **k):
        pass

    def relim(self, *a, **k):
        pass

    def invert_yaxis(self):
        self.vb.invertY(not self.vb.state['yInverted'])

    def invert_xaxis(self):
        self.vb.invertX(not self.vb.state['xInverted'])

    def _home_view(self):
        if self._is_colorbar:
            return
        cb = getattr(self, 'home_callback', None)
        if cb is not None:
            # An axes whose view a controller owns (the browser's scan area, widgets.ViewNavigator):
            # the controller's own home (its full extent).
            cb()
            return
        if self._home is not None:
            xa, xb, ya, yb = self._home
            self.vb.setRange(xRange=(xa, xb), yRange=(ya, yb), padding=0)
        else:
            self.vb.enableAutoRange(x=True, y=True)

    def _apply_toolbar_mode(self, mode):
        if self._is_colorbar:
            self.vb.setMouseEnabled(False, False)
            return
        on = bool(mode)
        self.vb.setMouseEnabled(on, on)
        if mode == 'zoom rect':
            self.vb.setMouseMode(pg.ViewBox.RectMode)
        else:
            self.vb.setMouseMode(pg.ViewBox.PanMode)

    # --- plotting ---

    @staticmethod
    def _parse_fmt(fmt):
        style = {}
        s = fmt
        for ls in ('--', '-.', '-', ':'):
            if ls in s:
                style['ls'] = ls
                s = s.replace(ls, '', 1)
                break
        if len(s) >= 2 and s[0] == 'C' and s[1].isdigit():
            style['color'] = s[:2]
            s = s[2:]
        for ch in s:
            if ch in _BASE:
                style['color'] = ch
            elif ch in _MARKERS:
                style['marker'] = ch
        if 'marker' in style and 'ls' not in style:
            style['ls'] = 'none'
        return style

    def plot(self, *args, **kw):
        """Curves and markers (matplotlib's plot): x, y[, fmt] groups; a 2D y draws one curve per column."""
        groups = []
        a = list(args)
        while a:
            if len(a) >= 2 and not isinstance(a[1], str):
                x, y = a[0], a[1]
                a = a[2:]
            else:
                y = a[0]
                x = None
                a = a[1:]
            fmt = None
            if a and isinstance(a[0], str):
                fmt = a[0]
                a = a[1:]
            groups.append((x, y, fmt))
        out = []
        for x, y, fmt in groups:
            out.extend(self._plot_one(x, y, fmt, dict(kw)))
        return out

    def _plot_one(self, x, y, fmt, kw):
        style = self._parse_fmt(fmt) if fmt else {}
        color = kw.pop('color', kw.pop('c', style.get('color')))
        ls = kw.pop('ls', kw.pop('linestyle', style.get('ls', '-')))
        lw = kw.pop('lw', kw.pop('linewidth', 1.5))
        marker = kw.pop('marker', style.get('marker'))
        ms = kw.pop('ms', kw.pop('markersize', 6.0))
        mec = kw.pop('mec', kw.pop('markeredgecolor', None))
        mfc = kw.pop('mfc', kw.pop('markerfacecolor', None))
        mew = kw.pop('mew', kw.pop('markeredgewidth', 1.0))
        alpha = kw.pop('alpha', None)
        label = kw.pop('label', None)
        zorder = kw.pop('zorder', None)
        picker = kw.pop('picker', None)
        visible = kw.pop('visible', True)
        yv = np.asarray(np.ma.filled(np.ma.asarray(y, dtype=np.float64), np.nan), dtype=np.float64)
        if x is None:
            xv = np.arange(yv.shape[0], dtype=np.float64)
        else:
            xv = np.asarray(np.ma.filled(np.ma.asarray(x, dtype=np.float64), np.nan), dtype=np.float64)
        if yv.ndim <= 1:
            if color is None:
                color = self._next_color()
            ln = Line2D(self, xv.ravel(), yv.ravel(), color=color, lw=lw, ls=ls, marker=marker, ms=ms, mec=mec,
                        mfc=mfc, mew=mew, alpha=alpha, label=label, zorder=zorder, picker=picker, visible=visible)
            return [self._register(ln)]
        # A 2D y: one curve per column.  Columns of one colour are drawn as one item (NaN breaks), so a
        # grid's thousand spectra are ten items instead of a thousand.
        n = yv.shape[1]
        if n == 0:
            return []
        cols = [color] * n if color is not None else [None] * n
        if color is None:
            cols = [self._next_color() for _ in range(n)]
        by_color = {}
        for j, c in enumerate(cols):
            by_color.setdefault(c, []).append(j)
        out = []
        xcol = xv if xv.ndim == 1 else None
        for c, js in by_color.items():
            parts_x, parts_y = [], []
            for j in js:
                parts_x.append(xcol if xcol is not None else xv[:, j])
                parts_y.append(yv[:, j])
                parts_x.append([np.nan])
                parts_y.append([np.nan])
            ln = Line2D(self, np.concatenate(parts_x), np.concatenate(parts_y), color=c, lw=lw, ls=ls, marker=marker,
                        ms=ms, mec=mec, mfc=mfc, mew=mew, alpha=alpha, label=label, zorder=zorder, picker=picker,
                        visible=visible)
            out.append(self._register(ln))
        return out

    def scatter(self, x, y, s=None, c=None, marker='o', label=None, zorder=None, **kw):
        ms = math.sqrt(s) if s is not None else 6.0
        return self.plot(x, y, ls='none', marker=marker, ms=ms, color=c if isinstance(c, str) else None,
                         label=label, zorder=zorder)[0]

    def axhline(self, y=0.0, xmin=0, xmax=1, **kw):
        return self._register(_RefLine(self, y, True, color=kw.get('color', kw.get('c', 'C0')),
                                       lw=kw.get('lw', kw.get('linewidth', 1.5)), ls=kw.get('ls', kw.get('linestyle', '-')),
                                       alpha=kw.get('alpha'), label=kw.get('label'), zorder=kw.get('zorder'),
                                       picker=kw.get('picker')))

    def axvline(self, x=0.0, ymin=0, ymax=1, **kw):
        return self._register(_RefLine(self, x, False, color=kw.get('color', kw.get('c', 'C0')),
                                       lw=kw.get('lw', kw.get('linewidth', 1.5)), ls=kw.get('ls', kw.get('linestyle', '-')),
                                       alpha=kw.get('alpha'), label=kw.get('label'), zorder=kw.get('zorder'),
                                       picker=kw.get('picker')))

    def axvspan(self, xmin, xmax, **kw):
        return self._register(_Span(self, xmin, xmax, True, color=kw.get('color', kw.get('fc', 'C0')),
                                    alpha=kw.get('alpha'), label=kw.get('label'), zorder=kw.get('zorder')))

    def axhspan(self, ymin, ymax, **kw):
        return self._register(_Span(self, ymin, ymax, False, color=kw.get('color', kw.get('fc', 'C0')),
                                    alpha=kw.get('alpha'), label=kw.get('label'), zorder=kw.get('zorder')))

    def imshow(self, X, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=None, vmax=None,
               origin=None, extent=None, zorder=None, label=None, **kw):
        im = AxesImage(self, X, cmap=cmap, norm=norm, extent=extent, origin=origin or 'upper', alpha=alpha,
                       zorder=zorder, label=label, vmin=vmin, vmax=vmax)
        self._register(im)
        if aspect is None or aspect == 'equal':
            self.set_aspect('equal')
        elif aspect != 'auto':
            self.set_aspect(aspect)
        if extent is None and (origin or 'upper') == 'upper' and not self.vb.state['yInverted']:
            self.vb.invertY(True)
        return im

    def pcolormesh(self, *args, cmap=None, norm=None, vmin=None, vmax=None, shading='flat', zorder=None,
                   alpha=None, **kw):
        if len(args) == 1:
            C = np.asarray(np.ma.filled(np.ma.asarray(args[0], dtype=np.float64), np.nan))
            X = np.arange(C.shape[1] + 1, dtype=np.float64)
            Y = np.arange(C.shape[0] + 1, dtype=np.float64)
        else:
            X, Y, C = args[:3]
        return self._register(QuadMesh(self, X, Y, C, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, zorder=zorder,
                                       alpha=alpha))

    def contourf(self, X, Y, Z, levels=10, cmap=None, zorder=None, **kw):
        return self._register(ContourSet(self, X, Y, Z, levels=levels, cmap=cmap, zorder=zorder))

    def add_collection(self, coll, autolim=True):
        coll._attach(self)
        return self._register(coll)

    def add_artist(self, a):
        return a

    def text(self, x, y, s, transform=None, **kw):
        coords = 'data'
        if transform is not None:
            kind = getattr(transform, 'kind', 'data')
            coords = {'axes': 'axes', 'figure': 'figure'}.get(kind, 'data')
        owner = self.figure if coords == 'figure' else self
        t = Text(owner, x, y, s, coords, **kw)
        if coords == 'figure':
            self.figure._texts.append(t)
            return t
        if coords == 'axes':
            self._axes_texts.append(t)
            t.item.setVisible(self._visible and t.get_visible())
        self._children.append(t)
        return t

    def annotate(self, text, xy, xytext=None, textcoords=None, arrowprops=None, **kw):
        return self._register(Annotation(self, text, xy, xytext=xytext, textcoords=textcoords,
                                         arrowprops=arrowprops, color=kw.get('color', 'k'),
                                         fontsize=kw.get('fontsize'), fontweight=kw.get('fontweight'),
                                         zorder=kw.get('zorder'), ha=kw.get('ha', 'left'), va=kw.get('va', 'baseline')))

    def secondary_yaxis(self, location='right', functions=None, **kw):
        a = self._make_side_axis('right')
        if functions is not None:
            a._mode = ('func', functions[0], functions[1])
        return SecondaryAxis(self, a)

    def inset_axes(self, bounds, **kw):
        return Axes(self.figure, rect=bounds, inset_parent=self)

    # --- legend ---

    def get_legend_handles_labels(self):
        handles, labels = [], []
        for a in self._children:
            lab = a.get_label()
            if isinstance(a, (Text, AxesImage)) or not lab or lab.startswith('_'):
                continue
            handles.append(a)
            labels.append(lab)
        return handles, labels

    def _remove_legend(self):
        leg = self._legend
        self._legend = None
        if leg is None:
            return
        try:
            leg.setParentItem(None)
            if leg.scene() is not None:
                leg.scene().removeItem(leg)
        except Exception:
            pass

    def get_legend(self):
        return self._legend

    def legend(self, *args, loc='best', frameon=True, fontsize=None, ncol=1, **kw):
        if len(args) >= 2:
            handles, labels = list(args[0]), list(args[1])
        elif len(args) == 1:
            labels = list(args[0])
            handles = [a for a in self._children if not isinstance(a, (Text, AxesImage))][:len(labels)]
        else:
            handles, labels = self.get_legend_handles_labels()
        if self._legend is not None:
            self._remove_legend()
        pairs = [(h, l) for h, l in zip(handles, labels) if l and not str(l).startswith('_')]
        if not pairs:
            return None
        self._legend_args = (args, dict(loc=loc, frameon=frameon, fontsize=fontsize, ncol=ncol))
        size = font_pt(fontsize, 10.0) * self.figure.font_zoom
        leg = pg.LegendItem(offset=None, verSpacing=0, horSpacing=4,
                            labelTextSize=f"{pt2px(size) * FONT_SCALE:.1f}px", colCount=int(ncol), frame=False,
                            pen=None if not frameon else pg.mkPen((204, 204, 204)),
                            brush=None if not frameon else pg.mkBrush(255, 255, 255, 204))
        leg.layout.setContentsMargins(2, 2, 2, 2)
        leg.setParentItem(self.vb)
        rows_h, widths = [], []
        # The legend's text is a real font at the figure's zoom (an HTML size is scaled differently by
        # Qt on high-DPI displays), shrunk when needed so the legend fits inside its panel.
        base_pt = font_pt(fontsize, 10.0)
        fnt = qfont(base_pt, zoom=self.figure.font_zoom)
        fm = QtGui.QFontMetricsF(fnt)
        text_w = max((fm.horizontalAdvance(str(l)) for _, l in pairs), default=0.0)
        avail = self._px_rect()[2] / max(int(ncol), 1) - 26 - 16
        if text_w > avail > 0:
            fnt = qfont(base_pt, zoom=self.figure.font_zoom * max(0.45, avail / text_w))
            fm = QtGui.QFontMetricsF(fnt)
        row_px = fm.height() * 1.15
        for h, l in pairs:
            sample = h._legend_sample() if hasattr(h, '_legend_sample') else None
            if sample is None:
                continue
            lab = pg.LabelItem('', justify='left')
            lab.item.setFont(fnt)
            lab.item.setDefaultTextColor(QtGui.QColor('black'))
            lab.item.setPlainText(str(l))
            lab.item.document().setDocumentMargin(1)
            row_h = int(math.ceil(row_px))
            lab_w = int(math.ceil(fm.horizontalAdvance(str(l)))) + 6
            lab.setFixedHeight(row_h)                 # fixed rows: the legend never changes its spacing
            lab.setFixedWidth(lab_w)
            sample.setFixedHeight(row_h)
            rows_h.append(row_h)
            widths.append(lab_w + 26 + 4)
            row = leg.layout.rowCount()
            col = 0
            if leg.columnCount > 1:
                n = len(leg.items)
                row, col = n // leg.columnCount, 2 * (n % leg.columnCount)
            leg.items.append((sample, lab))
            leg.layout.addItem(sample, row, col)
            leg.layout.addItem(lab, row, col + 1)
        # The legend's size fixed from its rows (pyqtgraph's updateSize re-measures rows that the
        # layout has stretched, so every call made the legend taller and its rows farther apart).
        leg.layout.setVerticalSpacing(0)
        leg.layout.setContentsMargins(2, 2, 2, 2)
        ncol = max(int(ncol), 1)
        n_rows = int(math.ceil(len(rows_h) / ncol)) if rows_h else 0
        lw = (max(widths) if widths else 0) * ncol + 4
        lh = (max(rows_h) if rows_h else 0) * n_rows + 4
        leg.updateSize = lambda: None
        leg.setMinimumSize(lw, lh)
        leg.setMaximumSize(lw, lh)
        leg.setGeometry(QtCore.QRectF(0, 0, lw, lh))
        loc = {'best': 'upper right', 0: 'upper right', 1: 'upper right', 2: 'upper left', 3: 'lower left',
               4: 'lower right'}.get(loc, loc)
        anchors = {'upper right': ((1, 0), (-4, 4)), 'upper left': ((0, 0), (4, 4)),
                   'lower left': ((0, 1), (4, -4)), 'lower right': ((1, 1), (-4, -4)),
                   'center': ((0.5, 0.5), (0, 0)), 'upper center': ((0.5, 0), (0, 4)),
                   'lower center': ((0.5, 1), (0, -4)), 'center left': ((0, 0.5), (4, 0)),
                   'center right': ((1, 0.5), (-4, 0)), 'right': ((1, 0.5), (-4, 0))}
        pos, off = anchors.get(loc, anchors['upper right'])
        leg.anchor(itemPos=pos, parentPos=pos, offset=off)
        leg.setZValue(100)
        self._legend = leg
        return leg


# ---------------------------------------------------------------------- #
#  Grids of subplots                                                       #
# ---------------------------------------------------------------------- #

class SubplotParams:
    def __init__(self):
        self.left, self.right, self.bottom, self.top = 0.125, 0.9, 0.11, 0.88
        self.wspace, self.hspace = 0.2, 0.2


class _GridSpecBase:
    def __init__(self, nrows, ncols, width_ratios=None, height_ratios=None, wspace=None, hspace=None):
        self.nrows, self.ncols = int(nrows), int(ncols)
        self.width_ratios = list(width_ratios) if width_ratios is not None else [1.0] * self.ncols
        self.height_ratios = list(height_ratios) if height_ratios is not None else [1.0] * self.nrows
        self.wspace, self.hspace = wspace, hspace

    def get_geometry(self):
        return self.nrows, self.ncols

    def _bounds(self, fig):
        raise NotImplementedError

    def get_grid_positions(self, fig):
        """(bottoms, tops, lefts, rights) of the cells, matplotlib's algorithm."""
        left, right, bottom, top, wspace, hspace = self._bounds(fig)
        tot_w, tot_h = right - left, top - bottom
        cell_h = tot_h / (self.nrows + hspace * (self.nrows - 1))
        sep_h = hspace * cell_h
        norm = cell_h * self.nrows / sum(self.height_ratios)
        cell_heights = [r * norm for r in self.height_ratios]
        sep_heights = [0] + [sep_h] * (self.nrows - 1)
        cell_hs = np.cumsum(np.column_stack([sep_heights, cell_heights]).flat)
        cell_w = tot_w / (self.ncols + wspace * (self.ncols - 1))
        sep_w = wspace * cell_w
        norm = cell_w * self.ncols / sum(self.width_ratios)
        cell_widths = [r * norm for r in self.width_ratios]
        sep_widths = [0] + [sep_w] * (self.ncols - 1)
        cell_ws = np.cumsum(np.column_stack([sep_widths, cell_widths]).flat)
        fig_tops, fig_bottoms = (top - cell_hs).reshape((-1, 2)).T
        fig_lefts, fig_rights = (left + cell_ws).reshape((-1, 2)).T
        return fig_bottoms, fig_tops, fig_lefts, fig_rights

    def __getitem__(self, key):
        if isinstance(key, tuple):
            r, c = key
        else:
            if isinstance(key, slice):
                idx = range(self.nrows * self.ncols)[key]
                rows = sorted({i // self.ncols for i in idx})
                cols = sorted({i % self.ncols for i in idx})
                return SubplotSpec(self, slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
            k = int(key)
            if k < 0:
                k += self.nrows * self.ncols
            r, c = divmod(k, self.ncols)
        rs = r if isinstance(r, slice) else slice(int(r) % self.nrows, int(r) % self.nrows + 1)
        cs = c if isinstance(c, slice) else slice(int(c) % self.ncols, int(c) % self.ncols + 1)
        return SubplotSpec(self, rs, cs)


class GridSpec(_GridSpecBase):
    def __init__(self, nrows, ncols, figure=None, left=None, bottom=None, right=None, top=None, wspace=None,
                 hspace=None, width_ratios=None, height_ratios=None):
        super().__init__(nrows, ncols, width_ratios, height_ratios, wspace, hspace)
        self.figure = figure
        self.left, self.bottom, self.right, self.top = left, bottom, right, top

    def _bounds(self, fig):
        sp = fig.subplotpars
        return (sp.left if self.left is None else self.left, sp.right if self.right is None else self.right,
                sp.bottom if self.bottom is None else self.bottom, sp.top if self.top is None else self.top,
                sp.wspace if self.wspace is None else self.wspace, sp.hspace if self.hspace is None else self.hspace)


class GridSpecFromSubplotSpec(_GridSpecBase):
    def __init__(self, nrows, ncols, subplot_spec, wspace=None, hspace=None, height_ratios=None, width_ratios=None):
        super().__init__(nrows, ncols, width_ratios, height_ratios, wspace, hspace)
        self._parent = subplot_spec

    def _bounds(self, fig):
        b = self._parent.get_position(fig)
        sp = fig.subplotpars
        return (b.x0, b.x1, b.y0, b.y1, sp.wspace if self.wspace is None else self.wspace,
                sp.hspace if self.hspace is None else self.hspace)


class SubplotSpec:
    def __init__(self, gridspec, rows, cols):
        self._gs, self.rows, self.cols = gridspec, rows, cols

    def get_gridspec(self):
        return self._gs

    def get_position(self, fig):
        bottoms, tops, lefts, rights = self._gs.get_grid_positions(fig)
        r = range(self._gs.nrows)[self.rows]
        c = range(self._gs.ncols)[self.cols]
        x0, x1 = min(lefts[list(c)]), max(rights[list(c)])
        y0, y1 = min(bottoms[list(r)]), max(tops[list(r)])
        return Bbox(x0, y0, x1 - x0, y1 - y0)


gridspec = types.SimpleNamespace(GridSpec=GridSpec, GridSpecFromSubplotSpec=GridSpecFromSubplotSpec,
                                 SubplotSpec=SubplotSpec)


# ---------------------------------------------------------------------- #
#  Events                                                                  #
# ---------------------------------------------------------------------- #

class MouseButton:
    LEFT, MIDDLE, RIGHT, BACK, FORWARD = 1, 2, 3, 8, 9


class MouseEvent:
    """matplotlib's MouseEvent: x, y in display pixels from the bottom left, inaxes, xdata, ydata, button, dblclick."""

    def __init__(self, name, canvas, x, y, button=None, dblclick=False, step=0, guiEvent=None, key=None):
        self.name = name
        self.canvas = canvas
        self.x, self.y = x, y
        self.button = button
        self.dblclick = dblclick
        self.step = step
        self.guiEvent = guiEvent
        self.key = key
        self.inaxes = None
        self.xdata = self.ydata = None


class KeyEvent:
    """
    matplotlib's KeyEvent: key (matplotlib's names: 'r', ' ', 'home', 'left', 'ctrl+left',
    'shift+left', 'tab', 'escape', 'delete', ...), x, y (the cursor in display pixels from the
    bottom left), inaxes, xdata, ydata (the axes under the cursor).  A callback that uses the key
    sets `handled = True`; the canvas then consumes the Qt event.
    """

    def __init__(self, name, canvas, key, x=None, y=None, guiEvent=None):
        self.name = name
        self.canvas = canvas
        self.key = key
        self.x, self.y = x, y
        self.guiEvent = guiEvent
        self.inaxes = None
        self.xdata = self.ydata = None
        self.handled = False


_QT_KEY_NAMES = None


def _key_name(ev):
    """A Qt key event as matplotlib's key name (None for a bare modifier)."""
    global _QT_KEY_NAMES
    K = Qt.Key
    if _QT_KEY_NAMES is None:
        _QT_KEY_NAMES = {K.Key_Space: ' ', K.Key_Home: 'home', K.Key_End: 'end', K.Key_Left: 'left',
                         K.Key_Right: 'right', K.Key_Up: 'up', K.Key_Down: 'down', K.Key_Tab: 'tab',
                         K.Key_Backtab: 'tab', K.Key_Escape: 'escape', K.Key_Delete: 'delete',
                         K.Key_Backspace: 'backspace', K.Key_Return: 'enter', K.Key_Enter: 'enter',
                         K.Key_PageUp: 'pageup', K.Key_PageDown: 'pagedown'}
    k = ev.key()
    if k in (K.Key_Shift, K.Key_Control, K.Key_Alt, K.Key_Meta):
        return None
    name = _QT_KEY_NAMES.get(k)
    if name is None:
        t = ev.text()
        if not t or not t.isprintable():
            return None
        name = t.lower()
    mods = ev.modifiers()
    M = Qt.KeyboardModifier
    prefix = ''
    if mods & M.ControlModifier:
        prefix += 'ctrl+'
    if mods & M.AltModifier:
        prefix += 'alt+'
    if (mods & M.ShiftModifier) and (len(name) > 1 or name == ' '):
        prefix += 'shift+'
    return prefix + name


class PickEvent:
    def __init__(self, canvas, mouseevent, artist):
        self.name = 'pick_event'
        self.canvas = canvas
        self.mouseevent = mouseevent
        self.artist = artist
        self.ind = []


class CloseEvent:
    def __init__(self, canvas):
        self.name = 'close_event'
        self.canvas = canvas


class _Toolbar:
    """The figure's navigation state: mode '' (the windows' own mouse handling), 'pan/zoom', or 'zoom rect'."""

    def __init__(self):
        self.mode = ''


_QT_BUTTONS = {Qt.MouseButton.LeftButton: 1, Qt.MouseButton.MiddleButton: 2, Qt.MouseButton.RightButton: 3,
               Qt.MouseButton.BackButton: 8, Qt.MouseButton.ForwardButton: 9}


class _PosOnly:
    """A stand-in event holding only a position (the canvas's _event reads position())."""

    def __init__(self, p):
        self._p = p

    def position(self):
        return self._p


class FigureCanvas(pg.GraphicsView):
    """
    The figure's drawing surface: one pyqtgraph scene in pixel coordinates,
    the Qt controls as child widgets, and matplotlib's event connection
    (mpl_connect / mpl_disconnect, draw_idle).  Mouse motion is coalesced:
    a burst of moves during a slow redraw reaches the handlers as one event
    at the latest position.
    """

    EVENTS = ('button_press_event', 'button_release_event', 'motion_notify_event', 'scroll_event',
              'pick_event', 'close_event', 'resize_event', 'key_press_event', 'key_release_event',
              'figure_enter_event', 'figure_leave_event', 'axes_enter_event', 'axes_leave_event', 'draw_event')

    def __init__(self, figure, w_px, h_px):
        super().__init__(background='w')
        self.figure = figure
        self._size_px = (int(w_px), int(h_px))
        self._cbs = {name: {} for name in self.EVENTS}
        self._cid = itertools.count(1)
        self.toolbar = _Toolbar()
        self.manager = None
        self._pending_motion = None
        self._motion_scheduled = False
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.resize(int(w_px), int(h_px))

    def sizeHint(self):
        return QtCore.QSize(*self._size_px)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        fig = getattr(self, 'figure', None)
        if fig is not None and getattr(fig, 'manager', None) is not None:
            fig._relayout()

    # --- matplotlib's connection API ---

    def mpl_connect(self, name, func):
        cid = next(self._cid)
        self._cbs.setdefault(name, {})[cid] = func
        return cid

    def mpl_disconnect(self, cid):
        for d in self._cbs.values():
            d.pop(cid, None)

    def _fire(self, name, event):
        for cid, func in list(self._cbs.get(name, {}).items()):
            try:
                func(event)
            except Exception:
                traceback.print_exc()

    def draw_idle(self):
        pass

    def draw(self):
        self.viewport().update()

    def flush_events(self):
        QtWidgets.QApplication.processEvents()

    def get_width_height(self):
        return self.width(), self.height()

    # --- Qt events -> matplotlib events ---

    def _event(self, name, ev, button=None, dblclick=False, step=0):
        pos = _evpos(ev)
        px, py = float(pos.x()), float(pos.y())
        H = self.viewport().height()
        e = MouseEvent(name, self, px, H - py, button=button, dblclick=dblclick, step=step, guiEvent=ev)
        ax = self.figure._axes_at(px, py)
        if ax is not None:
            e.inaxes = ax
            try:
                (xa, xb), (ya, yb) = ax.vb.viewRange()
                x0, y0, w, h = ax._px_rect()
                fx = (px - x0) / (w or 1.0)
                fy = (y0 + h - py) / (h or 1.0)
                if ax.vb.state['xInverted']:
                    fx = 1.0 - fx
                if ax.vb.state['yInverted']:
                    fy = 1.0 - fy
                e.xdata = xa + fx * (xb - xa)
                e.ydata = ya + fy * (yb - ya)
            except Exception:
                e.inaxes = None
        return e

    def mousePressEvent(self, ev):
        self._flush_motion()
        b = _QT_BUTTONS.get(ev.button())
        e = self._event('button_press_event', ev, button=b)
        self.figure._pick(e)
        self._fire('button_press_event', e)
        super().mousePressEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        self._flush_motion()
        b = _QT_BUTTONS.get(ev.button())
        e = self._event('button_press_event', ev, button=b, dblclick=True)
        self.figure._pick(e)
        self._fire('button_press_event', e)
        super().mouseDoubleClickEvent(ev)

    def mouseReleaseEvent(self, ev):
        self._flush_motion()
        b = _QT_BUTTONS.get(ev.button())
        self._fire('button_release_event', self._event('button_release_event', ev, button=b))
        super().mouseReleaseEvent(ev)

    def mouseMoveEvent(self, ev):
        pos = _evpos(ev)
        self._pending_motion = (QtCore.QPointF(pos), ev.buttons())
        if not self._motion_scheduled:
            self._motion_scheduled = True
            QtCore.QTimer.singleShot(0, self._flush_motion)
        if self.manager is not None:
            self.manager._show_coords(float(pos.x()), float(pos.y()))
        super().mouseMoveEvent(ev)

    def _flush_motion(self):
        self._motion_scheduled = False
        pm = self._pending_motion
        self._pending_motion = None
        if pm is None:
            return
        pos, buttons = pm

        class _E:
            def __init__(self, p):
                self._p = p

            def position(self):
                return self._p
        b = None
        for qb, nb in _QT_BUTTONS.items():
            if buttons & qb:
                b = nb
                break
        self._fire('motion_notify_event', self._event('motion_notify_event', _E(pos), button=b))

    def event(self, ev):
        # Tab / Backtab reach the key callbacks first (Qt would move the focus with them); a key the
        # callbacks do not handle moves the focus as usual.
        if (ev.type() == QtCore.QEvent.Type.KeyPress and self._cbs.get('key_press_event')
                and ev.key() in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab)):
            self._key_handled = False
            self.keyPressEvent(ev)
            if self._key_handled:
                return True
        return super().event(ev)

    def keyPressEvent(self, ev):
        """matplotlib's key_press_event, with the axes under the cursor; a callback setting handled consumes the key."""
        self._key_handled = False
        key = _key_name(ev)
        if key is None or not self._cbs.get('key_press_event'):
            super().keyPressEvent(ev)
            return
        pos = self.viewport().mapFromGlobal(QtGui.QCursor.pos())
        m = self._event('motion_notify_event', _PosOnly(QtCore.QPointF(pos)))
        e = KeyEvent('key_press_event', self, key, m.x, m.y, guiEvent=ev)
        e.inaxes, e.xdata, e.ydata = m.inaxes, m.xdata, m.ydata
        self._fire('key_press_event', e)
        if e.handled:
            self._key_handled = True
            ev.accept()
        else:
            super().keyPressEvent(ev)

    def wheelEvent(self, ev):
        dy = ev.angleDelta().y()
        if dy:
            step = dy / 120.0
            e = self._event('scroll_event', ev, button='up' if dy > 0 else 'down', step=step)
            self._fire('scroll_event', e)
        super().wheelEvent(ev)

    def leaveEvent(self, ev):
        if self.manager is not None:
            self.manager._show_coords(None, None)
        super().leaveEvent(ev)


class _Window(QtWidgets.QMainWindow):
    def __init__(self, manager):
        super().__init__()
        self._manager = manager

    def closeEvent(self, ev):
        self._manager._closed()
        super().closeEvent(ev)


def _icon(kind, size=22):
    """Small toolbar icons drawn here (pan: four arrows, zoom: a magnifier), as matplotlib's toolbar shows them."""
    pm = QtGui.QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    pen = QtGui.QPen(QtGui.QColor(30, 30, 30))
    pen.setWidthF(2.2)
    p.setPen(pen)
    c = size / 2.0
    if kind == 'pan':
        r = size * 0.4
        p.drawLine(QtCore.QPointF(c - r, c), QtCore.QPointF(c + r, c))
        p.drawLine(QtCore.QPointF(c, c - r), QtCore.QPointF(c, c + r))
        h = size * 0.14
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            tip = QtCore.QPointF(c + dx * r, c + dy * r)
            p.drawLine(tip, QtCore.QPointF(tip.x() - dx * h - dy * h, tip.y() - dy * h - dx * h))
            p.drawLine(tip, QtCore.QPointF(tip.x() - dx * h + dy * h, tip.y() - dy * h + dx * h))
    elif kind == 'zoom':
        r = size * 0.26
        p.drawEllipse(QtCore.QPointF(c - 2, c - 2), r, r)
        pen.setWidthF(3.2)
        p.setPen(pen)
        p.drawLine(QtCore.QPointF(c - 2 + r * 0.7, c - 2 + r * 0.7), QtCore.QPointF(size - 3, size - 3))
    p.end()
    return QtGui.QIcon(pm)


class NavigationToolbar(QtWidgets.QToolBar):
    """
    A figure's navigation toolbar (matplotlib's): Home (the views as the
    window set them), Pan (left drag pans, right drag / wheel zooms), Zoom
    (a rectangle), Save, and the cursor's coordinates.  Works for a figure
    in its own window and for a figure embedded in a layout.
    """

    def __init__(self, figure, parent=None, coordinates=True):
        super().__init__('navigation', parent)
        self.figure = figure
        self.setMovable(False)
        k = int(round(22 * screen_scale()))
        self.setIconSize(QtCore.QSize(k, k))
        st = self.style()
        sp = QtWidgets.QStyle.StandardPixmap
        a = self.addAction(st.standardIcon(sp.SP_DirHomeIcon), 'Home')
        a.setToolTip('Reset the views to where the window put them')
        a.triggered.connect(self._home)
        self.addSeparator()
        self.a_pan = self.addAction(_icon('pan'), 'Pan')
        self.a_pan.setCheckable(True)
        self.a_pan.setToolTip('Pan with the left button, zoom with the right button or the wheel')
        self.a_pan.toggled.connect(lambda on: self._mode('pan/zoom' if on else ''))
        self.a_zoom = self.addAction(_icon('zoom'), 'Zoom')
        self.a_zoom.setCheckable(True)
        self.a_zoom.setToolTip('Zoom to a rectangle drawn with the left button')
        self.a_zoom.toggled.connect(lambda on: self._mode('zoom rect' if on else ''))
        self.addSeparator()
        a = self.addAction(st.standardIcon(sp.SP_DialogSaveButton), 'Save')
        a.setToolTip('Save the figure: PDF / EPS / SVG / PNG / TIFF for publication (matplotlib), PNG / SVG as on screen, or the curves as CSV')
        a.triggered.connect(self._save)
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        self.addWidget(spacer)
        self.coords = QtWidgets.QLabel('')
        self.coords.setMinimumWidth(200)
        self.coords.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if coordinates:
            self.addWidget(self.coords)
        figure._toolbars.append(self)

    def _home(self):
        for ax in self.figure._axes:
            ax._home_view()

    def _mode(self, mode):
        for a, m in ((self.a_pan, 'pan/zoom'), (self.a_zoom, 'zoom rect')):
            if m != mode and a.isChecked():
                a.blockSignals(True)
                a.setChecked(False)
                a.blockSignals(False)
        self.figure.canvas.toolbar.mode = mode
        for ax in self.figure._axes:
            ax._apply_toolbar_mode(mode)

    # (dialog filter, extension, how): 'publication' through matplotlib, 'screen' through pyqtgraph.
    SAVE_FORMATS = [
        ("PDF, publication (vector, editable) (*.pdf)", '.pdf', 'publication'),
        ("EPS, publication (vector, editable) (*.eps)", '.eps', 'publication'),
        ("SVG, publication (vector, editable) (*.svg)", '.svg', 'publication'),
        ("PNG, publication, 300 dpi (*.png)", '.png', 'publication'),
        ("TIFF, publication, 300 dpi (*.tif)", '.tif', 'publication'),
        ("PNG, as on screen (pyqtgraph) (*.png)", '.png', 'screen'),
        ("SVG, as on screen (pyqtgraph) (*.svg)", '.svg', 'screen'),
        ("PNG, screenshot of the window (*.png)", '.png', 'screenshot'),
        ("CSV, the curves' data (*.csv)", '.csv', 'csv'),
    ]

    def _save(self):
        title = self.figure.manager.get_window_title() if self.figure.manager is not None else ''
        name = ''.join(ch if ch.isalnum() or ch in '._- ' else '_' for ch in (title or 'figure')).strip() or 'figure'
        formats = list(self.SAVE_FORMATS)
        if not _have_matplotlib():
            formats = [f for f in formats if f[2] != 'publication']
        last = getattr(NavigationToolbar, '_last_filter', None)
        filters = [f[0] for f in formats]
        path, flt = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save figure', name, ';;'.join(filters), last if last in filters else filters[0])
        if not path:
            return
        NavigationToolbar._last_filter = flt
        _, ext, how = next((f for f in formats if f[0] == flt), formats[0])
        # The chosen type decides: a missing or different extension is replaced by the type's.
        root, given = os.path.splitext(path)
        if given.lower() not in (ext, '.tiff' if ext == '.tif' else ext):
            path = (root if given.lower() in {f[1] for f in formats} else path) + ext
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            out = self.figure.savefig(path, how=how, dpi=300 if how == 'publication' else None)
            print(f"saved: {out}")
        except Exception as exc:
            traceback.print_exc()
            print(f"save failed: {exc}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def show_coords(self, px, py):
        fig = self.figure
        if px is None:
            self.coords.setText('')
            return
        ax = fig._axes_at(px, py)
        if ax is None or ax._is_colorbar:
            self.coords.setText('')
            return
        (xa, xb), (ya, yb) = ax.vb.viewRange()
        x0, y0, w, h = ax._px_rect()
        fx, fy = (px - x0) / (w or 1), (y0 + h - py) / (h or 1)
        if ax.vb.state['yInverted']:
            fy = 1 - fy
        if ax.vb.state['xInverted']:
            fx = 1 - fx
        self.coords.setText(f"x={xa + fx * (xb - xa):.5g}  y={ya + fy * (yb - ya):.5g}")


class FigureManager:
    """The figure's window: the navigation toolbar on top, then the canvas (matplotlib's Qt window)."""

    def __init__(self, figure, canvas, w_px, h_px):
        self.figure = figure
        self.canvas = canvas
        self.window = _Window(self)
        self.window.setCentralWidget(canvas)
        self.toolbar = NavigationToolbar(figure, self.window)
        self.window.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)
        th = self.toolbar.sizeHint().height()
        scr = QtGui.QGuiApplication.primaryScreen()
        if scr is not None:
            av = scr.availableGeometry()
            w_px = min(int(w_px), int(av.width() * 0.98))
            h_px = min(int(h_px), int(av.height() * 0.95) - th)
        # The info panel: the details cut from the panel titles, docked at the bottom, out of the
        # way (closable; hidden while empty).
        self.info = QtWidgets.QPlainTextEdit()
        self.info.setReadOnly(True)
        self.info.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.info.setFont(qfont(7.5))
        self.info.setFixedHeight(int(pt2px(7.5) * FONT_SCALE * 4.2))
        self.dock = QtWidgets.QDockWidget('info', self.window)
        self.dock.setWidget(self.info)
        self.dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
                              | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.window.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.dock)
        self.dock.hide()
        self._dock_closed = False
        self.dock.visibilityChanged.connect(self._dock_vis)
        self.window.resize(int(w_px), int(h_px) + th)
        self._title = f"Figure {figure.number}"
        self.window.setWindowTitle(self._title)

    def set_window_title(self, title):
        self._title = str(title)
        self.window.setWindowTitle(self._title)

    def get_window_title(self):
        return self._title

    def _dock_vis(self, visible):
        try:
            if not visible and self.info.toPlainText() and self.window.isVisible() and not self._setting:
                self._dock_closed = True          # closed by hand: stays closed
        except RuntimeError:                      # the window is being destroyed
            pass

    _setting = False

    def set_info(self, text):
        if text == self.info.toPlainText():
            return
        self.info.setPlainText(text)
        want = bool(text) and not self._dock_closed
        if want != self.dock.isVisible():
            # The window grows / shrinks by the panel so the figure keeps its size.
            self._setting = True
            h = self.dock.sizeHint().height()
            self.dock.setVisible(want)
            if self.figure._shown:
                self.window.resize(self.window.width(), self.window.height() + (h if want else -h))
            self._setting = False

    def show(self):
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()
        self.figure._shown = True

    def _closed(self):
        self.figure._on_close()

    def _show_coords(self, px, py):
        for tb in self.figure._toolbars:
            tb.show_coords(px, py)


class _EmbeddedManager:
    """The manager of a figure embedded in another window's layout: no window of its own."""

    def __init__(self, figure, canvas):
        self.figure = figure
        self.canvas = canvas
        self.window = None
        self._title = ''

    def set_window_title(self, title):
        self._title = str(title)

    def get_window_title(self):
        return self._title

    def show(self):
        self.figure._shown = True

    def _show_coords(self, px, py):
        for tb in self.figure._toolbars:
            tb.show_coords(px, py)


# ---------------------------------------------------------------------- #
#  Figure                                                                  #
# ---------------------------------------------------------------------- #

_figures = {}


class Figure:
    """A figure window: axes at figure fractions, figure texts, Qt controls, a navigation toolbar."""

    def __init__(self, figsize=(6.4, 4.8), num=None, embed=False):
        """embed=True: the figure lives in another window's layout (fig.canvas is the widget; NavigationToolbar(fig) its toolbar)."""
        _app()
        self._toolbars = []
        self._embedded = bool(embed)
        self.number = int(num) if num is not None else (max(_figures) + 1 if _figures else 1)
        w_px, h_px = float(figsize[0]) * DPI, float(figsize[1]) * DPI
        # The design size (matplotlib's figsize at DPI): the text is at FONT_SCALE x its points when
        # the canvas has this size and grows / shrinks with the canvas (font_zoom), so it stays
        # readable on any monitor and in a maximized window.  On a high-DPI screen the window opens
        # proportionally larger.
        self._design_px = (w_px, h_px)
        self.font_zoom = 1.0
        if not embed:
            k = screen_scale()
            w_px, h_px = w_px * k, h_px * k
        self.canvas = FigureCanvas(self, w_px, h_px)
        if not self._embedded:
            # A window of its own: white panels on the window colour (the browser's panel look).
            self.canvas.setBackground(QtGui.QColor(WINDOW_BG))
        self.canvas.scene().setItemIndexMethod(QtWidgets.QGraphicsScene.ItemIndexMethod.NoIndex)
        self._axes = []
        self._texts = []
        self._qt_controls = []         # objects with _place(W, H): the widgets of widgets.py
        self._suptitle = None
        self._shown = False
        self._closed = False
        self.subplotpars = SubplotParams()
        self.transFigure = _Transform('figure', self)
        self.dpi = DPI
        if self._embedded:
            self.manager = _EmbeddedManager(self, self.canvas)
            self._shown = True
        else:
            self.manager = FigureManager(self, self.canvas, w_px, h_px)
            _figures[self.number] = self
        self.canvas.manager = self.manager

    # --- the info panel: the details cut from the titles ---

    def _set_info(self, key, text):
        info = self.__dict__.setdefault('_info', {})
        text = str(text or '').strip()
        if info.get(key, '') == text:
            return
        if text:
            info[key] = text
        else:
            info.pop(key, None)
        if not getattr(self, '_info_pending', False):
            self._info_pending = True
            QtCore.QTimer.singleShot(50, self._push_info)

    def info_text(self):
        """The details of every panel's title, one line each (panels in their order on the figure)."""
        info = getattr(self, '_info', {})
        order = (['suptitle'] if 'suptitle' in info else []) + [ax for ax in self._axes if ax in info]
        return "\n".join(" ".join(info[k].split()) for k in order)

    def _push_info(self):
        self._info_pending = False
        try:
            text = self.info_text()
        except RuntimeError:
            return
        cb = getattr(self, 'on_info', None)
        if cb is not None:
            try:
                cb(text)
            except Exception:
                traceback.print_exc()
        if self.manager is not None and hasattr(self.manager, 'set_info') and not self._closed:
            try:
                self.manager.set_info(text)
            except RuntimeError:               # the window is already gone (closing, or Python exiting)
                pass

    # --- size ---

    def _canvas_size(self):
        if self._shown and (not self._embedded or self.canvas.isVisible()):
            vp = self.canvas.viewport()
            return float(max(vp.width(), 1)), float(max(vp.height(), 1))
        return float(self.canvas._size_px[0]), float(self.canvas._size_px[1])

    def get_size_inches(self):
        W, H = self._canvas_size()
        return np.array([W / DPI, H / DPI])

    def set_size_inches(self, w, h=None, forward=True):
        if h is None:
            w, h = w
        tw, th = int(round(float(w) * DPI)), int(round(float(h) * DPI))
        cw, ch = self._canvas_size()
        self.canvas._size_px = (tw, th)
        win = self.manager.window
        if win is not None:
            win.resize(win.width() + tw - int(cw), win.height() + th - int(ch))
        self._relayout()

    @property
    def axes(self):
        return list(self._axes)

    def get_axes(self):
        return list(self._axes)

    # --- layout ---

    def _relayout(self):
        if self._shown and (not self._embedded or self.canvas.isVisible()):
            vp = self.canvas.viewport()
            self.canvas._size_px = (vp.width(), vp.height())
        W0, H0 = self._design_px
        W, H = self._canvas_size()
        z = min(max(min(W / W0, H / H0), 1.0), 4.0)     # never below the designed size
        if abs(z - self.font_zoom) > 0.03 * self.font_zoom:
            self.font_zoom = z
            for tb in self._toolbars:
                try:
                    k = int(round(22 * screen_scale() * max(z, 1.0)))
                    tb.setIconSize(QtCore.QSize(k, k))
                    tb.coords.setFont(qfont(9, zoom=z))
                except RuntimeError:
                    pass
            for ax in list(self._axes):
                ax._refont(z)
            for t in self._texts:
                t._apply()
            if hasattr(self.manager, 'info'):
                try:
                    self.manager.info.setFont(qfont(7.5, zoom=z))
                    self.manager.info.setFixedHeight(int(pt2px(7.5) * FONT_SCALE * z * 4.2))
                except RuntimeError:
                    pass
        for ax in self._axes:
            ax._place()
        for t in self._texts:
            t._place()
        W, H = self._canvas_size()
        for c in self._qt_controls:
            try:
                c._place(W, H)
            except Exception:
                traceback.print_exc()

    def _axes_at(self, px, py):
        for ax in reversed(self._axes):
            if not ax._visible:
                continue
            x0, y0, w, h = ax._px_rect()
            if x0 <= px <= x0 + w and y0 <= py <= y0 + h:
                return ax
        return None

    def _pick(self, e):
        ax = e.inaxes
        if ax is None:
            return
        hits = [a for a in ax._children if a.pickable() and a.contains_px(e.x, e.y)]
        for a in hits:
            self.canvas._fire('pick_event', PickEvent(self.canvas, e, a))

    def _register_control(self, c):
        self._qt_controls.append(c)
        W, H = self._canvas_size()
        c._place(W, H)

    def _unregister_control(self, c):
        try:
            self._qt_controls.remove(c)
        except ValueError:
            pass

    # --- axes ---

    def add_axes(self, rect, sharex=None, sharey=None, **kw):
        if isinstance(rect, Axes):
            return rect
        b = rect.bounds if hasattr(rect, 'bounds') else rect
        ax = Axes(self, rect=b, sharex=sharex, sharey=sharey)
        if kw.get('facecolor') is not None:
            ax.patch.set_facecolor(kw['facecolor'])
        return ax

    def add_subplot(self, *args, sharex=None, sharey=None, **kw):
        if len(args) == 1 and isinstance(args[0], SubplotSpec):
            spec = args[0]
        else:
            if len(args) == 1:
                n = int(args[0])
                args = (n // 100, (n // 10) % 10, n % 10)
            if not args:
                args = (1, 1, 1)
            nrows, ncols, index = args
            gs = GridSpec(nrows, ncols)
            if isinstance(index, tuple):
                a, b = index
                spec = gs[(a - 1):b]
            else:
                spec = gs[int(index) - 1]
        return Axes(self, spec=spec, sharex=sharex, sharey=sharey)

    def subplots(self, nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, width_ratios=None,
                 height_ratios=None, gridspec_kw=None, **kw):
        gk = dict(gridspec_kw or {})
        gs = GridSpec(nrows, ncols, width_ratios=width_ratios or gk.get('width_ratios'),
                      height_ratios=height_ratios or gk.get('height_ratios'),
                      wspace=gk.get('wspace'), hspace=gk.get('hspace'))
        axs = np.empty((nrows, ncols), dtype=object)
        for r in range(nrows):
            for c in range(ncols):
                shx = axs[0, 0] if (sharex in (True, 'all') and (r or c)) else None
                shy = axs[0, 0] if (sharey in (True, 'all') and (r or c)) else None
                axs[r, c] = Axes(self, spec=gs[r, c], sharex=shx, sharey=shy)
        if squeeze:
            if nrows == 1 and ncols == 1:
                return axs[0, 0]
            if nrows == 1 or ncols == 1:
                return axs.ravel()
        return axs

    def add_gridspec(self, nrows=1, ncols=1, **kw):
        return GridSpec(nrows, ncols, figure=self, **kw)

    def subplots_adjust(self, left=None, bottom=None, right=None, top=None, wspace=None, hspace=None):
        sp = self.subplotpars
        for k, v in (('left', left), ('bottom', bottom), ('right', right), ('top', top), ('wspace', wspace),
                     ('hspace', hspace)):
            if v is not None:
                setattr(sp, k, float(v))
        for ax in self._axes:
            if ax._spec is not None:
                ax._rect = None
                ax._place()

    def delaxes(self, ax):
        ax.remove()

    def sca(self, ax):
        return ax

    def gca(self):
        return self._axes[-1] if self._axes else self.add_subplot(111)

    # --- texts ---

    def text(self, x, y, s, **kw):
        kw.pop('transform', None)
        t = Text(self, x, y, s, 'figure', **kw)
        self._texts.append(t)
        return t

    def _forget_text(self, t):
        try:
            self._texts.remove(t)
        except ValueError:
            pass
        if self._suptitle is t:
            self._suptitle = None

    def suptitle(self, t, fontsize=None, y=0.98, x=0.5, ha='center', va='top', **kw):
        if self._suptitle is not None:
            self._suptitle.remove()
        full = '' if t is None else str(t)
        t = short_title(full)
        self._set_info('suptitle', full if full.strip() != t else '')
        self._suptitle = self.text(x, y, t, fontsize=font_pt(fontsize, 12.0), ha=ha, va=va,
                                   fontweight=kw.get('fontweight'))
        return self._suptitle

    # --- colour bars ---

    def colorbar(self, mappable, cax=None, ax=None, fraction=0.15, pad=0.05, label=None, shrink=1.0, aspect=20,
                 **kw):
        mappable = getattr(mappable, 'mappable', mappable) if not isinstance(mappable, (AxesImage, QuadMesh)) else mappable
        parent_rect = None
        if cax is None:
            if ax is None:
                ax = self._axes[0] if self._axes else self.add_subplot(111)
            x0, y0, w, h = ax.get_position().bounds
            parent_rect = (x0, y0, w, h)
            W, H = self._canvas_size()
            cb_w = w * fraction
            pad_w = w * pad
            ax.set_position([x0, y0, w - cb_w - pad_w, h])
            hh = h * shrink
            cw = min(cb_w, (hh * H / float(aspect)) / W)
            cax = Axes(self, rect=[x0 + w - cb_w, y0 + (h - hh) / 2.0, cw, hh])
        cb = Colorbar(self, mappable, cax, label=label, parent_axes=ax, parent_rect=parent_rect)
        return cb

    # --- output, closing ---

    def savefig(self, path, dpi=None, how=None, **kw):
        """
        Save the figure (the plots and texts; the controls are left out of every format but the
        screenshot).  how:
          'publication' (default for .pdf .eps .ps .svg .tif .tiff, and for .png/.jpg with dpi
                         given): rebuilt and written by matplotlib (mplexport), vector files with
                         editable text, raster files at dpi (300 by default);
          'screen'      : as pyqtgraph draws it (pyqtgraph's ImageExporter / SVGExporter);
          'screenshot'  : the window's pixels, controls included (default for .png/.jpg/.bmp);
          'csv'         : the curves' data, one column pair per curve.
        Without matplotlib, the vector formats are written by Qt (QPdfWriter / QSvgGenerator).
        """
        path = str(path)
        ext = os.path.splitext(path)[1].lower()
        if how is None:
            if ext == '.csv':
                how = 'csv'
            elif ext in ('.pdf', '.eps', '.ps', '.svg', '.tif', '.tiff') or dpi is not None:
                how = 'publication'
            else:
                how = 'screenshot'
        if how == 'csv':
            return export_csv(self, path)
        if how == 'publication':
            from sm4_Browser import mplexport
            if mplexport.available():
                return mplexport.export(self, path, dpi=dpi or 300)
            print("matplotlib is not installed: the file is written by Qt instead (pip install matplotlib for "
                  "publication output).")
            if ext in ('.pdf', '.eps', '.ps', '.svg'):
                return export_vector(self, path)
            how = 'screen'
        if how == 'screen':
            import pyqtgraph.exporters as pex
            QuadItem.EXPORTING = ext == '.svg'
            try:
                ex = (pex.SVGExporter if ext == '.svg' else pex.ImageExporter)(self.canvas.scene())
                if ext != '.svg':
                    W, H = self._canvas_size()
                    ex.parameters()['width'] = int(W * max(1, (dpi or 2 * DPI) / DPI))
                ex.export(path)
            finally:
                QuadItem.EXPORTING = False
            return path
        self.canvas.grab().save(path)
        return path

    def show(self):
        self.manager.show()

    def clf(self):
        for ax in list(self._axes):
            ax.remove()
        for t in list(self._texts):
            t.remove()

    clear = clf

    def _on_close(self):
        if self._closed:
            return
        self._closed = True
        self.canvas._fire('close_event', CloseEvent(self.canvas))
        if _figures.get(self.number) is self:
            del _figures[self.number]

    def tight_layout(self, *a, **k):
        pass

    def legend(self, *a, **k):
        return None


def export_csv(fig, path):
    """Every curve of every visible panel as two columns (x, y), headed by panel title and label."""
    cols, heads = [], []
    for k, ax in enumerate(fig._axes):
        if not ax._visible:
            continue
        title = ax._title.get_text() if ax._title is not None else f"panel {k + 1}"
        for a in ax._children:
            if isinstance(a, Line2D) and a.get_visible() and len(a._x):
                lab = a.get_label()
                lab = '' if lab.startswith('_') else lab
                heads += [f"{title} | {lab} | {ax.get_xlabel() or 'x'}", f"{title} | {lab} | {ax.get_ylabel() or 'y'}"]
                cols += [a._x, a._y]
    n = max((len(c) for c in cols), default=0)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(','.join('"' + h.replace('"', "'") + '"' for h in heads) + '\n')
        for i in range(n):
            fh.write(','.join(('' if i >= len(c) or not np.isfinite(c[i]) else repr(float(c[i]))) for c in cols) + '\n')
    return path


def _render_scene(fig, painter, W, H):
    """Draw the figure's scene (no Qt controls) into a painter over (0, 0, W, H), caches off."""
    scene = fig.canvas.scene()
    cached = []
    for it in scene.items():
        if it.cacheMode() != QtWidgets.QGraphicsItem.CacheMode.NoCache:
            cached.append((it, it.cacheMode()))
            it.setCacheMode(QtWidgets.QGraphicsItem.CacheMode.NoCache)
    QuadItem.EXPORTING = True
    try:
        if not fig._embedded:
            painter.fillRect(QtCore.QRectF(0, 0, W, H), QtGui.QColor(WINDOW_BG))
        scene.render(painter, QtCore.QRectF(0, 0, W, H), QtCore.QRectF(0, 0, W, H))
    finally:
        QuadItem.EXPORTING = False
        for it, mode in cached:
            it.setCacheMode(mode)


def _have_matplotlib():
    try:
        import importlib.util
        return importlib.util.find_spec('matplotlib') is not None
    except Exception:
        return False


def _ghostscript():
    import shutil
    for name in ('gswin64c', 'gswin32c', 'gs'):
        p = shutil.which(name)
        if p:
            return p
    return None


def export_vector(fig, path):
    """
    The figure as a vector file: PDF (QPdfWriter, page = the figure's size), SVG
    (QSvgGenerator), or EPS (the PDF converted by Ghostscript, which must be installed;
    without it a PDF is written beside the requested name and the reason printed: Illustrator
    opens the PDF and SVG directly with every object editable).
    """
    W, H = fig._canvas_size()
    ext = os.path.splitext(path)[1].lower()
    if ext == '.svg':
        from pyqtgraph.Qt import QtSvg
        gen = QtSvg.QSvgGenerator()
        gen.setFileName(path)
        gen.setSize(QtCore.QSize(int(W), int(H)))
        gen.setViewBox(QtCore.QRectF(0, 0, W, H))
        gen.setResolution(int(DPI))
        gen.setTitle(fig.manager.get_window_title() if fig.manager is not None else 'figure')
        p = QtGui.QPainter(gen)
        _render_scene(fig, p, W, H)
        p.end()
        return path
    pdf = path if ext == '.pdf' else os.path.splitext(path)[0] + '.pdf'
    w = QtGui.QPdfWriter(pdf)
    w.setResolution(int(DPI))
    w.setPageSize(QtGui.QPageSize(QtCore.QSizeF(W / DPI * 72.0, H / DPI * 72.0), QtGui.QPageSize.Unit.Point))
    w.setPageMargins(QtCore.QMarginsF(0, 0, 0, 0))
    p = QtGui.QPainter(w)
    sx = w.width() / W
    p.scale(sx, sx)
    _render_scene(fig, p, W, H)
    p.end()
    if ext == '.pdf':
        return path
    gs = _ghostscript()
    if gs is None:
        print(f"EPS needs Ghostscript (gswin64c / gs on the PATH), which was not found; the figure was "
              f"written as {pdf} instead (Illustrator opens PDF and SVG with every object editable).")
        return pdf
    import subprocess
    subprocess.run([gs, '-q', '-dNOPAUSE', '-dBATCH', '-dNoOutputFonts' if False else '-dSAFER', '-sDEVICE=eps2write',
                    f'-sOutputFile={path}', pdf], check=True)
    try:
        os.remove(pdf)
    except OSError:
        pass
    return path


# ---------------------------------------------------------------------- #
#  pyplot functions                                                        #
# ---------------------------------------------------------------------- #

_RUNNING = [False]


def _app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        # OpenCV's Linux wheels point Qt at their own (Qt5) platform plugins when cv2 is imported
        # (drift.py uses cv2); PySide6 must use its own.
        p = os.environ.get('QT_QPA_PLATFORM_PLUGIN_PATH', '')
        if 'cv2' in p:
            os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
        app = pg.mkQApp('sm4_Browser')
        f = app.font()
        if f.pointSizeF() > 0:
            f.setPointSizeF(f.pointSizeF() * FONT_SCALE)
            app.setFont(f)
    return app


def figure(num=None, figsize=None, dpi=None, **kw):
    if isinstance(num, Figure):
        return num
    if num is not None and num in _figures:
        return _figures[num]
    return Figure(figsize=figsize or (6.4, 4.8), num=num)


def subplots(nrows=1, ncols=1, figsize=None, sharex=False, sharey=False, squeeze=True, **kw):
    fig = figure(figsize=figsize)
    axs = fig.subplots(nrows, ncols, sharex=sharex, sharey=sharey, squeeze=squeeze,
                       gridspec_kw=kw.get('gridspec_kw'))
    return fig, axs


def gcf():
    if _figures:
        return _figures[max(_figures)]
    return figure()


def fignum_exists(num):
    if isinstance(num, Figure):
        if num._embedded:
            return not num._closed
        return _figures.get(num.number) is num
    return num in _figures


def get_fignums():
    return sorted(_figures)


def close(fig=None):
    if fig is None:
        targets = [gcf()] if _figures else []
    elif fig == 'all':
        targets = list(_figures.values())
    elif isinstance(fig, Figure):
        targets = [fig]
    else:
        targets = [_figures[fig]] if fig in _figures else []
    for f in targets:
        try:
            f.manager.window.close()
        except Exception:
            pass
        f._on_close()


def _ipython_gui():
    """True when an IPython kernel (Spyder, Jupyter qtconsole) runs a Qt event loop or one could be turned on."""
    try:
        ip = get_ipython()                       # noqa: F821 (defined inside IPython)
    except NameError:
        return False
    if ip is None:
        return False
    loop = getattr(ip, 'active_eventloop', None)
    if loop and 'qt' in str(loop):
        return True
    try:
        ip.enable_gui('qt')
        return True
    except Exception:
        return False


def show(block=None):
    """
    Show every figure not yet shown.  Outside IPython the first call runs
    the Qt event loop until every window is closed (matplotlib's blocking
    show); inside a running loop, or in an IPython console with its Qt
    event loop (Spyder), it returns at once.
    """
    app = _app()
    for f in list(_figures.values()):
        if not f._shown:
            f.manager.show()
            f._relayout()
    if block is False:
        app.processEvents()
        return
    if _RUNNING[0] or QtCore.QThread.currentThread().loopLevel() > 0:
        return
    if block is None and _ipython_gui():
        return
    if not _figures and not any(w.isVisible() for w in _figures_main_windows):
        return
    _RUNNING[0] = True
    try:
        (app.exec() if hasattr(app, 'exec') else app.exec_())
    finally:
        _RUNNING[0] = False


def pause(interval):
    app = _app()
    show(block=False)
    deadline = QtCore.QDeadlineTimer(int(interval * 1000))
    while not deadline.hasExpired():
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)


def draw():
    pass


def ion():
    pass


def ioff():
    pass


def isinteractive():
    return True


class _Cycler:
    def __init__(self, colors):
        self._colors = list(colors)

    def by_key(self):
        return {'color': list(self._colors)}


rcParams = {'axes.prop_cycle': _Cycler(TAB10), 'figure.dpi': DPI, 'font.size': 10.0}

_figures_main_windows = []      # windows embedding qtplot figures (browser_qt), kept referenced
