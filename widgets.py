"""
The control tables and the canvas interactions every window shares.

  - SPECS: one specification per control name (its kind, its labels, its font size, a text box's
    width), LABELS (a name's label text where it differs from the name), SHORT_LABELS and
    short_form (the short labels a window taller than wide shows), so that a control of a given
    name is the same in every window.  The controls themselves are built from these tables in
    controls.py (controls.make and the composites), the one control set of every window.
  - The controls' fonts (_font, indicator_css, _refont_widgets) and filters_text (one console
    phrase of a filter box's state).
  - ViewNavigator (the scan area's zoom and pan), DragPath (the browser's section path),
    PairMarkers (the marker pairs of the drift window's two topos, each topo at its own scale, the
    partners seeded by the velocity the window gives).
  - figure_is_open.
The drawing of data lives in plots.py and never touches a control; a window owns its data and
state, draws through plots, and receives events from here and from controls.py.

automatic layout: the one rule for every window.  No control is placed by hand: every control is
one Qt widget in its window's Qt layout (controls.py, qtplot.Figure.set_layout), so nothing here
holds a position.
"""

if __package__ in (None, ''):
    # Run as a file (Spyder F5): make the package importable from wherever its folder sits.
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np
from sm4_Browser import qtplot as plt
from sm4_Browser.qtplot import QtWidgets, qfont
from sm4_Browser.topo import TOPO_DIRECTIONS


def figure_is_open(fig):
    """True while `fig` itself is still open (a closed figure's number may be reused by a later figure)."""
    try:
        return plt.fignum_exists(fig)
    except Exception:
        return False


# ---------------------------------------------------------------------- #
#  Specifications: name -> kind, size (inches), labels, font size          #
# ---------------------------------------------------------------------- #
# The sizes are the inches each control had when the windows placed their controls by hand; the
# layout now sizes every control to its text (automatic layout: the one rule for every window), and
# of the sizes only a text box's width (w) is still read (controls.Text).

def _spec(kind, w, h, fs=8, labels=None, **extra):
    d = {'kind': kind, 'w': float(w), 'h': float(h), 'fs': fs}
    if labels is not None:
        d['labels'] = list(labels)
    d.update(extra)
    return d


SPECS = {
    # --- toggles (one row) ---
    'Feenstra': _spec('check', 1.25, 0.38),
    'Feenstra (refit)': _spec('check', 1.35, 0.32, fs=8),          # the fit windows: the quantity of the next Refit
    'stack': _spec('check', 0.85, 0.38),
    'substack': _spec('check', 1.05, 0.38),
    'Average': _spec('check', 0.95, 0.38),
    'bin': _spec('check', 0.75, 0.33),
    'interp': _spec('check', 0.85, 0.33),
    '0Vsub': _spec('check', 0.95, 0.30),
    'broadened': _spec('check', 1.35, 0.30),
    'auto drift': _spec('check', 1.1, 0.38),                      # the tracker's drift (KAZE pairs)
    'manual drift': _spec('check', 1.25, 0.38),                   # the marker / typed drift (the MATLAB method)
    'stripe': _spec('check', 0.75, 0.38),                         # grids: the stripe matching on / off, one state per file (Drift.set_stripe_on)
    'stripe match': _spec('button', 1.2, 0.36),                   # grids: opens the stripe window (stripe.StripeWindow)
    'anchor to topo': _spec('check', 1.45, 0.38),                 # the MATLAB "drift for first point" (rigid shift): the browser's and the Topo window's toggle
    'cmap norm': _spec('check', 1.4, 0.38),
    'overlapping': _spec('check', 0.85, 0.38, fs=7),              # the line-average window: average only where every member covers a pixel
    'WKB background': _spec('check', 1.7, 0.32),
    'hold': _spec('check', 0.3, 0.26, fs=7, labels=['']),           # the fit edit window's hold rows (unlabeled)
    # --- toggle boxes (several rows) ---
    'pipeline': _spec('checks', 0.95, 1.68, fs=8, labels=['denoise', 'soft edge', 'calibrate', 'gate', '0-lock', 'Vbias lock', 'normalize']),   # denoise first, unindented, its soft edge row indented under it; the gate and lock rows indented under calibrate (PipelineBox.INDENT)
    'gate H': _spec('text', 0.40, 0.20, fs=7),                      # the gate's Hampel 'window,n_sigma' beside the gate row (PipelineBox)
    'gate SG': _spec('text', 0.40, 0.20, fs=7),                     # the gate's Savitzky-Golay 'length,order'
    'topo processors': _spec('checks', 1.4, 0.36, fs=8, labels=None, row_h=0.36),   # labels from topo.TOPO_PROCESSORS
    # --- radios ---
    'channel': _spec('radio', 1.1, 0.55, fs=8, labels=['dI/dV', 'IETS']),
    'sweeps': _spec('radio', 1.0, 0.95, fs=7, labels=['forward sweeps', 'reverse sweeps', 'all sweeps']),   # labels per file (sweeps.sweep_modes)
    'topo pass': _spec('radio', 1.05, 0.85, fs=8, labels=list(TOPO_DIRECTIONS)),      # topo.TOPO_DIRECTIONS
    'site': _spec('radio', 1.3, 0.85, fs=8, labels=['pulsing', 'clean', 'experiment']),
    'cal method': _spec('radio', 1.35, 0.55, fs=8, labels=['fitted stretch', '1/V_mod']),   # calibration_stretch_vmod's if-main
    'slow axis': _spec('radio', 1.0, 0.85, fs=8, labels=['off', 'auto', 'manual'], title=True),   # the stripe window: the velocity source, titled 'drift' (LABELS)
    'fast axis': _spec('radio', 1.0, 0.85, fs=8, labels=['off', 'auto', 'manual'], title=True),   # ... the placement along the line, titled 'stripe'
    'stripe pass': _spec('radio', 1.0, 0.6, fs=8, labels=['forward', 'reverse'], title=True),    # ... the pass drawn on top, titled 'pass'
    # --- buttons ---
    'run': _spec('button', 1.2, 0.36),
    'clear': _spec('button', 1.2, 0.36),
    'restore': _spec('button', 1.2, 0.36),
    'save': _spec('button', 1.2, 0.36),
    'remove last': _spec('button', 1.2, 0.36),
    'Send to fit': _spec('button', 1.4, 0.36, fs=8),
    'add new': _spec('button', 0.55, 0.12, fs=6),
    'add existing': _spec('button', 0.75, 0.12, fs=6),
    'edit alignment': _spec('button', 1.3, 0.36),
    'line → new avg': _spec('button', 1.35, 0.24, fs=7),            # the Line window: this line starts a line average
    'line → existing avg': _spec('button', 1.35, 0.24, fs=7),       # ... or joins an open one (the pop-up list)
    'map → new avg': _spec('button', 1.35, 0.24, fs=7),             # the Map window: the whole map starts an average
    'map → existing avg': _spec('button', 1.35, 0.24, fs=7),        # ... or joins an open one (the pop-up list)
    'add to plot': _spec('button', 0.8, 0.24, fs=7),                # the Map window: the panel above goes to the figure composer
    'new plot': _spec('button', 0.8, 0.24, fs=7),                   # the Map window: the figure composer's list emptied
    'apply': _spec('button', 0.9, 0.36),
    'auto': _spec('button', 0.9, 0.36),
    'close': _spec('button', 0.9, 0.36),
    'Grid view': _spec('button', 0.9, 0.36),
    'Topo Window': _spec('button', 1.5, 0.45),
    'section': _spec('button', 1.1, 0.36),                        # the browser: the section path (editing on / off)
    'cross section': _spec('button', 1.1, 0.36),                  # ... the three section features: its profile window,
    'measure': _spec('button', 1.1, 0.36),                        # ... the measure window (the points and lengths),
    'set scale': _spec('button', 1.1, 0.36),                      # ... and the set-scale window (section.py)
    'cal length': _spec('text', 0.9, 0.3, fs=9),                  # the set-scale window: the calibrated length (nm)
    'clear markers': _spec('button', 1.2, 0.36),                  # the manual-drift markers removed
    'clear points': _spec('button', 1.1, 0.36),                   # the stripe window: the dragged points dropped
    'crop': _spec('button', 0.6, 0.26, fs=7),                     # the Line and line-average windows: the view and the normalization to the cursors' ranges
    'uncrop': _spec('button', 0.7, 0.26, fs=7),                   # ... back to the full ranges
    'set from markers': _spec('button', 1.4, 0.36, fs=7),         # the drift window's velocity box filled from the marker draft (nothing stored)
    'apply velocity': _spec('button', 1.2, 0.36),                 # the drift window's velocity box stored as the pair's span (confirmed, adjusted, or typed)
    'drift': _spec('button', 0.8, 0.36),                          # the browser: the current file's manual drift window
    'drift window': _spec('button', 1.2, 0.36),                   # the Line / Map windows: the file's before/after topos with marker pairs
    'profile': _spec('button', 0.4, 0.22, fs=6),                 # a Line-window topo row: its line's forward/reverse profile window
    'load saved averages': _spec('button', 1.7, 0.45),
    'load': _spec('button', 1.3, 0.36),
    'rescan': _spec('button', 1.3, 0.36),
    'Line': _spec('button', 1.2, 0.45),
    'Map': _spec('button', 1.2, 0.45),
    '< Prev': _spec('button', 1.0, 0.45),
    'Next >': _spec('button', 1.0, 0.45),
    '< E': _spec('button', 0.55, 0.36),
    'E >': _spec('button', 0.55, 0.36),
    '< line': _spec('button', 0.6, 0.3, fs=7),
    'line >': _spec('button', 0.6, 0.3, fs=7),
    '< sweep': _spec('button', 0.6, 0.3, fs=7),                  # calibration's if-main: the previous / next sweep of the file
    'sweep >': _spec('button', 0.6, 0.3, fs=7),
    '< before': _spec('button', 0.62, 0.3, fs=7),                # the drift window: the earlier topo one step back / forward in time
    'before >': _spec('button', 0.62, 0.3, fs=7),
    '< after': _spec('button', 0.62, 0.3, fs=7),                 # ... and the later topo
    'after >': _spec('button', 0.62, 0.3, fs=7),
    'prev region': _spec('button', 0.78, 0.32, fs=6),
    'next region': _spec('button', 0.78, 0.32, fs=6),
    '+': _spec('button', 0.4, 0.3),
    '-': _spec('button', 0.4, 0.3),
    'f': _spec('button', 0.32, 0.22, fs=8),                          # the topo rows' pass button (shows f or r)
    'Add peak': _spec('button', 1.3, 0.45),
    'Refit': _spec('button', 1.3, 0.45),
    'edit background': _spec('button', 1.4, 0.45),
    'Revert fit': _spec('button', 1.3, 0.4),
    'Forward fit': _spec('button', 1.3, 0.4),
    'Edit peaks': _spec('button', 1.3, 0.4),
    'Save fit': _spec('button', 1.3, 0.4),
    # --- sliders ---
    # 'Markers', 'avgs' and 'E-markers' were sliders; controls.make builds them as whole-number boxes
    # (controls.Count) with these limits.  'file' (the browser's file slider) is the one slider left.
    'Markers': _spec('slider', 1.1, 0.12, fs=8, vmin=0, vmax=10, valstep=1),   # the drift window passes vmax=MARKER_PAIRS_MAX
    'avgs': _spec('slider', 1.1, 0.22, fs=8, vmin=0, vmax=10, valstep=1),
    'E-markers': _spec('slider', 1.1, 0.14, fs=8, vmin=0, vmax=8, valstep=1),     # the Line window's energy markers
    'file': _spec('slider', 7.0, 0.2, fs=9, vmin=0, vmax=1, valstep=1, valfmt='%d'),
    # --- text boxes ---
    'γ': _spec('text', 0.45, 0.24, fs=7),
    'clip %': _spec('text', 0.45, 0.24, fs=7),                       # the Map window's percentile clip of the colour scale (plots.percentile_range)
    'E lo': _spec('text', 0.55, 0.22, fs=7),                         # the Line window's crop cursors: the boxes riding with them,
    'E hi': _spec('text', 0.55, 0.22, fs=7),                         # ... the energy ones (V, three decimals) along the heatmap's top edge,
    'x lo': _spec('text', 0.55, 0.22, fs=7),                         # ... the position ones (Å, one decimal) along its right edge
    'x hi': _spec('text', 0.55, 0.22, fs=7),
    'name': _spec('text', 2.1, 0.26, fs=9),
    'line': _spec('text', 0.55, 0.3, fs=8),
    'scale factor': _spec('text', 0.65, 0.3, fs=9),
    'reference': _spec('text', 0.5, 0.3, fs=8),
    'width': _spec('text', 0.65, 0.30, fs=8),                        # the Feenstra broadening width (V)
    'rows': _spec('text', 0.42, 0.22, fs=8),                         # a topo row's line-index box
    'drift v': _spec('text', 1.2, 0.3, fs=8),                        # the typed manual velocity "Vx, Vy" (nm/h, calibrated by the file's scale)
    'param': _spec('text', 0.9, 0.26, fs=7),                         # the fit edit window's parameter boxes
}

# Widget names whose label text differs from the name (every other widget is labeled with its name).
LABELS = {'pipeline': None, 'topo processors': None, 'channel': None, 'topo pass': None, 'site': None,
          'hold': '', 'param': '', 'rows': '', 'width': '', 'file': 'file',
          'name': 'name ', 'line': 'line ', 'scale factor': 'scale factor ', 'reference': 'reference ',
          'cal length': 'calibrated length (nm) ',
          'γ': 'γ ', 'clip %': 'clip % ', 'drift v': 'v (nm/h, cal) ', 'E lo': '', 'E hi': '', 'x lo': '', 'x hi': '',
          'gate H': 'H ', 'gate SG': 'SG ',
          'slow axis': 'drift', 'fast axis': 'stripe', 'stripe pass': 'pass'}        # the radios' titles (Radio, title=True)

# The short form of every label that has one, used when a window is taller than it is wide (the
# figure's portrait flag, qtplot.Figure.set_portrait; the browser's window, browser._WindowZoom):
# a control whose label is a key here shows the value instead, and the full label when the window
# is wider than tall again.  Keys are the label texts as shown (LABELS applied), stripped.  A label
# not listed keeps its text in both modes.  One table for every window's controls
# (controls.Control.set_short), the browser's included.
SHORT_LABELS = {
    # toggles and boxes
    'calibrate': 'cal', 'normalize': 'norm', 'auto drift': 'auto', 'manual drift': 'manual',
    'anchor to topo': 'anchor', 'broadened': 'broad', 'substack': 'sub', 'Average': 'Avg',
    'Feenstra': 'Feen', 'Feenstra (refit)': 'Feen refit', 'cmap norm': 'cmap', 'clip %': 'clip', 'WKB background': 'WKB',
    'overlapping': 'overlap',
    'step through': 'step', 'topo processing': 'topo proc', 'filters': 'filters',
    'polyorder': 'order', 'Savitzky-Golay': 'SG',
    # radio options
    'forward': 'fwd', 'reverse': 'rev', 'average': 'avg', 'experiment': 'exp', 'pulsing': 'pulse',
    'forward sweeps': 'fwd', 'reverse sweeps': 'rev', 'all sweeps': 'all', 'fitted stretch': 'stretch',
    'topo pass': 'pass',
    'up to current': 'current', 'last 25': 'last',
    # buttons
    'stripe match': 'stripe', 'Topo Window': 'Topo', 'load saved averages': 'load avgs',
    'cross section': 'x-section', 'set scale': 'scale', 'add existing': 'existing', 'add new': 'new',
    'Send to fit': 'fit', 'apply velocity': 'apply v', 'set from markers': 'from markers',
    'drift window': 'drift', 'clear markers': 'clear', 'clear points': 'clear',
    'remove last': 'remove', 'edit alignment': 'align', 'edit background': 'bkg',
    'line → new avg': 'line→new', 'line → existing avg': 'line→exist', 'map → new avg': 'map→new',
    'map → existing avg': 'map→exist', 'add to plot': 'add', 'new plot': 'new', 'Grid view': 'grid',
    'prev region': '< region', 'next region': 'region >',
    '< Prev': '<', 'Next >': '>', 'Add peak': 'peak', 'Revert fit': 'revert', 'Forward fit': 'forward',
    'Edit peaks': 'peaks', 'Save fit': 'save',
    # text-box and slider labels
    'scale factor': 'scale', 'min (min)': 'min', 'width (V)': 'width', 'v (nm/h, cal)': 'v',
    'calibrated length (nm)': 'cal (nm)',
    'reference': 'ref', 'E-markers': 'E-mk', 'Markers': 'mk',
}


def short_form(text):
    """The short form of a label text (SHORT_LABELS), or the text itself when it has none."""
    t = str(text)
    return SHORT_LABELS.get(t.strip(), t)


# ---------------------------------------------------------------------- #
#  Qt helpers                                                              #
# ---------------------------------------------------------------------- #

WIDGET_FONT_SCALE = 1.1      # the controls' text: a little larger than matplotlib's, smaller than the plots'
                             # (the controls keep their sizes in inches)


def _font(fs, zoom=1.0):
    return qfont(fs * WIDGET_FONT_SCALE / plt.FONT_SCALE, zoom=zoom)


def indicator_css(zoom):
    """Check-box and radio indicators as large as the text around them."""
    n = max(13, int(round(13 * zoom)))
    return (f"QCheckBox::indicator, QRadioButton::indicator {{ width: {n}px; height: {n}px; }}"
            if zoom > 1.15 else "")


def _refont_widgets(widgets_fs, zoom):
    """(Qt widget, font size) pairs at a figure's zoom: the widget and every child widget."""
    css = indicator_css(zoom)
    for qw, fs in widgets_fs:
        f = _font(fs, zoom)
        for w in [qw] + qw.findChildren(QtWidgets.QWidget):
            w.setFont(f)
            if isinstance(w, (QtWidgets.QCheckBox, QtWidgets.QRadioButton)):
                base = w.property('base_css') or ''
                w.setStyleSheet(base + css)


def filters_text(fc):
    """
    One console phrase of a filter box's state, per filter its settings and
    targets: 'Hampel window 4, n_sigma 3 on dI/dV and I(V); SG off'.
    """
    hp, sp = fc.params()
    hd, sd = fc.params_didv()

    def one(name, p_iv, p_d):
        targets = [t for t, p in (('dI/dV', p_d), ('I(V)', p_iv)) if p is not None]
        if not targets:
            return f"{name} off"
        p = p_iv if p_iv is not None else p_d
        return f"{name} " + ", ".join(f"{k} {v:g}" for k, v in p.items()) + " on " + " and ".join(targets)

    return one('Hampel', hp, hd) + "; " + one('SG', sp, sd)


class ViewNavigator:
    """
    Zoom and pan of one axes (the browser's scan area): a scroll notch zooms
    by ZOOM_STEP about the cursor, both axes together so an equal aspect
    holds; a left-button drag inside the axes pans while the button is down;
    a double-click centres the view on the clicked point at the same zoom
    (the span is kept); the toolbar's Home button and the Home key return to
    the full extent (full_extent(): (x0, x1, y0, y1)).  With the cursor over
    the axes, R saves the current view and space returns to the saved view
    (a region of the absolute frame, so it holds across file changes; a
    console line says when nothing is saved yet).  Nothing happens while
    the toolbar's pan or zoom mode is active, nor while the navigator is
    frozen (freeze(True): the browser freezes it while the section path is
    edited; a drag in progress is dropped); hold(predicate) registers
    another controller's claim on a press (DragPath while editing), so no
    pan starts from it.
    The axes box never changes size or shape: the aspect is kept equal by
    adjusting the axis ranges (adjustable 'datalim'), and every view set
    here is first widened on its shorter side to the box's own aspect
    (fit_to_box), so the data are never reshaped to fit the box.
    The view persists across redraws: keep(x0, x1, y0, y1, focus) after a
    redraw sets the full extent and restores the last view; when `focus`
    (x0, x1, y0, y1: the file just loaded) is not entirely inside the view,
    the view relocates to it, keeping its zoom (its span) and recentring on
    the focus, widening only when the focus does not fit at that zoom.
    on_change runs callback() after every change (the window redraws its
    canvas).
    """

    ZOOM_STEP = 1.2

    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        self.full = None
        self.view = None
        self._drag = None
        self._cb = None
        self.saved = None          # the view saved with R (x0, x1, y0, y1), returned to with space
        self._holds = []           # predicates: a press for which any returns True is not a pan (DragPath)
        self.frozen = False        # freeze(True): no zoom, pan, centring, Home, R or space (the path is edited)
        ax.set_aspect('equal', adjustable='datalim')
        ax.home_callback = self.reset          # the toolbar's Home button: the full extent
        fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        fig.canvas.mpl_connect('button_press_event', self._on_press)
        fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        fig.canvas.mpl_connect('button_release_event', self._on_release)
        fig.canvas.mpl_connect('key_press_event', self._on_key)

    def on_change(self, callback):
        self._cb = callback

    def hold(self, predicate):
        """Register predicate(event): True when another controller takes this press, so no pan starts."""
        self._holds.append(predicate)

    def freeze(self, flag):
        """
        Frozen, the view does not move: the scroll zoom, the drag pan, the double-click centring,
        Home (button and key), R and space do nothing (the browser freezes it while the section
        path is edited, so the scan area responds to the path's points only).
        """
        self.frozen = bool(flag)
        if self.frozen:
            self._drag = None

    def _toolbar_busy(self):
        tb = getattr(self.fig.canvas, 'toolbar', None)
        return tb is not None and bool(getattr(tb, 'mode', ''))

    def box_aspect(self):
        """The axes box's width over height in inches (the figure fraction times the figure size)."""
        fw, fh = self.fig.get_size_inches()
        x0, y0, w, h = self.ax.get_position().bounds
        return (w * fw) / (h * fh) if h * fh > 0 else 1.0

    def fit_to_box(self, view):
        """The view widened on its shorter side, about its centre, to the box's aspect (equal data aspect in a fixed box)."""
        x0, x1, y0, y1 = [float(v) for v in view]
        w, h = max(x1 - x0, 1e-12), max(y1 - y0, 1e-12)
        a = self.box_aspect()
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        if w / h < a:
            w = h * a
        else:
            h = w / a
        return (cx - 0.5 * w, cx + 0.5 * w, cy - 0.5 * h, cy + 0.5 * h)

    def _apply(self):
        if self.view is None:
            return
        self.view = self.fit_to_box(self.view)
        x0, x1, y0, y1 = self.view
        self.ax.set_aspect('equal', adjustable='datalim')
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)

    def keep(self, x0, x1, y0, y1, focus=None):
        """
        After a redraw: the full extent is (x0, x1, y0, y1).  The previous
        view is kept; with `focus` given and not entirely inside it, the view
        relocates: same span, centred on the focus, widened to hold the focus
        when it does not fit.  No previous view: the full extent.
        """
        self.full = self.fit_to_box((float(x0), float(x1), float(y0), float(y1)))
        if self.view is None:
            self.view = self.full
        elif focus is not None:
            fx0, fx1, fy0, fy1 = [float(v) for v in focus]
            vx0, vx1, vy0, vy1 = self.view
            inside = fx0 >= vx0 and fx1 <= vx1 and fy0 >= vy0 and fy1 <= vy1
            if not inside:
                w, h = vx1 - vx0, vy1 - vy0
                fw, fh = fx1 - fx0, fy1 - fy0
                if fw > w or fh > h:                       # the focus does not fit at this zoom: widen
                    s = max(fw / max(w, 1e-12), fh / max(h, 1e-12)) * 1.1
                    w, h = w * s, h * s
                cx, cy = 0.5 * (fx0 + fx1), 0.5 * (fy0 + fy1)
                self.view = (cx - 0.5 * w, cx + 0.5 * w, cy - 0.5 * h, cy + 0.5 * h)
        self._apply()

    def reset(self):
        if self.frozen:
            return
        self.view = self.full
        self._apply()
        self._changed()

    def centre_on(self, x, y):
        """The current view moved to centre on (x, y), its span kept (the double-click)."""
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        hw, hh = 0.5 * (x1 - x0), 0.5 * (y1 - y0)
        self.view = (float(x) - hw, float(x) + hw, float(y) - hh, float(y) + hh)
        self._apply()
        self._changed()

    def save_view(self):
        """R: the current view saved."""
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        self.saved = (float(x0), float(x1), float(y0), float(y1))
        print(f"view saved: x {x0:.2f} to {x1:.2f}, y {y0:.2f} to {y1:.2f} (space returns to it).")

    def restore_view(self):
        """Space: back to the saved view (a console line when none is saved)."""
        if self.saved is None:
            print("no saved view yet: R (cursor over the scan area) saves the current one.")
            return
        self.view = self.saved
        self._apply()
        self._changed()

    def _on_key(self, event):
        if self._toolbar_busy() or self.frozen:
            return
        if event.key == 'home':
            self.reset()
            event.handled = True
        elif event.inaxes is self.ax and event.key == 'r':
            self.save_view()
            event.handled = True
        elif event.inaxes is self.ax and event.key == ' ':
            self.restore_view()
            event.handled = True

    def _changed(self):
        if self._cb is not None:
            self._cb()
        self.fig.canvas.draw_idle()

    def _on_scroll(self, event):
        if event.inaxes is not self.ax or self._toolbar_busy() or event.xdata is None or self.frozen:
            return
        f = 1.0 / self.ZOOM_STEP if event.button == 'up' else self.ZOOM_STEP
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        cx, cy = float(event.xdata), float(event.ydata)
        self.view = (cx + (x0 - cx) * f, cx + (x1 - cx) * f, cy + (y0 - cy) * f, cy + (y1 - cy) * f)
        self._apply()
        self._changed()

    def _on_press(self, event):
        if event.inaxes is not self.ax or self._toolbar_busy() or event.button != 1 or self.frozen:
            return
        if any(h(event) for h in self._holds):
            return
        if getattr(event, 'dblclick', False):
            self._drag = None
            if event.xdata is not None:
                self.centre_on(event.xdata, event.ydata)
            return
        self._drag = (event.x, event.y, self.ax.get_xlim(), self.ax.get_ylim())

    def _on_motion(self, event):
        if self._drag is None or event.x is None:
            return
        px, py, (x0, x1), (y0, y1) = self._drag
        # Pixel motion to data units through the axes' current scale.
        bbox = self.ax.get_window_extent()
        if bbox.width <= 0 or bbox.height <= 0:
            return
        dx = (event.x - px) * (x1 - x0) / bbox.width
        dy = (event.y - py) * (y1 - y0) / bbox.height
        self.view = (x0 - dx, x1 - dx, y0 - dy, y1 - dy)
        self._apply()
        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        if self._drag is not None:
            self._drag = None
            self._changed()


class DragPath:
    """
    The browser's section path on the scan area: a polyline through any
    number of points, drawn numbered (plots.draw_section_path).  Editing
    (set_editing(True), the browser's 'section' button) makes the axes
    respond to the path alone (the browser freezes the ViewNavigator): a
    left click away from every point appends a point, a press within
    GRAB_PX display pixels of a point drags it while the button is down, a
    right click on a point removes it.  Editing off, the points stay where
    they were and the clicks go back to the navigation.  points() gives the
    (n, 2) array of points in the axes' data units (None without any),
    set(points) places a path, clear() removes it (the editing state stays
    as it was), rescale(factor) multiplies every point by factor (the
    browser's scale timeline changed under the path), draw() redraws it
    after the axes were cleared, active() says whether there is a path.
    on_change(callback) runs callback() after every addition, drag or
    removal; on_edit(callback) runs callback(flag) when editing turns on
    or off.
    """

    GRAB_PX = 10

    def __init__(self, fig, ax, navigator=None):
        self.fig = fig
        self.ax = ax
        self.p = []                   # the points, (2,) arrays in data units
        self.editing = False
        self._arts = []
        self._drag = None
        self._cb = None
        self._cb_edit = None
        if navigator is not None:
            navigator.hold(lambda ev: self.editing)
        fig.canvas.mpl_connect('button_press_event', self._on_press)
        fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        fig.canvas.mpl_connect('button_release_event', self._on_release)

    def on_change(self, callback):
        self._cb = callback

    def on_edit(self, callback):
        self._cb_edit = callback

    def set_editing(self, flag):
        flag = bool(flag)
        if flag == self.editing:
            return
        self.editing = flag
        self._drag = None
        if self._cb_edit is not None:
            self._cb_edit(flag)

    def active(self):
        return bool(self.p)

    def points(self):
        return None if not self.p else np.array(self.p, dtype=np.float64).reshape(-1, 2)

    def set(self, points):
        self.p = [np.array(p, dtype=np.float64).reshape(2) for p in np.asarray(points, dtype=np.float64).reshape(-1, 2)]
        self.draw()

    def clear(self):
        self.p = []
        self._drag = None
        self._remove()
        self.fig.canvas.draw_idle()

    def rescale(self, factor):
        """Every point multiplied by factor (the drawing units changed: a new scale factor for the current file)."""
        self.p = [p * float(factor) for p in self.p]
        self.draw()

    def _remove(self):
        for a in self._arts:
            try:
                a.remove()
            except Exception:
                pass
        self._arts = []

    def draw(self):
        from sm4_Browser import plots
        self._remove()
        if self.p:
            self._arts = plots.draw_section_path(self.ax, np.array(self.p))
        self.fig.canvas.draw_idle()

    def _changed(self):
        if self._cb is not None:
            self._cb()

    def _grab(self, event):
        """The index of the point within GRAB_PX display pixels of a press inside the axes, else None."""
        if not self.p or event.inaxes is not self.ax or event.x is None:
            return None
        best = None
        for k, pt in enumerate(self.p):
            dx, dy = self.ax.transData.transform(pt) - (event.x, event.y)
            dist = float(np.hypot(dx, dy))
            if dist <= self.GRAB_PX and (best is None or dist < best[0]):
                best = (dist, k)
        return None if best is None else best[1]

    def _on_press(self, event):
        if not self.editing or event.inaxes is not self.ax or event.xdata is None:
            return
        k = self._grab(event)
        if event.button == 1:
            if k is not None:
                self._drag = (k, False)
            else:
                self.p.append(np.array([event.xdata, event.ydata], dtype=np.float64))
                self.draw()
                self._changed()
        elif event.button == 3 and k is not None:
            del self.p[k]
            self.draw()
            self._changed()

    def _on_motion(self, event):
        if self._drag is None or event.xdata is None or event.inaxes is not self.ax:
            return
        k, _ = self._drag
        self.p[k] = np.array([event.xdata, event.ydata], dtype=np.float64)
        self._drag = (k, True)
        self.draw()

    def _on_release(self, event):
        if self._drag is not None:
            _, moved = self._drag
            self._drag = None
            if moved:
                self._changed()


MARKER_PAIRS_MAX = 8          # marker pairs per span: one colour each (plots.MARKER_COLORS); the drift window's 'Markers' top


def _moved(m):
    """Whether a seeded marker was dragged off its seeded spot by more than half a pixel of its topo."""
    if m.get('seed') is None:
        return False
    tol = 0.5 * float(m.get('px_m') or 0.0)
    return float(np.linalg.norm(np.asarray(m['xy_m'], dtype=np.float64) - np.asarray(m['seed'], dtype=np.float64))) > tol


def _pair_line(k, ma, mb):
    """One readout line per marker pair of a draft: seeded and unmoved, seeded and adjusted by how much, or placed by hand."""
    seeded = mb if mb.get('seed') is not None else (ma if ma.get('seed') is not None else None)
    if seeded is None:
        return f"pair {k + 1}: placed by hand on both topos"
    if not _moved(seeded):
        return f"pair {k + 1}: seeded by the auto velocity, unmoved"
    off = (np.asarray(seeded['xy_m'], dtype=np.float64) - np.asarray(seeded['seed'], dtype=np.float64)) * 1e9
    return f"pair {k + 1}: seeded by the auto velocity, adjusted by ({off[0]:+.3f}, {off[1]:+.3f}) nm"


class PairMarkers:
    """
    The marker pairs of the drift window: two panels, the earlier topo's and
    the later's ((ax, path, name) each), every click in a panel placing a
    numbered marker on that panel's topo (press and release without
    motion), a press on a marker dragging it; each topo holds at most
    n_pairs markers (set_pairs; lowering it drops the highest-numbered, 0
    removes every marker; a click beyond the count raises it, on_count(n)
    telling the window);
    the k-th marker of the earlier topo pairs with the k-th of the later,
    pair k in its own colour (plots.marker_color).  resolve_on(path, x, y)
    -> dict with 'xy_m' (absolute metres), 't' (the pixel's acquisition
    time), and 'px_m' (the pixel size) gives a marker its data at placement
    and after a drag (x, y in the panel's drawing units).  `seed(path,
    xy_m, t)` -> (other path, xy_m, t, px_m) or None (optionally a fifth
    element, the velocity's source) gives the partner the current velocity
    places (drift.py, "Seeded markers"): a placed marker places its partner
    in the other panel at once when that panel is level with it (a click
    that completes a pair by hand seeds nothing); dragging the placed
    marker re-seeds an unmoved partner; dragging the partner adjusts it
    (moved = more than half a pixel off the seeded spot).  `to_display(xy_m,
    path)` -> drawing units of an absolute position on the panel of `path`
    (the drift window: each topo at its own scale); without it a seeded
    partner is placed with the clicked marker's own ratio of drawing units
    to metres (one scale for both panels).  `on_count(n)` is told when a
    click raised the pair count (the window's 'Markers' count box follows).
    The markers are a draft: the window applies them.  span() gives (earlier
    path, later path, markers (n, 6): x_A, y_A, t_A, x_B, y_B, t_B) or
    (None, None, None) while a topo has no marker; confirmable() is True
    when every pair is seeded and unmoved; seed_sources() the velocity
    sources that seeded the unmoved markers; pair_lines() the readout lines
    per pair; status() the marker counts per panel; set_pairs(n) and
    clear() change the draft (and fire); retarget(panels) takes new panels
    with an empty draft (the drift window stepping to another pair);
    restore(markers, to_display, kind) puts a stored span's markers back (a
    confirmed span's later markers as seeded and unmoved); draw() redraws
    the markers (the pairs' arrows on every panel, from the earlier
    marker's position to its partner's, plots.draw_drift_markers);
    on_change(callback) runs callback() after every placement, drag, or
    removal.
    """

    GRAB_PX = 10
    CLICK_PX = 3

    def __init__(self, fig, panels, resolve_on, n_pairs=1, seed=None, on_count=None, to_display=None):
        self.fig = fig
        self.panels = list(panels)
        self.resolve_on = resolve_on
        self.seed = seed
        # to_display(xy_m, path) -> drawing units of a position on the panel of `path`: each topo at its
        # own scale (the drift window); None: one drawing scale for both panels (a marker's own ratio).
        self.to_display = to_display
        self.on_count = on_count           # callback(n) when a click raised the pair count (the window's 'Markers' box)
        self.n_pairs = min(max(int(n_pairs), 0), MARKER_PAIRS_MAX)
        self.markers = {path: [] for _, path, _ in self.panels}     # path -> [{'xy', 'xy_m', 't', 'px_m', 'seed'}]
        self._arts = []
        self._drag = None
        self._press = None
        self._cb = None
        fig.canvas.mpl_connect('button_press_event', self._on_press)
        fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        fig.canvas.mpl_connect('button_release_event', self._on_release)

    def on_change(self, callback):
        self._cb = callback

    def _fire(self):
        if self._cb is not None:
            self._cb()

    def _panel_of(self, event):
        for ax, path, name in self.panels:
            if event.inaxes is ax:
                return ax, path, name
        return None

    def _other(self, path):
        return next(p for _, p, _ in self.panels if p != path)

    @staticmethod
    def _record(xy_m, t, xy, px_m=None, seed=None):
        return {'xy_m': np.asarray(xy_m, dtype=np.float64).reshape(2).copy(), 't': float(t),
                'xy': np.asarray(xy, dtype=np.float64).reshape(2).copy(),
                'px_m': None if px_m is None else float(px_m),
                'seed': None if seed is None else np.asarray(seed, dtype=np.float64).reshape(2).copy()}

    def set_pairs(self, n):
        self.n_pairs = min(max(int(n), 0), MARKER_PAIRS_MAX)
        changed = False
        for path, mine in self.markers.items():
            if len(mine) > self.n_pairs:
                del mine[self.n_pairs:]
                changed = True
        if changed:
            self.draw()
            self._fire()

    def clear(self):
        if any(self.markers.values()):
            for path in self.markers:
                self.markers[path] = []
            self.draw()
            self._fire()

    def retarget(self, panels):
        """
        The panels replaced (the drift window stepping to another pair of
        topos): the draft emptied and its artists dropped (the panels were
        cleared by the window), the new (ax, path, name) panels taken, the
        pair count kept; nothing fired.
        """
        for a in self._arts:
            try:
                a.remove()
            except Exception:
                pass
        self._arts = []
        self._drag = None
        self._press = None
        self.panels = list(panels)
        self.markers = {path: [] for _, path, _ in self.panels}

    def restore(self, markers, to_display=None, kind='adjusted'):
        """
        Markers from a stored span ((n, 6) metres and seconds; the earlier
        topo's columns first), to_display(xy_m) -> drawing units (None: the
        constructor's to_display(xy_m, path), each panel at its own scale);
        a confirmed span's later markers come back as seeded and unmoved.
        """
        (_, a, _), (_, b, _) = self.panels[0], self.panels[1]
        disp = (lambda xy_m, path: to_display(xy_m)) if to_display is not None else self.to_display
        self.markers = {a: [], b: []}
        for row in np.asarray(markers, dtype=np.float64).reshape(-1, 6):
            self.markers[a].append(self._record(row[0:2], row[2], disp(row[0:2], a)))
            self.markers[b].append(self._record(row[3:5], row[5], disp(row[3:5], b),
                                                seed=row[3:5] if kind == 'confirmed' else None))
        self.n_pairs = min(max(self.n_pairs, len(self.markers[a])), MARKER_PAIRS_MAX)
        self.draw()

    def _pairs(self):
        (_, a, _), (_, b, _) = self.panels[0], self.panels[1]
        return list(zip(self.markers[a], self.markers[b]))

    def span(self):
        (_, a, _), (_, b, _) = self.panels[0], self.panels[1]
        pairs = self._pairs()
        if not pairs:
            return None, None, None
        rows = [[*ma['xy_m'], ma['t'], *mb['xy_m'], mb['t']] for ma, mb in pairs]
        return a, b, np.array(rows, dtype=np.float64)

    def seed_sources(self):
        """The velocity sources that seeded the unmoved markers of the draft (e.g. {'auto (pair)'})."""
        out = set()
        for ma, mb in self._pairs():
            for m in (ma, mb):
                if m.get('seed') is not None and not _moved(m):
                    out.add(m.get('seed_source', 'auto'))
        return out

    def confirmable(self):
        """True when the span exists and every pair of it has a seeded, unmoved marker."""
        pairs = self._pairs()
        return bool(pairs) and all((ma.get('seed') is not None and not _moved(ma)) or
                                   (mb.get('seed') is not None and not _moved(mb)) for ma, mb in pairs)

    def pair_lines(self):
        """The readout lines per pair (seeded and unmoved, adjusted, or by hand)."""
        return [_pair_line(k, ma, mb) for k, (ma, mb) in enumerate(self._pairs())]

    def status(self):
        parts = [f"{len(self.markers[path])} on {name}" for _, path, name in self.panels]
        return "drift markers: " + ", ".join(parts)

    def _remove(self):
        for art in self._arts:
            try:
                art.remove()
            except Exception:
                pass
        self._arts = []

    def draw(self):
        from sm4_Browser import plots
        self._remove()
        (ax_a, a, _), (ax_b, b, _) = self.panels[0], self.panels[1]
        marks_a = [{'xy': m['xy'], 'side': 0, 'index': k + 1, 'seeded': m.get('seed') is not None, 'moved': _moved(m)}
                   for k, m in enumerate(self.markers[a])]
        marks_b = [{'xy': m['xy'], 'side': 1, 'index': k + 1, 'seeded': m.get('seed') is not None, 'moved': _moved(m)}
                   for k, m in enumerate(self.markers[b])]
        pairs = [(ma['xy'], mb['xy'], k + 1) for k, (ma, mb) in enumerate(self._pairs())]
        # The pairs' displacement arrows on every panel (the panels share the absolute frame).
        self._arts = plots.draw_drift_markers(ax_a, marks_a, pairs) + plots.draw_drift_markers(ax_b, marks_b, pairs)
        self.fig.canvas.draw_idle()

    def _grab(self, event):
        panel = self._panel_of(event)
        if panel is None or event.button != 1 or event.x is None:
            return None
        ax, path, _ = panel
        best = None
        for k, m in enumerate(self.markers[path]):
            dx, dy = ax.transData.transform(m['xy']) - (event.x, event.y)
            dist = float(np.hypot(dx, dy))
            if dist <= self.GRAB_PX and (best is None or dist < best[0]):
                best = (dist, path, k)
        return None if best is None else (best[1], best[2])

    def _on_press(self, event):
        g = self._grab(event)
        if g is not None:
            self._drag = g
            return
        panel = self._panel_of(event)
        if panel is not None and event.button == 1 and event.x is not None:
            self._press = (event.x, event.y, event.xdata, event.ydata, panel)

    def _on_motion(self, event):
        if self._drag is None or event.xdata is None:
            return
        path, k = self._drag
        ax = next(ax for ax, p, _ in self.panels if p == path)
        if event.inaxes is not ax:
            return
        self.markers[path][k]['xy'] = np.array([event.xdata, event.ydata], dtype=np.float64)
        self.draw()

    def _seed_from(self, path, m):
        """The partner (other path, xy_m, t, px_m) the auto velocity gives for a hand-placed marker m, or None."""
        if self.seed is None or m.get('seed') is not None:
            return None
        try:
            res = self.seed(path, m['xy_m'], m['t'])
        except Exception as exc:
            print(f"drift markers: no seed ({exc}).")
            return None
        if res is None or res[0] == path:
            return None
        return res

    def _try_seed(self, path, m, scale):
        """A just-placed marker m (panel `path`) seeds its partner in the other panel when that panel is exactly one marker behind."""
        res = self._seed_from(path, m)
        if res is None:
            return
        other, xy_m, t, px_m = res[:4]
        source = res[4] if len(res) > 4 else 'auto'
        k = next(i for i, x in enumerate(self.markers[path]) if x is m)    # by identity (dicts of arrays)
        if len(self.markers[other]) != k:
            return
        xy = self.to_display(xy_m, other) if self.to_display is not None else np.asarray(xy_m, dtype=np.float64) * scale
        rec = self._record(xy_m, t, xy, px_m, seed=xy_m)
        rec['seed_source'] = source
        self.markers[other].append(rec)
        name = next(n for _, p, n in self.panels if p == other)
        print(f"drift markers: marker {k + 1} placed on {name} by the {source} velocity (drag it to adjust).")

    def _reseed_partner(self, path, k, scale):
        """A dragged hand-placed marker moves its seeded, unmoved partner to the new seed."""
        other = self._other(path)
        if k >= len(self.markers[other]):
            return
        p = self.markers[other][k]
        if p.get('seed') is None or _moved(p):
            return
        res = self._seed_from(path, self.markers[path][k])
        if res is None or res[0] != other:
            return
        _, xy_m, t, px_m = res[:4]
        xy = self.to_display(xy_m, other) if self.to_display is not None else np.asarray(xy_m, dtype=np.float64) * scale
        rec = self._record(xy_m, t, xy, px_m, seed=xy_m)
        rec['seed_source'] = res[4] if len(res) > 4 else 'auto'
        self.markers[other][k] = rec

    def _scale(self, m):
        """Drawing units per metre of a marker (its 'xy' over its 'xy_m'); used for a seed only without to_display, when the panels share one scale."""
        xy, xy_m = np.asarray(m['xy'], dtype=np.float64), np.asarray(m['xy_m'], dtype=np.float64)
        j = int(np.argmax(np.abs(xy_m)))
        return float(xy[j] / xy_m[j]) if xy_m[j] != 0 else 1.0

    def _on_release(self, event):
        if self._drag is not None:
            path, k = self._drag
            self._drag = None
            m = self.markers[path][k]
            res = self.resolve_on(path, float(m['xy'][0]), float(m['xy'][1]))
            if res is not None:
                m['xy_m'], m['t'] = np.asarray(res['xy_m'], dtype=np.float64), float(res['t'])
                if res.get('px_m') is not None:
                    m['px_m'] = float(res['px_m'])
            if m.get('seed') is None:
                self._reseed_partner(path, k, self._scale(m))
            self.draw()
            self._fire()
            return
        if self._press is None:
            return
        x0, y0, xd, yd, (ax, path, name) = self._press
        self._press = None
        if event.x is None or np.hypot(event.x - x0, event.y - y0) > self.CLICK_PX or event.inaxes is not ax:
            return
        if len(self.markers[path]) >= self.n_pairs:
            if self.n_pairs >= MARKER_PAIRS_MAX:
                print(f"drift markers: {name} holds the most markers ({MARKER_PAIRS_MAX}); drag one or clear.")
                return
            self.n_pairs += 1                          # every click adds a pair; the window's 'Markers' box follows
            if self.on_count is not None:
                self.on_count(self.n_pairs)
        res = self.resolve_on(path, float(xd), float(yd))
        if res is None:
            print(f"drift markers: the click lies outside {name}.")
            return
        m = self._record(res['xy_m'], res['t'], [xd, yd], res.get('px_m'))
        self.markers[path].append(m)
        self._try_seed(path, m, self._scale(m))
        self.draw()
        self._fire()


if __name__ == '__main__':
    # Every control of the specification built once by name (controls.make), each in a box frame
    # titled with its name, in one wrapping row of the window's layout (automatic layout: the one
    # rule for every window), so that sizes and labels can be checked at a glance.
    from sm4_Browser import controls
    fig = plt.figure(figsize=(16, 9))
    flow = controls.FlowLayout()
    keep = []
    for name in SPECS:
        kw = {'labels': ['Slope Sub', 'Δz sub']} if name == 'topo processors' else {}
        c = controls.make(fig, name, **kw)
        keep.append(c)
        flow.addWidget(controls.box(c, title=name))
    root = QtWidgets.QVBoxLayout()
    root.addLayout(flow)
    root.addStretch(1)
    fig.set_layout(root)
    plt.show()
