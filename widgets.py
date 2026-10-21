"""
Every control of the package: one specification per widget name (SPECS: the
label, the kind, the size in inches, the font size) so that a widget of a
given name is identical in every window (the windows only say where it
sits), one wrapper per kind with one interface (value, set, on_change,
show, enable, place), and the composite controllers that several windows
share so that same-named controls are the same code: FilterControls,
PipelineBox, FeenstraSettings, StackControls, MarkerGroupControls,
TopoProcessorBox, ResampleControls, RunControls, GammaBox,
NameBox, StepButtons, AveragePicker, ViewNavigator (the scan area's zoom
and pan), DragPath (the browser's section path), PairMarkers (the marker
pairs of the drift window's two topos, each topo at its own scale, the
partners seeded by the velocity the window gives).  The drawing of data lives in
plots.py and never touches a widget; a window owns its data and state,
draws through plots, and receives events from here.

Every control is a native Qt widget (PySide6) laid over the figure's
pyqtgraph canvas (qtplot) at its figure fraction, with its specification's
size in inches: a click reaches the control directly (no redraw of the
figure), and the figure is only redrawn where the window's callback
changes it.

make(fig, name, x, y, ...) builds the widget named `name` at the figure
fraction (x, y) (its lower-left corner) with the size of its specification.
A window never builds a Qt widget itself.

Text size and short labels: a control's text follows its figure's zoom
(qtplot.window_zoom: the window's size relative to its design size, 0.5x to
4x), shrinking to at most 75 % of that when its box is too small and the box
then growing to fit, so no text is ever cut.  When the window is taller than
it is wide (qtplot.Figure.set_portrait) every label with a short form in
SHORT_LABELS shows it (Control.set_short), and the full label returns when
the window is wider than tall again; the browser's native controls use the
same table (browser.QControl.set_short).
"""

if __package__ in (None, ''):
    # Run as a file (Spyder F5): make the package importable from wherever its folder sits.
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np
from sm4_Browser import qtplot as plt
from sm4_Browser.qtplot import QtCore, QtGui, QtWidgets, qfont, qcolor, Bbox
from sm4_Browser.topo import TOPO_DIRECTIONS, toggle_processor

Qt = QtCore.Qt


def figure_is_open(fig):
    """True while `fig` itself is still open (a closed figure's number may be reused by a later figure)."""
    try:
        return plt.fignum_exists(fig)
    except Exception:
        return False


# ---------------------------------------------------------------------- #
#  Specifications: name -> kind, size (inches), labels, font size          #
# ---------------------------------------------------------------------- #

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
    'stripe match': _spec('button', 1.2, 0.36),                   # grids: opens the stripe window (stripe.StripeWindow)
    'anchor to topo': _spec('check', 1.45, 0.38),                 # the MATLAB "drift for first point" (rigid shift): the browser's and the Topo window's toggle
    'cmap norm': _spec('check', 1.4, 0.38),
    'overlapping': _spec('check', 0.85, 0.38, fs=7),              # the line-average window: average only where every member covers a pixel
    'WKB background': _spec('check', 1.7, 0.32),
    'hold': _spec('check', 0.3, 0.26, fs=7, labels=['']),           # the fit edit window's hold rows (unlabeled)
    # --- toggle boxes (several rows) ---
    'pipeline': _spec('checks', 0.95, 0.52, fs=8, labels=['calibrate', 'normalize']),
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
    '< spectra': _spec('button', 0.8, 0.32, fs=7),
    'spectra >': _spec('button', 0.8, 0.32, fs=7),
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
    'Markers': _spec('slider', 1.1, 0.12, fs=8, vmin=0, vmax=10, valstep=1),   # the drift window passes vmax=MARKER_PAIRS_MAX
    'avgs': _spec('slider', 1.1, 0.22, fs=8, vmin=0, vmax=10, valstep=1),
    'E-markers': _spec('slider', 1.1, 0.14, fs=8, vmin=0, vmax=8, valstep=1),     # the Line window's energy markers
    'file': _spec('slider', 7.0, 0.2, fs=9, vmin=0, vmax=1, valstep=1, valfmt='%d'),
    'spectra': _spec('slider', 1.8, 0.2, fs=9, vmin=0, vmax=1, valstep=1, valfmt='%d'),
    # --- text boxes ---
    'γ': _spec('text', 0.45, 0.24, fs=7),
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
          'hold': '', 'param': '', 'rows': '', 'width': '', 'file': 'file', 'spectra': 'spectra',
          'name': 'name ', 'line': 'line ', 'scale factor': 'scale factor ', 'reference': 'reference ',
          'cal length': 'calibrated length (nm) ',
          'γ': 'γ ', 'drift v': 'v (nm/h, cal) ', 'E lo': '', 'E hi': '', 'x lo': '', 'x hi': '',
          'slow axis': 'drift', 'fast axis': 'stripe', 'stripe pass': 'pass'}        # the radios' titles (Radio, title=True)

# The short form of every label that has one, used when a window is taller than it is wide (the
# figure's portrait flag, qtplot.Figure.set_portrait; the browser's window, browser._WindowZoom):
# a control whose label is a key here shows the value instead, and the full label when the window
# is wider than tall again.  Keys are the label texts as shown (LABELS applied), stripped.  A label
# not listed keeps its text in both modes.  One table for the figure windows' controls
# (Control.set_short) and the browser's native controls (browser.QControl.set_short) alike.
SHORT_LABELS = {
    # toggles and boxes
    'calibrate': 'cal', 'normalize': 'norm', 'auto drift': 'auto', 'manual drift': 'manual',
    'anchor to topo': 'anchor', 'broadened': 'broad', 'substack': 'sub', 'Average': 'Avg',
    'Feenstra': 'Feen', 'Feenstra (refit)': 'Feen refit', 'cmap norm': 'cmap', 'WKB background': 'WKB',
    'overlapping': 'overlap',
    'step through': 'step', 'topo processing': 'topo proc', 'filters (on the current)': 'filters',
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
    'prev region': '< region', 'next region': 'region >', '< spectra': '<', 'spectra >': '>',
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

# The boxes of several rows (check boxes, radios, the filter box) as light group boxes on the window
# colour, single toggles, buttons and text boxes as native Qt controls.
_FRAME_CSS = "QFrame#box { background: %s; border: 1px solid #c8c8c8; border-radius: 3px; }"
_FRAME_CSS_OFF = "QFrame#box { background: transparent; border: none; }"


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


class _LabelProxy:
    """
    A control's label text as matplotlib's Text (set_color, set_text, set_fontsize, get_text).
    `setter(text)`, when given, is the control's own way of changing that label, so that the text
    set here is the label the control shows in its full or short form (Control.set_short).
    """

    def __init__(self, qwidget, setter=None):
        self._w = qwidget
        self._setter = setter
        self._color = None

    def set_color(self, c):
        col = qcolor(c).name()
        if col != self._color:
            self._color = col
            base = f"color: {col};"
            self._w.setProperty('base_css', base)
            zoom = getattr(self._w, '_zoom_css', '')
            self._w.setStyleSheet(base + (self._w.property('ind_css') or ''))

    def set_text(self, s):
        if self._setter is not None:
            self._setter(str(s))
        else:
            self._w.setText(str(s))

    def get_text(self):
        return self._w.text()

    def set_fontsize(self, fs):
        self._w.setFont(_font(fs))


class _AxesProxy:
    """The control's place in the figure, as matplotlib's widget axes: set_position, get_position, set_visible, set_frame_on."""

    def __init__(self, control):
        self._c = control

    def set_position(self, rect):
        b = rect.bounds if hasattr(rect, 'bounds') else rect
        self._c._rect = [float(v) for v in b]
        self._c._relayout()

    def get_position(self):
        return Bbox(*self._c._rect)

    def set_visible(self, flag):
        self._c._set_visible(flag)

    def get_visible(self):
        return self._c._visible

    def set_frame_on(self, b):
        self._c._set_frame(b)

    def set_navigate(self, b):
        pass

    def remove(self):
        self._c.remove()

    def _qwidget(self):
        return self._c._main


# ---------------------------------------------------------------------- #
#  Wrappers                                                               #
# ---------------------------------------------------------------------- #

class Control:
    """
    One widget: its Qt widgets over the figure, its rectangle in figure
    fractions (the specification's size at the figure size it was made at,
    scaling with the window as matplotlib's widget axes do), and the
    interface every kind shares.  `ax` is its place (set_position,
    get_position, set_visible, set_frame_on); `widget` the main Qt widget.
    """

    def __init__(self, fig, name, x, y, labels=None, label=None):
        self.fig = fig
        self.name = name
        self.spec = dict(SPECS[name])
        if labels is not None:
            self.spec['labels'] = list(labels)
        self.label = LABELS.get(name, name) if label is None else label
        self._short = False       # short labels on: the window is taller than wide (set_short)
        self._cb = None
        self._quiet = False
        self._visible = True
        self._enabled = True
        self._frame_on = True
        w, h = self._size()
        self._rect = [float(x), float(y), w, h]
        self._parent = fig.canvas.viewport()
        self._qws = []            # every Qt widget of the control
        self._main = None
        self.widget = None
        self.eventson = True
        self.drawon = True
        self.ax = _AxesProxy(self)

    def _add(self, qw):
        qw.setParent(self._parent)
        self._qws.append(qw)
        qw.show()
        return qw

    def _finish(self):
        self.fig._register_control(self)

    def _size(self):
        fw, fh = self.fig.get_size_inches()
        return self.spec['w'] / float(fw), self.spec['h'] / float(fh)

    def _rect_px(self, W, H):
        x, y, w, h = self._rect
        return x * W, (1.0 - y - h) * H, w * W, h * H

    def _text_need(self, zoom):
        """(width, height) in pixels the control's own text needs at a font zoom, or None (no text inside)."""
        return None

    def shown(self, text):
        """The text a label shows: its short form (SHORT_LABELS) in the short mode, else as it is."""
        return short_form(text) if self._short else str(text)

    def _apply_labels(self):
        """Every label of the control set to its shown form (the kinds with text override this)."""
        return None

    def set_short(self, flag):
        """
        Short labels on or off (the figure's shape, qtplot.Figure.set_portrait: on when the
        window is taller than it is wide).  Every label with a short form in SHORT_LABELS
        switches, and the control is placed again so that its box fits the new text.
        """
        flag = bool(flag)
        if flag != self._short:
            self._short = flag
            self._apply_labels()
            self._relayout()

    def _set_zoom(self, z):
        if abs(z - getattr(self, '_zoom', 1.0)) > 1e-6:
            self._zoom = z
            _refont_widgets([(qw, self.spec['fs']) for qw in self._qws], z)

    def _place(self, W, H):
        z = getattr(self.fig, 'font_zoom', 1.0)
        x0, y0, w, h = self._rect_px(W, H)
        # The text is never cut: a box too small for its words first takes a slightly smaller font
        # (down to 75 %), then grows to fit (to the right, and upward for height).
        need = self._text_need(z)
        f = 1.0
        if need is not None and (need[0] > w or need[1] > h):
            f = max(0.75, min(w / need[0] if need[0] else 1.0, h / need[1] if need[1] else 1.0))
            need = self._text_need(z * f)
        self._set_zoom(z * f)
        if need is not None:
            if need[0] > w:
                w = need[0]
            if need[1] > h:
                y0 -= need[1] - h
                h = need[1]
        self._layout(int(round(x0)), int(round(y0)), max(int(round(w)), 1), max(int(round(h)), 1))

    def _layout(self, x, y, w, h):
        if self._main is not None:
            self._main.setGeometry(x, y, w, h)

    def _relayout(self):
        W, H = self.fig._canvas_size()
        self._place(W, H)

    def _set_visible(self, flag):
        self._visible = bool(flag)
        for qw in self._qws:
            qw.setVisible(self._visible)

    def _set_frame(self, b):
        self._frame_on = bool(b)

    def place(self, x, y):
        """Move the widget's lower-left corner to the figure fraction (x, y); the size stays the spec's."""
        self._rect[0], self._rect[1] = float(x), float(y)
        self._relayout()

    def place_in(self, x_in, y_in):
        """Move the widget's lower-left corner to (x_in, y_in) inches from the figure's lower left."""
        fw, fh = self.fig.get_size_inches()
        w, h = self._size()
        self._rect = [x_in / float(fw), y_in / float(fh), w, h]
        self._relayout()

    def place_center(self, cx, cy):
        """
        Move the widget so that its centre sits at the figure fraction
        (cx, cy), kept inside the figure; its size follows the figure's
        current size (as below_corner).
        """
        ww, hh = self._size()
        self._rect[2], self._rect[3] = ww, hh
        self.place(float(np.clip(cx - 0.5 * ww, 0.0, 1.0 - ww)), float(np.clip(cy - 0.5 * hh, 0.0, 1.0 - hh)))

    def below_corner(self, ax, gap_in=0.05, right_in=0.0):
        """
        Place the widget just below the bottom-right corner of `ax`, outside
        the axes: its right edge at the axes' right edge (shifted right by
        right_in inches), its top gap_in inches under the axes' bottom.
        """
        fw, fh = self.fig.get_size_inches()
        x0, y0, w, h = ax.get_position().bounds
        ww, hh = self._size()
        self._rect[2], self._rect[3] = ww, hh
        self.place(x0 + w - ww + right_in / float(fw), y0 - hh - gap_in / float(fh))

    def show(self, flag):
        self._set_visible(flag)
        self.enable(flag)

    def enable(self, flag):
        """The widget takes events (True) or ignores them (False)."""
        self._enabled = bool(flag)
        for qw in self._qws:
            qw.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, not self._enabled)

    def on_change(self, callback):
        self._cb = callback

    def _fire(self, *args):
        if self._cb is not None and not self._quiet and self.eventson:
            self._cb(*args)

    def remove(self):
        self.fig._unregister_control(self)
        for qw in self._qws:
            qw.hide()
            qw.deleteLater()
        self._qws = []

    def __getattr__(self, item):
        w = self.__dict__.get('widget')
        if w is None:
            raise AttributeError(item)
        return getattr(w, item)


class _Boxed(Control):
    """A control in a framed white box (matplotlib's check and radio buttons)."""

    def _make_frame(self):
        frame = QtWidgets.QFrame()
        frame.setObjectName('box')
        frame.setStyleSheet(_FRAME_CSS % '#f0f0f0')
        self._frame = frame
        self._main = self._add(frame)
        return frame

    def _set_frame(self, b):
        self._frame_on = bool(b)
        self._frame.setStyleSheet(_FRAME_CSS % '#f0f0f0' if b else _FRAME_CSS_OFF)


class Check(_Boxed):
    """A one-row toggle: value is its bool state; set(flag) never fires the callback; callback(flag)."""

    def __init__(self, fig, name, x, y, value=False, label=None):
        super().__init__(fig, name, x, y, label=label)
        frame = self._make_frame()
        self._set_frame(False)
        lay = QtWidgets.QHBoxLayout(frame)
        lay.setContentsMargins(4, 0, 2, 0)
        cb = QtWidgets.QCheckBox(self.label)
        cb.setFont(_font(self.spec['fs']))
        cb.setChecked(bool(value))
        lay.addWidget(cb)
        self.widget = cb
        self._labels = [_LabelProxy(cb, self._set_label)]
        cb.toggled.connect(lambda flag: self._fire(self.value))
        self._finish()

    def _set_label(self, text):
        self.label = str(text)
        self._apply_labels()
        self._relayout()

    def _apply_labels(self):
        self.widget.setText(self.shown(self.label))

    def _text_need(self, zoom):
        fm = QtGui.QFontMetricsF(_font(self.spec['fs'], zoom))
        ind = self.widget.style().pixelMetric(QtWidgets.QStyle.PixelMetric.PM_IndicatorWidth, None, self.widget)
        sp = self.widget.style().pixelMetric(QtWidgets.QStyle.PixelMetric.PM_CheckBoxLabelSpacing, None, self.widget)
        return (ind + sp + fm.horizontalAdvance(self.widget.text()) + 10, max(fm.height(), ind) + 2)

    @property
    def labels(self):
        return self._labels

    def get_status(self):
        return [self.value]

    @property
    def value(self):
        return bool(self.widget.isChecked())

    def set(self, flag):
        if self.value != bool(flag):
            self._quiet = True
            self.widget.blockSignals(True)
            try:
                self.widget.setChecked(bool(flag))
            finally:
                self.widget.blockSignals(False)
                self._quiet = False


class Checks(_Boxed):
    """A box of toggle rows: value is the list of states; set(i, flag) never fires; callback(label, states)."""

    def __init__(self, fig, name, x, y, values=None, labels=None):
        super().__init__(fig, name, x, y, labels=labels)
        labs = self.spec['labels']
        if self.spec.get('row_h'):
            self.spec['h'] = self.spec['row_h'] * len(labs) + 0.05
            self._rect[3] = self._size()[1]
        vals = [False] * len(labs) if values is None else [bool(v) for v in values]
        frame = self._make_frame()
        lay = QtWidgets.QVBoxLayout(frame)
        lay.setContentsMargins(4, 1, 2, 1)
        lay.setSpacing(0)
        self.boxes = []
        for lab, v in zip(labs, vals):
            cb = QtWidgets.QCheckBox(lab)
            cb.setFont(_font(self.spec['fs']))
            cb.setChecked(v)
            cb.toggled.connect(lambda flag, lab=lab: self._fire(lab, self.value))
            lay.addWidget(cb)
            self.boxes.append(cb)
        self.widget = frame
        self._labels = [_LabelProxy(cb, lambda s, i=i: self._set_label(i, s)) for i, cb in enumerate(self.boxes)]
        self._finish()

    def _set_label(self, i, text):
        self.spec['labels'][i] = str(text)
        self._apply_labels()
        self._relayout()

    def _apply_labels(self):
        for cb, lab in zip(self.boxes, self.spec['labels']):
            cb.setText(self.shown(lab))

    def _text_need(self, zoom):
        fm = QtGui.QFontMetricsF(_font(self.spec['fs'], zoom))
        st = self.boxes[0].style() if self.boxes else QtWidgets.QApplication.style()
        ind = st.pixelMetric(QtWidgets.QStyle.PixelMetric.PM_IndicatorWidth, None, None)
        tw = max((fm.horizontalAdvance(b.text()) for b in self.boxes), default=0.0)
        return (ind + 6 + tw + 12, len(self.boxes) * (max(fm.height(), ind) + 1) + 4)

    @property
    def labels(self):
        return list(self.spec['labels'])

    @property
    def label_texts(self):
        return self._labels

    def get_status(self):
        return self.value

    @property
    def value(self):
        return [bool(cb.isChecked()) for cb in self.boxes]

    def set(self, i, flag):
        cb = self.boxes[i]
        if cb.isChecked() != bool(flag):
            self._quiet = True
            cb.blockSignals(True)
            try:
                cb.setChecked(bool(flag))
            finally:
                cb.blockSignals(False)
                self._quiet = False

    def set_all(self, flags):
        for i, f in enumerate(flags):
            self.set(i, f)


class Radio(_Boxed):
    """
    A radio list: value is the selected label; set(label) never fires;
    callback(label).  A spec with title=True shows the widget's label
    (LABELS; the stripe window's 'drift' / 'stripe' / 'pass') as a title
    just above the box.
    """

    def __init__(self, fig, name, x, y, value=None, labels=None):
        super().__init__(fig, name, x, y, labels=labels)
        labs = self.spec['labels']
        active = labs.index(value) if value in labs else 0
        frame = self._make_frame()
        lay = QtWidgets.QVBoxLayout(frame)
        lay.setContentsMargins(4, 1, 2, 1)
        lay.setSpacing(0)
        self.group = QtWidgets.QButtonGroup(frame)
        self.group.setExclusive(True)
        self.buttons = []
        for k, lab in enumerate(labs):
            rb = QtWidgets.QRadioButton(lab)
            rb.setFont(_font(self.spec['fs']))
            rb.setChecked(k == active)
            self.group.addButton(rb, k)
            lay.addWidget(rb)
            rb.toggled.connect(lambda on, lab=lab: on and self._fire(lab))
            self.buttons.append(rb)
        self.widget = frame
        self._title = None
        if self.spec.get('title') and str(self.label).strip():
            t = self._add(QtWidgets.QLabel(self.shown(self.label)))
            t.setFont(_font(self.spec['fs']))
            t.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
            self._title = t
        self._finish()

    def _apply_labels(self):
        for rb, lab in zip(self.buttons, self.spec['labels']):
            rb.setText(self.shown(lab))
        if self._title is not None:
            self._title.setText(self.shown(self.label))

    def _layout(self, x, y, w, h):
        super()._layout(x, y, w, h)
        if self._title is not None:
            fm = QtGui.QFontMetrics(self._title.font())
            th = fm.height() + 2
            self._title.setGeometry(x + 4, y - th, max(w, fm.horizontalAdvance(self._title.text()) + 8), th)

    def _text_need(self, zoom):
        fm = QtGui.QFontMetricsF(_font(self.spec['fs'], zoom))
        st = self.buttons[0].style() if self.buttons else QtWidgets.QApplication.style()
        ind = st.pixelMetric(QtWidgets.QStyle.PixelMetric.PM_ExclusiveIndicatorWidth, None, None)
        tw = max((fm.horizontalAdvance(b.text()) for b in self.buttons), default=0.0)
        return (ind + 6 + tw + 12, len(self.buttons) * (max(fm.height(), ind) + 1) + 4)

    @property
    def labels(self):
        return list(self.spec['labels'])

    @property
    def value(self):
        k = self.group.checkedId()
        return self.spec['labels'][k] if k >= 0 else None

    value_selected = value

    def set(self, label):
        if label in self.labels and self.value != label:
            self._quiet = True
            rb = self.buttons[self.labels.index(label)]
            for b in self.buttons:
                b.blockSignals(True)
            try:
                rb.setChecked(True)
            finally:
                for b in self.buttons:
                    b.blockSignals(False)
                self._quiet = False


class PushButton(Control):
    """A button: callback() on a click; set_text changes the shown text (the name stays)."""

    def __init__(self, fig, name, x, y, label=None):
        super().__init__(fig, name, x, y, label=label)
        self._text = self.label        # the button's text as set (set_text); shown full or short
        b = QtWidgets.QPushButton(self.label)
        b.setFont(_font(self.spec['fs']))
        b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        b.setMinimumSize(1, 1)
        b.setStyleSheet("QPushButton { padding: 0px; }")
        b.clicked.connect(lambda checked=False: self._fire())
        self.widget = b
        self._main = self._add(b)
        self._finish()

    def _text_need(self, zoom):
        fm = QtGui.QFontMetricsF(_font(self.spec['fs'], zoom))
        return (fm.horizontalAdvance(self.widget.text()) + 10, fm.height() + 2)

    @property
    def value(self):
        return self.widget.text()

    def set_text(self, text):
        self._text = str(text)
        self._apply_labels()
        self._relayout()

    def _apply_labels(self):
        self.widget.setText(self.shown(self._text))


class Range(Control):
    """
    A slider: value is its number; set(v) never fires; callback(value) while
    it moves.  Its label sits left of it and its value right of it, as in
    matplotlib.  A release of the slider (or a click that moves it) is also
    passed to the figure as a button_release_event, as matplotlib's canvas
    reports it (the browser loads the chosen file on the release).
    """

    def __init__(self, fig, name, x, y, value=None, vmin=None, vmax=None, label=None):
        super().__init__(fig, name, x, y, label=label)
        s = self.spec
        self.vmin = float(s['vmin'] if vmin is None else vmin)
        self.vmax = float(s['vmax'] if vmax is None else vmax)
        self.valstep = float(s.get('valstep') or (self.vmax - self.vmin) / 1000.0 or 1.0)
        self.valfmt = s.get('valfmt') or '%g'
        self._steps = max(int(round((self.vmax - self.vmin) / self.valstep)), 1)
        sl = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        sl.setRange(0, self._steps)
        sl.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sl.setMinimumSize(1, 1)
        self.widget = sl
        self._main = self._add(sl)
        self._lab = self._add(QtWidgets.QLabel(self.label))
        self._lab.setFont(_font(s['fs']))
        self._lab.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._val = self._add(QtWidgets.QLabel(''))
        self._val.setFont(_font(s['fs']))
        self._val.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._color = '#1f77b4'
        self._apply_style()
        v0 = self.vmin if value is None else float(value)
        sl.blockSignals(True)
        sl.setValue(self._index(v0))
        sl.blockSignals(False)
        self._show_val()
        sl.valueChanged.connect(self._on_value)
        sl.sliderReleased.connect(self._released)
        self._finish()

    def _index(self, v):
        return int(np.clip(round((float(v) - self.vmin) / self.valstep), 0, self._steps))

    @property
    def val(self):
        return float(self.vmin + self.widget.value() * self.valstep)

    @property
    def value(self):
        return self.val

    def _show_val(self):
        v = self.val
        try:
            self._val.setText(self.valfmt % v)
        except TypeError:
            self._val.setText(f"{v:g}")

    def _apply_labels(self):
        self._lab.setText(self.shown(self.label))

    def _layout(self, x, y, w, h):
        self.widget.setGeometry(x, y, w, h)
        fm = QtGui.QFontMetrics(self._lab.font())
        lw = fm.horizontalAdvance(self._lab.text()) + 6
        th = max(h, fm.height())
        self._lab.setGeometry(x - lw - 2, y + (h - th) // 2, lw, th)
        vw = fm.horizontalAdvance('8' * max(len(self._val.text()), 3)) + 6
        self._val.setGeometry(x + w + 4, y + (h - th) // 2, vw, th)

    def _apply_style(self):
        c = qcolor(self._color).name()
        self.widget.setStyleSheet(
            "QSlider::groove:horizontal { height: 4px; background: #d9d9d9; border-radius: 2px; }"
            f"QSlider::sub-page:horizontal {{ background: {c}; border-radius: 2px; }}"
            f"QSlider::handle:horizontal {{ background: {c}; border: 1px solid #555555; width: 10px;"
            " margin: -4px 0; border-radius: 5px; }")

    def _on_value(self, k):
        self._show_val()
        self._fire(self.val)
        if not self.widget.isSliderDown():
            self._released()

    def _released(self):
        e = plt.MouseEvent('button_release_event', self.fig.canvas, 0, 0, button=1)
        self.fig.canvas._fire('button_release_event', e)

    def set(self, value):
        k = self._index(value)
        if self.widget.value() != k:
            self._quiet = True
            self.widget.blockSignals(True)
            try:
                self.widget.setValue(k)
            finally:
                self.widget.blockSignals(False)
                self._quiet = False
            self._show_val()

    set_val = set

    def set_color(self, color):
        """Fill the slider (its bar and knob) with a colour."""
        if color != self._color:
            self._color = color
            self._apply_style()


class Text(Control):
    """A text box: value is its text; set(text) never fires; callback(text) on submit (Enter, or leaving an edited box)."""

    def __init__(self, fig, name, x, y, value='', label=None):
        super().__init__(fig, name, x, y, label=label)
        e = QtWidgets.QLineEdit(str(value))
        e.setFont(_font(self.spec['fs']))
        e.setAlignment(Qt.AlignmentFlag.AlignCenter)
        e.setMinimumSize(1, 1)
        e.setStyleSheet("QLineEdit { padding: 0px; }")
        e.editingFinished.connect(lambda: self._fire(self.value))
        self.widget = e
        self._main = self._add(e)
        self._lab = self._add(QtWidgets.QLabel(self.label))
        self._lab.setFont(_font(self.spec['fs']))
        self._lab.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._labels = [_LabelProxy(self._lab, self._set_label)]
        if not self.label.strip():
            self._lab.hide()
        self._finish()

    @property
    def label_text(self):
        return self._labels[0]

    def _set_label(self, text):
        self.label = str(text)
        self._apply_labels()
        self._lab.setVisible(self._visible and bool(self.label.strip()))
        self._relayout()

    def _apply_labels(self):
        self._lab.setText(self.shown(self.label))

    def _layout(self, x, y, w, h):
        self.widget.setGeometry(x, y, w, h)
        fm = QtGui.QFontMetrics(self._lab.font())
        lw = fm.horizontalAdvance(self._lab.text()) + 4
        self._lab.setGeometry(x - lw - 1, y, lw, h)

    def _set_visible(self, flag):
        super()._set_visible(flag)
        if not self.label.strip():
            self._lab.hide()

    @property
    def value(self):
        return self.widget.text()

    @property
    def text(self):
        return self.widget.text()

    def set(self, text):
        if self.widget.text() != str(text):
            self._quiet = True
            self.widget.blockSignals(True)
            try:
                self.widget.setText(str(text))
            finally:
                self.widget.blockSignals(False)
                self._quiet = False

    set_val = set


_KINDS = {'check': Check, 'checks': Checks, 'radio': Radio, 'button': PushButton, 'slider': Range, 'text': Text}


def make(fig, name, x, y, **kw):
    """The widget named `name` at the figure fraction (x, y); kw as the kind's constructor takes them."""
    return _KINDS[SPECS[name]['kind']](fig, name, x, y, **kw)


def size_of(fig, name):
    """(width, height) of a named widget as figure fractions of `fig`."""
    fw, fh = fig.get_size_inches()
    s = SPECS[name]
    return s['w'] / float(fw), s['h'] / float(fh)


# ---------------------------------------------------------------------- #
#  Composite controllers                                                  #
# ---------------------------------------------------------------------- #

class FilterControls:
    """
    One framed box of Savitzky-Golay and Hampel controls inside a
    figure (`rect` = figure fraction [left, bottom, width, height]): two
    rows, Savitzky-Golay
    above Hampel, each with an unlabeled on/off toggle, the row name, and two
    short parameter boxes (window and polyorder for Savitzky-Golay, window and
    nσ for Hampel).  The two toggles are independent.

      hampel, sgolay : the parameter dicts the boxes start at
                       ({'window', 'n_sigma'}, {'window_length', 'polyorder'});
                       None uses HAMPEL_DEFAULT / SGOLAY_DEFAULT
      hampel_on, sgolay_on : initial toggle states (the spectra windows start
                       with every filter off)

    params() returns (hampel_filter_params or None, sgolay_filter_params or None)
    in the form filters.hampel / filters.sgolay take (None = off).  set_params(hampel,
    sgolay) sets the boxes and toggles from stored settings without firing the
    callback.  on_change(callback) registers a
    no-argument callback run after every toggle and every accepted edit.  An
    edit that does not parse (or breaks the filter's own rules) is reported
    on the console and the box is reset to its previous value.  set_rect(rect)
    moves the whole box (the fit window resizes with the panels it shows);
    show(flag) hides or shows it.
    """

    HAMPEL_DEFAULT = {'window': 4, 'n_sigma': 3}
    SGOLAY_DEFAULT = {'window_length': 6, 'polyorder': 3}

    # Row bottoms (fraction of the box height) and widget columns (x, width as
    # fractions of the box width); the widget height is 0.26 of the box height.
    _ROWS = {'sg': 0.57, 'hampel': 0.12}
    _COLS = {'chk': (0.03, 0.09), 'box1': (0.45, 0.12), 'box2': (0.86, 0.12)}
    FS = 7

    @classmethod
    def _widget_rect(cls, rect, col, row):
        left, bottom, width, height = rect
        x, w = cls._COLS[col]
        return [left + x * width, bottom + cls._ROWS[row] * height, w * width, 0.26 * height]

    def __init__(self, fig, rect, hampel=None, sgolay=None, hampel_on=False, sgolay_on=False):
        self.fig = fig
        self._hp = dict(self.HAMPEL_DEFAULT if hampel is None else hampel)
        self._sp = dict(self.SGOLAY_DEFAULT if sgolay is None else sgolay)
        self._on = [bool(hampel_on), bool(sgolay_on)]
        self._callback = None
        self._quiet = False
        self._rect = [float(v) for v in rect]
        self._visible = True
        parent = fig.canvas.viewport()
        fnt = _font(self.FS)

        self.frame = QtWidgets.QFrame(parent)
        self.frame.setObjectName('box')
        self.frame.setStyleSheet(_FRAME_CSS % '#f0f0f0')
        self.frame.show()

        def label(text, align=Qt.AlignmentFlag.AlignLeft):
            lab = QtWidgets.QLabel(text, parent)
            lab.setFont(fnt)
            lab.setAlignment(align | Qt.AlignmentFlag.AlignVCenter)
            lab.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            lab.show()
            return lab

        def check(state, i):
            cb = QtWidgets.QCheckBox('', parent)
            cb.setChecked(bool(state))
            cb.toggled.connect(lambda flag, i=i: self._toggle(i))
            cb.show()
            return cb

        def box(value, name):
            e = QtWidgets.QLineEdit(self._fmt(value), parent)
            e.setFont(fnt)
            e.setAlignment(Qt.AlignmentFlag.AlignCenter)
            e.setMinimumSize(1, 1)
            e.setStyleSheet("QLineEdit { padding: 0px; }")
            e.editingFinished.connect(lambda name=name, e=e: self._submit(name, e.text()))
            e.show()
            return e

        self._txt_sg_w = label('SG')
        self._txt_h_w = label('Hampel')
        self._txt_sg = _LabelProxy(self._txt_sg_w)
        self._txt_h = _LabelProxy(self._txt_h_w)
        self.chk_s = check(self._on[1], 1)
        self.box_wl = box(self._sp['window_length'], 'window_length')
        self.box_po = box(self._sp['polyorder'], 'polyorder')
        self.chk_h = check(self._on[0], 0)
        self.box_win = box(self._hp['window'], 'window')
        self.box_nsig = box(self._hp['n_sigma'], 'n_sigma')
        self._box_label_text = {'window_length': 'window', 'polyorder': 'polyorder', 'window': 'window',
                                'n_sigma': 'nσ'}
        self._box_labels = {k: label(t, Qt.AlignmentFlag.AlignRight) for k, t in self._box_label_text.items()}
        self._short = False
        self._placed = [(self.chk_s, 'chk', 'sg'), (self.box_wl, 'box1', 'sg'), (self.box_po, 'box2', 'sg'),
                        (self.chk_h, 'chk', 'hampel'), (self.box_win, 'box1', 'hampel'),
                        (self.box_nsig, 'box2', 'hampel')]
        fig._register_control(self)

    def _all(self):
        return ([self.frame, self._txt_sg_w, self._txt_h_w] + [w for w, _, _ in self._placed]
                + list(self._box_labels.values()))

    def _place(self, W, H):
        z = getattr(self.fig, 'font_zoom', 1.0)
        # The row (toggle, name, two labelled boxes) must fit the box's width: a smaller font when it
        # does not (never cut, never overlapping).
        fw = self._rect[2] * W
        fm = QtGui.QFontMetricsF(_font(self.FS, z))
        need = (28 + fm.horizontalAdvance('Hampel') + 8 + fm.horizontalAdvance('polyorder') + 8
                + fm.horizontalAdvance('window') + 8 + 2 * 0.12 * fw + 2 * fm.horizontalAdvance('88') + 24)
        f = max(0.6, min(1.0, fw / need)) if need > 0 else 1.0
        if abs(z * f - getattr(self, '_zoom', 1.0)) > 1e-6:
            self._zoom = z * f
            _refont_widgets([(qw, self.FS) for qw in self._all()], z * f)
        self.frame.lower()                         # other controls are never hidden under the box

        def px(r):
            x, y, w, h = r
            return int(round(x * W)), int(round((1.0 - y - h) * H)), max(int(round(w * W)), 1), max(int(round(h * H)), 1)
        self.frame.setGeometry(*px(self._rect))
        left, bottom, width, height = self._rect
        for (txt, fy) in ((self._txt_sg_w, 0.70), (self._txt_h_w, 0.25)):
            x, y, w, h = px([left + 0.14 * width, bottom + (fy - 0.13) * height, 0.3 * width, 0.26 * height])
            txt.setGeometry(x, y, w, h)
        names = {id(self.box_wl): 'window_length', id(self.box_po): 'polyorder', id(self.box_win): 'window',
                 id(self.box_nsig): 'n_sigma'}
        for w, col, row in self._placed:
            x, y, ww, hh = px(self._widget_rect(self._rect, col, row))
            if col == 'chk':
                s = min(ww, hh, 18)
                w.setGeometry(x, y + (hh - s) // 2, max(s, 16), s)
                continue
            w.setGeometry(x, y, ww, hh)
            lab = self._box_labels[names[id(w)]]
            fm = QtGui.QFontMetrics(lab.font())
            lw = fm.horizontalAdvance(lab.text()) + 4
            pad = int(round(0.12 * ww))
            lab.setGeometry(x - lw - pad, y, lw, hh)

    def set_short(self, flag):
        """Short labels on or off (the figure's shape, qtplot.Figure.set_portrait): the box labels with a short form switch ('polyorder' -> 'order')."""
        self._short = bool(flag)
        for k, lab in self._box_labels.items():
            t = self._box_label_text[k]
            lab.setText(short_form(t) if self._short else t)

    def set_rect(self, rect):
        """Move the box (frame and widgets) to another figure-fraction rect, e.g. after a figure resize."""
        self._rect = [float(v) for v in rect]
        W, H = self.fig._canvas_size()
        self._place(W, H)

    def show(self, flag):
        """The whole box shown or hidden (the browser hides it for files without dI/dV spectra); the settings stay."""
        self._visible = bool(flag)
        for w in self._all():
            w.setVisible(self._visible)

    @staticmethod
    def _fmt(v):
        return str(int(v)) if float(v) == int(v) else str(v)

    @staticmethod
    def _parse(name, text):
        """Value of a box entry, or a ValueError describing why it is not accepted."""
        v = float(text.strip())
        if name == 'window':
            if v != int(v) or v < 1:
                raise ValueError("Hampel 'window' must be a whole number >= 1 (points on each side).")
            return int(v)
        if name == 'n_sigma':
            if v <= 0:
                raise ValueError("Hampel 'n_sigma' must be > 0.")
            return v
        if name == 'window_length':
            if v != int(v) or v < 2:
                raise ValueError("Savitzky-Golay 'window_length' must be a whole number >= 2.")
            return int(v)
        if name == 'polyorder':
            if v != int(v) or v < 0:
                raise ValueError("Savitzky-Golay 'polyorder' must be a whole number >= 0.")
            return int(v)
        raise ValueError(name)

    def _box(self, name):
        return {'window': self.box_win, 'n_sigma': self.box_nsig,
                'window_length': self.box_wl, 'polyorder': self.box_po}[name]

    def _store(self, name):
        if name in self._hp:
            return self._hp
        return self._sp

    def _submit(self, name, text):
        if self._quiet:
            return
        store = self._store(name)
        try:
            value = self._parse(name, text)
            if name in ('window_length', 'polyorder'):
                wl = value if name == 'window_length' else self._sp['window_length']
                po = value if name == 'polyorder' else self._sp['polyorder']
                wl_eff = wl + 1 if wl % 2 == 0 else wl          # same rule as filters.sgolay
                if po >= wl_eff:
                    raise ValueError(f"Savitzky-Golay 'polyorder' ({po}) must be below the (odd) "
                                     f"window length ({wl_eff}).")
        except ValueError as exc:
            print(f"Filter controls: '{text}' not accepted for {name}: {exc}")
            self._set_box(name, store[name])
            return
        if value == store[name]:
            return
        store[name] = value
        self._set_box(name, value)
        self._fire()

    def _set_box(self, name, value):
        self._quiet = True
        try:
            b = self._box(name)
            b.blockSignals(True)
            b.setText(self._fmt(value))
            b.blockSignals(False)
        finally:
            self._quiet = False

    def _toggle(self, i):
        if self._quiet:
            return
        self._on[i] = not self._on[i]
        self._fire()

    def _set_toggle(self, i, state):
        chk = (self.chk_h, self.chk_s)[i]
        if bool(chk.isChecked()) != bool(state):
            self._quiet = True
            chk.blockSignals(True)
            try:
                chk.setChecked(bool(state))
            finally:
                chk.blockSignals(False)
                self._quiet = False
        self._on[i] = bool(state)

    def _fire(self):
        if self._callback is not None:
            self._callback()

    def on_change(self, callback):
        self._callback = callback

    def params(self):
        """(hampel_filter_params or None, sgolay_filter_params or None) as currently set."""
        return (dict(self._hp) if self._on[0] else None,
                dict(self._sp) if self._on[1] else None)

    def values(self):
        """(hampel dict, sgolay dict) of the boxes whatever the toggles say (the Line window's record)."""
        return dict(self._hp), dict(self._sp)

    def states(self):
        """(hampel_on, sgolay_on)."""
        return bool(self._on[0]), bool(self._on[1])

    def color_labels(self, hampel_color, sgolay_color=None):
        """The row names' colours (the fit windows: blue while a setting differs from the shown fit's, red once run with)."""
        self._txt_h.set_color(hampel_color)
        self._txt_sg.set_color(hampel_color if sgolay_color is None else sgolay_color)

    def set_params(self, hampel, sgolay):
        """Show stored settings: a dict sets the boxes and turns the filter on, None turns it off."""
        if hampel is not None:
            for k in ('window', 'n_sigma'):
                self._hp[k] = hampel[k]
                self._set_box(k, hampel[k])
        if sgolay is not None:
            for k in ('window_length', 'polyorder'):
                self._sp[k] = sgolay[k]
                self._set_box(k, sgolay[k])
        self._set_toggle(0, hampel is not None)
        self._set_toggle(1, sgolay is not None)


class PipelineBox:
    """
    The 'calibrate' / 'normalize' box (one box, two rows), at the same place
    in every window that has it (its top-right corner, TOP_RIGHT): the two
    pipeline steps switchable on their own, both off at open in every window
    that uses Feenstra (the callers say so).  state() gives
    {'calibrate': bool, 'normalize': bool}; set_state() sets the rows without
    firing; on_change(callback) runs callback(state) after a click.
    """

    TOP_RIGHT = (0.935, 0.93)

    def __init__(self, fig, xy=None, calibrate=True, normalize=True):
        x, y = self.TOP_RIGHT if xy is None else xy
        self.box = make(fig, 'pipeline', x, y, values=[bool(calibrate), bool(normalize)])
        self._cb = None
        self.box.on_change(lambda label, states: self._fire())

    def _fire(self):
        if self._cb is not None:
            self._cb(self.state())

    def on_change(self, callback):
        self._cb = callback

    def state(self):
        c, n = self.box.value
        return {'calibrate': bool(c), 'normalize': bool(n)}

    @property
    def calibrate(self):
        return self.state()['calibrate']

    @property
    def normalize(self):
        return self.state()['normalize']

    def set_state(self, calibrate=None, normalize=None):
        if calibrate is not None:
            self.box.set(0, bool(calibrate))
        if normalize is not None:
            self.box.set(1, bool(normalize))

    def show(self, flag):
        self.box.show(flag)


class FeenstraSettings:
    """
    The Feenstra settings boxes of a window: '0Vsub', 'broadened', and the
    width box (V), in a row at (x, y), each box the same in every window.
    sync(settings) shows a window's {'zero_sub', 'broadened', 'broadening'}
    without firing; on_change(callback) runs callback(**change) with the one
    changed setting (zero_sub=..., broadened=..., or broadening=...); a width
    that is not a positive number is reported and the box put back.
    """

    def __init__(self, fig, x, y, settings, gap=0.006):
        self.fig = fig
        self.gap = float(gap)
        fw, _ = fig.get_size_inches()
        self.chk_zero = make(fig, '0Vsub', x, y, value=settings['zero_sub'])
        x2 = x + SPECS['0Vsub']['w'] / fw + gap
        self.chk_broad = make(fig, 'broadened', x2, y, value=settings['broadened'])
        x3 = x2 + SPECS['broadened']['w'] / fw + gap
        self.box_width = make(fig, 'width', x3, y, value=f"{settings['broadening']:g}")
        self._cb = None
        self._settings = dict(settings)
        self.chk_zero.on_change(lambda flag: self._fire(zero_sub=flag))
        self.chk_broad.on_change(lambda flag: self._fire(broadened=flag))
        self.box_width.on_change(self._on_width)

    def on_change(self, callback):
        self._cb = callback

    def _fire(self, **change):
        if self._cb is not None:
            self._cb(**change)

    def _on_width(self, text):
        try:
            w = float(str(text).strip())
        except ValueError:
            w = -1.0
        if not np.isfinite(w) or w <= 0:
            print(f"Feenstra settings: '{text}' is not a positive broadening width in volts; the box is put back.")
            self.sync(self._settings)
            return
        self._fire(broadening=w)

    def sync(self, settings):
        """Show a window's settings {'zero_sub', 'broadened', 'broadening'} without firing."""
        self._settings = dict(settings)
        self.chk_zero.set(settings['zero_sub'])
        self.chk_broad.set(settings['broadened'])
        self.box_width.set(f"{settings['broadening']:g}")

    def show(self, flag):
        for c in (self.chk_zero, self.chk_broad, self.box_width):
            c.show(flag)

    def place(self, x, y):
        """Move the row's lower-left corner to the figure fraction (x, y) (the fit window relays out on resize)."""
        fw, _ = self.fig.get_size_inches()
        self.chk_zero.place(x, y)
        x2 = x + SPECS['0Vsub']['w'] / fw + self.gap
        self.chk_broad.place(x2, y)
        self.box_width.place(x2 + SPECS['broadened']['w'] / fw + self.gap, y)

    def color_labels(self, zero_color, broad_color):
        """The labels' colours ('0Vsub'; 'broadened' and the width box together): the fit windows' set-versus-run colours."""
        self.chk_zero.labels[0].set_color(zero_color)
        self.chk_broad.labels[0].set_color(broad_color)
        self.box_width.label_text.set_color(broad_color)


class StackControls:
    """
    The 'stack' and 'substack' toggles (and 'Average' when a window has
    marker groups), the same boxes in every window: stack offsets the curves
    of a panel, substack shows every member instead of one mean per group or
    file, Average averages the marker groups.  The console notes say when a
    toggle waits for another ('stack' applies with Average on, 'substack'
    with stack on), as every window says them.  on_change(callback) runs
    callback() after a click; stack / substack / average are the states.
    """

    def __init__(self, fig, stack_xy, substack_xy, average_xy=None, stack=False, substack=False,
                 average=False, needs_average=False, tag=''):
        self.chk_stack = make(fig, 'stack', *stack_xy, value=stack)
        self.chk_substack = make(fig, 'substack', *substack_xy, value=substack)
        self.chk_average = make(fig, 'Average', *average_xy, value=average) if average_xy is not None else None
        self.needs_average = bool(needs_average) and self.chk_average is not None
        self.tag = tag
        self._cb = None
        self.chk_stack.on_change(lambda flag: self._on_stack())
        self.chk_substack.on_change(lambda flag: self._on_substack())
        if self.chk_average is not None:
            self.chk_average.on_change(lambda flag: self._fire())

    def on_change(self, callback):
        self._cb = callback

    def _fire(self):
        if self._cb is not None:
            self._cb()

    @property
    def stack(self):
        return self.chk_stack.value

    @property
    def substack(self):
        return self.chk_substack.value

    @property
    def average(self):
        return self.chk_average.value if self.chk_average is not None else False

    def _prefix(self):
        return f"{self.tag}: " if self.tag else ""

    def _on_stack(self):
        if self.needs_average and not self.average:
            print(self._prefix() + "stack: applies with Average on"
                  + (" (set, shown once Average is on)." if self.stack else "."))
        self._fire()

    def _on_substack(self):
        if self.needs_average and not (self.average and self.stack):
            print(self._prefix() + "substack: applies with Average and stack on"
                  + (" (set, shown once both are on)." if self.substack else "."))
        elif not self.needs_average and not self.stack:
            print(self._prefix() + "substack applies with 'stack' on"
                  + (" (set, shown once 'stack' is on)." if self.substack else "."))
        self._fire()


class MarkerGroupControls:
    """
    The marker-group controls of the Line, Map, and line-average windows: the
    'avgs' slider (number of groups, 0 to max_groups), one 'Markers' slider
    per group (0 to 10 markers) with its 'add new' and 'add existing' buttons,
    the sliders stacked bottom to top in the order the window gives
    (refresh), each filled with its group's colour.  Group membership is by
    index: groups(sizes) gives the marker indices per group, group_of(sizes,
    i) a marker's group.  The avgs rule (grow_groups): growing keeps the
    existing groups and adds each new group, with the size of the last group,
    at the free slot of the even layout farthest from every existing centre;
    shrinking drops the last groups.
      on_avgs(cb(n)), on_markers(cb(group, m)), on_add_new(cb(group)),
      on_add_existing(cb(group, anchor_ax))
    stack_top() is the top of the visible slider stack (figure fraction), for
    the windows that raise their panels above it.
    """

    M_COLORS = ['#1f77b4', '#2ca02c', '#9467bd', '#00ced1', '#e377c2',
                '#17becf', '#bcbd22', '#7f7f7f', '#8c564b', '#d62728']

    def __init__(self, fig, x, y0, pitch, avgs_xy, max_groups=10, n_groups=1, gap=0.004):
        fw, fh = fig.get_size_inches()
        self.fig = fig
        self.x, self.y0, self.pitch = float(x), float(y0), float(pitch)
        self.max_groups = int(max_groups)
        self.rows = []
        w_slider = SPECS['Markers']['w'] / fw
        w_new = SPECS['add new']['w'] / fw
        for row in range(self.max_groups):
            y = self.y0 + row * self.pitch
            s = make(fig, 'Markers', x, y, value=1)
            b_new = make(fig, 'add new', x + w_slider + 0.028, y)          # right of the slider's value text
            b_exist = make(fig, 'add existing', x + w_slider + 0.028 + w_new + gap, y)
            for c in (s, b_new, b_exist):
                c.show(False)
            s.on_change(lambda val, row=row: self._on_marker(row, val))
            b_new.on_change(lambda row=row: self._on_new(row))
            b_exist.on_change(lambda row=row: self._on_exist(row))
            self.rows.append((s, b_new, b_exist))
        self.s_avgs = make(fig, 'avgs', *avgs_xy, value=n_groups, vmax=self.max_groups)
        self.s_avgs.on_change(lambda val: self._fire('avgs', int(val)))
        self._cbs = {}
        self._row_group = [None] * self.max_groups

    def on_avgs(self, cb):
        self._cbs['avgs'] = cb

    def on_markers(self, cb):
        self._cbs['markers'] = cb

    def on_add_new(self, cb):
        self._cbs['new'] = cb

    def on_add_existing(self, cb):
        self._cbs['exist'] = cb

    def _fire(self, key, *args):
        cb = self._cbs.get(key)
        if cb is not None:
            cb(*args)

    def _on_marker(self, row, val):
        g = self._row_group[row]
        if g is not None:
            self._fire('markers', g, int(val))

    def _on_new(self, row):
        g = self._row_group[row]
        if g is not None:
            self._fire('new', g)

    def _on_exist(self, row):
        g = self._row_group[row]
        if g is not None:
            self._fire('exist', g, self.rows[row][2].ax)

    # --- the group model shared by the windows ---

    @staticmethod
    def groups(sizes):
        """Marker indices per group: group g holds the next sizes[g] markers after the groups before it."""
        groups, start = [], 0
        for size in sizes:
            m = max(0, int(size))            # a group may hold no marker (its 'Markers' slider at 0)
            groups.append(list(range(start, start + m)))
            start += m
        return groups

    @staticmethod
    def group_of(sizes, i):
        start = 0
        for g, size in enumerate(sizes):
            start += max(0, int(size))
            if i < start:
                return g
        return max(len(sizes) - 1, 0)

    @classmethod
    def color(cls, k):
        return cls.M_COLORS[k % len(cls.M_COLORS)]

    @staticmethod
    def grow_groups(n, centers, slots, dist):
        """
        The centres of the groups added when 'avgs' grows from len(centers)
        to n: each at the free slot of `slots` (the even layout for n groups)
        farthest from every existing centre (ties: the first in slot order),
        in order.  `dist(a, b)` measures the centres.
        """
        centers = list(centers)
        added = []
        for _ in range(n - len(centers)):
            free = [s for s in slots if all(dist(s, c) > 1e-9 for c in centers)]
            if not free:
                free = list(slots)
            best = max(free, key=lambda s: (min(dist(s, c) for c in centers) if centers else 0.0,
                                            -slots.index(s)))
            centers.append(best)
            added.append(best)
        return added

    def refresh(self, order, sizes):
        """
        Show one slider per group, row 0 at the bottom: `order` lists the
        groups top to bottom (the window's ordering of its groups), `sizes`
        the marker count per group; the slider of group g is filled with
        colour g and set to its size without firing.
        """
        n = len(order)
        for row, (s, b_new, b_exist) in enumerate(self.rows):
            g = order[n - 1 - row] if row < n else None
            self._row_group[row] = g
            shown = g is not None
            for c in (s, b_new, b_exist):
                c.show(shown)
            if not shown:
                continue
            s.set_color(self.color(g))
            s.set(int(sizes[g]))
        self.s_avgs.set(len(sizes))

    def stack_top(self):
        """The top (figure fraction) of the visible slider stack."""
        n = sum(1 for g in self._row_group if g is not None)
        _, h = size_of(self.fig, 'Markers')
        return self.y0 + max(n - 1, 0) * self.pitch + h


class TopoProcessorBox:
    """
    The topo processor box ('Slope Sub', 'Δz sub', ... from topo.TOPO_PROCESSORS):
    one processor active at most.  A click on an inactive entry turns it on
    and the others off (their boxes cleared without firing); a click on the
    active entry turns it off (no processing).  active is the list of flags;
    on_change(callback) runs callback(active) after a click.
    """

    def __init__(self, fig, x, y, labels, active):
        self.box = make(fig, 'topo processors', x, y, labels=list(labels), values=list(active))
        self._cb = None
        self.box.on_change(self._on_click)

    def on_change(self, callback):
        self._cb = callback

    @property
    def active(self):
        return self.box.value

    def set_active(self, flags):
        self.box.set_all(flags)

    def _on_click(self, label, states):
        labels = self.box.labels
        if label not in labels:
            return
        i = labels.index(label)
        # The clicked box has already flipped itself; the rule (topo.toggle_processor) takes the
        # flags as they were before the click.
        before = list(states)
        before[i] = not before[i]
        new = toggle_processor(before, i)
        self.box.set_all(new)
        if self._cb is not None:
            self._cb(list(new))


class ResampleControls:
    """The 'bin' / 'interp' toggles: one of the two at most.  method is 'bin', 'interp', or None; on_change(cb(method))."""

    def __init__(self, fig, bin_xy, interp_xy, method=None):
        self.chk_bin = make(fig, 'bin', *bin_xy, value=method == 'bin')
        self.chk_interp = make(fig, 'interp', *interp_xy, value=method == 'interp')
        self._cb = None
        self.chk_bin.on_change(lambda flag: self._on('bin'))
        self.chk_interp.on_change(lambda flag: self._on('interp'))

    def on_change(self, callback):
        self._cb = callback

    @property
    def method(self):
        if self.chk_bin.value:
            return 'bin'
        if self.chk_interp.value:
            return 'interp'
        return None

    def set(self, method):
        self.chk_bin.set(method == 'bin')
        self.chk_interp.set(method == 'interp')

    def _on(self, label):
        # The clicked box has already flipped itself: on now means it was off (that method is
        # chosen, the other box cleared); off now means the method is turned off.
        clicked_on = self.chk_bin.value if label == 'bin' else self.chk_interp.value
        new = label if clicked_on else None
        self.set(new)
        if self._cb is not None:
            self._cb(new)


class RunControls:
    """A row of named buttons ('run', 'clear', 'restore', 'save', 'Send to fit', ...): on(name, cb) wires one."""

    def __init__(self, fig, placements):
        """placements: [(name, x, y), ...]."""
        self.buttons = {}
        for name, x, y in placements:
            self.buttons[name] = make(fig, name, x, y)

    def on(self, name, callback):
        self.buttons[name].on_change(callback)

    def __getitem__(self, name):
        return self.buttons[name]


class GammaBox:
    """
    The 'γ' box of a heatmap: the power-law exponent of its colour map (1 =
    linear).  A finite number > 0 is accepted; anything else is reported and
    the box put back.  on_change(callback) runs callback(gamma).  attach(ax)
    places the box just below the bottom-right corner of the heatmap's axes.
    """

    def __init__(self, fig, x=0.0, y=0.0, gamma=1.0):
        self.box = make(fig, 'γ', x, y, value=f"{gamma:g}")
        self.gamma = float(gamma)
        self._cb = None
        self.box.on_change(self._on_submit)

    def on_change(self, callback):
        self._cb = callback

    def attach(self, ax, gap_in=0.05):
        self.box.below_corner(ax, gap_in=gap_in)

    def show(self, flag):
        self.box.show(flag)

    @property
    def value(self):
        return self.gamma

    def set(self, gamma):
        self.gamma = float(gamma)
        self.box.set(f"{self.gamma:g}")

    def _on_submit(self, text):
        try:
            g = float(str(text).strip())
        except ValueError:
            g = -1.0
        if not np.isfinite(g) or g <= 0:
            print(f"γ: '{text}' is not a positive number; the box is put back to {self.gamma:g}.")
            self.box.set(f"{self.gamma:g}")
            return
        if g == self.gamma:
            return
        self.gamma = g
        if self._cb is not None:
            self._cb(g)


class NameBox:
    """The 'name' box of a window (an average's name): on_change(cb(text)); set(text) without firing."""

    def __init__(self, fig, x, y, name):
        self.box = make(fig, 'name', x, y, value=name)

    def on_change(self, callback):
        self.box.on_change(callback)

    @property
    def value(self):
        return self.box.value

    def set(self, text):
        self.box.set(text)


class StepButtons:
    """A labelled pair of step buttons (names such as '< line' / 'line >'): on_change(cb(d)) with d = -1 or +1."""

    def __init__(self, fig, left_name, right_name, x, y, gap=0.004):
        fw, _ = fig.get_size_inches()
        self.left = make(fig, left_name, x, y)
        self.right = make(fig, right_name, x + SPECS[left_name]['w'] / fw + gap, y)
        self._cb = None
        self.left.on_change(lambda: self._fire(-1))
        self.right.on_change(lambda: self._fire(+1))

    def on_change(self, callback):
        self._cb = callback

    def _fire(self, d):
        if self._cb is not None:
            self._cb(d)

    def show(self, flag):
        self.left.show(flag)
        self.right.show(flag)


class AveragePicker:
    """
    The pop-up list of the open averaging windows (the 'add existing' list),
    shared by the browser, the Line, the Map, and the line-average windows:
    one row per open window by index (name, entry count, 'holds this entry'
    when the window has the entry), opened just above the button that opened
    it (a Qt pop-up menu); a click on a row hands that window's index to the
    caller and closes the list; a click anywhere else closes it.
    """

    def __init__(self, fig, averages):
        self.fig = fig
        self.averages = averages
        self.menu = None
        self.items = []
        self.anchor = None
        self.on_pick = None

    def is_open(self):
        return self.menu is not None and self.menu.isVisible()

    def open(self, anchor_ax, entry, on_pick):
        """The list above `anchor_ax` (a control's place) for `entry`; on_pick(index) is called for the row picked."""
        self.close()
        open_avgs = self.averages.open()
        if not open_avgs:
            print("add existing: no average is open ('add new' starts one).")
            return
        self.items = sorted(open_avgs)
        self.anchor = anchor_ax
        self.on_pick = on_pick
        menu = QtWidgets.QMenu(self.fig.canvas)
        menu.setFont(qfont(8, zoom=getattr(self.fig, 'font_zoom', 1.0)))
        for i in self.items:
            win = open_avgs[i]
            text = f"{win.name}  ({len(win.entries)} entries)" + ("  holds this entry" if win.has(entry) else "")
            act = menu.addAction(text)
            act.triggered.connect(lambda checked=False, i=i: self._picked(i))
        self.menu = menu
        qw = anchor_ax._qwidget() if hasattr(anchor_ax, '_qwidget') else None
        if qw is not None:
            top_left = qw.mapToGlobal(QtCore.QPoint(0, 0))
        else:
            x0, y0, w, h = anchor_ax.get_position().bounds
            W, H = self.fig._canvas_size()
            top_left = self.fig.canvas.viewport().mapToGlobal(QtCore.QPoint(int(x0 * W), int((1 - y0 - h) * H)))
        menu.popup(top_left - QtCore.QPoint(0, menu.sizeHint().height()))

    def _picked(self, index):
        on_pick = self.on_pick
        self.close()
        if on_pick is not None:
            on_pick(index)

    def close(self):
        if self.menu is not None:
            try:
                self.menu.close()
                self.menu.deleteLater()
            except Exception:
                pass
        self.menu = None
        self.items = []
        self.anchor = None
        self.on_pick = None


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
    click raised the pair count (the window's 'Markers' slider follows).
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
        self.on_count = on_count           # callback(n) when a click raised the pair count (the window's slider)
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
            self.n_pairs += 1                          # every click adds a pair; the window's slider follows
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
    # Every widget of the specification drawn once on one figure, by name, so that sizes and
    # labels can be checked at a glance.
    import math
    names = list(SPECS)
    DESIGN = (16, 9)          # the demo window's design size (inches)
    fig = plt.figure(figsize=DESIGN)
    cols = 6
    rows = math.ceil(len(names) / cols)
    for k, name in enumerate(names):
        r, c = divmod(k, cols)
        x = 0.02 + c * 0.16
        y = 0.92 - (r + 1) * (0.86 / rows)
        kw = {}
        if name == 'topo processors':
            kw['labels'] = ['Slope Sub', 'Δz sub']
        make(fig, name, x, y, **kw)
        fig.text(x, y + SPECS[name]['h'] / 9.0 + 0.004, name, fontsize=6, ha='left', va='bottom', color='0.4')
    plt.show()
