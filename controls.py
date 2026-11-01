"""
The one set of controls of every window, the browser's included, so that a shared feature (the
calibrate box, the filter box, the step buttons, a number box, ...) works the same everywhere:
one class per kind (Check, Checks, Radio, Button, Text, Count, Slider, TextPanel) with one
interface (value, set (never fires), on_change, show, enable, set_short), and the composite
controllers several windows share (FilterControls, PipelineBox, FeenstraSettings, StackControls,
MarkerGroupControls, TopoProcessorBox, ResampleControls, RunControls, GammaBox, NameBox,
StepButtons, AveragePicker).  widgets.SPECS names every control (its kind, its font size, a text
box's width) and widgets.SHORT_LABELS its short label.

automatic layout: the one rule for every window (nothing placed by hand).  Every control is one Qt
widget, `qt`, that the window puts in its Qt layout (qtplot.Figure.set_layout for a figure window,
the browser's own layouts); a control never knows a position.  Groups of controls sit in box frames
(box), the same frame in every window.  A panel that holds only text scrolls (TextPanel); a window
never does.

Text size: a figure window's controls follow its text zoom (qtplot: the window's size relative to
its design size, reduced to fit the window, Figure._fit_controls); a control made without a figure
(fig=None, the browser) takes its window's font (browser._WindowZoom applies the same rule).  When
the window is taller than wide every label with a short form shows it (set_short).

Number boxes (Text, Count): select one or more digits and press Up or Down: the selected digits go
up or down by one in the place of the last selected digit, carrying into the digits to the left as
arithmetic does, and the same places stay selected; with nothing selected the digit just left of
the cursor steps.  Every number in a box steps on its own (a box holding "3,3" or "Vx, Vy"); a
Count stops at its limits.  Each press applies the new value at once, as Enter does.
"""

if __package__ in (None, ''):
    # Run as a file (Spyder F5): make the package importable from wherever its folder sits.
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import re
import weakref
from decimal import Decimal, InvalidOperation
import numpy as np
from sm4_Browser import qtplot as plt
from sm4_Browser.qtplot import QtCore, QtGui, QtWidgets, qfont, qcolor
from sm4_Browser import widgets
from sm4_Browser.widgets import SPECS, LABELS, short_form
from sm4_Browser.topo import toggle_processor

Qt = QtCore.Qt

# The box frame of every group of controls in every window (the light frame of the figure windows).
FRAME_CSS = "QFrame#box { background: #f0f0f0; border: 1px solid #c8c8c8; border-radius: 3px; }"
COLOR_FRAME_CSS = "QFrame#count { border: 2px solid %s; border-radius: 3px; }"


def _font(fs, zoom=1.0):
    """A control's font: its specification's points, the controls' scale, at a zoom (widgets._font)."""
    return widgets._font(fs, zoom)


def _label_of(name):
    """The label text of a control name (widgets.LABELS; the name itself without an entry), stripped."""
    lab = LABELS.get(name, name)
    return name if lab is None else str(lab).strip()


def _qw(item):
    """The Qt widget of a control, a composite (its `qt`) or a Qt widget."""
    return item.qt if hasattr(item, 'qt') else item


# ---------------------------------------------------------------------- #
#  Layout helpers: frames, rows, columns                                    #
# ---------------------------------------------------------------------- #

def box(*items, title=None, orient='h', spacing=4, margins=(4, 2, 4, 2), stretch=False):
    """
    A box frame holding `items` (controls, composites, Qt widgets or Qt layouts) in a row
    (orient='h') or a column ('v'), with a small title on top when given (automatic layout: the
    frame and its contents are placed by Qt).  A trailing stretch keeps the items together.
    """
    frame = QtWidgets.QFrame()
    frame.setObjectName('box')
    frame.setStyleSheet(FRAME_CSS)
    outer = QtWidgets.QVBoxLayout(frame)
    outer.setContentsMargins(*margins)
    outer.setSpacing(2)
    if title:
        t = QtWidgets.QLabel(title)
        t.setObjectName('box_title')
        frame._title_label = t
        outer.addWidget(t)
    inner = QtWidgets.QHBoxLayout() if orient == 'h' else QtWidgets.QVBoxLayout()
    inner.setContentsMargins(0, 0, 0, 0)
    inner.setSpacing(spacing)
    for it in items:
        if isinstance(it, QtWidgets.QLayout):
            inner.addLayout(it)
        elif it is not None:
            inner.addWidget(_qw(it))
    if stretch:
        inner.addStretch(1)
    outer.addLayout(inner)
    return frame


def row(*items, spacing=4, stretch=False, align=None):
    """A plain row (no frame) of items: a QHBoxLayout (automatic layout)."""
    lay = QtWidgets.QHBoxLayout()
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(spacing)
    for it in items:
        if it == 'stretch':
            lay.addStretch(1)
        elif isinstance(it, QtWidgets.QLayout):
            lay.addLayout(it)
        elif it is not None:
            if align is not None:
                lay.addWidget(_qw(it), 0, align)
            else:
                lay.addWidget(_qw(it))
    if stretch:
        lay.addStretch(1)
    return lay


def column(*items, spacing=4, stretch=False, align=None):
    """A plain column (no frame) of items: a QVBoxLayout (automatic layout)."""
    lay = QtWidgets.QVBoxLayout()
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(spacing)
    for it in items:
        if it == 'stretch':
            lay.addStretch(1)
        elif isinstance(it, QtWidgets.QLayout):
            lay.addLayout(it)
        elif it is not None:
            if align is not None:
                lay.addWidget(_qw(it), 0, align)
            else:
                lay.addWidget(_qw(it))
    if stretch:
        lay.addStretch(1)
    return lay


class FlowLayout(QtWidgets.QLayout):
    """
    A row of items that wraps onto more rows when the window is too narrow for one (automatic
    layout: the one rule for every window; a window never scrolls and nothing overlaps): items
    keep their preferred sizes, left to right, each row's items aligned at its bottom; with
    `push_last` the last item sits at the right end of its row (a window's right-hand group).
    Its height follows its width (heightForWidth).
    """

    def __init__(self, hspacing=6, vspacing=4, push_last=False):
        super().__init__()
        self._items = []
        self._hs = int(hspacing)
        self._vs = int(vspacing)
        self._push_last = bool(push_last)
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self._items.append(item)

    def addWidget(self, w, *args):
        super().addWidget(_qw(w))

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._arrange(QtCore.QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self):
        m = self.contentsMargins()
        w = h = 0
        for it in self._visible():
            s = it.sizeHint()
            w += s.width() + (self._hs if w else 0)
            h = max(h, s.height())
        return QtCore.QSize(w + m.left() + m.right(), h + m.top() + m.bottom())

    def minimumSize(self):
        size = QtCore.QSize()
        for it in self._visible():
            size = size.expandedTo(it.sizeHint())
        m = self.contentsMargins()
        return size + QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())

    def _visible(self):
        return [it for it in self._items if not (it.widget() is not None and it.widget().isHidden())]

    def _arrange(self, rect, apply):
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        items = self._visible()
        rows, cur, cur_w = [], [], 0
        for it in items:
            w = it.sizeHint().width()
            if cur and cur_w + self._hs + w > r.width():
                rows.append(cur)
                cur, cur_w = [], 0
            cur_w += (self._hs if cur else 0) + w
            cur.append(it)
        if cur:
            rows.append(cur)
        y = r.y()
        for k, rw in enumerate(rows):
            rh = max(it.sizeHint().height() for it in rw)
            if apply:
                x = r.x()
                last = rw[-1]
                for it in rw:
                    s = it.sizeHint()
                    if it is last and self._push_last and it is items[-1] and len(rw) > 1:
                        x = r.x() + r.width() - s.width()
                    it.setGeometry(QtCore.QRect(QtCore.QPoint(x, y + rh - s.height()), s))
                    x += s.width() + self._hs
            y += rh + (self._vs if k < len(rows) - 1 else 0)
        return y - r.y() + m.top() + m.bottom()


class UnderPanel(QtWidgets.QWidget):
    """
    Items kept centred under a panel's box (automatic layout: the one rule for every window): put
    in the same layout column as the panel's cell, its margins follow the room the panel's tick
    axes take in the cell (Axes.on_placed), so the items centre under the box itself, not under
    the cell with its axis labels.
    """

    def __init__(self, ax, *items, spacing=3):
        super().__init__()
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(spacing)
        lay.addStretch(1)
        for it in items:
            lay.addWidget(_qw(it))
        lay.addStretch(1)
        self._ax = ax
        ax.on_placed(self._follow)

    def _follow(self):
        ax = self._ax
        cell = getattr(ax, '_cell', None)
        if cell is None:
            return
        cx, cy, cw, ch = cell.rect_px()
        x0, y0, w, h = ax._px_rect()
        left = max(int(round(x0 - cx)), 0)
        right = max(int(round(cx + cw - (x0 + w))), 0)
        m = self.layout().contentsMargins()
        if (m.left(), m.right()) != (left, right):
            self.layout().setContentsMargins(left, 0, right, 0)


class RideBand(QtWidgets.QWidget):
    """
    A band of the window's layout beside a panel whose items ride along it at positions the window
    sets from its data (the Line window's crop-cursor boxes on their cursors): the band takes its
    place from the layout (automatic layout: the one rule for every window), as thick as its
    largest item, so the items never reach another control; each item centres on the canvas
    coordinate given (set_pos: x for a horizontal band, y for a vertical one), kept inside the band.
    """

    def __init__(self, orient='h'):
        super().__init__()
        self.orient = orient
        self._items = {}                  # key -> [widget, canvas coordinate or None]
        pol = QtWidgets.QSizePolicy.Policy
        if orient == 'h':
            self.setSizePolicy(pol.Expanding, pol.Fixed)
        else:
            self.setSizePolicy(pol.Fixed, pol.Expanding)

    def add(self, key, item):
        w = _qw(item)
        w.setParent(self)
        w.show()
        self._items[key] = [w, None]
        self.updateGeometry()

    def sizeHint(self):
        hints = [it[0].sizeHint() for it in self._items.values()]
        if self.orient == 'h':
            return QtCore.QSize(0, max([h.height() for h in hints] + [0]))
        return QtCore.QSize(max([h.width() for h in hints] + [0]), 0)

    def minimumSizeHint(self):
        return self.sizeHint()

    def set_pos(self, key, coord_px):
        """Centre item `key` at canvas coordinate `coord_px` along the band (x for 'h', y for 'v')."""
        self._items[key][1] = float(coord_px)
        self._apply(key)

    def _apply(self, key):
        w, c = self._items[key]
        if c is None:
            return
        s = w.sizeHint()
        vp = self.parentWidget()
        while vp is not None and vp.parentWidget() is not None and not isinstance(vp.parentWidget(), QtWidgets.QGraphicsView):
            vp = vp.parentWidget()
        origin = self.mapTo(vp, QtCore.QPoint(0, 0)) if vp is not None else QtCore.QPoint(0, 0)
        if self.orient == 'h':
            x = int(round(c - origin.x() - 0.5 * s.width()))
            x = max(0, min(x, self.width() - s.width()))
            w.setGeometry(x, 0, s.width(), self.height())
        else:
            y = int(round(c - origin.y() - 0.5 * s.height()))
            y = max(0, min(y, self.height() - s.height()))
            w.setGeometry(0, y, self.width(), s.height())

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        for k in self._items:
            self._apply(k)


def widget_of(layout):
    """A plain QWidget holding a layout (to put a row or a column where a widget is needed)."""
    w = QtWidgets.QWidget()
    layout.setContentsMargins(0, 0, 0, 0)
    w.setLayout(layout)
    return w


# ---------------------------------------------------------------------- #
#  The base                                                                #
# ---------------------------------------------------------------------- #

class _LabelProxy:
    """A control's label as matplotlib's Text (set_color, set_text, get_text, set_fontsize)."""

    def __init__(self, qwidget, setter=None):
        self._w = qwidget
        self._setter = setter
        self._color = None

    def set_color(self, c):
        col = qcolor(c).name()
        if col != self._color:
            self._color = col
            self._w.setStyleSheet(f"color: {col};")

    def set_text(self, s):
        if self._setter is not None:
            self._setter(str(s))
        else:
            self._w.setText(str(s))

    def get_text(self):
        return self._w.text()

    def set_fontsize(self, fs):
        self._w.setFont(_font(fs))


class _Anchor:
    """A control's place for the code that asks where it is (AveragePicker): its Qt widget."""

    def __init__(self, control):
        self._c = control

    def _qwidget(self):
        return self._c.qt

    def set_visible(self, flag):
        self._c.show(flag)

    def get_visible(self):
        return self._c._visible


class Control:
    """
    One control: its Qt widgets in one widget, `qt`, which the window's Qt layout places
    (automatic layout: the one rule for every window).  `fig` is the figure whose window holds it
    (its text zoom and short labels follow the figure), or None (the browser's window sets them).
    The interface every kind shares: value, set (never fires), on_change(callback), show, enable,
    set_short, remove; `widget` is the main Qt widget and unknown attributes go to it.
    """

    _all = weakref.WeakSet()      # every control made (the browser's window switches their short labels)

    def __init__(self, fig, name, label=None):
        self.fig = fig
        self.name = name
        self.spec = dict(SPECS.get(name, {'kind': None, 'w': 1.0, 'h': 0.3, 'fs': 8}))
        self.label = _label_of(name) if label is None else str(label)
        self._short = False
        self._cb = None
        self._quiet = False
        self._visible = True
        self._enabled = True
        self._zoom = None
        self.eventson = True
        self.qt = None
        self.widget = None
        Control._all.add(self)

    def _finish(self):
        if self.fig is not None:
            self.fig._register_control(self)

    # --- text size and short labels ---

    def _fs(self):
        return self.spec.get('fs', 8)

    def _set_zoom(self, z):
        """The control's text at the figure's control zoom z (a control without a figure takes its window's font)."""
        if self.fig is None or self.qt is None:
            return
        if self._zoom is not None and abs(z - self._zoom) < 1e-6:
            return
        self._zoom = z
        widgets._refont_widgets([(self.qt, self._fs())], z)
        self._after_zoom(z)

    def _after_zoom(self, z):
        return None

    def shown(self, text):
        """The text a label shows: its short form in the short mode, else as it is."""
        return short_form(text) if self._short else str(text)

    def _apply_labels(self):
        return None

    def set_short(self, flag):
        flag = bool(flag)
        if flag != self._short:
            self._short = flag
            self._apply_labels()

    # --- the shared interface ---

    def show(self, flag):
        self._visible = bool(flag)
        self.qt.setVisible(self._visible)

    def enable(self, flag):
        """The control takes events (True) or ignores them (False)."""
        self._enabled = bool(flag)
        self.qt.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, not self._enabled)

    def on_change(self, callback):
        self._cb = callback

    def _fire(self, *args):
        if self._cb is not None and not self._quiet and self.eventson:
            self._cb(*args)

    @property
    def ax(self):
        """The control's place (for AveragePicker's anchor): its Qt widget."""
        return _Anchor(self)

    def remove(self):
        if self.fig is not None:
            self.fig._unregister_control(self)
        if self.qt is not None:
            self.qt.hide()
            self.qt.deleteLater()

    def __getattr__(self, item):
        w = self.__dict__.get('widget')
        if w is None:
            raise AttributeError(item)
        return getattr(w, item)


# ---------------------------------------------------------------------- #
#  The kinds                                                               #
# ---------------------------------------------------------------------- #

class Check(Control):
    """A one-row toggle: value is its state; set(flag) never fires; callback(flag)."""

    def __init__(self, fig, name, value=False, label=None):
        super().__init__(fig, name, label)
        cb = QtWidgets.QCheckBox(self.label)
        cb.setChecked(bool(value))
        cb.toggled.connect(lambda flag: self._fire(bool(flag)))
        self.widget = self.qt = cb
        self._labels = [_LabelProxy(cb, self._set_label)]
        self._finish()

    def _set_label(self, text):
        self.label = str(text)
        self._apply_labels()

    def _apply_labels(self):
        self.widget.setText(self.shown(self.label))

    @property
    def labels(self):
        return self._labels

    @property
    def value(self):
        return bool(self.widget.isChecked())

    def get_status(self):
        return [self.value]

    def set(self, flag):
        if self.value != bool(flag):
            self._quiet = True
            blocked = self.widget.blockSignals(True)
            try:
                self.widget.setChecked(bool(flag))
            finally:
                self.widget.blockSignals(blocked)
                self._quiet = False

    def set_label_color(self, color):
        self._labels[0].set_color(color)


class Checks(Control):
    """
    A box frame of toggle rows: value is the list of states; set(i, flag) never fires;
    callback(label, states) after a click (the clicked row already flipped).  `indents` gives a
    left indent in pixels per row (the calibrate box's rows under 'calibrate').
    """

    def __init__(self, fig, name, values=None, labels=None, title=None, indents=None):
        super().__init__(fig, name)
        labs = list(labels if labels is not None else self.spec.get('labels') or [])
        self.spec['labels'] = labs
        vals = [False] * len(labs) if values is None else [bool(v) for v in values]
        self._title = title or ''
        frame = QtWidgets.QFrame()
        frame.setObjectName('box')
        frame.setStyleSheet(FRAME_CSS)
        lay = QtWidgets.QVBoxLayout(frame)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(1)
        self._title_w = None
        if self._title:
            self._title_w = QtWidgets.QLabel(self._title)
            lay.addWidget(self._title_w)
        self.boxes = []
        for k, (lab, v) in enumerate(zip(labs, vals)):
            cb = QtWidgets.QCheckBox(lab)
            cb.setChecked(v)
            cb.toggled.connect(lambda _f, lab=lab: self._fire(lab, self.value))
            ind = int(indents[k]) if indents is not None and k < len(indents) else 0
            if ind:
                r = QtWidgets.QHBoxLayout()
                r.setContentsMargins(ind, 0, 0, 0)
                r.addWidget(cb)
                r.addStretch(1)
                lay.addLayout(r)
            else:
                lay.addWidget(cb)
            self.boxes.append(cb)
        self.widget = self.qt = frame
        self._label_proxies = [_LabelProxy(cb, lambda s, i=i: self._set_label(i, s)) for i, cb in enumerate(self.boxes)]
        self._finish()

    def _set_label(self, i, text):
        self.spec['labels'][i] = str(text)
        self._apply_labels()

    def _apply_labels(self):
        for cb, lab in zip(self.boxes, self.spec['labels']):
            cb.setText(self.shown(lab))
        if self._title_w is not None:
            self._title_w.setText(self.shown(self._title))

    @property
    def labels(self):
        return list(self.spec['labels'])

    @property
    def label_texts(self):
        return self._label_proxies

    def get_status(self):
        return self.value

    @property
    def value(self):
        return [bool(cb.isChecked()) for cb in self.boxes]

    def set(self, i, flag):
        cb = self.boxes[i]
        if cb.isChecked() != bool(flag):
            self._quiet = True
            blocked = cb.blockSignals(True)
            try:
                cb.setChecked(bool(flag))
            finally:
                cb.blockSignals(blocked)
                self._quiet = False

    def set_all(self, flags):
        for i, f in enumerate(flags):
            self.set(i, f)


class Radio(Control):
    """
    A box frame of radio rows: value is the selected label; set(label) never fires;
    callback(label).  `title` (or a specification with title=True: its label) sits on top.
    """

    def __init__(self, fig, name, value=None, labels=None, title=None):
        super().__init__(fig, name)
        labs = list(labels if labels is not None else self.spec.get('labels') or [])
        self.spec['labels'] = labs
        if title is None and self.spec.get('title') and str(self.label).strip():
            title = self.label
        self._title = title or ''
        frame = QtWidgets.QFrame()
        frame.setObjectName('box')
        frame.setStyleSheet(FRAME_CSS)
        lay = QtWidgets.QVBoxLayout(frame)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(1)
        self._title_w = None
        if self._title:
            self._title_w = QtWidgets.QLabel(self._title)
            lay.addWidget(self._title_w)
        self.group = QtWidgets.QButtonGroup(frame)
        self.group.setExclusive(True)
        self.buttons = []
        active = labs.index(value) if value in labs else 0
        for k, lab in enumerate(labs):
            rb = QtWidgets.QRadioButton(lab)
            rb.setChecked(k == active)
            self.group.addButton(rb, k)
            lay.addWidget(rb)
            rb.toggled.connect(lambda on, lab=lab: on and self._fire(lab))
            self.buttons.append(rb)
        self.widget = self.qt = frame
        self._finish()

    def _apply_labels(self):
        for rb, lab in zip(self.buttons, self.spec['labels']):
            rb.setText(self.shown(lab))
        if self._title_w is not None:
            self._title_w.setText(self.shown(self._title))

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


class Button(Control):
    """A button: callback() on a click; set_text changes the shown text (the name stays)."""

    def __init__(self, fig, name, label=None):
        super().__init__(fig, name, label)
        self._text = self.label
        b = QtWidgets.QPushButton(self.label)
        b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        b.clicked.connect(lambda checked=False: self._fire())
        self.widget = self.qt = b
        self._finish()

    @property
    def value(self):
        return self.widget.text()

    def set_text(self, text):
        self._text = str(text)
        self._apply_labels()

    def _apply_labels(self):
        self.widget.setText(self.shown(self._text))


# The digit stepping of every number box -------------------------------------------------------

# A number in a box's text: a mantissa (sign, digits, point) not inside a word, or an exponent's digits with their sign.
_NUM = re.compile(r'(?<=[eE])[-+]?\d+|(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)')


def _places(tok):
    """Per character of a number token its power of ten (None for the sign and the point)."""
    body = tok.lstrip('+-')
    lead = len(tok) - len(body)
    dot = body.find('.')
    int_len = len(body) if dot < 0 else dot
    out = [None] * lead
    for i, ch in enumerate(body):
        if ch == '.':
            out.append(None)
        elif i < int_len:
            out.append(int_len - 1 - i)
        else:
            out.append(-(i - int_len))
    return out


def step_text(text, s0, s1, d, vmin=None, vmax=None, integer=False):
    """
    The text after one Up (d = +1) or Down (d = -1) press, and the new selection (a0, a1), or
    None when no digit is there.  [s0, s1) is the selection; with none (s0 == s1, the cursor at
    s0) the digit just left of the cursor steps (a point in between is skipped) and the cursor
    stays right of it.  The selected digits go up or down by one in the place of the last selected
    digit, carrying into the digits to the left as arithmetic does, and the same places stay
    selected; the number keeps its decimals; vmin / vmax clamp it, `integer` keeps it whole.
    """
    cursor = s0 == s1
    if cursor:
        k = s1 - 1
        if k >= 0 and text[k] == '.':
            k -= 1
        if k < 0 or not text[k].isdigit():
            return None
        s0, s1 = k, k + 1
    for m in _NUM.finditer(text):
        a, b = m.start(), m.end()
        if not (a <= s0 and s1 <= b and s0 < b):
            continue
        tok = m.group(0)
        pl = _places(tok)
        sel = [pl[i - a] for i in range(s0, s1) if pl[i - a] is not None]
        if not sel:
            return None
        p_lo, p_hi = min(sel), max(sel)
        try:
            v = Decimal(tok)
        except InvalidOperation:
            return None
        body = tok.lstrip('+-')
        ndec = len(body) - body.find('.') - 1 if '.' in body else 0
        nv = v + d * (Decimal(10) ** p_lo)
        if vmin is not None and nv < Decimal(str(vmin)):
            nv = Decimal(str(vmin))
        if vmax is not None and nv > Decimal(str(vmax)):
            nv = Decimal(str(vmax))
        if integer:
            nv = nv.to_integral_value()
        new = format(nv.quantize(Decimal(1).scaleb(-ndec)), 'f') if ndec else format(nv.to_integral_value(), 'f')
        if body.endswith('.') and '.' not in new:
            new += '.'
        if new.startswith('-') and Decimal(new) == 0:
            new = new[1:]
        if tok.startswith('+') and not new.startswith('-'):
            new = '+' + new
        npl = _places(new)
        idx = [i for i, p in enumerate(npl) if p is not None and p_lo <= p <= p_hi]
        if not idx:                       # the places selected are gone (10 -> 9): the units
            idx = [i for i, p in enumerate(npl) if p == 0] or [len(new) - 1]
        out = text[:a] + new + text[b:]
        if cursor:
            return out, (a + idx[-1] + 1, a + idx[-1] + 1)
        return out, (a + idx[0], a + idx[-1] + 1)
    return None


class _DigitEdit(QtWidgets.QLineEdit):
    """A line edit whose Up / Down keys step the selected digits (step_text); `owner._stepped` applies the result."""

    def __init__(self, text, owner):
        super().__init__(text)
        self._owner = owner

    def keyPressEvent(self, ev):
        key = ev.key()
        if key in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            d = 1 if key == Qt.Key.Key_Up else -1
            if self.hasSelectedText():
                s0 = self.selectionStart()
                s1 = s0 + len(self.selectedText())
            else:
                s0 = s1 = self.cursorPosition()
            r = step_text(self.text(), s0, s1, d, *self._owner._limits())
            if r is not None:
                text, (a0, a1) = r
                self.setText(text)
                if a1 > a0:
                    self.setSelection(a0, a1 - a0)
                else:
                    self.setCursorPosition(a0)
                self._owner._stepped(text)
            ev.accept()
            return
        super().keyPressEvent(ev)


class Text(Control):
    """
    A text box with its label on the left: value is its text; set(text) never fires;
    callback(text) on Enter, or on leaving the box after an edit, and at once on every Up / Down
    digit step (step_text).  The box's width is its specification's (inches), at the text zoom.
    """

    def __init__(self, fig, name, value='', label=None, width=None):
        super().__init__(fig, name, label)
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        self._lab = QtWidgets.QLabel(self.label)
        lay.addWidget(self._lab)
        if not self.label.strip():
            self._lab.hide()
        e = _DigitEdit(str(value), self)
        e.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._base_w = int(width) if width is not None else int(round(self.spec.get('w', 0.6) * plt.DPI))
        if fig is None:
            e.setMaximumWidth(self._base_w)          # the browser's window scales it (browser._WindowZoom)
        else:
            e.setFixedWidth(self._base_w)
            # A fixed width with a fixed policy: a line edit's own policy is 'expanding', which would make
            # every row or box holding this one claim the spare width of its window's layout.
            e.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, e.sizePolicy().verticalPolicy())
        e.editingFinished.connect(self._submit)
        e.returnPressed.connect(self._enter)
        lay.addWidget(e)
        self._last = str(value)
        self._entered = False
        self.widget = e
        self.qt = w
        self._labels = [_LabelProxy(self._lab, self._set_label)]
        self._finish()

    def _limits(self):
        return None, None, False

    def _after_zoom(self, z):
        self.widget.setFixedWidth(int(round(self._base_w * z)))

    def _enter(self):
        self._entered = True

    def _submit(self):
        text = self.widget.text()
        entered, self._entered = self._entered, False
        if text == self._last and not entered:
            return
        self._last = text
        self._fire(text)

    def _stepped(self, text):
        self._last = text
        self._fire(text)

    @property
    def label_text(self):
        return self._labels[0]

    def _set_label(self, text):
        self.label = str(text)
        self._apply_labels()
        self._lab.setVisible(bool(self.label.strip()))

    def _apply_labels(self):
        self._lab.setText(self.shown(self.label))

    def set_label_color(self, color):
        self._labels[0].set_color(color)

    @property
    def value(self):
        return self.widget.text()

    @property
    def text(self):
        return self.widget.text()

    def set(self, text):
        text = str(text)
        self._last = text
        if self.widget.text() != text:
            self._quiet = True
            blocked = self.widget.blockSignals(True)
            try:
                self.widget.setText(text)
            finally:
                self.widget.blockSignals(blocked)
                self._quiet = False

    set_val = set


class Count(Text):
    """
    A whole-number box with limits (the marker and group counts that were sliders): value and val
    are the number; set(n) never fires; callback(n) on Enter, on leaving the box after an edit, and
    on every Up / Down step, which stops at vmin / vmax.  An entry that is not a whole number within
    the limits is reported on the console and the box put back.  set_color(c) draws the box's frame
    in c (a marker group's colour, so the box tells which average it belongs to).
    """

    def __init__(self, fig, name, value=0, vmin=None, vmax=None, label=None, width=None):
        spec = SPECS.get(name, {})
        self.vmin = int(spec.get('vmin', 0) if vmin is None else vmin)
        self.vmax = int(spec.get('vmax', 10) if vmax is None else vmax)
        super().__init__(fig, name, str(int(round(float(value)))), label=label,
                         width=width if width is not None else 34)
        self._n = int(round(float(value)))
        self._color = None
        self.widget.setObjectName('count')

    def _limits(self):
        return self.vmin, self.vmax, True

    def _submit(self):
        text = self.widget.text()
        entered, self._entered = self._entered, False
        if text == self._last and not entered:
            return
        self._take(text)

    def _stepped(self, text):
        self._take(text)

    def _take(self, text):
        try:
            v = float(str(text).strip())
            if v != int(v) or not (self.vmin <= v <= self.vmax):
                raise ValueError
        except ValueError:
            print(f"{self.label or self.name}: '{text}' is not a whole number from {self.vmin} to {self.vmax}; "
                  f"the box is put back to {self._n}.")
            self.set(self._n)
            return
        self._n = int(v)
        self._last = str(self._n)
        if self.widget.text() != self._last:
            self.set(self._n)
        self._fire(self._n)

    @property
    def value(self):
        return self._n

    @property
    def val(self):
        return float(self._n)

    def set(self, n):
        self._n = int(round(float(n)))
        super().set(str(self._n))

    set_val = set

    def set_color(self, color):
        """The box's frame in `color` (a marker group's colour)."""
        if color != self._color:
            self._color = color
            self.widget.setStyleSheet(f"QLineEdit#count {{ border: 2px solid {qcolor(color).name()}; border-radius: 3px; }}")


class Slider(Control):
    """
    An integer slider with its label left of it and its value right of it (the browser's file
    slider, the one slider kept): callback(value) while it moves; release(callback) registers
    callback() for the end of a move (the mouse released after a drag, or at once for a keyboard /
    page-click step: the browser loads the file then).
    """

    def __init__(self, fig, name, value=0, vmin=0, vmax=1, label=None):
        super().__init__(fig, name, label)
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        self._lab = QtWidgets.QLabel(self.label)
        lay.addWidget(self._lab)
        sl = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        sl.setRange(int(vmin), int(vmax))
        sl.setValue(int(value))
        sl.setPageStep(max(1, (int(vmax) - int(vmin)) // 20))
        self.slider = sl
        self.readout = QtWidgets.QLabel(str(int(value)))
        self.readout.setMinimumWidth(36)
        lay.addWidget(sl, 1)
        lay.addWidget(self.readout)
        self._release = None
        sl.valueChanged.connect(self._changed)
        sl.sliderReleased.connect(self._released)
        self.widget = sl
        self.qt = w
        self._finish()

    def _apply_labels(self):
        self._lab.setText(self.shown(self.label))

    def _changed(self, v):
        self.readout.setText(str(v))
        if self._quiet:
            return
        self._fire(v)
        if not self.slider.isSliderDown():          # keyboard, wheel, or page click: no drag to wait for
            self._released()

    def _released(self):
        if self._release is not None and not self._quiet:
            self._release()

    def release(self, callback):
        self._release = callback

    @property
    def value(self):
        return self.slider.value()

    def set(self, value):
        v = int(round(value))
        if self.slider.value() != v:
            self._quiet = True
            blocked = self.slider.blockSignals(True)
            try:
                self.slider.setValue(v)
                self.readout.setText(str(v))
            finally:
                self.slider.blockSignals(blocked)
                self._quiet = False

    def set_range(self, vmin, vmax):
        self._quiet = True
        blocked = self.slider.blockSignals(True)
        try:
            self.slider.setRange(int(vmin), int(vmax))
            self.slider.setPageStep(max(1, (int(vmax) - int(vmin)) // 20))
        finally:
            self.slider.blockSignals(blocked)
            self._quiet = False


class TextPanel(Control):
    """
    A panel that holds only text (a window's readout or help text): it scrolls when the text needs
    more room than the layout gives it (a window never scrolls); the text can be selected and
    copied.  The interface of a matplotlib text: set_text, get_text, set_visible, get_visible,
    set_color, set_fontsize; `mono` for a monospace readout.
    """

    def __init__(self, fig, name='readout', text='', fs=7, mono=False, lines=None, fixed=False):
        super().__init__(fig, name, label='')
        self._fixed = bool(fixed)
        self.spec['fs'] = fs
        e = QtWidgets.QPlainTextEdit()
        e.setReadOnly(True)
        e.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        e.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap if mono
                          else QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        e.setStyleSheet("QPlainTextEdit { background: transparent; }")
        e.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        e.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        pol = QtWidgets.QSizePolicy.Policy
        e.setSizePolicy(pol.Expanding, pol.Expanding)
        self._mono = bool(mono)
        self._lines = lines
        self._text = ''
        self.widget = self.qt = e
        self._finish()
        self.set_text(text)

    def _set_zoom(self, z):
        if self.fig is None:
            return
        self._zoom = z
        f = _font(self._fs(), z)
        if self._mono:
            f.setFamily('monospace')
            f.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        self.widget.setFont(f)
        if self._lines:
            # `lines` lines of text tall: exactly (fixed) or at least; more text scrolls.
            fm = QtGui.QFontMetrics(f)
            h = int(fm.lineSpacing() * self._lines + 2 * self.widget.document().documentMargin() + 4)
            if self._fixed:
                self.widget.setFixedHeight(h)
            else:
                self.widget.setMinimumHeight(h)

    def sizeHint(self):
        return self.widget.sizeHint()

    def set_text(self, s):
        s = '' if s is None else str(s)
        if s != self._text:
            self._text = s
            self.widget.setPlainText(s)

    def get_text(self):
        return self._text

    def set_visible(self, flag):
        self.show(flag)

    def get_visible(self):
        return self._visible

    def set_color(self, c):
        self.widget.setStyleSheet(f"QPlainTextEdit {{ background: transparent; color: {qcolor(c).name()}; }}")

    def set_fontsize(self, fs):
        self.spec['fs'] = fs
        self._zoom = None
        if self.fig is not None:
            self._set_zoom(self.fig.control_zoom)


class Label(Control):
    """A short text beside controls (a row's name): set_text, get_text, set_visible, set_color, as a matplotlib text."""

    def __init__(self, fig, name='label', text='', fs=8, wrap=False):
        super().__init__(fig, name, label='')
        self.spec['fs'] = fs
        self._text = str(text)
        self.widget = self.qt = QtWidgets.QLabel(self._text)
        if wrap:
            # A long line wraps rather than widening the window; the text can be selected and copied.
            self.widget.setWordWrap(True)
            self.widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._finish()

    def set_text(self, s):
        self._text = '' if s is None else str(s)
        self.widget.setText(self._text)

    def get_text(self):
        return self._text

    def set_visible(self, flag):
        self.show(flag)

    def get_visible(self):
        return self._visible

    def set_color(self, c):
        self.widget.setStyleSheet(f"color: {qcolor(c).name()};")


_KINDS = {'check': Check, 'checks': Checks, 'radio': Radio, 'button': Button, 'text': Text}
_COUNTS = ('Markers', 'avgs', 'E-markers')     # the counts that were sliders: whole-number boxes now


def make(fig, name, **kw):
    """The control named `name` (its kind from widgets.SPECS; the counts that were sliders are Count boxes, 'file' the one Slider)."""
    if name in _COUNTS:
        return Count(fig, name, **kw)
    kind = SPECS[name]['kind']
    if kind == 'slider':
        return Slider(fig, name, **kw)
    return _KINDS[kind](fig, name, **kw)


# ---------------------------------------------------------------------- #
#  Composite controllers                                                   #
# ---------------------------------------------------------------------- #

class StepButtons:
    """
    A pair of step buttons (names such as '< line' / 'line >') in one row: on_change(cb(d)) with
    d = -1 for the left one, +1 for the right.  chevrons=True shows them as '‹' and '›' shapes.
    """

    CHEVRONS = ('‹', '›')

    def __init__(self, fig, left_name, right_name, chevrons=False):
        self.left = Button(fig, left_name, label=self.CHEVRONS[0] if chevrons else None)
        self.right = Button(fig, right_name, label=self.CHEVRONS[1] if chevrons else None)
        if chevrons:
            for b in (self.left, self.right):
                b.spec['fs'] = 13
                b.widget.setStyleSheet("QPushButton { font-weight: bold; padding: 0px 10px; }")
        self._cb = None
        self.left.on_change(lambda: self._fire(-1))
        self.right.on_change(lambda: self._fire(+1))
        self.qt = widget_of(row(self.left, self.right, spacing=3))

    def on_change(self, callback):
        self._cb = callback

    def _fire(self, d):
        if self._cb is not None:
            self._cb(d)

    def show(self, flag):
        self.qt.setVisible(bool(flag))


class FilterControls:
    """
    One box of Savitzky-Golay and Hampel controls: two rows, Savitzky-Golay above Hampel, each with
    two unlabeled on/off toggles under the headers 'dI/dV' and 'I(V)' (the filter on the channel's
    columns, the filter on the recorded current: the same settings, each target its own toggle),
    the row name, and two parameter boxes (window and polyorder for Savitzky-Golay, window and nσ
    for Hampel).  The four toggles are independent.  `title` heads the box when given.

      hampel, sgolay : the parameter dicts the boxes start at ({'window', 'n_sigma'},
                       {'window_length', 'polyorder'}); None uses HAMPEL_DEFAULT / SGOLAY_DEFAULT
      hampel_on, sgolay_on : the I(V) toggles at the start; hampel_didv_on, sgolay_didv_on the dI/dV ones

    params() returns (hampel or None, sgolay or None) of the I(V) toggles in the form filters.hampel /
    filters.sgolay take (None = off); params_didv() the same for the dI/dV toggles.  set_params
    sets the boxes and toggles from stored settings without firing.  on_change(callback) registers
    a no-argument callback run after every toggle and every accepted edit.  An edit that does not
    parse (or breaks the filter's own rules) is reported on the console and the box is put back.
    """

    HAMPEL_DEFAULT = {'window': 4, 'n_sigma': 3}
    SGOLAY_DEFAULT = {'window_length': 6, 'polyorder': 3}

    def __init__(self, fig, hampel=None, sgolay=None, hampel_on=False, sgolay_on=False,
                 hampel_didv_on=False, sgolay_didv_on=False, title=None):
        self.fig = fig
        self._hp = dict(self.HAMPEL_DEFAULT if hampel is None else hampel)
        self._sp = dict(self.SGOLAY_DEFAULT if sgolay is None else sgolay)
        self._on = [bool(hampel_on), bool(sgolay_on)]            # the I(V) toggles (Hampel, SG)
        self._on_d = [bool(hampel_didv_on), bool(sgolay_didv_on)]   # the dI/dV toggles (Hampel, SG)
        self._callback = None
        self._quiet = False
        self._title = title or ''
        frame = QtWidgets.QFrame()
        frame.setObjectName('box')
        frame.setStyleSheet(FRAME_CSS)
        outer = QtWidgets.QVBoxLayout(frame)
        outer.setContentsMargins(4, 2, 4, 2)
        outer.setSpacing(1)
        self._title_w = None
        if self._title:
            self._title_w = QtWidgets.QLabel(self._title)
            outer.addWidget(self._title_w)
        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(2)
        outer.addLayout(grid)
        self._hdr_d = QtWidgets.QLabel('dI/dV')
        self._hdr_i = QtWidgets.QLabel('I(V)')
        for lab in (self._hdr_d, self._hdr_i):
            lab.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        self.chk_sd = Check(fig, 'sgolay_didv', self._on_d[1], label='')
        self.chk_s = Check(fig, 'sg', self._on[1], label='')
        self.chk_hd = Check(fig, 'hampel_didv', self._on_d[0], label='')
        self.chk_h = Check(fig, 'hampel', self._on[0], label='')
        self._name_s = QtWidgets.QLabel('SG')
        self._name_h = QtWidgets.QLabel('Hampel')
        self.box_wl = Text(fig, 'window_length', self._fmt(self._sp['window_length']), label='window', width=36)
        self.box_po = Text(fig, 'polyorder', self._fmt(self._sp['polyorder']), label='polyorder', width=36)
        self.box_win = Text(fig, 'window', self._fmt(self._hp['window']), label='window', width=36)
        self.box_nsig = Text(fig, 'n_sigma', self._fmt(self._hp['n_sigma']), label='nσ', width=36)
        self._labels_w = [self._hdr_d, self._hdr_i, self._name_s, self._name_h]
        grid.addWidget(self._hdr_d, 0, 0)
        grid.addWidget(self._hdr_i, 0, 1)
        rows = [(self.chk_sd, self.chk_s, self._name_s, self.box_wl, self.box_po),
                (self.chk_hd, self.chk_h, self._name_h, self.box_win, self.box_nsig)]
        for r, items in enumerate(rows, start=1):
            for c, it in enumerate(items):
                grid.addWidget(_qw(it), r, c, Qt.AlignmentFlag.AlignHCenter if c < 2 else Qt.AlignmentFlag.AlignLeft)
        self.chk_h.on_change(lambda flag: self._toggle(0))
        self.chk_s.on_change(lambda flag: self._toggle(1))
        self.chk_hd.on_change(lambda flag: self._toggle(2))
        self.chk_sd.on_change(lambda flag: self._toggle(3))
        for name in ('window', 'n_sigma', 'window_length', 'polyorder'):
            self._box(name).on_change(lambda text, name=name: self._submit(name, text))
        self.qt = frame
        self._zoom = None
        self._short = False
        if fig is not None:
            fig._register_control(self)

    # text size and short labels (the box's own labels; its toggles and boxes are controls of their own)
    def _set_zoom(self, z):
        if self.fig is None:
            return
        self._zoom = z
        f = _font(7, z)
        for w in self._labels_w + ([self._title_w] if self._title_w is not None else []):
            w.setFont(f)

    def set_short(self, flag):
        self._short = bool(flag)
        if self._title_w is not None:
            self._title_w.setText(short_form(self._title) if flag else self._title)

    def show(self, flag):
        self.qt.setVisible(bool(flag))

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
        self._box(name).set(self._fmt(value))

    def _toggle(self, i):
        """Toggle i flipped: 0 Hampel on I(V), 1 SG on I(V), 2 Hampel on dI/dV, 3 SG on dI/dV."""
        if self._quiet:
            return
        if i < 2:
            self._on[i] = not self._on[i]
        else:
            self._on_d[i - 2] = not self._on_d[i - 2]
        self._fire()

    def _set_toggle(self, i, state):
        (self.chk_h, self.chk_s, self.chk_hd, self.chk_sd)[i].set(bool(state))
        if i < 2:
            self._on[i] = bool(state)
        else:
            self._on_d[i - 2] = bool(state)

    def _fire(self):
        if self._callback is not None:
            self._callback()

    def on_change(self, callback):
        self._callback = callback

    def params(self):
        """(hampel_filter_params or None, sgolay_filter_params or None) of the I(V) toggles as currently set."""
        return (dict(self._hp) if self._on[0] else None,
                dict(self._sp) if self._on[1] else None)

    def params_didv(self):
        """(hampel_filter_params or None, sgolay_filter_params or None) of the dI/dV toggles: the same boxes."""
        return (dict(self._hp) if self._on_d[0] else None,
                dict(self._sp) if self._on_d[1] else None)

    def values(self):
        """(hampel dict, sgolay dict) of the boxes whatever the toggles say (the Line window's record)."""
        return dict(self._hp), dict(self._sp)

    def states(self):
        """(hampel_on, sgolay_on) of the I(V) toggles."""
        return bool(self._on[0]), bool(self._on[1])

    def states_didv(self):
        """(hampel_on, sgolay_on) of the dI/dV toggles."""
        return bool(self._on_d[0]), bool(self._on_d[1])

    def color_labels(self, hampel_color, sgolay_color=None):
        """The row names' colours (the fit windows: blue while a setting differs from the shown fit's, red once run with)."""
        self._name_h.setStyleSheet(f"color: {qcolor(hampel_color).name()};")
        self._name_s.setStyleSheet(f"color: {qcolor(hampel_color if sgolay_color is None else sgolay_color).name()};")

    def set_params(self, hampel, sgolay, hampel_didv=None, sgolay_didv=None):
        """
        Show stored settings: a dict sets the boxes and turns the target's filter on, None turns it
        off (hampel, sgolay: the I(V) toggles; hampel_didv, sgolay_didv: the dI/dV toggles, the
        same boxes, so the I(V) dict sets them when both are given).
        """
        for d in (hampel_didv, hampel):
            if d is not None:
                for k in ('window', 'n_sigma'):
                    self._hp[k] = d[k]
                    self._set_box(k, d[k])
        for d in (sgolay_didv, sgolay):
            if d is not None:
                for k in ('window_length', 'polyorder'):
                    self._sp[k] = d[k]
                    self._set_box(k, d[k])
        self._set_toggle(0, hampel is not None)
        self._set_toggle(1, sgolay is not None)
        self._set_toggle(2, hampel_didv is not None)
        self._set_toggle(3, sgolay_didv is not None)


class PipelineBox:
    """
    The 'calibrate' / 'gate' / 'normalize' box: the two pipeline steps switchable on their own, and
    under 'calibrate', indented, its 'gate' row (calibration.gate_sweeps: the safety on the
    per-sweep gains and offsets, on at open) with two small boxes, 'H' holding the gate's Hampel
    'window,n_sigma' and 'SG' its Savitzky-Golay 'length,order' (calibration.GATE_HAMPEL /
    GATE_SGOLAY at open).  hs='beside' puts the two boxes left of the box on the gate row (the
    figure windows); hs='rows' puts them on a row each under the gate row inside the box (the
    browser, whose settings column stays narrow).  A box edit that does not parse, or breaks the
    gate's rules (calibration.gate_settings), is reported on the console and the box put back.  The
    gate row is ignored while 'calibrate' is off.  state() gives {'calibrate': bool, 'normalize':
    bool, 'gate': calibration.GateSettings}; set_state() sets the rows and boxes without firing;
    on_change(callback) runs callback(state) after a click or an accepted edit.
    """

    def __init__(self, fig, calibrate=True, normalize=True, gate=None, hs='beside'):
        from sm4_Browser import calibration
        self.fig = fig
        self._gate = calibration.gate_settings(gate)
        labels = list(SPECS['pipeline']['labels'])
        self.box = Checks(fig, 'pipeline', values=[bool(calibrate), bool(self._gate.on), bool(normalize)], labels=labels)
        self.box_h = Text(fig, 'gate H', self._fmt(self._gate.hampel), width=36)
        self.box_sg = Text(fig, 'gate SG', self._fmt(self._gate.sgolay), width=36)
        if hs == 'rows':
            lay = self.box.qt.layout()
            for i, t in enumerate((self.box_h, self.box_sg)):
                r = QtWidgets.QHBoxLayout()
                r.setContentsMargins(18, 0, 0, 0)
                r.addWidget(t.qt)
                r.addStretch(1)
                lay.insertLayout(2 + i, r)
            self.qt = self.box.qt
        else:
            # The two boxes left of the toggle box, on the gate row: one grid, the box frame behind
            # the three rows of column 2, so the boxes share the gate row's height exactly.
            w = QtWidgets.QWidget()
            grid = QtWidgets.QGridLayout(w)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(4)
            grid.setVerticalSpacing(0)
            frame = self.box.qt
            lay = frame.layout()
            cbs = list(self.box.boxes)
            for cb in cbs:
                lay.removeWidget(cb)                             # the frame keeps its look, the rows go in the grid
            grid.addWidget(frame, 0, 2, 3, 1)
            for r, cb in enumerate(cbs):
                cell = QtWidgets.QHBoxLayout()
                cell.setContentsMargins(6, 3 if r == 0 else 0, 6, 3 if r == len(cbs) - 1 else 0)
                cell.addWidget(cb)
                grid.addLayout(cell, r, 2)
                cb.raise_()
            grid.addWidget(self.box_h.qt, 1, 0)
            grid.addWidget(self.box_sg.qt, 1, 1)
            self.qt = w
        self._cb = None
        self._quiet = False
        self.box.on_change(lambda label, states: self._on_toggle())
        self.box_h.on_change(lambda text: self._on_edit('hampel', text))
        self.box_sg.on_change(lambda text: self._on_edit('sgolay', text))

    @staticmethod
    def _fmt(pair):
        return f"{pair[0]:g},{pair[1]:g}"

    def _fire(self):
        if self._cb is not None and not self._quiet:
            self._cb(self.state())

    def _on_toggle(self):
        c, g, n = self.box.value[:3]
        self._gate = self._gate._replace(on=bool(g))
        self._fire()

    def _on_edit(self, which, text):
        """An accepted 'H' or 'SG' edit changes the gate settings and fires; a refused one is reported and the box put back."""
        from sm4_Browser import calibration
        if self._quiet:
            return
        try:
            parts = [float(p) for p in str(text).replace(';', ',').split(',')]
            if len(parts) != 2:
                raise ValueError(f"'{text}': two numbers are needed, separated by a comma.")
            trial = self._gate._replace(**{which: (parts[0], parts[1])})
            new = calibration.gate_settings(trial)
        except ValueError as exc:
            print(f"gate '{'H' if which == 'hampel' else 'SG'}' box: {exc} The box is put back.")
            self._set_boxes()
            return
        if new == self._gate:
            self._set_boxes()
            return
        self._gate = new
        self._set_boxes()
        self._fire()

    def _set_boxes(self):
        self._quiet = True
        try:
            self.box_h.set(self._fmt(self._gate.hampel))
            self.box_sg.set(self._fmt(self._gate.sgolay))
        finally:
            self._quiet = False

    def on_change(self, callback):
        self._cb = callback

    def state(self):
        c, g, n = self.box.value[:3]
        return {'calibrate': bool(c), 'normalize': bool(n), 'gate': self._gate._replace(on=bool(g))}

    @property
    def calibrate(self):
        return self.state()['calibrate']

    @property
    def normalize(self):
        return self.state()['normalize']

    @property
    def gate(self):
        """The gate as set: calibration.GateSettings (its `on` the gate row)."""
        return self.state()['gate']

    def set_state(self, calibrate=None, normalize=None, gate=None):
        """The rows and boxes set without firing; `gate` a GateSettings, dict, bool or None (unchanged)."""
        from sm4_Browser import calibration
        if calibrate is not None:
            self.box.set(0, bool(calibrate))
        if normalize is not None:
            self.box.set(2, bool(normalize))
        if gate is not None:
            self._gate = calibration.gate_settings(gate)
            self.box.set(1, bool(self._gate.on))
            self._set_boxes()

    def show(self, flag):
        self.qt.setVisible(bool(flag))


class FeenstraSettings:
    """
    The Feenstra settings of a window in one row: '0Vsub', 'broadened', and the width box (V).
    sync(settings) shows a window's {'zero_sub', 'broadened', 'broadening'} without firing;
    on_change(callback) runs callback(**change) with the one changed setting (zero_sub=...,
    broadened=..., or broadening=...); a width that is not a positive number is reported and the
    box put back.
    """

    def __init__(self, fig, settings):
        self.fig = fig
        self.chk_zero = Check(fig, '0Vsub', settings['zero_sub'])
        self.chk_broad = Check(fig, 'broadened', settings['broadened'])
        self.box_width = Text(fig, 'width', f"{settings['broadening']:g}", label='width (V)')
        self._cb = None
        self._settings = dict(settings)
        self.chk_zero.on_change(lambda flag: self._fire(zero_sub=flag))
        self.chk_broad.on_change(lambda flag: self._fire(broadened=flag))
        self.box_width.on_change(self._on_width)
        self.qt = widget_of(row(self.chk_zero, self.chk_broad, self.box_width, spacing=6))

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
        self.qt.setVisible(bool(flag))

    def color_labels(self, zero_color, broad_color):
        """The labels' colours ('0Vsub'; 'broadened' and the width box together): the fit windows' set-versus-run colours."""
        self.chk_zero.labels[0].set_color(zero_color)
        self.chk_broad.labels[0].set_color(broad_color)
        self.box_width.label_text.set_color(broad_color)


class StackControls:
    """
    The 'stack' and 'substack' toggles (and 'Average' when a window has marker groups), the same
    in every window: stack offsets the curves of a panel, substack shows every member instead of
    one mean per group or file, Average averages the marker groups.  The console notes say when a
    toggle waits for another ('stack' applies with Average on, 'substack' with stack on), as every
    window says them.  on_change(callback) runs callback() after a click; stack / substack / average
    are the states.  The window puts chk_stack, chk_substack and chk_average in its layout.
    """

    def __init__(self, fig, stack=False, substack=False, average=None, needs_average=False, tag=''):
        self.chk_stack = Check(fig, 'stack', stack)
        self.chk_substack = Check(fig, 'substack', substack)
        self.chk_average = Check(fig, 'Average', bool(average)) if average is not None else None
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


class _MarkerBox(QtWidgets.QScrollArea):
    """
    The marker box: one row per group, top to bottom, inside a scroll area that is as tall as its
    rows until the window's cap (MarkerGroupControls.set_height_cap) and scrolls past it.
    """

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.inner = QtWidgets.QWidget()
        self.inner.setStyleSheet("background: transparent;")
        self.rows = QtWidgets.QVBoxLayout(self.inner)
        self.rows.setContentsMargins(0, 0, 0, 0)
        self.rows.setSpacing(2)
        self.setWidget(self.inner)
        self.viewport().setStyleSheet("background: transparent;")
        self.cap = None                   # callable: the largest height (px) the box may take, or None
        self.inner.installEventFilter(self)

    def eventFilter(self, obj, ev):
        # The rows' size changed (a new text size, a row shown or hidden): the box asks the layout again.
        if obj is self.inner and ev.type() == QtCore.QEvent.Type.LayoutRequest:
            self.updateGeometry()
        return False

    def _content(self):
        return self.inner.sizeHint()

    def sizeHint(self):
        c = self._content()
        h = c.height()
        cap = self.cap() if self.cap is not None else None
        if cap is not None and h > cap:
            return QtCore.QSize(c.width() + self.verticalScrollBar().sizeHint().width(), int(max(cap, 1)))
        return QtCore.QSize(c.width(), h)

    def minimumSizeHint(self):
        s = self.sizeHint()
        cap = self.cap() if self.cap is not None else None
        h = s.height() if cap is None else min(s.height(), int(max(cap, 1)))
        return QtCore.QSize(s.width(), h)


class MarkerGroupControls:
    """
    The marker-group controls of the Line, Map, and line-average windows: the 'avgs' box (number
    of groups, 0 to max_groups) and the marker box: one row per group with its 'Markers' count box
    (0 to 10 markers; its frame in the group's colour, so a row tells which average it is) and its
    'add new' and 'add existing' buttons, the rows top to bottom in the order the window gives
    (refresh).  The box is as tall as its rows until set_height_cap's limit, then it scrolls.
    Group membership is by index: groups(sizes) gives the marker indices per group, group_of(sizes,
    i) a marker's group.  The avgs rule (grow_groups): growing keeps the existing groups and adds
    each new group, with the size of the last group, at the free slot of the even layout farthest
    from every existing centre; shrinking drops the last groups.
      on_avgs(cb(n)), on_markers(cb(group, m)), on_add_new(cb(group)),
      on_add_existing(cb(group, anchor))
    The window puts `box` (the marker box) and `s_avgs` (the 'avgs' box) in its layout.
    """

    M_COLORS = ['#1f77b4', '#2ca02c', '#9467bd', '#00ced1', '#e377c2',
                '#17becf', '#bcbd22', '#7f7f7f', '#8c564b', '#d62728']

    def __init__(self, fig, max_groups=10, n_groups=1, markers_max=None):
        self.fig = fig
        self.max_groups = int(max_groups)
        self.rows = []
        self.box = _MarkerBox()
        for r in range(self.max_groups):
            s = Count(fig, 'Markers', value=1, vmax=markers_max)
            b_new = Button(fig, 'add new')
            b_exist = Button(fig, 'add existing')
            w = widget_of(row(s, b_new, b_exist, spacing=3, stretch=True))
            w.setVisible(False)
            self.box.rows.addWidget(w)
            s.on_change(lambda val, r=r: self._on_marker(r, val))
            b_new.on_change(lambda r=r: self._on_new(r))
            b_exist.on_change(lambda r=r: self._on_exist(r))
            self.rows.append((s, b_new, b_exist, w))
        self.box.rows.addStretch(1)
        self.s_avgs = Count(fig, 'avgs', value=n_groups, vmax=self.max_groups)
        self.s_avgs.on_change(lambda val: self._fire('avgs', int(val)))
        self._cbs = {}
        self._row_group = [None] * self.max_groups
        self.qt = self.box

    def set_height_cap(self, fn):
        """fn() gives the largest height (px) the marker box takes; past it the box scrolls."""
        self.box.cap = fn
        self.box.updateGeometry()

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

    def _on_marker(self, r, val):
        g = self._row_group[r]
        if g is not None:
            self._fire('markers', g, int(val))

    def _on_new(self, r):
        g = self._row_group[r]
        if g is not None:
            self._fire('new', g)

    def _on_exist(self, r):
        g = self._row_group[r]
        if g is not None:
            self._fire('exist', g, self.rows[r][2])

    # --- the group model shared by the windows ---

    @staticmethod
    def groups(sizes):
        """Marker indices per group: group g holds the next sizes[g] markers after the groups before it."""
        groups, start = [], 0
        for size in sizes:
            m = max(0, int(size))            # a group may hold no marker (its 'Markers' box at 0)
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
        The centres of the groups added when 'avgs' grows from len(centers) to n: each at the free
        slot of `slots` (the even layout for n groups) farthest from every existing centre (ties:
        the first in slot order), in order.  `dist(a, b)` measures the centres.
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
        One row per group: `order` lists the groups top to bottom (the window's ordering of its
        groups), `sizes` the marker count per group; the row of group g has its box framed in
        colour g and set to its size without firing.
        """
        n = len(order)
        for r, (s, b_new, b_exist, w) in enumerate(self.rows):
            g = order[r] if r < n else None
            self._row_group[r] = g
            w.setVisible(g is not None)
            if g is None:
                continue
            s.set_color(self.color(g))
            s.set(int(sizes[g]))
        self.s_avgs.set(len(sizes))
        self.box.updateGeometry()


class TopoProcessorBox:
    """
    The topo processor box ('Slope Sub', 'Δz sub', ... from topo.TOPO_PROCESSORS): one processor
    active at most.  A click on an inactive entry turns it on and the others off (their boxes
    cleared without firing); a click on the active entry turns it off (no processing).  active is
    the list of flags; on_change(callback) runs callback(active) after a click.
    """

    def __init__(self, fig, labels, active, title=None):
        self.box = Checks(fig, 'topo processors', values=list(active), labels=list(labels), title=title)
        self._cb = None
        self.box.on_change(self._on_click)
        self.qt = self.box.qt

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

    def show(self, flag):
        self.box.show(flag)


class ResampleControls:
    """The 'bin' / 'interp' toggles: one of the two at most.  method is 'bin', 'interp', or None; on_change(cb(method))."""

    def __init__(self, fig, method=None):
        self.chk_bin = Check(fig, 'bin', method == 'bin')
        self.chk_interp = Check(fig, 'interp', method == 'interp')
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
    """Named buttons ('run', 'clear', 'restore', 'save', 'Send to fit', ...): on(name, cb) wires one; the window places buttons[name]."""

    def __init__(self, fig, names):
        self.buttons = {name: Button(fig, name) for name in names}

    def on(self, name, callback):
        self.buttons[name].on_change(callback)

    def __getitem__(self, name):
        return self.buttons[name]


class GammaBox:
    """
    The 'γ' box of a heatmap: the power-law exponent of its colour map (1 = linear).  A finite
    number > 0 is accepted; anything else is reported and the box put back.  on_change(callback)
    runs callback(gamma).  The window puts `qt` under its heatmap in its layout.
    """

    def __init__(self, fig, gamma=1.0):
        self.box = Text(fig, 'γ', f"{gamma:g}")
        self.gamma = float(gamma)
        self._cb = None
        self.box.on_change(self._on_submit)
        self.qt = self.box.qt

    def on_change(self, callback):
        self._cb = callback

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

    def __init__(self, fig, name):
        self.box = Text(fig, 'name', name)
        self.qt = self.box.qt

    def on_change(self, callback):
        self.box.on_change(callback)

    @property
    def value(self):
        return self.box.value

    def set(self, text):
        self.box.set(text)


class AveragePicker:
    """
    The pop-up list of the open averaging windows (the 'add existing' list), the same in every
    window: one row per open window by index (name, entry count, 'holds this entry' when the window
    has the entry), opened just above the control that opened it (a Qt pop-up menu); a click on a
    row hands that window's index to the caller and closes the list; a click anywhere else closes it.
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

    def open(self, anchor, entry, on_pick):
        """The list above `anchor` (a control, its place, or a Qt widget) for `entry`; on_pick(index) is called for the row picked."""
        self.close()
        open_avgs = self.averages.open()
        if not open_avgs:
            print("add existing: no average is open ('add new' starts one).")
            return
        self.items = sorted(open_avgs)
        self.anchor = anchor
        self.on_pick = on_pick
        if hasattr(anchor, '_qwidget'):
            qw = anchor._qwidget()
        elif hasattr(anchor, 'qt'):
            qw = anchor.qt
        else:
            qw = anchor
        menu = QtWidgets.QMenu(qw)
        if self.fig is not None:
            menu.setFont(qfont(8, zoom=getattr(self.fig, 'font_zoom', 1.0)))
        for i in self.items:
            win = open_avgs[i]
            text = f"{win.name}  ({len(win.entries)} entries)" + ("  holds this entry" if win.has(entry) else "")
            act = menu.addAction(text)
            act.triggered.connect(lambda checked=False, i=i: self._picked(i))
        self.menu = menu
        top_left = qw.mapToGlobal(QtCore.QPoint(0, 0))
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
