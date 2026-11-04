"""
Feenstra's normalized conductance (dI/dV)/(I/V): the division of a
calibrated dI/dV column by the same column's broadened I/V (divide_by_iv,
with iv_zero_offset, iv_corrected, iv_broadened), the settings helpers
(feenstra_width, feenstra_titles, energy_has_zero), the zero-crossing
diagnostic of the current (iv_zero_crossings), and the Feenstra companion
window (FeenstraWindow) of the Line, Map, averaging, and line-average
windows.

The current the division works on is the recorded I(V) through the Hampel
and Savitzky-Golay filters, with its value at 0 V subtracted when '0Vsub'
is on, then broadened.  Where the filtering happens: the pipeline
(sts.STS.process_arrays) filters the current once, before the calibration,
and calls divide_by_iv with the filters off, so the calibration and the
division use the same filtered current; the Feenstra windows' current and
I/V panels, and the __main__ demonstrations, call iv_corrected /
iv_broadened with the windows' filter settings and filter here.  The
dI/dV columns have their own filter toggles in the pipeline.  The method follows Feenstra's
normalization (the code names no citation; see the README).
"""

if __package__ in (None, ''):
    # Run as a file (Spyder F5): make the package importable from wherever its folder sits.
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np
from sm4_Browser import qtplot as plt
from sm4_Browser import widgets, plots, filters, controls
from sm4_Browser.qtplot import QtWidgets, QtCore

Qt = QtCore.Qt


# ---------------------------------------------------------------------- #
#  The division                                                           #
# ---------------------------------------------------------------------- #

def iv_zero_offset(energy, iv):
    """
    Per column, the current at 0 V: the 0 V sample when the axis holds
    one, otherwise the linear interpolation between the two samples that
    bracket 0 V.  None when 0 V lies outside the axis (no extrapolation:
    such a sweep has nothing to subtract).  Units of `iv`.
    """
    energy = np.asarray(energy, dtype=np.float64)
    iv = np.asarray(iv, dtype=np.float64)
    order = np.argsort(energy)
    e = energy[order]
    if len(e) == 0 or e[0] > 0.0 or e[-1] < 0.0:
        return None
    return np.array([np.interp(0.0, e, iv[order, j]) for j in range(iv.shape[1])], dtype=np.float64)


def iv_corrected(energy, iv, zero_sub=False, hampel=None, sgolay=None):
    """
    The current the Feenstra division works on: `iv` through the Hampel
    and Savitzky-Golay filters when their parameter dicts are given
    (filters.filter_lia, the two in that order; the pipeline,
    sts.STS.process_arrays, filters the current once before the
    calibration and passes it here with the filters off, so the same
    filtered current reaches the calibration and the division; the Feenstra
    windows' current panels filter here), then, with zero_sub, each column
    minus its own value at 0 V (iv_zero_offset), a vertical shift only.
    Unchanged when 0 V lies outside the axis.  Original energy order, units
    of `iv`.
    """
    iv = filters.filter_lia(np.asarray(iv, dtype=np.float64), hampel, sgolay)
    if not zero_sub:
        return iv
    off = iv_zero_offset(energy, iv)
    if off is None:
        return iv
    return iv - off[None, :]


def iv_broadened(energy, iv, broadening=0.0, zero_sub=False, hampel=None, sgolay=None):
    """
    The denominator of divide_by_iv: each column's broadened I/V at every
    energy, in the original energy order (the units of `iv` per volt: A/V
    for the loader's current in A).  broadening > 0 gives the
    exponentially weighted mean of I/V described there; broadening <= 0
    the plain ratio with the 0 V sample interpolated between its
    neighbours.  The current is first filtered (hampel, sgolay: the
    windows' filter settings, None = off) and, with zero_sub, has its value
    at 0 V subtracted before I/V is formed (iv_corrected).  Used by the
    division and shown by the Feenstra windows.
    """
    if iv is None:
        raise ValueError("Feenstra needs the 'Current' channel, which this file does not have.")
    energy = np.asarray(energy, dtype=np.float64)
    iv = iv_corrected(energy, iv, zero_sub, hampel, sgolay)
    order = np.argsort(energy)
    e = energy[order]
    nz = e != 0.0
    ratio = np.zeros((len(e), iv.shape[1]))
    ratio[nz, :] = iv[order, :][nz, :] / e[nz][:, None]
    if broadening > 0:
        W = np.exp(-np.abs(e[:, None] - e[None, :]) / float(broadening))   # W[i, k]
        W[:, ~nz] = 0.0                                                      # no source at 0 V
        denom = (W @ ratio) / (W @ nz.astype(np.float64))[:, None]
    else:
        denom = ratio
        if not nz.all():
            for j in range(iv.shape[1]):
                denom[~nz, j] = np.interp(e[~nz], e[nz], ratio[nz, j])
    out = np.empty_like(denom)
    out[order, :] = denom
    return out


def divide_by_iv(energy, curves, iv, broadening=0.0, zero_sub=False, hampel=None, sgolay=None):
    """
    Normalized conductance (Feenstra): each column of the calibrated dI/dV
    divided point by point by the same column's broadened I/V, a
    dimensionless ratio.  I(V) is taken as recorded, through the Hampel
    and Savitzky-Golay filters when their settings are given (hampel,
    sgolay; the dI/dV columns have their own filters in the pipeline) and, with zero_sub, with
    its own value at 0 V subtracted first (iv_corrected, a vertical shift
    of the current only; the bias axis is never touched), V =
    energy in volts, and the broadened denominator at V_i is the
    exponentially weighted mean of I/V over the measured points of that
    column,
        (sum_k (I_k / V_k) exp(-|V_k - V_i| / broadening))
        / (sum_k exp(-|V_k - V_i| / broadening)),
    which keeps the denominator finite where the current is small; a
    sample exactly at 0 V (I/V = 0/0) carries no weight as a source and
    still receives a denominator from its neighbours.  broadening <= 0
    gives the plain ratio, with the 0 V sample's I/V linearly interpolated
    between its two neighbours.  Every index and energy value is kept.
    Columns of curves and iv are the same spectra (same page order).  The
    division is defined per spectrum: the numerator and the denominator
    come from the same junction, which is what lets their dependence on
    the tip-sample spacing cancel; averages are taken of the ratios, never
    of a ratio of averages.
    """
    if iv is None:
        raise ValueError("Feenstra needs the 'Current' channel, which this file does not have.")
    curves = np.asarray(curves, dtype=np.float64)
    order = np.argsort(np.asarray(energy, dtype=np.float64))
    denom = iv_broadened(energy, iv, broadening, zero_sub, hampel, sgolay)[order, :]
    out = np.empty_like(curves)
    out[order, :] = curves[order, :] / denom
    return out


# ---------------------------------------------------------------------- #
#  Settings helpers                                                        #
# ---------------------------------------------------------------------- #

def feenstra_width(broadening, broadened):
    """The broadening width the division uses: the width in volts with 'broadened' on, 0 (the plain I/V) off."""
    return float(broadening) if broadened else 0.0


def feenstra_titles(zero_sub, broadened, broadening, has_zero=True):
    """
    The panel titles of the Feenstra settings: 'iv' (the current the division
    works on), 'ivv' (its I/V), and 'ratio' (what the ratio was made with).
    has_zero False: the data's axis does not reach 0 V, so '0Vsub' has
    nothing to subtract and the titles say so.
    """
    how = f"broadened ({float(broadening):g} V)" if broadened else "plain"
    if zero_sub and not has_zero:
        note = " (no 0 V in range: nothing subtracted)"
        return {'iv': "I(V)" + note, 'ivv': f"I/V, {how}" + note, 'ratio': f"(dI/dV)/(I/V), I/V {how}" + note}
    sub = ", I(0 V) subtracted" if zero_sub else ""
    return {'iv': "I(V) - I(0 V)" if zero_sub else "I(V)",
            'ivv': f"I/V, {how}{sub}",
            'ratio': f"(dI/dV)/(I/V), I/V {how}{sub}"}


def energy_has_zero(energy):
    """True when 0 V lies on or inside the energy axis (iv_zero_offset has something to work on)."""
    e = np.asarray(energy, dtype=np.float64)
    return bool(len(e)) and float(e.min()) <= 0.0 <= float(e.max())


def iv_zero_crossings(energy, iv, acquired):
    """
    The zero crossing of the recorded current (the loader's data, in A) of
    every acquired sweep: per sweep
    (V0 in volts, the bias where the current crosses zero, the sign change
    nearest 0 V, linearly interpolated; the current at the nominal 0 V
    sample in pA; the slope near 0 V in pA/V over the two neighbours either
    side), as an (n_acquired, 3) array (NaN where a sweep never crosses),
    and the energy of the nominal 0 V sample.  A diagnostic of the bias
    offset that '0Vsub' removes (checks/zero_crossing.py runs it over the
    saved averages).
    """
    e = np.asarray(energy, dtype=np.float64)
    o = np.argsort(e)
    es = e[o]
    cur = np.asarray(iv, dtype=np.float64)[o][:, np.asarray(acquired, dtype=bool)] * 1e12
    k0 = int(np.argmin(np.abs(es)))
    out = []
    for j in range(cur.shape[1]):
        y = cur[:, j]
        s = np.sign(y)
        k = np.flatnonzero(s[:-1] * s[1:] < 0)            # sign changes anywhere
        if len(k) == 0:
            out.append((np.nan, y[k0], np.nan))
            continue
        kk = k[np.argmin(np.abs(es[k] - 0.0))]            # the one nearest 0 V
        v0 = es[kk] - y[kk] * (es[kk + 1] - es[kk]) / (y[kk + 1] - y[kk])
        lo, hi = max(k0 - 2, 0), min(k0 + 2, len(es) - 1)
        sl = (y[hi] - y[lo]) / (es[hi] - es[lo]) if hi > lo else np.nan
        out.append((v0, y[k0], sl))
    return np.array(out, dtype=np.float64).reshape(-1, 3), float(es[k0])


# ---------------------------------------------------------------------- #
#  The companion window                                                    #
# ---------------------------------------------------------------------- #

class FeenstraWindow:
    """
    The Feenstra companion of a window with its own 'Feenstra' toggle (the
    Line, the Map, an averaging window, a line-average window): four
    versions of the window's spectra side by side, the dI/dV as the main
    window shows it with the toggle off, the current in amps as the division
    works on it, its I/V in A/V as the division divides by it, and the
    Feenstra ratio.  It opens with its main window whatever the toggle's
    state (the line-average window opens it after its first run), follows
    it live (refresh is called by the main window after every redraw), has
    no marker controls (the markers are drawn as the main window has them),
    its own 'stack' and 'substack' toggles for the curve panels, and closes
    with the main window's figure.

    Its '0Vsub' box (under the current panel), 'broadened' box and width box
    (under the I/V panel) are the main window's Feenstra settings
    (controls.FeenstraSettings): a change goes to the main window
    (set_feenstra_settings), which recomputes its own Feenstra curves with
    it and refreshes this window; the boxes always show the main window's
    state (feenstra_settings).  '0Vsub' subtracts each sweep's current at
    0 V before I/V is formed (iv_corrected: a vertical shift of the current
    only, nothing moves on the bias axis; a sweep whose axis does not reach
    0 V is left as it is); 'broadened' off divides by the plain I/V.

    Its pipeline box (controls.PipelineBox, top right, as in every window:
    'denoise', 'calibrate', 'gate' with its boxes, '0-lock', 'Vbias lock',
    'normalize') holds this window's own state of those rows, independent
    of the main window's (calibrate and normalize off at open, the gate on,
    the locks and denoise off): the four panels are the main window's
    view for the main window's state with these rows overriding it
    (refresh: sts.PipelineState.with_).  'normalize' turns the per-location
    unit-area normalization of all four panels on or off together (the
    dI/dV and the ratio through the pipeline's unit-area step, the current
    and the I/V through normalization.normalize_abs); 'calibrate' turns the
    calibration of the dI/dV off, so the dI/dV panel shows the lock-in
    output as recorded (in its page's units) and the ratio panel that
    output divided by the I/V; the current and I/V panels do not change
    with it (the current is the calibration's reference); 'denoise' divides
    the per-point gap factor out of the dI/dV page and the current, so the
    current panel is I/g and the I/V panel its I/V, and the ratio panel is
    the dI/dV panel divided by the I/V panel under every state with
    'normalize' off; the gate and the locks act on the calibration as in the
    main window.  A change of any row recomputes (the main window memoizes
    its views by the state).  Every heatmap ('line' and 'map' kinds) has
    its own 'γ' box under its bottom-right corner.

    The main window's feenstra_view(state) describes what to draw, for the
    state this window asks for: the main window's own PipelineState with
    this window's box overriding its calibrate, normalize and gate fields
    (the gate carries the locks and the denoise row) and nothing else, so
    every panel is a field of sts.STS.view for that state:
      kind     : 'line' (four heatmaps with the topo stripe and the marker
                 curve panels under them), 'map' (four maps at the current
                 energy with the marker curve panels under them), or 'sweeps'
                 (four sweep panels in one row, the averaging window)
      title    : the figure title
      energy   : the ascending energy axis of the curve panels
      versions : dict with keys 'didv', 'iv', 'ivv', 'ratio'; each a dict with
                 'title', 'ylabel', optional 'note' (the panel shows the note
                 and nothing else), and for 'line' / 'map' 'top' (the 2D
                 array), 'markers' and 'groups' (plots.draw_marker_curves
                 items), for 'sweeps' 'entries' ([{'label', 'color', 'V',
                 'sweeps' (npts, n)}]) and 'average' ((V0, curve) or None)
      texts    : the texts of plots.draw_marker_curves ('line' / 'map')
      avg_on   : the main window's Average state ('line' / 'map')
      extent   : imshow extent of the top panels ('map')
      pos, kept, shear, stripe, stripe_pos : 'line': the position axis, the
                 kept rows (the ones drawn, at pos[kept]: plots.RowHeatmap),
                 their within-sweep shears with the main window's drift
                 toggle on (else None), the stripe profile (never moved by
                 the drift; the rows' drift lines drawn on it), and the
                 positions of its cells (the pixels' own recorded locations)
      shift    : 'map': with the main window's drift toggle on, the pixel
                 shifts (n_lines, n_fast, 2; plots.PixelMap), else None (a
                 map shows the drift by its pixels' positions, never by
                 drift lines)
      marker_lines : 'line': [(position, colour)] and 'centers': [position]
      marker_points: 'map': [((x, y), colour)] and 'centers': [(x, y)];
                 marker_diameter: a pixel's diagonal (the markers' size,
                 plots.draw_map_marker)
      cmap     : the heatmap / map colour map
    """

    KEYS = ('didv', 'iv', 'ivv', 'ratio')
    SIMPLE = {'didv': 'dI/dV', 'iv': 'I(V)', 'ivv': 'I/V', 'ratio': '(dI/dV)/(I/V)'}   # the panel titles
    STACK_OFFSET_FRAC = 0.25

    DESIGN = (18, 8)       # the design size (inches): the text zoom of the window is its size relative to this (qtplot.window_zoom)

    def __init__(self, parent, title):
        self.parent = parent
        self.title = title
        self.fig = plt.figure(figsize=self.DESIGN)
        try:
            self.fig.canvas.manager.set_window_title(f"Feenstra: {title}")
        except Exception:
            pass
        self.axes = []                     # built at the first refresh (the kind decides the layout)
        self.kind = None
        self.cols = []
        self._topo_cbar = None
        # This window's own pipeline rows (top right, as in every window; calibrate and normalize off at
        # open, the gate on, the locks and denoise off): they override the main window's state (refresh).
        self.pipeline = controls.PipelineBox(self.fig, calibrate=False, normalize=False)
        self.pipeline.on_change(lambda state: self.refresh())
        # The main window's Feenstra settings: '0Vsub', 'broadened' and the width box, under the
        # current panel's column.
        self.settings = controls.FeenstraSettings(self.fig, self._settings())
        self.settings.on_change(self._apply)
        self.stacks = controls.StackControls(self.fig, needs_average=False, tag='Feenstra window')
        self.stacks.on_change(self.refresh)
        # One gamma box per heatmap column (shown for the 'line' and 'map' kinds).
        self.gammas = {key: controls.GammaBox(self.fig) for key in self.KEYS}
        for g in self.gammas.values():
            g.on_change(lambda value: self.refresh())
            g.show(False)
        # automatic layout: the one rule for every window (nothing placed by hand).  The pipeline box at
        # the top right; under it the panels' grid, rebuilt for each view kind (_build_axes), its last
        # row holding the settings under the current column and 'stack' / 'substack' under the last.
        self._grid = QtWidgets.QGridLayout()
        self._grid.setHorizontalSpacing(6)
        self._grid.setVerticalSpacing(2)
        self._stack_row = controls.widget_of(controls.row(self.stacks.chk_stack, self.stacks.chk_substack, spacing=8))
        root = QtWidgets.QVBoxLayout()
        title_cell = self.fig.cell(min_px=(40, 24), stretch=(True, False))   # the window's title, centred in it
        self.fig.set_title_cell(title_cell)
        root.addLayout(controls.row(title_cell, self.pipeline))
        root.addLayout(self._grid, 1)
        self.fig.set_layout(root)
        pfig = getattr(parent, '_int_fig', None) or getattr(parent, 'fig', None)
        if pfig is not None:
            pfig.canvas.mpl_connect('close_event', lambda event: self.close())
        self.refresh()
        plt.show(block=False)

    def is_open(self):
        """True while this window's figure is open (widgets.figure_is_open: the Figure object itself, not its number)."""
        return widgets.figure_is_open(self.fig)

    def close(self):
        """Close the figure when it is open (the main window's close_event calls this)."""
        if self.is_open():
            plt.close(self.fig)

    # --- the main window's Feenstra settings ---

    def _settings(self):
        """The main window's settings {'zero_sub', 'broadened', 'broadening'}."""
        return self.parent.feenstra_settings()

    def _apply(self, **changes):
        """Hand a settings change to the main window; it recomputes and refreshes this window, or refuses (boxes put back)."""
        if not self.parent.set_feenstra_settings(**changes):
            self.settings.sync(self._settings())
            self.fig.canvas.draw_idle()

    @property
    def stack_on(self):
        """The state of this window's own 'stack' toggle (the curve panels offset vertically)."""
        return self.stacks.stack

    @property
    def substack_on(self):
        """The state of this window's own 'substack' toggle (every member instead of one mean per group or file)."""
        return self.stacks.substack

    @staticmethod
    def _wrap(title, width=36):
        """A panel title on two lines when it is long: broken at the comma or space nearest the middle."""
        title = str(title)
        if len(title) <= width:
            return title
        mid = len(title) // 2
        commas = [i for i, ch in enumerate(title) if ch == ',' and 8 < i < len(title) - 8]
        depth, spaces = 0, []
        for i, ch in enumerate(title):               # spaces outside parentheses only
            depth += (ch == '(') - (ch == ')')
            if ch == ' ' and depth == 0 and 8 < i < len(title) - 8:
                spaces.append(i)
        cuts = commas or spaces
        if not cuts:
            return title
        k = min(cuts, key=lambda i: abs(i - mid))
        return title[:k + 1].rstrip() + "\n" + title[k + 1:].lstrip()

    def _build_axes(self, kind):
        """
        The axes of the four columns for a view kind, replacing the previous
        ones, each in a cell of the window's layout grid (automatic layout:
        the one rule for every window), the old cells removed: 'sweeps' one
        panel per column (the gamma boxes hidden); 'line' and 'map' a top
        panel (the heatmap or map) with its colour bar and a curve panel under
        it, and for 'line' a topo stripe left of the top panel sharing its y
        axis, plus one topo colour bar at the right; the gamma boxes under the
        top panels; the settings under the current column, 'stack' /
        'substack' under the last.  Sets self.cols (one dict of axes per
        column) and self.kind.
        """
        for ax in self.axes:
            try:
                ax.remove()
            except Exception:
                pass
        if self._topo_cbar is not None:
            try:
                self._topo_cbar.remove()
            except Exception:
                pass
            self._topo_cbar = None
        self.axes = []
        fig = self.fig
        grid = self._grid
        # The grid emptied: the old cells go (their axes are gone above), the controls are put back below.
        while grid.count():
            it = grid.takeAt(0)
            w = it.widget()
            if isinstance(w, plt.PanelCell):
                w.setParent(None)
                w.deleteLater()
        for c in range(grid.columnCount()):
            grid.setColumnStretch(c, 0)
        for r in range(grid.rowCount()):
            grid.setRowStretch(r, 0)
        cols = []
        for c, key in enumerate(self.KEYS):
            base = 3 * c                      # the column's grid columns: stripe, top panel, colour bar
            if kind == 'sweeps':
                cell = fig.cell()
                grid.addWidget(cell, 0, base, 3, 3)
                ax = fig.add_axes(cell=cell)
                cols.append({'panel': ax})
                self.axes.append(ax)
                self.gammas[key].show(False)
                grid.setColumnStretch(base + 1, 1)
                continue
            d = {}
            if kind == 'line':
                cs = fig.cell(inner_w=12)
                grid.addWidget(cs, 0, base)
                d['stripe'] = fig.add_axes(cell=cs)
                self.axes.append(d['stripe'])
            ct, cc, cu = fig.cell(), fig.cell(inner_w=8), fig.cell()
            grid.addWidget(ct, 0, base + 1)
            grid.addWidget(cc, 0, base + 2)
            grid.addWidget(cu, 2, base, 1, 3)
            d['top'] = fig.add_axes(cell=ct, sharey=d.get('stripe'))
            d['cbar'] = fig.add_axes(cell=cc)
            d['curves'] = fig.add_axes(cell=cu)
            self.axes.extend([d['top'], d['cbar'], d['curves']])
            cols.append(d)
            grid.addWidget(self.gammas[key].qt, 1, base + 1, Qt.AlignmentFlag.AlignRight)   # under its heatmap
            self.gammas[key].show(True)
            grid.setColumnStretch(base + 1, 1)
        if kind == 'line':
            tc = fig.cell(inner_w=7)
            grid.addWidget(tc, 0, 12)
            self._topo_cbar = plots.draw_topo_cbar(fig, cell=tc)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(2, 1)
        grid.addWidget(self.settings.qt, 3, 3, 1, 3, Qt.AlignmentFlag.AlignLeft)      # under the current column
        grid.addWidget(self._stack_row, 3, 9, 1, 3, Qt.AlignmentFlag.AlignRight)      # under the last column
        # The top row across the columns (the maps or heatmaps, their colour bars, the stripes and
        # the topo ramp): one height, a map's aspect setting it (qtplot Figure.align_row).
        top_row = [a for d in cols for a in (d.get('stripe'), d.get('top'), d.get('cbar'), d.get('panel')) if a is not None]
        if self._topo_cbar is not None:
            top_row.append(self._topo_cbar)
        self.fig.align_row(*top_row)
        curves_row = [d['curves'] for d in cols if 'curves' in d]
        if curves_row:
            self.fig.align_row(*curves_row)              # the curve panels: one row too
        self.cols = cols
        self.kind = kind

    @staticmethod
    def _sig(a):
        """A cheap fingerprint of an array (shape and sums), None for None."""
        if a is None:
            return None
        a = np.asarray(a, dtype=np.float64)
        with np.errstate(invalid='ignore', over='ignore'):
            return (a.shape, float(np.nansum(a)), float(np.nansum(a * a)), int(np.isnan(a).sum()))

    def _simple(self, key, v):
        """The panel's title: what it shows, nothing more (the rest goes to the info panel)."""
        if key == 'didv' and 'IETS' in str(v.get('title', '')):
            return 'IETS'
        return self.SIMPLE[key]

    def _fingerprint(self, view, v, key):
        """
        What the top panels of one column depend on (not the markers): the
        kind, note and title, the data array's fingerprint, the gamma, the
        drift (shear, shift), the stripe and positions, the
        extent, the colour map, the energy range and the limits.  Equal
        fingerprints on consecutive refreshes skip the rebuild of the
        heatmap / map (refresh).
        """
        return (view['kind'], v.get('note'), v.get('title'), self._sig(v.get('top')), self.gammas[key].value,
                self._sig(view.get('shear')), self._sig(view.get('shift')), self._sig(view.get('stripe')),
                self._sig(view.get('pos')), self._sig(view.get('kept')), tuple(view.get('extent') or ()),
                str(view.get('cmap')), tuple(view.get('e_range') or ()),
                tuple(view.get('limits') or ()))

    def _draw_top(self, view, v, key, col, kind):
        """The heatmap (line) or map of one column with its colour bar (and the stripe)."""
        top, cax = col['top'], col['cbar']
        img = np.asarray(v['top'], dtype=np.float64)
        if kind == 'line':
            # The kept spectra's rows at their positions, sheared as the main window's are.
            pos = np.asarray(view['pos'], dtype=np.float64)
            kept = np.asarray(view.get('kept', np.ones(len(pos), dtype=bool)), dtype=bool)
            # The main window's 'E range' box (view['e_range']): the colour scale from the cells
            # inside it, and the energy limits at it.
            rng = view.get('e_range')
            lo, hi = plots.energy_range_scale(img[kept], view['energy'], rng)
            im = plots.RowHeatmap(top, img[kept], view['energy'], pos[kept], view.get('cmap', 'jet'),
                                  gamma=self.gammas[key].value, shear=view.get('shear'), vmin=lo, vmax=hi)
            x0, x1, y0, y1 = im.extent()
            if rng is not None:
                x0, x1 = float(rng[0]), float(rng[1])
            top.set_xlim(x0, x1)
            top.set_ylim(y0, y1)
        elif view.get('shift') is not None:
            im = plots.PixelMap(top, img, view['extent'], view.get('cmap', 'jet'), gamma=self.gammas[key].value,
                                shift=view['shift'])
        else:
            im = plots.draw_heatmap(top, img, view['extent'], view.get('cmap', 'jet'), gamma=self.gammas[key].value,
                                    aspect='equal', origin='lower', interpolation='none')
        plots.draw_colorbar(self.fig, im, cax)
        col['im'] = im                                   # the drawn image (its array is what the panel shows)
        top.set_title(f"{self._simple(key, v)}: " + v.get('title', ''), fontsize=8)
        if kind == 'line':
            plots.draw_topo_stripe(col['stripe'], view.get('stripe_pos', view['pos']), view['stripe'])
            sh = view.get('shear')
            if sh is not None and len(sh):
                plots.draw_stripe_drift_lines(col['stripe'], pos[kept], np.asarray(sh))
            col['stripe'].set_ylabel("Position (Å)" if key == self.KEYS[0] else "")
            if key != self.KEYS[0]:
                col['stripe'].tick_params(axis='y', labelleft=False)   # the first column carries the positions
            top.tick_params(labelleft=False)             # the stripe carries the position labels
            top.set_xlabel("Energy (eV)")
        else:
            l, r, b, t = view.get('limits') or view['extent']      # the Map's widened bounds with a drift source on
            top.set_xlim(min(l, r), max(l, r))
            top.set_ylim(min(b, t), max(b, t))
            top.set_aspect('equal', adjustable='box')
            top.set_xlabel("x (nm, scan frame)")
            top.set_ylabel("y (nm, scan frame)")

    def refresh(self):
        """
        Redraw from the main window's current view, feenstra_view(state) for
        the main window's state with this window's own pipeline rows
        (calibrate, normalize, the gate with its locks and denoise)
        overriding it (nothing when this window is closed; an exception from
        the main window is printed and ends the
        refresh): the axes rebuilt when the kind changed, the settings boxes
        synced to the main window's, then per column the sweep panel
        ('sweeps'), or the heatmap / map rebuilt only when its fingerprint
        changed, the marker overlays replaced, and the marker curve panel
        redrawn (plots.draw_marker_curves); a column with a note shows the
        note alone.  The figure title is 'Feenstra normalization: ' and the
        view's title.
        """
        if not self.is_open():
            return
        flags = self.pipeline.state()
        try:
            state = self.parent.state.with_(calibrate=flags['calibrate'], normalize=flags['normalize'], gate=flags['gate'])
            view = self.parent.feenstra_view(state)
        except Exception as exc:
            print(f"Feenstra window: {exc}")
            return
        if view is None:
            return
        kind = view['kind']
        if kind != self.kind:
            self._build_axes(kind)
        self.settings.sync(self._settings())
        for col, key in zip(self.cols, self.KEYS):
            v = view['versions'].get(key) or {}
            note = v.get('note')
            if kind == 'sweeps':
                ax = col['panel']
                ax.clear()
                if note:
                    ax.text(0.5, 0.5, note, ha='center', va='center', transform=ax.transAxes, wrap=True)
                else:
                    plots.draw_sweeps(ax, v.get('entries'), v.get('average'), self.stack_on, self.substack_on,
                                      self.STACK_OFFSET_FRAC)
                ax.set_title(f"{self._simple(key, v)}: " + v.get('title', ''), fontsize=9)
                ax.set_xlabel("Energy (eV)")
                ax.set_ylabel(v.get('ylabel', ''))
                continue
            top, cax, axc = col['top'], col['cbar'], col['curves']
            # The heatmap / map, its colour bar, and the stripe are rebuilt only when what they show
            # changed (the data, the gamma, the drift, the layout); a marker dragged in the main window
            # only moves the marker overlays and redraws the curve panels.
            fp = self._fingerprint(view, v, key)
            if fp != col.get('_fp'):
                col['_fp'] = fp
                col['marks'] = []
                top.clear()
                cax.clear()
                if 'stripe' in col:
                    col['stripe'].clear()
                if note:
                    axc.clear()
                    top.text(0.5, 0.5, note, ha='center', va='center', transform=top.transAxes, wrap=True)
                    top.set_xticks([])
                    top.set_yticks([])
                    top.set_title(f"{self._simple(key, v)}: " + v.get('title', ''), fontsize=8)
                    continue
                self._draw_top(view, v, key, col, kind)
            elif note:
                continue
            for art in col.get('marks', []):
                try:
                    art.remove()
                except Exception:
                    pass
            marks = []
            if kind == 'line':
                # On the heatmaps the marker lines follow their rows' shear as in the main window
                # ('marker_paths' / 'center_paths' over 'path_x'); on the stripe they are level.
                px = view.get('path_x')
                paths = view.get('marker_paths') if px is not None else None
                cpaths = view.get('center_paths') if px is not None else None
                for i, (pos_val, color) in enumerate(view['marker_lines']):
                    sh = paths[i] if paths is not None and i < len(paths) else None
                    if sh is not None:
                        marks.extend(top.plot(px, float(pos_val) + sh, color=color, ls='--', lw=2, alpha=0.7))
                    else:
                        marks.append(top.axhline(y=pos_val, color=color, ls='--', lw=2, alpha=0.7))
                    marks.append(col['stripe'].axhline(y=pos_val, color=color, ls='--', lw=2, alpha=0.7))
                if view.get('avg_on'):
                    for g, c in enumerate(view['centers']):
                        sh = cpaths[g] if cpaths is not None and g < len(cpaths) else None
                        if sh is not None:
                            marks.extend(top.plot(px, float(c) + sh, color='black', ls='-', lw=2.5, alpha=0.9))
                        else:
                            marks.append(top.axhline(y=c, color='black', ls='-', lw=2.5, alpha=0.9))
                        marks.append(col['stripe'].axhline(y=c, color='black', ls='-', lw=2.5, alpha=0.9))
            else:
                for (x, y), color in view['marker_points']:
                    marks.append(plots.draw_map_marker(top, (x, y), color, view['marker_diameter']))
                if view.get('avg_on'):
                    for (x, y) in view['centers']:
                        marks.extend(top.plot([x], [y], marker='X', ms=10, color='black', ls='none'))
            col['marks'] = marks
            texts = dict(view['texts'])
            texts['ylabel'] = v.get('ylabel', texts.get('ylabel', ''))
            texts['ylabel_offset'] = texts['ylabel'] + " (offset)"
            plots.draw_marker_curves(axc, view['energy'], v['markers'], v['groups'], view.get('avg_on', False),
                                     self.stack_on, self.substack_on, texts, legend_fontsize=6.5, show_title=False)
            if kind == 'line' and view.get('e_range') is not None:
                axc.set_xlim(float(view['e_range'][0]), float(view['e_range'][1]))   # the main window's 'E range'
            axc.title.set_fontsize(7)
        self.fig.suptitle("Feenstra normalization: " + view.get('title', self.title), fontsize=10, y=0.985)
        self.fig.canvas.draw_idle()


if __name__ == '__main__':
    # The Feenstra division on one file: the ratio's range per sweep with and without the
    # broadening and the 0 V subtraction (nothing is drawn).
    from sm4_Browser import sm4file, calibration
    path = r'experiment_dir/file.sm4'
    f = sm4file.Sm4File(path)
    cols = np.flatnonzero(f.acquired)
    ref = None if f.lia_ref is None or f.iv_ref is None else (f.lia_ref[:, cols], f.iv_ref[:, cols])
    cal, _ = calibration.calibrate(f.energy, f.lia[:, cols], f.iv[:, cols], f.vmod, V_bias=f.V_bias,
                                   I_setpoint=f.I_setpoint, ref=ref)
    for broadening, zero_sub in ((0.0, True), (0.2, True), (0.0, False)):
        r = divide_by_iv(f.energy, cal, f.iv[:, cols], broadening, zero_sub)
        t = feenstra_titles(zero_sub, broadening > 0, broadening, energy_has_zero(f.energy))['ratio']
        print(f"{t}: ratio from {np.nanmin(r):.4g} to {np.nanmax(r):.4g}")
    cr, e0 = iv_zero_crossings(f.energy, f.iv, f.acquired)
    print(f"zero crossings (mV) of the current: {np.array2string(cr[:, 0] * 1e3, precision=1)}; nominal 0 V sample at {e0 * 1e3:+.3f} mV")
