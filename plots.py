"""
Every drawing of data into axes (qtplot: matplotlib's interface on
pyqtgraph), and nothing else: no widget is built or read here.  The windows own their data and state, draw through
these functions, and take their events from widgets.py.

  heat_norm, draw_heatmap, update_heatmap, draw_colorbar : STS heatmaps and
      maps with the colour-map gamma (a PowerNorm for gamma != 1)
  RowHeatmap, row_edges : a heatmap whose rows sit at their own positions
      (the Line window: the kept spectra at their recorded positions), each
      row's cells sheared along the position axis by the drift during its
      sweep
  PixelMap : a map whose pixels sit at their own shifted positions (the Map
      window with drift correction on)
  draw_topo_stripe, draw_marker_curves, draw_sweeps : the Line / Map / Feenstra
      panels
  TOPO_CMAP, draw_topo_absolute, draw_topo_frame, draw_point_sweeps,
      draw_grid_stripes, draw_drift_lines, draw_topo_cbar : topography drawing
  draw_section_line, draw_drift_markers : the browser's cross-section line
      and the manual-drift markers on the scan area
  draw_profiles : the raw forward/reverse topo profiles and cross sections
      (the profile window) as curves of height against position
  draw_profile_pairs : several forward/reverse pairs in one axes (the
      profile plot): one colour per pair, forward solid, reverse dashed
"""

if __package__ in (None, ''):
    # Run as a file (Spyder F5): make the package importable from wherever its folder sits.
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np
from sm4_Browser import qtplot as plt
from sm4_Browser import qtplot as mcolors           # Normalize, PowerNorm, TwoSlopeNorm, LinearSegmentedColormap
from sm4_Browser.qtplot import ScalarMappable, PolyCollection, LineCollection, Affine2D, QuadItem, to_cmap

# The drift lines (every sweep's start to its end): one colour on every page that shows them
# for the auto drift (the tracker), another for the manual drift (markers or typed), and a
# third for a span confirmed from seeded markers (the auto velocity, confirmed by hand), so
# the three kinds are told apart on sight.
DRIFT_LINE_COLOR = '#ffffff'
DRIFT_LINE_EDGE = 'black'
MANUAL_DRIFT_COLOR = '#ff2a2a'
CONFIRMED_DRIFT_COLOR = '#ffd84a'
# The manual-drift markers: pair k in its own colour (both markers and the arrow), the same
# in the browser and the drift window; red, green, blue first, then yellow, magenta, cyan,
# orange, purple: the eight chromatic basic colour terms (Boynton and Olson) that read apart
# on the topo colour map, so the 'Markers' sliders run 1 to len(MARKER_COLORS)
# (widgets.MARKER_PAIRS_MAX).  The earlier marker of a pair is a circle, the later a square.
MARKER_COLORS = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff', '#ff8000', '#8000ff']
SECTION_LINE_COLOR = '#ff00ff'        # the browser's cross-section line (draggable endpoints)
PARTNER_FRAME_COLOR = 'black'         # the drift window: the other topo's recorded frame on each panel
# The profile window: the forward and reverse passes of a topo line, and the cross section's two passes.
PROFILE_FORWARD_COLOR = '#1f77b4'
PROFILE_REVERSE_COLOR = '#d62728'


def drift_color(source):
    """
    The drift-line colour of a file's velocity source (Drift result
    'source'): manual sources take MANUAL_DRIFT_COLOR, 'auto (confirmed)'
    CONFIRMED_DRIFT_COLOR, the tracker's DRIFT_LINE_COLOR.
    """
    source = source or ''
    if source.startswith('manual'):
        return MANUAL_DRIFT_COLOR
    if source == 'auto (confirmed)':
        return CONFIRMED_DRIFT_COLOR
    return DRIFT_LINE_COLOR


# The Line window's energy markers (E-markers, vertical lines at an energy): their own colours, apart from
# the position markers' group colours (widgets.MarkerGroupControls.M_COLORS) and readable on the jet heatmap.
E_MARKER_COLORS = ['#000000', '#ff00ff', '#ff7f0e', '#ffffff', '#8b4513', '#ffd700', '#808000', '#4b0082']


def e_marker_color(k):
    return E_MARKER_COLORS[k % len(E_MARKER_COLORS)]


def marker_color(index):
    """The colour of marker pair `index` (from 1): MARKER_COLORS in turn."""
    return MARKER_COLORS[(int(index) - 1) % len(MARKER_COLORS)]

# The topography colour map (the scan area, the Topo window, the embedded topos, the stripes, the topo colour bars).
TOPO_CMAP = mcolors.LinearSegmentedColormap.from_list("custom_topo", ["black", "firebrick", "yellow"])
# Single-point markers on the scan area, chosen to stand out from the topo ramp
# (black, firebrick, yellow) and from each other (the loaded-point colour is the browser's).
POINT_CURRENT_COLOR = '#39ff14'     # the current single point (its first sweep): lime cross
POINT_SWEEP_COLOR = '#ff8c00'       # the later sweeps of the current point with drift correction on: orange dots


# ---------------------------------------------------------------------- #
#  STS heatmaps and maps                                                   #
# ---------------------------------------------------------------------- #

def heat_norm(gamma, vmin, vmax):
    """
    The colour normalization of an STS heatmap: a PowerNorm with exponent
    gamma over [vmin, vmax] (((x - vmin) / (vmax - vmin)) ** gamma), a
    plain Normalize for gamma 1, so a gamma of 1 draws exactly as before.
    vmin is the data minimum everywhere, so no value falls below it.
    """
    vmin, vmax = float(vmin), float(vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    if abs(float(gamma) - 1.0) < 1e-12:
        return mcolors.Normalize(vmin=vmin, vmax=vmax)
    return mcolors.PowerNorm(float(gamma), vmin=vmin, vmax=vmax)


def finite_range(img):
    """(min, max) of the finite values of an array, (0, 1) when it has none."""
    a = np.asarray(img, dtype=np.float64)
    ok = np.isfinite(a)
    if not ok.any():
        return 0.0, 1.0
    return float(np.min(a[ok])), float(np.max(a[ok]))


def energy_range_scale(img, energy, e_range):
    """
    (vmin, vmax) of a row heatmap's colour scale over an energy range: the
    finite_range of the cells whose energy (the columns' axis `energy`)
    lies inside e_range (lo, hi), the Line window's 'E range' box; (None,
    None) without a range or without a column inside it (the heatmap then
    scales over every cell).
    """
    if e_range is None:
        return None, None
    e = np.asarray(energy, dtype=np.float64)
    cols = (e >= float(e_range[0])) & (e <= float(e_range[1]))
    a = np.asarray(img, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != len(e) or not cols.any():
        return None, None
    return finite_range(a[:, cols])


def draw_heatmap(ax, img, extent, cmap, gamma=1.0, vmin=None, vmax=None, origin='upper', aspect='auto',
                 interpolation=None):
    """
    An STS heatmap or map into `ax` (imshow): its own min and max as the
    colour limits unless vmin / vmax are given, the colour-map gamma applied.
    Returns the AxesImage.
    """
    lo, hi = finite_range(img)
    vmin = lo if vmin is None else float(vmin)
    vmax = hi if vmax is None else float(vmax)
    kw = {'origin': origin, 'aspect': aspect, 'cmap': cmap, 'norm': heat_norm(gamma, vmin, vmax)}
    if extent is not None:
        kw['extent'] = extent
    if interpolation is not None:
        kw['interpolation'] = interpolation
    return ax.imshow(img, **kw)


def update_heatmap(im, img=None, gamma=1.0, vmin=None, vmax=None):
    """New data and/or limits and gamma for an existing image or RowHeatmap (its own min and max unless given)."""
    if img is not None:
        im.set_data(img)
    data = im.get_array()
    lo, hi = finite_range(data)
    vmin = lo if vmin is None else float(vmin)
    vmax = hi if vmax is None else float(vmax)
    im.set_norm(heat_norm(gamma, vmin, vmax))


def row_edges(rows):
    """
    The cell edges (n + 1) of rows at positions `rows` (n, monotonic): the
    midpoints between neighbours, the outer edges half a neighbour spacing
    beyond the ends (a single row gets edges half a unit either side).
    """
    r = np.asarray(rows, dtype=np.float64)
    if len(r) == 1:
        return np.array([r[0] - 0.5, r[0] + 0.5])
    mid = 0.5 * (r[1:] + r[:-1])
    return np.concatenate([[r[0] - (mid[0] - r[0])], mid, [r[-1] + (r[-1] - mid[-1])]])


class RowHeatmap:
    """
    A heatmap whose rows sit at their own positions along the vertical axis:
    img (n_rows, n_e) over an ascending energy axis `energy` (n_e), row r at
    position rows[r] (its cell from the midpoint to its lower neighbour to
    the midpoint to its upper, row_edges), and, with `shear` (n_rows, n_e +
    1), every row's n_e + 1 cell edges displaced along the position axis by
    its own shear (the drift during its sweep: the row's cells slide
    relative to one another, nothing moves along the energy axis).  All
    cells are one quad item (qtplot.QuadItem: one filled path per colour,
    cached as a pixmap, so markers dragged over it never refill it).  The
    interface of an AxesImage as the windows use it: set_data, get_array,
    set_norm, norm, cmap, remove; `mappable` is the ScalarMappable for the
    colour bar (draw_colorbar takes the object itself).  NaN cells are not
    drawn.
    """

    def __init__(self, ax, img, energy, rows, cmap, gamma=1.0, shear=None, vmin=None, vmax=None):
        self.ax = ax
        self.energy = np.asarray(energy, dtype=np.float64)
        self.rows = np.asarray(rows, dtype=np.float64)
        self.img = np.asarray(img, dtype=np.float64)
        self.x_edges = row_edges(self.energy)
        self.y_edges = row_edges(self.rows)
        self.shear = None if shear is None else np.asarray(shear, dtype=np.float64)
        lo, hi = finite_range(self.img)
        self.norm = heat_norm(gamma, lo if vmin is None else vmin, hi if vmax is None else vmax)
        self.cmap = to_cmap(cmap)
        self.mappable = ScalarMappable(norm=self.norm, cmap=self.cmap)
        self.mappable.set_array(self.img[np.isfinite(self.img)] if np.isfinite(self.img).any() else np.zeros(1))
        self.item = None
        self._quads = None
        self._build()

    def _geometry(self):
        n, n_e = self.img.shape
        S = np.zeros((n, n_e + 1))
        if self.shear is not None:
            m = min(n, len(self.shear))
            S[:m] = np.where(np.isfinite(self.shear[:m]), self.shear[:m], 0.0)
        xl, xr = self.x_edges[:-1][None, :], self.x_edges[1:][None, :]
        yb, yt = self.y_edges[:-1][:, None], self.y_edges[1:][:, None]
        sl, sr = S[:, :-1], S[:, 1:]
        X = np.broadcast_to
        quads = np.stack([np.stack([X(xl, (n, n_e)), yb + sl], -1), np.stack([X(xr, (n, n_e)), yb + sr], -1),
                          np.stack([X(xr, (n, n_e)), yt + sr], -1), np.stack([X(xl, (n, n_e)), yt + sl], -1)],
                         axis=2)
        self._quads = quads.reshape(-1, 4, 2)

    def _colors(self):
        v = self.img.ravel()
        c = self.cmap(np.asarray(self.norm(v), dtype=np.float64), bytes=True)
        c[~np.isfinite(v)] = 0
        return c

    def _build(self):
        if self.img.ndim != 2 or self.img.size == 0:
            self.remove()
            return
        self._geometry()
        if self.item is None:
            self.item = QuadItem()
            self.item.setZValue(1.0)
            self.ax.vb.addItem(self.item)
        self.item.setData(self._quads, self._colors())

    def set_data(self, img):
        img = np.asarray(img, dtype=np.float64)
        if img.shape != self.img.shape:
            self.img = img
            self._build()
        else:
            self.img = img
            if self.item is not None:
                self.item.setData(self._quads, self._colors())
        self.mappable.set_array(self.img[np.isfinite(self.img)] if np.isfinite(self.img).any() else np.zeros(1))

    def get_array(self):
        return self.img

    def set_norm(self, norm):
        self.norm = norm
        if self.item is not None:
            self.item.setData(self._quads, self._colors())
        self.mappable.set_norm(norm)

    def set_shear(self, shear):
        """New shears (or None): the cells are rebuilt."""
        self.shear = None if shear is None else np.asarray(shear, dtype=np.float64)
        self._build()

    def extent(self):
        """(x0, x1, y0, y1): the outer edges, the shear included."""
        y0, y1 = float(np.min(self.y_edges)), float(np.max(self.y_edges))
        if self.shear is not None and np.isfinite(self.shear).any():
            y0 = min(y0, float(np.nanmin(self.y_edges[:-1, None] + self.shear)))
            y1 = max(y1, float(np.nanmax(self.y_edges[1:, None] + self.shear)))
        return float(self.x_edges[0]), float(self.x_edges[-1]), y0, y1

    def remove(self):
        if self.item is not None:
            try:
                self.ax.vb.removeItem(self.item)
            except Exception:
                pass
            self.item = None


class PixelMap:
    """
    A map whose pixels sit at their own positions: img (n_lines, n_fast)
    over the frame `extent` [l, r, b, t], pixel (i, j) a quad at its nominal
    cell displaced by shift[i, j] (n_lines, n_fast, 2; the frame's units;
    zero where None), drawn as one quad item (qtplot.QuadItem).  The
    interface the windows use: set_data, get_array, set_norm, norm, cmap,
    remove, and `mappable` for the colour bar.  NaN pixels are not drawn.
    """

    def __init__(self, ax, img, extent, cmap, gamma=1.0, shift=None, vmin=None, vmax=None, zorder=1.0):
        self.ax = ax
        self.img = np.asarray(img, dtype=np.float64)
        self.extent = [float(v) for v in extent]
        self.shift = None if shift is None else np.asarray(shift, dtype=np.float64)
        self.zorder = zorder
        lo, hi = finite_range(self.img)
        self.norm = heat_norm(gamma, lo if vmin is None else vmin, hi if vmax is None else vmax)
        self.cmap = to_cmap(cmap)
        self.mappable = ScalarMappable(norm=self.norm, cmap=self.cmap)
        self.item = None
        self._build()

    def _quads(self):
        n_lines, n_fast = self.img.shape
        l, r, b, t = self.extent
        dx, dy = (r - l) / n_fast, (t - b) / n_lines
        jj, ii = np.meshgrid(np.arange(n_fast), np.arange(n_lines))
        x0 = l + jj * dx
        y0 = b + ii * dy
        quads = np.stack([np.stack([x0, y0], axis=-1), np.stack([x0 + dx, y0], axis=-1),
                          np.stack([x0 + dx, y0 + dy], axis=-1), np.stack([x0, y0 + dy], axis=-1)], axis=2)
        if self.shift is not None and self.shift.shape[:2] == self.img.shape:
            s = np.where(np.isfinite(self.shift), self.shift, 0.0)
            quads = quads + s[:, :, None, :]
        return quads.reshape(-1, 4, 2)

    def _colors(self):
        v = self.img.ravel()
        c = self.cmap(np.asarray(self.norm(v), dtype=np.float64), bytes=True)
        c[~np.isfinite(v)] = 0
        return c

    def _build(self):
        if self.item is None:
            self.item = QuadItem()
            self.item.setZValue(self.zorder)
            self.ax.vb.addItem(self.item)
        self._q = self._quads()
        self.item.setData(self._q, self._colors())
        ok = np.isfinite(self.img).ravel()
        vals = self.img.ravel()[ok]
        self.mappable.set_array(vals if len(vals) else np.zeros(1))

    def set_data(self, img):
        self.img = np.asarray(img, dtype=np.float64)
        self._build()

    def get_array(self):
        return self.img

    def set_norm(self, norm):
        self.norm = norm
        if self.item is not None:
            self.item.setData(self._q, self._colors())
        self.mappable.set_norm(norm)

    def set_shift(self, shift):
        self.shift = None if shift is None else np.asarray(shift, dtype=np.float64)
        self._build()

    def remove(self):
        if self.item is not None:
            try:
                self.ax.vb.removeItem(self.item)
            except Exception:
                pass
            self.item = None


def draw_colorbar(fig, im, cax, label=None, labelsize=None):
    """
    The colour bar of an image (or a RowHeatmap / PixelMap, through its
    mappable) into `cax`.  A colour bar already in `cax` is updated in place
    (the colour map, the norm's ticks, the label) instead of being rebuilt.
    """
    m = getattr(im, 'mappable', im)
    cb = getattr(cax, '_colorbar', None)
    if cb is not None and cb.ax is cax and cax in fig.axes:
        m.colorbar = cb
        cb.update_normal(m)
        if label is not None:
            cb.set_label(label)
    else:
        cax.clear()
        cb = fig.colorbar(m, cax=cax)
        if label is not None:
            cb.set_label(label)
    if labelsize is not None:
        cax.tick_params(labelsize=labelsize)
    return cb


def draw_topo_stripe(ax, pos, profile):
    """
    The topo stripe of the Line window into `ax`: the profile as one column
    of cells along the position axis, each cell at its sample's position
    (row_edges of `pos` when pos has one position per sample; cells spread
    evenly from pos[0] to pos[-1] otherwise, the line-average windows), the
    two-slope colour scale between the profile's finite minimum and maximum
    (a flat profile has no range: no norm), NaN samples (pixels not kept)
    not drawn, no x ticks, the title 'Topo'.  Shared with the Feenstra
    companion and the line-average windows.
    """
    ax.clear()
    prof = np.asarray(profile, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    y_edges = row_edges(pos) if len(pos) == len(prof) else np.linspace(pos[0], pos[-1], len(prof) + 1)
    x_edges = np.array([0.0, 1.0])
    ok = np.isfinite(prof)
    lo, hi = (float(np.min(prof[ok])), float(np.max(prof[ok]))) if ok.any() else (0.0, 0.0)
    # A flat profile (a row the scan never reached) has no range for the two-slope scale.
    norm = (mcolors.TwoSlopeNorm(vmin=lo, vcenter=(lo + hi) / 2, vmax=hi)
            if hi > lo else None)
    ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(prof)[:, None],
                  cmap=TOPO_CMAP, norm=norm, shading='flat')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_title("Topo", fontsize=9)


def draw_stripe_drift_lines(ax, pos, shear, x=0.5, color=None):
    """
    The drift lines of the Line window's rows on the topo stripe: for every
    row a vertical line at x through its anchor position pos[k] plus its
    along-line drift at every energy-cell edge of its sweep, shear[k] (the
    row's shear, Drift.row_shear, in the stripe's units: a polyline, so a
    drift that turns within the sweep is drawn as far as it went), with a
    dot at the anchor; `shear` may also be (n,) ends only; rows with a NaN
    value are skipped.  `color` (default DRIFT_LINE_COLOR; MANUAL_DRIFT_COLOR
    for a manual velocity, drift_color) is the line colour.  Returns the
    artists.
    """
    color = DRIFT_LINE_COLOR if color is None else color
    pos = np.asarray(pos, dtype=np.float64)
    sh = np.asarray(shear, dtype=np.float64)
    if sh.ndim == 1:
        sh = sh[:, None]
    ok = np.isfinite(pos) & np.all(np.isfinite(sh), axis=1)
    arts = []
    if not ok.any():
        return arts
    segs = [np.column_stack([np.full(sh.shape[1] + 1, x), np.concatenate([[p], p + e])]) for p, e in zip(pos[ok], sh[ok])]
    lc = LineCollection(segs, colors=color, linewidths=1.2, zorder=3.0)
    ax.add_collection(lc)
    arts.append(lc)
    dots, = ax.plot(np.full(ok.sum(), x), pos[ok], ls='none', marker='o', ms=2.5, color=color,
                    mec=DRIFT_LINE_EDGE, mew=0.4, zorder=3.1)
    arts.append(dots)
    return arts


def draw_marker_curves(ax, energy, markers, groups, avg_on, stack_on, substack_on, texts,
                       offset_frac=0.25, legend_fontsize=8, show_title=True):
    """
    The marker curve panel of the Line and Map windows (and of their Feenstra
    companions) into `ax`:
      markers : [{'curve', 'color', 'label', 'key'}] per marker, in marker
                order ('key': the marker's place along the ordering axis,
                the position along the line or y on the map)
      groups  : [{'members': [marker indices], 'average': curve, 'color',
                  'label', 'center': key of the group's centre}] per group
      texts   : {'ylabel', 'ylabel_offset', 'title_markers', 'stack_where'}
    Average off: every marker's curve in its colour with its label.  Average
    on: one black average per group.  Average and stack on: one stack of
    blocks, each block the group's average at the bottom and, with substack,
    its members above it (highest key at the top) in the group's colour, the
    blocks ordered by their centres (lowest at the bottom); offsets
    offset_frac of the largest curve range per curve, one full range between
    blocks; the legend lists the top block first.
    """
    # Fast path while a marker is dragged: the same curves (count, colours, labels' layout) with new
    # data are updated in place instead of clearing and rebuilding the panel.
    if not (avg_on and stack_on):
        if avg_on:
            items = [(g['average'], 'black', 2.5, g['label']) for g in groups]
        else:
            items = [(m['curve'], m['color'], 2, m['label']) for m in markers]
        sig = (bool(avg_on), tuple(c for _, c, _, _ in items), texts['ylabel'], texts['title_markers'],
               len(groups), id(energy) if False else len(energy))
        prev = getattr(ax, '_mc_state', None)
        if prev is not None and prev[0] == sig and all(ln in ax._children for ln in prev[1]):
            for ln, (curve, _, _, label) in zip(prev[1], items):
                ln.set_data(energy, curve)
                ln.set_label(label)
            # The legend is rebuilt from scratch (never edited in place, so it cannot grow).
            ax.legend(loc='upper right', frameon=False, fontsize=legend_fontsize)
            if avg_on and show_title:
                ax.set_title(f"Group averages: {len(groups)} group(s)")
            ax.keep_legend_clear()
            return
    ax.clear()
    ax._mc_state = None
    ylabel = texts['ylabel']
    title = texts['title_markers']
    if avg_on and stack_on:
        blocks = []
        for g in groups:
            shown = sorted(g['members'], key=lambda i: -markers[i]['key']) if substack_on else []
            blocks.append((g['center'], shown, g['average'], g))
        blocks.sort(key=lambda b: b[0])            # lowest centre first: drawn at the bottom
        curves = [markers[i]['curve'] for _, shown, _, _ in blocks for i in shown]
        ranges = [float(np.nanmax(c) - np.nanmin(c)) for c in curves if np.any(np.isfinite(c))]
        ranges += [float(np.nanmax(b[2]) - np.nanmin(b[2])) for b in blocks if np.any(np.isfinite(b[2]))]
        max_range = max(ranges + [0.0])
        step = offset_frac * max_range
        y0 = 0.0
        for center, shown, avg, g in blocks:
            ax.plot(energy, avg + y0, color='black', lw=2.5, label=g['label'])
            for j, i in enumerate(reversed(shown)):            # lowest key just above the average
                ax.plot(energy, markers[i]['curve'] + y0 + (j + 1) * step, color=g['color'], lw=1.2)
            y0 += (len(shown) + 1) * step + max_range
        ylabel = texts['ylabel_offset']
        title = (f"Stacked group averages: {len(groups)} group(s)"
                 + (" with their members" if substack_on else "") + texts['stack_where'])
        handles, labels = ax.get_legend_handles_labels()       # listed top block first, as drawn
        ax.legend(handles[::-1], labels[::-1], loc='upper right', frameon=False, fontsize=legend_fontsize)
        ax.keep_legend_clear()
    elif avg_on:
        lines = []
        for g in groups:
            lines += ax.plot(energy, g['average'], color='black', lw=2.5, label=g['label'])
        title = f"Group averages: {len(groups)} group(s)"
        ax.legend(loc='upper right', frameon=False, fontsize=legend_fontsize)
        ax.keep_legend_clear()
        ax._mc_state = (sig, lines)
    else:
        lines = []
        for m in markers:
            lines += ax.plot(energy, m['curve'], color=m['color'], lw=2, label=m['label'])
        ax.legend(loc='upper right', frameon=False, fontsize=legend_fontsize)
        ax.keep_legend_clear()
        ax._mc_state = (sig, lines)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(ylabel)
    ax.set_title(title if show_title else '')
    if not show_title:
        ax.figure._set_info(ax, title)


def draw_sweeps(ax, entries, average, stack_on, substack_on, offset_frac=0.25):
    """
    One sweep panel of the averaging window's companion: every kept sweep
    of every entry in the entry's colour (or, stacked with 'substack' off,
    one mean per entry), the run's average in black; stacked: offset by
    offset_frac of the largest range per curve, the average at the
    bottom, as the averaging window stacks.
      entries : [{'label', 'color', 'V', 'sweeps' (npts, n)}]
      average : (V0, curve) or None
    """
    entries = entries or []
    curves = []                                            # (V, curve, colour, label, lw)
    for e in entries:
        V = e['V']
        sw = np.asarray(e['sweeps'], dtype=np.float64)
        if sw.ndim == 1:
            sw = sw[:, None]
        if sw.shape[1] == 0:
            continue
        if stack_on and not substack_on:
            curves.append((V, np.mean(sw, axis=1), e['color'], e['label'] + " (mean)", 1.2))
        else:
            for j in range(sw.shape[1]):
                curves.append((V, sw[:, j], e['color'], e['label'] if j == 0 else None, 0.9))
    n = len(curves)
    ranges = [float(np.nanmax(c) - np.nanmin(c)) for _, c, _, _, _ in curves if np.any(np.isfinite(c))]
    if average is not None and np.any(np.isfinite(average[1])):
        ranges.append(float(np.nanmax(average[1]) - np.nanmin(average[1])))
    max_range = max(ranges + [0.0])
    offset = offset_frac * max_range if stack_on else 0.0
    n_slots = n + (1 if (stack_on and average is not None) else 0)
    for k, (V, c, color, label, lw) in enumerate(curves):
        ax.plot(V, c + (n_slots - 1 - k) * offset, lw=lw, color=color, label=label)
    if average is not None:
        ax.plot(average[0], average[1], color='black', lw=1.8, label='average')
    if curves:
        ax.legend(loc='upper right', frameon=False, fontsize=7)


# ---------------------------------------------------------------------- #
#  Topography                                                              #
# ---------------------------------------------------------------------- #

def draw_topo_absolute(ax, sm4, img, scalefactor, **imshow_kw):
    """
    Draw a processed image at the file's absolute position and rotation
    (nm after the scale factor).  Returns the AxesImage.  The array is drawn
    at the resolution it was saved at: interpolation='none', so no smoothing,
    anti-aliasing, or interpolation is applied; on screen each display pixel
    shows the nearest data pixel (matplotlib treats 'none' as 'nearest' on
    screen backends), and vector output (pdf, svg) receives the raw pixels.
    """
    k = 1e9 / scalefactor
    l, r, b, t = [v * k for v in sm4.frame_extent_m()]
    cx, cy = sm4.center_m[0] * k, sm4.center_m[1] * k
    im = ax.imshow(img, origin='lower', extent=[l, r, b, t], interpolation='none', **imshow_kw)
    im.set_transform(Affine2D().rotate_deg(sm4.angle_deg).translate(cx, cy) + ax.transData)
    return im


def draw_topo_frame(ax, sm4, img, scalefactor, **imshow_kw):
    """Draw a processed image in its own scan frame (nm after the scale factor), aspect locked."""
    k = 1e9 / scalefactor
    l, r, b, t = [v * k for v in sm4.frame_extent_m()]
    im = ax.imshow(img, origin='lower', extent=[l, r, b, t], interpolation='none', **imshow_kw)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(min(l, r), max(l, r))
    ax.set_ylim(min(b, t), max(b, t))
    return im


def draw_point_sweeps(ax, xy, corrected, ends=None, lines=None, color=None):
    """
    The sweeps of the current single point at positions xy (n, 2; the drawing
    units of ax).  The first sweep (the start) is the lime cross, large and
    drawn first; with drift correction on, every later sweep is a smaller
    orange dot, drawn in time order so that a later sweep lies on top of the
    earlier ones, and the larger start marker stays visible even when the
    drift is too small to separate them, the sweep starts connected dot to
    dot in time order, and every sweep gets its drift line (draw_drift_lines):
    with `lines` (n, P, 2; Drift.sweep_lines) the polyline through the sweep,
    following the drift function, else with `ends` (n, 2) the segment from
    its start to its end.  Without correction every sweep sits at the
    recorded position and only the cross is drawn.  `color` is the drift
    lines' colour (draw_drift_lines).
    """
    ax.plot(xy[:1, 0], xy[:1, 1], ls='none', marker='X', ms=13, color=POINT_CURRENT_COLOR,
            mec='black', mew=0.8, zorder=3.0, label='first sweep')
    if corrected and len(xy) > 1:
        ok = np.all(np.isfinite(xy), axis=1)
        ax.plot(xy[ok, 0], xy[ok, 1], ls='-', lw=0.8, color=POINT_SWEEP_COLOR, alpha=0.8, zorder=2.9)
        for j in range(1, len(xy)):
            ax.plot(xy[j:j + 1, 0], xy[j:j + 1, 1], ls='none', marker='o', ms=6, color=POINT_SWEEP_COLOR,
                    mec='black', mew=0.5, zorder=3.0 + 0.001 * j, label='later sweeps' if j == 1 else None)
        ax.legend(loc='upper right', frameon=False, fontsize=8)
    if corrected and lines is not None:
        draw_drift_lines(ax, lines, markers=False, color=color)
    elif corrected and ends is not None:
        draw_drift_lines(ax, np.stack([xy, np.asarray(ends, dtype=np.float64)], axis=1), markers=False, color=color)


def draw_drift_lines(ax, lines, markers=True, zorder=3.2, color=None):
    """
    The drift lines of a file's sweeps: `lines` (n, P, 2) in the drawing
    units of ax, each row a sweep's polyline from its start through its
    position at every energy point of the sweep (Drift.sweep_lines; P = 2
    for plain start-end pairs); drawn as it is per sweep (`color`, default
    DRIFT_LINE_COLOR, MANUAL_DRIFT_COLOR for a manual velocity, over a dark
    edge so it shows on any topo colour), and, with `markers`, a dot at
    every start.  Rows with a NaN value are skipped.  Returns the artists.
    """
    color = DRIFT_LINE_COLOR if color is None else color
    L = np.asarray(lines, dtype=np.float64)
    if L.ndim == 2:
        L = L.reshape(-1, 2, 2)
    ok = np.all(np.isfinite(L), axis=(1, 2))
    arts = []
    if not ok.any():
        return arts
    segs = L[ok]
    edge = LineCollection(segs, colors=DRIFT_LINE_EDGE, linewidths=2.0, zorder=zorder, alpha=0.6)
    core = LineCollection(segs, colors=color, linewidths=0.9, zorder=zorder + 0.01)
    ax.add_collection(edge)
    ax.add_collection(core)
    arts.extend([edge, core])
    if markers:
        dots, = ax.plot(segs[:, 0, 0], segs[:, 0, 1], ls='none', marker='o', ms=2.0, color=color,
                        mec=DRIFT_LINE_EDGE, mew=0.3, zorder=zorder + 0.02)
        arts.append(dots)
    return arts


def draw_section_line(ax, p0, p1):
    """
    The browser's cross-section line on the scan area: the segment from p0
    to p1 (drawing units) in SECTION_LINE_COLOR with its two endpoints as
    handles (a filled circle at p0, the start of the section's position
    axis, a hollow one at p1).  Returns [line, start handle, end handle].
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    line, = ax.plot([p0[0], p1[0]], [p0[1], p1[1]], ls='-', lw=1.6, color=SECTION_LINE_COLOR, zorder=4.0)
    h0, = ax.plot([p0[0]], [p0[1]], ls='none', marker='o', ms=8, color=SECTION_LINE_COLOR, mec='black', mew=0.8, zorder=4.1)
    h1, = ax.plot([p1[0]], [p1[1]], ls='none', marker='o', ms=8, mfc='white', mec=SECTION_LINE_COLOR, mew=1.6, zorder=4.1)
    return [line, h0, h1]


def draw_topo_outline(ax, sm4, scalefactor, zorder=6.0):
    """
    The recorded frame of a topography (Sm4File) on an axes of absolute
    positions (nm after the scale factor): the closed polygon through its
    four corners (Sm4File.corners_m: the recorded centre, angle and extent,
    no drift correction) as a solid thin line in PARTNER_FRAME_COLOR, no
    text.  The drift window draws it last on each panel, for the other
    topo of the pair, so the zorder is above the topo image, the spectra
    positions, the markers (4.2, 4.3) and the arrows (4.15).  Returns the
    artists.
    """
    c = np.asarray(sm4.corners_m(), dtype=np.float64) * (1e9 / scalefactor)
    closed = np.vstack([c, c[:1]])
    line, = ax.plot(closed[:, 0], closed[:, 1], ls='-', lw=0.8, color=PARTNER_FRAME_COLOR, zorder=zorder)
    return [line]


def draw_drift_markers(ax, markers, pairs):
    """
    The manual-drift markers on the scan area or a drift-window panel:
    `markers` a list of dicts with 'xy' (drawing units), 'side' (0 on the
    earlier topo, 1 on the later, None while only one topo carries
    markers), 'index' (its number within its topo, from 1: the pair number,
    marker_color), 'seeded' (placed by the auto velocity) and 'moved'
    (dragged off its seeded spot); drawn in the pair's colour, a circle on
    the earlier topo (and while the side is unknown), a square on the
    later, hollow while seeded and unmoved, filled with a black edge
    otherwise, the number beside it; `pairs` a list of (xy_A, xy_B, index)
    drawn as arrows from the earlier marker to its partner in the pair's
    colour (the MATLAB arrow).  Returns the artists.
    """
    arts = []
    for m in markers:
        xy = np.asarray(m['xy'], dtype=np.float64)
        color = marker_color(m.get('index', 1) or 1)
        shape = 's' if m.get('side') == 1 else 'o'
        hollow = bool(m.get('seeded')) and not bool(m.get('moved'))
        pt, = ax.plot([xy[0]], [xy[1]], ls='none', marker=shape, ms=9,
                      mfc='none' if hollow else color, mec=color if hollow else 'black', mew=2.0 if hollow else 0.8,
                      zorder=4.2)
        txt = ax.annotate(str(m.get('index', '')), xy=(xy[0], xy[1]), xytext=(6, 6), textcoords='offset points',
                          color=color, fontsize=8, fontweight='bold', zorder=4.3)
        arts.extend([pt, txt])
    for pair in pairs:
        a, b = np.asarray(pair[0], dtype=np.float64), np.asarray(pair[1], dtype=np.float64)
        color = marker_color(pair[2] if len(pair) > 2 else 1)
        arr = ax.annotate('', xy=(b[0], b[1]), xytext=(a[0], a[1]),
                          arrowprops=dict(arrowstyle='->', color=color, lw=1.6, shrinkA=0, shrinkB=0),
                          zorder=4.15)
        arts.append(arr)
    return arts


def draw_profiles(ax, curves, xlabel, ylabel, title=None):
    """
    Height against position into `ax` (the profile window): `curves` a list
    of (label, x, z, color), each drawn as a line with small dots at the
    samples (NaN samples break the line: pixels not acquired or outside the
    topo), a legend, the axis labels, and the title.
    """
    ax.clear()
    for label, x, z, color in curves:
        x = np.asarray(x, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        ax.plot(x, z, ls='-', lw=1.2, marker='.', ms=3.5, color=color, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=9)
    if curves:
        ax.legend(loc='best', frameon=False, fontsize=8)
    ax.grid(True, lw=0.4, alpha=0.4)


def draw_profile_pairs(ax, pairs, xlabel, ylabel, title=None):
    """
    Several forward/reverse profile pairs in one axes (the profile plot):
    `pairs` a list of dicts with 'label', 'forward' (x, z) and 'reverse'
    ((x, z) or None), every pair in its own colour from the axes' colour
    cycle in the order added, the forward pass a solid line, the reverse a
    dashed one, small dots at the samples, a legend of "<label> fwd" /
    "<label> rev", the axis labels, the title, and the grid.  Nothing is
    aligned or offset: every pair as it is.
    """
    ax.clear()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for k, pair in enumerate(pairs):
        color = colors[k % len(colors)]
        for key, ls, tag in (('forward', '-', 'fwd'), ('reverse', '--', 'rev')):
            xz = pair.get(key)
            if xz is None:
                continue
            x = np.asarray(xz[0], dtype=np.float64)
            z = np.asarray(xz[1], dtype=np.float64)
            ax.plot(x, z, ls=ls, lw=1.2, marker='.', ms=3.0, color=color, label=f"{pair.get('label', '')} {tag}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=9)
    if pairs:
        ax.legend(loc='best', frameon=False, fontsize=7)
    ax.grid(True, lw=0.4, alpha=0.4)


def draw_topo_cbar(fig, rect, label='topo'):
    """
    The topo colour bar of a page: a thin vertical axes at `rect` (figure
    fraction) showing the TOPO_CMAP ramp, low at the bottom, no ticks and no
    numbers, the word `label` above it.  One per page that uses the topo
    colour map.  Returns the axes.
    """
    ax = fig.add_axes(rect)
    ax.set_navigate(False)
    ramp = np.linspace(0.0, 1.0, 256)[:, None]
    ax.imshow(ramp, aspect='auto', origin='lower', cmap=TOPO_CMAP, extent=[0, 1, 0, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(label, fontsize=7, pad=2)
    return ax


def _pixel_quads(path, half_width):
    """
    Quadrilaterals (n, 4, 2) covering a row of pixels centred on the points of
    `path` (n, 2): along the path each pixel spans from the midpoint to its
    predecessor to the midpoint to its successor (half a step beyond the ends),
    across it +/- half_width along the local normal.
    """
    p = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    n = len(p)
    if n == 1:
        d = np.array([[1.0, 0.0]])
        starts = p - half_width * d
        ends = p + half_width * d
    else:
        seg = np.diff(p, axis=0)                                           # (n - 1, 2)
        d = np.vstack([seg[:1], 0.5 * (seg[:-1] + seg[1:]), seg[-1:]]) if n > 2 else np.vstack([seg, seg])
        mids = 0.5 * (p[:-1] + p[1:])
        starts = np.vstack([p[:1] - 0.5 * seg[:1], mids])
        ends = np.vstack([mids, p[-1:] + 0.5 * seg[-1:]])
    norms = np.linalg.norm(d, axis=1)
    d = d / np.where(norms > 0, norms, 1.0)[:, None]
    normal = np.column_stack([-d[:, 1], d[:, 0]]) * half_width
    return np.stack([starts + normal, ends + normal, ends - normal, starts - normal], axis=1)


def draw_grid_stripes(ax, grid, fwd, rev, pos_f, pos_r=None, width=None, top='forward'):
    """
    The grid's own topo stripes drawn on the scan area with every pixel at
    its own position: fwd and rev are the processed forward and reverse
    topo pages of the grid (ny, nx), rev None without a reverse page; pos_f
    and pos_r (ny, nx, 2) the position of every pixel in the axes' units
    (the recorded location from the page geometry, Sm4File.pixel_xy_m, with
    the drift toggle off; the placement's corrected position with it on),
    NaN for a pixel not drawn (never acquired, or a forward pixel whose
    spectrum is not kept).  Along each row the pixels are cells from the
    midpoint to the previous pixel to the midpoint to the next
    (_pixel_quads), `width` across (the recorded slow step; the pixel pitch
    along the row when None).  Each pass is coloured on its own scale (its
    own min to max, as the topo images are drawn; the reverse rows span a
    smaller height range than the forward rows, so one shared scale would
    leave them black).  `top` names the pass drawn on top of the other
    ('forward': the reverse band under the forward one; 'reverse': the
    other way round), the browser's, the Topo window's and the stripe
    window's pass radio.  Nothing is drawn when fwd is None.
    """
    if fwd is None or pos_f is None:
        return

    def scale_of(a):
        """A Normalize from a pass's own finite min to max (None when it holds no finite pixel)."""
        v = a[np.isfinite(a)]
        if len(v) == 0:
            return None
        lo, hi = float(np.min(v)), float(np.max(v))
        return mcolors.Normalize(lo, hi if hi > lo else lo + 1.0)

    def band(img, pos, norm):
        quads, colors = [], []
        for i in range(img.shape[0]):
            ok = np.isfinite(img[i]) & np.all(np.isfinite(pos[i]), axis=1)
            if ok.sum() == 0:
                continue
            p = pos[i][ok]
            if width is not None:
                half = 0.5 * float(width)
            elif len(p) > 1:
                half = 0.5 * float(np.median(np.linalg.norm(np.diff(p, axis=0), axis=1)))
            else:
                continue
            q = _pixel_quads(p, half)
            quads.extend(q)
            colors.extend(TOPO_CMAP(norm(img[i][ok])))
        return quads, colors

    norm_f = scale_of(fwd)
    if norm_f is None:
        return
    quads_f, colors_f = band(np.asarray(fwd, dtype=np.float64), np.asarray(pos_f, dtype=np.float64), norm_f)
    if rev is not None and pos_r is not None:
        norm_r = scale_of(rev)
        if norm_r is not None:
            quads_r, colors_r = band(np.asarray(rev, dtype=np.float64), np.asarray(pos_r, dtype=np.float64), norm_r)
            if quads_r:
                ax.add_collection(PolyCollection(np.array(quads_r), facecolors=colors_r, edgecolors='none',
                                                 zorder=1.5 if top == 'reverse' else 1.4, antialiased=False))
    if quads_f:
        ax.add_collection(PolyCollection(np.array(quads_f), facecolors=colors_f, edgecolors='none',
                                         zorder=1.4 if top == 'reverse' else 1.5, antialiased=False))


if __name__ == '__main__':
    # The topo colour bar and a gamma series of one heatmap, as a check of the drawing helpers.
    DESIGN = (8, 4)           # the demo window's design size (inches)
    fig = plt.figure(figsize=DESIGN)
    img = np.outer(np.linspace(0, 1, 50), np.linspace(0, 1, 80))
    for k, g in enumerate((0.5, 1.0, 2.0)):
        ax = fig.add_axes([0.05 + 0.3 * k, 0.2, 0.24, 0.6])
        im = draw_heatmap(ax, img, None, 'jet', gamma=g)
        ax.set_title(f"γ = {g}")
        draw_colorbar(fig, im, fig.add_axes([0.05 + 0.3 * k + 0.25, 0.2, 0.01, 0.6]))
    draw_topo_cbar(fig, [0.96, 0.2, 0.012, 0.6])
    plt.show()
