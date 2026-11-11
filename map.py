"""
Per-energy maps of a grid file: the Map window (a tiled grid of every energy,
or one energy beside the file's embedded topography with energy arrows and a
curve panel), its marker groups (the Line window's method in 2D), its own
'Feenstra' toggle and Feenstra companion, the 'calibrate' / 'normalize' box,
the 'cmap norm' colour-scale toggle, one 'γ' box per energy slice, the
contour slice (a draggable level line on the topo colour bars; the topo's
iso-line at that height on every map, contour.py) with its 'Contour' toggle,
and Height shading with its 'Height' toggle (every map coloured by its value
and the topo's height together, topo_ldos_blend.TopoLdosBlend, with the
colour square, its two bars and the maps' legends).
Also in the window: the 'channel' radio (dI/dV or IETS), the 'topo pass'
radio of the embedded topo, the 'sweeps' radio of a file with several
sweeps per pixel, the filter box, the window's own 'auto drift' / 'manual
drift' / 'stripe' toggles, 'drift window' and 'stripe match' (the pixels and the topo at their
drift-corrected positions), 'map → new avg' / 'map → existing avg' (the
whole map into an averaging window), the figure composer buttons ('add to
plot' under each panel, 'new plot'; composer.py), and 'Send to fit'.  The
frame is the grid's scan frame in nm after the file's scale factor (the
`scalefactor` argument: the scale in force at the file's time, which the
browser takes from its timeline, scales.Scales).
"""

if __package__ in (None, ''):
    # Run as a file (Spyder F5): make the package importable from wherever its folder sits.
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from datetime import datetime
import numpy as np
from sm4_Browser import qtplot as plt
from sm4_Browser import fit, widgets, plots, calibration, controls, contour, records
from sm4_Browser.qtplot import QtWidgets, QtCore

Qt = QtCore.Qt
from sm4_Browser.sts import STS, curve_units, PipelineState, say, state_record, state_from_record
from sm4_Browser.averaging import Averages, Selection
from sm4_Browser.feenstra import (FeenstraWindow, feenstra_width, feenstra_titles, energy_has_zero)
from sm4_Browser.drift import DriftWindow
from sm4_Browser.composer import FigureComposer
from sm4_Browser.topo import processed_topo, process_topo, average_passes, TOPO_CMAP, TOPO_DIRECTIONS


def height_source(height, centres, pitch):
    """
    The topography as drawn, made ready for the height lookup of Height shading (heights_at):
    `height` (ny, nx) the height fraction of every topo pixel (NaN: not drawn), `centres` (ny, nx, 2)
    the pixels' centres as drawn (the frame's units), `pitch` (x, y) one topo pixel's size.  Returns
    (tree, values, points, pitch): a scipy cKDTree over the centres of the pixels with a finite height,
    in units of the pixel pitch, their heights, the same points and the pitch; None when no pixel has
    a finite height.
    """
    from scipy.spatial import cKDTree
    h = np.asarray(height, dtype=np.float64).ravel()
    c = np.asarray(centres, dtype=np.float64).reshape(-1, 2)
    p = np.array([abs(float(pitch[0])) or 1.0, abs(float(pitch[1])) or 1.0])
    ok = np.isfinite(h) & np.all(np.isfinite(c), axis=1)
    if not ok.any():
        return None
    pts = c[ok] / p
    return cKDTree(pts), h[ok], pts, p


def heights_at(source, at):
    """
    The height fraction of the topography as drawn under the points `at` (..., 2; the frame's units):
    for every point the height of the nearest drawn topo pixel (by its centre, the distance counted
    in pixel pitches; `source` from height_source).  A point that is not finite, or whose nearest
    topo pixel's centre is more than one pixel pitch away along x or along y (the point lies outside
    the topography as drawn), has no height: NaN.  Shape at.shape[:-1].
    """
    at = np.asarray(at, dtype=np.float64)
    out = np.full(at.shape[:-1], np.nan)
    if source is None:
        return out
    tree, values, pts, p = source
    q = at.reshape(-1, 2) / p
    ok = np.all(np.isfinite(q), axis=1)
    if not ok.any():
        return out
    _, idx = tree.query(q[ok])
    near = np.all(np.abs(pts[idx] - q[ok]) <= 1.0 + 1e-9, axis=1)
    flat = out.reshape(-1)
    flat[np.flatnonzero(ok)] = np.where(near, values[idx], np.nan)
    return out


class Map:
    """
    Grid file: a toggle ('Grid view' / 'Single view') between (a) a tiled
    grid of the embedded topography (first cell, the curve panel under it)
    and every energy's map to their right, every cell titled ('Topo', the
    map's energy), and (b) one energy beside the file's own embedded
    topography with energy arrows ('< E', 'E >') and the curve panel (one
    panel, moved between the views).  Maps are the channel (dI/dV, or IETS
    by the 'channel' radio) through the one pipeline path (Sm4File.view for
    the window's state, sts.PipelineState: the filter box's Hampel and
    Savitzky-Golay filters, the pipeline box's denoise, calibrate, gate,
    locks and normalize rows, the division by the broadened I/V with
    'Feenstra' on; view(state).shown, memoized by the state),
    every location calibrated against its own I(V) and unit-area normalized
    as the 'calibrate' / 'normalize' box says (both off at open), energy
    ascending, reshaped to (n_lines, n_fast) in the scan frame (self.extent:
    the recorded frame in nm after the scale factor; a never-acquired pixel
    is NaN and drawn transparent).  Filter controls at the bottom.  The
    'topo pass' radio picks the pass of the embedded topo ('average': the
    two passes averaged, each normalized on its own, topo.average_passes;
    with a placement on, the reverse rows resampled onto the forward rows'
    placed positions first, _averaged_pass); the 'sweeps' radio of a
    file with several sweeps per pixel, at the bottom right under '< E' /
    'E >', picks the sweep view the maps are made of (Sm4File.with_sweeps).
    The window is arranged by one Qt layout (automatic layout: the one rule
    for every window; nothing placed by hand): the pipeline box on top, the
    panels in the middle (the single view's and the grid view's on two
    pages that swap), the controls in columns at the bottom.

    Colour scale: with 'cmap norm' off (the default) every energy slice is
    coloured on one scale, the minimum and maximum over the whole cube, in
    the single view and in every grid map; on, each map has its own minimum
    and maximum.  The 'clip %' box beside it (0 at open, 0 <= p < 50) clips
    the colour scale to the p-th and (100 - p)-th percentiles of the
    values (plots.percentile_range: the whole cube with 'cmap norm' off,
    each map its own with it on), so that a few extreme pixels do not set
    the colouring; a pixel beyond the limits takes the end colour, the
    colour bar spans the clipped limits, and nothing else changes (the map
    arrays, the marker curves, the companion, the sends and the composer
    read the same arrays).  Every energy slice has its own 'γ' (the
    power-law exponent of the colour map, plots.heat_norm, applied on top
    of the limits): one box under each grid map, and the single view's box
    edits the current slice's value.

    Markers (both views): the Line window's marker method in 2D.  A
    marker is a grid pixel (line, fast); it is drawn as a solid circle as
    wide as the pixel's diagonal (overlapping its neighbours slightly),
    centred on the pixel, on the map and on the embedded topo, in the grid view on
    the topo cell and on every energy's map (each map at its own energy's
    pixel shifts).  The 'avgs' box
    (0 to MAX_GROUPS) is the number of groups; every group has its own row
    in the marker box: its 'Markers' box (0 to 10 markers; a group at 0
    keeps its centre), framed in the group's colour, with 'add new' and
    'add existing' beside it, the rows top to bottom in the order of the
    groups' centres (highest y at the top).  The marker box grows with the
    groups, the panels giving way, until it is as tall as the panel row;
    past that it keeps that height and scrolls (controls.py's number
    boxes: digits step with Up / Down).  Group membership is by index.  A group's m
    markers are the m pixels nearest its centre (ties by angle); new groups
    start at the free slot of an even layout along the grid's diagonal with
    the last group's size; fewer groups drops the last groups.  The window
    has its own 'Feenstra' toggle (starting at the caller's state,
    `divide_iv`: the browser opens it on; independent of the browser's
    after; the browser hands over the Feenstra settings) and a Feenstra
    companion window (feenstra.FeenstraWindow) that opens with it and
    follows it: the map at the current energy in four versions with the
    marker curves under each.  Dragging: a press on a
    marker (any panel it is drawn on) moves its centre with the cursor, and
    the pixel under its centre is the marker's (the one whose spectrum it
    shows); with
    Average on, a press on a group's centre (a black X on a map) moves the
    whole group by the same pixel offset, clipped so no member leaves the
    grid.  The curve panel shows one spectrum per marker (from the cube: the
    window's filters, flags, Feenstra, and channel), or with Average on one
    average per group (the plain mean, Averages.mean_rows, a marker on a
    never-acquired pixel left out), with 'stack' and 'substack' as in the
    Line window.  'add new' and 'add existing' beside each 'Markers' box
    put that group into the averages as the browser's buttons add a single
    point: a Selection of its markers' columns (each marker one sweep) goes
    to a new average, or to the open average picked from the pop-up list
    (controls.AveragePicker).  'Send to fit' sends the marker spectra
    (Average off) or the group averages (Average on) to the single-point
    fit, as the Line window does (calibrated and normalized, the
    fit's pipeline).  'map → new avg' / 'map → existing avg' put the whole
    map (every acquired pixel's spectrum as one sweep of one entry) into a
    new or an open averaging window.  'save' (in the column with 'Send to
    fit') writes the marker record averages/<file stem>_map_markers.json
    (marker_record: the file and its sweep view, the pipeline state, the
    view, the energy step, the gammas, 'cmap norm' and 'clip %', the topo
    pass and processors, the drift toggles, the contour slice's level and
    its traces, the marker groups and markers, the stack states), one per
    file, overwritten; the saved averages window lists it and the browser
    reopens the map from it with everything restored (restore_markers).
    'add to plot' under the topo, the
    map and the curve panel sends that panel as shown to the figure
    composer (composer.FigureComposer, one per window); 'new plot' empties
    it.  One topo colour bar sits right of the embedded topo in the single
    view and one beside the topo cell in the grid view.

    Contour slice (both views; WSxM's flooding, as outlines): a horizontal
    level line on each topo colour bar (one level, the two lines move
    together; plots.CONTOUR_LEVEL_COLOR, at half the range at open), dragged
    by a press on it (the line follows the cursor's height while the button
    is down, wherever the cursor is sideways: the bar is narrow).  Its
    height is a fraction (z_frac) of the drawn topo's colour range: the pass
    on top (the 'topo pass' radio's), on its own range as drawn (each pass
    has its own with a drift source on).  The topo's iso-line at that height
    (contour.iso_segments: marching squares on the processed pass in nm) is
    drawn on the single view's map and on every grid map as single-colour
    traces (plots.draw_contour: CONTOUR_COLOR, 1 pt, above the
    pixels, under the markers), never as topo pixels and never on the topo
    panels themselves, at the topo pixels' positions as drawn
    (contour.to_frame: the frame, plus the pass's chained pixel shifts with
    a drift source on); the same traces on every energy's map, since the
    surface does not move with the energy while the maps' pixels do.  A
    never-acquired pixel bounds no trace.  The traces go to the publication
    files with the maps (mplexport: a line collection).  The 'Contour'
    toggle (on at open) shows or hides the slice: off, no trace is drawn
    and the level lines are hidden.  The drift toggles and the drift
    window are described at __init__.

    Height shading ('Height', off at open; both views; display only,
    nothing is recomputed): on, a map is no longer coloured by the maps'
    colour map alone; every pixel's colour is blend(s, t) of the window's
    TopoLdosBlend (topo_ldos_blend.py, made from the topo colour map and
    the maps' colour map as they are): s the map's value through the norm
    it has without the shading ('cmap norm', 'clip %' and the slice's γ),
    t the height of the topo under the pixel as a fraction of the drawn
    topo's own range (_heights: the pass on top, the image the contour
    slice reads; the topo pixel drawn nearest where the map pixel is drawn
    at that energy, so with a drift source on the heights follow the shift
    within the sweep; a map pixel drawn outside the topography as drawn
    has no height and is not drawn).  The lowest height shows the maps'
    colour map exactly, and every colour keeps the lightness (CIELAB L*)
    of the maps' colour map's colour for its value at every height: a
    shaded map is as light and dark as the unshaded map at every pixel
    drawn, and the height shows as colour only.  'Height' and 'Contour'
    are never on together: turning one on turns the other off; both may
    be off.  The topo panels keep the topo colour map.
    With 'Height' on (_draw_height_panels) the
    colour square (value left to right, height bottom to top) sits between
    the topo's ramp and the map in the single view and in the first map
    position of the grid view (the maps one place on, _arrange_grid), in a
    map panel's shape, with the topo bar at its left (the heights in nm)
    and the value bar under it; with one scale for every map shown ('cmap
    norm' off; in the grid view also every slice at the same γ) the value
    bar carries that scale and the maps have no colour bar of their own,
    otherwise it shows 'low' and 'high' and every map has its legend under
    it in place of its colour bar (the blend over the map's own scale,
    'Topo' written vertically at its left).  The sizes are plots.BLEND_BAR,
    BLEND_GAP and BLEND_LEGEND, fractions of a map panel's height (qtplot
    Figure.attach).  The Feenstra companion's four maps and a map sent to
    the figure composer are shaded the same way (_shading_view).
    """

    MAX_GROUPS = 10
    STACK_OFFSET_FRAC = 0.25
    M_COLORS = controls.MarkerGroupControls.M_COLORS
    # The Feenstra companion shows heatmaps (its own 'γ' boxes and pipeline box).
    feenstra_heatmaps = True

    DESIGN = (16, 9)       # the design size (inches): the text zoom of the window is its size relative to this (qtplot.window_zoom)

    def __init__(self, grid, active_procs, scalefactor, hampel_defaults, sgolay_defaults, cmap=plots.MAP_CMAP,
                 divide_iv=False, iv_broadening=0.0, channel='didv', averages=None, fit_settings=None,
                 iv_zero_sub=True, iv_broadened=False, drift=None, auto_on=False, manual_on=False):
        """
        The Map window of a grid file, opened at once (plt.show at the end).
          grid            : the grid's sm4file.Sm4File (a sweep view of it for a file with
                            several sweeps per pixel)
          active_procs    : the topo processor flags, one per topo.TOPO_PROCESSORS entry
                            (the browser's: 'Slope Sub', 'Δz sub')
          scalefactor     : the file's x/y scale factor: recorded metres / scalefactor are the
                            frame's nm (the browser hands the scale in force at the file's time)
          hampel_defaults, sgolay_defaults : the filter box's starting values (every filter off)
          cmap            : the maps' colour map (plots.MAP_CMAP by default)
          divide_iv, iv_broadening, iv_zero_sub, iv_broadened : the window's own 'Feenstra'
                            toggle's starting state and the Feenstra settings (the companion's
                            boxes; feenstra_width gives the width the division uses)
          channel         : 'didv' or 'iets'; a channel the file lacks falls back to the first
                            it holds
          averages        : the averaging.Averages registry of the 'add' buttons (the
                            browser's); None makes one at the first add
          fit_settings    : the keyword arguments of Fit.select_model for 'Send to fit', plus
                            'use_cache'
          drift, auto_on, manual_on : the directory's drift.Drift and the starting states of
                            the window's own 'auto drift' / 'manual drift' toggles (off without
                            a Drift)
        """
        self.grid = grid
        self.scalefactor = scalefactor
        self.cmap = cmap
        # The window's own 'auto drift' and 'manual drift' toggles (set from the caller's states at
        # opening, independent after): with a drift source on, every pixel of every energy slice sits at
        # its spectrum's chained position (where the sweep started, by the time from the grid's first
        # spectrum: Drift.result 'xy_corr', the same chain that places the topo) plus the drift within
        # the sweep up to that energy (the sweep's dwell per point times the energy index: 'shift'), both
        # from the file's velocity (the before/after topo pair's, auto or manual).  Drift.result is asked
        # with the grid's stored fast-axis state (stripe None), so when the stripe matching is set for the
        # grid its placement of the lines along the fast axis and their measured forward compression
        # enter the chained positions and the within-sweep shifts as well, and the result is made even
        # with both toggles off (then with no slow-axis drift).  The embedded topo is drawn as quads with
        # every pixel at its chained position (_draw_topo, plots.PixelMap: nothing is clipped), the frame
        # widens to hold every shifted pixel (_drift_bounds; with a toggle on), and the spectra's drift
        # lines are drawn on the single view's topo in the auto or the manual colour (with a toggle on;
        # never on a map, whose pixels' positions show the drift).  'drift
        # window' opens the file's before and after topos for marker pairs (drift.DriftWindow), whose
        # span refreshes the maps at once (Drift.subscribe, below).
        self.drift = drift
        self.auto_on = bool(auto_on and drift is not None)
        self.manual_on = bool(manual_on and drift is not None)
        self._drift_windows = []
        self.stripe_window = None                   # the stripe window opened by 'stripe match' (one at a time)
        self._drift_source = None                   # the velocity's source of the last result (the lines' colour)
        self._drift_res = None                      # Drift.result of the grid, once per toggle state (False: none)
        self._bounds = None                         # _drift_bounds, memoized with _drift_res
        self._pixel_maps = {}                       # drift on: the PixelMap of the single view and of every grid map
        self._topo_maps = {}                        # drift on: the topo passes' PixelMaps per axes (id(ax) -> [...])
        self._grid_topo_stale = True                # the grid view's topo cell needs drawing
        self.divide_iv = bool(divide_iv)
        self.iv_broadening = float(iv_broadening)
        self.iv_zero_sub = bool(iv_zero_sub)        # the Feenstra settings (the companion's boxes)
        self.iv_broadened = bool(iv_broadened)
        self.calibrate = False                      # the 'calibrate' / 'normalize' box: both off at open, as in
        self.normalize = False                      # every window that uses Feenstra
        self.gate = calibration.GATE_DEFAULT        # the box's 'gate' row and its 'H' / 'SG' boxes
        self.cmap_norm = False                      # 'cmap norm': each map its own colour scale
        self.clip_percent = 0.0                     # 'clip %': the colour scale's percentile clip (plots.percentile_range), 0 = none
        held = grid.channels()                      # the maps start on a channel the file holds
        self.channel = channel if channel in held else (held[0] if held else 'didv')
        self.order = np.argsort(grid.energy)
        self.V = grid.energy[self.order]
        self.n_e = len(self.V)
        self.e_index = int(np.argmin(np.abs(self.V)))
        self.gammas = np.ones(self.n_e)             # one γ per energy slice
        self.view_mode = 'single'                  # 'single' or 'grid' (the pipeline's view is view(state))
        self.active_procs = list(active_procs)
        self.direction = 'forward'
        self.has_topo = bool(grid.topo_pages)
        k = 1e9 / scalefactor
        l, r, b, t = [v * k for v in grid.frame_extent_m()]
        self.extent = [l, r, b, t]
        self.hampel_defaults = hampel_defaults
        self.sgolay_defaults = sgolay_defaults
        self.averages = averages                    # the Browser's registry, or one made at the first add
        self.fit_settings = dict(fit_settings) if fit_settings else {}
        self._avg_windows = []                      # averaging windows opened from here stay referenced
        self._fits = []                 # sent fits stay referenced
        self._avg_picker = None                     # the pop-up list of the open averages (made at the first use)
        self.feenstra = None                        # the Feenstra companion (opened at the end of __init__)
        self.composer = None                        # the figure composer (composer.FigureComposer), made at the first add
        self._single_im = None                      # the single view's drawn map (its colour limits for the composer)
        self.z_frac = 0.5                           # the contour slice: its level as a fraction of the drawn topo's colour range
        self._contour_memo = None                   # the slice's segments in the frame for the level and the topo drawn (dropped with the topo)
        self._contour_artists = []                  # the slice's traces on the maps of the view shown
        self._level_lines = []                      # the slice's level line on each topo colour bar (single view, grid view)
        self._topo_im = None                        # drift off: the single view's topo image (its colour range scales the slice)
        # Height shading and the contour slice's toggle ('Height', off at open; 'Contour', on at open;
        # never on together, _on_height / _on_contour): with 'Height' on every map is coloured by its
        # value and the height of the topo drawn under each of its pixels (plots.blend_colors with the
        # window's TopoLdosBlend of the topo colour map and the maps' colour map; _heights).
        self.height_on = False
        self.contour_on = True
        self.blend = plots.make_blend(TOPO_CMAP, self.cmap)
        self._height_memo = {}                      # the heights of the maps' pixels per energy slice, and the lookup source (dropped with the topo)
        self._height_vis = None                     # the state the Height panels' visibility was last set for (_draw_height_panels)
        self._grid_legends = {}                     # the legend under each grid map (energy index -> axes), made at the first need

        # Marker state: one group of two markers at the grid's centre.
        self.n_lines, self.n_fast = int(grid.n_lines), int(grid.n_fast)
        self.group_sizes = [2]
        self.marker_px = self.place_group(self.layout_centers(1)[0], 2)
        self.group_ctr = []                         # the centre of every group (kept by a group without markers)
        self._sync_centres()
        self._active_obj = None
        self._marker_artists = []

        self.fig = plt.figure(figsize=self.DESIGN)
        fig = self.fig
        # The controls (controls.py: the one set of every window).
        self.filters = controls.FilterControls(fig, hampel=hampel_defaults, sgolay=sgolay_defaults,
                                               hampel_on=False, sgolay_on=False)
        self.filters.on_change(self.recompute)
        self.rad_dir = controls.make(fig, 'topo pass', value='forward')
        self.rad_dir.on_change(self._on_direction)
        self.rad_channel = controls.make(fig, 'channel', value=STS.CHANNEL_NAMES[self.channel])
        self.rad_channel.on_change(self._on_channel)
        # Several sweeps per pixel (sweeps.py): the 'sweeps' radio chooses the view the maps are made of.
        self.rad_sweeps = None
        if getattr(grid, 'sweep_info', None) is not None:
            self.rad_sweeps = controls.make(fig, 'sweeps', labels=list(grid.sweep_modes()), value=grid.sweep_mode)
            self.rad_sweeps.on_change(self._on_sweeps)
        self.chk_feenstra = controls.make(fig, 'Feenstra', value=self.divide_iv)
        self.chk_feenstra.on_change(self._on_feenstra_toggle)
        self.pipeline = controls.PipelineBox(fig, calibrate=self.calibrate, normalize=self.normalize, gate=self.gate)
        self.pipeline.on_change(self._on_pipeline_change)
        self.chk_cmap_norm = controls.make(fig, 'cmap norm', value=self.cmap_norm)
        self.chk_cmap_norm.on_change(self._on_cmap_norm)
        self.box_clip = controls.Text(fig, 'clip %', f"{self.clip_percent:g}")
        self.box_clip.on_change(self._on_clip)
        self.chk_height = controls.make(fig, 'Height', value=self.height_on)
        self.chk_height.on_change(self._on_height)
        self.chk_contour = controls.make(fig, 'Contour', value=self.contour_on)
        self.chk_contour.on_change(self._on_contour)
        self.chk_auto = controls.make(fig, 'auto drift', value=self.auto_on)
        self.chk_auto.on_change(lambda flag: self._on_drift_toggle('auto', flag))
        self.chk_manual = controls.make(fig, 'manual drift', value=self.manual_on)
        self.chk_manual.on_change(lambda flag: self._on_drift_toggle('manual', flag))
        self.btn_drift_win = controls.make(fig, 'drift window')
        self.btn_drift_win.on_change(self._on_drift_window)
        # The grid's stripe matching: the 'stripe' toggle (one state per file, Drift.set_stripe_on) and
        # the 'stripe match' button, the row under the drift toggles.
        self.chk_stripe = controls.make(fig, 'stripe', value=False)
        self.chk_stripe.on_change(self._on_stripe_toggle)
        self.btn_stripe_win = controls.make(fig, 'stripe match')
        self.btn_stripe_win.on_change(self._on_stripe_window)
        self._sync_stripe_toggle()
        self.btn_view = controls.make(fig, 'Grid view')
        # Its label names the view a click switches to ('Grid view' / 'Single view', _apply_view): the
        # settings section lists it (controls.settings_text).
        self.btn_view.stateful = True
        self.btn_view.setting_label = 'switch to'
        self.btn_view.on_change(self.toggle_view)
        self.steps = controls.StepButtons(fig, '< E', 'E >')
        self.steps.on_change(self.step)
        self.groups_ctl = controls.MarkerGroupControls(fig, max_groups=self.MAX_GROUPS, n_groups=len(self.group_sizes))
        self.groups_ctl.on_avgs(self._on_avgs_change)
        self.groups_ctl.on_markers(self._on_group_slider)
        self.groups_ctl.on_add_new(self._on_add_new)
        self.groups_ctl.on_add_existing(self._on_add_existing)
        self.stacks = controls.StackControls(fig, average=False, needs_average=True)
        self.stacks.on_change(self._redraw_markers)
        self.btn_send = controls.make(fig, 'Send to fit')
        self.btn_send.on_change(self._send_to_fit)
        self.btn_save = controls.make(fig, 'save')
        self.btn_save.on_change(self._on_save)
        # The whole map into an averaging window: 'map → new avg' starts one, 'map → existing avg' lists
        # the open ones (map_selection: every acquired pixel's spectrum as one sweep of one entry).
        self.btn_map_new = controls.make(fig, 'map → new avg')
        self.btn_map_new.on_change(self._on_map_new)
        self.btn_map_exist = controls.make(fig, 'map → existing avg')
        self.btn_map_exist.on_change(self._on_map_existing)
        # The figure composer (composer.FigureComposer): 'add to plot' under the topo, the map and the curve
        # panel sends that panel as shown; 'new plot' empties the figure.
        self.btn_plot_topo = controls.make(fig, 'add to plot')
        self.btn_plot_topo.on_change(lambda: self._on_add_to_plot('topo'))
        self.btn_plot_map = controls.make(fig, 'add to plot')
        self.btn_plot_map.on_change(lambda: self._on_add_to_plot('map'))
        self.btn_plot_curve = controls.make(fig, 'add to plot')
        self.btn_plot_curve.on_change(lambda: self._on_add_to_plot('spectra'))
        self.btn_new_plot = controls.make(fig, 'new plot')
        self.btn_new_plot.on_change(self._on_new_plot)
        self.gamma_single = controls.GammaBox(fig, gamma=self.gammas[self.e_index])
        self.gamma_single.on_change(self._on_gamma_single)

        # automatic layout: the one rule for every window (nothing placed by hand).  One Qt layout on the
        # canvas holds the controls and the panels' cells (qtplot.PanelCell); Qt sizes and places them.
        # Top: the pipeline box at the right.  Middle: the panels: the single view's cells (topo, its
        # ramp, map, its colour bar, curves) or, in their place, the grid view's page (the topo with a ramp
        # beside it and the curve panel under it, one map per energy to their right, each map's γ box under it; the one
        # curve panel moves between the views, _place_curve), with 'add to plot' under the topo, the map and the
        # curve panel in both views and the single view's γ box under its map.  Bottom: the controls
        # in columns, left to right in the order they always had; the sweeps choice under the '< E' /
        # 'E >' buttons at the bottom right.
        self._cell_topo = fig.cell()
        self._cell_topo_cbar = fig.cell(inner_w=9)
        self._cell_square = fig.cell()                       # Height shading's colour square (shown with 'Height' on)
        self._cell_square.setVisible(False)
        self._cell_map = fig.cell()
        self._cell_cbar = fig.cell(inner_w=14)
        self._cell_curve = fig.cell()
        self._single_cells = (self._cell_topo, self._cell_topo_cbar, self._cell_map, self._cell_cbar, self._cell_curve)
        self._grid_page = QtWidgets.QWidget()
        self._grid_layout = QtWidgets.QGridLayout(self._grid_page)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        self._grid_layout.setHorizontalSpacing(4)
        self._grid_layout.setVerticalSpacing(1)
        self._grid_page.setVisible(False)
        sg = QtWidgets.QGridLayout()
        self._single_layout = sg                             # the curve panel's place in the single view (_place_curve)
        sg.setContentsMargins(0, 0, 0, 0)
        sg.setHorizontalSpacing(8)
        sg.setVerticalSpacing(2)
        # Columns: the topo, its ramp, the colour square of Height shading (column 2: hidden, and without
        # stretch, until 'Height' is on, _draw_height_panels), the map, its colour bar, the curves.
        for c, cell in zip((0, 1, 3, 4, 5), self._single_cells):
            sg.addWidget(cell, 0, c)
        sg.addWidget(self._cell_square, 0, 2)
        sg.addWidget(self._grid_page, 0, 0, 1, 6)            # the grid view takes the single view's place
        left = Qt.AlignmentFlag.AlignLeft
        sg.addWidget(self.btn_plot_topo.qt, 1, 0, left)
        sg.addLayout(controls.row(self.btn_plot_map, 'stretch', self.gamma_single), 1, 3)
        sg.addWidget(self.btn_plot_curve.qt, 1, 5, left)
        for c, s in ((0, 28), (3, 28), (5, 26)):
            sg.setColumnStretch(c, s)
        sg.setRowStretch(0, 1)
        # The columns of controls in one row that wraps onto more rows in a narrow window
        # (controls.FlowLayout), the view and energy group at the right end.
        strip = controls.FlowLayout(push_last=True)
        strip.addWidget(self.filters)
        strip.addWidget(controls.box(self.chk_auto, self.rad_dir, orient='v'))
        strip.addWidget(controls.box(self.chk_feenstra, self.rad_channel, orient='v'))
        strip.addWidget(controls.box(self.btn_new_plot, self.btn_map_exist, self.btn_map_new, orient='v'))
        strip.addWidget(controls.box(controls.row(self.chk_manual, self.btn_drift_win),
                                     controls.row(self.chk_stripe, self.btn_stripe_win),
                                     self.groups_ctl.box, orient='v'))
        strip.addWidget(controls.box(self.stacks.chk_average, self.groups_ctl.s_avgs, orient='v'))
        strip.addWidget(controls.box(self.btn_send, self.btn_save, self.stacks.chk_stack, orient='v'))
        strip.addWidget(controls.box(controls.row(self.chk_cmap_norm, self.box_clip),
                                     controls.row(self.chk_height, self.chk_contour), self.stacks.chk_substack, orient='v'))
        strip.addWidget(controls.box(self.btn_view, self.steps, self.rad_sweeps, orient='v'))
        root = QtWidgets.QVBoxLayout()
        root.setSpacing(4)
        root.addLayout(controls.row('stretch', self.pipeline))
        root.addLayout(sg, 1)
        root.addLayout(strip)
        fig.set_layout(root)
        # The marker box grows with the groups, the panels giving way, until it is as tall as the panel
        # row; past that it keeps that height and scrolls.
        self.groups_ctl.set_height_cap(lambda: 0.5 * (self.groups_ctl.box.height() + self._cell_topo.height()))

        # Single view axes.
        self.ax_topo = fig.add_axes(cell=self._cell_topo)
        self.ax_map = fig.add_axes(cell=self._cell_map)
        self.cax_map = fig.add_axes(cell=self._cell_cbar)
        self.ax_curve = fig.add_axes(cell=self._cell_curve)
        self.ax_curve.keep_clear_of(self.cax_map)        # the curves never run into the colour bar's labels
        self.ax_topo_cbar = plots.draw_topo_cbar(fig, cell=self._cell_topo_cbar)
        # Height shading's panels of the single view (hidden until 'Height' is on, _draw_height_panels): the
        # colour square with its topo bar and value bar (plots.BlendKey) in its cell between the topo's ramp
        # and the map, and the legend attached under the map (Figure.attach; shown with 'cmap norm' on).
        self.key_single = plots.BlendKey(fig, self._cell_square)
        self.key_single.set_visible(False)
        self.ax_map_legend = fig.attach(self.ax_map, 'bottom', plots.BLEND_LEGEND, plots.BLEND_GAP)
        self.ax_map_legend.set_visible(False)
        # One row: the topo and the map keep their aspect (the frame, or the drift's wider box) and set
        # the height; the two colour bars, the colour square and the curve panel take it (qtplot Figure.align_row).
        fig.align_row(self.ax_topo, self.ax_topo_cbar, self.key_single.ax, self.ax_map, self.cax_map, self.ax_curve)
        # Grid view axes (built once, on their page until toggled): the topo in the first column's top
        # cell with the curve panel under it (the single view's panel, moved there while the grid view is
        # shown: _place_curve), the maps in the columns to their right in order of energy (grid_shape,
        # _grid_slot), one γ box under each map.
        self._grid_cols, self._grid_rows = self.grid_shape(self.n_e)
        gl = self._grid_layout
        cell = fig.cell(min_px=(20, 20))
        gl.addWidget(cell, 0, 0)
        self.grid_topo_ax = fig.add_axes(cell=cell)
        self.grid_topo_ax.set_xticks([])
        self.grid_topo_ax.set_yticks([])
        # The grid view's topo colour bar beside the topo cell (the single view's is hidden there), as narrow
        # as the maps' bars; it carries one of the contour slice's two level lines (_draw_level_lines).
        tcell = fig.cell(min_px=(4, 20), inner_w=5)
        gl.addWidget(tcell, 0, 1)
        self.grid_topo_cbar = plots.draw_topo_cbar(fig, cell=tcell)
        self.grid_axes, self.grid_caxes, self.grid_gammas = [], [], []
        self._grid_cells, self._grid_ccells = [], []         # every map's cell and its colour bar's cell (_arrange_grid moves them)
        for e in range(self.n_e):
            rr, cc = self._grid_slot(e)
            cell = fig.cell(min_px=(20, 20))
            ccell = fig.cell(min_px=(4, 20), inner_w=5)
            self._grid_cells.append(cell)
            self._grid_ccells.append(ccell)
            gl.addWidget(cell, 2 * rr, 2 * cc)
            gl.addWidget(ccell, 2 * rr, 2 * cc + 1)
            ax = fig.add_axes(cell=cell)
            cax = fig.add_axes(cell=ccell)
            ax.set_xticks([])
            ax.set_yticks([])
            self.grid_axes.append(ax)
            self.grid_caxes.append(cax)
            g = controls.GammaBox(fig, gamma=self.gammas[e])
            g.box.setting_label = f"γ {self.V[e]:.2f} V"        # its map's energy, as the map's title (settings_text)
            g.on_change(lambda value, e=e: self._on_gamma_grid(e, value))
            gl.addWidget(g.qt, 2 * rr + 1, 2 * cc, Qt.AlignmentFlag.AlignRight)
            self.grid_gammas.append(g)
        for cc in range(self._grid_cols + 1):
            gl.setColumnStretch(2 * cc, 1)
        for rr in range(self._grid_rows):
            gl.setRowStretch(2 * rr, 1)
        # Height shading's colour square of the grid view (plots.BlendKey): its cell is kept on the grid's
        # page, outside the layout and hidden, until 'Height' is on; it then takes the first map position
        # and the maps move one place on (_arrange_grid).
        self._grid_square_cell = fig.cell(min_px=(20, 20))
        self._grid_square_cell.setParent(self._grid_page)
        self._grid_square_cell.setVisible(False)
        self.key_grid = plots.BlendKey(fig, self._grid_square_cell)
        self.key_grid.set_visible(False)
        # Each row of cells is one row: the maps (and the topo) set the height, their colour bars take it.
        self._align_grid_rows()
        self.grid_ims = [None] * self.n_e
        self.maps = None
        self.fig.canvas.mpl_connect('pick_event', self._on_pick)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        # Every manual change of the drift (a span applied or cleared in a drift window), made
        # in any drift window, refreshes the maps (Drift.subscribe); the subscription ends with
        # the window.
        if self.drift is not None:
            fig = self.fig
            self.drift.subscribe(self._refresh_drift, alive=lambda: plt.fignum_exists(fig))
            fig.canvas.mpl_connect('close_event', lambda event: self.drift.unsubscribe(self._refresh_drift))
        self._draw_level_lines()
        self._draw_topo(self.ax_topo)
        self.recompute()
        self._apply_view()
        # The Feenstra companion: the four versions of the map, opened whatever the toggle's state.
        self.feenstra = FeenstraWindow(self, f"{grid.name}: map")
        plt.show()

    def is_open(self):
        """True while the window's figure exists (tested on the Figure itself, not its number)."""
        return widgets.figure_is_open(self.fig)

    @property
    def avg_on(self):
        """The 'Average' state (controls.StackControls); stack_on and substack_on are the 'stack' and 'substack' states."""
        return self.stacks.average

    @property
    def stack_on(self):
        return self.stacks.stack

    @property
    def substack_on(self):
        return self.stacks.substack

    def set_divide_iv(self, flag, broadening=0.0):
        """The window's 'Feenstra' state (or its broadening) set and the maps recomputed (the toggle's handler and callers outside the window)."""
        self.divide_iv = bool(flag)
        self.iv_broadening = float(broadening)
        self.recompute()

    def _on_feenstra_toggle(self, flag):
        """The window's own 'Feenstra' toggle: the maps recomputed with the division on or off; a file that cannot be divided (no Current channel: recompute printed why) puts the toggle back."""
        flag = bool(flag)
        if flag == self.divide_iv:
            return
        old = self.divide_iv
        before = self.maps
        self.set_divide_iv(flag, self.iv_broadening)
        if self.maps is before:                     # the file cannot be divided: recompute printed why
            self.divide_iv = old
            self.chk_feenstra.set(old)
            return
        t = feenstra_titles(self.iv_zero_sub, self.iv_broadened, self.iv_broadening, energy_has_zero(self.grid.energy))
        print(f"Feenstra {'on' if flag else 'off'}: the map window shows " + (t['ratio'] + "." if flag else "dI/dV."))

    def _on_pipeline_change(self, state):
        """The 'calibrate' / 'gate' / 'normalize' box: the maps recomputed with the flags; refused, the box goes back."""
        old = (self.calibrate, self.normalize, self.gate)
        self.calibrate, self.normalize, self.gate = state['calibrate'], state['normalize'], state.get('gate', self.gate)
        before = self.maps
        self.recompute()
        if self.maps is before:
            self.calibrate, self.normalize, self.gate = old
            self.pipeline.set_state(*old)
            return
        state = {'calibrate': self.calibrate, 'normalize': self.normalize, 'gate': self.gate}
        print(f"{controls.PipelineBox.text(state)}: the map window shows {self._quantity()}.")

    def _on_cmap_norm(self, flag):
        """'cmap norm': each map its own colour scale (on) or one scale over the whole cube (off)."""
        self.cmap_norm = bool(flag)
        self.draw_single()
        if self.view_mode == 'grid':
            self.draw_grid()
        self.fig.canvas.draw_idle()

    def _on_clip(self, text):
        """'clip %': the colour scale's percentile clip, 0 <= p < 50 (plots.percentile_range); an edit outside it, or not a number, is reported and the box put back."""
        try:
            p = float(str(text).strip())
            if not np.isfinite(p) or p < 0.0 or p >= 50.0:
                raise ValueError("the clip must be a percentage from 0 up to (not including) 50")
        except ValueError as exc:
            print(f"clip % box: '{text}' not accepted ({exc}). The box is put back.")
            self.box_clip.set(f"{self.clip_percent:g}")
            return
        if p == self.clip_percent:
            return
        self.clip_percent = p
        self.box_clip.set(f"{p:g}")
        self.draw_single()
        if self.view_mode == 'grid':
            self.draw_grid()
        self.fig.canvas.draw_idle()

    # --- 'Height' and 'Contour' (never on together; display only, nothing is recomputed) ---

    def _on_height(self, flag):
        """'Height': Height shading on or off (the maps coloured by their value and the topo's height, _heights; the colour square and the legends shown, _draw_height_panels); turning it on turns 'Contour' off.  Refused, the toggle put back, without a topo page."""
        flag = bool(flag)
        if flag and not self.has_topo:
            print("Map window: no topography page in this file; 'Height' stays off.")
            self.chk_height.set(False)
            self.fig.canvas.draw_idle()
            return
        if flag == self.height_on:
            return
        self.height_on = flag
        if flag and self.contour_on:
            self.contour_on = False
            self.chk_contour.set(False)
        self._apply_shading()

    def _on_contour(self, flag):
        """'Contour': the contour slice on or off (its traces on the maps and its level line on the topo colour bars); turning it on turns 'Height' off."""
        flag = bool(flag)
        if flag == self.contour_on:
            return
        self.contour_on = flag
        if flag and self.height_on:
            self.height_on = False
            self.chk_height.set(False)
        self._apply_shading()

    def _apply_shading(self):
        """The 'Height' / 'Contour' state applied to the window (the toggles' handlers, restore_markers): the level lines shown with 'Contour', the maps' images dropped (made again shaded or not), the grid view's cells arranged for the state (_arrange_grid), the maps of the view shown drawn anew (the contour slice and the Height panels with them), and the panels of the view shown or hidden (_apply_view).  The maps themselves are not recomputed."""
        for h in self._level_lines:
            h.set_visible(self.contour_on)
        self._drop_map_images()
        self._arrange_grid()
        if self.maps is not None:
            self.draw_single()
            if self.view_mode == 'grid':
                self.draw_grid()
        self._apply_view()

    def _drop_map_images(self):
        """Every drawn map of the grid view removed from its panel, images and PixelMaps alike, and forgotten: the next draw makes them again (the single view's map is made anew at every draw)."""
        for im in self.grid_ims:
            if im is not None and not isinstance(im, plots.PixelMap):
                try:
                    im.remove()
                except Exception:
                    pass
        self._clear_pixel_maps()

    # --- the drift (the window's own 'auto drift' / 'manual drift' toggles, the drift window) ---

    def _drift_active(self):
        """True with a Drift at hand and a toggle on (a velocity source; the stripe matching alone places the pixels as well, _drift_result)."""
        return self.drift is not None and (self.auto_on or self.manual_on)

    def _drift_tag(self):
        """The titles' placement text: '' without a result, ', stripe matched' for the stripe matching alone (source 'none (...)'), else ', drift corrected (<source>)'."""
        if self._drift_result() is None:
            return ""
        src = self._drift_source
        if src and str(src).startswith('none'):
            return ", stripe matched"
        return ", drift corrected" + (f" ({src})" if src else "")

    def _drift_result(self):
        """
        Drift.result of the grid for the window's toggles and the grid's
        stored fast-axis state (the chained placement: 'xy_corr' the
        corrected start of every spectrum, 'pix' the corrected topo pixels
        per pass, 'shift' the drift within every sweep, 'source' the
        velocity's rule), computed once per toggle state (_drift_res; False
        stands for "none"); None with both toggles off and the stripe
        matching off, or when no correction can be made (the console says
        why).  With both toggles off but the stripe matching set, the
        result still places the lines along the fast axis with no slow-axis
        drift (Drift.result's 'none (stripe match only)' model).
        """
        stripe = self.drift is not None and self.drift.fast_state(self.grid, None, self.auto_on, self.manual_on) != 'off'
        if not (self._drift_active() or stripe):
            return None
        if self._drift_res is None:
            try:
                res = self.drift.result(self.grid, None, None, self.auto_on, self.manual_on)
            except Exception as exc:
                print(f"Map window: drift correction not available ({exc}); pixels unshifted.")
                res = None
            self._drift_res = res if res is not None else False
            self._drift_source = res.get('source') if res is not None else None
        return self._drift_res or None

    def _to_frame_shift(self, d_m):
        """Absolute displacements (..., 2) in metres into the frame's units (nm after the scale factor), same shape."""
        d = np.asarray(d_m, dtype=np.float64)
        return (d.reshape(-1, 2) @ self.grid.rotation() * 1e9 / self.scalefactor).reshape(d.shape)

    def _chain_shift(self):
        """The chained shift of every spectrum's start (its corrected position minus the recorded one), (N, 2) metres; None without a result."""
        res = self._drift_result()
        if res is None:
            return None
        xy = np.asarray(self.grid.spec_xy_m, dtype=np.float64)
        corr = np.asarray(res['xy_corr'], dtype=np.float64)
        n = min(len(xy), len(corr))
        return corr[:n] - xy[:n]

    def _frame_shift(self, e):
        """
        The pixel shifts of energy slice e (ascending index) in the frame
        (nm after the scale factor), (n_lines, n_fast, 2): every pixel's
        chained shift (where its spectrum started, by the time from the
        grid's first spectrum: Drift.result 'xy_corr') plus the drift within
        its sweep up to that energy point (the sweep's dwell per point times
        the energy index: 'shift'), rotated into the scan frame; None
        without a result (_drift_result) or without energy points.
        """
        res = self._drift_result()
        chain = self._chain_shift()
        if res is None or chain is None or res.get('shift') is None:
            return None
        g = self.grid
        n = g.n_lines * g.n_fast
        col = int(self.order[e])
        s = np.full((n, 2), np.nan)
        m = min(n, len(chain), res['shift'].shape[0])
        s[:m] = chain[:m] + res['shift'][:m, col, :]
        return self._to_frame_shift(s).reshape(g.n_lines, g.n_fast, 2)

    def _topo_shift(self, direction):
        """
        The chained shift of every pixel of a topo pass in the frame's units,
        (ny, nx, 2; NaN where no spectrum places the pixel): Drift.result
        'pix' minus the recorded pixel locations; None without a result, a
        placement of that pass, or when the page's shape differs from it.
        """
        res = self._drift_result()
        pix = None if res is None else (res.get('pix') or {}).get(direction)
        if pix is None:
            return None
        rec = self.grid.pixel_xy_m(direction)
        pix = np.asarray(pix, dtype=np.float64)
        if pix.shape != rec.shape:
            return None
        return self._to_frame_shift(pix - rec)

    def _pixel_centres(self, shape):
        """The nominal centres of the pixels of an image of `shape` (ny, nx) over the frame (nm), (ny, nx, 2), as plots.PixelMap places its cells (row i from the bottom edge, column j from the left; a reversed extent counts signed)."""
        ny, nx = int(shape[0]), int(shape[1])
        l, r, b, t = self.extent
        dx, dy = (r - l) / nx, (t - b) / ny
        jj, ii = np.meshgrid(np.arange(nx), np.arange(ny))
        return np.stack([l + (jj + 0.5) * dx, b + (ii + 0.5) * dy], axis=-1)

    def _averaged_pass(self):
        """
        The 'average' topo pass with a placement on: (img, shift), the two
        processed passes averaged where they were measured
        (topo.average_passes: the reverse row of every scan line resampled
        onto the forward row's placed positions, each pass normalized on
        its own, then the mean), in nm, on the forward pass's pixels at
        its chained shifts ((ny, nx, 2), _topo_shift); a pixel the forward
        placement leaves out is NaN.  The positions are the nominal pixel
        centres in the frame plus each pass's shifts (_pixel_centres, as
        plots.PixelMap draws the passes).  None without two topo pages or
        without both passes' placements (said on the console; _draw_topo
        then draws the passes as placed).
        """
        g = self.grid
        if len(g.topo_pages) < 2:
            return None
        sh_f, sh_r = self._topo_shift('forward'), self._topo_shift('reverse')
        f = process_topo(g.topo_raw('forward'), self.active_procs)
        r = process_topo(g.topo_raw('reverse'), self.active_procs)
        if (sh_f is None or sh_r is None or f.shape != r.shape or sh_f.shape[:2] != f.shape
                or sh_r.shape[:2] != r.shape):
            print("Map window: the average needs both passes placed; the passes are drawn as placed.")
            return None
        c = self._pixel_centres(f.shape)
        placed_f = np.all(np.isfinite(sh_f), axis=-1)
        placed_r = np.all(np.isfinite(sh_r), axis=-1)
        pos_f = np.where(placed_f[..., None], c + sh_f, np.nan)
        pos_r = np.where(placed_r[..., None], c + sh_r, np.nan)
        img = average_passes(f, r, pos_f, pos_r) * 1e9
        return np.where(placed_f, img, np.nan), sh_f

    def _drift_bounds(self):
        """
        The limits [l, r, b, t] whenever a result places the pixels: the
        bounding box of everything drawn, exactly (edge to edge, no
        margin): every acquired map pixel's cell at every energy (the maps
        are NaN only where never acquired), every topo pixel drawn in the
        single view's topo at its shift (the passes as placed, or the
        average on the forward placement; plots.pixel_bounds; every placed
        pixel of both passes before that topo is drawn), and the spectra's
        drift lines (_frame_lines; the markers sit on the pixels).  Every
        energy counts, so the view holds still while the energy is stepped.
        The recorded frame without a result, or when nothing is drawn.
        Memoized with the result (_bounds; dropped by a pass change too).
        """
        if self._bounds is not None:
            return self._bounds
        l, r, b, t = self.extent
        frame = [min(l, r), max(l, r), min(b, t), max(b, t)]
        res = self._drift_result()
        chain = self._chain_shift()
        g = self.grid
        boxes = []
        if res is not None and chain is not None and res.get('shift') is not None:
            n = g.n_lines * g.n_fast
            m = min(n, len(chain), res['shift'].shape[0])
            s = np.full((n, self.n_e, 2), np.nan)
            s[:m] = chain[:m, None, :] + res['shift'][:m][:, self.order, :]
            sh = self._to_frame_shift(s).reshape(g.n_lines, g.n_fast, self.n_e, 2).transpose(2, 0, 1, 3)
            acq = np.zeros(n, dtype=bool)
            a = np.asarray(g.acquired, dtype=bool)[:n]
            acq[:len(a)] = a
            boxes.append(plots.pixel_bounds(np.where(acq.reshape(g.n_lines, g.n_fast), 0.0, np.nan), self.extent, sh))
        drawn = self._topo_maps.get(id(self.ax_topo)) or []
        if drawn:
            boxes += [plots.pixel_bounds(pm.img, self.extent, pm.shift) for pm in drawn]
        else:
            for direction in ('forward', 'reverse'):
                sh = self._topo_shift(direction)
                if sh is None:
                    continue
                try:
                    img = process_topo(g.topo_raw(direction), self.active_procs)
                except Exception:
                    continue
                if sh.shape[:2] != img.shape:
                    continue
                boxes.append(plots.pixel_bounds(np.where(np.all(np.isfinite(sh), axis=-1), img, np.nan), self.extent, sh))
        pts = [np.array([[bx[0], bx[2]], [bx[1], bx[3]]]) for bx in boxes if bx is not None]
        box = plots.drawn_box(*pts, self._frame_lines()) if res is not None else None
        self._bounds = list(box) if box is not None else frame
        return self._bounds

    def _draw_topo(self, ax, small=False):
        """
        The embedded topo into `ax` (the single view's topo panel; the grid
        view's topo cell with small=True): with a drift source on, the
        two passes as quads at their chained positions (plots.PixelMap:
        nothing is clipped, a pixel no spectrum places is not drawn), the
        pass the 'topo pass' radio names on top, or for 'average' one layer,
        the two passes averaged where they were measured, on the forward
        pass's placement (_averaged_pass, topo.average_passes); off, the
        chosen pass at the recorded pixels, or their average
        (topo.processed_topo; plots.draw_topo_frame).  The limits are
        _frame_limits'.  The single view's topo carries the spectra's drift
        lines whenever a result places the pixels (plots.draw_drift_lines,
        in the source's colour; _frame_lines); the grid view's cell is
        titled 'Topo' (_grid_title, the maps' title function).  The contour
        slice's memo is dropped (the slice follows the topo drawn,
        _contour_segments) and the single view's drift-off image is kept
        (_topo_im: its colour range scales the slice, _topo_level).
        """
        for pm in self._topo_maps.pop(id(ax), []):
            try:
                pm.remove()
            except Exception:
                pass
        ax.clear()
        self._contour_memo = None
        self._height_memo = {}
        g = self.grid
        fs = 7 if small else 9
        if not self.has_topo:
            ax.text(0.5, 0.5, "no topography page in this file", ha='center', va='center',
                    transform=ax.transAxes, fontsize=fs)
            return
        corrected = self._topo_shift('forward') is not None
        if corrected:
            maps = []
            avg = self._averaged_pass() if self.direction == 'average' else None
            if avg is not None:
                # 'average': one layer, both passes averaged where they were measured, on the forward placement.
                img, sh = avg
                maps.append(plots.PixelMap(ax, img, self.extent, TOPO_CMAP, shift=sh, zorder=1.0))
                what = "average on the forward placement" + self._drift_tag()
            else:
                top = 'reverse' if self.direction == 'reverse' else 'forward'
                passes = ['forward', 'reverse'] if top == 'reverse' else ['reverse', 'forward']    # the last drawn on top
                for p in passes:
                    if p == 'reverse' and len(g.topo_pages) < 2:
                        continue
                    sh = self._topo_shift(p)
                    img = process_topo(g.topo_raw(p), self.active_procs) * 1e9
                    if sh is None or sh.shape[:2] != img.shape:
                        print(f"Map window: the {p} topo pass has no placement; not drawn.")
                        continue
                    img = np.where(np.all(np.isfinite(sh), axis=-1), img, np.nan)
                    maps.append(plots.PixelMap(ax, img, self.extent, TOPO_CMAP, shift=sh,
                                               zorder=1.0 if p == top else 0.5))
                what = (f"both passes, {top} on top" if len(maps) > 1 else passes[-1]) + self._drift_tag()
            self._topo_maps[id(ax)] = maps
            if ax is self.ax_topo:
                self._topo_im = None
            ax.set_aspect('equal', adjustable='box')
        else:
            # 'average' is topo.processed_topo's (the two passes, each normalized on its own); one page: the forward.
            direction = 'forward' if (self.direction == 'average' and len(g.topo_pages) < 2) else self.direction
            img = processed_topo(g, direction, self.active_procs) * 1e9
            im = plots.draw_topo_frame(ax, g, img, self.scalefactor, cmap=TOPO_CMAP)
            if ax is self.ax_topo:
                self._topo_im = im
            what = direction
        self._frame_limits(ax)
        if small:
            self._grid_title(ax, "Topo")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_title(f"Topography: {g.name}  [{what}]", fontsize=fs)
            ax.set_xlabel("x (nm, scan frame)")
            ax.set_ylabel("y (nm, scan frame)")
            # The spectra's drift lines on the single view's topo only (never on a map: the maps show the
            # drift by their pixels' positions), drawn with the topo, so a marker move never redraws them.
            lines = self._frame_lines()
            if lines is not None:
                plots.draw_drift_lines(ax, lines, color=plots.drift_color(self._drift_source))

    def _frame_lines(self):
        """
        The spectra's drift lines in the frame (nm), (n, npts, 2): every
        sweep's polyline from its corrected start through its position at
        every energy point (Drift.sweep_lines: the model's drift plus the
        line's forward compression along the line where measured),
        whenever a result places the pixels (a toggle on, or the stripe
        matching alone); None otherwise.  Memoized per toggle state
        (_lines_memo; a stripe change drops it through _refresh_drift).
        """
        if self._drift_result() is None:
            return None
        key = (self.auto_on, self.manual_on)
        memo = getattr(self, '_lines_memo', None)
        if memo is not None and memo[0] == key:
            return memo[1]
        self._lines_memo = (key, self._frame_lines_compute())
        return self._lines_memo[1]

    def _frame_lines_compute(self):
        """The drift lines of _frame_lines computed: Drift.sweep_lines (absolute metres) into the frame's nm."""
        try:
            L = self.drift.sweep_lines(self.grid, None, None, self.auto_on, self.manual_on)
        except Exception as exc:
            print(f"Map window: drift lines not available ({exc}).")
            return None
        if L is None:
            return None
        g = self.grid
        k = 1e9 / self.scalefactor
        return (g.to_frame_m(L.reshape(-1, 2)) * k).reshape(L.shape)

    def _on_drift_toggle(self, which, flag):
        """The window's 'auto drift' / 'manual drift' toggles: on only with a drift model at hand (else the toggle goes back)."""
        flag = bool(flag)
        chk = {'auto': self.chk_auto, 'manual': self.chk_manual}[which]
        if flag and self.drift is None:
            print("Map window: no drift model at hand; the toggle stays off.")
            chk.set(False)
            self.fig.canvas.draw_idle()
            return
        current = {'auto': self.auto_on, 'manual': self.manual_on}[which]
        if flag == current:
            return
        if which == 'auto':
            self.auto_on = flag
        else:
            self.manual_on = flag
        self._refresh_drift()
        print(f"Map window: {which} drift {'on' if flag else 'off'}.")

    def _sync_stripe_toggle(self):
        """The 'stripe' toggle shows the grid's stripe matching under the window's drift setting (Drift.stripe_on); off without a drift model."""
        chk = getattr(self, 'chk_stripe', None)
        if chk is None:
            return
        on = self.drift is not None and self.drift.stripe_on(self.grid, self.auto_on, self.manual_on)
        chk.set(bool(on))

    def _on_stripe_toggle(self, flag):
        """'stripe': the grid's stripe matching on or off under every drift setting at once (Drift.set_stripe_on: one state per file, every window's toggle follows); refused, the toggle put back, without a drift model or topo pages."""
        if self.drift is None or not self.grid.topo_pages:
            print("Map window: no drift model at hand, or no topo pages; the stripe toggle stays off.")
            self.chk_stripe.set(False)
            self.fig.canvas.draw_idle()
            return
        self.drift.set_stripe_on(self.grid, flag)
        self._sync_stripe_toggle()

    def _on_stripe_window(self):
        """'stripe match': the grid's stripe window (stripe.StripeWindow), one at a time, opened at the window's drift setting with the pass of the 'topo pass' radio on top."""
        if self.drift is None or not self.grid.topo_pages:
            print("Map window: no drift model at hand, or no topo pages; no stripe window.")
            return
        w = self.stripe_window
        if w is not None and w.is_open():
            w.raise_()
            return
        from sm4_Browser.stripe import StripeWindow
        try:
            self.stripe_window = StripeWindow(self.drift, self.grid, self.scalefactor, list(self.active_procs),
                                              setting=self.drift.slow_setting(self.grid, self.auto_on, self.manual_on),
                                              pass_top='reverse' if self.rad_dir.value == 'reverse' else 'forward')
        except ValueError as exc:
            print(f"Map window: stripe window not opened ({exc}).")

    def _on_drift_window(self):
        """'drift window': the file's before and after topos with marker pairs (drift.DriftWindow; partners seeded by the velocity governing the pair); a change applied there reaches every subscriber of the Drift, this window included."""
        if self.drift is None:
            print("Map window: no drift model at hand; no drift window.")
            return
        try:
            w = DriftWindow(self.drift, self.grid, self.scalefactor, list(self.active_procs))
        except ValueError as exc:
            print(f"Map window: drift window not opened ({exc}).")
            return
        self._drift_windows.append(w)

    def _refresh_drift(self):
        """The drift state applied anew (a toggle, a sweep view, a change in any drift or stripe window): the 'stripe' toggle synced, the memoized result, bounds and lines dropped, the pixel maps removed, the topo and the maps redrawn."""
        self._sync_stripe_toggle()
        self._drift_res = None
        self._bounds = None
        self._lines_memo = None
        self._clear_pixel_maps()
        self._draw_topo(self.ax_topo)
        self._grid_topo_stale = True
        self.draw_single()
        if self.view_mode == 'grid':
            self.draw_grid()
        self.fig.canvas.draw_idle()

    def _clear_pixel_maps(self):
        """Every PixelMap of the maps removed and the grid view's images forgotten (made again at the next draw)."""
        for pm in self._pixel_maps.values():
            try:
                pm.remove()
            except Exception:
                pass
        self._pixel_maps = {}
        self.grid_ims = [None] * self.n_e

    def _on_gamma_single(self, value):
        """The single view's γ box: the current slice's γ (shared with its grid box)."""
        self.gammas[self.e_index] = float(value)
        self.grid_gammas[self.e_index].set(value)
        self.draw_single()
        if self.view_mode == 'grid':
            self.draw_grid()
        self.fig.canvas.draw_idle()

    def _on_gamma_grid(self, e, value):
        """A grid map's γ box: that slice's γ (shared with the single view's box when it shows the slice)."""
        self.gammas[e] = float(value)
        if e == self.e_index:
            self.gamma_single.set(value)
        self.draw_grid()
        if e == self.e_index:
            self.draw_single()
        self.fig.canvas.draw_idle()

    # --- the Feenstra settings (the companion's '0Vsub', 'broadened', and width boxes) ---

    def iv_width(self):
        """The broadening width the division uses (feenstra_width)."""
        return feenstra_width(self.iv_broadening, self.iv_broadened)

    def feenstra_settings(self):
        """The window's Feenstra settings: {'zero_sub', 'broadened', 'broadening'}."""
        return {'zero_sub': self.iv_zero_sub, 'broadened': self.iv_broadened, 'broadening': self.iv_broadening}

    def set_feenstra_settings(self, zero_sub=None, broadened=None, broadening=None):
        """
        A Feenstra setting changed (the companion's boxes; None = unchanged):
        the maps are recomputed with it (the division when the window's
        toggle is on), the companion refreshed, and every fit sent
        from this window receives it (Fit.set_feenstra).  Returns False
        (nothing changed) when the file cannot be processed with it.
        """
        new = (self.iv_zero_sub if zero_sub is None else bool(zero_sub),
               self.iv_broadened if broadened is None else bool(broadened),
               self.iv_broadening if broadening is None else float(broadening))
        old = (self.iv_zero_sub, self.iv_broadened, self.iv_broadening)
        if new == old:
            return True
        self.iv_zero_sub, self.iv_broadened, self.iv_broadening = new
        before = self.maps
        self.recompute()
        if self.maps is before:                     # recompute printed why
            self.iv_zero_sub, self.iv_broadened, self.iv_broadening = old
            return False
        note = ("0 V subtraction " + ('on' if new[0] else 'off') if new[0] != old[0]
                else "broadening " + ('on' if new[1] else 'off') if new[1] != old[1]
                else f"broadening {new[2]:g} V")
        t = feenstra_titles(self.iv_zero_sub, self.iv_broadened, self.iv_broadening, energy_has_zero(self.grid.energy))
        print(f"{note}: the map window shows " + (t['ratio'] + "." if self.divide_iv else "dI/dV."))
        for an in list(self._fits):
            try:
                an.set_feenstra(self.iv_width(), self.iv_zero_sub)
            except Exception as exc:
                print(f"Map window: a fit did not take the Feenstra settings: {exc}")
        return True

    def _feenstra_refresh(self):
        """The companion redrawn when it is open (after every marker or map change)."""
        w = getattr(self, 'feenstra', None)
        if w is not None and w.is_open():
            w.refresh()

    @property
    def state(self):
        """
        The window's PipelineState (sts.PipelineState.of): the channel, the
        filter box's four settings, the pipeline box's rows (calibrate,
        normalize, the gate with its locks and denoise), the window's own
        'Feenstra' toggle and the Feenstra settings; no energy crop.  Every
        drawn quantity comes from view(self.state).
        """
        return PipelineState.of(self.filters, {'calibrate': self.calibrate, 'normalize': self.normalize, 'gate': self.gate},
                                self.feenstra_settings(), self.channel, self.divide_iv)

    def view(self, state=None):
        """
        What the window draws for a state (the window's own when None):
        the grid's view (Sm4File.view: didv, current, ivv, ratio, shown,
        every spectrum in the file's column order), one computation per
        state (the memo is dropped when the sweep view changes, _on_sweeps);
        the pipeline's console lines said once.
        """
        state = self.state if state is None else state
        memo = self.__dict__.setdefault('_view_memo', {})
        if state not in memo:
            v = self.grid.view(state)
            say(v.notes)
            memo[state] = v
        return memo[state]

    def _cube_of(self, columns):
        """Raw columns (npts, N) of the grid reshaped like the maps (energy ascending, never acquired NaN)."""
        g = self.grid
        n = g.n_lines * g.n_fast
        c = np.asarray(columns, dtype=np.float64)[self.order, :][:, :n].copy()
        c[:, ~g.acquired[:n]] = np.nan
        return c.reshape(self.n_e, g.n_lines, g.n_fast)

    def _curve_items(self, cube):
        """
        The plots.draw_marker_curves items of the current marker state for a
        cube (n_e, n_lines, n_fast): per marker its pixel's curve (NaN for a
        never-acquired pixel), colour, label '(x, y) nm', key = its y (the
        ordering axis); per group its members, the plain mean of its
        acquired members' curves (NaN when none), colour, label, centre y.
        """
        markers = []
        for i, px in enumerate(self.marker_px):
            x, y = self.pixel_xy(px)
            markers.append({'curve': cube[:, px[0], px[1]], 'color': self.marker_color(i),
                            'label': f'({x:.2f}, {y:.2f}) nm', 'key': float(y)})
        groups = []
        for g, members in enumerate(self.groups()):
            rows = [cube[:, self.marker_px[i][0], self.marker_px[i][1]] for i in members
                    if self.pixel_acquired(self.marker_px[i])]
            avg = Averages.mean_rows(np.vstack(rows)) if rows else np.full(self.n_e, np.nan)
            groups.append({'members': list(members), 'average': avg, 'color': self.M_COLORS[g % len(self.M_COLORS)],
                           'label': self.group_label(g, members), 'center': self.group_center_xy(members, g)[1]})
        return markers, groups

    def _curve_texts(self, quantity=None):
        """The texts of plots.draw_marker_curves for the window's quantity (or another: the companion's)."""
        q = self._quantity() if quantity is None else quantity
        return {'ylabel': f"{q} (arb. units)", 'ylabel_offset': f"{q} (arb. units, offset)",
                'title_markers': f"{q} at markers", 'stack_where': ", blocks by centre (highest y at the top)"}

    def feenstra_view(self, state=None):
        """
        The companion's view (feenstra.FeenstraWindow) for a state (the
        window's own when None; the companion asks with its own calibrate,
        normalize and gate rows overriding it): the map at the current
        energy in four versions, every one a field of view(state) reshaped
        like the maps (_cube_of): the dI/dV of the pipeline with the
        division off, the current the division works on (the denoised
        current I/g with 'denoise' on; its value at 0 V subtracted with
        '0Vsub'), its I/V as the division divides by it (broadened with
        'broadened'), and the ratio with the division on, with the marker
        curves of each in the current marker state; a file without the
        Current channel gives notes for the current, I/V, and ratio panels,
        the IETS channel a note for the ratio panel.  With the state's
        `normalize` on the current and the I/V are unit area per pixel
        (normalization.normalize_abs); with `calibrate` off the dI/dV is the
        lock-in output as recorded and the ratio that output divided by the I/V.
        """
        state = self.state if state is None else state
        g = self.grid
        v = self.view(state)
        calibrate, normalize = state.calibrate, state.normalize
        q = ("d2I/dV2 (IETS)" if state.channel == 'iets' else "dI/dV") + ("" if calibrate else ", uncalibrated")
        at = f"at {self.V[self.e_index]:.3f} V"
        t = feenstra_titles(state.iv_zero_sub, state.iv_broadened, state.iv_broadening, energy_has_zero(g.energy))
        nn = "" if normalize else ", not normalized"
        versions = {
            'didv': {'title': f"{q} (division off){nn} {at}",
                     'ylabel': f"{q} ({curve_units(state.channel, calibrate, normalize, g.units_lia, g.units_iets, g.units_iv)})"},
            'iv': {'title': f"{t['iv']} {at}", 'ylabel': "I (unit area)" if normalize else "I (A)"},
            'ivv': {'title': f"{t['ivv']} {at}", 'ylabel': "I/V (unit area)" if normalize else "I/V (A/V)"},
            'ratio': {'title': f"{t['ratio']} (division on){nn} {at}", 'ylabel': "(dI/dV)/(I/V) (arb. units)"},
        }
        cubes = {'didv': self._cube_of(v.didv)}
        if v.current is None:
            for key in ('iv', 'ivv', 'ratio'):
                versions[key]['note'] = "no Current channel in this file"
        else:
            cubes['iv'] = self._cube_of(v.current)
            cubes['ivv'] = self._cube_of(v.ivv)
            if state.channel == 'iets':
                versions['ratio']['note'] = "the division applies to dI/dV only"
            else:
                cubes['ratio'] = self._cube_of(v.ratio)
        for key, cube in cubes.items():
            markers, groups = self._curve_items(cube)
            versions[key].update({'top': cube[self.e_index], 'markers': markers, 'groups': groups})
        return self._feenstra_view_dict(g, at, versions, q)

    def _feenstra_view_dict(self, g, at, versions, q):
        """The feenstra_view dict of kind 'map' (the keys feenstra.FeenstraWindow lists) around the versions."""
        return {'kind': 'map',
                'title': f"{g.name}: map {at} ({self.e_index + 1} of {self.n_e})",
                'energy': self.V,
                'versions': versions,
                'texts': self._curve_texts(q),
                'avg_on': self.avg_on,
                'extent': self.extent,
                'limits': self._drift_bounds() if self._drift_result() is not None else None,
                'shift': self._frame_shift(self.e_index),
                'marker_points': [(self.marker_xy(px, True), self.marker_color(i)) for i, px in enumerate(self.marker_px)],
                'marker_diameter': self.pixel_diagonal(),
                'centers': [self.group_center_xy(m, g) for g, m in enumerate(self.groups())],
                'cmap': self.cmap,
                **self._shading_view()}

    def _shading_view(self):
        """Height shading as the companion and the composer take it, for the shown energy: {'blend': the window's TopoLdosBlend, 'height': the height fraction under every pixel (_heights), 'z_range': the heights (lo, hi) of the topo drawn in nm, 'aspect': a map panel's shape}; every value None with 'Height' off or without heights."""
        height = self._heights(self.e_index) if self.height_on else None
        src = self._topo_level() if height is not None else None
        if height is None or src is None:
            return {'blend': None, 'height': None, 'z_range': None, 'aspect': None}
        return {'blend': self.blend, 'height': height, 'z_range': plots.finite_range(src[0]),
                'aspect': self._frame_aspect()}

    def recompute(self):
        """
        The maps: every spectrum of the grid through the one pipeline path
        for the window's state (view(self.state).shown: the filter box's
        settings, the pipeline box's rows, the division when 'Feenstra' is
        on with the Feenstra settings, the channel), energy ascending,
        reshaped to (n_e, n_lines, n_fast) with never-acquired pixels NaN
        (_cube_of); then the single view (and the grid view when shown)
        redrawn.  A grid that cannot be processed (no Current channel for
        the division, no such channel) leaves self.maps as it was with the
        reason printed: the callers test `self.maps is before` to put their
        control back.
        """
        try:
            self.maps = self._cube_of(self.view().shown)
        except Exception as exc:
            print(f"Map window: {exc}")
            return
        self.draw_single()
        if self.view_mode == 'grid':
            self.draw_grid()
        self.fig.canvas.draw_idle()

    def _quantity(self):
        """Name of the displayed quantity: dI/dV, (dI/dV)/(I/V), or d2I/dV2 (IETS), with the flags that are off."""
        q = ("(dI/dV)/(I/V)" if self.divide_iv else "dI/dV") if self.channel == 'didv' else "d2I/dV2 (IETS)"
        return q + ("" if self.calibrate else ", uncalibrated") + ("" if self.normalize else ", not normalized")

    def _limits(self, img=None):
        """(vmin, vmax) of a map's colour scale: with 'cmap norm' off one range over the whole cube, with it on the given map's own; both through the 'clip %' box (plots.percentile_range: the p-th and (100 - p)-th percentiles, the finite range at 0).  The single view, the grid view and update_heatmap all take their limits here."""
        if self.maps is None:
            return None, None
        return plots.percentile_range(img if self.cmap_norm else self.maps, self.clip_percent)

    def draw_single(self):
        """
        The single view's map at the current energy (e_index): a
        plots.PixelMap with the pixel shifts when a drift result places
        them (_frame_shift), else plots.draw_heatmap; the colour scale of
        _limits and the slice's γ; the colour bar labelled with the
        quantity and its units; the frame limits; the title ("... drift
        corrected (<source>)" with shifts); the γ box set to the slice; then
        the contour slice on the maps of the view shown
        (_draw_contour_artists), the markers and the curve panel
        (_redraw_markers).  The topo panel is _draw_topo's.
        """
        ax = self.ax_map
        ax.clear()
        self.cax_map.clear()
        self._pixel_maps.pop('single', None)
        m = self.maps[self.e_index]
        lo, hi = self._limits(m)
        shift = self._frame_shift(self.e_index)
        height = self._heights(self.e_index) if self.height_on else None      # 'Height': the topo's height under every pixel
        if shift is not None:
            im = plots.PixelMap(ax, m, self.extent, self.cmap, gamma=self.gammas[self.e_index], shift=shift,
                                vmin=lo, vmax=hi, blend=self.blend if height is not None else None, height=height)
            self._pixel_maps['single'] = im
            ax.set_aspect('equal', adjustable='box')
        elif height is not None:
            im = plots.BlendImage(ax, m, height, self.extent, self.blend, self.cmap, gamma=self.gammas[self.e_index],
                                  vmin=lo, vmax=hi)
        else:
            im = plots.draw_heatmap(ax, m, self.extent, self.cmap, gamma=self.gammas[self.e_index], vmin=lo, vmax=hi,
                                    origin='lower', aspect='equal', interpolation='none')
        self._single_im = im
        self._frame_limits(ax)
        q = self._quantity()
        plots.draw_colorbar(self.fig, im, self.cax_map, label=self._cbar_label())
        ax.set_title(f"{q.split(',')[0]} map at {self.V[self.e_index]:.3f} V: {q}, energy {self.e_index + 1} of {self.n_e}"
                     + ("" if self.cmap_norm else ", one colour scale for every energy")
                     + (self._drift_tag() if shift is not None else ""), fontsize=10)
        ax.set_xlabel("x (nm, scan frame)")
        ax.set_ylabel("y (nm, scan frame)")
        self.gamma_single.set(self.gammas[self.e_index])
        # The contour slice, the markers and the curve panel follow the maps (the topo panel is _draw_topo's).
        self._draw_contour_artists()
        self._redraw_markers()
        if self.height_on and self.view_mode == 'single':
            self._draw_height_panels()

    def _cbar_label(self):
        """The label of the maps' colour bar: the displayed quantity (_quantity) and its units; with 'Height' on the label of the value bar and of the single view's legend."""
        units = ("arb. units" if self.divide_iv and self.channel == 'didv'
                 else curve_units(self.channel, self.calibrate, self.normalize, self.grid.units_lia,
                                  self.grid.units_iets, self.grid.units_iv))
        return f"{self._quantity()} ({units})"

    def draw_grid(self):
        """
        The grid view: the topo cell when stale (_draw_topo, small), then
        every energy's map on the colour scale of _limits with its own γ:
        with pixel shifts a new plots.PixelMap each time (the old one
        removed), else an image made once (plots.draw_heatmap) and updated
        after (plots.update_heatmap); the map titled by its energy at the
        first draw (_grid_title); each map's colour bar redrawn.  A redrawn topo cell
        loses its markers with its clear: they are drawn again
        (_draw_marker_artists) in the grid view.  Every map's colour limits
        are _limits (the cube's or its own, through the clip).
        """
        topo_redrawn = self._grid_topo_stale
        if self._grid_topo_stale:
            self._draw_topo(self.grid_topo_ax, small=True)
            self._grid_topo_stale = False
        if topo_redrawn and self.view_mode == 'grid':
            self._draw_marker_artists()
        for e, ax in enumerate(self.grid_axes):
            m = self.maps[e]
            lo, hi = self._limits(m)
            shift = self._frame_shift(e)
            height = self._heights(e) if self.height_on else None             # 'Height': the topo's height under every pixel
            if shift is None and self.grid_ims[e] is not None \
                    and isinstance(self.grid_ims[e], plots.BlendImage) != (height is not None):
                # The image kept is not of the form wanted now (shaded or not): it is made again below.
                try:
                    self.grid_ims[e].remove()
                except Exception:
                    pass
                self.grid_ims[e] = None
            if shift is not None:
                old = self._pixel_maps.pop(e, None)
                if old is not None:
                    old.remove()
                if self.grid_ims[e] is not None and not isinstance(self.grid_ims[e], plots.PixelMap):
                    try:
                        self.grid_ims[e].remove()
                    except Exception:
                        pass
                    self.grid_ims[e] = None
                pm = plots.PixelMap(ax, m, self.extent, self.cmap, gamma=self.gammas[e], shift=shift, vmin=lo, vmax=hi,
                                    blend=self.blend if height is not None else None, height=height)
                self._pixel_maps[e] = pm
                if self.grid_ims[e] is None:
                    self._frame_limits(ax)
                    ax.set_aspect('equal', adjustable='box')
                    self._grid_title(ax, f"{self.V[e]:.2f} V")
                self.grid_ims[e] = pm
            elif self.grid_ims[e] is None:
                if height is not None:
                    self.grid_ims[e] = plots.BlendImage(ax, m, height, self.extent, self.blend, self.cmap,
                                                        gamma=self.gammas[e], vmin=lo, vmax=hi)
                else:
                    self.grid_ims[e] = plots.draw_heatmap(ax, m, self.extent, self.cmap, gamma=self.gammas[e],
                                                          vmin=lo, vmax=hi, origin='lower', aspect='equal',
                                                          interpolation='none')
                self._frame_limits(ax)
                self._grid_title(ax, f"{self.V[e]:.2f} V")
            else:
                if height is not None:
                    self.grid_ims[e].height = height                          # the topo drawn may have changed
                plots.update_heatmap(self.grid_ims[e], m, gamma=self.gammas[e], vmin=lo, vmax=hi)
            plots.draw_colorbar(self.fig, self.grid_ims[e], self.grid_caxes[e], labelsize=5)
        if self.height_on and self.view_mode == 'grid':
            self._draw_height_panels()

    GRID_TITLE_PT = 7      # the grid view's cell titles: one size, so every row of cells has one title band

    def _grid_title(self, ax, text):
        """A grid view cell's title ('Topo', or a map's energy '<bias> V'), one size for every cell (GRID_TITLE_PT): every row of cells then has the same title band (the row rule, Figure.align_row) and the same height."""
        ax.set_title(text, fontsize=self.GRID_TITLE_PT)

    @staticmethod
    def grid_shape(n_e):
        """The grid view's (map columns, rows) for n_e maps: about square, ceil(sqrt(n_e)) map columns, with at least two rows so that the curve panel under the topo cell has a row (then ceil(n_e / 2) map columns)."""
        k = int(np.ceil(np.sqrt(n_e)))
        r = int(np.ceil(n_e / k))
        if r < 2:
            r, k = 2, int(np.ceil(n_e / 2))
        return k, r

    def _grid_slot(self, e):
        """The grid view cell (row, column) of energy e's map: in order of energy, row by row, in the map columns right of the topo's column (column 0: the topo, the curve panel under it); with 'Height' on one place on, the first map position being the colour square's (_arrange_grid)."""
        k = int(e) + (1 if self.height_on else 0)
        return k // self._grid_cols, k % self._grid_cols + 1

    def _align_grid_rows(self):
        """Each row of the grid view's cells made one row (qtplot Figure.align_row): the maps (and in the first row the topo and Height shading's colour square) set the height, their colour bars take it."""
        for rr in range(self._grid_rows):
            members = [self.grid_topo_ax, self.grid_topo_cbar, self.key_grid.ax] if rr == 0 else []
            for e in range(self.n_e):
                if self._grid_slot(e)[0] == rr:
                    members += [self.grid_axes[e], self.grid_caxes[e]]
            self.fig.align_row(*members)

    def _arrange_grid(self):
        """
        The grid view's cells placed for the 'Height' state: off, the maps in order of energy from the
        first map position (grid_shape of n_e, as the window opens); on, Height shading's colour square
        in the first map position and the maps one place on (grid_shape of n_e + 1).  Every map's cell,
        its colour bar's cell and its γ box are taken out of the page's grid and put at their slots
        (_grid_slot), the column and row stretches set for the new shape, the rows made rows again
        (_align_grid_rows).  The curve panel's place follows at the next _place_curve (_apply_view).
        """
        gl = self._grid_layout
        for c in range(gl.columnCount()):
            gl.setColumnStretch(c, 0)
        for r in range(gl.rowCount()):
            gl.setRowStretch(r, 0)
        self._grid_cols, self._grid_rows = self.grid_shape(self.n_e + (1 if self.height_on else 0))
        gl.removeWidget(self._grid_square_cell)
        if self.height_on:
            gl.addWidget(self._grid_square_cell, 0, 2)       # the first map position: row 0, the first map column
        for e in range(self.n_e):
            rr, cc = self._grid_slot(e)
            for w in (self._grid_cells[e], self._grid_ccells[e], self.grid_gammas[e].qt):
                gl.removeWidget(w)
            gl.addWidget(self._grid_cells[e], 2 * rr, 2 * cc)
            gl.addWidget(self._grid_ccells[e], 2 * rr, 2 * cc + 1)
            gl.addWidget(self.grid_gammas[e].qt, 2 * rr + 1, 2 * cc, Qt.AlignmentFlag.AlignRight)
        for cc in range(self._grid_cols + 1):
            gl.setColumnStretch(2 * cc, 1)
        for rr in range(self._grid_rows):
            gl.setRowStretch(2 * rr, 1)
        self._align_grid_rows()

    def _place_curve(self):
        """The one curve panel in the view shown: in the single view right of the map's colour bar (the single view's row), in the grid view under the topo cell, as wide as the topo's column and as tall as the rows below it."""
        cell = self._cell_curve
        self._single_layout.removeWidget(cell)               # out of both first: placed again in the same view too (_apply_shading)
        self._grid_layout.removeWidget(cell)
        if self.view_mode == 'grid':
            self._grid_layout.addWidget(cell, 2, 0, 2 * (self._grid_rows - 1), 1)
        else:
            self._single_layout.addWidget(cell, 0, 5)

    def _on_channel(self, label):
        """The 'channel' radio: the maps are recomputed from the chosen channel; a file without it puts the radio back."""
        channel = {v: k for k, v in STS.CHANNEL_NAMES.items()}.get(label, 'didv')
        if channel == self.channel:
            return
        old = self.channel
        self.channel = channel
        before = self.maps
        self.recompute()
        if self.maps is before:                     # the file has no such channel: recompute printed why
            self.channel = old
            self.rad_channel.set(STS.CHANNEL_NAMES[old])

    def _on_sweeps(self, label):
        """
        The 'sweeps' radio: the maps made from the chosen sweep view of every pixel ('forward sweeps',
        'reverse sweeps', 'all sweeps', 'sweep n'); the pixel shifts (each sweep at its own time), the markers'
        curves, and the companion follow.
        """
        if label == self.grid.sweep_mode:
            return
        try:
            self.grid = self.grid.with_sweeps(label)
        except ValueError as exc:
            print(f"Map window: {exc}")
            self.rad_sweeps.set(self.grid.sweep_mode)
            return
        print(f"Map window: the maps show the {label} sweep(s) of every pixel.")
        self._view_memo = {}                       # the views by state: the grid changed
        self._drift_res = None
        self._bounds = None
        self._lines_memo = None
        self._clear_pixel_maps()
        self._draw_topo(self.ax_topo)
        self._grid_topo_stale = True
        self.recompute()
        self._redraw_markers()

    def _on_direction(self, label):
        """The 'topo pass' radio: the embedded topo redrawn from the chosen pass, or the two averaged ('average'), in both views; the box of what is drawn follows (the average sits on the forward placement alone); nothing without a topo page."""
        if label == self.direction or not self.grid.topo_pages:
            return
        self.direction = label
        self._bounds = None
        self._draw_topo(self.ax_topo)
        self._grid_topo_stale = True
        if self.view_mode == 'grid':
            self.draw_grid()
        self.draw_single()
        self.fig.canvas.draw_idle()

    def _frame_limits(self, ax):
        """Axis limits of the scan frame in normal orientation (the extent itself may be reversed); whenever a result places the pixels (a drift source on, or the stripe matching alone), the box of everything drawn, edge to edge (_drift_bounds)."""
        l, r, b, t = self._drift_bounds() if self._drift_result() is not None else self.extent
        ax.set_xlim(min(l, r), max(l, r))
        ax.set_ylim(min(b, t), max(b, t))

    def _apply_view(self):
        """Show the cells, axes and γ boxes of the current view (single or grid), hide the other's; the one curve panel moved into the view shown (_place_curve); the toggle button names the other view."""
        single = self.view_mode == 'single'
        self._place_curve()
        for cell in self._single_cells:
            cell.setVisible(single or cell is self._cell_curve)
        self._grid_page.setVisible(not single)
        self.gamma_single.show(single)
        for ax in (self.ax_topo, self.ax_map, self.cax_map, self.ax_topo_cbar):
            ax.set_visible(single)
        for ax in [self.grid_topo_ax, self.grid_topo_cbar] + self.grid_axes + self.grid_caxes:
            ax.set_visible(not single)
        self._height_vis = None                     # the loops above showed every colour bar of the view: set anew
        self._draw_height_panels()
        self.btn_view.set_text('Grid view' if single else 'Single view')
        self.fig.canvas.draw_idle()

    def toggle_view(self):
        """'Grid view' / 'Single view': the other view (the grid drawn when switched to), the markers and the contour slice drawn on its panels (_draw_marker_artists, _draw_contour_artists)."""
        self.view_mode = 'grid' if self.view_mode == 'single' else 'single'
        if self.view_mode == 'grid':
            self.draw_grid()
        self._draw_marker_artists()
        self._draw_contour_artists()
        self._apply_view()

    def step(self, d):
        """'< E' / 'E >': the single view's energy index moved by d (-1 or +1), clipped to the axis."""
        self.e_index = int(np.clip(self.e_index + d, 0, self.n_e - 1))
        self.draw_single()
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------ #
    #  Marker groups (the Line window's method in 2D: a marker is a pixel) #
    # ------------------------------------------------------------------ #

    def pixel_xy(self, px):
        """Centre of grid pixel (line, fast) in the map's frame (nm)."""
        l, r, b, t = self.extent
        line, fast = px
        return (l + (fast + 0.5) * (r - l) / self.n_fast, b + (line + 0.5) * (t - b) / self.n_lines)

    def pixel_diagonal(self):
        """A grid pixel's diagonal, corner to corner, in the frame (nm): the diameter of a marker (plots.draw_map_marker)."""
        l, r, b, t = self.extent
        return float(np.hypot((r - l) / self.n_fast, (t - b) / self.n_lines))

    def xy_pixel(self, x, y):
        """The grid pixel (line, fast) under a point of the frame, clipped to the grid."""
        l, r, b, t = self.extent
        fast = int(np.floor((x - l) / ((r - l) / self.n_fast)))
        line = int(np.floor((y - b) / ((t - b) / self.n_lines)))
        return (int(np.clip(line, 0, self.n_lines - 1)), int(np.clip(fast, 0, self.n_fast - 1)))

    def pixel_col(self, px):
        """Global column of the file's spectra for grid pixel (line, fast)."""
        return int(px[0]) * self.n_fast + int(px[1])

    def pixel_acquired(self, px):
        """Whether the spectrum of grid pixel (line, fast) is kept (STS.acquired; False past the recorded spectra)."""
        col = self.pixel_col(px)
        return bool(col < len(self.grid.acquired) and self.grid.acquired[col])

    def layout_centers(self, n):
        """n group centres (line, fast) evenly along the grid's diagonal from 10% to 90%."""
        if n <= 1:
            return [((self.n_lines - 1) / 2.0, (self.n_fast - 1) / 2.0)]
        fr = np.linspace(0.1, 0.9, n)
        return [(float(f * (self.n_lines - 1)), float(f * (self.n_fast - 1))) for f in fr]

    def place_group(self, center, m):
        """The m grid pixels nearest `center` (ties by angle), the 2D form of 'one step apart'."""
        cl, cf = center
        lines, fasts = np.meshgrid(np.arange(self.n_lines), np.arange(self.n_fast), indexing='ij')
        dl, df = lines.ravel() - cl, fasts.ravel() - cf
        d = np.hypot(dl, df)
        ang = np.arctan2(dl, df)
        order = np.lexsort((ang, d))
        m = int(min(max(0, m), lines.size))
        return [(int(lines.ravel()[i]), int(fasts.ravel()[i])) for i in order[:m]]

    def groups(self):
        """Marker indices per group (membership by index, controls.MarkerGroupControls.groups)."""
        return controls.MarkerGroupControls.groups(self.group_sizes)

    def group_of(self, i):
        """The group index of marker i."""
        return controls.MarkerGroupControls.group_of(self.group_sizes, i)

    DEFAULT_GROUP_SIZE = 2     # markers of a group added while there is no group to copy the size from

    def group_center(self, members, g=None):
        """The mean pixel (line, fast) of a group's markers; a group without markers keeps its last centre (group_ctr)."""
        if members:
            return (float(np.mean([self.marker_px[i][0] for i in members])),
                    float(np.mean([self.marker_px[i][1] for i in members])))
        ctr = getattr(self, 'group_ctr', [])
        if g is not None and g < len(ctr) and ctr[g] is not None:
            return tuple(ctr[g])
        return self.layout_centers(1)[0]

    def _sync_centres(self):
        """The stored group centres follow the groups that hold markers."""
        groups = self.groups()
        ctr = (list(getattr(self, 'group_ctr', [])) + [None] * len(groups))[:len(groups)]
        for g, members in enumerate(groups):
            ctr[g] = self.group_center(members, g)
        self.group_ctr = ctr
        return ctr

    def group_center_xy(self, members, g=None, on_map=True, e=None):
        """A group's centre in the frame (nm) as drawn: the mean of its markers' drawn positions (marker_xy, on a map at energy slice e, the shown energy when e is None, or on the topo), or pixel_xy of its kept centre without markers."""
        if members:
            pts = np.array([self.marker_xy(self.marker_px[i], on_map, e) for i in members], dtype=np.float64)
            return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
        return self.pixel_xy(self.group_center(members, g))

    def _marker_shift(self, on_map=True, e=None):
        """Every pixel's shift as drawn, (n_lines, n_fast, 2) in the frame (nm): on a map the shift of energy slice e (_frame_shift; the shown energy when e is None), on the topo the chained shift of the sweep's start; None without a result."""
        if on_map:
            return self._frame_shift(self.e_index if e is None else e)
        chain = self._chain_shift()
        if chain is None:
            return None
        g = self.grid
        n = g.n_lines * g.n_fast
        s = np.full((n, 2), np.nan)
        m = min(n, len(chain))
        s[:m] = chain[:m]
        return self._to_frame_shift(s).reshape(g.n_lines, g.n_fast, 2)

    def marker_xy(self, px, on_map=True, e=None):
        """Where grid pixel (line, fast) is drawn (nm): its centre plus its shift as drawn (_marker_shift: on a map at energy slice e, the shown energy when e is None), so that a marker sits on its pixel; the centre alone without a result."""
        x, y = self.pixel_xy(px)
        sh = self._marker_shift(on_map, e)
        if sh is not None:
            s = sh[int(px[0]), int(px[1])]
            if np.all(np.isfinite(s)):
                return (float(x + s[0]), float(y + s[1]))
        return (float(x), float(y))

    def xy_pixel_drawn(self, x, y, on_map=True, e=None):
        """The grid pixel drawn nearest a point of the frame (every pixel at its shift as drawn, _marker_shift: on a map at energy slice e, the shown energy when e is None); xy_pixel without a result."""
        sh = self._marker_shift(on_map, e)
        if sh is None:
            return self.xy_pixel(x, y)
        l, r, b, t = self.extent
        fx = l + (np.arange(self.n_fast) + 0.5) * (r - l) / self.n_fast
        fy = b + (np.arange(self.n_lines) + 0.5) * (t - b) / self.n_lines
        d = np.hypot(fx[None, :] + sh[:, :, 0] - x, fy[:, None] + sh[:, :, 1] - y)
        d = np.where(np.isfinite(d), d, np.inf)
        if not np.isfinite(d).any():
            return self.xy_pixel(x, y)
        k = int(np.argmin(d))
        return (int(k // self.n_fast), int(k % self.n_fast))

    def group_order(self):
        """Group indices by their centres' y on the frame, highest first (the order of the marker box's rows)."""
        groups = self.groups()
        return sorted(range(len(groups)), key=lambda g: (-self.group_center_xy(groups[g], g)[1], g))

    def marker_color(self, i):
        """Average off: one colour per marker; Average on: the group's colour."""
        k = self.group_of(i) if self.avg_on else i
        return self.M_COLORS[k % len(self.M_COLORS)]

    def group_label(self, g, members):
        """The group's legend label: 'avg <g+1>: (x, y) nm' of its centre, or 'no markers'."""
        if not members:
            return f"avg {g + 1}: no markers"
        cx, cy = self.group_center_xy(members, g)
        return f"avg {g + 1}: ({cx:.2f}, {cy:.2f}) nm"

    def _marker_panels(self):
        """
        The panels the markers are drawn on in the view shown, as (axes,
        on_map, e), e the energy slice of a map (None on a topo): the single
        view's map at the shown energy and its topo; the grid view's topo
        cell and every energy's map.  A file without a topo page has no
        topo panel.
        """
        topo = self.grid_topo_ax if self.view_mode == 'grid' else self.ax_topo
        panels = [(topo, False, None)] if self.has_topo else []
        if self.view_mode == 'grid':
            return panels + [(ax, True, e) for e, ax in enumerate(self.grid_axes)]
        return [(self.ax_map, True, self.e_index)] + panels

    def _draw_marker_artists(self):
        """
        The marker artists drawn anew (the old ones removed) on every panel
        of the view shown (_marker_panels): every marker as a solid circle in
        its colour, as wide as a pixel's diagonal, with a white edge ring
        (plots.draw_map_marker, pixel_diagonal), centred on its pixel as drawn in that panel
        (marker_xy: a map at its own energy slice's shift, a topo at the
        chain's), and with Average on
        every non-empty group's centre as a black X at the mean of its
        members' drawn positions.  Picking: on a map with Average on the
        centres are picked (the markers are not, so a group moves as one);
        elsewhere the markers are picked (a press inside the circle) and the
        centres never (an artist without a picker is unpickable).  Nothing
        is drawn before the first
        recompute (returns None); else returns the groups.
        """
        for art in self._marker_artists:
            try:
                art.remove()
            except (ValueError, NotImplementedError):
                pass
        self._marker_artists = []
        if self.maps is None:
            return None
        self._sync_centres()
        groups = self.groups()
        d = self.pixel_diagonal()
        for ax, on_map, e in self._marker_panels():
            pickable = not (self.avg_on and on_map)
            for i, px in enumerate(self.marker_px):
                h = plots.draw_map_marker(ax, self.marker_xy(px, on_map, e), self.marker_color(i), d,
                                          label=f'marker_{i}', pickable=pickable)
                self._marker_artists.append(h)
            if self.avg_on:
                avg_pick = {'picker': 6} if on_map else {}
                for g, members in enumerate(groups):
                    if not members:
                        continue                         # a group without markers has no centre mark
                    x, y = self.group_center_xy(members, g, on_map, e)
                    h, = ax.plot([x], [y], marker='X', ms=10, color='black', ls='none',
                                 label=f'_avg_{g}', **avg_pick)
                    self._marker_artists.append(h)
        return groups

    def _redraw_markers(self):
        """
        Redraw the marker artists on the panels of the view shown
        (_draw_marker_artists), then the curve panel, the slider stack and
        the companion.  Nothing is drawn before the first recompute.
        """
        groups = self._draw_marker_artists()
        if groups is None:
            return
        self._draw_curves(groups)
        self._refresh_group_sliders()
        self.fig.canvas.draw_idle()
        self._feenstra_refresh()

    def _draw_curves(self, groups):
        """The curve panel: one spectrum per marker, or one average per group (stacked when asked); plots.draw_marker_curves."""
        markers, group_items = self._curve_items(self.maps)
        plots.draw_marker_curves(self.ax_curve, self.V, markers, group_items, self.avg_on, self.stack_on,
                                 self.substack_on, self._curve_texts(), offset_frac=self.STACK_OFFSET_FRAC)
        self.ax_curve.title.set_fontsize(9)
        # One label between the map's colour bar and the curves: the bar's (the quantity and its units,
        # the curves' too); the curve panel carries none of its own.
        self.ax_curve.set_ylabel('')

    # --- the contour slice (the level line on the topo colour bars, the traces on the maps) ---

    def _topo_level(self):
        """
        The contour slice's source: (img, z, shift), the processed topo pass
        drawn on top in nm (the 'topo pass' radio's, the one the single
        view's topo shows: the last quad map drawn with a drift source on,
        the image kept by _draw_topo off), the level z in nm at z_frac of
        that pass's own colour range as drawn (the quad map's norm, the
        image's colour limits), and the pass's chained pixel shifts
        ((ny, nx, 2) in the frame; None without a drift result).  None
        without a topo page or before the topo is drawn.
        """
        if not self.has_topo:
            return None
        maps = self._topo_maps.get(id(self.ax_topo)) or []
        if maps:
            pm = maps[-1]                                   # the last drawn is on top (_draw_topo)
            img, shift, lo, hi = pm.img, pm.shift, pm.norm.vmin, pm.norm.vmax
        elif self._topo_im is not None:
            img, shift = self._topo_im.get_array(), None
            lo, hi = self._topo_im.get_clim()
        else:
            return None
        if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
            return None
        return img, float(lo + self.z_frac * (hi - lo)), shift

    def _contour_segments(self):
        """The slice's traces in the frame, (n, 2, 2): contour.iso_segments on the pass on top at the level of _topo_level, placed by contour.to_frame at the pixels as drawn (the extent, the pass's shifts); memoized until the level or the topo changes (_contour_memo); empty without a topo."""
        if self._contour_memo is None:
            src = self._topo_level()
            if src is None:
                self._contour_memo = np.zeros((0, 2, 2))
            else:
                img, z, shift = src
                self._contour_memo = contour.to_frame(contour.iso_segments(img, z), self.extent, img.shape, shift)
        return self._contour_memo

    def _contour_panels(self):
        """The map panels the contour slice is drawn on in the view shown: the single view's map, or every grid map (never a topo panel)."""
        return list(self.grid_axes) if self.view_mode == 'grid' else [self.ax_map]

    def _draw_contour_artists(self):
        """The contour slice drawn anew (the old traces removed) on every map panel of the view shown (_contour_panels; plots.draw_contour with the segments of _contour_segments); nothing before the first recompute, without a topo page or with 'Contour' off."""
        for art in self._contour_artists:
            try:
                art.remove()
            except (ValueError, NotImplementedError):
                pass
        self._contour_artists = []
        if self.maps is None or not self.has_topo or not self.contour_on:
            return
        segs = self._contour_segments()
        if not len(segs):
            return
        for ax in self._contour_panels():
            h = plots.draw_contour(ax, segs)
            if h is not None:
                self._contour_artists.append(h)

    def _draw_level_lines(self):
        """The contour slice's level line on each topo colour bar (the single view's and the grid view's; a bar spans 0 to 1, the fraction of the colour range): one pickable horizontal line per bar at z_frac, labelled '_zlevel' (_on_pick), drawn once; _set_level moves them."""
        self._level_lines = []
        for ax in (self.ax_topo_cbar, self.grid_topo_cbar):
            h = ax.axhline(self.z_frac, color=plots.CONTOUR_LEVEL_COLOR, lw=1.5, label='_zlevel', picker=6,
                           zorder=3.0)
            self._level_lines.append(h)

    def _set_level(self, frac):
        """The contour slice's level set to `frac` of the colour range (clipped to 0 .. 1): both level lines moved, the traces made anew (the memo dropped) and redrawn on the view shown."""
        frac = float(np.clip(frac, 0.0, 1.0))
        if frac == self.z_frac:
            return
        self.z_frac = frac
        for h in self._level_lines:
            h.set_ydata([frac, frac])
        self._contour_memo = None
        self._draw_contour_artists()
        self.fig.canvas.draw_idle()

    def _ramp_fraction(self, event, ax):
        """The cursor's height on the topo colour bar `ax` as the fraction of its range, from the event's canvas pixel row and the bar's drawn box (Axes.get_window_extent, both from the bottom): the drag follows the cursor's height even when it leaves the narrow bar sideways.  None without a position."""
        if event.y is None:
            return None
        x0, y0, w, h = ax.get_window_extent().bounds
        if not h:
            return None
        return (float(event.y) - y0) / h

    # --- Height shading (the heights under the maps' pixels, the colour square, the bars and the legends) ---

    def _heights(self, e):
        """
        The height fraction of the topography under every pixel of energy
        slice e's map, (n_lines, n_fast), for Height shading: the topo is
        the one the contour slice reads (_topo_level: the pass on top as
        the single view's topo draws it, in nm, with its chained pixel
        shifts), its heights as fractions of its own finite range
        (plots.height_fraction: 0 the lowest, 1 the highest); a map pixel
        takes the height of the topo pixel drawn nearest where the map
        pixel itself is drawn at that energy (its centre plus
        _frame_shift(e); height_source / heights_at), so with a drift
        source on the heights follow the shift within the sweep; a map
        pixel drawn outside the topography as drawn (more than one topo
        pixel pitch from the nearest drawn topo pixel) has no height (NaN)
        and is not drawn.  Without pixel shifts every energy has the same
        heights.  Memoized until the topo is drawn again (_height_memo,
        dropped by _draw_topo).  None without a topo page or before the
        topo is drawn.
        """
        if not self.has_topo:
            return None
        shift = self._frame_shift(e)
        key = int(e) if shift is not None else -1
        memo = self._height_memo
        if key in memo:
            return memo[key]
        src = self._topo_level()
        if src is None:
            return None
        img, _, tshift = src
        img = np.asarray(img, dtype=np.float64)
        if 'source' not in memo:
            centres = self._pixel_centres(img.shape)
            if tshift is not None and np.shape(tshift)[:2] == img.shape:
                centres = centres + np.where(np.isfinite(tshift), tshift, 0.0)     # as plots.PixelMap places the pixels
            l, r, b, t = self.extent
            pitch = (abs(r - l) / img.shape[1], abs(t - b) / img.shape[0])
            memo['source'] = height_source(plots.height_fraction(img), centres, pitch)
        at = self._pixel_centres((self.n_lines, self.n_fast))
        if shift is not None:
            at = at + np.where(np.isfinite(shift), shift, 0.0)
        memo[key] = heights_at(memo['source'], at)
        return memo[key]

    def _one_scale(self):
        """Whether one colour scale applies to every map of the grid view: 'cmap norm' off and every energy slice at the same γ (a slice's γ is part of its scale)."""
        return (not self.cmap_norm) and float(np.max(self.gammas)) == float(np.min(self.gammas))

    def _frame_aspect(self):
        """The shape of a map panel, its height over its width: of the limits _frame_limits sets (the frame, or the box of everything drawn with a placement on); None for an empty frame."""
        l, r, b, t = self._drift_bounds() if self._drift_result() is not None else self.extent
        w, h = abs(r - l), abs(t - b)
        return (h / w) if (w > 0 and h > 0) else None

    def _grid_legend(self, e):
        """The legend's axes under grid map e (attached under the map, qtplot Figure.attach: plots.BLEND_LEGEND tall, plots.BLEND_GAP under it), made at the first need."""
        ax = self._grid_legends.get(e)
        if ax is None:
            ax = self.fig.attach(self.grid_axes[e], 'bottom', plots.BLEND_LEGEND, plots.BLEND_GAP)
            self._grid_legends[e] = ax
        return ax

    def _draw_height_panels(self):
        """
        The panels of Height shading for the window's state, shown or
        hidden and (with 'Height' on) drawn for the view shown.  On: the
        colour square with the topo bar at its left and the value bar under
        it (plots.BlendKey): in the single view between the topo's ramp and
        the map, in the grid view in the first map position; a square has a
        map panel's shape (_frame_aspect); the topo bar carries the heights
        of the topo drawn in nm.  With one scale for every map shown
        ('cmap norm' off; in the grid view also every slice at the same γ,
        _one_scale) the value bar carries that scale with the maps'
        colour-bar label (_cbar_label) and the maps have no colour bar of
        their own; otherwise the value bar shows 'low' and 'high' and every
        map has its legend under it in place of its colour bar
        (plots.draw_blend_legend: the single view's attached under its map,
        a grid map's made at the first need, _grid_legend).  Off: none of
        it is shown and the maps' colour bars are as without it.  The
        visibility is set only when the state changed (_height_vis;
        _apply_view sets it anew).
        """
        single = self.view_mode == 'single'
        on = bool(self.height_on and self.has_topo and self.maps is not None)
        per_single = on and self.cmap_norm
        per_grid = on and not self._one_scale()
        state = (single, on, per_single, per_grid)
        if state != self._height_vis:
            self._height_vis = state
            self._cell_square.setVisible(single and on)
            self._single_layout.setColumnStretch(2, 28 if (single and on) else 0)
            self.key_single.set_visible(single and on)
            self.ax_map_legend.set_visible(single and per_single)
            self._cell_cbar.setVisible(single and not on)
            self.cax_map.set_visible(single and not on)
            self._grid_square_cell.setVisible(on)
            self.key_grid.set_visible((not single) and on)
            for e in range(self.n_e):
                self._grid_ccells[e].setVisible(not on)
                self.grid_caxes[e].set_visible((not single) and not on)
            for lg in self._grid_legends.values():
                lg.set_visible((not single) and per_grid)
        if not on:
            return
        src = self._topo_level()
        z_range = plots.finite_range(src[0]) if src is not None else (0.0, 1.0)
        aspect = self._frame_aspect()
        label = self._cbar_label()
        if single:
            im = self._single_im
            if im is None:
                return
            self.key_single.draw(self.blend, z_range, None if per_single else im.norm, None if per_single else label,
                                 aspect=aspect, labelsize=8)
            if per_single:
                plots.draw_blend_legend(self.ax_map_legend, self.blend, im.norm, label, labelsize=8)
        else:
            drawn = [im for im in self.grid_ims if im is not None]
            norm = drawn[0].norm if (drawn and not per_grid) else None
            self.key_grid.draw(self.blend, z_range, norm, label if norm is not None else None, aspect=aspect, labelsize=5)
            if per_grid:
                for e in range(self.n_e):
                    if self.grid_ims[e] is None:
                        continue
                    lg = self._grid_legend(e)
                    lg.set_visible(True)
                    plots.draw_blend_legend(lg, self.blend, self.grid_ims[e].norm, None, labelsize=5, nbins=3)

    # --- dragging ---

    def _on_pick(self, event):
        """
        A press on a marker ('marker_<i>'), a group centre ('_avg_<g>') or
        the contour slice's level line ('_zlevel', either topo colour bar):
        it becomes the dragged object.  Markers overlap (each is a pixel's
        diagonal across), so of the ones under the press the one whose
        centre is nearest it is taken; the press's offset from that centre
        is kept (_grab), so that the drag moves the centre and the centre
        picks the pixel (_on_motion).
        """
        artist = event.artist
        label = getattr(artist, 'get_label', lambda: '')()
        if not isinstance(label, str):
            return
        if label == '_zlevel':
            if not self.contour_on:
                return                                  # 'Contour' off: the level lines are hidden and take no press
            # The contour slice's level line: the drag follows the cursor's height on the bar it was picked on.
            self._active_obj = ('zlevel', artist.axes)
            self._pick_dist = 0.0
            self._grab = (0.0, 0.0)
            return
        if label.startswith('marker_'):
            idx = int(label.split('_')[1])
            obj = ('marker', idx) if idx < len(self.marker_px) else None
        elif label.startswith('_avg_'):
            g = int(label.split('_')[2])
            obj = ('avg', g) if g < len(self.group_sizes) else None
        else:
            obj = None
        if obj is None:
            return
        c = artist.get_center() if hasattr(artist, 'get_center') else (artist.get_xdata()[0], artist.get_ydata()[0])
        me = getattr(event, 'mouseevent', None)
        mx, my = (getattr(me, 'xdata', None), getattr(me, 'ydata', None)) if me is not None else (None, None)
        if mx is None or my is None:
            mx, my = c
        dist = float(np.hypot(mx - c[0], my - c[1]))
        if self._active_obj is not None and dist >= getattr(self, '_pick_dist', np.inf):
            return
        self._active_obj = obj
        self._pick_dist = dist
        self._grab = (float(mx - c[0]), float(my - c[1]))

    def _on_motion(self, event):
        """Drag (on the single view's map or topo, or the grid view's topo cell or any energy's map): the dragged centre (the cursor less the press's offset from it, _grab) picks the pixel drawn nearest it in that panel (a map at its own energy slice's shift, a topo at the chain's): a marker goes to that pixel; a group's centre moves every member by the same whole-pixel offset, clipped so no member leaves the grid.  The contour slice's level line follows the cursor's height on the colour bar it was picked on (_ramp_fraction, _set_level) while the button is down, wherever the cursor is sideways."""
        if self._active_obj is None:
            return
        if self._active_obj[0] == 'zlevel':
            frac = self._ramp_fraction(event, self._active_obj[1])
            if frac is not None:
                self._set_level(frac)
            return
        if event.xdata is None or event.ydata is None:
            return
        grid_e = next((e for e, ax in enumerate(self.grid_axes) if ax is event.inaxes), None)
        if event.inaxes is self.ax_map:
            on_map, e = True, None
        elif grid_e is not None:
            on_map, e = True, grid_e
        elif event.inaxes is self.ax_topo or event.inaxes is self.grid_topo_ax:
            on_map, e = False, None
        else:
            return
        kind, idx = self._active_obj
        gx, gy = getattr(self, '_grab', (0.0, 0.0))
        target = self.xy_pixel_drawn(event.xdata - gx, event.ydata - gy, on_map, e)   # the pixel drawn under the dragged centre
        if kind == 'marker':
            if self.marker_px[idx] == target:
                return
            self.marker_px[idx] = target
        else:
            members = self.groups()[idx]
            if not members:
                return
            cl, cf = self.group_center(members, idx)
            dl, df = int(round(target[0] - cl)), int(round(target[1] - cf))
            lines = [self.marker_px[i][0] for i in members]
            fasts = [self.marker_px[i][1] for i in members]
            dl = int(np.clip(dl, -min(lines), self.n_lines - 1 - max(lines)))
            df = int(np.clip(df, -min(fasts), self.n_fast - 1 - max(fasts)))
            if dl == 0 and df == 0:
                return
            for i in members:
                l0, f0 = self.marker_px[i]
                self.marker_px[i] = (l0 + dl, f0 + df)
        self._redraw_markers()

    def _on_release(self, event):
        """End of a drag."""
        self._active_obj = None
        self._grab = (0.0, 0.0)

    # --- marker boxes and toggles ---

    def _on_group_slider(self, g, m):
        """A group's 'Markers' box: that group gets the new number of pixels nearest its centre; the others stay."""
        if m == int(self.group_sizes[g]):
            return
        groups = self.groups()
        center = self.group_center(groups[g], g)
        new_px = []
        for gi, members in enumerate(groups):
            if gi == g:
                new_px.extend(self.place_group(center, m))
            else:
                new_px.extend(self.marker_px[i] for i in members)
        self.group_sizes[g] = m
        self.marker_px = new_px
        self._sync_centres()
        if m == 0:
            self.group_ctr[g] = center                  # an emptied group keeps its place
        self._redraw_markers()

    def _on_avgs_change(self, n):
        """'avgs': the number of groups.  Growing adds each new group with the last group's size (DEFAULT_GROUP_SIZE when there is none or it is empty) at the free slot of the diagonal layout farthest from the others (controls.MarkerGroupControls.grow_groups); shrinking drops the last groups with their markers."""
        groups = self.groups()
        if n == len(groups):
            return
        self._sync_centres()
        if n < len(groups):
            keep = sum(max(0, int(s)) for s in self.group_sizes[:n])
            self.marker_px = self.marker_px[:keep]
            self.group_sizes = self.group_sizes[:n]
            self.group_ctr = self.group_ctr[:n]
        else:
            m = int(self.group_sizes[-1]) if self.group_sizes else 0
            m = m if m > 0 else self.DEFAULT_GROUP_SIZE
            centers = [self.group_center(members, g) for g, members in enumerate(groups)]
            dist = lambda a, b: float(np.hypot(a[0] - b[0], a[1] - b[1]))
            for best in controls.MarkerGroupControls.grow_groups(n, centers, self.layout_centers(n), dist):
                self.marker_px.extend(self.place_group(best, m))
                self.group_sizes.append(m)
                self.group_ctr.append(tuple(best))
        self._redraw_markers()

    def _refresh_group_sliders(self):
        """One row of the marker box per group ('Markers' box framed in the group's colour, 'add new', 'add existing'), ordered as the groups' centres (highest y at the top); the layout lets the panels give way as the box grows (the box's height cap, __init__)."""
        self.groups_ctl.refresh(self.group_order(), self.group_sizes)

    # --- add to averages ---

    def _averages(self):
        """The averaging.Averages registry the 'add' buttons use: the caller's, else one made here at the first use on the file's folder (its new windows' Feenstra toggle on, with this window's broadening, settings, channel and fit_settings)."""
        if self.averages is None:
            self.averages = Averages(os.path.dirname(os.path.abspath(self.grid.path)),
                                     self.hampel_defaults, self.sgolay_defaults,
                                     fit_settings=self.fit_settings,
                                     divide_iv=True, iv_broadening=self.iv_broadening, channel=self.channel,
                                     iv_zero_sub=self.iv_zero_sub, iv_broadened=self.iv_broadened)
        return self.averages

    def group_selection(self, g):
        """The Selection of group g (its markers' columns; never-acquired pixels left out with a console line), or None."""
        members = self.groups()[g]
        cols, positions, skipped = [], [], []
        for i in members:
            px = self.marker_px[i]
            x, y = self.pixel_xy(px)
            if not self.pixel_acquired(px):
                skipped.append((x, y))
                continue
            cols.append(self.pixel_col(px))
            positions.append([float(x), float(y)])
        if skipped:
            print("add: marker(s) at " + ", ".join(f"({x:.2f}, {y:.2f})" for x, y in skipped)
                  + " nm sit on never-acquired spectra and are left out.")
        if not cols:
            return None
        return Selection(self.grid, cols, {'line': None, 'positions': positions})

    def _group_selection_checked(self, g):
        """The Selection of a group, or None (the reason printed)."""
        try:
            selection = self.group_selection(g)
        except Exception as exc:
            print(f"add: {exc}")
            return None
        if selection is None:
            print(f"add: avg {g + 1} has no marker on an acquired spectrum; nothing added.")
        return selection

    def _picker(self):
        """The pop-up list of the open averages (controls.AveragePicker), made at the first use."""
        if self._avg_picker is None:
            self._avg_picker = controls.AveragePicker(self.fig, self._averages())
        return self._avg_picker

    def _on_add_new(self, g):
        """'add new' of a marker-box row: that group starts a new average (each marker one sweep)."""
        self._picker().close()
        selection = self._group_selection_checked(g)
        if selection is None:
            return
        try:
            w = self._averages().add('', selection, channel=self.channel)
        except Exception as exc:
            print(f"add new: {exc}")
            return
        self._avg_windows.append(w)
        self.fig.canvas.draw_idle()

    def _on_add_existing(self, g, anchor_ax):
        """'add existing' of a marker-box row: the list of the open averages; a row picked adds the group to it."""
        picker = self._picker()
        if picker.is_open():
            picker.close()
            return
        selection = self._group_selection_checked(g)
        if selection is None:
            return

        def add_to(index):
            w = self._averages().open().get(index)
            if w is None:
                return
            if w.has(selection):
                print(f"{w.name} already holds {selection.entry_label} ({selection.name}).")
                return
            try:
                w.add(selection)
            except Exception as exc:
                print(f"add to {w.name}: {exc}")
                return
            print(f"{w.name}: added {selection.entry_label} ({selection.name}; {len(w.entries)} entries).")
            self._avg_windows.append(w)
            self.fig.canvas.draw_idle()

        picker.open(anchor_ax, selection, add_to)

    # --- the whole map into an average ('map → new avg', 'map → existing avg') ---

    def map_selection(self):
        """
        The whole map as one entry of an averaging window: every acquired
        pixel's spectrum as one sweep (averaging.Selection of those columns,
        its positions the pixel centres in the frame, 'map' True); None
        without an acquired pixel (the console says so).
        """
        n = self.n_lines * self.n_fast
        acq = np.asarray(self.grid.acquired, dtype=bool)[:n]
        cols = np.flatnonzero(acq)
        if not len(cols):
            print("map → avg: no acquired pixel in this map.")
            return None
        positions = []
        for c in cols:
            x, y = self.pixel_xy(divmod(int(c), self.n_fast))
            positions.append([float(x), float(y)])
        return Selection(self.grid, cols, {'line': None, 'positions': positions, 'map': True})

    def _on_map_new(self):
        """'map → new avg': the whole map starts a new averaging window (one entry, every acquired pixel one sweep)."""
        self._picker().close()
        selection = self.map_selection()
        if selection is None:
            return
        try:
            w = self._averages().add('', selection, channel=self.channel)
        except Exception as exc:
            print(f"map → new avg: {exc}")
            return
        self._avg_windows.append(w)
        self.fig.canvas.draw_idle()

    def _on_map_existing(self):
        """'map → existing avg': the list of the open averages; a row picked adds the whole map to it."""
        picker = self._picker()
        if picker.is_open():
            picker.close()
            return
        selection = self.map_selection()
        if selection is None:
            return

        def add_to(index):
            w = self._averages().open().get(index)
            if w is None:
                return
            if w.has(selection):
                print(f"{w.name} already holds {selection.entry_label} ({selection.name}).")
                return
            try:
                w.add(selection)
            except Exception as exc:
                print(f"map → {w.name}: {exc}")
                return
            print(f"{w.name}: added {selection.entry_label} ({selection.name}; {len(w.entries)} entries).")
            self._avg_windows.append(w)
            self.fig.canvas.draw_idle()

        picker.open(self.btn_map_exist.ax, selection, add_to)

    # --- the figure composer ('add to plot' x 3, 'new plot') ---

    # --- the record (averages/<file stem>_map_markers.json) ---

    def marker_record(self):
        """
        The marker record of this map (records.MAP_RECORD_FORMAT), every
        value JSON-plain: the file (path and name), the scale factor the
        window was made with, the grid's sweep view, the pipeline state
        (sts.state_record: the channel, the filter boxes and their four on /
        off states, calibrate, normalize, the gate with the locks and
        denoise, divide_iv, the Feenstra settings), the view ('single' or
        'grid'), the energy step (index, and the energy for information),
        the gammas (one per slice), 'cmap norm' and 'clip %', the topo pass
        and the processor flags, the drift toggles and the stripe toggle,
        the 'Height' and 'Contour' toggles,
        the contour slice's level as the fraction of the drawn topo's colour
        range (z_frac; the slice is rebuilt from it at load) with the level
        in metres and the traces themselves in nm for information (the
        load compares the rebuilt slice with them and says when they
        differ), the group sizes, their centres as (line, fast) and the
        markers as pixels (with their positions in nm for information),
        the Average / stack / substack states, and the time of writing.
        """
        g = self.grid
        level = self._topo_level()
        segs = self._contour_segments() if (self.has_topo and self.maps is not None) else np.zeros((0, 2, 2))
        segs = np.asarray(segs, dtype=np.float64).reshape(-1, 2, 2)
        centres = self._sync_centres()
        return {'format': records.MAP_RECORD_FORMAT,
                'path': os.path.abspath(g.path),
                'name': g.name,
                'scalefactor': float(self.scalefactor),
                'sweeps': getattr(g, 'sweep_mode', None),
                **state_record(self.state, self.filters.values()),
                'view': self.view_mode,
                'e_index': int(self.e_index),
                'energy_V': float(self.V[self.e_index]),
                'gammas': [float(v) for v in self.gammas],
                'cmap_norm': bool(self.cmap_norm),
                'clip_percent': float(self.clip_percent),
                'direction': self.direction,
                'active_procs': [bool(a) for a in self.active_procs],
                'auto_drift': bool(self.auto_on),
                'manual_drift': bool(self.manual_on),
                'stripe': bool(self.chk_stripe.value),
                'z_frac': float(self.z_frac),
                'height': bool(self.height_on),
                'contour': bool(self.contour_on),
                'contour_level_m': None if level is None else float(level[1]) * 1e-9,
                'contour_segments_nm': [[[float(x), float(y)] for x, y in seg] for seg in segs],
                'group_sizes': [int(v) for v in self.group_sizes],
                'group_centers': [[float(c[0]), float(c[1])] for c in centres],
                'markers': [[int(px[0]), int(px[1])] for px in self.marker_px],
                'markers_nm': [[float(x), float(y)] for x, y in (self.marker_xy(px, True) for px in self.marker_px)],
                'average': bool(self.avg_on),
                'stack': bool(self.stack_on),
                'substack': bool(self.substack_on),
                'saved': datetime.now().isoformat(timespec='seconds')}

    def _on_save(self):
        """'save': the marker record averages/<file stem>_map_markers.json (one per file, overwritten; the sweep view inside), its path printed."""
        try:
            directory = os.path.dirname(os.path.abspath(self.grid.path))
            path = records.write_json(records.map_record_path(directory, self.grid.path), self.marker_record())
        except Exception as exc:
            print(f"save: {exc}")
            return
        print(f"Map window: marker record written: {path}")

    def restore_markers(self, rec):
        """
        Put a marker record's state into the open window (the browser's
        loader, browser.load_map_record, after the window is opened on the
        record's file and sweep view): the pipeline state through
        sts.state_from_record into the filter box and the pipeline box
        (without firing) and the window's attributes; the view, the energy
        index (refused with a console line when outside the file's axis),
        the gammas (their count checked), 'cmap norm', 'clip %', the topo
        pass, the processor flags, the drift toggles and the stripe toggle
        (through their own handlers, so the drift model's one-state-per-file
        rules hold; refused as they refuse without a drift model), the
        contour slice's level, the group sizes, centres and markers (clipped
        to the grid; a record whose group sizes and markers disagree keeps
        the window's markers), the stack controls, the 'Height' and
        'Contour' toggles (a record without them, format 1, leaves Height
        off and Contour on; _apply_shading); then the maps recomputed,
        the topo, the markers, the contour slice and the companion drawn
        anew, and the rebuilt slice compared with the record's traces (a
        console line when they differ: the topo's pass, processing or drift
        state is not what it was at the save).  Every part that cannot be
        applied is reported and skipped.
        """
        st = state_from_record(rec, self.state)
        hp, sp, hd, sd = st.filters()
        try:
            self.filters.set_params(hp, sp, hd, sd)
            for store, d in ((self.filters._hp, rec.get('hampel')), (self.filters._sp, rec.get('sgolay'))):
                if isinstance(d, dict):
                    store.update(d)                       # the box values, as the Line window keeps them
        except Exception as exc:
            print(f"restore markers: filter settings not applied ({exc}).")
        self.calibrate, self.normalize, self.gate = st.calibrate, st.normalize, st.gate
        self.pipeline.set_state(self.calibrate, self.normalize, self.gate)
        self.divide_iv = st.divide_iv
        self.chk_feenstra.set(self.divide_iv)
        self.iv_zero_sub, self.iv_broadened, self.iv_broadening = st.iv_zero_sub, st.iv_broadened, st.iv_broadening
        if st.channel != self.channel:
            if st.channel in self.grid.channels():
                self.channel = st.channel
                self.rad_channel.set(STS.CHANNEL_NAMES[st.channel])
            else:
                print(f"restore markers: {self.grid.name} has no {st.channel} channel; the window keeps {self.channel}.")
        want_view = rec.get('view', self.view_mode)
        if want_view not in ('single', 'grid'):
            print(f"restore markers: the record's view '{want_view}' is not 'single' or 'grid'; the window keeps its own.")
            want_view = self.view_mode
        e = rec.get('e_index')
        if e is not None:
            if 0 <= int(e) < self.n_e:
                self.e_index = int(e)
            else:
                print(f"restore markers: the record's energy index {e} is outside the file's {self.n_e} energies; "
                      f"the window keeps {self.e_index}.")
        gm = rec.get('gammas')
        if gm is not None:
            gm = np.asarray(gm, dtype=np.float64).ravel()
            if gm.size == self.n_e and np.all(np.isfinite(gm)) and np.all(gm > 0):
                self.gammas = gm.copy()
                for box, v in zip(self.grid_gammas, self.gammas):
                    box.set(v)
                self.gamma_single.set(self.gammas[self.e_index])
            else:
                print(f"restore markers: the record's gammas ({gm.size}) do not fit the file's {self.n_e} energies, "
                      "or are not positive; the window keeps its own.")
        if 'cmap_norm' in rec:
            self.cmap_norm = bool(rec['cmap_norm'])
            self.chk_cmap_norm.set(self.cmap_norm)
        cp = rec.get('clip_percent')
        if cp is not None:
            try:
                cp = float(cp)
                if not (np.isfinite(cp) and 0.0 <= cp < 50.0):
                    raise ValueError(cp)
                self.clip_percent = cp
                self.box_clip.set(f"{cp:g}")
            except (TypeError, ValueError):
                print(f"restore markers: the record's clip % {cp!r} is not a percentage from 0 up to 50; the window keeps "
                      f"{self.clip_percent:g}.")
        d = rec.get('direction')
        if d is not None:
            if d in TOPO_DIRECTIONS and self.grid.topo_pages:
                self.direction = d
                self.rad_dir.set(d)
            elif d not in TOPO_DIRECTIONS:
                print(f"restore markers: the record's topo pass '{d}' is unknown; the window keeps {self.direction}.")
        ap = rec.get('active_procs')
        if ap is not None:
            if len(ap) == len(self.active_procs):
                self.active_procs = [bool(a) for a in ap]
            else:
                print(f"restore markers: the record's {len(ap)} processor flags do not fit the {len(self.active_procs)} "
                      "processors; the window keeps its own.")
        for which, key, chk in (('auto', 'auto_drift', self.chk_auto), ('manual', 'manual_drift', self.chk_manual)):
            if key in rec:
                self._on_drift_toggle(which, bool(rec[key]))
                chk.set(self.auto_on if which == 'auto' else self.manual_on)
        if 'stripe' in rec and bool(rec['stripe']) != bool(self.chk_stripe.value):
            self._on_stripe_toggle(bool(rec['stripe']))
        zf = rec.get('z_frac')
        if zf is not None:
            try:
                zf = float(zf)
                if not np.isfinite(zf):
                    raise ValueError(zf)
                self.z_frac = float(np.clip(zf, 0.0, 1.0))
                for h in self._level_lines:
                    h.set_ydata([self.z_frac, self.z_frac])
            except (TypeError, ValueError):
                print(f"restore markers: the record's z_frac {zf!r} is not a number; the window keeps {self.z_frac:g}.")
        sizes = [max(0, int(v)) for v in rec.get('group_sizes', [])]
        markers = [(int(px[0]), int(px[1])) for px in rec.get('markers', [])]
        if 'group_sizes' in rec and sum(sizes) == len(markers) and len(sizes) <= self.MAX_GROUPS:
            self.group_sizes = sizes
            self.marker_px = [(int(np.clip(line, 0, self.n_lines - 1)), int(np.clip(fast, 0, self.n_fast - 1)))
                              for line, fast in markers]
            ctr = rec.get('group_centers')
            self.group_ctr = ([(float(c[0]), float(c[1])) for c in ctr] if ctr is not None and len(ctr) == len(sizes)
                              else [None] * len(sizes))
            self._sync_centres()
        elif sizes or markers:
            print("restore markers: the group sizes and the marker positions do not match; markers kept.")
        self.stacks.chk_average.set(bool(rec.get('average', False)))
        self.stacks.chk_stack.set(bool(rec.get('stack', False)))
        self.stacks.chk_substack.set(bool(rec.get('substack', False)))
        # The window drawn anew: the maps for the state, the topo for the pass and the processors, the
        # view, the markers, the contour slice and the companion.
        self._view_memo = {}
        self._bounds = None
        self._lines_memo = None
        self._clear_pixel_maps()
        self._contour_memo = None
        self._draw_topo(self.ax_topo)
        self._grid_topo_stale = True
        before = self.maps
        self.recompute()
        if self.maps is before:
            print("restore markers: the record's pipeline state could not be applied; the window keeps its own.")
        if want_view != self.view_mode:
            self.toggle_view()
        if 'height' in rec or 'contour' in rec:
            # The 'Height' and 'Contour' toggles (a format 1 record has neither key: the window keeps
            # Height off and Contour on); never both on, and no Height without a topo page.
            self.height_on = bool(rec.get('height', False)) and self.has_topo
            self.contour_on = bool(rec.get('contour', True)) and not self.height_on
            self.chk_height.set(self.height_on)
            self.chk_contour.set(self.contour_on)
            self._apply_shading()
        self._redraw_markers()
        saved = rec.get('contour_segments_nm')
        if saved is not None and self.has_topo and self.maps is not None:
            segs = np.asarray(self._contour_segments(), dtype=np.float64).reshape(-1, 2, 2)
            sv = np.asarray(saved, dtype=np.float64).reshape(-1, 2, 2)
            same = segs.shape == sv.shape and (segs.size == 0 or np.allclose(segs, sv, rtol=1e-9, atol=1e-9))
            if not same:
                print(f"restore markers: the contour slice rebuilt from the record ({len(segs)} traces) differs from the "
                      f"saved one ({len(sv)} traces): the topo's pass, processing or drift state is not what it was at "
                      "the save.")

    def _composer(self):
        """The window's figure composer (composer.FigureComposer, titled '<file>: figure'), made at the first use."""
        if self.composer is None:
            self.composer = FigureComposer(f"{self.grid.name}: figure")
        return self.composer

    def _on_add_to_plot(self, which):
        """
        'add to plot' under a panel: that panel as shown goes to the figure
        composer (composer.FigureComposer).  'map': the shown slice with its
        colour limits, gamma, the axes limits and its drift shift, at its
        bias, and with 'Height' on its shading (_shading_view: the blend,
        the heights under its pixels, the topo's heights and the map's
        colour-bar label for the composer's colour square);
        'topo': the embedded topo as drawn (the quads with their
        shifts with a drift source on); 'spectra': every marker's spectrum
        (Average off) or every group's average (Average on), each with the
        marker's pixel centre or the group's centre, and the curve panel's
        limits (the curve's own finite range with Average and stack on,
        since the panel then shows offset curves).  A refusal (no maps yet,
        no topo page) is printed as 'add to plot: ...'.
        """
        c = self._composer()
        try:
            if self.maps is None:
                raise ValueError("no maps yet")
            if which == 'map':
                e = self.e_index
                im = self._single_im
                vmin, vmax = (im.norm.vmin, im.norm.vmax) if im is not None else (None, None)
                shading = self._shading_view()
                c.add_map(self.maps[e], self.extent, [*self.ax_map.get_xlim(), *self.ax_map.get_ylim()],
                          vmin, vmax, self.gammas[e], self.cmap, self.V[e], shift=self._frame_shift(e),
                          blend=shading['blend'], height=shading['height'], z_range=shading['z_range'],
                          label=self._cbar_label())
            elif which == 'topo':
                if not self.has_topo:
                    raise ValueError("no topography page in this file")
                quads = self._topo_maps.get(id(self.ax_topo)) or []
                if quads:
                    layers = [(pm.img, pm.shift) for pm in quads]
                else:
                    layers = [(processed_topo(self.grid, self.direction, self.active_procs) * 1e9, None)]
                c.add_topo(layers, self.extent, [*self.ax_topo.get_xlim(), *self.ax_topo.get_ylim()])
            else:
                markers, groups = self._curve_items(self.maps)
                xlim = self.ax_curve.get_xlim()
                q = self._quantity().split(',')[0]

                def lim_of(curve):
                    if self.stack_on and self.avg_on:
                        ok = np.isfinite(curve)
                        return (float(np.min(curve[ok])), float(np.max(curve[ok]))) if ok.any() else (0.0, 1.0)
                    return self.ax_curve.get_ylim()

                if self.avg_on:
                    for g, item in enumerate(groups):
                        if not item['members']:
                            continue
                        c.add_spectrum(self.V, item['average'], xlim, lim_of(item['average']),
                                       self.group_center_xy(item['members'], g), q)
                else:
                    for i, item in enumerate(markers):
                        c.add_spectrum(self.V, item['curve'], xlim, lim_of(item['curve']),
                                       self.marker_xy(self.marker_px[i], True), q)
        except Exception as exc:
            print(f"add to plot: {exc}")

    def _on_new_plot(self):
        """'new plot': the composer's list emptied (the figure, if open, shows nothing)."""
        if self.composer is not None:
            self.composer.clear()

    # --- send to fit ---

    def _send_to_fit(self):
        """
        Send the marker spectra (Average off: one fit per marker) or
        the group averages (Average on: one per group) to the single-point
        fit, as the Line window does: the grid's raw columns at the
        markers through the pipeline with the window's filters, calibration
        and normalization on (a console line says so when the window's boxes
        are off), divided by I/V when Feenstra is on (quantity 'ratio'); the
        unfiltered counterpart goes along for the noise estimate, the raw
        columns for refits.  dI/dV only; needs the Current channel.  Each
        send is labelled '<file path> | marker at (x, y) nm' or '... |
        average of n markers at ...' (' over I/V' with Feenstra on, so the
        ratio's fit histories are separate); with fit_settings 'use_cache'
        on an up-to-date fit history of that label opens instead of a fresh
        model selection (Fit.has_cache / from_cache).  A refusal is printed
        as 'Send to fit: ...'.
        """
        try:
            self._send()
        except Exception as exc:
            print(f"Send to fit: {exc}")

    def _send(self):
        """The send of _send_to_fit; raises on a refusal (IETS shown, no Current channel)."""
        g = self.grid
        if self.channel != 'didv':
            raise ValueError("the fit models dI/dV; the window shows IETS.")
        if g.iv is None:
            raise ValueError("no 'Current' spectroscopy page in this file.")
        switched = [n for n, on in (('calibration', self.calibrate), ('normalization', self.normalize)) if not on]
        if switched:
            print("Send to fit: " + " and ".join(switched) + " switched on for the sent curves.")
        hp, sp = self.filters.params()
        hd, sd = self.filters.params_didv()
        div = self.divide_iv
        quantity = 'ratio' if div else 'didv'
        suffix = " over I/V" if div else ""
        energy = g.energy
        V = np.sort(energy)
        e_order = np.argsort(energy)
        iv_sorted = np.asarray(g.iv, dtype=np.float64)[e_order, :]
        i_ref = int(np.argmin(np.abs(V)))
        i_bias = int(np.argmin(np.abs(V - g.V_bias)))

        def refs(cols):
            # The forward-sweep reference of these pixels (a reverse-sweep view), None otherwise.
            if g.lia_ref is None or g.iv_ref is None:
                return None, None
            return g.lia_ref[:, cols], g.iv_ref[:, cols]

        def processed(cols, hampel, sgolay, hampel_didv, sgolay_didv):
            # The pipeline on the raw columns with calibrate and normalize at their defaults (both on: the
            # fit's pipeline), energy ascending; areas: the normalization areas per column (amps).
            lia = g.lia[:, cols]
            iv = g.iv[:, cols]
            c, areas = STS.process_arrays(energy, lia, iv, hampel, sgolay, div, self.iv_width(),
                                          return_areas=True, channel='didv', iv_zero_sub=self.iv_zero_sub,
                                          vmod=g.vmod, V_bias=g.V_bias, I_setpoint=g.I_setpoint, ref=refs(cols),
                                          hampel_didv=hampel_didv, sgolay_didv=sgolay_didv, name=g.name)
            return c[e_order, :], areas

        sends = []
        groups = self.groups() if self.avg_on else [[i] for i in range(len(self.marker_px))]
        for members in groups:
            cols, places, skipped = [], [], []
            for i in members:
                px = self.marker_px[i]
                x, y = self.pixel_xy(px)
                if not self.pixel_acquired(px):
                    skipped.append((x, y))
                    continue
                cols.append(self.pixel_col(px))
                places.append((x, y))
            if skipped:
                print("Send to fit: marker(s) at " + ", ".join(f"({x:.2f}, {y:.2f})" for x, y in skipped)
                      + " nm sit on never-acquired spectra and are left out.")
            if not cols:
                continue
            cf, areas = processed(cols, hp, sp, hd, sd)
            cu, _ = processed(cols, None, None, None, None)
            areas_A = np.asarray(areas, dtype=np.float64)
            delta_I = iv_sorted[i_bias, cols] - iv_sorted[i_ref, cols]
            where = ", ".join(f"({x:.2f}, {y:.2f})" for x, y in places)
            if self.avg_on:
                data = Averages.mean_rows(cf.T)
                unf = Averages.mean_rows(cu.T)
                label = f"{g.path} | average of {len(cols)} markers at {where} nm{suffix}"
                stage = 'average'
            else:
                data, unf = cf[:, 0], cu[:, 0]
                label = f"{g.path} | marker at {where} nm{suffix}"
                stage = 'column'
            sends.append((data, unf, areas_A, delta_I, label, g.lia[:, cols], g.iv[:, cols], stage, refs(cols)))

        settings = dict(self.fit_settings)
        use_cache = settings.pop('use_cache', True)
        for data, unf, areas_A, delta_I, label, lia_raw, iv_raw, stage, (lia_ref, iv_ref) in sends:
            print("\n" + "=" * 70)
            if use_cache and fit.Fit.has_cache(label):
                an = fit.Fit.from_cache(fit.Fit.cache_path(label))
            else:
                an = fit.Fit.from_curve(V, data, unf, g.V_bias, g.I_setpoint, areas_A, label,
                                        hampel_filter_params=hp, sgolay_filter_params=sp,
                                        hampel_didv_params=hd, sgolay_didv_params=sd,
                                        delta_I_window_A=delta_I,
                                        lia_raw=lia_raw, iv_raw=iv_raw, lia_ref=lia_ref, iv_ref=iv_ref,
                                        units_lia=g.units_lia, units_iv=g.units_iv, vmod=g.vmod,
                                        energy_raw=energy,
                                        filter_stage=stage,
                                        quantity=quantity, iv_broadening=self.iv_width(),
                                        iv_zero_sub=self.iv_zero_sub)
                an.select_model(**settings)
            an.plot()
            self._fits.append(an)



if __name__ == '__main__':
    from sm4_Browser import sm4file
    path = r'experiment_dir/file.sm4'
    # Pre-fill of the Hampel / Savitzky-Golay boxes (filters start off), as in the browser block.
    hampel_defaults = {'window': 4, 'n_sigma': 3}
    sgolay_defaults = {'window_length': 6, 'polyorder': 3}
    # The map in the browser's start state: Slope Sub off, Δz sub on (one flag per processor),
    # Feenstra on with 0.2 V broadening.
    window = Map(sm4file.Sm4File(path), [False, True], 2.75, hampel_defaults, sgolay_defaults,
                 divide_iv=True, iv_broadening=0.2)
