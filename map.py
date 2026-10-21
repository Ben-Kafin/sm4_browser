"""
Per-energy maps of a grid file: the Map window (a tiled grid of every energy,
or one energy beside the file's embedded topography with energy arrows and a
curve panel), its marker groups (the Line window's method in 2D), its own
'Feenstra' toggle and Feenstra companion, the 'calibrate' / 'normalize' box,
the 'cmap norm' colour-scale toggle, and one 'γ' box per energy slice.
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
import numpy as np
from sm4_Browser import qtplot as plt
from sm4_Browser import fit, widgets, plots
from sm4_Browser.sts import STS, curve_units
from sm4_Browser.averaging import Averages, Selection
from sm4_Browser.feenstra import (FeenstraWindow, feenstra_width, feenstra_titles, energy_has_zero, iv_corrected,
                                  iv_broadened)
from sm4_Browser.normalization import normalize_abs
from sm4_Browser.drift import DriftWindow
from sm4_Browser.composer import FigureComposer
from sm4_Browser.topo import processed_topo, process_topo, TOPO_CMAP


class Map:
    """
    Grid file: a toggle ('Grid view' / 'Single view') between (a) a tiled
    grid of the embedded topography (first cell) and every energy, and (b)
    one energy beside the file's own embedded topography with energy arrows
    ('< E', 'E >') and a curve panel.  Maps are the channel (dI/dV, or IETS
    by the 'channel' radio) through the pipeline (Sm4File.curves,
    STS.process_arrays: the filter box's Hampel and Savitzky-Golay filters
    on the current, the division by the broadened I/V with 'Feenstra' on),
    every location calibrated against its own I(V) and unit-area normalized
    as the 'calibrate' / 'normalize' box says (both off at open), energy
    ascending, reshaped to (n_lines, n_fast) in the scan frame (self.extent:
    the recorded frame in nm after the scale factor; a never-acquired pixel
    is NaN and drawn transparent).  Filter controls at the bottom.  The
    'topo pass' radio picks the pass of the embedded topo ('average' counts
    as forward: a grid's passes are never averaged); the 'sweeps' radio of a
    file with several sweeps per pixel picks the sweep view the maps are
    made of (Sm4File.with_sweeps).

    Colour scale: with 'cmap norm' off (the default) every energy slice is
    coloured on one scale, the minimum and maximum over the whole cube, in
    the single view and in every grid map; on, each map has its own minimum
    and maximum.  Every energy slice has its own 'γ' (the power-law exponent
    of the colour map, plots.heat_norm): one box under each grid map, and the
    single view's box edits the current slice's value.

    Markers (the single view): the Line window's marker method in 2D.  A
    marker is a grid pixel (line, fast); it is drawn as a small square at the
    pixel's centre on the map and on the embedded topo.  The 'avgs' slider
    (0 to MAX_GROUPS) is the number of groups; every group has its own
    'Markers' slider (0 to 10 markers; a group at 0 keeps its centre) with
    'add new' and 'add existing' beside it, the sliders stacked bottom to
    top in the order of the groups' centres (highest y at the top), each
    filled with its group's colour; the panels move up when the stack would
    otherwise run into them.  Group membership is by index.  A group's m
    markers are the m pixels nearest its centre (ties by angle); new groups
    start at the free slot of an even layout along the grid's diagonal with
    the last group's size; fewer groups drops the last groups.  The window
    has its own 'Feenstra' toggle (starting at the caller's state,
    `divide_iv`: the browser opens it on; independent of the browser's
    after; the browser hands over the Feenstra settings) and a Feenstra
    companion window (feenstra.FeenstraWindow) that opens with it and
    follows it: the map at the current energy in four versions with the
    marker curves under each.  Dragging: a press on a
    marker (either panel) moves it to the pixel under the cursor; with
    Average on, a press on a group's centre (a black X on the map) moves the
    whole group by the same pixel offset, clipped so no member leaves the
    grid.  The curve panel shows one spectrum per marker (from the cube: the
    window's filters, flags, Feenstra, and channel), or with Average on one
    average per group (the plain mean, Averages.mean_rows, a marker on a
    never-acquired pixel left out), with 'stack' and 'substack' as in the
    Line window.  'add new' and 'add existing' beside each 'Markers' slider
    put that group into the averages as the browser's buttons add a single
    point: a Selection of its markers' columns (each marker one sweep) goes
    to a new average, or to the open average picked from the pop-up list
    (widgets.AveragePicker).  'Send to fit' sends the marker spectra
    (Average off) or the group averages (Average on) to the single-point
    fit, as the Line window does (calibrated and normalized, the
    fit's pipeline).  'map → new avg' / 'map → existing avg' put the whole
    map (every acquired pixel's spectrum as one sweep of one entry) into a
    new or an open averaging window.  'add to plot' under the topo, the
    map and the curve panel sends that panel as shown to the figure
    composer (composer.FigureComposer, one per window); 'new plot' empties
    it.  One topo colour bar sits right of the embedded topo.  The drift
    toggles and the drift window are described at __init__.
    """

    MAX_GROUPS = 10
    STACK_OFFSET_FRAC = 0.25
    # Panels of the single view (figure fractions); the bottom rises with the slider stack.
    PANEL_BOTTOM_MIN, PANEL_H_MAX = 0.17, 0.75
    PANEL_TOP = PANEL_BOTTOM_MIN + PANEL_H_MAX
    TOPO_X, TOPO_W = 0.04, 0.28
    MAP_X, MAP_W = 0.36, 0.28
    CBAR_X, CBAR_W = 0.645, 0.012
    CURVE_X, CURVE_W = 0.72, 0.26
    TOPO_CBAR = [0.326, 0.14, 0.008, 0.78]
    # The 'Markers' slider stack.
    SLIDER_X = 0.45
    SLIDER_Y0, SLIDER_PITCH = 0.02, 0.025
    PANEL_CLEARANCE = 0.06
    AVGS_XY = (0.665, 0.02)
    M_COLORS = widgets.MarkerGroupControls.M_COLORS
    # The Feenstra companion shows heatmaps (its own 'γ' boxes and pipeline box).
    feenstra_heatmaps = True

    DESIGN = (16, 9)       # the design size (inches): the text zoom of the window is its size relative to this (qtplot.window_zoom)

    def __init__(self, grid, active_procs, scalefactor, hampel_defaults, sgolay_defaults, cmap='jet',
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
          cmap            : the maps' colour map
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
        # lines are drawn on both panels in the auto or the manual colour (with a toggle on).  'drift
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
        self.cmap_norm = False                      # 'cmap norm': each map its own colour scale
        held = grid.channels()                      # the maps start on a channel the file holds
        self.channel = channel if channel in held else (held[0] if held else 'didv')
        self.order = np.argsort(grid.energy)
        self.V = grid.energy[self.order]
        self.n_e = len(self.V)
        self.e_index = int(np.argmin(np.abs(self.V)))
        self.gammas = np.ones(self.n_e)             # one γ per energy slice
        self.view = 'single'
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

        # Marker state: one group of two markers at the grid's centre.
        self.n_lines, self.n_fast = int(grid.n_lines), int(grid.n_fast)
        self.group_sizes = [2]
        self.marker_px = self.place_group(self.layout_centers(1)[0], 2)
        self.group_ctr = []                         # the centre of every group (kept by a group without markers)
        self._sync_centres()
        self._active_obj = None
        self._marker_artists = []
        self._panel_bottom = self.PANEL_BOTTOM_MIN

        self.fig = plt.figure(figsize=self.DESIGN)
        # The controls (widgets.py).
        self.filters = widgets.FilterControls(self.fig, [0.04, 0.005, 0.19, 0.125],
                                              hampel=hampel_defaults, sgolay=sgolay_defaults,
                                              hampel_on=False, sgolay_on=False)
        self.filters.on_change(self.recompute)
        self.rad_dir = widgets.make(self.fig, 'topo pass', 0.26, 0.015, value='forward')
        self.rad_dir.on_change(self._on_direction)
        self.rad_channel = widgets.make(self.fig, 'channel', 0.34, 0.015, value=STS.CHANNEL_NAMES[self.channel])
        self.rad_channel.on_change(self._on_channel)
        # Several sweeps per pixel (sweeps.py): the 'sweeps' radio chooses the view the maps are made of.
        self.rad_sweeps = None
        if getattr(grid, 'sweep_info', None) is not None:
            self.rad_sweeps = widgets.make(self.fig, 'sweeps', 0.01, 0.80, labels=list(grid.sweep_modes()),
                                           value=grid.sweep_mode)
            self.rad_sweeps.on_change(self._on_sweeps)
        self.chk_feenstra = widgets.make(self.fig, 'Feenstra', 0.34, 0.08, value=self.divide_iv)
        self.chk_feenstra.on_change(self._on_feenstra_toggle)
        self.pipeline = widgets.PipelineBox(self.fig, calibrate=self.calibrate, normalize=self.normalize)
        self.pipeline.on_change(self._on_pipeline_change)
        self.chk_cmap_norm = widgets.make(self.fig, 'cmap norm', 0.825, 0.065, value=self.cmap_norm)
        self.chk_cmap_norm.on_change(self._on_cmap_norm)
        self.chk_auto = widgets.make(self.fig, 'auto drift', 0.26, 0.112, value=self.auto_on)
        self.chk_auto.on_change(lambda flag: self._on_drift_toggle('auto', flag))
        self.chk_manual = widgets.make(self.fig, 'manual drift', 0.53, 0.112, value=self.manual_on)
        self.chk_manual.on_change(lambda flag: self._on_drift_toggle('manual', flag))
        self.btn_drift_win = widgets.make(self.fig, 'drift window', 0.612, 0.114)
        self.btn_drift_win.on_change(self._on_drift_window)
        # The grid's stripe matching: the 'stripe' toggle (one state per file, Drift.set_stripe_on) and
        # the 'stripe match' button, the row under the drift toggles, left of 'Average'.
        self.chk_stripe = widgets.make(self.fig, 'stripe', 0.54, 0.072, value=False)
        self.chk_stripe.on_change(self._on_stripe_toggle)
        self.btn_stripe_win = widgets.make(self.fig, 'stripe match', 0.588, 0.074)
        self.btn_stripe_win.on_change(self._on_stripe_window)
        self._sync_stripe_toggle()
        self.btn_view = widgets.make(self.fig, 'Grid view', 0.92, 0.065)
        self.btn_view.on_change(self.toggle_view)
        self.steps = widgets.StepButtons(self.fig, '< E', 'E >', 0.895, 0.01)
        self.steps.on_change(self.step)
        self.groups_ctl = widgets.MarkerGroupControls(self.fig, self.SLIDER_X, self.SLIDER_Y0, self.SLIDER_PITCH,
                                                      self.AVGS_XY, max_groups=self.MAX_GROUPS,
                                                      n_groups=len(self.group_sizes))
        self.groups_ctl.on_avgs(self._on_avgs_change)
        self.groups_ctl.on_markers(self._on_group_slider)
        self.groups_ctl.on_add_new(self._on_add_new)
        self.groups_ctl.on_add_existing(self._on_add_existing)
        self.stacks = widgets.StackControls(self.fig, (0.76, 0.01), (0.82, 0.01), (0.665, 0.065),
                                            needs_average=True)
        self.stacks.on_change(self._redraw_markers)
        self.btn_send = widgets.make(self.fig, 'Send to fit', 0.73, 0.065)
        self.btn_send.on_change(self._send_to_fit)
        # The whole map into an averaging window: 'map → new avg' starts one, 'map → existing avg' lists
        # the open ones (map_selection: every acquired pixel's spectrum as one sweep of one entry).
        self.btn_map_new = widgets.make(self.fig, 'map → new avg', 0.45, 0.012)
        self.btn_map_new.on_change(self._on_map_new)
        self.btn_map_exist = widgets.make(self.fig, 'map → existing avg', 0.45, 0.046)
        self.btn_map_exist.on_change(self._on_map_existing)
        # The figure composer (composer.FigureComposer): 'add to plot' under the topo, the map and the curve
        # panel sends that panel as shown; 'new plot' empties the figure.
        self.btn_plot_topo = widgets.make(self.fig, 'add to plot', self.TOPO_X, 0.135)
        self.btn_plot_topo.on_change(lambda: self._on_add_to_plot('topo'))
        self.btn_plot_map = widgets.make(self.fig, 'add to plot', self.MAP_X, 0.135)
        self.btn_plot_map.on_change(lambda: self._on_add_to_plot('map'))
        self.btn_plot_curve = widgets.make(self.fig, 'add to plot', self.CURVE_X, 0.135)
        self.btn_plot_curve.on_change(lambda: self._on_add_to_plot('spectra'))
        self.btn_new_plot = widgets.make(self.fig, 'new plot', 0.43, 0.114)
        self.btn_new_plot.on_change(self._on_new_plot)
        self.gamma_single = widgets.GammaBox(self.fig, gamma=self.gammas[self.e_index])
        self.gamma_single.on_change(self._on_gamma_single)

        # Single view axes.
        self.ax_topo = self.fig.add_axes([self.TOPO_X, self.PANEL_BOTTOM_MIN, self.TOPO_W, self.PANEL_H_MAX])
        self.ax_map = self.fig.add_axes([self.MAP_X, self.PANEL_BOTTOM_MIN, self.MAP_W, self.PANEL_H_MAX])
        self.cax_map = self.fig.add_axes([self.CBAR_X, self.PANEL_BOTTOM_MIN, self.CBAR_W, self.PANEL_H_MAX])
        self.ax_curve = self.fig.add_axes([self.CURVE_X, self.PANEL_BOTTOM_MIN, self.CURVE_W, self.PANEL_H_MAX])
        self.ax_curve.keep_clear_of(self.cax_map)        # the curves never run into the colour bar's labels
        self.ax_topo_cbar = plots.draw_topo_cbar(self.fig, self.TOPO_CBAR)
        self.gamma_single.attach(self.ax_map, gap_in=0.2)
        # Grid view axes (built once, hidden until toggled), one γ box under each map.
        n_cells = self.n_e + 1                      # the topo first, then one map per energy
        ncols = int(np.ceil(np.sqrt(n_cells)))
        nrows = int(np.ceil(n_cells / ncols))
        # The grid stays under the pipeline box (top right); each cell keeps room under its map
        # for the 'γ' box, the energy written inside the map.
        gl, gb, gw, gh = 0.03, 0.14, 0.95, 0.78
        cw, ch = gw / ncols, gh / nrows
        # The topo in the first cell (drawn by _draw_topo as the single view's topo is).
        self.grid_topo_ax = self.fig.add_axes([gl + 0.08 * cw, gb + gh - ch + 0.38 * ch, 0.66 * cw, 0.60 * ch])
        self.grid_topo_ax.set_xticks([])
        self.grid_topo_ax.set_yticks([])
        self.grid_axes, self.grid_caxes, self.grid_gammas = [], [], []
        for e in range(self.n_e):
            rr, cc = divmod(e + 1, ncols)
            x0 = gl + cc * cw
            y0 = gb + gh - (rr + 1) * ch
            ax = self.fig.add_axes([x0 + 0.08 * cw, y0 + 0.38 * ch, 0.66 * cw, 0.60 * ch])
            cax = self.fig.add_axes([x0 + 0.76 * cw, y0 + 0.38 * ch, 0.04 * cw, 0.60 * ch])
            ax.set_xticks([])
            ax.set_yticks([])
            self.grid_axes.append(ax)
            self.grid_caxes.append(cax)
            g = widgets.GammaBox(self.fig, gamma=self.gammas[e])
            g.attach(ax, gap_in=0.01)
            g.on_change(lambda value, e=e: self._on_gamma_grid(e, value))
            g.show(False)
            self.grid_gammas.append(g)
        self.grid_ims = [None] * self.n_e
        self.grid_labels = [None] * self.n_e      # the grid view's energy label per cell, made once (_grid_label)
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
        """The 'Average' state (widgets.StackControls); stack_on and substack_on are the 'stack' and 'substack' states."""
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
        """The 'calibrate' / 'normalize' box: the maps recomputed with the flags; refused, the box goes back."""
        old = (self.calibrate, self.normalize)
        self.calibrate, self.normalize = state['calibrate'], state['normalize']
        before = self.maps
        self.recompute()
        if self.maps is before:
            self.calibrate, self.normalize = old
            self.pipeline.set_state(*old)
            return
        print(f"calibration {'on' if self.calibrate else 'off'}, normalization {'on' if self.normalize else 'off'}: "
              f"the map window shows {self._quantity()}.")

    def _on_cmap_norm(self, flag):
        """'cmap norm': each map its own colour scale (on) or one scale over the whole cube (off)."""
        self.cmap_norm = bool(flag)
        self.draw_single()
        if self.view == 'grid':
            self.draw_grid()
        self.fig.canvas.draw_idle()

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

    def _drift_bounds(self):
        """
        The limits [l, r, b, t] whenever a result places the pixels: the
        bounding box of everything drawn, exactly (edge to edge, no
        margin): every acquired map pixel's cell at every energy (the maps
        are NaN only where never acquired) and every placed topo pixel's
        cell at their shifts (plots.pixel_bounds), and the spectra's drift
        lines (_frame_lines; the markers sit on the pixels).  Every energy
        counts, so the view holds still while the energy is stepped.  The
        recorded frame without a result, or when nothing is drawn.
        Memoized with the result (_bounds).
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
        pass the 'topo pass' radio names on top ('average' counts as
        forward: a grid's passes are never averaged); off, the chosen pass
        at the recorded pixels (plots.draw_topo_frame).  The limits are
        _frame_limits'.
        """
        for pm in self._topo_maps.pop(id(ax), []):
            try:
                pm.remove()
            except Exception:
                pass
        ax.clear()
        g = self.grid
        fs = 7 if small else 9
        if not self.has_topo:
            ax.text(0.5, 0.5, "no topography page in this file", ha='center', va='center',
                    transform=ax.transAxes, fontsize=fs)
            return
        corrected = self._topo_shift('forward') is not None
        if corrected:
            top = 'reverse' if self.direction == 'reverse' else 'forward'
            passes = ['forward', 'reverse'] if top == 'reverse' else ['reverse', 'forward']    # the last drawn on top
            maps = []
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
            self._topo_maps[id(ax)] = maps
            ax.set_aspect('equal', adjustable='box')
            what = (f"both passes, {top} on top" if len(maps) > 1 else passes[-1]) + self._drift_tag()
        else:
            direction = 'reverse' if self.direction == 'reverse' else 'forward'      # a grid's passes are never averaged
            img = processed_topo(g, direction, self.active_procs) * 1e9
            plots.draw_topo_frame(ax, g, img, self.scalefactor, cmap=TOPO_CMAP)
            what = direction
        self._frame_limits(ax)
        if small:
            ax.set_title("Topo", fontsize=fs)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_title(f"Topography: {g.name}  [{what}]", fontsize=fs)
            ax.set_xlabel("x (nm, scan frame)")
            ax.set_ylabel("y (nm, scan frame)")

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
        if self.view == 'grid':
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
        if self.view == 'grid':
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

    def _cube(self, divide_iv, calibrate=None, normalize=None):
        """
        The processed cube (n_e, n_lines, n_fast) with the division as given
        (recompute's reshaping) and the calibrate / normalize flags (the
        window's unless given: the companion's own).
        """
        g = self.grid
        hp, sp = self.filters.params()
        c = g.curves(hp, sp, divide_iv, self.iv_width(), channel=self.channel, iv_zero_sub=self.iv_zero_sub,
                     normalize=self.normalize if normalize is None else normalize,
                     calibrate=self.calibrate if calibrate is None else calibrate)[self.order, :]
        n = g.n_lines * g.n_fast
        c = c[:, :n].copy()
        c[:, ~g.acquired[:n]] = np.nan
        return c.reshape(self.n_e, g.n_lines, g.n_fast)

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

    def feenstra_view(self, calibrate=True, normalize=True):
        """
        The companion's view (feenstra.FeenstraWindow): the map at the current
        energy in four versions, the dI/dV of the pipeline with the division
        off, the current in amps as the division works on it (its value at
        0 V subtracted with '0Vsub'), its I/V in A/V as the division divides
        by it (broadened with 'broadened'), and the ratio with the division
        on, with the marker curves of each in the current marker state; a
        file without the Current channel gives notes for the current, I/V,
        and ratio panels, the IETS channel a note for the ratio panel.
        `calibrate` and `normalize` are the companion's own flags: the dI/dV
        and the ratio through the pipeline with them; with `normalize` on the
        current and the I/V are unit area per pixel (normalize_abs).
        """
        g = self.grid
        q = ("d2I/dV2 (IETS)" if self.channel == 'iets' else "dI/dV") + ("" if calibrate else ", uncalibrated")
        at = f"at {self.V[self.e_index]:.3f} V"
        zs, width = self.iv_zero_sub, self.iv_width()
        t = feenstra_titles(zs, self.iv_broadened, self.iv_broadening, energy_has_zero(g.energy))
        nn = "" if normalize else ", not normalized"
        versions = {
            'didv': {'title': f"{q} (division off){nn} {at}",
                     'ylabel': f"{q} ({curve_units(self.channel, calibrate, normalize, g.units_lia, g.units_iets, g.units_iv)})"},
            'iv': {'title': f"{t['iv']} {at}", 'ylabel': "I (unit area)" if normalize else "I (A)"},
            'ivv': {'title': f"{t['ivv']} {at}", 'ylabel': "I/V (unit area)" if normalize else "I/V (A/V)"},
            'ratio': {'title': f"{t['ratio']} (division on){nn} {at}", 'ylabel': "(dI/dV)/(I/V) (arb. units)"},
        }
        # The cubes depend on these settings only, never on the markers: computed once per setting.
        hp, sp = self.filters.params()
        memo_key = (bool(calibrate), bool(normalize), repr(hp), repr(sp), zs, width, self.channel)
        memo = getattr(self, '_cube_memo', None)
        if memo is not None and memo[0] == memo_key:
            cubes = memo[1]
        else:
            cubes = self._feenstra_cubes(calibrate, normalize, zs, width)
            self._cube_memo = (memo_key, cubes)
        if g.iv is None:
            for key in ('iv', 'ivv', 'ratio'):
                versions[key]['note'] = "no Current channel in this file"
        elif self.channel == 'iets':
            versions['ratio']['note'] = "the division applies to dI/dV only"
        for key, cube in cubes.items():
            markers, groups = self._curve_items(cube)
            versions[key].update({'top': cube[self.e_index], 'markers': markers, 'groups': groups})
        return self._feenstra_view_dict(g, at, versions, q)

    def _feenstra_cubes(self, calibrate, normalize, zs, width):
        """The companion's cubes (n_e, n_lines, n_fast) per version."""
        g = self.grid
        cubes = {'didv': self._cube(False, calibrate, normalize)}
        if g.iv is not None:
            hp, sp = self.filters.params()
            iv_a = iv_corrected(g.energy, g.iv, zs, hp, sp)
            ivv = iv_broadened(g.energy, g.iv, width, zs, hp, sp)
            if normalize:
                iv_a = normalize_abs(g.energy, iv_a)[0]
                ivv = normalize_abs(g.energy, ivv)[0]
            cubes['iv'] = self._cube_of(iv_a)
            cubes['ivv'] = self._cube_of(ivv)
            if self.channel != 'iets':
                cubes['ratio'] = self._cube(True, calibrate, normalize)
        return cubes

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
                'drift_lines': self._frame_lines(),
                'marker_points': [(self.marker_xy(px, True), self.marker_color(i)) for i, px in enumerate(self.marker_px)],
                'centers': [self.group_center_xy(m, g) for g, m in enumerate(self.groups())],
                'cmap': self.cmap}

    def recompute(self):
        """
        The maps: every spectrum of the grid through the pipeline
        (Sm4File.curves with the filter box's settings, the division when
        'Feenstra' is on with the Feenstra settings, the channel, and the
        calibrate / normalize flags), energy ascending, reshaped to (n_e,
        n_lines, n_fast) with never-acquired pixels NaN; then the single
        view (and the grid view when shown) redrawn.  A grid that cannot be
        processed (no Current channel for the division, no such channel)
        leaves self.maps as it was with the reason printed: the callers
        test `self.maps is before` to put their control back.
        """
        hp, sp = self.filters.params()
        g = self.grid
        try:
            c = g.curves(hp, sp, self.divide_iv, self.iv_width(), channel=self.channel,
                         iv_zero_sub=self.iv_zero_sub, normalize=self.normalize,
                         calibrate=self.calibrate)[self.order, :]                                  # (n_e, N)
        except Exception as exc:
            print(f"Map window: {exc}")
            return
        n = g.n_lines * g.n_fast
        c = c[:, :n].copy()
        c[:, ~g.acquired[:n]] = np.nan                        # never acquired: drawn transparent
        self.maps = c.reshape(self.n_e, g.n_lines, g.n_fast)
        self.draw_single()
        if self.view == 'grid':
            self.draw_grid()
        self.fig.canvas.draw_idle()

    def _quantity(self):
        """Name of the displayed quantity: dI/dV, (dI/dV)/(I/V), or d2I/dV2 (IETS), with the flags that are off."""
        q = ("(dI/dV)/(I/V)" if self.divide_iv else "dI/dV") if self.channel == 'didv' else "d2I/dV2 (IETS)"
        return q + ("" if self.calibrate else ", uncalibrated") + ("" if self.normalize else ", not normalized")

    def _limits(self):
        """(vmin, vmax) of the colour scale: the whole cube with 'cmap norm' off, None (each map its own) on."""
        if self.cmap_norm or self.maps is None:
            return None, None
        return plots.finite_range(self.maps)

    def draw_single(self):
        """
        The single view's map at the current energy (e_index): a
        plots.PixelMap with the pixel shifts when a drift result places
        them (_frame_shift), else plots.draw_heatmap; the colour scale of
        _limits and the slice's γ; the colour bar labelled with the
        quantity and its units; the frame limits; the title ("... drift
        corrected (<source>)" with shifts); the γ box set to the slice; then
        the markers and the curve panel (_redraw_markers).  The topo panel
        is _draw_topo's.
        """
        ax = self.ax_map
        ax.clear()
        self.cax_map.clear()
        self._pixel_maps.pop('single', None)
        m = self.maps[self.e_index]
        lo, hi = self._limits()
        shift = self._frame_shift(self.e_index)
        if shift is not None:
            im = plots.PixelMap(ax, m, self.extent, self.cmap, gamma=self.gammas[self.e_index], shift=shift,
                                vmin=lo, vmax=hi)
            self._pixel_maps['single'] = im
            ax.set_aspect('equal', adjustable='box')
        else:
            im = plots.draw_heatmap(ax, m, self.extent, self.cmap, gamma=self.gammas[self.e_index], vmin=lo, vmax=hi,
                                    origin='lower', aspect='equal', interpolation='none')
        self._single_im = im
        self._frame_limits(ax)
        q = self._quantity()
        units = ("arb. units" if self.divide_iv and self.channel == 'didv'
                 else curve_units(self.channel, self.calibrate, self.normalize, self.grid.units_lia,
                                  self.grid.units_iets, self.grid.units_iv))
        plots.draw_colorbar(self.fig, im, self.cax_map, label=f"{q} ({units})")
        ax.set_title(f"{q.split(',')[0]} map at {self.V[self.e_index]:.3f} V: {q}, energy {self.e_index + 1} of {self.n_e}"
                     + ("" if self.cmap_norm else ", one colour scale for every energy")
                     + (self._drift_tag() if shift is not None else ""), fontsize=10)
        ax.set_xlabel("x (nm, scan frame)")
        ax.set_ylabel("y (nm, scan frame)")
        self.gamma_single.set(self.gammas[self.e_index])
        # The markers and the curve panel follow the maps (the topo panel is _draw_topo's).
        self._redraw_markers()

    def draw_grid(self):
        """
        The grid view: the topo cell when stale (_draw_topo, small), then
        every energy's map on the colour scale of _limits with its own γ:
        with pixel shifts a new plots.PixelMap each time (the old one
        removed), else an image made once (plots.draw_heatmap) and updated
        after (plots.update_heatmap); the energy written inside the map at
        the first draw; each map's colour bar redrawn.
        """
        if self._grid_topo_stale:
            self._draw_topo(self.grid_topo_ax, small=True)
            self._grid_topo_stale = False
        lo, hi = self._limits()
        for e, ax in enumerate(self.grid_axes):
            m = self.maps[e]
            shift = self._frame_shift(e)
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
                pm = plots.PixelMap(ax, m, self.extent, self.cmap, gamma=self.gammas[e], shift=shift, vmin=lo, vmax=hi)
                self._pixel_maps[e] = pm
                if self.grid_ims[e] is None:
                    self._frame_limits(ax)
                    ax.set_aspect('equal', adjustable='box')
                    self._grid_label(e, ax)
                self.grid_ims[e] = pm
            elif self.grid_ims[e] is None:
                self.grid_ims[e] = plots.draw_heatmap(ax, m, self.extent, self.cmap, gamma=self.gammas[e],
                                                      vmin=lo, vmax=hi, origin='lower', aspect='equal',
                                                      interpolation='none')
                self._frame_limits(ax)
                self._grid_label(e, ax)
            else:
                plots.update_heatmap(self.grid_ims[e], m, gamma=self.gammas[e], vmin=lo, vmax=hi)
            plots.draw_colorbar(self.fig, self.grid_ims[e], self.grid_caxes[e], labelsize=5)

    def _grid_label(self, e, ax):
        """The energy label of grid cell e ('<bias> V', top left), made once and kept across the cell's redraws."""
        if self.grid_labels[e] is None:
            self.grid_labels[e] = ax.text(0.03, 0.95, f"{self.V[e]:.2f} V", transform=ax.transAxes, fontsize=6,
                                          ha='left', va='top', bbox=dict(facecolor='white', alpha=0.75, lw=0, pad=1))
        return self.grid_labels[e]

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
        self._drift_res = None
        self._bounds = None
        self._lines_memo = None
        self._clear_pixel_maps()
        self._draw_topo(self.ax_topo)
        self._grid_topo_stale = True
        self.recompute()
        self._redraw_markers()

    def _on_direction(self, label):
        """The 'topo pass' radio: the embedded topo redrawn from the chosen pass (both views); nothing without a topo page."""
        if label == self.direction or not self.grid.topo_pages:
            return
        self.direction = label
        self._draw_topo(self.ax_topo)
        self._grid_topo_stale = True
        if self.view == 'grid':
            self.draw_grid()
        self.draw_single()
        self.fig.canvas.draw_idle()

    def _frame_limits(self, ax):
        """Axis limits of the scan frame in normal orientation (the extent itself may be reversed); whenever a result places the pixels (a drift source on, or the stripe matching alone), the box of everything drawn, edge to edge (_drift_bounds)."""
        l, r, b, t = self._drift_bounds() if self._drift_result() is not None else self.extent
        ax.set_xlim(min(l, r), max(l, r))
        ax.set_ylim(min(b, t), max(b, t))

    def _apply_view(self):
        """Show the axes and γ boxes of the current view (single or grid), hide the other's; the toggle button names the other view."""
        single = self.view == 'single'
        for ax in (self.ax_topo, self.ax_map, self.cax_map, self.ax_curve, self.ax_topo_cbar):
            ax.set_visible(single)
        self.gamma_single.show(single)
        for ax in [self.grid_topo_ax] + self.grid_axes + self.grid_caxes:
            ax.set_visible(not single)
        for g in self.grid_gammas:
            g.show(not single)
        self.btn_view.set_text('Grid view' if single else 'Single view')
        self.fig.canvas.draw_idle()

    def toggle_view(self):
        """'Grid view' / 'Single view': the other view (the grid drawn when switched to)."""
        self.view = 'grid' if self.view == 'single' else 'single'
        if self.view == 'grid':
            self.draw_grid()
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
        """Marker indices per group (membership by index, widgets.MarkerGroupControls.groups)."""
        return widgets.MarkerGroupControls.groups(self.group_sizes)

    def group_of(self, i):
        """The group index of marker i."""
        return widgets.MarkerGroupControls.group_of(self.group_sizes, i)

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

    def group_center_xy(self, members, g=None, on_map=True):
        """A group's centre in the frame (nm) as drawn: the mean of its markers' drawn positions (marker_xy, on the map at the shown energy or on the topo), or pixel_xy of its kept centre without markers."""
        if members:
            pts = np.array([self.marker_xy(self.marker_px[i], on_map) for i in members], dtype=np.float64)
            return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
        return self.pixel_xy(self.group_center(members, g))

    def _marker_shift(self, on_map=True):
        """Every pixel's shift as drawn, (n_lines, n_fast, 2) in the frame (nm): on the map the shift of the shown energy (_frame_shift), on the topo the chained shift of the sweep's start; None without a result."""
        if on_map:
            return self._frame_shift(self.e_index)
        chain = self._chain_shift()
        if chain is None:
            return None
        g = self.grid
        n = g.n_lines * g.n_fast
        s = np.full((n, 2), np.nan)
        m = min(n, len(chain))
        s[:m] = chain[:m]
        return self._to_frame_shift(s).reshape(g.n_lines, g.n_fast, 2)

    def marker_xy(self, px, on_map=True):
        """Where grid pixel (line, fast) is drawn (nm): its centre plus its shift as drawn (_marker_shift), so that a marker sits on its pixel; the centre alone without a result."""
        x, y = self.pixel_xy(px)
        sh = self._marker_shift(on_map)
        if sh is not None:
            s = sh[int(px[0]), int(px[1])]
            if np.all(np.isfinite(s)):
                return (float(x + s[0]), float(y + s[1]))
        return (float(x), float(y))

    def xy_pixel_drawn(self, x, y, on_map=True):
        """The grid pixel drawn nearest a point of the frame (every pixel at its shift as drawn, _marker_shift); xy_pixel without a result."""
        sh = self._marker_shift(on_map)
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
        """Group indices by their centres' y on the frame, highest first (the stack and slider order)."""
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

    def _redraw_markers(self):
        """
        Redraw the marker artists on the map and the topo panel, then the
        curve panel, the slider stack and the companion: with a drift
        toggle on the spectra's drift lines (plots.draw_drift_lines, in the
        source's colour), every marker as a hollow square in its colour,
        and with Average on every non-empty group's centre as a black X.
        Picking: on the map with Average on the centres are picked (the
        markers are not, so a group moves as one); elsewhere the markers
        are picked and the centres never (a Line2D without a picker is
        unpickable).  Nothing is drawn before the first recompute.
        """
        for art in self._marker_artists:
            try:
                art.remove()
            except (ValueError, NotImplementedError):
                pass
        self._marker_artists = []
        if self.maps is None:
            return
        self._sync_centres()
        groups = self.groups()
        lines = self._frame_lines()
        for ax in (self.ax_map, self.ax_topo):
            if ax is self.ax_topo and not self.has_topo:
                continue
            if lines is not None:
                self._marker_artists.extend(plots.draw_drift_lines(ax, lines, color=plots.drift_color(self._drift_source)))
            on_map = ax is self.ax_map
            marker_pick = {} if (self.avg_on and on_map) else {'picker': 6}
            # Every marker on its pixel as drawn (marker_xy: the map's shift at the shown energy, the
            # topo's chained shift), the centres at the mean of their members' drawn positions.
            for i, px in enumerate(self.marker_px):
                x, y = self.marker_xy(px, on_map)
                h, = ax.plot([x], [y], marker='s', ms=9, mfc='none', mec=self.marker_color(i), mew=1.8,
                             ls='none', label=f'marker_{i}', **marker_pick)
                self._marker_artists.append(h)
            if self.avg_on:
                avg_pick = {'picker': 6} if on_map else {}
                for g, members in enumerate(groups):
                    if not members:
                        continue                         # a group without markers has no centre mark
                    x, y = self.group_center_xy(members, g, on_map)
                    h, = ax.plot([x], [y], marker='X', ms=10, color='black', ls='none',
                                 label=f'_avg_{g}', **avg_pick)
                    self._marker_artists.append(h)
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

    # --- dragging ---

    def _on_pick(self, event):
        """A press on a marker ('marker_<i>') or a group centre ('_avg_<g>'): it becomes the dragged object."""
        artist = event.artist
        label = getattr(artist, 'get_label', lambda: '')()
        if not isinstance(label, str):
            return
        if label.startswith('marker_'):
            idx = int(label.split('_')[1])
            if idx < len(self.marker_px):
                self._active_obj = ('marker', idx)
        elif label.startswith('_avg_'):
            g = int(label.split('_')[2])
            if g < len(self.group_sizes):
                self._active_obj = ('avg', g)

    def _on_motion(self, event):
        """Drag (on the map or the topo): a marker to the pixel under the cursor; a group's centre by the same whole-pixel offset for every member, clipped so no member leaves the grid."""
        if self._active_obj is None or event.xdata is None or event.ydata is None:
            return
        if event.inaxes not in (self.ax_map, self.ax_topo):
            return
        kind, idx = self._active_obj
        target = self.xy_pixel_drawn(event.xdata, event.ydata, event.inaxes is self.ax_map)   # the pixel drawn under the cursor
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

    # --- sliders and toggles ---

    def _on_group_slider(self, g, m):
        """A 'Markers' slider: that group gets the new number of pixels nearest its centre; the others stay."""
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
        """'avgs': the number of groups.  Growing adds each new group with the last group's size (DEFAULT_GROUP_SIZE when there is none or it is empty) at the free slot of the diagonal layout farthest from the others (widgets.MarkerGroupControls.grow_groups); shrinking drops the last groups with their markers."""
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
            for best in widgets.MarkerGroupControls.grow_groups(n, centers, self.layout_centers(n), dist):
                self.marker_px.extend(self.place_group(best, m))
                self.group_sizes.append(m)
                self.group_ctr.append(tuple(best))
        self._redraw_markers()

    def _refresh_group_sliders(self):
        """One 'Markers' slider and its buttons per group, ordered as the groups' centres (highest y at the top); the single view's panels raised above the stack (PANEL_CLEARANCE) when it would run into them, the γ box re-attached."""
        self.groups_ctl.refresh(self.group_order(), self.group_sizes)
        top = self.groups_ctl.stack_top()
        bottom = max(self.PANEL_BOTTOM_MIN, top + self.PANEL_CLEARANCE)
        if abs(bottom - self._panel_bottom) > 1e-9:
            self._panel_bottom = bottom
            h = self.PANEL_TOP - bottom
            for ax, x, w in ((self.ax_topo, self.TOPO_X, self.TOPO_W), (self.ax_map, self.MAP_X, self.MAP_W),
                             (self.cax_map, self.CBAR_X, self.CBAR_W), (self.ax_curve, self.CURVE_X, self.CURVE_W)):
                ax.set_position([x, bottom, w, h])
            self.ax_topo_cbar.set_position([self.TOPO_CBAR[0], bottom, self.TOPO_CBAR[2], h])
            self.gamma_single.attach(self.ax_map, gap_in=0.2)

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
        """The pop-up list of the open averages (widgets.AveragePicker), made at the first use."""
        if self._avg_picker is None:
            self._avg_picker = widgets.AveragePicker(self.fig, self._averages())
        return self._avg_picker

    def _on_add_new(self, g):
        """'add new' of a slider row: that group starts a new average (each marker one sweep)."""
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
        """'add existing' of a slider row: the list of the open averages; a row picked adds the group to it."""
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
        bias; 'topo': the embedded topo as drawn (the quads with their
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
                c.add_map(self.maps[e], self.extent, [*self.ax_map.get_xlim(), *self.ax_map.get_ylim()],
                          vmin, vmax, self.gammas[e], self.cmap, self.V[e], shift=self._frame_shift(e))
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
        div = self.divide_iv
        quantity = 'ratio' if div else 'didv'
        suffix = " over I/V" if div else ""
        energy = g.energy
        V = np.sort(energy)
        e_order = np.argsort(energy)
        iv_sorted = np.asarray(g.iv, dtype=np.float64)[e_order, :]
        i_ref = int(np.argmin(np.abs(V)))
        i_bias = int(np.argmin(np.abs(V - g.V_bias)))

        def processed(cols, hampel, sgolay):
            # The pipeline on the raw columns with calibrate and normalize at their defaults (both on: the
            # fit's pipeline), energy ascending; areas: the normalization areas per column (amps).
            lia = g.lia[:, cols]
            iv = g.iv[:, cols]
            c, areas = STS.process_arrays(energy, lia, iv, hampel, sgolay, div, self.iv_width(),
                                          return_areas=True, channel='didv', iv_zero_sub=self.iv_zero_sub,
                                          vmod=g.vmod)
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
            cf, areas = processed(cols, hp, sp)
            cu, _ = processed(cols, None, None)
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
            sends.append((data, unf, areas_A, delta_I, label, g.lia[:, cols], g.iv[:, cols], stage))

        settings = dict(self.fit_settings)
        use_cache = settings.pop('use_cache', True)
        for data, unf, areas_A, delta_I, label, lia_raw, iv_raw, stage in sends:
            print("\n" + "=" * 70)
            if use_cache and fit.Fit.has_cache(label):
                an = fit.Fit.from_cache(fit.Fit.cache_path(label))
            else:
                an = fit.Fit.from_curve(V, data, unf, g.V_bias, g.I_setpoint, areas_A, label,
                                        hampel_filter_params=hp, sgolay_filter_params=sp,
                                        delta_I_window_A=delta_I,
                                        lia_raw=lia_raw, iv_raw=iv_raw,
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
    window = Map(sm4file.Sm4File(path), [False, True], 2.75, hampel_defaults, sgolay_defaults, cmap='jet',
                 divide_iv=True, iv_broadening=0.2)
