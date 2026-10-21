"""
The stripe window: the stripe matching of one grid file, set up once for
every slow-axis setting, opened by the browser's and the Topo window's
'stripe match' button.

What it shows, everything at the grid file's scale factor (the scalefactor
argument, the browser's scale_of(grid), 1e9 / scalefactor nm per recorded
metre; the reference topos too, unlike the scan area, which draws each file
at its own scale).  The earlier topos under the grid (the stripe reference,
Drift.stripe_reference), each placed as the scan area places it under the
slow-axis setting chosen here (Drift.corrected_topo), the grid's own topo
stripes at their placed positions (both passes, the pass the 'stripe pass'
radio names drawn on top), the spectra's placed positions, and one handle
at every point of the placement that can be moved: every line's forward
end (where its reverse pass begins) and every line's return point (one
point with the next line's start, the recorded separation kept).  The
grid's start is anchored and has no handle.

The three radios, each titled above its box.  'drift' (off, auto,
manual; the 'slow axis' widget) chooses the velocity source the window
works under, the same three sources the browser's drift toggles give, and
with it which of the grid's three stored sets is shown and edited
(Drift.stripe_sets: one fast-axis state and one set of points per
setting), so every setting can be prepared from this one window.
'stripe' (off, auto, manual; the 'fast axis' widget) is that set's state:
off, the velocity alone places the lines (no matching); auto, the extrema
matching (drift_correct_grid); manual, the matching with the dragged
points.  'pass' (forward, reverse; the 'stripe pass' widget) chooses
which pass is drawn on top.

Dragging.  A press on a handle and a drag moves that point along its
line only.  The mouse's motion is projected on the direction of the line
as placed under the setting in force (its corrected first to last
spectrum, so with a velocity applied it is the drifted scan direction),
the motion across it is discarded, and the point's coordinate along the
scan axis changes by that projection (times the cosine between the placed
direction and the recorded axis, 1 with drift off).  The handle is then
redrawn where the placement puts the point, on the placed line: the
placement moves a point along the recorded scan axis, the point's slow-axis
shift being fixed by its time, so the drawn displacement and the drifted
line direction differ by the drift over one spectrum, picometres.  The
first drag pins every point of the
current placement (Drift.pin_stripe_points) and sets the fast axis to
manual, so the dragged point alone moves: a forward end stretches the
line's forward pass and its reverse pass; a return point stretches the
line's reverse pass and moves the next line's start with it, that line's
own forward end staying where it is.  The window redraws while dragging;
the release writes the settings store (the 'drift' section of
settings_<stamp>.json, Drift._save_manual) and tells every window
(Drift.notify).  'clear points' drops the set's points, back to the
matching.

Everything applies at once wherever the grid is drawn: the scan area, the
Topo window, the Line window's heatmap (its rows sheared within their sweeps by the placement's forward compression),
the Map window's pixels and embedded topo.
"""

if __package__ in (None, ''):
    # Run as a file (Spyder F5): make the package importable from wherever its folder sits.
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
from sm4_Browser import qtplot as plt
from sm4_Browser import widgets, plots
from sm4_Browser.topo import grid_stripe_layers


class StripeWindow:
    """
    The window (the module docstring).  StripeWindow(drift, grid, scalefactor, active_procs,
    setting, pass_top): `drift` the experiment's Drift, `grid` the loaded grid file (kind 'grid'
    with topo pages, else ValueError), `scalefactor` the grid's scale factor (the one drawing
    scale of the window), `active_procs` the topo processor toggles (topo.process_topo),
    `setting` the slow-axis setting to open at ('off', 'auto', 'manual'; the opener's setting in
    force; anything else gives 'auto'), `pass_top` the pass drawn on top ('forward' or 'reverse').
    The window subscribes its draw to the Drift (every manual change anywhere redraws it) and
    unsubscribes when closed.
    """

    FLAGS = {'off': (False, False), 'auto': (True, False), 'manual': (False, True)}   # the toggles each setting means
    HANDLE_COLORS = {'E': '#ffd700', 'R': '#00ffff'}                                   # forward ends, return points

    DESIGN = (12, 8.8)     # the design size (inches): the text zoom of the window is its size relative to this (qtplot.window_zoom)

    def __init__(self, drift, grid, scalefactor, active_procs, setting='auto', pass_top='forward'):
        if grid.kind != 'grid' or not grid.topo_pages:
            raise ValueError(f"{grid.name} is not a grid with topo pages; no stripe matching.")
        self.drift = drift
        self.grid = grid
        self.scalefactor = float(scalefactor)
        self.k = 1e9 / self.scalefactor
        self.active = list(active_procs)
        self.setting = setting if setting in drift.SLOW_SETTINGS else 'auto'
        self.top = pass_top if pass_top in ('forward', 'reverse') else 'forward'
        self.fig = plt.figure(figsize=self.DESIGN)
        try:
            self.fig.canvas.manager.set_window_title(f"stripe matching: {grid.name}")
        except Exception:
            pass
        self.ax = self.fig.add_axes([0.06, 0.19, 0.91, 0.76])
        self.rad_slow = widgets.make(self.fig, 'slow axis', 0.03, 0.015, value=self.setting)
        self.rad_slow.on_change(self._on_slow)
        self.rad_fast = widgets.make(self.fig, 'fast axis', 0.14, 0.015, value=drift.fast_state(grid, self.setting))
        self.rad_fast.on_change(self._on_fast)
        self.rad_pass = widgets.make(self.fig, 'stripe pass', 0.25, 0.015, value=self.top)
        self.rad_pass.on_change(self._on_pass)
        self.btn_clear = widgets.make(self.fig, 'clear points', 0.36, 0.03)
        self.btn_clear.on_change(self._on_clear)
        self.txt = self.fig.text(0.47, 0.015, '', fontsize=7, family='monospace', ha='left', va='bottom')
        self._handles = []                  # [(artist, key, line index, position (2,) metres, direction (2,))]
        self._drag = None                   # (key, direction u (2,), start position (2,) m, start coordinate s0 (m along the fast axis), cos = u . fast)
        self._res = None                    # the placement drawn last (result(), Drift.result under the window's setting)
        self.fig.canvas.mpl_connect('pick_event', self._on_pick)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        fig = self.fig
        drift.subscribe(self.draw, alive=lambda: plt.fignum_exists(fig))
        fig.canvas.mpl_connect('close_event', lambda event: drift.unsubscribe(self.draw))
        self.draw()
        plt.show()

    def is_open(self):
        """True while the window is open (the figure itself is tested, not its reusable number)."""
        return plt.fignum_exists(self.fig)

    def raise_(self):
        """The window brought to the front."""
        try:
            self.fig.canvas.manager.show()
        except Exception:
            pass

    # --- the placement under the window's setting ---

    def flags(self):
        """(auto, manual) the window's slow-axis setting means."""
        return self.FLAGS[self.setting]

    def result(self):
        """Drift.result of the grid under the setting with its stored fast-axis state (stripe None); None when the Drift makes none (no velocity with the fast axis off, no kept spectrum) or raises (reported)."""
        auto, manual = self.flags()
        try:
            return self.drift.result(self.grid, None, None, auto, manual)
        except Exception as exc:
            print(f"stripe window: no placement ({exc}).")
            return None

    def line_direction(self, res, i):
        """The unit direction of line i as placed (its first to its last kept spectrum, corrected), else the recorded fast axis."""
        g = self.grid
        fast, _ = g.grid_axes()
        cols = np.arange(i * g.n_fast, min((i + 1) * g.n_fast, g.n_spectra))
        acq = np.asarray(g.acquired, dtype=bool)[:g.n_spectra]
        kept = cols[acq[cols]]
        if res is not None and len(kept) >= 2:
            xy = np.asarray(res['xy_corr'], dtype=np.float64)
            d = xy[kept[-1]] - xy[kept[0]]
            n = float(np.linalg.norm(d))
            if np.isfinite(n) and n > 0:
                return d / n
        return np.asarray(fast, dtype=np.float64)

    def handle_positions(self, res):
        """
        [(key, line i, position (2,) metres)] of every movable point of a
        placement: 'E:i' at the corrected position of line i's last kept
        spectrum, 'R:i' at the corrected position of its return point (the
        reverse pixel nearest the line's start; without a reverse page the
        next line's first spectrum).  Empty without a placement with the
        fast axis on.
        """
        out = []
        if res is None or not res.get('stripe') or not res.get('lines'):
            return out
        g = self.grid
        xy = np.asarray(res['xy_corr'], dtype=np.float64)
        acq = np.asarray(g.acquired, dtype=bool)[:g.n_spectra]
        fast, _ = g.grid_axes()
        origin = np.asarray(g.spec_xy_m, dtype=np.float64)[0]
        pix_r = (res.get('pix') or {}).get('reverse')
        lines = {ln['i']: ln for ln in res['lines']}
        for i, ln in lines.items():
            cols = np.arange(i * g.n_fast, min((i + 1) * g.n_fast, g.n_spectra))
            kept = cols[acq[cols]]
            if len(kept) and np.all(np.isfinite(xy[kept[-1]])):
                out.append((f"E:{i}", i, xy[kept[-1]].copy()))
            if ln.get('R') is None:
                continue
            pos = None
            if pix_r is not None and ln.get('row') is not None and 0 <= int(ln['row']) < pix_r.shape[0]:
                row = np.asarray(pix_r[int(ln['row'])], dtype=np.float64)
                ok = np.all(np.isfinite(row), axis=1)
                if ok.any():
                    s = (row[ok] - origin) @ fast
                    pos = row[ok][int(np.argmin(s))].copy()
            if pos is None and (i + 1) in lines:
                nxt = np.arange((i + 1) * g.n_fast, min((i + 2) * g.n_fast, g.n_spectra))
                nk = nxt[acq[nxt]]
                if len(nk) and np.all(np.isfinite(xy[nk[0]])):
                    pos = xy[nk[0]].copy()
            if pos is not None:
                out.append((f"R:{i}", i, pos))
        return out

    # --- drawing ---

    def draw(self):
        """
        The axes drawn anew for the window's setting and pass: the reference topos (the oldest at
        the bottom), the grid's stripes (topo.grid_stripe_layers, plots.draw_grid_stripes; at the
        corrected pixel positions when there is a placement, else the recorded ones), the kept
        spectra's placed positions (cyan dots), the grid's start (white square), the handles
        (handle_positions), the limits edge to edge around everything drawn (plots.fit_limits), the
        title, the status text; the 'fast axis' radio set to the stored state.
        """
        ax = self.ax
        ax.clear()
        self._handles = []
        g, k = self.grid, self.k
        auto, manual = self.flags()
        # The reference topos as the scan area places them under the setting, the oldest at the bottom.
        try:
            ref = self.drift.stripe_reference(g, auto, manual)
            entries = list(ref.entries) if ref else []
        except Exception as exc:
            print(f"stripe window: no reference topo ({exc}).")
            entries = []
        corners = []
        for e in reversed(entries):
            f = e['f']
            try:
                img = self.drift.corrected_topo(f, 'forward', self.active, None, auto, manual) * 1e9
                plots.draw_topo_absolute(ax, f, img, self.scalefactor, cmap=plots.TOPO_CMAP)
                corners.append(f.corners_m() * k)
            except Exception as exc:
                print(f"stripe window: {f.name} not drawn ({exc}).")
        res = self.result()
        self._res = res
        pix = res['pix'] if (res is not None and res.get('pix') and res['pix'].get('forward') is not None) else None
        xy = np.asarray(res['xy_corr'] if res is not None else g.spec_xy_m, dtype=np.float64)
        xy = np.where(np.isfinite(xy), xy, np.asarray(g.spec_xy_m, dtype=np.float64)[:len(xy)])
        acq = np.asarray(g.acquired, dtype=bool)[:len(xy)]
        quads = None
        try:
            main, second, pos_m, pos_s = grid_stripe_layers(g, self.top, self.active, pix)
            if main is not None:
                width = (float(np.linalg.norm(g.spec_xy_m[g.n_fast] - g.spec_xy_m[0])) * k
                         if g.n_lines > 1 and len(g.spec_xy_m) > g.n_fast else None)
                quads = plots.draw_grid_stripes(ax, g, main, second, pos_m * k, None if pos_s is None else pos_s * k,
                                                width, top=self.top)
        except Exception as exc:
            print(f"stripe window: the grid's stripes not drawn ({exc}).")
        ax.plot(xy[acq, 0] * k, xy[acq, 1] * k, ls='none', marker='o', ms=2.0, color='cyan', mec='black', mew=0.3,
                zorder=2.0)
        ax.plot([xy[0, 0] * k], [xy[0, 1] * k], ls='none', marker='s', ms=8, color='white', mec='black', mew=0.8,
                zorder=2.1, label='grid start (anchored)')
        # The handles.
        for key, i, pos in self.handle_positions(res):
            kind = key[0]
            h, = ax.plot([pos[0] * k], [pos[1] * k], ls='none', marker='o' if kind == 'E' else 'D', ms=7,
                         mfc=self.HANDLE_COLORS[kind], mec='black', mew=0.8, zorder=2.5, picker=8, label=key)
            self._handles.append((h, key, i, pos, self.line_direction(res, i)))
        # Limits: everything drawn, edge to edge (plots.fit_limits): the reference topos' frames, the
        # grid's stripes, the kept spectra, the start marker and the handles.
        ax.set_aspect('equal', adjustable='box')
        handles = np.array([h[3] for h in self._handles], dtype=np.float64).reshape(-1, 2) * k
        plots.fit_limits(ax, *corners, quads, xy[acq] * k, xy[:1] * k, handles)
        ax.set_xlabel("x (nm, absolute)")
        ax.set_ylabel("y (nm, absolute)")
        ax.ticklabel_format(useOffset=False, style='plain')
        fast = self.drift.fast_state(g, self.setting)
        src = res.get('source') if res is not None else 'none'
        n_pts = len(self.drift.stripe_points(g, self.setting))
        ax.set_title(f"Stripe matching: {g.name}  [drift {self.setting} ({src}); stripe {fast}"
                     + (f", {n_pts} point(s)" if n_pts else "") + f"; {self.top} pass on top]", fontsize=9)
        self.rad_fast.set(fast)
        self._status()
        self.fig.canvas.draw_idle()

    def _status(self, extra=''):
        """The status text at the bottom: the drift setting and its velocity source, the stripe state and the point count, the lines placed and factors measured, the drag hints, and `extra` (the running offset of a drag)."""
        res = self._res
        lines = [f"drift {self.setting}: " + (f"velocity {res['source']}" if res is not None else "no velocity, no placement"),
                 f"stripe {self.drift.fast_state(self.grid, self.setting)}; "
                 f"{len(self.drift.stripe_points(self.grid, self.setting))} manual point(s)"]
        if res is not None and res.get('lines'):
            fitted = sum(1 for ln in res['lines'] if ln.get('b_f_measured'))
            lines.append(f"{len(res['lines'])} line(s) placed, {fitted} forward factor(s) measured")
        lines.append("drag a handle along its line: circle = forward end, diamond = return point")
        lines.append("'clear points': back to the matching")
        if extra:
            lines.append(extra)
        self.txt.set_text("\n".join(lines))

    # --- the controls ---

    def _on_slow(self, label):
        """'slow axis': the window switches to that setting's stored set and redraws (nothing stored)."""
        if label not in self.drift.SLOW_SETTINGS or label == self.setting:
            return
        self.setting = label
        self.draw()

    def _on_fast(self, label):
        """'fast axis': the set's state stored (Drift.set_fast_state writes the settings store; its notify redraws this window too)."""
        if label not in self.drift.FAST_STATES:
            return
        if label == self.drift.fast_state(self.grid, self.setting):
            return
        self.drift.set_fast_state(self.grid, self.setting, label)      # notify redraws this window too

    def _on_pass(self, label):
        """'stripe pass': the pass drawn on top; redrawn."""
        if label in ('forward', 'reverse') and label != self.top:
            self.top = label
            self.draw()

    def _on_clear(self):
        """'clear points': the set's points dropped (Drift.clear_stripe_points writes the settings store; its notify redraws)."""
        self.drift.clear_stripe_points(self.grid, self.setting)         # notify redraws

    # --- dragging a handle along its line ---

    def _on_pick(self, event):
        """
        A press on a handle starts a drag: on the first drag of a set (fast axis not yet manual, or
        no points) the placement is pinned (Drift.pin_stripe_points, not yet written) and the fast
        axis set to manual in memory; the drag keeps the key, the line's placed direction, the start
        position and coordinate, and the cosine between that direction and the recorded fast axis.
        """
        for h, key, i, pos, u in self._handles:
            if event.artist is h:
                res = self._res
                if res is None or not res.get('lines'):
                    return
                if self.drift.fast_state(self.grid, self.setting) != 'manual' or not self.drift.stripe_points(self.grid, self.setting):
                    # The first drag: every point of the placement pinned, the fast axis manual.
                    self.drift.pin_stripe_points(self.grid, self.setting, res['lines'], commit=False)
                    st = self.drift._stripe_set(self.grid.path, self.setting, create=True)
                    st['fast'] = 'manual'
                pts = self.drift.stripe_points(self.grid, self.setting)
                if key not in pts:
                    return
                fast, _ = self.grid.grid_axes()
                self._drag = (key, u, pos.copy(), float(pts[key]), float(np.dot(u, fast)))
                return

    def _on_motion(self, event):
        """The drag: the cursor's motion from the start projected on the line's direction, the point moved to s0 + projection x cos (Drift.set_stripe_point without commit), the window redrawn with the running offset."""
        if self._drag is None or event.xdata is None or event.ydata is None or event.inaxes is not self.ax:
            return
        key, u, pos0, s0, cos = self._drag
        cursor = np.array([event.xdata, event.ydata], dtype=np.float64) / self.k
        delta = float(np.dot(cursor - pos0, u))                        # the motion along the line only
        self.drift.set_stripe_point(self.grid, self.setting, key, s0 + delta * cos, commit=False)
        self.draw()
        self._status(f"{key}: {delta * 1e10:+.1f} A along the line")

    def _on_release(self, event):
        """The drag ends: the point committed (the settings store written, every window told) and reported."""
        if self._drag is None:
            return
        key, u, pos0, s0, cos = self._drag
        self._drag = None
        pts = self.drift.stripe_points(self.grid, self.setting)
        if key in pts:
            self.drift.set_stripe_point(self.grid, self.setting, key, pts[key], commit=True)   # the settings store + every window
        print(f"stripe matching: {self.grid.name}, drift {self.setting}: {key} at "
              f"{pts.get(key, s0) * 1e10:+.2f} A along the fast axis.")


if __name__ == '__main__':
    # The stripe window on the stripe check's synthetic experiment.
    import io, contextlib, tempfile
    from sm4_Browser.checks import check_stripe as cs, synth
    with contextlib.redirect_stdout(io.StringIO()):
        tmp = tempfile.mkdtemp(prefix='sm4stripe_win_')
        exp, gpath = cs.experiment(tmp, np.array([12e-12, -6e-12]), b_true=(0.92, 1.08))
        synth.install(exp)
        d, load = cs.drift_for(exp)
        g = load(gpath)
    StripeWindow(d, g, 1.0, [False, True], setting='auto')
