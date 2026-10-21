"""
Topography processing: slope subtraction and Δz subtraction of an image (per
scan line) and of a profile, the topography processors (one active at a
time) and their directions, the range matching of averaged topographies,
the processed topography of a file, the Topo window, the raw cross section
of the loaded topos along the browser's section path (cross_section: every
segment of the path in order, each topo at its own scale), the profile window
(ProfileWindow: raw height against position, the forward and reverse
passes of a grid line from the Line window, or the browser's cross
section), and the profile plots (ProfilePlots, ProfilePlot: the pairs sent
from profile windows with 'add new' / 'add existing', overlaid in one axes,
one colour per pair, forward solid, reverse dashed).  The drawing of
topographies lives in plots.py.
"""

if __package__ in (None, ''):
    # Run as a file (Spyder F5): make the package importable from wherever its folder sits.
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np
from sm4_Browser import qtplot as plt
from scipy.ndimage import map_coordinates
from sm4_Browser import plots


def slope_subtract_image(img):
    """
    Linear slope removed from every scan line (row) independently.  NaN
    pixels (not acquired) are ignored in the fit and stay NaN; a row with
    fewer than two finite pixels is left as it is.
    """
    img = np.asarray(img, dtype=np.float64)
    x = np.arange(img.shape[1], dtype=np.float64)
    out = img.copy()
    for i in range(img.shape[0]):
        ok = np.isfinite(img[i, :])
        if np.sum(ok) < 2:
            continue
        p = np.polyfit(x[ok], img[i, ok], 1)
        out[i, ok] = img[i, ok] - np.polyval(p, x[ok])
    return out


def slope_subtract_profile(topo_x, topo_profile):
    """Linear slope subtraction (the line fitted to the finite samples; NaN samples, pixels not kept, stay NaN)."""
    y = np.asarray(topo_profile, dtype=np.float64)
    x = np.asarray(topo_x, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 2:
        return y.copy()
    slope_fit = np.polyfit(x[ok], y[ok], 1)
    out = y.copy()
    out[ok] = y[ok] - np.polyval(slope_fit, x[ok])
    return out


def delta_z_subtract_profile(topo_x, topo_profile):
    """
    Δz subtraction, RHK's line delta z subtract (the image window's Data
    Mode: "the average value of each line is set to zero"): the profile
    minus the mean of its finite samples.  A profile with no finite sample
    is returned as it is.  The offset that follows is left to the
    min-to-zero step that follows every processing.
    """
    y = np.asarray(topo_profile, dtype=np.float64)
    ok = np.isfinite(y)
    if not ok.any():
        return y.copy()
    return y - float(np.mean(y[ok]))


def delta_z_subtract_image(img):
    """
    Δz subtraction of every scan line (row) independently
    (delta_z_subtract_profile on the row: the row minus the mean of its
    acquired pixels); NaN pixels (not acquired) are ignored in the mean and
    stay NaN.
    """
    img = np.asarray(img, dtype=np.float64)
    x = np.arange(img.shape[1], dtype=np.float64)
    out = img.copy()
    for i in range(img.shape[0]):
        out[i, :] = delta_z_subtract_profile(x, img[i, :])
    return out


# The topography processors: one active at a time (the toggles enforce it), Δz sub on by default.
TOPO_PROCESSORS = [
    ("Slope Sub", slope_subtract_image, False),
    ("Δz sub", delta_z_subtract_image, True),
]
# The profile forms of the same processors (the Line window's stripe rows), in the same order.
PROFILE_PROCESSORS = [
    ("Slope Sub", slope_subtract_profile, False),
    ("Δz sub", delta_z_subtract_profile, True),
]

TOPO_CMAP = plots.TOPO_CMAP                      # the topography colour map (plots.py)
TOPO_DIRECTIONS = ('forward', 'reverse', 'average')


def processor_labels():
    return [label for (label, _, _) in TOPO_PROCESSORS]


def processor_defaults():
    return [default for (_, _, default) in TOPO_PROCESSORS]


def process_topo(img, active):
    """Apply the active registry steps in order, then shift the minimum (of the acquired pixels) to zero."""
    out = np.asarray(img, dtype=np.float64).copy()
    for (label, func, _default), on in zip(TOPO_PROCESSORS, active):
        if on:
            out = func(out)
    if np.any(np.isfinite(out)):
        out = out - np.nanmin(out)
    return out


def process_profile(topo_x, profile, active):
    """A profile through the active profile processor (PROFILE_PROCESSORS, one at most), no offset step."""
    q = np.array(profile, dtype=np.float64)
    for (label, func, _default), on in zip(PROFILE_PROCESSORS, active):
        if on:
            q = func(topo_x, q)
    return q


def toggle_processor(active, index):
    """
    The processor flags after a click on entry `index`: one active at most.
    A click on an inactive entry turns it on and every other off; a click on
    the active entry turns it off (none active means no processing).
    Returns the new list of flags.
    """
    active = list(active)
    if active[index]:
        active[index] = False
    else:
        active = [False] * len(active)
        active[index] = True
    return active


def match_ranges(arrays):
    """
    Range matching of topographies to be averaged: the array with the largest
    finite range (nanmax minus nanmin) is the reference and is returned as it
    is; every other array a with range R_a > 0 is mapped linearly so that its
    minimum and maximum equal the reference's,
    ref_min + (a - a_min) * R_ref / R_a; a flat array (R_a = 0) becomes
    ref_min everywhere (NaN pixels stay NaN).  Works on 1D profiles and 2D
    images alike.  Returns the list of matched arrays in the given order.
    """
    arrays = [np.asarray(a, dtype=np.float64) for a in arrays]
    if not arrays:
        return []
    stats = []
    for a in arrays:
        if np.any(np.isfinite(a)):
            lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        else:
            lo, hi = np.nan, np.nan
        stats.append((lo, hi, hi - lo))
    ranges = [r if np.isfinite(r) else -np.inf for _, _, r in stats]
    ref = int(np.argmax(ranges))
    ref_lo, ref_hi, ref_r = stats[ref]
    if not np.isfinite(ref_r):
        return [a.copy() for a in arrays]
    out = []
    for k, a in enumerate(arrays):
        lo, hi, r = stats[k]
        if k == ref or not np.isfinite(r):
            out.append(a.copy())
        elif r > 0:
            out.append(ref_lo + (a - lo) * (ref_r / r))
        else:
            b = np.full(a.shape, ref_lo)
            b[~np.isfinite(a)] = np.nan
            out.append(b)
    return out


def stripe_profile(topo_x, profiles, active=None):
    """
    The stripe profile of raw row profiles (the Line and line-average
    windows): every profile through the active processor (one of
    PROFILE_PROCESSORS at most; `active` None takes the registry defaults),
    then range matching to the profile with the largest range
    (match_ranges), then the mean, then minus its minimum.
    """
    if active is None:
        active = processor_defaults()
    processed = [process_profile(topo_x, p, active) for p in profiles]
    matched = np.asarray(match_ranges(processed), dtype=np.float64)
    with np.errstate(invalid='ignore'):
        profile = np.nanmean(matched, axis=0) if np.isfinite(matched).any() else np.full(matched.shape[1], np.nan)
    if np.isfinite(profile).any():
        profile = profile - np.nanmin(profile)
    return profile


def processed_topo(sm4, direction, active):
    """
    The processed topography of a file for a direction: 'forward' or
    'reverse' is process_topo of that pass; 'average' processes the two
    passes separately, matches their ranges (match_ranges), averages them
    pixelwise (a pixel acquired in one pass only keeps that pass), and shifts
    the minimum to zero.  A file with a single topographic page gives its
    forward pass for 'average'.
    """
    if direction == 'average':
        if len(sm4.topo_pages) < 2:
            return process_topo(sm4.topo_raw('forward'), active)
        f = process_topo(sm4.topo_raw('forward'), active)
        r = process_topo(sm4.topo_raw('reverse'), active)
        f, r = match_ranges([f, r])
        both = np.isfinite(f) & np.isfinite(r)
        out = np.where(np.isfinite(f), f, r)
        out[both] = 0.5 * (f[both] + r[both])
        if np.any(np.isfinite(out)):
            out = out - np.nanmin(out)
        return out
    return process_topo(sm4.topo_raw(direction), active)


def grid_stripe_layers(grid, direction, active, pix=None):
    """
    The grid's own topo pixels as the scan area and the Topo window draw
    them (plots.draw_grid_stripes): the images through the active
    processors and every pixel's position (absolute metres, NaN for a pixel
    not drawn).  Positions: recorded (Sm4File.pixel_xy_m; a forward pixel
    only where its kept spectrum sits, found by location,
    Sm4File.spectrum_pixels) with `pix` None (the drift toggles off), else
    the drift's corrected positions (`pix` = Drift result 'pix').
    `direction` is the 'topo pass' choice for the grid's topo, which
    applies with `pix` None (the drift off): 'forward' or 'reverse' gives
    that pass alone ('average' counts as forward: the two passes of a grid
    are never averaged, averaging is for topo images).  With `pix` given
    (the drift on) both passes are always returned at their own corrected
    positions (the forward pass by its points' times, the reverse rigid at
    the forward end, see drift.Drift.grid_chain), whatever `direction`
    says; the caller draws the pass of its choice on top
    (plots.draw_grid_stripes).  Returns (main image, second image or None,
    main positions, second positions or None); (None, None, None, None)
    without topo pages.
    """
    if not grid.topo_pages:
        return None, None, None, None
    two = len(grid.topo_pages) > 1
    if pix is not None and pix.get('forward') is not None:
        pos_f = np.asarray(pix['forward'], dtype=np.float64)
        pos_r = None if pix.get('reverse') is None else np.asarray(pix['reverse'], dtype=np.float64)
        if two and pos_r is not None:
            return (process_topo(grid.topo_raw('forward'), active), process_topo(grid.topo_raw('reverse'), active),
                    pos_f, pos_r)
        return process_topo(grid.topo_raw('forward'), active), None, pos_f, None
    else:
        pos_f = grid.pixel_xy_m('forward')
        idx, _ = grid.spectrum_pixels('forward')
        acq = np.asarray(grid.acquired, dtype=bool)[:len(idx)]
        drawn = np.zeros(pos_f.shape[:2], dtype=bool)
        for k in np.flatnonzero(acq & (idx[:, 0] >= 0)):
            drawn[idx[k, 0], idx[k, 1]] = True
        pos_f = np.where(drawn[..., None], pos_f, np.nan)
        pos_r = grid.pixel_xy_m('reverse') if two else None
    if direction == 'reverse' and two and pos_r is not None:
        return process_topo(grid.topo_raw('reverse'), active), None, pos_r, None
    return process_topo(grid.topo_raw('forward'), active), None, pos_f, None


class TopoWindow:
    """
    The most recent topography in its own scan frame, aspect locked, with
    its own 'auto drift' and 'manual drift' toggles (bottom left; set from
    the browser's states at opening and independent after that): with
    either on, the image is skewed pixel by pixel by the drift during its
    scan (Drift.corrected_topo, the velocity from the sources on) and the
    spectra file shown with it (a single point's sweeps, or a grid's
    spectra) gets its corrected positions and its drift lines (a marker at
    every sweep's start and a line to its end, in the auto or the manual
    colour; the sweeps of a single point connected dot to dot); with both
    off, the processed image and the recorded positions.  A grid's own topo
    pixels are drawn over the image at their positions (grid_stripe_layers)
    as the scan area draws them, the pass chosen by the window's 'topo
    pass' radio ('average' counts as forward: a grid's two passes are never
    averaged): with both drift toggles off that pass alone at the recorded
    positions; with a source on both passes at their corrected positions,
    the pass the radio names on top.  'stripe match' (grid files) opens the
    stripe window (stripe.StripeWindow); the grid's stored fast-axis state
    for the setting in force places the lines (Drift.result with stripe
    None); the browser's anchor_on state is taken at opening (no toggle of
    its own here; no window builds the 'anchor to topo' toggle today, so it
    is off).  Its colour bar carries the height scale (nm).  Every length
    is drawn at `scalefactor`: the window takes the scale the caller gives
    and keeps it (the browser passes the current file's).
      topo        : the topography (Sm4File)
      active_procs: the topo processor flags (TOPO_PROCESSORS, one on at most)
      scalefactor : the scale factor the frame and every position are drawn at
                    (nm = metres x 1e9 / scalefactor)
      marker_xy_m : (kept for callers without `spec`) positions (n, 2) drawn as
                    a point's sweeps
      title       : the title's file text (the topo's name when None)
      direction   : the topo's pass shown ('forward', 'reverse' or 'average')
      spec        : the spectra file drawn on it (Sm4File of kind 'point', 'grid',
                    or 'phase'), or None
      drift       : the directory's Drift, or None (no correction possible; the
                    toggles then start off and cannot be turned on)
      auto_on, manual_on : the toggles' starting states
      anchor_on   : the state taken from the browser (always False today)
      spec_dir    : the grid's topo pass at opening ('forward' or 'reverse')
      stripe_on   : accepted and unused
    """

    DESIGN = (7.5, 7.4)    # the design size (inches): the text zoom of the window is its size relative to this (qtplot.window_zoom)

    def __init__(self, topo, active_procs, scalefactor, marker_xy_m=None, title=None, direction='forward',
                 spec=None, drift=None, auto_on=False, manual_on=False, anchor_on=False,
                 spec_dir='forward', stripe_on=None):
        from sm4_Browser import widgets
        self.topo = topo
        self.active = list(active_procs)
        self.scalefactor = float(scalefactor)
        self.direction = direction
        self.title = title or topo.name
        self.spec = spec
        self.marker_xy_m = marker_xy_m
        self.drift = drift
        self.auto_on = bool(auto_on and drift is not None)
        self.manual_on = bool(manual_on and drift is not None)
        self.anchor_on = bool(anchor_on)
        self.spec_dir = spec_dir if spec_dir in ('forward', 'reverse') else 'forward'
        self.stripe_window = None
        self.fig, self.ax = plt.subplots(figsize=self.DESIGN)
        self.fig.subplots_adjust(bottom=0.17)
        self.cbar = None
        self.chk_auto = widgets.make(self.fig, 'auto drift', 0.02, 0.015, value=self.auto_on)
        self.chk_auto.on_change(lambda flag: self._on_drift_toggle('auto', flag))
        self.chk_manual = widgets.make(self.fig, 'manual drift', 0.18, 0.015, value=self.manual_on)
        self.chk_manual.on_change(lambda flag: self._on_drift_toggle('manual', flag))
        self.btn_stripe = widgets.make(self.fig, 'stripe match', 0.36, 0.015)
        self.btn_stripe.on_change(self._on_stripe_window)
        is_grid = spec is not None and spec.kind == 'grid' and bool(spec.topo_pages)
        self.btn_stripe.show(is_grid and drift is not None)
        # The grid's own topo pass (the 'topo pass' radio of the scan area, for the grid's topo).
        self.txt_dir = self.fig.text(0.80, 0.135, 'line/map topo', fontsize=8, ha='left', va='bottom', visible=is_grid)
        self.rad_dir = widgets.make(self.fig, 'topo pass', 0.80, 0.015, value=self.spec_dir)
        self.rad_dir.on_change(self._on_spec_dir)
        self.rad_dir.show(is_grid)
        self.draw()
        plt.show()

    def _drift_active(self):
        return self.drift is not None and (self.auto_on or self.manual_on)

    def _image(self):
        """The image drawn (nm): skewed by the drift during its scan with a source on (Drift.corrected_topo; the plain processed image when that fails, said on the console), else processed_topo."""
        if self._drift_active():
            try:
                return self.drift.corrected_topo(self.topo, self.direction, self.active, None,
                                                 self.auto_on, self.manual_on) * 1e9
            except Exception as exc:
                print(f"Topo window: drift correction of the image not applied ({exc}).")
        return processed_topo(self.topo, self.direction, self.active) * 1e9

    def draw(self):
        """
        Everything drawn anew: the image in its scan frame with the height colour bar, the spectra
        file's positions in that frame (corrected with a source on, with their drift lines in the
        source's colour), a grid's own topo pixels, and the title with the pass and drift state.
        """
        ax = self.ax
        ax.clear()
        if self.cbar is not None:
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None
        img = self._image()
        im = plots.draw_topo_frame(ax, self.topo, img, self.scalefactor, cmap=TOPO_CMAP)
        self.cbar = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="height (nm)")
        k = 1e9 / self.scalefactor
        f = self.spec
        corrected = False
        source = None
        d = None
        if f is not None and f.kind in ('point', 'grid', 'phase') and len(f.spec_xy_m):
            xy = f.spec_xy_m
            lines = None
            stripe = (f.kind == 'grid' and self.drift is not None
                      and self.drift.fast_state(f, None, self.auto_on, self.manual_on) != 'off')
            if (self._drift_active() or stripe) and f.kind != 'phase':
                # The grid's stored fast-axis state places the lines, with drift none as well.
                d = self.drift.result(f, None, None, self.auto_on, self.manual_on, self.anchor_on)
                if d is not None:
                    xy = np.where(np.isfinite(d['xy_corr']), d['xy_corr'], f.spec_xy_m)
                    # Every sweep's drift line through its sweep (Drift.sweep_lines), in the frame.
                    L = self.drift.sweep_lines(f, None, None, self.auto_on, self.manual_on, self.anchor_on)
                    if L is None:
                        L = np.stack([d['xy_corr'], d['xy_end']], axis=1)
                    lines = (self.topo.to_frame_m(L.reshape(-1, 2)) * k).reshape(L.shape)
                    corrected = True
                    source = d.get('source')
            uv = self.topo.to_frame_m(xy) * k
            color = plots.drift_color(source)
            if f.kind == 'grid':
                acq = np.asarray(f.acquired, dtype=bool)[:len(uv)]
                # The grid's own topo pixels over the image, in the frame (the scan area's stripes).
                if f.topo_pages:
                    try:
                        pix = d['pix'] if (corrected and d is not None and d.get('pix')) else None
                        main, second, pos_m, pos_s = grid_stripe_layers(f, self.spec_dir, self.active, pix)
                        if main is not None:
                            to_frame = lambda p: (self.topo.to_frame_m(np.asarray(p).reshape(-1, 2)) * k).reshape(np.asarray(p).shape)
                            width = (float(np.linalg.norm(f.spec_xy_m[f.n_fast] - f.spec_xy_m[0])) * k
                                     if f.n_lines > 1 and len(f.spec_xy_m) > f.n_fast else None)
                            plots.draw_grid_stripes(ax, f, main, second, to_frame(pos_m),
                                                    None if pos_s is None else to_frame(pos_s), width,
                                                    top=self.spec_dir)
                    except Exception as exc:
                        print(f"Topo window: the grid's topo pixels not drawn ({exc}).")
                ax.plot(uv[acq, 0], uv[acq, 1], ls='none', marker='o', ms=2.5, color='cyan', mec='black', mew=0.3,
                        label=f'spectra ({int(acq.sum())} kept)')
                if lines is not None:
                    plots.draw_drift_lines(ax, lines[acq], color=color)
                ax.legend(loc='upper right', frameon=False, fontsize=8)
            else:
                show = uv if corrected or f.kind == 'phase' else uv[:1]
                plots.draw_point_sweeps(ax, show, corrected=corrected, lines=lines, color=color)
        elif self.marker_xy_m is not None:
            uv = self.topo.to_frame_m(self.marker_xy_m) * k
            plots.draw_point_sweeps(ax, uv, corrected=len(uv) > 1)
        ax.set_xlabel("x (nm, scan frame)")
        ax.set_ylabel("y (nm, scan frame)")
        tag = ""
        if self._drift_active():
            tag = "  drift corrected" + (f" ({source})" if source else "") + ("  stripe matched" if (
                d is not None and d.get('stripe')) else "")
        if f is not None and f.kind == 'grid' and f.topo_pages:
            tag += f"  grid topo [{self.spec_dir}]"
        ax.set_title("Topography: " + self.title + f"  [{self.direction}]" + tag, fontsize=9)
        self.fig.canvas.draw_idle()

    def _on_drift_toggle(self, which, flag):
        """The window's 'auto drift' / 'manual drift' toggles: on only with a drift model at hand (else the toggle goes back)."""
        chk = {'auto': self.chk_auto, 'manual': self.chk_manual}[which]
        if flag and self.drift is None:
            print("Topo window: no drift model at hand; the toggle stays off.")
            chk.set(False)
            self.fig.canvas.draw_idle()
            return
        if which == 'auto':
            self.auto_on = bool(flag)
        else:
            self.manual_on = bool(flag)
        self.draw()

    def _on_stripe_window(self):
        """'stripe match': the grid's stripe window (stripe.StripeWindow), one at a time, opened at the setting in force here."""
        f = self.spec
        if self.drift is None or f is None or f.kind != 'grid' or not f.topo_pages:
            print("Topo window: no grid with topo pages, or no drift model; no stripe window.")
            return
        w = self.stripe_window
        if w is not None and w.is_open():
            w.raise_()
            return
        from sm4_Browser.stripe import StripeWindow
        try:
            self.stripe_window = StripeWindow(self.drift, f, self.scalefactor, list(self.active),
                                              setting=self.drift.slow_setting(f, self.auto_on, self.manual_on),
                                              pass_top=self.spec_dir)
        except ValueError as exc:
            print(f"Topo window: stripe window not opened ({exc}).")

    def _on_spec_dir(self, label):
        """The 'topo pass' radio: the grid's own topo pass drawn alone (drift off) or on top (drift on); 'average' counts as forward."""
        if label in TOPO_DIRECTIONS:
            self.spec_dir = 'reverse' if label == 'reverse' else 'forward'
            self.draw()


# ---------------------------------------------------------------------- #
#  Raw profiles: the cross section of the loaded topos, the profile window #
# ---------------------------------------------------------------------- #

def _sample_raw(topo, img, xy_m):
    """Raw heights of a topo page at absolute positions (M, 2), bilinear, NaN outside the page or on unacquired pixels."""
    uv = topo.to_frame_m(xy_m)
    j = uv[:, 0] / topo.xscale_m + (topo.nx - 1) / 2.0
    i = uv[:, 1] / topo.yscale_m + (topo.ny - 1) / 2.0
    return map_coordinates(np.asarray(img, dtype=np.float64), [i, j], order=1, mode='constant', cval=np.nan)


def cross_section(topos, points, scale_of):
    """
    The raw cross section of the loaded topographies along a path: `points`
    (n, 2) in the scan area's drawing units (calibrated nm), the path's
    points in order, two or more.  Samples along the path at the finest
    pixel pitch among the topos given (each pitch in drawing units at that
    topo's own scale; a hundredth of the path's length without any topo with
    an image); at every sample the topmost topo whose footprint
    contains it (`topos` oldest first, as the browser draws them: the last
    one containing the sample wins) supplies the raw height (Sm4File.topo_raw,
    in metres as the loader gives it, no processor, no normalization), its
    forward page and, when the file has one, its reverse page; NaN where no
    topo or no acquired pixel lies under the sample.  A sample is converted
    into each topo's recorded frame with that topo's scale, scale_of(path)
    (the scale in force at the topo's time, scales.Scales; or one number for
    every topo), so files of different scales are cut where they truly lie.
    Returns (x: the distance along the path from its first point, calibrated
    nm, z_forward (nm), z_reverse (nm, all NaN without reverse pages), the
    names of the topos that supplied samples).
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        raise ValueError("a cross section needs a path of two or more points")
    sc = scale_of if callable(scale_of) else (lambda p, v=float(scale_of): v)
    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    L = float(seg_len.sum())
    pitches = [min(abs(t.xscale_m), abs(t.yscale_m)) * 1e9 / sc(t.path) for t in topos if t.nx and t.ny]
    pitch = min(pitches) if pitches else L / 100.0
    n = max(int(round(L / pitch)) + 1, 2) if L > 0 and pitch > 0 else 2
    s = np.linspace(0.0, L, n)
    # Every sample on the path: the segment it falls in, and its position along it.
    starts = np.concatenate([[0.0], np.cumsum(seg_len)])
    idx = np.clip(np.searchsorted(starts, s, side='right') - 1, 0, len(seg) - 1)
    frac = np.where(seg_len[idx] > 0, (s - starts[idx]) / np.where(seg_len[idx] > 0, seg_len[idx], 1.0), 0.0)
    draw = pts[idx] + frac[:, None] * seg[idx]
    zf = np.full(n, np.nan)
    zr = np.full(n, np.nan)
    owner = np.full(n, -1)
    names = []
    for i in range(len(topos) - 1, -1, -1):              # the topmost first
        t = topos[i]
        free = owner < 0
        if not free.any():
            break
        p_m = draw * (sc(t.path) * 1e-9)                   # this topo's recorded metres
        inside = np.array([t.contains_m(p) for p in p_m]) & free
        if not inside.any():
            continue
        zf[inside] = _sample_raw(t, t.topo_raw('forward'), p_m[inside]) * 1e9
        if len(t.topo_pages) > 1:
            zr[inside] = _sample_raw(t, t.topo_raw('reverse'), p_m[inside]) * 1e9
        owner[inside] = i
        names.append(t.name)
    return s, zf, zr, names


class ProfileWindow:
    """
    Raw height against position in one axes: the forward and reverse
    passes of a grid line (the Line window's 'profile' button) or the
    browser's cross section, as curves (label, x, z, colour) through
    plots.draw_profiles; nothing processed or normalized.  update(curves,
    title, pair) redraws it with new curves.  With a ProfilePlots registry
    the window carries 'add new' and 'add existing' (the browser's buttons):
    'add new' opens a profile plot holding the window's pair; 'add existing'
    drops the list of the open profile plots (widgets.AveragePicker) and
    adds the pair to the chosen one.  The pair is {'label', 'forward': (x,
    z), 'reverse': (x, z) or None}, the curves as shown.  Closed by hand it
    is simply gone; is_open() says whether it still exists.
    """

    DESIGN = (8.5, 4.2)    # the design size (inches): the text zoom of the window is its size relative to this (qtplot.window_zoom)

    def __init__(self, curves, title, xlabel="position (nm)", ylabel="height (nm, raw)", pair=None, registry=None):
        from sm4_Browser import widgets
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.registry = registry
        self.pair = pair
        self.fig, self.ax = plt.subplots(figsize=self.DESIGN)
        self.fig.subplots_adjust(bottom=0.2 if registry is not None else 0.15, left=0.09, right=0.98, top=0.9)
        try:
            self.fig.canvas.manager.set_window_title(title)
        except Exception:
            pass
        self.btn_new = self.btn_existing = self.picker = None
        if registry is not None:
            self.btn_new = widgets.make(self.fig, 'add new', 0.80, 0.02)
            self.btn_existing = widgets.make(self.fig, 'add existing', 0.875, 0.02)
            self.btn_new.on_change(self._on_add_new)
            self.btn_existing.on_change(self._on_add_existing)
            self.picker = widgets.AveragePicker(self.fig, registry)
        self.update(curves, title, pair)
        plt.show()

    def is_open(self):
        from sm4_Browser import widgets
        return widgets.figure_is_open(self.fig)

    def update(self, curves, title, pair=None):
        if pair is not None:
            self.pair = pair
        plots.draw_profiles(self.ax, curves, self.xlabel, self.ylabel, title)
        self.fig.canvas.draw_idle()

    def _on_add_new(self):
        if self.picker is not None:
            self.picker.close()
        if self.pair is None:
            print("add new: no profile pair in this window.")
            return
        w = self.registry.new(self.pair)
        print(f"{w.name}: opened with {self.pair['label']}.")

    def _on_add_existing(self):
        if self.pair is None:
            print("add existing: no profile pair in this window.")
            return
        if self.picker.is_open():
            self.picker.close()
            return
        self.picker.open(self.btn_existing.ax, self.pair['label'], self._on_pick)

    def _on_pick(self, index):
        w = self.registry.add(index, self.pair)
        if w is not None:
            print(f"{w.name}: {self.pair['label']} added ({len(w.entries)} pair(s)).")


class ProfilePlots:
    """
    The registry of the profile plots (one per experiment browser, handed to
    the Line windows): profile1, profile2, ... by index.  new(pair) opens
    the next plot with a pair; add(index, pair) adds to an open one; open()
    gives {index: window} of the plots still open (the picker's list, with
    each window's name, entries, and has(label)).
    """

    def __init__(self):
        self.windows = {}

    def open(self):
        return {i: w for i, w in self.windows.items() if w.is_open()}

    def next_index(self):
        return (max(self.windows) + 1) if self.windows else 1

    def new(self, pair):
        w = ProfilePlot(self.next_index())
        self.windows[w.index] = w
        w.add(pair)
        return w

    def add(self, index, pair):
        w = self.open().get(index)
        if w is None:
            print(f"add existing: profile{index} is not open.")
            return None
        w.add(pair)
        return w


class ProfilePlot:
    """
    One profile plot: every pair added (ProfileWindow's 'add new' / 'add
    existing') overlaid in one axes, position against raw height, as the
    pairs are (no alignment, no offset), one colour per pair in the order
    added, the forward pass solid and the reverse dashed
    (plots.draw_profile_pairs), the legend naming each pair's source.
    `entries` is the list of pairs; has(label) says whether a pair of that
    label is in; name is 'profile<index>'.
    """

    DESIGN = (9.0, 4.6)    # the design size (inches): the text zoom of the window is its size relative to this (qtplot.window_zoom)

    def __init__(self, index):
        from sm4_Browser import widgets
        self.index = int(index)
        self.name = f"profile{self.index}"
        self.entries = []
        self.fig, self.ax = plt.subplots(figsize=self.DESIGN)
        self.fig.subplots_adjust(bottom=0.14, left=0.09, right=0.98, top=0.92)
        try:
            self.fig.canvas.manager.set_window_title(self.name)
        except Exception:
            pass
        self._widgets = widgets
        plt.show()

    def is_open(self):
        return self._widgets.figure_is_open(self.fig)

    def has(self, label):
        return any(p.get('label') == label for p in self.entries)

    def add(self, pair):
        self.entries.append({'label': pair.get('label', ''), 'forward': pair.get('forward'), 'reverse': pair.get('reverse')})
        self.redraw()

    def redraw(self):
        plots.draw_profile_pairs(self.ax, self.entries, "position (nm)", "height (nm, raw)",
                                 f"Height profiles: {self.name}, {len(self.entries)} pair(s), forward solid, reverse dashed")
        self.fig.canvas.draw_idle()


if __name__ == '__main__':
    from sm4_Browser import sm4file
    path = r'experiment_dir/file.sm4'
    window = TopoWindow(sm4file.Sm4File(path), [False, True], 2.75)   # Slope Sub off, Δz sub on (one flag per processor)
