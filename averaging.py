"""
Averaging of spectra: the one averaging method of the package and the
registry of an experiment's open averaging windows (Averages), a Selection
(chosen spectra of a grid file as an averaging entry), the resampling
helpers that bring files of another energy step onto one grid (grid_steps,
same_step, common_range, grid_map, to_grid, point_weights), the avgN
averaging window (MultiAverageWindow), and send_sets_to_fit, the send of
an average built from several raw sets to the single-point fit
(used by the averaging and the line-average windows).
"""

if __package__ in (None, ''):
    # Run as a file (Spyder F5): make the package importable from wherever its folder sits.
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from datetime import datetime
import numpy as np
from sm4_Browser import qtplot as plt
from scipy.integrate import cumulative_trapezoid
from sm4_Browser import sm4file, fit, widgets, records
from sm4_Browser.sts import STS, curve_units_of
from sm4_Browser.feenstra import (feenstra_width, feenstra_titles, energy_has_zero, iv_corrected, iv_broadened,
                                  FeenstraWindow)


# ---------------------------------------------------------------------- #
#  Bringing files of another energy step onto one grid (the avgN run)      #
# ---------------------------------------------------------------------- #

def grid_steps(axes):
    """The median energy step of every axis in the list."""
    return [float(np.median(np.diff(V))) for V in axes]


def same_step(steps, names, tol):
    """With 'bin' and 'interp' off every file must have the first file's step (ValueError otherwise)."""
    for name, st in zip(names, steps):
        if abs(st - steps[0]) > tol:
            raise ValueError(f"{name} has an energy step of {st * 1e3:.3f} meV, "
                             f"{names[0]} has {steps[0] * 1e3:.3f} meV ('bin' or 'interp' brings "
                             f"files of different steps onto one grid).")


def common_range(axes, step):
    """(lo, hi) of the energy range the axes share; ValueError when it is shorter than a step."""
    lo = max(float(V[0]) for V in axes)
    hi = min(float(V[-1]) for V in axes)
    if hi - lo < step:
        raise ValueError(f"the files share no energy range (overlap {lo:.3f} to {hi:.3f} V).")
    return lo, hi


def grid_map(V, V0, lo, hi, step, st, method, tol, name, ref_name, label):
    """
    The map that brings one file's columns onto the run's grid V0 (see
    to_grid) and the note for the console: a file at the grid's step must
    coincide with the grid points on the overlap ({'kind': 'mask'}); with
    'bin' its points are averaged inside half a grid step either side of
    each grid point (refused if a bin would be empty); with 'interp' they
    are interpolated linearly at the grid points.  ValueError when refused.
    """
    if abs(st - step) <= tol:
        m = (V >= lo - tol) & (V <= hi + tol)
        if int(m.sum()) != len(V0) or np.max(np.abs(V[m] - V0)) > tol:
            raise ValueError(f"the energy values of {name} do not coincide with those "
                             f"of {ref_name} on the overlap {lo:.3f} to {hi:.3f} V.")
        return {'kind': 'mask', 'mask': m}, f"{label} on the grid"
    if method == 'bin':
        bins = [np.nonzero((V >= g - 0.5 * step - tol) & (V < g + 0.5 * step - tol))[0] for g in V0]
        sizes = [len(rows) for rows in bins]
        if min(sizes) == 0:
            raise ValueError(f"{name} ({st * 1e3:.3f} meV) leaves a bin of the "
                             f"{step * 1e3:.3f} meV grid empty.")
        mp = {'kind': 'bin', 'bins': bins, 'counts': np.array(sizes, dtype=np.float64),
              'weight': step / st}
        return mp, (f"{label} ({st * 1e3:.3f} meV) binned, {min(sizes)} to {max(sizes)} points per bin, "
                    f"each point weighted {step / st:.3g}")
    mp = {'kind': 'interp', 'V': V, 'grid': V0, 'weight': step / st}
    return mp, (f"{label} ({st * 1e3:.3f} meV) interpolated, each point weighted "
                f"{step / st:.3g}")


def to_grid(columns, mp):
    """
    Columns of one entry (npts_file, n) brought onto the run's grid by its
    map mp (see grid_map): {'kind': 'mask', 'mask': m} takes the rows of
    the common range; {'kind': 'bin', 'bins': [row indices per grid
    point], 'counts': points per bin, 'weight': grid step / file step}
    averages the rows of each bin; {'kind': 'interp', 'V': axis, 'grid':
    V, 'weight': grid step / file step} interpolates every column
    linearly at the grid points.
    """
    c = np.asarray(columns, dtype=np.float64)
    if mp['kind'] == 'mask':
        return c[mp['mask'], :]
    if mp['kind'] == 'bin':
        return np.vstack([c[rows, :].mean(axis=0) for rows in mp['bins']])
    return np.column_stack([np.interp(mp['grid'], mp['V'], c[:, j]) for j in range(c.shape[1])]) \
        if c.shape[1] else np.zeros((len(mp['grid']), 0))


def point_weights(mp, n):
    """
    Weight of every grid point of a file's sweeps in the average (n,):
    the file's measured points per grid step, i.e. grid step / file step
    for a file binned (above 1) or interpolated (below 1) onto the grid,
    1 for a file on the grid.  The same constant at every grid point: a
    binned point is not weighted by its own bin's count, which alternates
    (2, 3, 2, 3 for 10 into 25 meV) and would put a sawtooth on the
    average.
    """
    if mp is not None and mp['kind'] in ('bin', 'interp'):
        return np.full(n, mp['weight'])
    return np.ones(n)


# ---------------------------------------------------------------------- #
#  Selection: chosen spectra of a grid file as an averaging entry           #
# ---------------------------------------------------------------------- #

class Selection:
    """
    Chosen spectra of a grid file (the columns at a marker group of the Line
    or Map window) with the file's identity, behaving as a single-point file
    for the averaging window: MultiAverageWindow.add, the records, reload,
    and the send read the same attributes a Sm4File has.  Every marker's
    column is one sweep of the entry.

    sm4   : the grid file's Sm4File
    cols  : the chosen columns as global indices of the file's spectra, in
            the order given
    where : {'line': k or None, 'positions': [...]}: the Line window gives
            its chunk index and the marker positions in Å along the line, the
            Map window None and [[x_nm, y_nm], ...] of the marker pixels
    """

    MAX_PLACES = 4         # places named in entry_label before '+n'

    def __init__(self, sm4, cols, where):
        if sm4.spectra is None:
            raise ValueError(f"{sm4.name} has no spectra.")
        self.file = sm4
        self.path = sm4.path
        self.name = sm4.name
        self.stamp = sm4.stamp
        self.kind = 'selection'
        self.cols = np.asarray(cols, dtype=int)
        self.where = {'line': None, 'positions': []} if where is None else dict(where)
        self.spectra = sm4.spectra.subset(self.cols)
        self.key = (self.path, tuple(int(c) for c in self.cols))

    # --- pass-throughs, as Sm4File has them ---

    @property
    def energy(self):
        return self.spectra.energy

    @property
    def lia(self):
        return self.spectra.lia

    @property
    def iets(self):
        return self.spectra.iets

    @property
    def iv(self):
        return self.spectra.iv

    @property
    def zscale_iv(self):
        return self.spectra.zscale_iv

    @property
    def units_lia(self):
        return self.spectra.units_lia

    @property
    def units_iets(self):
        return self.spectra.units_iets

    @property
    def units_iv(self):
        return self.spectra.units_iv

    @property
    def vmod(self):
        return self.spectra.vmod

    @property
    def V_bias(self):
        return self.spectra.V_bias

    @property
    def I_setpoint(self):
        return self.spectra.I_setpoint

    @property
    def n_spectra(self):
        return self.spectra.n_spectra

    @property
    def n_acquired(self):
        return self.spectra.n_acquired

    @property
    def acquired(self):
        return self.spectra.acquired

    @property
    def spec_xy_m(self):
        return self.spectra.spec_xy_m

    @property
    def spec_time_s(self):
        return self.spectra.spec_time_s

    def channels(self):
        return self.spectra.channels()

    def curves(self, hampel=None, sgolay=None, divide_iv=False, iv_broadening=0.0, channel='didv',
               iv_zero_sub=False, normalize=True, calibrate=True):
        """The chosen columns through the pipeline (npts, n), original energy order (STS.process)."""
        return self.spectra.process(channel, hampel, sgolay, divide_iv, iv_broadening, iv_zero_sub=iv_zero_sub,
                                    normalize=normalize, calibrate=calibrate)

    @property
    def where_text(self):
        """'line 1: 12.3, 14.8 Å' or '(1.23, 4.56), (1.30, 4.56) nm' (MAX_PLACES named, then '+n')."""
        pos = list(self.where.get('positions') or [])
        shown = pos[:self.MAX_PLACES]
        more = len(pos) - len(shown)
        tail = f" +{more}" if more > 0 else ""
        line = self.where.get('line')
        if line is not None:
            body = ", ".join(f"{float(p):.1f}" for p in shown)
            return f"line {int(line)}: {body}{tail} Å"
        body = ", ".join(f"({float(p[0]):.2f}, {float(p[1]):.2f})" for p in shown)
        return f"{body}{tail} nm"

    @property
    def entry_label(self):
        """The entry's label in an averaging window: the file's time of day and the place."""
        return f"{sm4file.stamp_text(self.stamp)[11:]} {self.where_text}"

    def record_entry(self):
        """The fields a record stores for this entry beyond path, name, n_sweeps."""
        return {'kind': 'selection', 'cols': [int(c) for c in self.cols],
                'sweeps': getattr(self.file, 'sweep_mode', None),      # the sweep view the columns index
                'where': {'line': None if self.where.get('line') is None else int(self.where['line']),
                          'positions': [([float(v) for v in p] if np.ndim(p) else float(p))
                                        for p in (self.where.get('positions') or [])]}}


# ---------------------------------------------------------------------- #
#  The averaging method and the registry                                   #
# ---------------------------------------------------------------------- #

class Averages:
    """
    The one averaging method of the package and the registry of an
    experiment's open averaging windows.

    Math (one copy; the avgN window, the Line, Map, and line-average
    windows, and the fit all reach it): average(cols, weights) is
    the weighted pooled mean of per-set column arrays, the plain mean of
    every pooled column when every weight is 1; mean_rows(rows) the plain
    mean of curves given as rows.  The line-average windows' mean over the
    members covering each pixel is align.average_stack.

    Registry: the experiment directory (its 'averages' subfolder holds the
    records and fit histories), the defaults handed to new windows (the
    browser's: the Hampel / SG box values, fit_settings, the Feenstra
    state and settings, the channel), and the open MultiAverageWindows by
    index.  The browser owns one and hands it to the Line and Map windows;
    a Line or Map run stand-alone makes its own on the file's folder.
    """

    def __init__(self, directory, hampel_defaults, sgolay_defaults, fit_settings=None,
                 divide_iv=True, iv_broadening=0.0, channel='didv', iv_zero_sub=True, iv_broadened=False):
        self.directory = os.path.abspath(directory)
        self.hampel_defaults = hampel_defaults
        self.sgolay_defaults = sgolay_defaults
        self.fit_settings = dict(fit_settings) if fit_settings else {}
        self.divide_iv = bool(divide_iv)
        self.iv_broadening = float(iv_broadening)
        self.iv_zero_sub = bool(iv_zero_sub)       # the Feenstra settings handed to new windows (the browser's)
        self.iv_broadened = bool(iv_broadened)
        self.channel = channel if channel in STS.CHANNELS else 'didv'
        self.windows = {}          # index -> MultiAverageWindow (closed ones are dropped by open())

    # --- the averaging math ---

    @staticmethod
    def average(cols, weights=None):
        """
        The average of per-set column arrays (npts, n) in a list: every point
        of every column weighted by its set's weight per point (weights: one
        (npts,) array per set, 1 for a set on the axis; None or a list of
        another length means every weight 1), the plain mean of the pooled
        columns when every weight is 1.
        Returns (average, per-set centres [the plain mean of each set]).
        """
        ws = weights
        if not ws or len(ws) != len(cols):
            ws = [np.ones(a.shape[0]) for a in cols]
        num = sum((w[:, None] * a).sum(axis=1) for w, a in zip(ws, cols))
        den = sum(w * a.shape[1] for w, a in zip(ws, cols))
        return num / den, [np.mean(a, axis=1) for a in cols]

    @staticmethod
    def mean_rows(rows):
        """The plain mean of curves given as rows (n_curves, n_energies)."""
        rows = np.asarray(rows, dtype=np.float64)
        return np.mean(rows, axis=0)

    # --- the registry ---

    def open(self):
        """index -> window of the open averaging windows (closed ones are dropped)."""
        self.windows = {i: w for i, w in self.windows.items() if w.is_open()}
        return self.windows

    @staticmethod
    def saved_indices_of(directory):
        """The 'index' of every record *_average.json in the directory's 'averages' subfolder."""
        return records.saved_indices_of(directory)

    def saved_indices(self):
        return self.saved_indices_of(self.directory)

    def next_index(self):
        """1 + the highest index among the open windows and the saved records."""
        return 1 + max([0] + list(self.open()) + self.saved_indices())

    def by_name(self, name):
        """The open window whose name equals `name` (exact), or None."""
        for w in self.open().values():
            if w.name == name:
                return w
        return None

    def names(self):
        """The open windows' names, by index."""
        return [w.name for _, w in sorted(self.open().items())]

    def new(self, index=None, name=None, divide_iv=None, iv_broadening=None, channel=None,
            iv_zero_sub=None, iv_broadened=None):
        """
        Open an averaging window with the registry's defaults (an argument
        given replaces its default; index None takes the next free one),
        register it, and return it.
        """
        index = self.next_index() if index is None else int(index)
        w = MultiAverageWindow(index, self.directory, self.hampel_defaults, self.sgolay_defaults,
                               fit_settings=self.fit_settings,
                               divide_iv=self.divide_iv if divide_iv is None else divide_iv,
                               iv_broadening=self.iv_broadening if iv_broadening is None else iv_broadening,
                               channel=self.channel if channel is None else channel,
                               name=name,
                               iv_zero_sub=self.iv_zero_sub if iv_zero_sub is None else iv_zero_sub,
                               iv_broadened=self.iv_broadened if iv_broadened is None else iv_broadened)
        self.windows[index] = w
        return w

    def add(self, target, entry, channel=None):
        """
        Add an entry (a single-point Sm4File or a Selection) to an averaging
        window: `target` None or '' opens a new window (on `channel` when the
        entry holds it, else the entry's first channel) and adds the entry;
        a name adds to the open window of that name, ValueError when none is
        open.  A window that already holds the entry says so (has).  Returns
        the window.
        """
        target = (target or '').strip()
        if target:
            w = self.by_name(target)
            if w is None:
                raise ValueError(f"no open average named '{target}' (an empty name starts a new one; "
                                 f"open: {', '.join(self.names()) or 'none'}).")
        else:
            held = entry.channels()
            if not held:
                raise ValueError(f"{entry.name} has no dI/dV or IETS channel.")
            ch = channel if channel in held else (self.channel if self.channel in held else held[0])
            w = self.new(channel=ch)
        if w.has(entry):
            print(f"{w.name} already holds {entry.entry_label} ({entry.name}).")
            return w
        w.add(entry)
        print(f"{w.name}: added {entry.entry_label} ({entry.name}; {len(w.entries)} entries).")
        return w

    def holding(self, path):
        """Indices of the open windows holding an entry of that file path (a point file or a selection)."""
        return sorted(i for i, w in self.open().items() if any(e['file'].path == path for e in w.entries))


# ---------------------------------------------------------------------- #
#  The send of an average built from raw sets to the fit          #
# ---------------------------------------------------------------------- #

SETPOINT_TOL = 1e-6    # relative: V_bias and I_setpoint must agree to this for a send
CURVE_TOL = 1e-9       # relative: a fit history's stored curve must equal the average to this to be reused
ENERGY_TOL = 1e-6      # V: energy steps and values must agree to this


def compare_with_cache(state, V, data_unfiltered):
    """
    (same, text): whether a fit history's stored energy axis and unfiltered
    curve equal V and data_unfiltered (CURVE_TOL, relative to the largest
    stored value), and a description of the comparison.
    """
    Vc = np.asarray(state.get('V'), dtype=np.float64).ravel()
    dc = np.asarray(state.get('data_unfiltered'), dtype=np.float64).ravel()
    if len(Vc) != len(V) or len(dc) != len(data_unfiltered):
        return False, f"{len(dc)} stored points, {len(data_unfiltered)} in the average"
    dV = float(np.max(np.abs(Vc - V)))
    dd = float(np.max(np.abs(dc - data_unfiltered)))
    scale = max(float(np.max(np.abs(dc))), 1e-300)
    same = dV <= ENERGY_TOL and dd <= CURVE_TOL * scale
    return same, (f"largest difference {dd:.3g} ({dd / scale:.2e} of the largest stored value), "
                  f"energy axis difference {dV:.3g} V")


def send_sets_to_fit(label, V0, data, data_unfiltered, parts, hampel, sgolay, divide_iv, iv_width,
                        iv_zero_sub, resample, fit_settings, cache, tag, describe, removed_text=''):
    """
    Send an average built from several files' raw columns to the
    single-point fit (Fit); the one send of the averaging and the
    line-average windows.
      label            : the fit label ('<averages folder> | <name>'), whose stem names the fit history
      V0               : the average's energy axis (the run's grid)
      data             : the average as drawn (the window's filters; the plain mean)
      data_unfiltered  : the same average with every filter off
      parts            : [{'file': the entry (energy, lia, iv, units_lia, units_iv, vmod, acquired,
                           V_bias, I_setpoint, name), 'cols': the kept columns of its lia / iv}] in added order; the
                         first part is the anchor file
      hampel, sgolay   : the window's filter settings (None = off)
      divide_iv, iv_width, iv_zero_sub : the window's Feenstra state and settings
      resample         : 'bin', 'interp', or None (the refit brings a raw set of another step onto
                         the axis the way the run did)
      fit_settings  : the browser's settings for Fit.select_model (plus 'use_cache')
      cache            : the fit history path of this average (records.cache_path_of)
      tag              : the window's tag for the console
      describe         : a one-line description of the send for the console
    Per kept sweep, in pooled order: the normalization area (amps) as the
    drawn pipeline gives it (calibration and normalization on: the fit's
    pipeline) and I(V_bias) - I(V_ref) (amps, raw current); and one raw set
    per file (its own axis, the kept sweeps' raw LIA and I(V) columns) so
    that the fit reruns the pipeline under any filter setting.
    Setpoint: the files' own (V_bias, I_setpoint) when they all agree
    (SETPOINT_TOL).  When they differ the anchor is formed from the average
    itself: V_bias = the common-range point nearest the first file's bias,
    I_setpoint = (integral of the unsmoothed average from the point nearest
    0 V to that V_bias) / mean(1/areas), so that the fit's
    setpoint anchor (model integral = I_setpoint x mean(1/areas)) constrains
    the model to integrate to what the averaged data integrates to over
    that window; delta_I_window_A is then measured at that V_bias on every
    file's own I(V).  With 'use_cache' on, an existing fit history is
    reopened only when its stored energy axis and unfiltered curve equal
    the current average (CURVE_TOL); otherwise a fresh enumeration runs and
    its cache replaces the old one.  Returns the Fit, or None (reason printed).
    """
    first = parts[0]['file']
    for p in parts:
        f = p['file']
        if f.iv is None:
            print(f"send to fit refused: {f.name} has no 'Current' channel "
                  f"(the normalization areas in amps are needed).")
            return None
    tol = SETPOINT_TOL
    pA = 1e12
    differing = [p['file'] for p in parts[1:]
                 if (abs(p['file'].V_bias - first.V_bias) > tol * max(abs(first.V_bias), 1e-12)
                     or abs(p['file'].I_setpoint - first.I_setpoint) > tol * max(abs(first.I_setpoint), 1e-30))]
    V0 = np.asarray(V0, dtype=np.float64)
    i_ref0 = int(np.argmin(np.abs(V0)))
    i_bias0 = int(np.argmin(np.abs(V0 - first.V_bias)))
    V_bias, I_setpoint = first.V_bias, first.I_setpoint
    if differing:
        # Anchor from the average itself (see the docstring): V_bias = common-range
        # point nearest the first file's bias; I_setpoint set below from the areas.
        V_bias = float(V0[i_bias0])
        if i_bias0 == i_ref0:
            print(f"send to fit refused: the common range's point nearest {first.name}'s bias "
                  f"({first.V_bias:.4f} V) is its 0 V point, so no anchor window remains.")
            return None
    areas, delta_I, raw_sets = [], [], []
    for p in parts:
        f = p['file']
        cols = np.asarray(p['cols'], dtype=int)
        if cols.size == 0:
            continue
        _, ar = STS.process_arrays(f.energy, f.lia, f.iv, hampel, sgolay, divide_iv, iv_width,
                                   return_areas=True, acquired=f.acquired, iv_zero_sub=iv_zero_sub, vmod=f.vmod)
        order = np.argsort(f.energy)
        Vf = f.energy[order]
        iv_sorted = f.iv[order, :]
        i_ref = int(np.argmin(np.abs(Vf)))
        i_bias = int(np.argmin(np.abs(Vf - V_bias)))
        areas.extend(ar[cols])
        delta_I.extend(iv_sorted[i_bias, cols] - iv_sorted[i_ref, cols])
        raw_sets.append(STS.from_arrays(f.energy.copy(), f.lia[:, cols].copy(), f.iv[:, cols].copy(),
                                        f.units_lia, f.units_iv, f.vmod))
    areas = np.array(areas, dtype=np.float64)
    delta_I = np.array(delta_I, dtype=np.float64)
    if differing:
        X = cumulative_trapezoid(data_unfiltered, V0, initial=0)
        X = X - X[i_ref0]
        I_setpoint = float(X[i_bias0] / np.mean(1.0 / areas))
        print("send to fit: the files' setpoints differ ("
              + "; ".join(f"{f.name}: {f.V_bias:.4f} V, {f.I_setpoint * pA:.3f} pA"
                          for f in [first] + differing)
              + f"). Anchor from the average itself: V_bias = {V_bias:.4f} V (common-range point nearest "
              f"{first.name}'s {first.V_bias:.4f} V), I_setpoint = {I_setpoint * pA:.3f} pA = integral of the "
              f"unsmoothed average from {V0[i_ref0]:.4f} V to {V_bias:.4f} V divided by mean(1/areas).")
    settings = dict(fit_settings)
    use_cache = settings.pop('use_cache', True)
    print("\n" + "=" * 70)
    print(f"send to fit: {label} ({describe}{removed_text})")
    reuse = False
    if use_cache and fit.Fit.has_cache(label):
        state = records.read_cache(cache)
        same, text = ((False, "not readable") if state is None
                      else compare_with_cache(state, V0, data_unfiltered))
        if same:
            reuse = True
        else:
            print(f"{tag}: the fit history {cache} belongs to a different composition of this "
                  f"average ({text}); a fresh fit replaces it.")
    try:
        if reuse:
            an = fit.Fit.from_cache(cache)
        else:
            an = fit.Fit.from_curve(V0, data, data_unfiltered, V_bias, I_setpoint,
                                    areas, label,
                                    hampel_filter_params=hampel, sgolay_filter_params=sgolay,
                                    delta_I_window_A=delta_I,
                                    quantity='ratio' if divide_iv else 'didv',
                                    iv_broadening=iv_width, iv_zero_sub=iv_zero_sub,
                                    raw_sets=raw_sets,
                                    resample=resample)
            an.select_model(**settings)
        an.plot()
    except Exception as exc:
        print(f"send to fit: {exc}")
        return None
    return an


# ---------------------------------------------------------------------- #
#  The avgN window                                                         #
# ---------------------------------------------------------------------- #

class MultiAverageWindow:
    """
    One indexed average (avg<N>, or the name typed in its name box) of
    single-point files and marker-group selections added with the browser's
    'add new' / 'add existing' buttons and the Line, Map, and line-average
    windows' 'add' buttons.
    Left panel: every acquired sweep of every added file (the window's Hampel /
    SG controls on its dI/dV or IETS channel (the 'channel' choice; a file
    without the chosen channel cannot be added or shown), calibrated against
    the raw I(V) and unit-area normalized as the 'calibrate' / 'normalize'
    box says (both off at open), divided by I/V
    when the window's own 'Feenstra' toggle is on (on at open, independent of
    the browser's toggle; off reverts every sweep and the average to dI/dV;
    never applied to IETS), one colour per file; a change of the
    controls rebuilds every file's sweeps and repeats the run); overlaid, or
    offset vertically by the 'stack' toggle (STACK_OFFSET_FRAC of the largest
    curve's range per step, the first added file's first sweep at the top and,
    after a run, the average at the bottom; with 'substack' off (the default)
    the stack shows one curve per file instead, the mean of its kept sweeps,
    and a right-click removes nothing there, with it on every sweep;
    'substack' acts with 'stack' on; the two panels always fill the usable
    width together, un-stacked as one third / two thirds, stacked with the
    widths chosen so that one curve's range in the stack has the same
    height-to-width ratio as the average's range in the right panel; after a
    run only the energy range common to the files is shown).  Right panel:
    the average after a successful 'run', the plain mean of all kept sweeps
    of all added files (every sweep counts once) on the run's grid, the
    energy range common to the files.  With the 'bin' and 'interp' toggles
    off the run goes through only when every file has the same energy step
    and the same energy values on that range (ENERGY_TOL); otherwise the
    console names the file that differs.  With 'bin' on (one of the two at
    most) files of a finer step are binned onto the grid of the coarsest
    file, with 'interp' on files of a coarser step are interpolated onto the
    grid of the finest file; either way every point of such a file counts in
    the average with grid step / file step, the file's measured points per
    grid step (2.5 for 10 meV binned onto 25 meV, 0.4 for 25 meV
    interpolated onto 10 meV; see _on_run, _average_of); the left panel
    then shows the sweeps as the
    grid has them, and a send carries the setting so that the fit
    rebuilds the curve the same way.
    Adding a file clears the average until the next run.  A right-click on a
    sweep in the left panel (the nearest sweep within PICK_PX pixels, ignored
    while the toolbar's zoom or pan mode is active) removes that sweep from
    the panel and from the average, which is recomputed at once when it
    exists; a removed sweep counts as never loaded, so a file whose sweeps
    are all removed is absent from the panel, the run's checks (energy step,
    energy values, common range) and the average, and 'run' goes through
    without it; 'restore' puts back the most recently removed sweep (when
    its file was empty at the last run, that run is cleared: 'run' again).
    'clear'
    empties the window (the index stays).  The name box at the top left
    (default avg<N>) names the average: 'save' writes the record
    <stem>_average.json in the subfolder 'averages' of the
    experiment directory (see record; the stem is the name cleaned as the fit
    history cleans its label, records.stem_of); 'Send to fit' writes it
    too and sends the average to the single-point fit
    (send_sets_to_fit; the sent curves are always calibrated and
    normalized, the fit's pipeline, with a console line when the window's
    boxes are off), whose fit history is <stem>_fits.pkl in the same
    subfolder.  A new name typed into the box renames the record and the fit
    history already written for this average (see _on_rename).
    restore(...) rebuilds the window from a record and
    open_fit(...) compares the rebuilt average with a fit history and opens
    the fit (the saved averages window).  Its Feenstra companion
    (feenstra.FeenstraWindow) opens with it.
    An entry is a single-point Sm4File or a Selection (chosen columns of a
    grid file); each of its acquired columns is one sweep.
    """

    RECORD_FORMAT = records.RECORD_FORMAT
    AVG_SUBDIR = records.AVG_SUBDIR

    ENERGY_TOL = ENERGY_TOL
    STACK_OFFSET_FRAC = 0.25
    PICK_PX = 6            # right-click must land within this many pixels of a sweep
    # Usable width 0.06 to 0.98 with a 0.07 gap, the two panels always filling it
    # together: un-stacked, left third (overlay) and right two thirds (average);
    # stacked, the widths follow from the aspect rule (see draw).  The panels' top
    # is 0.90, leaving room for the name box above the left panel.
    PANEL_Y, PANEL_H = 0.24, 0.66
    NAME_XY = (0.09, 0.945)                        # the name box (its label 'name ' sits to its left)
    LEFT_X, GAP, RIGHT_EDGE = 0.06, 0.07, 0.98
    PANELS_WIDTH = RIGHT_EDGE - LEFT_X - GAP       # 0.85, shared by the two panels
    LEFT_WIDTH = PANELS_WIDTH / 3                  # un-stacked left panel (one third)
    RIGHT_WIDTH = PANELS_WIDTH - LEFT_WIDTH        # un-stacked average panel (two thirds)
    RIGHT_X = LEFT_X + LEFT_WIDTH + GAP
    # The Feenstra companion has sweep panels and no heatmaps.
    feenstra_heatmaps = False

    DESIGN = (15, 7)       # the design size (inches): the text zoom of the window is its size relative to this (qtplot.window_zoom)

    def __init__(self, index, directory, hampel_defaults, sgolay_defaults, fit_settings=None,
                 divide_iv=True, iv_broadening=0.0, channel='didv', name=None,
                 iv_zero_sub=True, iv_broadened=False):
        """
        index     : the average's index N (kept in the record; new windows count on from it)
        directory : the experiment directory; the send label is
                    '<directory>/averages | <name>', so the fit history is
                    <directory>/averages/<stem>_fits.pkl
        divide_iv : starting state of the window's own 'Feenstra' toggle (default on:
                    the sweeps are the normalized conductance); independent of the
                    browser's toggle
        iv_broadening : the broadening width of I/V in volts (divide_by_iv)
        iv_zero_sub, iv_broadened : the Feenstra settings (the companion's '0Vsub' and
                    'broadened' boxes): the current's value at 0 V subtracted before
                    I/V is formed; I/V broadened by iv_broadening (off, the default:
                    the plain I/V)
        channel   : the channel the window averages, 'didv' or 'iets' (the 'channel'
                    radio of the window changes it for every added file)
        name      : the average's name (window title, record and fit history names);
                    None gives avg<N>; editable in the window's name box
        """
        self.index = int(index)
        self.divide_iv = bool(divide_iv)
        self.iv_broadening = float(iv_broadening)
        self.iv_zero_sub = bool(iv_zero_sub)
        self.iv_broadened = bool(iv_broadened)
        self.calibrate = False                      # the 'calibrate' / 'normalize' box: both off at open, as in
        self.normalize = False                      # every window that uses Feenstra
        self.channel = channel if channel in STS.CHANNELS else 'didv'
        self.directory = os.path.abspath(directory)
        self.avg_dir = records.avg_dir_of(self.directory)
        self.name = ''
        self._set_name((name or '').strip() or f"avg{self.index}")
        self._saved_stem = None    # the stem of the record this window has written (None until a save)
        self.entries = []          # dicts: file, V (ascending), sweeps (npts, n), label
        self.average = None        # (V, lo, hi, maps) after a successful run; maps bring each entry onto V (see _to_grid)
        self.resample = None       # 'bin' / 'interp' toggles: how files of another energy step reach the grid; None = off
        self.removed = []          # (entry index, sweep index) in removal order
        self._lines = []           # (entry index, sweep index, Line2D) of the sweeps drawn in the left panel
        self.fit_settings = dict(fit_settings) if fit_settings else {}
        self._fits = []   # sent and loaded fits stay referenced so their widgets stay live
        self.fig = plt.figure(figsize=self.DESIGN)
        try:
            self.fig.canvas.manager.set_window_title(self.tag)
        except Exception:
            pass
        self.ax = self.fig.add_axes([self.LEFT_X, self.PANEL_Y, self.LEFT_WIDTH, self.PANEL_H])
        self.ax_avg = self.fig.add_axes([self.RIGHT_X, self.PANEL_Y, self.RIGHT_WIDTH, self.PANEL_H])
        # The controls (widgets.py): the name box at the top left, the filter box, the channel
        # radio, stack / substack, bin / interp, the window's own 'Feenstra' toggle, the
        # 'calibrate' / 'normalize' box (top right, as in every window), and the button row.
        self.box_name = widgets.NameBox(self.fig, *self.NAME_XY, name=self.name)
        self.box_name.on_change(self._on_rename)
        self.filters = widgets.FilterControls(self.fig, [0.06, 0.03, 0.22, 0.15],
                                              hampel=hampel_defaults, sgolay=sgolay_defaults,
                                              hampel_on=False, sgolay_on=False)
        self.filters.on_change(self._on_filter_change)
        self.radio_channel = widgets.make(self.fig, 'channel', 0.50, 0.12, value=STS.CHANNEL_NAMES[self.channel])
        self.radio_channel.on_change(self._on_channel)
        self.stacks = widgets.StackControls(self.fig, (0.285, 0.06), (0.348, 0.06), needs_average=False, tag=self.tag)
        self.stacks.on_change(self.draw)
        self.resamples = widgets.ResampleControls(self.fig, (0.285, 0.12), (0.343, 0.12))
        self.resamples.on_change(self._on_resample_toggle)
        self.chk_feenstra = widgets.make(self.fig, 'Feenstra', 0.41, 0.12, value=self.divide_iv)
        self.chk_feenstra.on_change(self._on_feenstra_toggle)
        self.pipeline = widgets.PipelineBox(self.fig, calibrate=self.calibrate, normalize=self.normalize)
        self.pipeline.on_change(self._on_pipeline_change)
        self.buttons = widgets.RunControls(self.fig, [('run', 0.425, 0.05), ('clear', 0.52, 0.05),
                                                      ('restore', 0.615, 0.05), ('Send to fit', 0.71, 0.05),
                                                      ('save', 0.84, 0.05)])
        self.buttons.on('run', self._on_run)
        self.buttons.on('clear', self._on_clear)
        self.buttons.on('restore', self._on_restore)
        self.buttons.on('Send to fit', self._on_send_to_fit)
        self.buttons.on('save', self._on_save)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.feenstra = None               # the Feenstra companion (FeenstraWindow), opened below
        self.draw()
        self.feenstra = FeenstraWindow(self, self.name)
        plt.show()

    def is_open(self):
        return widgets.figure_is_open(self.fig)

    @property
    def stacked(self):
        return self.stacks.stack

    @property
    def substack(self):
        return self.stacks.substack

    # --- the Feenstra companion ---

    def _feenstra_refresh(self):
        w = getattr(self, 'feenstra', None)
        if w is not None and w.is_open():
            w.refresh()

    def feenstra_settings(self):
        """The window's Feenstra settings (the companion's boxes): {'zero_sub', 'broadened', 'broadening'}."""
        return {'zero_sub': self.iv_zero_sub, 'broadened': self.iv_broadened, 'broadening': self.iv_broadening}

    def iv_width(self):
        """The broadening width the division uses (feenstra_width)."""
        return feenstra_width(self.iv_broadening, self.iv_broadened)

    def set_feenstra_settings(self, zero_sub=None, broadened=None, broadening=None):
        """
        A Feenstra setting changed (the companion's boxes; None = unchanged):
        every file's sweeps are rebuilt with it and the run repeated (as
        set_divide_iv does), and every fit sent from this window
        receives it (Fit.set_feenstra), so a fit window with its own
        'Feenstra (refit)' toggle on shows and refits the same quantity.
        Returns False (nothing changed) when a file cannot be processed with it.
        """
        new = (self.iv_zero_sub if zero_sub is None else bool(zero_sub),
               self.iv_broadened if broadened is None else bool(broadened),
               self.iv_broadening if broadening is None else float(broadening))
        old = (self.iv_zero_sub, self.iv_broadened, self.iv_broadening)
        if new == old:
            return True
        self.iv_zero_sub, self.iv_broadened, self.iv_broadening = new
        note = ("0 V subtraction " + ('on' if new[0] else 'off') if new[0] != old[0]
                else "broadening " + ('on' if new[1] else 'off') if new[1] != old[1]
                else f"broadening {new[2]:g} V")
        if not self._rebuild_sweeps(note):
            self.iv_zero_sub, self.iv_broadened, self.iv_broadening = old
            return False
        for an in list(self._fits):
            try:
                an.set_feenstra(self.iv_width(), self.iv_zero_sub)
            except Exception as exc:
                print(f"{self.tag}: a fit did not take the Feenstra settings: {exc}")
        return True

    def feenstra_view(self, calibrate=True, normalize=True):
        """
        The companion's view (FeenstraWindow): per active entry its kept
        sweeps in the four versions, dI/dV (the pipeline with the division
        off), the current in amps as the division works on it (its value at
        0 V subtracted with '0Vsub'), its I/V in A/V as the division divides
        by it (broadened with 'broadened'), and the ratio (the division on),
        each on the run's grid when a run exists (the entry's map, as the
        window's sweeps are) and with the run's average (the pooled mean of
        _average_of); the window's filters and channel; `calibrate` and
        `normalize` are the companion's own flags (its dI/dV and ratio
        through the pipeline with them, the current and I/V unit-area
        normalized with normalize_abs when `normalize` is on).  An entry
        without the Current channel gives notes for the current, I/V, and
        ratio panels; the IETS channel gives a note for the ratio panel.
        """
        from sm4_Browser.normalization import normalize_abs
        hp, sp = self.filters.params()
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        active = self._active()
        run = self.average is not None
        tol = self.ENERGY_TOL
        window = (self.average[1], self.average[2]) if run else None
        q = ("d2I/dV2 (IETS)" if self.channel == 'iets' else "dI/dV") + ("" if calibrate else ", uncalibrated")
        zs, width = self.iv_zero_sub, self.iv_width()
        t = feenstra_titles(zs, self.iv_broadened, self.iv_broadening,
                            all(energy_has_zero(self.entries[i]['file'].energy) for i in active))
        nn = "" if normalize else ", not normalized"
        versions = {
            'didv': {'title': f"{q} (division off){nn}",
                     'ylabel': f"{q} ({curve_units_of([self.entries[i]['file'] for i in active], self.channel, calibrate, normalize)})",
                     'entries': [], 'sets': []},
            'iv': {'title': t['iv'] + (" (unit area)" if normalize else ""), 'ylabel': "I (unit area)" if normalize else "I (A)",
                   'entries': [], 'sets': []},
            'ivv': {'title': t['ivv'] + (" (unit area)" if normalize else ""), 'ylabel': "I/V (unit area)" if normalize else "I/V (A/V)",
                    'entries': [], 'sets': []},
            'ratio': {'title': t['ratio'] + " (division on)" + nn + ("" if calibrate else ", uncalibrated"),
                      'ylabel': "(dI/dV)/(I/V) (arb. units)", 'entries': [], 'sets': []},
        }
        if self.channel == 'iets':
            versions['ratio']['note'] = "the division applies to dI/dV only"
        for i in active:
            e = self.entries[i]
            f = e['file']
            kept = self._kept(i)
            order = np.argsort(f.energy)
            mp = self.average[3][i] if run else None
            color = colors[i % len(colors)]
            label = e['label']
            per = {}
            per['didv'] = f.curves(hp, sp, False, width, channel=self.channel, iv_zero_sub=zs,
                                   normalize=normalize, calibrate=calibrate)[order, :][:, f.acquired][:, kept]
            if f.iv is not None:
                iv_a = iv_corrected(f.energy, f.iv, zs, hp, sp)
                ivv = iv_broadened(f.energy, f.iv, width, zs, hp, sp)
                if normalize:
                    iv_a = normalize_abs(f.energy, iv_a)[0]
                    ivv = normalize_abs(f.energy, ivv)[0]
                per['iv'] = iv_a[order, :][:, f.acquired][:, kept]
                per['ivv'] = ivv[order, :][:, f.acquired][:, kept]
                if self.channel != 'iets':
                    per['ratio'] = f.curves(hp, sp, True, width, channel=self.channel, iv_zero_sub=zs,
                                            normalize=normalize, calibrate=calibrate)[order, :][:, f.acquired][:, kept]
            else:
                for key in ('iv', 'ivv', 'ratio'):
                    versions[key].setdefault('note', f"{f.name} has no Current channel")
            for key, sw in per.items():
                if mp is not None:
                    V, sw_g = self.average[0], self._to_grid(sw, mp)
                    versions[key]['sets'].append((i, sw_g))
                elif window:
                    m = (e['V'] >= window[0] - tol) & (e['V'] <= window[1] + tol)
                    V, sw_g = e['V'][m], sw[m, :]
                else:
                    V, sw_g = e['V'], sw
                versions[key]['entries'].append({'label': label, 'color': color, 'V': V, 'sweeps': sw_g})
        for key, v in versions.items():
            v['average'] = None
            if run and v['sets'] and 'note' not in v:
                avg, _ = self._average_of(v['sets'])
                v['average'] = (self.average[0], avg)
            v.pop('sets', None)
        return {'kind': 'sweeps', 'title': f"{self.name}: {len(active)} entries", 'energy': None, 'versions': versions}

    # --- name, record (averages/<stem>_average.json) and fit history (averages/<stem>_fits.pkl) ---

    def _set_name(self, name):
        """Set the name and what follows from it: tag, stem, send label, window title, name box."""
        self.name = str(name)
        self.tag = self.name
        self.stem = records.stem_of(self.name)
        self.label = f"{self.avg_dir} | {self.name}"
        fig = getattr(self, 'fig', None)
        if fig is not None:
            try:
                fig.canvas.manager.set_window_title(self.name)
            except Exception:
                pass
        box = getattr(self, 'box_name', None)
        if box is not None:
            box.set(self.name)

    def _on_rename(self, text):
        """
        The name box: a new name for this average.  Its stem must be non-empty
        and must not be the stem of another record in the folder (else a
        console line and the box reverts).  When this window has written a
        record, that record is renamed to the new stem with its 'name' and
        'label' rewritten, and the fit history of the old stem, when it
        exists, is renamed too with its stored label rewritten (the pickle
        format kept).  With no record on disk nothing is renamed; the next
        save uses the new name.
        """
        new = str(text).strip()
        if new == self.name:
            return
        stem = records.stem_of(new)
        if not stem:
            print(f"{self.tag}: '{new}' leaves no file name (letters, digits, '.', '_', '-'); name kept.")
            self._set_name(self.name)
            return
        new_rec = records.record_path(self.directory, stem)
        if os.path.isfile(new_rec) and stem != self._saved_stem:
            print(f"{self.tag}: a saved average already uses the file name {os.path.basename(new_rec)}; name kept.")
            self._set_name(self.name)
            return
        old_name, old_stem = self.name, self._saved_stem
        self._set_name(new)
        if old_stem is None or old_stem == stem:
            print(f"{old_name} renamed {self.name}" + ("" if old_stem is None else " (same file names)") + ".")
            return
        renamed = records.rename_record(self.directory, old_stem, stem, self.name, self.label, self.tag)
        self._saved_stem = stem
        print(f"{old_name} renamed {self.name}" + (f"; renamed: {', '.join(renamed)}." if renamed else "."))

    def cache_path(self):
        return records.cache_path_of(self.directory, self.stem)

    def record(self):
        """
        Plain dict describing the average: the index, the name and the send
        label, the entries in added order (kind 'point' or 'selection',
        absolute path, base name, number of acquired sweeps; a selection also
        its columns and where they are), the removed sweeps in removal order (indices among the
        acquired sweeps), the channel ('didv' or 'iets'), the filter settings
        (Hampel, SG; None = off), the 'bin' / 'interp' setting ('resample':
        'bin', 'interp', or None), the Feenstra state and settings, the
        calibrate / normalize flags, the range, point count, and grid step of the
        last run, and the time of writing.
        """
        hp, sp = self.filters.params()
        run = None
        if self.average is not None:
            V0 = self.average[0]
            run = {'lo': float(self.average[1]), 'hi': float(self.average[2]),
                   'n_points': int(len(V0)),
                   'step_V': float(np.median(np.diff(V0))) if len(V0) > 1 else None}
        files = []
        for e in self.entries:
            f = e['file']
            entry = {'kind': 'point', 'path': os.path.abspath(f.path), 'name': f.name,
                     'n_sweeps': int(e['sweeps'].shape[1])}
            if isinstance(f, Selection):
                entry.update(f.record_entry())
            files.append(entry)
        return {'format': self.RECORD_FORMAT,
                'index': self.index,
                'name': self.name,
                'label': self.label,
                'directory': self.directory,
                'files': files,
                'removed': [[int(i), int(j)] for i, j in self.removed],
                'channel': self.channel,
                'hampel': hp,
                'sgolay': sp,
                'resample': self.resample,
                'divide_iv': self.divide_iv,
                'iv_broadening_V': self.iv_broadening,
                'iv_broadened': self.iv_broadened,
                'iv_zero_sub': self.iv_zero_sub,
                'calibrate': self.calibrate,
                'normalize': self.normalize,
                'run': run,
                'saved': datetime.now().isoformat(timespec='seconds')}

    def save(self):
        """Write the record (to a temporary file, then replaced in place).  Returns its path."""
        path = records.write_json(records.record_path(self.directory, self.stem), self.record())
        self._saved_stem = self.stem
        return path

    def _on_save(self):
        if len(self.entries) == 0:
            print(f"{self.tag} save: no files added.")
            return
        print(f"{self.tag}: record written: {self.save()}")

    def restore(self, files, removed, hampel, sgolay, resample=None, channel='didv', calibrate=True, normalize=True):
        """
        Rebuild the window from a record: the channel, the calibrate /
        normalize flags, the files added in order, the sweeps removed in the
        recorded order (so 'restore' walks them back), the filter controls
        and the 'bin' / 'interp' boxes set (a record written before the
        resampling has no setting: both boxes off), then 'run'.  The caller
        has resolved the files and checked the removed indices against the
        acquired sweeps.
        """
        self._set_channel(channel)
        self.calibrate, self.normalize = bool(calibrate), bool(normalize)
        self.pipeline.set_state(self.calibrate, self.normalize)
        self.filters.set_params(hampel, sgolay)   # the sweeps are built with these settings
        self._set_resample(resample)
        for f in files:
            self.add(f)
        self.removed = [(int(i), int(j)) for i, j in removed]
        print(f"{self.tag}: rebuilt from its record: {len(self.entries)} files, "
              f"{len(self.removed)} removed sweep(s), {STS.CHANNEL_NAMES[self.channel]}, "
              f"Hampel {hampel}, SG {sgolay}, {self.resample or 'bin / interp off'}"
              + ("" if self.calibrate else ", calibration off") + ("" if self.normalize else ", normalization off") + ".")
        self._on_run()
        self.draw()

    def open_fit(self, cache):
        """
        Compare the rebuilt average (after a run) with the fit history at
        `cache` on the console, then open its fit figure (from_cache).
        """
        state = records.read_cache(cache)
        if state is None:
            print(f"{self.tag}: {cache} is not a readable fit history; no fit opened.")
            return
        sweeps = self._avg_sweeps()
        if sweeps is None or sweeps.shape[1] == 0:
            print(f"{self.tag}: no average to compare with the fit history (the run did not go through).")
        elif (state.get('quantity', 'didv') == 'ratio') != self.divide_iv:
            print(f"{self.tag}: the fit history holds "
                  + ("(dI/dV)/(I/V)" if state.get('quantity', 'didv') == 'ratio' else "dI/dV")
                  + f" and the window shows {'(dI/dV)/(I/V)' if self.divide_iv else 'dI/dV'}; no comparison is made.")
        else:
            same, text = compare_with_cache(state, self.average[0], self._unfiltered_average())
            print(f"{self.tag}: self check against {cache}: {text}; "
                  + ("the fit history belongs to this average." if same
                     else "the fit history belongs to a different composition of this average."))
        try:
            an = fit.Fit.from_cache(cache)
            an.plot()
        except Exception as exc:
            print(f"{self.tag}: fit history not opened: {exc}")
            return
        self._fits.append(an)

    # --- entries and sweeps ---

    def has(self, point):
        """Whether an entry with the same key is held (a file path; a selection's path and columns)."""
        return any(e['file'].key == point.key for e in self.entries)

    def _sweeps_of(self, point, hampel=None, sgolay=None, use_controls=True, calibrate=None, normalize=None):
        """
        A file's acquired sweeps through the pipeline (the window's Hampel / SG
        settings on the channel, or the given ones when use_controls is False;
        the window's calibrate / normalize flags unless given), rows in
        ascending energy order, never-acquired sweeps left out.
        """
        if use_controls:
            hampel, sgolay = self.filters.params()
        order = np.argsort(point.energy)
        c = point.curves(hampel, sgolay, self.divide_iv, self.iv_width(), channel=self.channel,
                         iv_zero_sub=self.iv_zero_sub,
                         normalize=self.normalize if normalize is None else normalize,
                         calibrate=self.calibrate if calibrate is None else calibrate)
        return c[order, :][:, point.acquired]

    def _rebuild_sweeps(self, note):
        """
        Every file's sweeps rebuilt with the current settings (removed sweeps
        stay removed) and, when a run existed, the run repeated at once (its
        checks concern the energy axes, which do not change).  Returns False
        (nothing changed) when a file cannot be processed.
        """
        try:
            new = [self._sweeps_of(e['file']) for e in self.entries]
        except Exception as exc:
            print(f"{self.tag}: {note} not applied: {exc}")
            return False
        for e, sw in zip(self.entries, new):
            e['sweeps'] = sw
        print(f"{self.tag}: {note}; sweeps rebuilt.")
        if self.average is not None:
            self._on_run()
        else:
            self.draw()
        return True

    def _on_filter_change(self):
        hp, sp = self.filters.params()
        self._rebuild_sweeps(f"Hampel {'on' if hp else 'off'}, "
                             f"SG {'on' if sp else 'off'} on the {STS.CHANNEL_NAMES[self.channel]}")

    def _on_pipeline_change(self, state):
        """The 'calibrate' / 'normalize' box: every file's sweeps rebuilt with the flags and the run repeated."""
        old = (self.calibrate, self.normalize)
        self.calibrate, self.normalize = state['calibrate'], state['normalize']
        if not self._rebuild_sweeps(f"calibration {'on' if self.calibrate else 'off'}, "
                                    f"normalization {'on' if self.normalize else 'off'}"):
            self.calibrate, self.normalize = old
            self.pipeline.set_state(*old)

    def _set_channel(self, channel):
        """Set self.channel ('didv' or 'iets') and the radio to it, without rebuilding."""
        self.channel = channel if channel in STS.CHANNELS else 'didv'
        radio = getattr(self, 'radio_channel', None)
        if radio is not None:
            radio.set(STS.CHANNEL_NAMES[self.channel])

    def _on_channel(self, label):
        """
        The 'channel' radio: every file's sweeps are rebuilt from the chosen
        channel (removed sweeps stay removed) and, when a run existed, the run
        is repeated; a file without the channel puts the radio back.
        """
        channel = {v: k for k, v in STS.CHANNEL_NAMES.items()}.get(label, 'didv')
        if channel == self.channel:
            return
        old = self.channel
        self.channel = channel
        if not self._rebuild_sweeps(f"channel {STS.CHANNEL_NAMES[channel]}"):
            self.channel = old
        self._set_channel(self.channel)

    def add(self, point):
        if self.has(point):
            return
        order = np.argsort(point.energy)
        sweeps = self._sweeps_of(point)
        self.entries.append({'file': point, 'V': point.energy[order].astype(np.float64),
                             'sweeps': sweeps, 'cols': np.flatnonzero(point.acquired),
                             'label': point.entry_label})
        self.average = None                    # a new file undoes a previous run
        self.draw()

    def set_divide_iv(self, flag, broadening=0.0):
        """
        The window's 'Feenstra' state (or its broadening) changed: every file's
        sweeps are rebuilt (removed sweeps stay removed) and, when a run
        existed, the run is repeated at once (its checks concern the energy
        axes, which do not change).  The toggle box follows the state that
        results (a file that cannot be divided puts it back).
        """
        flag, broadening = bool(flag), float(broadening)
        if flag == self.divide_iv and broadening == self.iv_broadening:
            self.chk_feenstra.set(self.divide_iv)
            return
        old = (self.divide_iv, self.iv_broadening)
        self.divide_iv, self.iv_broadening = flag, broadening
        if not self._rebuild_sweeps(f"Feenstra {'on' if flag else 'off'}"):
            self.divide_iv, self.iv_broadening = old
        self.chk_feenstra.set(self.divide_iv)

    def _on_feenstra_toggle(self, flag):
        """The window's own 'Feenstra' toggle: the sweeps and the average follow it (set_divide_iv)."""
        self.set_divide_iv(bool(flag), self.iv_broadening)

    def _on_resample_toggle(self, method):
        """
        'bin' / 'interp': one of the two at most (the controller clears the
        other box).  After a run the run is repeated with the new setting (a
        repeated run that is refused leaves no average: 'run' again).
        """
        self.resample = method
        print(f"{self.tag}: files of another energy step are "
              + {'bin': "binned onto the coarsest grid", 'interp': "interpolated onto the finest grid",
                 None: "not accepted (bin / interp off)"}[self.resample] + ".")
        if self.average is not None:
            self.average = None
            self._on_run()
            if self.average is None:
                self.draw()

    def _set_resample(self, method):
        """Set self.resample ('bin', 'interp', or None) and the two boxes to it, without redrawing."""
        self.resample = method if method in ('bin', 'interp') else None
        self.resamples.set(self.resample)

    @staticmethod
    def _to_grid(columns, mp):
        """Columns of one entry brought onto the run's grid by its map (to_grid)."""
        return to_grid(columns, mp)

    def _kept(self, i):
        """Indices of the sweeps of entry i that have not been removed."""
        gone = {j for ii, j in self.removed if ii == i}
        return [j for j in range(self.entries[i]['sweeps'].shape[1]) if j not in gone]

    def _active(self):
        """
        Indices of the files that still have a kept sweep.  A removed sweep
        counts as never loaded: a file whose sweeps are all removed takes no
        part in the run's checks, the average, the panel, or a send.
        """
        return [i for i in range(len(self.entries)) if self._kept(i)]

    def _avg_sweeps(self):
        """
        Remaining sweeps of the files that took part in the last run, on its
        grid, (npts, n_kept); None without a run.  A file with no map (empty
        at the run) contributes nothing.
        """
        if self.average is None:
            return None
        sets = self._avg_sets()
        if not sets:
            return np.zeros((len(self.average[0]), 0))
        return np.hstack([s for _, s in sets])

    def _avg_sets(self, hampel=None, sgolay=None, use_controls=True, calibrate=None, normalize=None):
        """
        The kept sweeps of the files that took part in the last run, one array
        (npts, n_kept) per file on the run's grid (brought there by the file's
        map, _to_grid), as [(entry index, array)] in added order; [] without a
        run.  With use_controls the window's sweeps as built (the controls'
        settings); otherwise the files are run through the pipeline again
        with the given settings (None = off) and flags.
        """
        if self.average is None:
            return []
        maps = self.average[3]
        sets = []
        for i, (e, mp) in enumerate(zip(self.entries, maps)):
            kept = self._kept(i)
            if mp is None or not kept:
                continue
            sw = e['sweeps'] if use_controls else self._sweeps_of(e['file'], hampel, sgolay, use_controls=False,
                                                                   calibrate=calibrate, normalize=normalize)
            sets.append((i, self._to_grid(sw[:, kept], mp)))
        return sets

    def _point_weights(self, i):
        """Weight of every grid point of entry i's sweeps in the average (point_weights of its map)."""
        return point_weights(self.average[3][i], len(self.average[0]))

    def _average_of(self, sets):
        """
        The average of per-file sweep arrays [(entry index, (npts, n_kept))]:
        every point of every sweep weighted by its file's measured points per
        grid step (_point_weights: grid step / file step for a binned or
        interpolated file, 1 for a file on the grid), so that a file counts
        by how much it measured; with every file on the grid it is the plain
        mean of every sweep (Averages.average).  Returns (average, {entry index:
        the file's plain mean}); None for [].
        """
        if not sets:
            return None
        avg, centres = Averages.average([a for _, a in sets], [self._point_weights(i) for i, _ in sets])
        return avg, {i: c for (i, _), c in zip(sets, centres)}

    def _file_average(self, sweeps):
        """One file's plain mean of its sweep columns."""
        return np.mean(sweeps, axis=1)

    def _unfiltered_average(self, calibrate=True, normalize=True):
        """The average of the last run with every filter off (calibration and normalization on unless given)."""
        return self._average_of(self._avg_sets(None, None, use_controls=False, calibrate=calibrate,
                                               normalize=normalize))[0]

    @staticmethod
    def _px_distance(x, y, xy):
        """Pixel distance from (x, y) to the polyline with display-space vertices xy (n, 2)."""
        xy = xy[np.all(np.isfinite(xy), axis=1)]
        if len(xy) == 0:
            return np.inf
        if len(xy) == 1:
            return float(np.hypot(xy[0, 0] - x, xy[0, 1] - y))
        p, d = xy[:-1], np.diff(xy, axis=0)
        L2 = np.sum(d ** 2, axis=1)
        t = ((x - p[:, 0]) * d[:, 0] + (y - p[:, 1]) * d[:, 1]) / np.where(L2 > 0, L2, 1.0)
        t = np.clip(t, 0.0, 1.0)
        return float(np.sqrt(np.min((p[:, 0] + t * d[:, 0] - x) ** 2 + (p[:, 1] + t * d[:, 1] - y) ** 2)))

    def _on_click(self, event):
        # Right-click in the left panel removes the nearest sweep (within PICK_PX pixels).
        if event.button != 3 or event.inaxes is not self.ax:
            return
        tb = getattr(self.fig.canvas, 'toolbar', None)
        if tb is not None and getattr(tb, 'mode', ''):
            return
        if getattr(self, '_per_file_view', False):
            print(f"{self.tag}: the stacked panel shows one average per file; sweeps are removed with "
                  f"'stack' off or 'substack' on.")
            return
        best, best_d = None, np.inf
        for i, j, line in self._lines:
            xy = self.ax.transData.transform(np.column_stack([line.get_xdata(), line.get_ydata()]))
            d = self._px_distance(event.x, event.y, xy)
            if d < best_d:
                best, best_d = (i, j), d
        if best is None or best_d > self.PICK_PX:
            return
        self.removed.append(best)
        e = self.entries[best[0]]
        print(f"{self.tag}: removed sweep {best[1] + 1} of {e['label']} ({e['file'].name}); "
              f"{len(self.removed)} removed in total.")
        self.draw()

    def _on_restore(self):
        if not self.removed:
            print(f"{self.tag}: nothing to restore.")
            return
        i, j = self.removed.pop()
        e = self.entries[i]
        print(f"{self.tag}: restored sweep {j + 1} of {e['label']} ({e['file'].name}); "
              f"{len(self.removed)} still removed.")
        # A file that was empty at the last run never had its energy axis checked
        # against the others: that run is cleared, 'run' again.
        if self.average is not None and self.average[3][i] is None:
            self.average = None
            print(f"{self.tag}: {e['file'].name} took no part in the last run; run again.")
        self.draw()

    def _on_run(self):
        """
        Average the kept sweeps of the files that still have one (see _active)
        on one grid: the energy steps, the energy values on the overlap and
        the common range are checked over those files only; a file with no
        kept sweep gets no map and is skipped.
        With 'bin' and 'interp' off every file must have the same energy step
        and the same energy values on the common range (ENERGY_TOL).  With one
        of them on, the grid is the points, on the common range, of the file
        with the coarsest step ('bin') or the finest ('interp'); a file at the
        grid's step must coincide with the grid points (as before), any other
        file is brought onto the grid: 'bin' averages its points inside half
        a grid step either side of each grid point (refused if a bin would be
        empty), 'interp' interpolates linearly at the grid points (the grid
        lies inside every file's range, so nothing is extrapolated); either
        map keeps the weight grid step / file step with which the file's
        points count in the average (_average_of).
        """
        if len(self.entries) == 0:
            print(f"{self.tag}: no files added.")
            return
        active = self._active()
        if not active:
            print(f"{self.tag} refused: every sweep has been removed ('restore' puts them back).")
            return
        tol = self.ENERGY_TOL
        ents = [self.entries[i] for i in active]
        steps = grid_steps([e['V'] for e in ents])
        method = self.resample
        try:
            if method is None:
                same_step(steps, [e['file'].name for e in ents], tol)
                ref = 0
            else:
                ref = int(np.argmax(steps)) if method == 'bin' else int(np.argmin(steps))
            step = steps[ref]
            lo, hi = common_range([e['V'] for e in ents], step)
        except ValueError as exc:
            print(f"{self.tag} refused: {exc}")
            return
        in_range = lambda V: (V >= lo - tol) & (V <= hi + tol)
        V0 = ents[ref]['V'][in_range(ents[ref]['V'])]
        maps = [None] * len(self.entries)
        notes = []
        for i, e, st in zip(active, ents, steps):
            try:
                maps[i], note = grid_map(e['V'], V0, lo, hi, step, st, method, tol,
                                         e['file'].name, ents[ref]['file'].name, e['label'])
            except ValueError as exc:
                print(f"{self.tag} refused: {exc}")
                return
            notes.append(note)
        n_kept = sum(len(self._kept(i)) for i in active)
        self.average = (V0, lo, hi, maps)
        n_empty = len(self.entries) - len(active)
        print(f"{self.tag}: {n_kept} sweeps from {len(active)} files on {lo:.3f} to {hi:.3f} V "
              f"({len(V0)} points, step {step * 1e3:.3f} meV"
              + (f", {method}: " + "; ".join(notes) if method else "") + ")"
              + (f", {len(self.removed)} removed sweep(s) excluded" if self.removed else "")
              + (f", {n_empty} file(s) with every sweep removed left out." if n_empty else "."))
        self.draw()

    def _on_clear(self):
        self.entries = []
        self.average = None
        self.removed = []
        self.draw()

    def _on_send_to_fit(self):
        """
        Send the average (the right panel's curve) to the single-point
        fit (send_sets_to_fit).  Needs a successful run of the
        dI/dV channel (the fit models dI/dV; an IETS average is
        refused); refused when a file has no 'Current' channel.  The sent
        curves are always calibrated and normalized (the fit's pipeline); a
        console line says so when the window's boxes are off.  The record
        averages/<stem>_average.json is written first (creating the
        subfolder); the label '<directory>/averages | <name>' names the fit
        cache <directory>/averages/<stem>_fits.pkl.
        """
        if self.average is None:
            print("send to fit: run the average first.")
            return
        if self.channel != 'didv':
            print(f"send to fit refused: the fit models dI/dV, the window averages "
                  f"{STS.CHANNEL_NAMES[self.channel]}.")
            return
        sweeps = self._avg_sweeps()
        if sweeps.shape[1] == 0:
            print("send to fit: every sweep has been removed ('restore' puts them back).")
            return
        switched = [n for n, on in (('calibration', self.calibrate), ('normalization', self.normalize)) if not on]
        if switched:
            print("Send to fit: " + " and ".join(switched) + " switched on for the sent curves.")
        hp, sp = self.filters.params()
        # The files taking part: those with a kept sweep (a file emptied by removals is as if never loaded).
        active = self._active()
        parts = [{'file': self.entries[i]['file'], 'cols': self.entries[i]['cols'][self._kept(i)]} for i in active]
        data, _ = self._average_of(self._avg_sets(hp, sp, use_controls=False, calibrate=True, normalize=True))
        data_unfiltered = self._unfiltered_average()
        describe = (f"{sweeps.shape[1]} sweeps from {len(active)} files "
                    + ", ".join(self.entries[i]['label'] for i in active)
                    + (" minus " + " ".join(f"{i + 1}:{j + 1}" for i, j in self.removed) if self.removed else "") + "; "
                    + ("mean with binned points weighted by their point density" if self.resample == 'bin'
                       else "mean with interpolated points weighted by their point density" if self.resample == 'interp'
                       else "plain mean"))
        print(f"{self.tag}: record written: {self.save()}")
        an = send_sets_to_fit(self.label, self.average[0], data, data_unfiltered, parts, hp, sp,
                                 self.divide_iv, self.iv_width(), self.iv_zero_sub, self.resample,
                                 self.fit_settings, self.cache_path(), self.tag, describe)
        if an is not None:
            self._fits.append(an)

    def draw(self):
        ax = self.ax
        ax.clear()
        self._lines = []
        # Left panel: the files with a kept sweep (an emptied file is absent).  Overlaid:
        # every remaining sweep (built with the controls' settings), one colour per file.
        # Stacked with 'substack' on: the same, offset.  Stacked with 'substack' off: one
        # curve per file, the plain mean of its kept sweeps, in the file's colour.  After
        # a run the sweeps are shown as the run's grid has them (its range; binned or
        # interpolated when the run brought the file onto the grid); before a run on
        # the files' own axes.
        tol = self.ENERGY_TOL
        window = (self.average[1], self.average[2]) if self.average is not None else None
        active = self._active()
        n_empty = len(self.entries) - len(active)
        stacked, substack = self.stacked, self.substack
        self._per_file_view = stacked and not substack
        curves = []                      # (entry index, sweep index or None, V, curve)
        for i in active:
            e = self.entries[i]
            mp = self.average[3][i] if self.average is not None else None
            if mp is not None:
                V, sw = self.average[0], self._to_grid(e['sweeps'], mp)
            else:
                m = ((e['V'] >= window[0] - tol) & (e['V'] <= window[1] + tol)) if window \
                    else np.ones(len(e['V']), bool)
                V, sw = e['V'][m], e['sweeps'][m, :]
            if self._per_file_view:
                curves.append((i, None, V, self._file_average(sw[:, self._kept(i)])))
            else:
                for j in self._kept(i):
                    curves.append((i, j, V, sw[:, j]))
        n = len(curves)
        n_sweeps = sum(len(self._kept(i)) for i in active)
        n_removed = len(self.removed)
        max_range = max((float(np.ptp(c)) for _, _, _, c in curves), default=0.0)
        offset = self.STACK_OFFSET_FRAC * max_range if stacked else 0.0
        # The average (after a run): the plain mean of every kept sweep.
        avg_curve = None
        avg_sweeps = self._avg_sweeps()
        if avg_sweeps is not None and avg_sweeps.shape[1] > 0:
            avg, _ = self._average_of(self._avg_sets())
            avg_curve = (self.average[0], avg)
        # Stacked: the average (after a run) sits at the bottom, the curves above it,
        # and the two panels share PANELS_WIDTH so that one curve's range in the
        # stack has the same height-to-width ratio as the average's range in the
        # right panel: with r = max_range / yspan, w_stack = W_avg * r and
        # w_stack + W_avg = PANELS_WIDTH give w_stack = PANELS_WIDTH * r / (1 + r).
        # w_stack is kept within [0.05, LEFT_WIDTH]; the average panel takes the rest.
        n_slots = n + (1 if (stacked and avg_curve is not None) else 0)
        if stacked and n:
            yspan = (n_slots - 1) * offset + max_range
            r = max_range / yspan if yspan > 0 else 1.0
            w_left = float(np.clip(self.PANELS_WIDTH * r / (1.0 + r), 0.05, self.LEFT_WIDTH))
        else:
            w_left = self.LEFT_WIDTH
        ax.set_position([self.LEFT_X, self.PANEL_Y, w_left, self.PANEL_H])
        self.ax_avg.set_position([self.LEFT_X + w_left + self.GAP, self.PANEL_Y,
                                  self.PANELS_WIDTH - w_left, self.PANEL_H])
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        labelled = set()                 # entries whose legend label has been attached to a curve
        for k, (i, j, V, c) in enumerate(curves):
            e = self.entries[i]
            label = None
            if i not in labelled:
                labelled.add(i)
                label = f"{e['label']} ({len(self._kept(i))} of {e['sweeps'].shape[1]} sweeps" \
                        + (", averaged)" if j is None else ")")
            line, = ax.plot(V, c + (n_slots - 1 - k) * offset, lw=1.2 if j is None else 0.9,
                            color=colors[i % len(colors)], label=label)
            if j is not None:
                self._lines.append((i, j, line))
        if stacked and avg_curve is not None:
            ax.plot(avg_curve[0], avg_curve[1], color='black', lw=1.6, label='average')
        removed_text = f" ({n_removed} removed)" if n_removed else ""
        empty_text = f" ({n_empty} empty)" if n_empty else ""
        per_file_text = ", stacked per-file means" if self._per_file_view else (", stacked" if stacked else "")
        flags_text = ("" if self.calibrate else ", uncalibrated") + ("" if self.normalize else ", not normalized")
        title = (f"Sweeps: {self.tag}: {len(self.entries)} file(s) added{empty_text}, {n_sweeps} sweeps{removed_text}"
                 + (", Feenstra" if self.divide_iv and self.channel == 'didv' else "")
                 + (f", {STS.CHANNEL_NAMES[self.channel]}" if self.channel != 'didv' else "")
                 + flags_text + per_file_text
                 + (f", common range {window[0]:.2f} to {window[1]:.2f} V" if window else "")) if self.entries \
            else (f"Sweeps: {self.tag}: no entries added ('add new' / 'add existing' in the browser, "
                  f"'add new' / 'add existing' at a marker group in the Line, Map, or line-average window)")
        if stacked and n:
            title += "\n" + ", ".join(self.entries[i]['label'] for i in active)
        ax.set_title(title, fontsize=8 if stacked else 9)
        if n and not stacked:
            ax.legend(loc='upper right', frameon=False, fontsize=8)
        ax.set_xlabel("Energy (eV)")
        q = ("(dI/dV)/(I/V)" if self.divide_iv else "dI/dV") if self.channel == 'didv' else "d2I/dV2 (IETS)"
        units = ("arb. units" if self.divide_iv and self.channel == 'didv'
                 else curve_units_of([self.entries[i]['file'] for i in active], self.channel, self.calibrate,
                                     self.normalize))
        ax.set_ylabel(f"{q} ({units}" + (", offset)" if stacked and n > 1 else ")"))

        ax2 = self.ax_avg
        ax2.clear()
        if avg_curve is not None:
            ax2.plot(avg_curve[0], avg_curve[1], color='black', lw=1.8)
            kinds = {mp['kind'] for mp in self.average[3] if mp is not None}
            how = (f"mean of {avg_sweeps.shape[1]} sweeps, binned points weighted by their point density" if 'bin' in kinds
                   else f"mean of {avg_sweeps.shape[1]} sweeps, interpolated points weighted by their point density"
                   if 'interp' in kinds else f"plain mean of {avg_sweeps.shape[1]} sweeps")
            ax2.set_title(f"Average: {self.tag}: {how}{removed_text}, "
                          f"{len(active)} files: " + ", ".join(self.entries[i]['label'] for i in active), fontsize=9)
        elif self.average is not None:
            ax2.text(0.5, 0.5, "every sweep removed ('restore')", ha='center', va='center',
                     transform=ax2.transAxes)
            ax2.set_title(f"Average: {self.tag}", fontsize=9)
        else:
            ax2.text(0.5, 0.5, "no average yet ('run')", ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title(f"Average: {self.tag}", fontsize=9)
        ax2.set_xlabel("Energy (eV)")
        ax2.set_ylabel(f"{q} ({units})")
        self.fig.canvas.draw_idle()
        self._feenstra_refresh()


if __name__ == '__main__':
    # Single-point files opened in one avgN window (the next free index of their folder's
    # 'averages' subfolder, the window's own Feenstra toggle on with the browser's broadening);
    # the average runs when 'run' is pressed.
    files = [                                     # the files of the saved average avg1
        r'experiment_dir/file.sm4',
        r'experiment_dir/file.sm4',
        r'experiment_dir/file.sm4',
        r'experiment_dir/file.sm4',
        r'experiment_dir/file.sm4',
        r'experiment_dir/file.sm4',
        r'experiment_dir/file.sm4',
    ]

    # Pre-fill of the Hampel / Savitzky-Golay boxes (filters start off).
    hampel_defaults = {'window': 4, 'n_sigma': 3}
    sgolay_defaults = {'window_length': 6, 'polyorder': 3}

    # Settings for "send to fit" (passed to Fit.select_model).
    fit_settings = {
        'known_features': [],
        'n_jobs': -1,
        'max_width_ratio': 3,
        'noise_z': 2.5,
        'use_cache': True,
    }

    directory = os.path.dirname(os.path.abspath(files[0]))
    # The next free index: the records now carry their index inside (their file names carry the
    # average's name), so it is read from them (Averages.saved_indices_of).
    index = 1 + max([0] + Averages.saved_indices_of(directory))
    window = MultiAverageWindow(index, directory, hampel_defaults, sgolay_defaults,
                                fit_settings=fit_settings, divide_iv=True, iv_broadening=0.2)
    for path in files:
        window.add(sm4file.Sm4File(path))
