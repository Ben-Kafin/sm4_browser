"""
Several sweeps per pixel in a grid file: how a file's spectroscopy columns group into pixels and
sweeps, the direction of every sweep, and the views the windows work on (one column per pixel).

The grouping is read from the file, never assumed:
  - pixels: the columns' recorded locations (RHK_SpecDrift_Xcoord / Ycoord); columns within the
    location tolerance of each other are one pixel; the number of sweeps per pixel k is the number
    of columns at every location (the same everywhere, else the file is taken as one sweep per
    column and a console line says why);
  - order within a pixel: the recorded times (RHK_SpecDrift_Time) when the file has them, else the
    file's column order (page by page);
  - direction of a sweep (+1: the same bias direction as the pixel's first sweep, -1: the other).
    A forward sweep leaves the setpoint: it starts at the axis end holding the setpoint bias
    (RHK_Bias) with the feedback just opened, so its current starts at the setpoint current
    (RHK_Current); a reverse sweep follows it back from the far end, starting where the forward
    sweep ended, and the pixel's first sweep is always forward.
      * sweeps on pages of their own (one page per direction): the page's energy axis
        (RHK_Xoffset + i RHK_Xscale); a page whose axis runs the other way is reversed onto the
        first page's axis and its columns are reverse sweeps;
      * sweeps sharing a page (the instrument's layout: a pixel's sweeps as consecutive columns at
        one recorded location, every column stored in the page's axis order whichever way it ran):
        a column whose never-written samples sit at the start of the axis was stopped while
        running down the axis (reverse); a column stored in acquisition order against the page's
        axis is recognised by its I(V) matching the pixel's first sweep only when reversed
        (correlation above SWEEP_FLIP_CORR and better than as stored by SWEEP_FLIP_MARGIN) and is
        reversed onto the axis; every other sweep takes the pair rule: the file's 'Topography'
        spectroscopy page records the tip height, held while the feedback is open, so a pixel's
        consecutive sweeps with one tip height are one feedback period (feedback_periods), and
        in a period the stored order is forward, reverse, forward, ... (period_reverse); a period
        of one sweep is forward only.  Without a tip height such a sweep is taken as forward
        (said on the console).
After grouping, every column holds its values on one ascending-index energy axis (the first
page's), so every pipeline step works on it unchanged; the direction is kept for the clock (the
time of every energy point, Sm4File.energy_times / energy_edge_times), for the calibration's
Vbias lock (a reverse sweep is pinned at its own first sample, STS.sweep_pins) and for the
stopped-sweep test (a sweep is stopped when its last acquired sample, in acquisition order, was
never written).  The current's report (STS._direction_report) says on the console how far the
forward sweeps start from the setpoint current and how closely a sweep pair meets at the far
end; it is a report, never a filter.

Views: for a grid (STS.sweep_view) one column per pixel, in grid order, made from the chosen
sweeps of every pixel: 'forward sweeps' (the mean of the pixel's +1 sweeps), 'reverse sweeps'
(of its -1 sweeps), 'all sweeps' (the mean of all its sweeps), or 'sweep n' (its n-th sweep).
The raw channel and current columns are averaged over the chosen kept sweeps (calibration, the
Feenstra division and normalization then run per pixel as for a single sweep); a pixel without
a chosen kept sweep is a never-acquired column.  For a single-point file (one pixel) a view is
the chosen sweeps themselves (STS.subset of chosen_columns), one column per sweep.

This module holds the grouping of columns into pixels by location (group_locations), the
correlation of the reversed-storage test (_corr), the pair rule's helpers (feedback_periods,
period_reverse) and the view helpers (sweep_modes, default_mode, chosen_columns); the grouping
itself, the direction tests and the reordering of the columns live in sts.STS._group_sweeps and
STS._from_sweep_pages, which read a file's sweep_info dict: 'k' (sweeps per pixel), 'n_pix',
'pixel' and 'sweep' (per column), 'dir' (per column, +1 / -1), 'acq_desc' (per column: acquired
down the axis), 'cols_of' (per pixel, the column indices), 'locs' (n_pix, 2) and 'notes'.
"""

import numpy as np

FORWARD, REVERSE, ALL = 'forward sweeps', 'reverse sweeps', 'all sweeps'   # the view names (radio labels, records)
SWEEP_FLIP_CORR = 0.9
SWEEP_FLIP_MARGIN = 0.2


def sweep_modes(info):
    """The view modes a file offers (its sweeps), the default first; () for one sweep per pixel."""
    if info is None or info['k'] <= 1:
        return ()
    modes = []
    dirs = info['dir']
    if np.any(dirs > 0):
        modes.append(FORWARD)
    if np.any(dirs < 0):
        modes.append(REVERSE)
    modes.append(ALL)
    modes.extend(f"sweep {s + 1}" for s in range(info['k']))
    return tuple(modes)


def default_mode(info):
    """The view a file opens with: the first of sweep_modes (FORWARD when any sweep is forward), None for one sweep per pixel."""
    m = sweep_modes(info)
    return m[0] if m else None


def chosen_columns(info, mode):
    """
    For every pixel (grid order) the column indices the view `mode` takes
    (FORWARD: the pixel's +1 sweeps; REVERSE: its -1 sweeps; ALL: every
    sweep; 'sweep n': its n-th sweep, 1-based, in acquisition order), a
    list of int arrays, one per pixel, possibly empty.  `info` is
    STS.sweep_info; ValueError for an unknown mode.
    """
    sw, d = info['sweep'], info['dir']
    out = []
    for p in range(info['n_pix']):
        cols = info['cols_of'][p]
        if mode == FORWARD:
            c = cols[d[cols] > 0]
        elif mode == REVERSE:
            c = cols[d[cols] < 0]
        elif mode == ALL:
            c = cols
        elif mode.startswith('sweep '):
            n = int(mode.split()[1]) - 1
            c = cols[sw[cols] == n]
        else:
            raise ValueError(f"unknown sweep view '{mode}'")
        out.append(np.asarray(c, dtype=int))
    return out


def group_locations(xy, tol):
    """
    Pixel index of every column from its recorded location (columns within tol of a pixel's first
    column join it), pixels numbered in the order their first column appears.  Returns (pixel (N,),
    pixel locations (n_pix, 2)).
    """
    xy = np.asarray(xy, dtype=np.float64)
    N = len(xy)
    pixel = np.full(N, -1, dtype=int)
    locs = []
    # A grid of cells of size tol makes the search linear.
    cell = {}
    for c in range(N):
        key = tuple(np.floor(xy[c] / max(tol, 1e-30)).astype(np.int64))
        found = -1
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for p in cell.get((key[0] + dx, key[1] + dy), ()):
                    if np.hypot(*(xy[c] - locs[p])) <= tol:
                        found = p
                        break
                if found >= 0:
                    break
            if found >= 0:
                break
        if found < 0:
            found = len(locs)
            locs.append(xy[c].copy())
            cell.setdefault(key, []).append(found)
        pixel[c] = found
    return pixel, np.array(locs).reshape(-1, 2)


def feedback_periods(tip_height, cols_of, k, ever):
    """
    The sweeps per feedback period of a file holding k sweeps at every pixel
    (STS._group_sweeps): within a pixel, consecutive sweeps (its columns in
    acquisition order, `cols_of`) with one tip height (`tip_height` per
    column, NaN never written) form one period.  Only started columns
    (`ever`) count, and the last period of a pixel stopped before its end
    is left out as cut short.  Returns (g, None) when every complete period
    holds the same number g of sweeps and g divides k; else (None, why).
    """
    if tip_height is None:
        return None, "no tip height (no usable 'Topography' spectroscopy page)"
    th = np.asarray(tip_height, dtype=np.float64)
    ever = np.asarray(ever, dtype=bool)
    lengths = []
    for c in cols_of:
        c = np.asarray(c, dtype=int)
        started = c[ever[c] & np.isfinite(th[c])]
        if started.size == 0:
            continue
        h = th[started]
        runs, n = [], 1
        for j in range(1, len(h)):
            if h[j] == h[j - 1]:
                n += 1
            else:
                runs.append(n)
                n = 1
        runs.append(n)
        lengths.extend(runs if started.size == c.size else runs[:-1])
    if not lengths:
        return None, "no pixel completed a feedback period"
    g = int(lengths[0])
    if any(n != g for n in lengths):
        return None, (f"the tip height groups the sweeps unevenly ({min(lengths)} to {max(lengths)} per "
                      f"feedback period)")
    if k % g:
        return None, f"{g} sweeps per feedback period do not divide the {k} sweeps per pixel"
    return g, None


def period_reverse(s, g):
    """Whether the sweep at index s of a pixel (0-based, acquisition order) is a reverse one with g sweeps per feedback period: the period's stored order is forward, reverse, forward, ..."""
    return (int(s) % int(g)) % 2 == 1


def _corr(a, b):
    """
    Pearson correlation of two columns over the samples finite in both
    (the reversed-storage test of STS._group_sweeps); NaN with fewer than
    three such samples or when either column is constant.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / den) if den > 0 else np.nan
