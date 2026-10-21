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
  - direction of a sweep (+1: the same bias direction as the pixel's first sweep, -1: the other):
      * sweeps on pages of their own (one page per direction): the page's energy axis
        (RHK_Xoffset + i RHK_Xscale); a page whose axis runs the other way is reversed onto the
        first page's axis and its columns are reverse sweeps;
      * sweeps sharing a page: a column whose never-written samples sit at the start of the axis
        was stopped while running down the axis (reverse); a column stored in acquisition order
        against the page's axis is recognised by its I(V) matching the pixel's first sweep only
        when reversed (correlation above SWEEP_FLIP_CORR and better than as stored by
        SWEEP_FLIP_MARGIN); it is reversed onto the axis.  Otherwise the direction is not
        recorded and the sweep is taken as running like the page (said on the console).
After grouping, every column holds its values on one ascending-index energy axis (the first
page's), so every pipeline step works on it unchanged; the direction is kept for the clock (the
time of every energy point, Sm4File.energy_times / energy_edge_times) and for the stopped-sweep
test (a sweep is stopped when its last acquired sample, in acquisition order, was never written).

Views (STS.sweep_view): one column per pixel, in grid order, made from the chosen sweeps of every
pixel: 'forward sweeps' (the mean of the pixel's +1 sweeps), 'reverse sweeps' (of its -1 sweeps),
'all sweeps' (the mean of all its sweeps), or 'sweep n' (its n-th sweep).  The raw channel and current columns
are averaged over the chosen kept sweeps (calibration, the Feenstra division and normalization
then run per pixel as for a single sweep); a pixel without a chosen kept sweep is a
never-acquired column.

This module holds the grouping of columns into pixels by location (group_locations), the
correlation of the reversed-storage test (_corr), and the view helpers (sweep_modes,
default_mode, chosen_columns); the grouping itself, the direction tests and the reordering of
the columns live in sts.STS._group_sweeps and STS._from_sweep_pages, which read a file's
sweep_info dict: 'k' (sweeps per pixel), 'n_pix', 'pixel' and 'sweep' (per column), 'dir' (per
column, +1 / -1), 'acq_desc' (per column: acquired down the axis), 'cols_of' (per pixel, the
column indices), 'locs' (n_pix, 2) and 'notes'.
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
