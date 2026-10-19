"""
Alignment of grid lines for a line average (lineavg.py), on arrays only:
the dynamic time warping (DTW) of a line's spectrum positions against the
main line's, the per-location map it gives, the equalizing of the stretch
over every line (the main included), the marker-section map of the
alignment editor, the common energy and position grids, the resampling of
heatmaps and stripe profiles onto them, and the average with or without
crop.  Nothing here draws or holds a widget.

Positions are in Å along a line (each line's own axis starts at 0); maps
are non-decreasing arrays of the same length as the line's positions.

The alignment, step by step (lineavg.LineAverage.run):
  1. DTW: for every line but the main, the monotonic path over all of its
     spectrum positions against the main's that minimizes the summed cost
     (dtw_path), cost per pair = (1 - Pearson r of the two dI/dV curves)
     + (1 - Pearson r of the two current curves) on the common energy
     grid (pair_cost); one-to-many allowed, no band.  dtw_map turns the
     path into the main-frame position of every location (the mean of the
     main positions paired with it; a run of locations paired with the
     main's first (last) position alone is laid out before (after) it at
     the line's own step, so a line may extend beyond the main).  The main
     keeps displacement 0 and stretch 0 at this step.
  2. Equalizing, stretch only (equalize_stretch): every line's log-stretch
     into the main frame, ln(d mapped / d own), is taken as a function of
     the main-frame position (the main's is 0 everywhere; a line is
     unstretched where it is absent); the correction is their equal-weight
     mean over all N lines, main included, at every point; the common
     frame is the main frame re-scaled pointwise by exp(-mean), anchored at
     the main's centre, so the N stretches sum to zero everywhere (two
     lines: equal and opposite halves of the DTW stretch), the main's net
     displacement stays 0, and every line keeps the displacement the DTW
     gave it.
  3. The common grids (common_energy, common_positions), the resampling of
     every line's heatmap and stripe onto them (resample_columns,
     resample_profile), and the average (average_stack).
After `apply` in the editor, marker_map replaces the DTW map of every
line (linear between consecutive markers, the DTW map continued and
shifted to join outside the outer markers); the equalizing is applied on
top in the same way.
"""

if __package__ in (None, ''):
    # Run as a file (Spyder F5): make the package importable from wherever its folder sits.
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np


# ---------------------------------------------------------------------- #
#  Energy grid and curves on it                                            #
# ---------------------------------------------------------------------- #

def common_energy(energies, tol=1e-9):
    """
    The common energy grid of several lines: the finest energy step among
    them over the range they all share, the points of the finest-step line
    within that range (so lines sharing one axis keep their own points).
    `energies` are the lines' axes in any order.  ValueError when the
    shared range is shorter than a step.  Returns the ascending grid.
    """
    axes = [np.sort(np.asarray(e, dtype=np.float64)) for e in energies]
    steps = [float(np.median(np.diff(a))) if len(a) > 1 else np.inf for a in axes]
    lo = max(float(a[0]) for a in axes)
    hi = min(float(a[-1]) for a in axes)
    k = int(np.argmin(steps))
    if hi - lo < steps[k]:
        raise ValueError(f"the lines share no energy range (overlap {lo:.3f} to {hi:.3f} V).")
    a = axes[k]
    m = (a >= lo - tol) & (a <= hi + tol)
    return a[m].copy()


def on_energy_grid(curves, energy, grid):
    """
    Columns (npts, N) on their own axis `energy` (any order) interpolated
    linearly onto the ascending `grid` (n_grid, N); a NaN column stays NaN.
    """
    c = np.asarray(curves, dtype=np.float64)
    e = np.asarray(energy, dtype=np.float64)
    order = np.argsort(e)
    e, c = e[order], c[order, :]
    out = np.full((len(grid), c.shape[1]), np.nan)
    for j in range(c.shape[1]):
        col = c[:, j]
        if np.all(np.isfinite(col)):
            out[:, j] = np.interp(grid, e, col)
    return out


# ---------------------------------------------------------------------- #
#  DTW                                                                     #
# ---------------------------------------------------------------------- #

def pearson_cost(a, b):
    """
    1 - Pearson r between every column of a (npts, n) and every column of
    b (npts, m): an (n, m) cost matrix.  A column without variance (or
    with a non-finite value) is uncorrelated with everything: cost 1.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok_a = np.all(np.isfinite(a), axis=0)
    ok_b = np.all(np.isfinite(b), axis=0)
    za = np.zeros_like(a)
    zb = np.zeros_like(b)
    for z, x, ok in ((za, a, ok_a), (zb, b, ok_b)):
        for j in np.flatnonzero(ok):
            col = x[:, j] - np.mean(x[:, j])
            s = np.sqrt(np.sum(col * col))
            if s > 0:
                z[:, j] = col / s
            else:
                ok[j] = False
    r = za.T @ zb
    cost = 1.0 - r
    cost[~ok_a, :] = 1.0
    cost[:, ~ok_b] = 1.0
    return cost


def pair_cost(didv_a, didv_b, iv_a=None, iv_b=None):
    """
    The DTW cost between the locations of line a (columns of didv_a (n_e,
    n), iv_a (n_e, n) or None) and of line b: (1 - r of the dI/dV curves)
    + (1 - r of the current curves); the dI/dV term alone when either line
    has no current.  Both on the common energy grid.
    """
    cost = pearson_cost(didv_a, didv_b)
    if iv_a is not None and iv_b is not None:
        cost = cost + pearson_cost(iv_a, iv_b)
    return cost


def dtw_path(cost):
    """
    The monotonic path from (0, 0) to (n-1, m-1) through an (n, m) cost
    matrix (steps (1, 0), (0, 1), (1, 1)) that minimizes the summed cost:
    every location of either sequence paired with at least one of the
    other, one-to-many allowed, no band.  Returns ([(i, j), ...], total).
    """
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    D = np.full((n, m), np.inf)
    D[0, 0] = cost[0, 0]
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            best = np.inf
            if i > 0:
                best = min(best, D[i - 1, j])
            if j > 0:
                best = min(best, D[i, j - 1])
            if i > 0 and j > 0:
                best = min(best, D[i - 1, j - 1])
            D[i, j] = cost[i, j] + best
    i, j = n - 1, m - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        cands = []
        if i > 0 and j > 0:
            cands.append((D[i - 1, j - 1], i - 1, j - 1))
        if i > 0:
            cands.append((D[i - 1, j], i - 1, j))
        if j > 0:
            cands.append((D[i, j - 1], i, j - 1))
        _, i, j = min(cands, key=lambda t: t[0])
        path.append((i, j))
    path.reverse()
    return path, float(D[n - 1, m - 1])


def dtw_map(path, pos_a, pos_main):
    """
    The main-frame position of every location of line a from a DTW path
    over (location of a, location of the main): the mean of the main
    positions paired with it, except at the ends: a run of a's first
    locations paired with the main's first position alone is laid out
    before it at a's own step (the run's last location at the main's first
    position), and a run of a's last locations paired with the main's last
    position alone after it; a's first (last) location paired with several
    of the main's first (last) positions takes the last (first) of them, so
    the main's own overhang is not covered by stretched data.  Returns a
    non-decreasing array of len(pos_a).
    """
    pos_a = np.asarray(pos_a, dtype=np.float64)
    pos_main = np.asarray(pos_main, dtype=np.float64)
    n, m = len(pos_a), len(pos_main)
    pairs = [[] for _ in range(n)]
    for i, j in path:
        pairs[i].append(j)
    step_a = float(abs(pos_a[-1] - pos_a[0]) / max(n - 1, 1)) if n > 1 else float(abs(pos_main[-1] - pos_main[0]) / max(m - 1, 1))
    out = np.array([float(np.mean(pos_main[js])) for js in pairs])
    # The starting run paired with the main's first position alone.
    k = 0
    while k < n and pairs[k] == [0]:
        k += 1
    k -= 1                                     # the run's last location (-1: no run)
    if k >= 0 and m > 1:
        for i in range(k + 1):
            out[i] = pos_main[0] - (k - i) * step_a
    # The ending run paired with the main's last position alone.
    k2 = n - 1
    while k2 >= 0 and pairs[k2] == [m - 1]:
        k2 -= 1
    k2 += 1                                    # the run's first location (n: no run)
    if k2 <= n - 1 and m > 1 and k2 > k:
        for i in range(k2, n):
            out[i] = pos_main[-1] + (i - k2) * step_a
    # The first (last) location paired with several of the main's first (last) positions.
    if len(pairs[0]) > 1 and pairs[0][0] == 0 and not (k >= 0):
        out[0] = pos_main[pairs[0][-1]]
    if len(pairs[-1]) > 1 and pairs[-1][-1] == m - 1 and not (k2 <= n - 1):
        out[-1] = pos_main[pairs[-1][0]]
    return np.maximum.accumulate(out)


def fill_map(mapped, acquired, pos_own):
    """
    A map known at the acquired locations only (`mapped` of len(pos_own),
    meaningful where `acquired`): the others filled by linear interpolation
    over the acquired neighbours, and beyond the first or last acquired
    location continued at the line's own step.
    """
    mapped = np.array(mapped, dtype=np.float64)
    acquired = np.asarray(acquired, dtype=bool)
    idx = np.arange(len(mapped), dtype=np.float64)
    acq = np.flatnonzero(acquired)
    if acq.size == 0:
        return np.asarray(pos_own, dtype=np.float64).copy()
    if acq.size == len(mapped):
        return mapped
    out = np.interp(idx, idx[acq], mapped[acq])
    pos_own = np.asarray(pos_own, dtype=np.float64)
    step = float(abs(pos_own[-1] - pos_own[0]) / max(len(pos_own) - 1, 1))
    first, last = acq[0], acq[-1]
    for i in range(first):
        out[i] = mapped[first] - (first - i) * step
    for i in range(last + 1, len(mapped)):
        out[i] = mapped[last] + (i - last) * step
    return np.maximum.accumulate(out)


def align_to_main(didv_a, iv_a, pos_a, acq_a, didv_main, iv_main, pos_main, acq_main):
    """
    The DTW alignment of line a against the main line: the acquired
    locations of both take part (pair_cost over their curves on the common
    energy grid), and the map of every location of a (dtw_map, then
    fill_map for the never-acquired ones).  Returns (map, total cost).
    """
    ia = np.flatnonzero(np.asarray(acq_a, dtype=bool))
    im = np.flatnonzero(np.asarray(acq_main, dtype=bool))
    if ia.size == 0 or im.size == 0:
        raise ValueError("a line with no acquired spectrum cannot be aligned.")
    cost = pair_cost(np.asarray(didv_a)[:, ia], np.asarray(didv_main)[:, im],
                     None if iv_a is None else np.asarray(iv_a)[:, ia],
                     None if iv_main is None else np.asarray(iv_main)[:, im])
    path, total = dtw_path(cost)
    mapped_acq = dtw_map(path, np.asarray(pos_a)[ia], np.asarray(pos_main)[im])
    mapped = np.full(len(pos_a), np.nan)
    mapped[ia] = mapped_acq
    return fill_map(mapped, np.asarray(acq_a, dtype=bool), pos_a), total


# ---------------------------------------------------------------------- #
#  Maps: evaluation, inversion, marker sections                            #
# ---------------------------------------------------------------------- #

def map_at(p, pos_own, mapped):
    """
    A map (own positions -> mapped positions, both non-decreasing) at any
    own positions p: linear between the locations, continued at the end
    slopes outside them (a line's own step beyond its ends).
    """
    pos_own = np.asarray(pos_own, dtype=np.float64)
    mapped = np.asarray(mapped, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if len(pos_own) < 2:
        return mapped[0] + (p - pos_own[0])
    out = np.interp(p, pos_own, mapped)
    s0 = (mapped[1] - mapped[0]) / (pos_own[1] - pos_own[0]) if pos_own[1] != pos_own[0] else 1.0
    s1 = (mapped[-1] - mapped[-2]) / (pos_own[-1] - pos_own[-2]) if pos_own[-1] != pos_own[-2] else 1.0
    lo, hi = p < pos_own[0], p > pos_own[-1]
    out = np.where(lo, mapped[0] + (p - pos_own[0]) * s0, out)
    out = np.where(hi, mapped[-1] + (p - pos_own[-1]) * s1, out)
    return out


def inverse_map_at(q, pos_own, mapped):
    """The own position of a line at mapped positions q (the inverse of map_at; ties in the map resolved by their mean)."""
    pos_own = np.asarray(pos_own, dtype=np.float64)
    mapped = np.asarray(mapped, dtype=np.float64)
    u, inv = np.unique(mapped, return_inverse=True)
    own_u = np.array([np.mean(pos_own[inv == k]) for k in range(len(u))])
    return map_at(q, u, own_u)


def marker_map(pos_own, dtw_mapped, markers_own, markers_main):
    """
    The marker-section map of a line: linear between consecutive markers
    (marker j of the line at its own position markers_own[j] lands at the
    main's own position markers_main[j]), and outside the outer markers
    the DTW map continued, shifted so that it joins the marker map at the
    outer marker.  The markers are paired by index and must be in the same
    order in the line and in the main (ValueError otherwise).  With no
    marker on either side the DTW map is returned as it is.  Returns the
    map at the line's locations (non-decreasing).
    """
    pos_own = np.asarray(pos_own, dtype=np.float64)
    dtw_mapped = np.asarray(dtw_mapped, dtype=np.float64)
    mo = np.asarray(markers_own, dtype=np.float64)
    mm = np.asarray(markers_main, dtype=np.float64)
    if len(mo) == 0 and len(mm) == 0:
        # No marker (the editor's 'Markers' slider at 0): no section to pin, the DTW map stands.
        return np.maximum.accumulate(dtw_mapped)
    if len(mo) != len(mm) or len(mo) == 0:
        raise ValueError("the line and the main must have the same markers.")
    order = np.argsort(mm)
    mo, mm = mo[order], mm[order]
    if np.any(np.diff(mo) < 0):
        raise ValueError("the markers must be in the same order in every line.")
    out = np.empty(len(pos_own))
    if len(mo) > 1:
        inside = (pos_own >= mo[0]) & (pos_own <= mo[-1])
        out[inside] = np.interp(pos_own[inside], mo, mm)
    else:
        inside = np.zeros(len(pos_own), dtype=bool)
    before, after = pos_own < mo[0], pos_own > mo[-1]
    d0 = map_at(mo[0], pos_own, dtw_mapped)
    d1 = map_at(mo[-1], pos_own, dtw_mapped)
    out[before] = dtw_mapped[before] + (mm[0] - d0)
    out[after] = dtw_mapped[after] + (mm[-1] - d1)
    if len(mo) == 1:
        out[pos_own == mo[0]] = mm[0]
    return np.maximum.accumulate(out)


# ---------------------------------------------------------------------- #
#  Equalizing the stretch                                                  #
# ---------------------------------------------------------------------- #

def log_stretch(pos_own, mapped):
    """The log-stretch of a map at every location, ln(d mapped / d own) (np.gradient); 0 where the slope is not positive."""
    pos_own = np.asarray(pos_own, dtype=np.float64)
    mapped = np.asarray(mapped, dtype=np.float64)
    if len(pos_own) < 2:
        return np.zeros(len(pos_own))
    slope = np.gradient(mapped, pos_own)
    out = np.zeros(len(pos_own))
    ok = slope > 0
    out[ok] = np.log(slope[ok])
    return out


def equalize_stretch(maps, positions, main):
    """
    The common frame of N lines from their main-frame maps (maps[k] at the
    line's own positions[k]; the main's map is its own positions): every
    line's log-stretch as a function of the main-frame position (0 where
    the line is absent), their equal-weight mean mu over all N lines at
    every point of a fine main-frame axis (the main's step over the union
    of the mapped ranges), and the frame Q(q) = q_c + integral from the
    main's centre q_c of exp(-mu), so that the lines' stretches into the
    common frame sum to zero everywhere, the main's net displacement stays
    0, and the displacements the maps gave are left alone.  Returns
    (common maps [the common position of every location of every line],
    axis, Q on it).
    """
    N = len(maps)
    pm = np.asarray(positions[main], dtype=np.float64)
    step = float(abs(pm[-1] - pm[0]) / max(len(pm) - 1, 1)) if len(pm) > 1 else 1.0
    lo = min(float(np.min(mp)) for mp in maps)
    hi = max(float(np.max(mp)) for mp in maps)
    axis = np.arange(lo - step, hi + 1.5 * step, step)
    mu = np.zeros(len(axis))
    for k in range(N):
        if k == main:
            continue
        lam = log_stretch(positions[k], maps[k])
        q = np.asarray(maps[k], dtype=np.float64)
        u, inv = np.unique(q, return_inverse=True)
        lam_u = np.array([np.mean(lam[inv == i]) for i in range(len(u))])
        lam_axis = np.interp(axis, u, lam_u, left=0.0, right=0.0) if len(u) > 1 else np.zeros(len(axis))
        mu += lam_axis
    mu /= float(N)
    scale = np.exp(-mu)
    q_c = 0.5 * (float(pm[0]) + float(pm[-1]))
    Q = np.concatenate([[0.0], np.cumsum(0.5 * (scale[1:] + scale[:-1]) * np.diff(axis))])
    Q = Q - np.interp(q_c, axis, Q) + q_c
    common = [np.interp(np.asarray(mp, dtype=np.float64), axis, Q) for mp in maps]
    return common, axis, Q


# ---------------------------------------------------------------------- #
#  Position grid, resampling, average                                      #
# ---------------------------------------------------------------------- #

def common_positions(common_maps, positions, crop):
    """
    The common position grid: the finest step among the lines (each line's
    own step), over the union of the lines' common ranges (crop off) or
    their intersection (crop on).  ValueError when the intersection is
    empty.  Returns the ascending grid.
    """
    steps = [float(abs(p[-1] - p[0]) / max(len(p) - 1, 1)) for p in positions if len(p) > 1]
    step = min(steps) if steps else 1.0
    los = [float(np.min(m)) for m in common_maps]
    his = [float(np.max(m)) for m in common_maps]
    if crop:
        lo, hi = max(los), min(his)
        if hi - lo < step:
            raise ValueError("crop: the lines share no position range.")
    else:
        lo, hi = min(los), max(his)
    n = int(np.floor((hi - lo) / step + 1e-9)) + 1
    return lo + step * np.arange(n)


def _merge_ties(q, rows):
    """Locations with the same common position merged: their rows averaged (NaN-aware); returns (unique q, rows)."""
    u, inv = np.unique(q, return_inverse=True)
    if len(u) == len(q):
        order = np.argsort(q)
        return q[order], rows[order]
    out = np.full((len(u),) + rows.shape[1:], np.nan)
    with np.errstate(invalid='ignore'):
        for k in range(len(u)):
            out[k] = np.nanmean(rows[inv == k], axis=0)
    return u, out


def resample_rows(q, rows, grid):
    """
    Rows (one per location, (n, ...)) at common positions q (non-decreasing)
    interpolated linearly along position at the `grid` points: a grid point
    between two locations takes their interpolation when both are acquired
    (finite), NaN when either is not acquired; outside the first and last
    location NaN (half a step of coverage is added at each end: a grid
    point within half the local step of an end location takes that
    location).
    """
    q = np.asarray(q, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.float64)
    q, rows = _merge_ties(q, rows)
    n = len(q)
    out = np.full((len(grid),) + rows.shape[1:], np.nan)
    if n == 0:
        return out
    if n == 1:
        return out
    finite = np.all(np.isfinite(rows.reshape(n, -1)), axis=1)
    step0 = q[1] - q[0]
    step1 = q[-1] - q[-2]
    for g, x in enumerate(np.asarray(grid, dtype=np.float64)):
        if x < q[0] - 0.5 * step0 or x > q[-1] + 0.5 * step1:
            continue
        if x <= q[0]:
            if finite[0]:
                out[g] = rows[0]
            continue
        if x >= q[-1]:
            if finite[-1]:
                out[g] = rows[-1]
            continue
        j = int(np.searchsorted(q, x, side='right')) - 1
        j = min(max(j, 0), n - 2)
        if not (finite[j] and finite[j + 1]):
            continue
        t = (x - q[j]) / (q[j + 1] - q[j]) if q[j + 1] > q[j] else 0.0
        out[g] = (1.0 - t) * rows[j] + t * rows[j + 1]
    return out


def resample_columns(columns, energy, grid_energy, common_pos, grid_pos):
    """
    A line's columns (npts, N) on its own energy axis, at common positions
    common_pos (N,): on the common energy grid (on_energy_grid), then along
    position at grid_pos (resample_rows).  Returns (n_grid_energy, n_grid_pos),
    NaN where the line does not cover a pixel.
    """
    c = on_energy_grid(columns, energy, grid_energy)          # (n_e, N)
    rows = resample_rows(np.asarray(common_pos), c.T, grid_pos)   # (n_pos, n_e)
    return rows.T


def resample_profile(profile, common_pos, grid_pos):
    """A stripe profile (n,) at common positions (n,) onto the position grid (NaN outside its coverage)."""
    return resample_rows(np.asarray(common_pos), np.asarray(profile, dtype=np.float64)[:, None], grid_pos)[:, 0]


def average_stack(stack, crop):
    """
    The average of a stack of resampled arrays (n_lines, ...): with crop
    off the mean over the lines covering each pixel (a pixel no line
    covers is NaN); with crop on a pixel is averaged only where every line
    covers it (NaN otherwise).
    """
    a = np.asarray(stack, dtype=np.float64)
    with np.errstate(invalid='ignore'):
        if crop:
            full = np.all(np.isfinite(a), axis=0)
            out = np.full(a.shape[1:], np.nan)
            if full.any():
                out[full] = np.mean(a[:, full], axis=0)
            return out
        return np.nanmean(a, axis=0)


def stripe_positions(pos, n_cells):
    """The positions of a stripe's cells along a line: n_cells cell centres spanning pos[0] to pos[-1] (as drawn)."""
    pos = np.asarray(pos, dtype=np.float64)
    edges = np.linspace(pos[0], pos[-1], n_cells + 1)
    return 0.5 * (edges[1:] + edges[:-1])


if __name__ == '__main__':
    # Checks on arrays: a line against itself gives the identity map at cost 0; a line
    # against a copy stretched by 1.2 is undone by the equalizing to within the grid.
    rng = np.random.default_rng(0)
    E = np.linspace(-2.0, -0.8, 121)
    pos = np.arange(32) * 1.2
    base = np.exp(-((E[:, None] + 1.4 + 0.3 * np.sin(pos[None, :] / 6.0)) / 0.2) ** 2) + 0.1 * rng.standard_normal((121, 32))
    acq = np.ones(32, dtype=bool)
    mp, total = align_to_main(base, None, pos, acq, base, None, pos, acq)
    print("identity:", np.allclose(mp, pos), "cost", round(total, 6))
    common, axis, Q = equalize_stretch([pos, mp], [pos, pos], 0)
    print("equalized identity:", np.allclose(common[0], pos, atol=1e-6), np.allclose(common[1], pos, atol=1e-6))
    grid = common_positions(common, [pos, pos], crop=False)
    print("grid points:", len(grid), "step", round(float(grid[1] - grid[0]), 3))
    r = resample_columns(base, E, E, common[1], grid)
    print("resampled finite:", int(np.isfinite(r).sum()), "of", r.size)
