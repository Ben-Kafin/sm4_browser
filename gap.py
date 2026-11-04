"""
The 'denoise' step of the pipeline (sts.STS.process_arrays; the pipeline
box's first row, off at open): the per-point gap factor g of every sweep,
read from the recorded current against the Feenstra ratio of the same
sweep, divided out of the dI/dV page and the current before the
calibration.  The calibration then regresses the denoised current on the
integral of the denoised dI/dV page, with its gate and locks as set.

What g is.  The tip-sample spacing of one sweep fluctuates from sample to
sample, and the spacing multiplies the current and the lock-in output of
a sample by the same factor g.  Feenstra's ratio (dI/dV)/(I/V) is free
of g for that reason, and it is also, with I0 = I - I(0 V), the
logarithmic derivative d ln|I0| / d ln|V|; integrated in ln|V| it gives
the current the sweep would have had with g = 1 at every sample.  The
recorded current divided by that current is g, point by point, with
nothing fitted to the dI/dV shape and nothing smoothed.

The step, per sweep (denoise):
1. the calibration as set (calibration.calibrate with the window's gate
   and locks), used for its a and b only, to form the calibrated page the
   ratio needs;
2. I0 = I - I(0 V) (feenstra.iv_corrected with zero_sub; a window that
   does not reach 0 V is left as it is: its current keeps its own offset,
   and g is only as good as that offset is small);
3. F = the plain 0-sub Feenstra ratio of the calibrated page
   (feenstra.divide_by_iv with broadening 0);
4. on each side of 0 V (the point at 0 V belongs to neither) with at
   least GAP_MIN_POINTS points whose ratio is finite: ln I_rebuilt = the
   cumulative trapezoid of F against ln|V| from the window's edge toward
   0 V, shifted so that the mean of ln|I0| - ln I_rebuilt over that side is 0
   (g has mean 1 on each side), g_raw = |I0| / I_rebuilt at every point of
   the side;
5. the soft edge (the 'soft edge' row under 'denoise', on at open): each
   point's correction scaled by the share of it that is signal, w, read
   off a curve measured on the set of sweeps the step ran on
   (weight_curve), g = 1 + (g_raw - 1) w (soft_edge); with the row off,
   g = g_raw;
6. the page and the current of the sweep divided by g.

Every point with a finite ratio gets its own g: there is no band around
0 V (removed 2026-11-03; the band of 10 % of the sweep's maximum |I0|
skipped 32 of 141 points on a metal-like sweep whose current is small
near 0 V only because V is, while the current channel's own point-to-
point noise there was 0.02 pA against a 0.15 pA imprint of g).  The
points left at g = 1 are the point at 0 V itself, a point whose ratio
is not finite, and the points of a side with fewer than GAP_MIN_POINTS.

The soft edge.  The step's own error grows where the current is small,
since the rebuilt current there rests on the ratio of two small numbers,
while the spacing jitter the step removes is the same at every current.
Over every usable point of every sweep, g_raw - 1 is binned by |I0| in
N_BINS log-spaced bins from the 1st percentile of the current to its
maximum; the rms of g_raw - 1 per bin (a bin with fewer than
BIN_MIN_POINTS points has none) is the measured spread of g there.  The
largest rms over the upper half of the bins is the jitter; a bin's excess
over it, excess^2 = rms^2 - jitter^2, is the step's own error at that
current, and w = jitter^2 / (jitter^2 + excess^2) is the share of the
measured g_raw - 1 that is signal: 1 where the current is large, falling
toward 0 where it is tiny, so that those points stay near their recorded
values by measurement rather than by a band.  Bins without an rms are
interpolated in log current, and w is made non-decreasing with the
current.  When the curve cannot be measured (no bin of the upper half
has an rms, or the jitter reads 0) w is 1 everywhere and the console
note says so.  Decided 2026-11-04.

The dI/dV channel only; the per-sweep level (the setpoint scatter) and
the lock-in's own additive noise are not g and are left to the area
normalization and to averaging.  With the Vbias lock on, the setpoint
pin of a denoised forward sweep is I_setpoint / g at the setpoint point
(setpoint_divided; the 0-lock pin and a reverse sweep's pin read the
denoised current themselves).  Decided 2026-11-02.
"""

from typing import NamedTuple
import numpy as np
from scipy.integrate import cumulative_trapezoid
from sm4_Browser import calibration, feenstra


# A side of 0 V needs at least this many points with a finite ratio for its integral (a guard for
# a window that barely crosses 0 V, not a threshold on the data).
GAP_MIN_POINTS = 3
# The soft edge's curve (weight_curve): the bins of |I0|, log-spaced, and the points a bin needs
# for its rms.
N_BINS = 12
BIN_MIN_POINTS = 8


class DenoiseStep(NamedTuple):
    """
    The step's own pieces for the columns it ran on, every array (npts, N) in the original energy
    order (denoise returns it; sts.STS.stages keeps it as Stages.step, so the rebuild panels tool
    draws the step without running any of it again): cal, the page a · LIA + b of the step's
    regression (calibration.calibrate on the filtered pair with the window's gate and locks; the
    fit forms this page and the ratio only, it is never the calibration the windows show); fits,
    that regression's ColumnFit list (a, b, c, the pins, the gate's replacement), one per column;
    Fcal, the ratio the step integrates (the plain 0-sub Feenstra ratio of cal); Isub0, the
    filtered current with its value at 0 V subtracted; I_rebuilt, the current rebuilt from Fcal
    (gap_factor; NaN where not rebuilt: the point at 0 V, a point whose ratio is not finite, and
    a side of 0 V with too few points); g_raw, the per-point gap factor before the soft edge
    (|Isub0| / I_rebuilt, 1 where not rebuilt); w, each point's share of its correction that is
    signal (soft_edge; 1 everywhere when the curve was not measured or the row is off); curve, the
    soft edge's measured curve (weight_curve's dict: centres, rms, w, jitter, edges; None when it
    could not be measured or the row is off); g, the gap factor applied, 1 + (g_raw - 1) w; kept,
    the points per column left as recorded (N,): where g_raw is 1.
    """
    cal: object
    fits: object
    Fcal: object
    Isub0: object
    I_rebuilt: object
    g_raw: object
    w: object
    curve: object
    g: object
    kept: object


def _zero_point(e):
    """
    The point at 0 V as the file records it, on the ascending axis `e` (npts,): the axis is built
    from float32 fields, so the nominal 0 V point can sit at a few 1e-8 V; any |V| below a
    thousandth of the median spacing is that point.  (npts,) bool.
    """
    step = np.median(np.abs(np.diff(e))) if e.size > 1 else 1.0
    return np.abs(e) < 1e-3 * step


def gap_factor(energy, ratio, iv0):
    """
    The per-point gap factor of every column: `ratio` the plain 0-sub
    Feenstra ratio (dI/dV)/(I0/V) of the calibrated page and `iv0` the
    current with its value at 0 V subtracted, both (npts, N) in the
    original energy order.  Per column: on each side of 0 V (the point at
    0 V itself belongs to neither) with at least GAP_MIN_POINTS points
    whose ratio is finite, ln I_rebuilt is the cumulative trapezoid of
    the ratio against ln|V| from the window's edge toward 0 V, shifted so that
    mean(ln|I0| - ln I_rebuilt) = 0 over those points, and
    g = |I0| / I_rebuilt at every one of them.  Returns (g (npts, N), the
    number of points left at g = 1 per column (N,): the point at 0 V, a
    point whose ratio is not finite, a side with too few points,
    I_rebuilt (npts, N), the rebuilt current, NaN where none was rebuilt),
    the arrays in the original energy order.  This is g before the soft
    edge (denoise applies it).
    """
    energy = np.asarray(energy, dtype=np.float64)
    ratio = np.asarray(ratio, dtype=np.float64)
    iv0 = np.asarray(iv0, dtype=np.float64)
    order = np.argsort(energy)
    e = energy[order]
    zero = _zero_point(e)
    g = np.ones_like(iv0)
    rebuilt = np.full_like(iv0, np.nan)
    kept = np.zeros(iv0.shape[1], dtype=int)
    for j in range(iv0.shape[1]):
        i0 = iv0[order, j]
        f = ratio[order, j]
        usable = np.isfinite(f) & (i0 != 0.0)
        gj = np.ones_like(i0)
        rj = np.full_like(i0, np.nan)
        for side in ((e < 0.0) & ~zero, (e > 0.0) & ~zero):
            pts = np.flatnonzero(side & usable)
            if pts.size < GAP_MIN_POINTS:
                continue
            # From the window's edge toward 0 V: ascending energies on the negative side, descending
            # on the positive side; d ln|I0| = F d ln|V| on both.  The trapezoid is taken against
            # ln|V|, not V: it is then exact wherever the current is a power law of V between two
            # points, which it nearly is near 0 V, where F / V against V overestimated the change
            # of ln|I0| by 6 % per step on a 25 mV axis (the current doubles between the last points).
            run = pts if e[pts[0]] < 0.0 else pts[::-1]
            v = e[run]
            ln_rec = cumulative_trapezoid(f[run], np.log(np.abs(v)), initial=0.0)
            ln_rec = ln_rec - np.mean(ln_rec - np.log(np.abs(i0[run])))
            rec = np.exp(ln_rec)
            rj[run] = rec
            gj[run] = np.abs(i0[run]) / rec
        g[order, j] = gj
        rebuilt[order, j] = rj
        kept[j] = int(np.sum(gj == 1.0))
    return g, kept, rebuilt


def weight_curve(iv0, g_raw, usable, n_bins=N_BINS):
    """
    The soft edge's curve, measured on the set of sweeps the step ran on: over every usable point
    (`usable` (npts, N): a finite ratio and a non-zero current, the point at 0 V out) of every
    column, g_raw - 1 is binned by |I0| (`iv0`, the current with its value at 0 V subtracted) in
    n_bins log-spaced bins from the 1st percentile of the current to its maximum; the rms of
    g_raw - 1 per bin (a bin with fewer than BIN_MIN_POINTS points has none) is the measured
    spread of g there.  The largest rms over the upper half of the bins is the jitter, the spacing
    noise the step removes; a bin's excess over it, excess^2 = rms^2 - jitter^2, is the step's own
    error at that current, and w = jitter^2 / (jitter^2 + excess^2) (= min(1, jitter^2 / rms^2))
    the share of the measured g_raw - 1 that is signal.  Bins without an rms are interpolated in
    log current (flat beyond the measured ones), and w is made non-decreasing with the current
    (the running minimum from the high-current end: the share of signal does not fall as the
    current rises).  Returns dict(centres, rms, w, jitter, edges), the currents in the units of
    `iv0`; None when the curve cannot be measured (no usable point, no bin of the upper half with
    an rms, or a jitter of 0): the caller then applies no soft edge.
    """
    iv0 = np.asarray(iv0, dtype=np.float64)
    g_raw = np.asarray(g_raw, dtype=np.float64)
    usable = np.asarray(usable, dtype=bool)
    cur = np.abs(iv0[usable])
    dev = g_raw[usable] - 1.0
    ok = (cur > 0) & np.isfinite(cur) & np.isfinite(dev)
    cur, dev = cur[ok], dev[ok]
    if cur.size == 0:
        return None
    lo, hi = float(np.percentile(cur, 1)), float(cur.max())
    if not (hi > lo > 0.0):
        return None
    edges = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
    idx = np.clip(np.digitize(cur, edges) - 1, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    rms = np.array([np.sqrt(np.mean(dev[idx == i] ** 2)) if counts[i] >= BIN_MIN_POINTS else np.nan
                    for i in range(n_bins)])
    centres = np.sqrt(edges[:-1] * edges[1:])
    upper = rms[n_bins // 2:]
    if not np.isfinite(upper).any():
        return None
    jitter = float(np.nanmax(upper))                      # the plateau's level: the jitter the step removes (its top,
    if not jitter > 0.0:                                  #  so the plateau's own scatter is not read as excess)
        return None
    excess2 = np.clip(rms ** 2 - jitter ** 2, 0.0, None)
    w = jitter ** 2 / (jitter ** 2 + excess2)             # = min(1, jitter^2 / rms^2)
    good = np.isfinite(w)
    w = np.interp(np.log(centres), np.log(centres[good]), w[good])      # bins too thin to measure: interpolated
    w = np.minimum.accumulate(w[::-1])[::-1]              # the share of signal does not fall as the current rises
    return dict(centres=centres, rms=rms, w=w, jitter=jitter, edges=edges)


def soft_edge(g_raw, iv0, curve):
    """
    The weighted correction: g = 1 + (g_raw - 1) w(|I0|), w read off the measured curve
    (weight_curve) by interpolation in log current, flat beyond its ends (a current below a
    thousandth of the first bin centre reads as that).  Returns (g, w), arrays like g_raw.
    """
    c, wv = curve['centres'], curve['w']
    cur = np.maximum(np.abs(np.asarray(iv0, dtype=np.float64)), c[0] * 1e-3)
    w = np.interp(np.log(cur), np.log(c), wv, left=wv[0], right=wv[-1])
    return 1.0 + (np.asarray(g_raw, dtype=np.float64) - 1.0) * w, w


def denoise(energy, lia, iv, vmod=None, V_bias=None, I_setpoint=None, gate=None, sweeps=None):
    """
    The denoise step on the acquired columns of one file: `lia` the dI/dV
    page and `iv` the current, both (npts, N) through the pipeline's
    filters, in the original energy order; vmod, V_bias, I_setpoint,
    gate and sweeps as calibration.calibrate takes them (the gate and the
    locks as the window sets them; this step's own regression uses them
    for its page and ratio only, never as the calibration; the gate's
    soft_edge flag is the 'soft edge' row).  Returns (lia / g, iv / g,
    step, note): the two pages divided by the per-point gap factor, the
    step's own pieces (DenoiseStep: cal, fits, Fcal, Isub0, I_rebuilt,
    g_raw, w, curve, g, kept) and the console line (the median and range
    of the per-sweep rms of g - 1, how many points per sweep stayed as
    recorded, and the soft edge: its jitter and the current at which w
    crosses 0.5, or that it was not measured, or off).  A ValueError of
    the regression (a column it cannot determine, a lock it cannot pin)
    is left to the caller, which then applies nothing.
    """
    settings = calibration.gate_settings(gate)
    energy = np.asarray(energy, dtype=np.float64)
    lia = np.asarray(lia, dtype=np.float64)
    iv = np.asarray(iv, dtype=np.float64)
    cal, fits = calibration.calibrate(energy, lia, iv, vmod, V_bias=V_bias, I_setpoint=I_setpoint, gate=settings,
                                      sweeps=sweeps)
    Isub0 = feenstra.iv_corrected(energy, iv, zero_sub=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        Fcal = feenstra.divide_by_iv(energy, cal, iv, 0.0, True, None, None)
    g_raw, kept, I_rebuilt = gap_factor(energy, Fcal, Isub0)
    order = np.argsort(energy)
    zero = np.empty(energy.size, dtype=bool)
    zero[order] = _zero_point(energy[order])
    usable = np.isfinite(Fcal) & (Isub0 != 0.0) & ~zero[:, None]
    curve = weight_curve(Isub0, g_raw, usable) if settings.soft_edge else None
    if curve is not None:
        g, w = soft_edge(g_raw, Isub0, curve)
    else:
        g, w = g_raw.copy(), np.ones_like(g_raw)
    rms = np.sqrt(np.mean((g - 1.0) ** 2, axis=0))
    npts = g.shape[0]
    if kept.size:
        if curve is not None:
            wv, c = curve['w'], curve['centres']
            if wv[0] < 0.5 <= wv[-1]:
                half = f"w = 0.5 near |I - I(0 V)| = {float(np.interp(0.5, wv, c)):.2g} (the current's units)"
            else:
                half = "w " + ("at least" if wv[0] >= 0.5 else "below") + " 0.5 in every bin"
            soft = f"soft edge: jitter {curve['jitter'] * 100:.1f} %, {half}"
        elif settings.soft_edge:
            soft = (f"soft edge not measured (no bin of the upper half of the current range has {BIN_MIN_POINTS} "
                    f"points): w = 1 everywhere")
        else:
            soft = "soft edge off"
        note = (f"denoise: per-sweep gap factor rms median {np.median(rms) * 100:.1f} % "
                f"(from {rms.min() * 100:.1f} to {rms.max() * 100:.1f} %); {kept.min()} to {kept.max()} of {npts} points "
                f"per sweep left as recorded (the point at 0 V, and any point without a finite ratio); {soft}")
    else:
        note = "denoise: no acquired columns"
    step = DenoiseStep(cal=cal, fits=fits, Fcal=Fcal, Isub0=Isub0, I_rebuilt=I_rebuilt, g_raw=g_raw, w=w, curve=curve,
                       g=g, kept=kept)
    return lia / g, iv / g, step, note


def setpoint_divided(energy, g, V_bias, I_setpoint):
    """
    The Vbias lock's pin currents for denoised columns: I_setpoint / g at
    the setpoint-bias sample (calibration.anchor_index), one per column
    (N,).  None when there is no setpoint current or no sample at the
    setpoint bias: the caller then passes I_setpoint as it is and the
    calibration refuses the lock as it does without denoising.
    """
    if I_setpoint is None or not np.isfinite(I_setpoint):
        return None
    try:
        k = calibration.anchor_index(energy, V_bias)
    except ValueError:
        return None
    return float(I_setpoint) / np.asarray(g, dtype=np.float64)[k, :]
