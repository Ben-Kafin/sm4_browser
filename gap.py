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
4. the band |I0| < GAP_BAND x the sweep's maximum |I0|, where the ratio is
   not conditioned: g = 1 there, those samples stay as recorded;
5. on each side of 0 V with at least GAP_MIN_POINTS samples outside the
   band: ln I_rec = the cumulative trapezoid of F / V from the window's
   edge toward 0 V, shifted so that the mean of ln|I0| - ln I_rec over
   that side is 0 (g has mean 1 on each side), g = |I0| / I_rec;
6. the page and the current of the sweep divided by g.

The dI/dV channel only; the per-sweep level (the setpoint scatter) and
the lock-in's own additive noise are not g and are left to the area
normalization and to averaging.  With the Vbias lock on, the setpoint
pin of a denoised forward sweep is I_setpoint / g at the setpoint sample
(setpoint_divided; the 0-lock pin and a reverse sweep's pin read the
denoised current themselves).  Decided 2026-11-02.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from sm4_Browser import calibration, feenstra


# The band around the current's zero where the Feenstra ratio is not conditioned: samples whose
# |I - I(0 V)| is below this fraction of the sweep's maximum keep g = 1 (left as recorded).
GAP_BAND = 0.10
# A side of 0 V needs at least this many samples outside the band for its integral.
GAP_MIN_POINTS = 3


def gap_factor(energy, ratio, iv0):
    """
    The per-point gap factor of every column: `ratio` the plain 0-sub
    Feenstra ratio (dI/dV)/(I0/V) of the calibrated page and `iv0` the
    current with its value at 0 V subtracted, both (npts, N) in the
    original energy order.  Per column: the band |I0| < GAP_BAND x
    max|I0| takes g = 1; on each side of 0 V (the sample at 0 V itself
    belongs to neither) with at least GAP_MIN_POINTS samples outside the
    band and with a finite ratio, ln I_rec is the cumulative trapezoid of
    ratio / V from the window's edge toward 0 V, shifted so that
    mean(ln|I0| - ln I_rec) = 0 over those samples, and g = |I0| / I_rec
    there.  Returns (g (npts, N), the number of samples left at g = 1 per
    column (N,)), g in the original energy order.
    """
    energy = np.asarray(energy, dtype=np.float64)
    ratio = np.asarray(ratio, dtype=np.float64)
    iv0 = np.asarray(iv0, dtype=np.float64)
    order = np.argsort(energy)
    e = energy[order]
    g = np.ones_like(iv0)
    kept = np.zeros(iv0.shape[1], dtype=int)
    for j in range(iv0.shape[1]):
        i0 = iv0[order, j]
        f = ratio[order, j]
        top = np.max(np.abs(i0)) if i0.size else 0.0
        usable = (np.abs(i0) >= GAP_BAND * top) & np.isfinite(f) & (top > 0)
        gj = np.ones_like(i0)
        for side in (e < 0.0, e > 0.0):
            pts = np.flatnonzero(side & usable)
            if pts.size < GAP_MIN_POINTS:
                continue
            # From the window's edge toward 0 V: ascending energies on the negative side, descending
            # on the positive side; d ln|I0| = F dV / V on both.
            run = pts if e[pts[0]] < 0.0 else pts[::-1]
            v = e[run]
            ln_rec = cumulative_trapezoid(f[run] / v, v, initial=0.0)
            ln_rec = ln_rec - np.mean(ln_rec - np.log(np.abs(i0[run])))
            gj[run] = np.abs(i0[run]) / np.exp(ln_rec)
        g[order, j] = gj
        kept[j] = int(np.sum(gj == 1.0))
    return g, kept


def denoise(energy, lia, iv, vmod=None, V_bias=None, I_setpoint=None, gate=None, sweeps=None):
    """
    The denoise step on the acquired columns of one file: `lia` the dI/dV
    page and `iv` the current, both (npts, N) through the pipeline's
    filters, in the original energy order; vmod, V_bias, I_setpoint,
    gate and sweeps as calibration.calibrate takes them (the gate and the
    locks as the window sets them; this step's own regression uses them
    and is otherwise discarded).  Returns (lia / g, iv / g, g, note):
    the two pages divided by the per-point gap factor, g itself (npts, N)
    and the console line (the median and range of the per-sweep rms of
    g - 1, and how many samples per sweep stayed as recorded).  A
    ValueError of the regression (a column it cannot determine, a lock it
    cannot pin) is left to the caller, which then applies nothing.
    """
    lia = np.asarray(lia, dtype=np.float64)
    iv = np.asarray(iv, dtype=np.float64)
    cal, _ = calibration.calibrate(energy, lia, iv, vmod, V_bias=V_bias, I_setpoint=I_setpoint, gate=gate,
                                   sweeps=sweeps)
    iv0 = feenstra.iv_corrected(energy, iv, zero_sub=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = feenstra.divide_by_iv(energy, cal, iv, 0.0, True, None, None)
    g, kept = gap_factor(energy, ratio, iv0)
    rms = np.sqrt(np.mean((g - 1.0) ** 2, axis=0))
    npts = g.shape[0]
    if kept.size:
        note = (f"denoise: per-sweep gap factor rms median {np.median(rms) * 100:.1f} % "
                f"(from {rms.min() * 100:.1f} to {rms.max() * 100:.1f} %); {kept.min()} to {kept.max()} of {npts} samples "
                f"per sweep left as recorded (|I - I(0 V)| below {GAP_BAND * 100:.0f} % of the sweep's maximum)")
    else:
        note = "denoise: no acquired columns"
    return lia / g, iv / g, g, note


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
