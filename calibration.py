"""
Calibration of the lock-in channels against the co-acquired I(V) current:
one gain for the file from the modulation amplitude (1 / V_mod for the
first harmonic, 4 / V_mod² for the second; the joint fit over the columns
when none is recorded, and printed beside it as a check either way), one
offset per sweep from the regression anchored at the setpoint (the point
at the setpoint bias, whose current is the setpoint current: the feedback
loop was opened there), and its double-derivative counterpart for the
IETS (second-harmonic) channel.  Nothing refers to 0 V.  A sweep
that does not pass the setpoint bias is refused (anchor_index); a pixel's
reverse sweeps, which start at the far end, take the a, b of its forward
sweeps (the `ref` columns).  calibrate and calibrate_iets return
(calibrated, fits), every fit entry carrying a, b, c, rms, X / Z, model
and residual.

For the dI/dV channel the regressed offsets are a diagnostic, not the
offset applied: the modulation starts with the sweep, so the first points
of every sweep are the lock-in's and the current channel's filters
settling (settling_points: the held stretch, left out of the regression);
the lock-in's offset is one number for the file, the 1 / rms²-weighted
mean of the sweeps' b, raised to the bound below which some calibrated
point is negative (file_offset); and a gap that moves during a sweep
(creep after the loop opened) multiplies the current and the lock-in by
the same factor s(V), the junction's transmission relative to the
setpoint gap, which the two channels fix together: ln s = ln(I / I_ref) −
∫ (a·LIA + b) / I dV from the sweep's start.  gap_factor reads it where
the channels disagree beyond the noise (the creep regions) and holds it
elsewhere; (a·LIA + b) / s is the dI/dV at the setpoint gap and I / s the
current there.  The IETS channel keeps its regression (s needs the first
harmonic).  Plain arrays in, columns = spectra; the pipeline
(sts.STS.process_arrays) calls these on the acquired columns only, after
the channel and the current have been through their filters and before
the Feenstra division.  Run as a file, the __main__
block draws one file's calibration (its inputs, the fit in the I(V) frame
with the corrected current, the corrected curve in the dI/dV frame, and
the gap factor with its regions), one sweep at a time.
"""

if __package__ in (None, ''):
    # Run as a file (Spyder F5): make the package importable from wherever its folder sits.
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from typing import NamedTuple
import numpy as np
from scipy.integrate import cumulative_trapezoid


class ColumnFit(NamedTuple):
    """
    One column's dI/dV regression, I(V) − I_set = a·X(V) + b·(V − V_bias)
    with X referenced at the anchor sample, the one at the setpoint bias
    (calibrate).  The arrays are in the column's original energy order
    (npts,).
    """
    a: float               # dI/dV scale, the file's: I units per V per LIA unit: 1/V when both are in A
    b: float               # dI/dV offset, this column's: I units per V
    c: float               # the setpoint current I_set, I units (not fitted: the anchor)
    X: np.ndarray          # ∫ LIA dV referenced at the anchor: X_raw − X_raw(V_bias) (LIA units·V)
    model: np.ndarray      # a·X + b·(V − V_bias) + c (I units)
    residual: np.ndarray   # I(V) − model (I units)
    rms: float             # rms of the residual (I units)
    a_joint: float = float('nan')   # the joint fit's gain over the file's columns (the check, or the gain without V_mod)
    gain: str = ''         # the gain's source: 'modulation' (1 / V_mod) or 'joint'
    held: int = 0          # the points after the anchor left out of the b regression (the settling stretch)


class IetsColumnFit(NamedTuple):
    """
    One column's IETS regression, I(V) − I_set = a·Z(V) + b·U²/2 + c·U with
    U = V − V_bias and Z referenced at the anchor sample, the one at the
    setpoint bias (calibrate_iets).  The arrays are in the column's
    original energy order (npts,).
    """
    a: float               # d2I/dV2 scale, the file's: I units per V² per second-harmonic unit
    b: float               # d2I/dV2 offset, this column's: I units per V²
    c: float               # the unknown dI/dV at the anchor (I units per V)
    d: float               # the setpoint current I_set, I units (not fitted: the anchor)
    Z: np.ndarray          # the double integral of the second-harmonic signal, both integrations referenced at the anchor (second-harmonic units·V²)
    model: np.ndarray      # a·Z + b·U²/2 + c·U + d (I units)
    residual: np.ndarray   # I(V) − model (I units)
    rms: float             # rms of the residual (I units)
    a_joint: float = float('nan')   # the joint fit's gain over the file's columns
    gain: str = ''         # the gain's source: 'modulation' (4 / V_mod²) or 'joint'


ANCHOR_TOL = 0.5     # the setpoint bias must lie within this fraction of the energy step of a sample

# The steps after the regression (dI/dV channel): the held stretch, the file's offset, the gap factor.
SETTLE_N_TAU = 5.0       # the held stretch at the start of a sweep: this many time constants of the slower of the
                         # lock-in filter and the current low-pass (a one-pole filter is within 1 % after 4.6)
REGION_THRESHOLD = 3.0   # a creep region: the channels' disagreement rate beyond this many times its own noise
REGION_MIN_RUN = 3       # ... for at least this many consecutive points of the sweep
I_FLOOR_FACTOR = 3.0     # |I| below this many times the file's current noise carries no gap information: s is held


class FileOffset(NamedTuple):
    """The lock-in's offset for a file (file_offset): the one b applied to every sweep of the dI/dV channel."""
    b: float          # the offset applied: b_mean, raised to `bound` when the floor is on and b_mean lies below it
    b_mean: float     # the 1 / rms²-weighted mean of the sweeps' regressed b (I units per V)
    bound: float      # −min(a·LIA) over the columns: below this b some calibrated point is negative
    floored: bool     # True when b is the bound rather than b_mean
    n: int            # the sweeps in the mean
    noise: float      # the file's current noise: the median rms residual of the regressions (I units)


class GapFactor(NamedTuple):
    """One sweep's gap factor s(V) (gap_factor); the arrays in the column's original energy order (npts,)."""
    s: np.ndarray         # the junction's transmission at each point relative to the setpoint gap (1 at the start)
    lns: np.ndarray       # ln s (−2κ·δz, κ the decay constant of the current with the gap)
    iv_corr: np.ndarray   # the current at the setpoint gap: (I − I(0 V)) / s + I(0 V) (I units)
    regions: tuple        # the creep regions ((V_lo, V_hi), ...) in V
    s_end: float          # s at the sweep's last point
    lns_max: float        # the largest |ln s| over the sweep
    noise: float          # the disagreement rate's noise in this sweep (1/V); NaN when it could not be estimated
    held: int             # the points held at the start (the settling stretch)
    n_region: int         # the points inside creep regions
    anchored: bool        # False when the sweep does not start at an end of the window: s ≡ 1 (or the given start)


def settling_points(lockin_cutoff_hz, lowpass_hz, dwell_s, n_tau=SETTLE_N_TAU):
    """
    The held stretch at the start of every sweep, in points.  The
    modulation starts with the sweep, so the lock-in output (low-pass
    filtered with time constant 1 / (2π f_c), f_c the file's
    RHK_Lockin0_FilterCutoffFrequency) and the current channel (its own
    low-pass, RHK_LowPassFilter1_CutoffFrequency) reach their values only
    after a few time constants: n_tau of the longer of the two, divided by
    the dwell per point, rounded up.  Returns (points, seconds); (0, 0.0)
    when no cutoff is recorded or the dwell is unknown or not positive.
    """
    taus = []
    for f in (lockin_cutoff_hz, lowpass_hz):
        try:
            f = float(f)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f) and f > 0:
            taus.append(1.0 / (2.0 * np.pi * f))
    try:
        dwell = float(dwell_s)
    except (TypeError, ValueError):
        dwell = 0.0
    if not taus or not np.isfinite(dwell) or dwell <= 0:
        return 0, 0.0
    seconds = float(n_tau) * max(taus)
    return int(np.ceil(seconds / dwell - 1e-9)), seconds


def held_mask(energy, V_bias, settle):
    """
    The held stretch as a mask over the points in the order of `energy`:
    True at the `settle` points after the anchor (the sample at V_bias,
    which must lie at an end of the window; _held_indices).  All False for
    settle 0, an anchor inside the window, or no anchor.
    """
    energy = np.asarray(energy, dtype=np.float64)
    mask = np.zeros(energy.size, dtype=bool)
    try:
        sort_idx = np.argsort(energy)
        i_a = anchor_index(energy[sort_idx], V_bias)
    except ValueError:
        return mask
    mask[sort_idx[_held_indices(energy.size, i_a, settle)]] = True
    return mask


def _held_indices(n, i_a, settle):
    """
    The sorted-axis indices of the held stretch: the `settle` points after
    the anchor of a sweep that leaves the setpoint from an end of the
    window (anchor index i_a, n points); none for settle 0 or an anchor
    inside the window (the sweep's start is then not known from the axis).
    """
    settle = int(max(0, settle))
    if settle == 0 or n < 2:
        return np.zeros(0, dtype=int)
    if i_a == 0:
        return np.arange(1, min(settle, n - 1) + 1)
    if i_a == n - 1:
        return np.arange(n - 2, max(n - 2 - settle, -1), -1)
    return np.zeros(0, dtype=int)


def anchor_index(energy, V_bias):
    """
    The anchor sample: the index in `energy` (any order) of the sample at
    the setpoint bias V_bias (RHK_Bias), within ANCHOR_TOL of the energy
    step.  The feedback loop was opened there at the setpoint current, so
    the current of that sample is known: I_setpoint.  Raises ValueError
    when V_bias is not recorded or no sample lies at it: a sweep that never
    passes the bias it was stabilised at has no known current and is not
    calibrated (the pipeline says so once on the console).
    """
    e = np.asarray(energy, dtype=np.float64)
    if V_bias is None or not np.isfinite(V_bias):
        raise ValueError("no setpoint bias (RHK_Bias) recorded; the calibration anchors at the setpoint.")
    if e.size == 0:
        raise ValueError("no energy samples.")
    i = int(np.argmin(np.abs(e - float(V_bias))))
    step = float(np.median(np.abs(np.diff(np.sort(e))))) if e.size > 1 else 0.0
    if abs(e[i] - float(V_bias)) > max(ANCHOR_TOL * step, 1e-12):
        raise ValueError(f"no sample at the setpoint bias {float(V_bias):+.4f} V (the sweep runs "
                         f"{e.min():+.4f} to {e.max():+.4f} V); the file is not calibrated.")
    return i


def _anchor(energy, V_bias, I_setpoint):
    """The sorted axis, its sort order, the anchor index on it, and U = V − V_bias; the setpoint checked."""
    energy = np.asarray(energy, dtype=np.float64)
    if I_setpoint is None or not np.isfinite(I_setpoint):
        raise ValueError("no setpoint current (RHK_Current) recorded; the calibration anchors at the setpoint.")
    sort_idx = np.argsort(energy)
    e_sorted = energy[sort_idx]
    i_a = anchor_index(e_sorted, V_bias)
    return e_sorted, sort_idx, i_a, e_sorted - e_sorted[i_a]


def _ref_column(ref, j, n):
    """The reference columns (lia, iv) of column j, or None when no reference holds a finite column there."""
    if ref is None or ref[0] is None or ref[1] is None:
        return None
    rc, ri = np.asarray(ref[0], dtype=np.float64), np.asarray(ref[1], dtype=np.float64)
    if rc.ndim != 2 or ri.ndim != 2 or rc.shape[0] != n or ri.shape[0] != n or j >= rc.shape[1] or j >= ri.shape[1]:
        return None
    if not (np.isfinite(rc[:, j]).all() and np.isfinite(ri[:, j]).all()):
        return None
    return rc[:, j], ri[:, j]


def gain_from_modulation(vmod, order=1):
    """
    The lock-in gain from the modulation amplitude V_mod (V, peak): for the
    first harmonic dI/dV = LIA / V_mod, so 1 / V_mod; for the second
    harmonic d2I/dV2 = 4 S / V_mod², so 4 / V_mod².  None when V_mod is not
    recorded (or not positive).
    """
    if vmod is None or not np.isfinite(vmod) or float(vmod) <= 0:
        return None
    return 1.0 / float(vmod) if order == 1 else 4.0 / float(vmod) ** 2


def _joint_gain(regressors, targets, nuisance):
    """
    One gain for every column with its own nuisance parameters: the least
    squares of Σ_j ‖ΔI_j − a X_j − B β_j‖² over a and every β_j, the β_j
    eliminated by projecting X_j and ΔI_j orthogonal to the columns of B
    (the same B for every column: U, or (U²/2, U)).  NaN when no column
    leaves anything to regress on.
    """
    B = np.asarray(nuisance, dtype=np.float64)
    if B.ndim == 1:
        B = B[:, None]
    G = np.linalg.pinv(B.T @ B) @ B.T
    num = den = 0.0
    for X, dI in zip(regressors, targets):
        Xp = X - B @ (G @ X)
        dIp = dI - B @ (G @ dI)
        num += float(np.dot(dIp, Xp))
        den += float(np.dot(Xp, Xp))
    return num / den if den > 0 else float('nan')


# vmod (the modulation amplitude, V) is the gain's source (gain_from_modulation); the pipeline
# (STS.process_arrays) passes the file's RHK_CH1Drive_Amplitude.
def calibrate(energy, curves, iv, vmod=None, V_bias=None, I_setpoint=None, ref=None, settle=0):
    """
    Calibrate dI/dV (LIA) curves against the co-acquired I(V) current
    channel: one gain for the file, one offset per curve, anchored at the
    setpoint.  For the dI/dV channel the pipeline applies the file's
    offset (file_offset) and the gap factor (gap_factor) after this
    regression; the per-curve b_j are then the diagnostic of each sweep.

    Physics
    -------
    The true dI/dV relates to the raw LIA signal by:

        true dI/dV(V) = a · LIA(V) + b

    The feedback loop was opened at the setpoint bias V_bias with the
    current at its setpoint I_set, so the sweep's sample at V_bias (its
    first, in a sweep that leaves the setpoint) has a known current.
    Integrating both sides from V_bias to V:

        I(V) − I_set = a · X(V) + b · (V − V_bias)

    where X(V) ≡ ∫ LIA(V') dV' from V_bias to V is the cumulative integral
    of the experimental LIA signal (cumulative trapezoidal rule, referenced
    at the anchor sample).

    The gain a is the lock-in's and does not change from spectrum to
    spectrum within a file: it is 1 / V_mod from the recorded modulation
    amplitude (gain_from_modulation; the lock-in's first harmonic is V_mod
    times dI/dV).  The offset b_j is fitted per spectrum, with ΔI = I −
    I_set and U = V − V_bias, as the least squares of

        S_j(b) = Σₙ [ΔI_j(Vₙ) − a·X_j(Vₙ) − b·Uₙ]²,   b_j = Σ (ΔI_j − a·X_j)·U / Σ U²

    Fitting a per spectrum as well is not done: on a window where the LIA
    is nearly flat, X is nearly proportional to U and a trades off against
    b (one spectrum's data fix only a·mean(LIA) + b), which spreads the
    gain over a wide range across the spectra of one file and flips its
    sign in some.  The joint fit of one a for every regressed column, each
    with its own b (_joint_gain), is computed as well: it is the gain when
    the file records no modulation amplitude, and otherwise the console
    comparison with 1 / V_mod (the pipeline prints both once per file).

    The calibrated dI/dV is then  a · LIA_j(V) + b_j.  The relation is
    exact in U, so the fit is unbiased whatever the bias window, crossing
    0 V or not, and in either sweep direction (the energies are sorted);
    nothing refers to 0 V.  A sweep that does not pass V_bias
    (anchor_index) or a file without the setpoint is refused (ValueError):
    it is not calibrated.  With `settle` > 0 the `settle` points after
    the anchor (the held stretch, settling_points: the filters settling
    after the modulation started with the sweep) are left out of the sums
    of S_j and of the rms; the anchor stays, and X and the model still
    cover every point.  The joint fit's sums are not changed by it.

    Parameters
    ----------
    energy : ndarray, shape (npts,)
        The energy axis of the columns (V), any order.
    curves : ndarray, shape (npts, N)
        The LIA data as the loader gives it (physical units, the page's
        RHK_Zunits; through the dI/dV filters when they are on) in the
        file's column ordering, i.e. *before* the spatial-reversal /
        transpose done for plotting.
    iv : ndarray or None
        Matching I(V) data with the same shape and column ordering, in the
        Current page's units (A): the recorded current, through the
        pipeline's filters when they are on.  None raises ValueError.
    vmod : float or None
        The modulation amplitude (V, peak; RHK_CH1Drive_Amplitude): the
        gain's source.  None: the joint fit is the gain.
    V_bias, I_setpoint : float
        The setpoint (RHK_Bias in V, RHK_Current in A): the anchor.
    ref : (lia_ref, iv_ref) or None
        Columns aligned with `curves` whose sweeps did start at the
        setpoint: a pixel's forward sweeps (their raw mean), for the views
        of its reverse sweeps or of all its sweeps (sts.STS.sweep_view).
        When given, the regression of column j runs on ref column j and
        its b_j is applied to curves[:, j]; a column whose reference is
        not finite (a pixel without a forward sweep) is regressed on
        itself.
    settle : int
        The held stretch in points (settling_points): the points after the
        anchor left out of the b regression.  0: every point counts.

    Returns
    -------
    calibrated : ndarray, same shape as *curves*, float64: a * LIA_j + b_j
        per column, in I units per V (A/V for pages in A)
    fits : list of ColumnFit, one per column (a the file's gain, b, c =
        I_set, X, model, residual, rms over the points in the regression,
        a_joint the joint fit's gain, gain 'modulation' or 'joint', held
        the points left out; the arrays in the original energy
        ordering; X, the model and the residual of the regressed columns,
        the reference's when one was used)

    With no modulation amplitude and every column's X proportional to U
    (for instance all-zero LIA columns) the joint gain is undefined and
    ValueError is raised; the pipeline passes the acquired columns only.
    """
    if iv is None:
        raise ValueError(
            "No I(V) current channel found in this SM4 file. "
            "Cannot perform linear-regression calibration."
        )
    e_sorted, sort_idx, i_a, U = _anchor(energy, V_bias, I_setpoint)
    energy = np.asarray(energy, dtype=np.float64)
    curves = np.asarray(curves, dtype=np.float64)
    iv_data = np.asarray(iv, dtype=np.float64)
    I_set = float(I_setpoint)
    n_curves = curves.shape[1]
    # The held stretch (the settling of the filters after the modulation started): out of the b sums.
    held = _held_indices(len(e_sorted), i_a, settle)
    use = np.ones(len(e_sorted), dtype=bool)
    use[held] = False
    Uu = U[use]
    UU = float(np.dot(Uu, Uu))

    # The regressed data per column (the reference's when given), on the sorted axis.
    Xs, dIs, srcs = [], [], []
    for j in range(n_curves):
        rc = _ref_column(ref, j, curves.shape[0])
        src_lia = curves[:, j] if rc is None else rc[0]
        src_iv = iv_data[:, j] if rc is None else rc[1]
        # X(V) = cumulative integral of LIA, referenced at the anchor sample (V_bias).
        X_raw = cumulative_trapezoid(src_lia[sort_idx], e_sorted, initial=0)
        Xs.append(X_raw - X_raw[i_a])
        # ΔI(V) = I(V) − I_set: the loop was opened at V_bias with the current at the setpoint.
        dIs.append(src_iv[sort_idx] - I_set)
        srcs.append(src_iv)

    # The gain: 1 / V_mod, or the joint fit over every column (its own offset each) without one.
    a_joint = _joint_gain(Xs, dIs, U)
    g = gain_from_modulation(vmod, 1)
    a, source = (g, 'modulation') if g is not None else (a_joint, 'joint')
    if not np.isfinite(a):
        raise ValueError("the calibration gain cannot be determined: no modulation amplitude recorded and "
                         "every column's integrated LIA is proportional to V − V_bias.")

    calibrated = np.empty_like(curves, dtype=np.float64)
    fits = []
    U_orig = np.empty_like(U)
    U_orig[sort_idx] = U
    for j in range(n_curves):
        # The offset: the least squares of ΔI − a·X against U over the points in the regression.
        b_j = float(np.dot((dIs[j] - a * Xs[j])[use], Uu) / UU) if UU > 0 else 0.0
        # Calibrated dI/dV in original energy ordering (the column's own LIA, whichever columns were regressed).
        calibrated[:, j] = a * curves[:, j] + b_j
        X_orig = np.empty_like(Xs[j])
        X_orig[sort_idx] = Xs[j]
        model = a * X_orig + b_j * U_orig + I_set
        residual = srcs[j] - model
        r_use = residual[sort_idx][use]
        fits.append(ColumnFit(float(a), b_j, I_set, X_orig, model, residual,
                              float(np.sqrt(np.mean(r_use * r_use))), float(a_joint), source, int(held.size)))

    return calibrated, fits


def file_offset(fits, lia, floor=True):
    """
    The lock-in's offset for a file: one b for every sweep of the dI/dV
    channel.  The lock-in's offset does not change from sweep to sweep,
    while a per-sweep b also absorbs a gap that moved during that sweep
    (gap_factor), so the file's b is the mean of the regressed b_j
    (calibrate's fits) weighted by 1 / rms_j², the inverse variance of
    each (the standard error of b_j is rms_j over the same factor for
    every sweep).  With `floor` on, a b_mean below −min(a·LIA) over the
    given columns `lia` (the filtered lock-in columns the offset is
    applied to) is raised to that bound: the lowest calibrated point of
    the file then sits at zero and none is negative; the result says when
    that acted.  Returns a FileOffset (b applied, b_mean, bound, floored,
    n sweeps, noise = the median rms residual, the file's current noise).
    Raises ValueError without a finite fit.
    """
    b = np.array([f.b for f in fits], dtype=np.float64)
    r = np.array([f.rms for f in fits], dtype=np.float64)
    ok = np.isfinite(b) & np.isfinite(r)
    if not ok.any():
        raise ValueError("no regressed sweep: the file's offset cannot be formed.")
    b, r = b[ok], r[ok]
    w = np.where(r > 0, 1.0 / np.maximum(r, 1e-300) ** 2, 0.0)
    b_mean = float(np.sum(w * b) / np.sum(w)) if np.sum(w) > 0 else float(np.mean(b))
    a = float(fits[0].a)
    lia = np.asarray(lia, dtype=np.float64)
    al = a * lia[np.isfinite(lia)] if lia.size else lia
    bound = float(-np.min(al)) if al.size else float('-inf')
    floored = bool(floor and np.isfinite(bound) and b_mean < bound)
    return FileOffset(bound if floored else b_mean, b_mean, bound, floored, int(b.size), float(np.median(r)))


def gap_factor(energy, g, iv, V_bias, settle=0, i_floor=0.0, threshold=REGION_THRESHOLD, min_run=REGION_MIN_RUN,
               start='setpoint', lns0=0.0, iv_zero=None):
    """
    The gap factor s(V) of every column: the junction's transmission at
    the moment the sweep passed V relative to the setpoint gap, read from
    the two channels together.

    Physics
    -------
    A gap that moves during a sweep (creep after the loop opened; the
    tip was not moved on purpose) multiplies the current and the lock-in
    by the same factor s, exp(−2κ·δz) to lowest order, the lock-in
    tracking the current's change with bias at the instantaneous gap.
    With g = a·LIA + b the calibrated dI/dV at the instantaneous gap and
    I₀ the current the sweep would have had at a still gap,

        I = s · I₀,    g = s · dI₀/dV,    I₀ = I_ref + ∫ g / s dV,

    so the log of the first line, differentiated and combined with the
    second, gives the gap's log-rate as the channels' disagreement,
    (dI/dV − g) / I, and integrating it from the sweep's start removes
    the derivative:

        ln s(V) = ln( I(V) / I_ref ) − ∫ g / I dV'    (from the start),

    I_ref the current at the start.  (a·LIA + b) / s is the dI/dV at the
    setpoint gap and I / s the current there, which is then the integral
    of that dI/dV by construction.  The Feenstra ratio is unchanged by s.

    The rule
    --------
    s changes only where the channels disagree beyond the noise.  The
    disagreement rate (dI/dV − g) / I is formed from the derivative of
    the (filtered) current for detection only; its noise is the
    median-based scale of the rate over this sweep's own points (right
    while the creep occupies less than half the sweep); a creep region is
    a run of at least `min_run` consecutive points beyond `threshold`
    times that noise, closed where the rate comes back inside.  Inside a
    region ln s accumulates by the closed form above (the trapezoid
    increment between consecutive points); outside it is constant.  The
    first `settle` points after the start (the held stretch,
    settling_points) are held at the starting value, the integral and
    the current ratio starting at the end of that stretch.  Points with
    |I| below `i_floor` (I_FLOOR_FACTOR times the file's current noise:
    the current says nothing about the gap there, as around 0 V) take no
    part in detection and s is held across them.  The current at 0 V
    (`iv_zero`, per column: the current channel's offset, what '0Vsub'
    subtracts) is removed before the ratio when given; the corrected
    current keeps it: (I − I(0 V)) / s + I(0 V).

    A sweep that starts at the setpoint (`start` 'setpoint': a forward
    sweep) starts at the anchor, which must be at an end of the window;
    its ln s starts at `lns0` (0).  A reverse sweep (`start` 'far') starts
    at the other end of the window, right after its forward sweep ended
    there: it is anchored to it, `lns0` its forward's ln s_end (per
    column, or one value), and its ln s continues from there.  A column
    whose anchor lies inside the window is not corrected (s ≡ exp(lns0),
    anchored False).

    Parameters: energy (npts,) any order; g (npts, N) the calibrated
    dI/dV a·LIA + b at the instantaneous gap; iv (npts, N) the recorded
    current through the pipeline's filters; V_bias the setpoint bias;
    settle, i_floor, threshold, min_run, start, lns0, iv_zero as above.
    Returns a list of GapFactor, one per column, the arrays in the
    original energy order.
    """
    energy = np.asarray(energy, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    iv = np.asarray(iv, dtype=np.float64)
    if g.ndim == 1:
        g = g[:, None]
    if iv.ndim == 1:
        iv = iv[:, None]
    if g.shape != iv.shape or g.shape[0] != energy.size:
        raise ValueError(f"gap_factor: dI/dV {g.shape}, current {iv.shape} and energy ({energy.size}) must agree.")
    if start not in ('setpoint', 'far'):
        raise ValueError(f"start must be 'setpoint' or 'far' (got '{start}').")
    n, N = g.shape
    sort_idx = np.argsort(energy)
    e_sorted = energy[sort_idx]
    i_a = anchor_index(e_sorted, V_bias)
    lns0 = np.broadcast_to(np.asarray(lns0, dtype=np.float64), (N,))
    zero = (np.zeros(N) if iv_zero is None
            else np.broadcast_to(np.asarray(iv_zero, dtype=np.float64), (N,)))
    anchored = n >= 2 and i_a in (0, n - 1)
    from_low = (i_a == 0) == (start == 'setpoint')      # the sweep runs up the sorted axis
    seq = np.arange(n) if from_low else np.arange(n - 1, -1, -1)
    i_floor = max(float(i_floor), 0.0)
    min_run = max(1, int(min_run))
    out = []
    for j in range(N):
        z = float(zero[j]) if np.isfinite(zero[j]) else 0.0
        gj = g[sort_idx, j]
        Ij = iv[sort_idx, j] - z
        l0 = float(lns0[j]) if np.isfinite(lns0[j]) else 0.0
        lns_s = np.full(n, l0)
        regions, n_region, noise, held = (), 0, float('nan'), 0
        if anchored:
            Vq, Iq, gq = e_sorted[seq], Ij[seq], gj[seq]
            held = int(min(max(0, int(settle)), n - 1))
            k0 = held
            valid = np.isfinite(Iq) & np.isfinite(gq) & (np.abs(Iq) >= i_floor) & (Iq != 0)
            if valid[k0]:
                valid &= np.sign(Iq) == np.sign(Iq[k0])
            # The increments of ln s between consecutive points (trapezoid), from the end of the held stretch.
            d = np.zeros(n)
            kk = np.arange(k0 + 1, n)
            if kk.size:
                both = valid[kk] & valid[kk - 1]
                with np.errstate(divide='ignore', invalid='ignore'):
                    inc = (np.log(Iq[kk] / Iq[kk - 1])
                           - 0.5 * (gq[kk] / Iq[kk] + gq[kk - 1] / Iq[kk - 1]) * (Vq[kk] - Vq[kk - 1]))
                d[kk] = np.where(both & np.isfinite(inc), inc, 0.0)
            # Detection: the disagreement rate of the two channels and its noise in this sweep.
            flagged = np.zeros(n, dtype=bool)
            with np.errstate(divide='ignore', invalid='ignore'):
                dIdV = np.gradient(Ij, e_sorted)[seq]
                r = (dIdV - gq) / Iq
            cand = valid & np.isfinite(r)
            cand[:k0 + 1] = False
            if cand.sum() >= 3:
                rc = r[cand]
                med = float(np.median(rc))
                noise = float(1.4826 * np.median(np.abs(rc - med)))
                if np.isfinite(noise) and noise > 0:
                    flagged = cand & (np.abs(r - med) > float(threshold) * noise)
            # The creep regions: runs of at least min_run flagged points.
            regs = []
            in_reg = np.zeros(n, dtype=bool)
            k = 0
            while k < n:
                if not flagged[k]:
                    k += 1
                    continue
                k1 = k
                while k1 + 1 < n and flagged[k1 + 1]:
                    k1 += 1
                if k1 - k + 1 >= min_run:
                    in_reg[k:k1 + 1] = True
                    regs.append((float(min(Vq[k], Vq[k1])), float(max(Vq[k], Vq[k1]))))
                k = k1 + 1
            lns_s[seq] = l0 + np.cumsum(np.where(in_reg, d, 0.0))
            regions, n_region = tuple(regs), int(in_reg.sum())
        lns = np.empty(n)
        lns[sort_idx] = lns_s
        s = np.exp(lns)
        iv_corr = (iv[:, j] - z) / s + z
        s_end = float(np.exp(lns_s[seq[-1]])) if anchored else float(np.exp(l0))
        out.append(GapFactor(s, lns, iv_corr, regions, s_end, float(np.max(np.abs(lns))), noise, held, n_region,
                             bool(anchored)))
    return out


# vmod: the gain's source, 4 / V_mod² for the second harmonic (gain_from_modulation).
def calibrate_iets(energy, curves, iv, vmod=None, V_bias=None, I_setpoint=None, ref=None):
    """
    Calibrate IETS (second-harmonic lock-in) curves against the co-acquired
    I(V) current channel, the double-derivative counterpart of calibrate:
    one gain for the file, the offset and the anchor slope per curve,
    anchored at the setpoint.

    The true d2I/dV2 relates to the raw second-harmonic signal S by

        true d2I/dV2(V) = a · S(V) + b

    Integrating twice from the anchor sample (the setpoint bias V_bias,
    where the current is I_set), with U = V − V_bias:

        I(V) − I_set = a · Z(V) + b · U² / 2 + c · U

    where Z(V) is the double cumulative trapezoidal integral of S, both
    integrations referenced at the anchor, and c absorbs the unknown dI/dV
    at the anchor.  The gain a is 4 / V_mod² from the recorded modulation
    amplitude (gain_from_modulation; the lock-in's second harmonic is
    V_mod² / 4 times d2I/dV2), or the joint fit of one a over every
    regressed column with its own (b, c) (_joint_gain) when none is
    recorded; the joint fit is computed either way for the console
    comparison.  For each spectrum j the least squares in (b, c) with a
    fixed gives the 2 by 2 normal equations

        [[Σ(U²/2)², ΣU²/2·U], [ΣU²/2·U, ΣU²]] · [b; c] = [Σ(ΔI − a·Z)·U²/2 ; Σ(ΔI − a·Z)·U]

    The calibrated curve is a · S_j(V) + b_j (in the Current page's units
    per V²), in the original energy ordering.  As in calibrate the relation
    is exact in U, so nothing refers to 0 V, and a sweep that does not pass
    V_bias is refused (anchor_index).

    Parameters: as calibrate (energy in V, any order; curves the
    second-harmonic page's data as the loader gives it, through the dI/dV
    filters when they are on; iv the recorded current through the
    pipeline's filters, None raises ValueError; vmod the modulation
    amplitude; V_bias, I_setpoint the setpoint; ref the reference columns
    of the sweeps that started at the setpoint).

    Returns
    -------
    calibrated : ndarray, same shape as *curves*, float64
    fits : list of IetsColumnFit, one per column (a, b, c, d = I_set, Z,
        model, residual, rms, a_joint, gain; the arrays in the original
        energy ordering)
    """
    if iv is None:
        raise ValueError(
            "No I(V) current channel found in this SM4 file. "
            "Cannot perform linear-regression calibration."
        )
    e_sorted, sort_idx, i_a, U = _anchor(energy, V_bias, I_setpoint)
    energy = np.asarray(energy, dtype=np.float64)
    curves = np.asarray(curves, dtype=np.float64)
    iv_data = np.asarray(iv, dtype=np.float64)
    I_set = float(I_setpoint)
    n_curves = curves.shape[1]
    Q = 0.5 * U * U
    B = np.column_stack([Q, U])
    M2 = B.T @ B

    Zs, dIs, srcs = [], [], []
    for j in range(n_curves):
        rc = _ref_column(ref, j, curves.shape[0])
        src_s = curves[:, j] if rc is None else rc[0]
        src_iv = iv_data[:, j] if rc is None else rc[1]
        # Y(V) = cumulative integral of S, referenced at the anchor; Z(V) = its integral, referenced too.
        Y_raw = cumulative_trapezoid(src_s[sort_idx], e_sorted, initial=0)
        Y = Y_raw - Y_raw[i_a]
        Z_raw = cumulative_trapezoid(Y, e_sorted, initial=0)
        Zs.append(Z_raw - Z_raw[i_a])
        dIs.append(src_iv[sort_idx] - I_set)
        srcs.append(src_iv)

    a_joint = _joint_gain(Zs, dIs, B)
    g = gain_from_modulation(vmod, 2)
    a, source = (g, 'modulation') if g is not None else (a_joint, 'joint')
    if not np.isfinite(a):
        raise ValueError("the calibration gain cannot be determined: no modulation amplitude recorded and "
                         "every column's double integral is in the span of U²/2 and U.")

    calibrated = np.empty_like(curves, dtype=np.float64)
    fits = []
    U_orig = np.empty_like(U)
    U_orig[sort_idx] = U
    for j in range(n_curves):
        rhs = B.T @ (dIs[j] - a * Zs[j])
        b_j, c_j = np.linalg.solve(M2, rhs)
        calibrated[:, j] = a * curves[:, j] + b_j
        Z_orig = np.empty_like(Zs[j])
        Z_orig[sort_idx] = Zs[j]
        model = a * Z_orig + b_j * 0.5 * U_orig * U_orig + c_j * U_orig + I_set
        residual = srcs[j] - model
        fits.append(IetsColumnFit(float(a), float(b_j), float(c_j), I_set, Z_orig, model, residual,
                                  float(np.sqrt(np.mean(residual * residual))), float(a_joint), source))

    return calibrated, fits


if __name__ == '__main__':
    # One file's dI/dV calibration by this method drawn, one sweep at a time (the file, the starting
    # sweep, and the current filters set here; '< sweep' / 'sweep >' step through the acquired columns
    # in file order, point, line, or map alike).  Row 1, the calibration's inputs: the raw LIA, its
    # integral X referenced at the anchor (the sample at the setpoint bias), the recorded current
    # (every acquired column faint, the shown sweep bold; the filtered current when the filters are
    # on).  Row 2: the fit in the I(V) frame with its parts a·X, b·(V − V_bias), c = I_set, the
    # residual, and the corrected current I / s (a the file's gain from V_mod, b the sweep's regressed
    # offset); the dI/dV frame with a·LIA + b_file (the file's offset) and its correction (a·LIA +
    # b_file) / s beside the numerical derivative of the (filtered) current; the gap factor s(V) with
    # its creep regions shaded and the held stretch hatched.  The file's lines (the gain, the offset,
    # the held stretch, the regions and the negative columns) and every column's parameters printed
    # once; the shown sweep's on every step.
    from sm4_Browser import qtplot as plt
    from sm4_Browser import sm4file, widgets, filters
    from sm4_Browser.feenstra import iv_zero_offset
    path = r'experiment_dir/file.sm4'
    column = 0        # the starting sweep: an index among the acquired columns (0 = the first)
    hampel = None     # the current filters as the windows set them (None = off), e.g. {'window': 4, 'n_sigma': 3}
    sgolay = None     # e.g. {'window_length': 6, 'polyorder': 3}

    f = sm4file.Sm4File(path)
    cols = np.flatnonzero(f.acquired)
    if cols.size == 0:
        raise ValueError(f"{f.name}: no acquired spectra.")
    if f.iv is None:
        raise ValueError(f"{f.name}: no 'Current' page; nothing to calibrate against.")
    if not 0 <= column < cols.size:
        raise ValueError(f"column must be 0..{cols.size - 1} (the acquired columns); got {column}.")
    V = np.asarray(f.energy, dtype=np.float64)
    lia = f.lia[:, cols]
    iv_raw = np.asarray(f.iv[:, cols], dtype=np.float64)
    filtered = hampel is not None or sgolay is not None
    iv = filters.filter_lia(iv_raw, hampel, sgolay)   # the recorded current through the filters (a copy when off)
    # The held stretch from the file's filters and dwell; the regression without it.
    settle, settle_s = settling_points(getattr(f.spectra, 'lockin_cutoff_hz', None),
                                       getattr(f.spectra, 'lowpass_hz', None), f.dwell_s)
    cal, fits = calibrate(V, lia, iv, f.vmod, V_bias=f.V_bias, I_setpoint=f.I_setpoint, ref=(f.lia_ref, f.iv_ref),
                          settle=settle)
    held = held_mask(V, f.V_bias, settle)
    # The file's offset (the floor on), the gap factor of every sweep, the corrected channel and current.
    off = file_offset(fits, lia[~held, :], floor=True)
    g = fits[0].a * lia + off.b
    gfs = gap_factor(V, g, iv, f.V_bias, settle=settle, i_floor=I_FLOOR_FACTOR * off.noise,
                     iv_zero=iv_zero_offset(V, iv))
    s_all = np.column_stack([gf.s for gf in gfs])
    corrected = g / s_all
    iv_corr = np.column_stack([gf.iv_corr for gf in gfs])
    uL = f.units_lia or 'no units'                    # the LIA page's RHK_Zunits (the loader's data are in it)
    uI = f.units_iv or 'no units'                     # the Current page's RHK_Zunits
    ua = "1/V" if uI == uL else f"{uI}/({uL}·V)"      # the units of the stretch a
    pA = 1e12
    n_sweeps = int(cols.size)
    order = np.argsort(V)
    Vs = V[order]
    i0 = anchor_index(Vs, f.V_bias)                   # the anchor: the sample at the setpoint bias
    Us = Vs - Vs[i0]
    print(f"{f.name}: {n_sweeps} acquired column(s), bias {V.min():+.3f} to {V.max():+.3f} V"
          f"; anchor at the setpoint: V_bias = {Vs[i0]:+.4f} V, I_setpoint = {f.I_setpoint * pA:.3f} pA"
          + (f"; current filters Hampel {hampel}, SG {sgolay}" if filtered else "; current filters off"))
    print(f"gain a = {fits[0].a:.4g} {ua} " + ("from the modulation amplitude (1/V_mod, V_mod = "
                                                f"{f.vmod * 1e3:.3g} mV)" if fits[0].gain == 'modulation'
                                                else "from the joint fit over the columns (no modulation amplitude recorded)")
          + f"; the joint fit gives {fits[0].a_joint:.4g} {ua}; the offset b per sweep")
    print(f"offset b = {sm4file.si_text(off.b, uI + '/V')} for the file: the 1/rms²-weighted mean of the "
          f"{off.n} regressed b is {sm4file.si_text(off.b_mean, uI + '/V')}, the bound −min(a·LIA) "
          f"{sm4file.si_text(off.bound, uI + '/V')}" + (" (the floor acted)" if off.floored else "")
          + f"; current noise {sm4file.si_text(off.noise, uI)}")
    print(f"held stretch {settle} point(s) ({settle_s * 1e3:.3g} ms, "
          f"{SETTLE_N_TAU:g} time constants of the slower filter; lock-in cutoff "
          f"{getattr(f.spectra, 'lockin_cutoff_hz', None)}, low-pass {getattr(f.spectra, 'lowpass_hz', None)}, "
          f"dwell {f.dwell_s})")
    n_reg = sum(1 for gf in gfs if gf.regions)
    n_pts = sum(gf.n_region for gf in gfs)
    n_neg = int(np.sum(np.any(corrected[~held, :] < 0, axis=0)))
    print(f"gap correction: creep regions in {n_reg} of {n_sweeps} sweeps ({100.0 * n_pts / corrected[~held, :].size:.1f} % "
          f"of the points); {n_neg} of {n_sweeps} corrected columns dip below zero outside the held stretch")
    for j, cf, gf in zip(cols, fits, gfs):
        regs = ", ".join(f"{lo:+.3f} to {hi:+.3f} V" for lo, hi in gf.regions) or "none"
        print(f"column {j}: b = {sm4file.si_text(cf.b, uI + '/V')}, "
              f"I_set = {sm4file.si_text(cf.c, uI)}, rms residual = {sm4file.si_text(cf.rms, uI)}; "
              f"creep regions {regs}; s_end = {gf.s_end:.4f}, largest |ln s| = {gf.lns_max:.4f}"
              + ("" if gf.anchored else "; not anchored (the sweep does not start at an end of the window)"))

    X_all = np.column_stack([c.X for c in fits])
    faint = dict(color='0.6', lw=0.5, alpha=0.5)
    bold = dict(color='C0', lw=1.2)
    to_pA = (lambda y: y * pA, lambda y: y / pA)      # the right-hand axes in pA (a Current page in A)

    held_span = (V[held].min(), V[held].max()) if held.any() else None

    DESIGN = (17, 9.5)        # the demo window's design size (inches)
    fig = plt.figure(figsize=DESIGN)
    # Fixed margins (a strip at the bottom for the sweep buttons), so every rebuild lands identically.
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.12, top=0.92, wspace=0.28, hspace=0.32)
    state = {'k': int(column), 'axes': []}

    def draw(k):
        """The six panels and the colour bar rebuilt for sweep k; the title and the sweep text updated."""
        for ax in state['axes']:
            ax.remove()
        state['axes'] = []
        cf = fits[k]
        gf = gfs[k]
        lia_k = lia[:, k]
        iv_k = iv[:, k]
        print(f"column {cols[k]}: a = {cf.a:.4g} ({cf.gain}), b = {cf.b:.4g} (file's b {off.b:.4g}), "
              f"s_end = {gf.s_end:.4f}, largest |ln s| = {gf.lns_max:.4f}")

        axes = fig.subplots(2, 3)
        state['axes'].extend(axes.flat)
        fig.suptitle(f"{f.name}: calibration of column {cols[k]} anchored at the setpoint  (bias {V.min():+.3f} to "
                     f"{V.max():+.3f} V, V_bias = {Vs[i0]:+.3f} V" + (", current filtered" if filtered else "") + ")")

        # Row 1: the inputs of the calibration.
        ax = axes[0, 0]
        ax.plot(V, lia, **faint)
        ax.plot(V, lia_k, **bold, label=f"column {cols[k]}")
        ax.set_ylabel(f"raw dI/dV: the lock-in output ({uL})")
        ax.set_title("raw dI/dV: LIA(V)")
        ax.legend(fontsize=8)

        ax = axes[0, 1]
        ax.plot(V, X_all, **faint)
        ax.plot(V, cf.X, **bold)
        ax.axvline(Vs[i0], color='0.5', lw=0.8, ls=':')
        ax.set_ylabel(f"X = ∫ LIA dV from V_bias ({uL}·V)")
        ax.set_title(f"the regressor X(V) = ∫ LIA dV from V_bias = {Vs[i0]:+.3f} V (0 there by construction; "
                     "the current it builds is in the panel below)")

        ax = axes[0, 2]
        ax.plot(V, iv_raw, **faint)
        if filtered:
            ax.plot(V, iv_raw[:, k], color='0.4', lw=0.6, alpha=0.6, label="recorded")
        ax.plot(V, iv_k, **bold, label="filtered" if filtered else "recorded")
        ax.set_ylabel(f"I(V) ({uI})")
        ax.set_title("recorded current I(V)" + (" and its filtered version" if filtered else ""))
        ax.legend(fontsize=8)
        if uI == 'A':
            ax.secondary_yaxis('right', functions=to_pA).set_ylabel("pA")

        # Row 2: the calibration.
        ax = axes[1, 0]
        if filtered:
            ax.plot(V, iv_raw[:, k], '.', ms=2, color='0.75', label="I(V) recorded")
        ax.plot(V, iv_k, '.', ms=2, color='0.3', label="I(V)" + (" filtered" if filtered else ""))
        ax.plot(V, cf.model, color='C3', lw=1.2, label="a·X + b·(V − V_bias) + I_set")
        ax.plot(V, cf.a * cf.X, '--', color='C0', lw=1, label="a·X")
        ax.plot(V, cf.b * (V - Vs[i0]), ':', color='C2', lw=1, label="b·(V − V_bias)")
        ax.axhline(cf.c, ls='-.', color='C1', lw=1, label="c = I_set")
        ax.plot(V, iv_corr[:, k], color='C4', lw=1.0, label="I / s: the current at the setpoint gap")
        ax.axvline(Vs[i0], color='0.5', lw=0.8, ls=':')
        ax.set_ylabel(f"I ({uI})")
        ax.set_title("the fit in the I(V) frame: I − I_set = a·X + b·(V − V_bias); the corrected current")
        ax.legend(loc='upper left', fontsize=8)
        ax.text(0.02, 0.03, f"a = {cf.a:.4g} {ua}\nb = {sm4file.si_text(cf.b, uI + '/V')}\n"
                            f"I_set = {sm4file.si_text(cf.c, uI)}\nrms = {sm4file.si_text(cf.rms, uI, 3)}",
                transform=ax.transAxes, fontsize=8, va='bottom', bbox=dict(boxstyle='round', fc='w', alpha=0.8))
        ins = ax.inset_axes([0.6, 0.08, 0.38, 0.28])
        ins.plot(V, cf.residual, color='0.3', lw=0.6)
        ins.axhline(0, color='k', lw=0.5)
        ins.set_title(f"residual I − model ({uI})", fontsize=7)
        ins.tick_params(labelsize=6)

        ax = axes[1, 1]
        didv_num = np.empty_like(V)
        didv_num[order] = np.gradient(iv_k[order], Vs)
        ax.plot(V, didv_num, color='0.6', lw=0.6, alpha=0.7,
                label="numerical dI/dV of the " + ("filtered" if filtered else "recorded") + " I(V)")
        ax.plot(V, g[:, k], '--', color='C3', lw=1.0, label="a·LIA + b_file (the file's offset)")
        ax.plot(V, corrected[:, k], color='C3', lw=1.3, label="(a·LIA + b_file) / s: dI/dV at the setpoint gap")
        ax.axhline(0, color='k', lw=0.5)
        ax.set_ylabel(f"dI/dV ({uI}/V)")
        ax.set_title("the dI/dV frame: a scales the LIA, b_file shifts it, s undoes the gap's motion")
        ax.legend(fontsize=8)
        if uI == 'A':
            ax.secondary_yaxis('right', functions=to_pA).set_ylabel("pA/V")

        ax = axes[1, 2]
        for lo, hi in gf.regions:
            ax.axvspan(lo, hi, color='C1', alpha=0.25, lw=0)
        if held_span is not None:
            ax.axvspan(held_span[0], held_span[1], facecolor='none', edgecolor='0.4', hatch='//', lw=0)
        ax.plot(V, gf.s, color='C0', lw=1.3, label="s(V)")
        ax.axhline(1, color='k', lw=0.5)
        ax.axvline(Vs[i0], color='0.5', lw=0.8, ls=':')
        ax.set_ylabel("s = exp(ln(I / I_ref) − ∫ (a·LIA + b_file) / I dV)")
        ax.set_title("the gap factor: creep regions shaded, the held stretch hatched")
        ax.legend(fontsize=8, loc='upper right')
        ax.text(0.02, 0.03, f"s_end = {gf.s_end:.4f}\nlargest |ln s| = {gf.lns_max:.4f}\n"
                            f"regions: {len(gf.regions)} ({gf.n_region} points)\n"
                            f"rate noise = {gf.noise:.3g} 1/V" + ("" if gf.anchored else "\nnot anchored"),
                transform=ax.transAxes, fontsize=8, va='bottom', bbox=dict(boxstyle='round', fc='w', alpha=0.8))

        for ax in list(axes[0]) + list(axes[1]):
            ax.set_xlabel("Energy (eV)")
        sweep_text.set_text(f"sweep {k + 1} of {n_sweeps} (file column {cols[k]})")
        fig.canvas.draw_idle()

    def on_step(d):
        """'< sweep' / 'sweep >': the previous / next sweep; the ends are not passed (a console line)."""
        new = state['k'] + int(d)
        if new < 0 or new >= n_sweeps:
            print(f"sweep {new + 1}: the file has sweeps 1..{n_sweeps}.")
            return
        state['k'] = new
        draw(new)

    # The sweep buttons bottom centre, the sweep text beside them.
    bw, bh = widgets.size_of(fig, '< sweep')
    steps = widgets.StepButtons(fig, '< sweep', 'sweep >', 0.5 - bw - 0.002, 0.015)
    steps.on_change(on_step)
    sweep_text = fig.text(0.5 + bw + 0.012, 0.015 + bh / 2, '', va='center', fontsize=9)
    draw(state['k'])
    plt.show()
