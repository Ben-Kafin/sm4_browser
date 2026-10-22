"""
Calibration of the lock-in channels against the co-acquired I(V) current,
one least squares per sweep with nothing locked, and the gate on the
per-sweep gains and offsets.  The true dI/dV relates to the lock-in
output by a·LIA + b, so the recorded current is

    I(V) = a · X(V) + b · V + c,     X(V) = ∫ LIA dV

the constant c the constant of integration: it carries where the
integral starts and the current's own level, so no sample is privileged
and no current (the setpoint's or any recorded one) is subtracted.  For
each sweep a, b and c are the least squares of that relation over every
point of the sweep (the normal equations of the sum of squares in a, b,
c); the calibrated dI/dV is a · LIA + b.  The gain the modulation
amplitude would give, 1 / V_mod, is carried beside the fitted a as the
check and is not used.

The gate (gate_sweeps; GateSettings, the window's 'gate' row with its
'H' and 'SG' boxes): one sweep with a disturbed current must not break
the calibration.  Along the acquisition order, a sweep's a is compared
with the mean a of the two sweeps touching it; a departure beyond
GATE_FACTOR times the robust scale of those departures over the file,
confirmed against the touching sweeps that passed, marks the sweep, and
it takes the trend's a with its b and c refitted.  The same test on b
then marks sweeps that take the trend's pair.  The trend is the
sequence of the per-sweep values through the package's Hampel filter
(filters.hampel, window and n_sigma from the 'H' box) and then a
Savitzky-Golay smoothing (length and order from the 'SG' box): where a
sweep's regression should sit among the sweeps around it.  The gate
needs as many sweeps as the Savitzky-Golay length and runs on dI/dV
only.

The IETS (second-harmonic) channel is the double-derivative
counterpart: I(V) = a · Z(V) + b · V² / 2 + c · V + d with Z the double
integral of the second-harmonic signal, four parameters per sweep, the
calibrated d2I/dV2 a · S + b, no gate.  calibrate and calibrate_iets
return (calibrated, fits), every fit entry carrying the parameters,
their standard errors, X / Z, the model and the residual; the dI/dV
entries also the sweep's own pair beside the applied one, the stage
that replaced it, the trend's pair and the Gate record.  Plain arrays
in, columns = spectra; the pipeline (sts.STS.process_arrays) calls these
on the acquired columns only, after the channel and the current have
been through their filters and before the Feenstra division.  Run as a
file, the __main__ block draws one file's calibration (its inputs, the
fit in the I(V) frame, the calibrated curve in the dI/dV frame, and the
cost surface), one sweep at a time.
"""

if __package__ in (None, ''):
    # Run as a file (Spyder F5): make the package importable from wherever its folder sits.
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from typing import NamedTuple
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import savgol_filter
from sm4_Browser import filters


# The gate's tolerance: this many times the robust scale (1.4826 × MAD) of the departures of the
# per-sweep values from the mean of their two touching sweeps.
GATE_FACTOR = 2.0
# The trend's defaults, the window's 'H' and 'SG' boxes: the Hampel filter's window (sweeps on
# each side) and n_sigma, the Savitzky-Golay length (sweeps, odd) and polynomial order.
GATE_HAMPEL = (3, 3.0)
GATE_SGOLAY = (9, 2)


class GateSettings(NamedTuple):
    """The gate as a window sets it: on / off, the Hampel (window, n_sigma), the Savitzky-Golay (length, order)."""
    on: bool = True
    hampel: tuple = GATE_HAMPEL
    sgolay: tuple = GATE_SGOLAY

    @property
    def text(self):
        """The short form shown beside the toggle: 'H 3,3 SG 9,2'."""
        return f"H {self.hampel[0]:g},{self.hampel[1]:g} SG {self.sgolay[0]:g},{self.sgolay[1]:g}"


GATE_DEFAULT = GateSettings()


def gate_settings(gate):
    """
    The GateSettings of `gate`: None (the defaults), a GateSettings, a dict
    with 'on', 'hampel', 'sgolay' (missing keys the defaults), or a bool
    (on / off at the defaults).  Validated: the Hampel window a whole number
    of at least 1 and n_sigma positive; the Savitzky-Golay length an odd
    whole number greater than the order, the order a whole number of at
    least 0.  ValueError otherwise, naming the box.
    """
    if gate is None:
        return GATE_DEFAULT
    if isinstance(gate, GateSettings):
        g = gate
    elif isinstance(gate, (bool, np.bool_)):
        g = GateSettings(on=bool(gate))
    elif isinstance(gate, dict):
        g = GateSettings(on=bool(gate.get('on', True)), hampel=tuple(gate.get('hampel', GATE_HAMPEL)),
                         sgolay=tuple(gate.get('sgolay', GATE_SGOLAY)))
    else:
        raise ValueError(f"gate settings must be None, a bool, a dict or GateSettings; got {gate!r}.")
    try:
        hw, hs = g.hampel
        sl, so = g.sgolay
        hw, hs, sl, so = float(hw), float(hs), float(sl), float(so)
    except (TypeError, ValueError):
        raise ValueError("gate settings: 'H' needs window,n_sigma and 'SG' needs length,order.")
    if hw != int(hw) or hw < 1 or not np.isfinite(hs) or hs <= 0:
        raise ValueError(f"gate 'H' box: the window must be a whole number of at least 1 and n_sigma positive (got {hw:g},{hs:g}).")
    if sl != int(sl) or so != int(so) or so < 0 or sl <= so or int(sl) % 2 == 0:
        raise ValueError(f"gate 'SG' box: the length must be an odd whole number greater than the order, the order at least 0 (got {sl:g},{so:g}).")
    return GateSettings(bool(g.on), (int(hw), hs), (int(sl), int(so)))


class Gate(NamedTuple):
    """The gate's record of one calibrate call: what ran and what it replaced (indices in the call's column order)."""
    settings: GateSettings
    ran: bool                  # False: the gate was off or the call had fewer columns than the Savitzky-Golay length
    note: str                  # why it did not run ('' when it ran)
    tol_a: float               # the tolerance on a (I units per V per LIA unit); NaN when it did not run
    tol_b: float               # the tolerance on b (I units per V)
    replaced_a: np.ndarray     # the columns replaced on the a stage (the trend's a, b and c refitted)
    replaced_b: np.ndarray     # the columns replaced on the b stage (the trend's pair)
    trend_a: np.ndarray        # the trend of a along the columns (the a stage's), NaN when it did not run
    trend_b: np.ndarray        # the trend of b (after the a stage)


_NO_GATE = Gate(GATE_DEFAULT, False, '', float('nan'), float('nan'), np.zeros(0, dtype=int), np.zeros(0, dtype=int),
                np.zeros(0), np.zeros(0))


class ColumnFit(NamedTuple):
    """
    One column's dI/dV regression, I(V) = a·X(V) + b·V + c, X the cumulative
    integral of the LIA from the lowest-energy sample (calibrate), and what
    the gate made of it.  The arrays are in the column's original energy
    order (npts,).
    """
    a: float               # dI/dV scale applied: I units per V per LIA unit: 1/V when both are in A
    b: float               # dI/dV offset applied: I units per V
    c: float               # the constant of integration of the applied pair's refit (I units)
    X: np.ndarray          # ∫ LIA dV from the lowest-energy sample (LIA units·V)
    model: np.ndarray      # a·X + b·V + c with the sweep's own least squares (I units)
    residual: np.ndarray   # I(V) − model (I units)
    rms: float             # rms of the residual (I units)
    se: tuple = (float('nan'),) * 3   # the standard errors of the sweep's own (a, b, c) from the residual
    a_mod: float = float('nan')       # the gain the modulation amplitude would give, 1 / V_mod (the check; NaN when not recorded)
    a_own: float = float('nan')       # the sweep's own least squares (a, b, c); equal to a, b, c when the gate left it
    b_own: float = float('nan')
    c_own: float = float('nan')
    replaced: str = ''                # '' (its own pair), 'a' (the a stage: the trend's a, b and c refitted), 'ab' (the b stage: the trend's pair)
    trend: tuple = (float('nan'),) * 2   # the trend's (a, b) at this column, NaN when the gate did not run
    gate: Gate = _NO_GATE             # the call's gate record (the same object in every column's entry)


class IetsColumnFit(NamedTuple):
    """
    One column's IETS regression, I(V) = a·Z(V) + b·V²/2 + c·V + d, Z the
    double cumulative integral of the second-harmonic signal from the
    lowest-energy sample (calibrate_iets).  The arrays are in the column's
    original energy order (npts,).
    """
    a: float               # d2I/dV2 scale, this column's: I units per V² per second-harmonic unit
    b: float               # d2I/dV2 offset, this column's: I units per V²
    c: float               # the dI/dV constant of integration (I units per V)
    d: float               # the current's constant of integration (I units)
    Z: np.ndarray          # the double integral of the second-harmonic signal from the lowest-energy sample (second-harmonic units·V²)
    model: np.ndarray      # a·Z + b·V²/2 + c·V + d (I units)
    residual: np.ndarray   # I(V) − model (I units)
    rms: float             # rms of the residual (I units)
    se: tuple = (float('nan'),) * 4   # the standard errors of (a, b, c, d)
    a_mod: float = float('nan')       # the gain the modulation amplitude would give, 4 / V_mod² (the check)


def gain_from_modulation(vmod, order=1):
    """
    The lock-in gain the modulation amplitude V_mod (V, peak) would give:
    for the first harmonic dI/dV = LIA / V_mod, so 1 / V_mod; for the
    second harmonic d2I/dV2 = 4 S / V_mod², so 4 / V_mod².  None when V_mod
    is not recorded (or not positive).  The check carried beside the fitted
    gain; the regression does not use it.
    """
    if vmod is None or not np.isfinite(vmod) or float(vmod) <= 0:
        return None
    return 1.0 / float(vmod) if order == 1 else 4.0 / float(vmod) ** 2


def _solve(A, y, what):
    """
    The least squares of y against the columns of A (npts, k), the columns
    scaled to unit norm for the solve (the integral of a current in A is
    1e-10 of V): (parameters, standard errors, residual).  Raises
    ValueError naming `what` when the columns are dependent (a column of
    zeros, or a regressor in the span of the others): the parameters are
    then not determined.
    """
    norms = np.sqrt(np.sum(A * A, axis=0))
    if not np.all(norms > 0):
        raise ValueError(f"{what}: a regressor is identically zero; its parameters are not determined.")
    As = A / norms
    sol, _, rank, _ = np.linalg.lstsq(As, y, rcond=None)
    if rank < A.shape[1]:
        raise ValueError(f"{what}: the regressors are dependent; the parameters are not determined.")
    residual = y - As @ sol
    dof = max(A.shape[0] - A.shape[1], 1)
    cov = float(residual @ residual) / dof * np.linalg.inv(As.T @ As)
    return sol / norms, np.sqrt(np.abs(np.diag(cov))) / norms, residual


def _touching_flags(x, factor):
    """
    The touching-sweep test on the sequence x (N,) in acquisition order:
    every value against the mean of the two values touching it (one at the
    ends); the tolerance `factor` times the robust scale (1.4826 × MAD about
    the median) of those departures; a value beyond it is re-tested against
    the touching values that passed, so a sweep that merely touches an
    outlier is not dragged along.  Returns (the flagged indices, the
    tolerance); nothing flagged when the scale is 0 or not finite.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    touching = [[i for i in (j - 1, j + 1) if 0 <= i < n] for j in range(n)]
    ref = np.array([np.mean(x[t]) for t in touching])
    d = x - ref
    ok = np.isfinite(d)
    if not ok.any():
        return np.zeros(0, dtype=int), float('nan')
    tol = float(factor) * 1.4826 * float(np.median(np.abs(d[ok] - np.median(d[ok]))))
    if not (np.isfinite(tol) and tol > 0):
        return np.zeros(0, dtype=int), tol
    first = set(np.flatnonzero(ok & (np.abs(d) > tol)).tolist())
    flagged = []
    for j in sorted(first):
        passed = [i for i in touching[j] if i not in first]
        if passed and abs(x[j] - np.mean(x[passed])) <= tol:
            continue                               # it fits the sweeps that passed: a false alarm
        flagged.append(j)
    return np.array(flagged, dtype=int), tol


def _trend(x, settings):
    """The trend of the sequence x along the sweeps: the Hampel filter, then the Savitzky-Golay smoothing (GateSettings)."""
    x = np.asarray(x, dtype=np.float64)
    xh = filters.hampel(x[:, None], window=settings.hampel[0], n_sigma=settings.hampel[1])[:, 0]
    return savgol_filter(xh, settings.sgolay[0], settings.sgolay[1], mode='interp')


def gate_sweeps(a, b, refit, gate=None, factor=GATE_FACTOR):
    """
    The gate on the per-sweep gains a and offsets b (N,) in acquisition
    order, `refit(j, a_j)` giving the (b, c) of sweep j refitted with its
    a held (calibrate's).  Stage 1: the touching-sweep test on a
    (_touching_flags); a flagged sweep takes the trend's a (_trend) and
    its b and c refitted.  Stage 2: the test on the b that came out of
    stage 1; a flagged sweep takes the trend's a and b, the trends drawn
    on the sequences after stage 1.  Returns (a, b, c-or-None per column,
    Gate): the applied values and the record.  With the gate off, or
    fewer sweeps than the Savitzky-Golay length, nothing is replaced and
    the record says why.
    """
    settings = gate_settings(gate)
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    n = a.size
    c = [None] * n
    if not settings.on:
        return a, b, c, Gate(settings, False, "the gate is off", float('nan'), float('nan'),
                             np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.full(n, np.nan), np.full(n, np.nan))
    if n < settings.sgolay[0]:
        return a, b, c, Gate(settings, False, f"{n} sweep(s), fewer than the Savitzky-Golay length {settings.sgolay[0]}",
                             float('nan'), float('nan'), np.zeros(0, dtype=int), np.zeros(0, dtype=int),
                             np.full(n, np.nan), np.full(n, np.nan))
    rep_a, tol_a = _touching_flags(a, factor)
    trend_a = _trend(a, settings)
    for j in rep_a:
        a[j] = trend_a[j]
        b[j], c[j] = refit(j, a[j])
    rep_b, tol_b = _touching_flags(b, factor)
    trend_a2 = _trend(a, settings)
    trend_b = _trend(b, settings)
    for j in rep_b:
        a[j], b[j] = trend_a2[j], trend_b[j]
        c[j] = None
    return a, b, c, Gate(settings, True, '', tol_a, tol_b, rep_a, rep_b, trend_a2, trend_b)


# vmod (the modulation amplitude, V) gives the check carried beside the fitted gain (gain_from_modulation);
# V_bias, I_setpoint and ref are accepted for the callers and not used: the regression locks nothing.
def calibrate(energy, curves, iv, vmod=None, V_bias=None, I_setpoint=None, ref=None, gate=None):
    """
    Calibrate dI/dV (LIA) curves against the co-acquired I(V) current
    channel: per sweep the least squares of a, b, c, nothing locked, then
    the gate.

    Physics
    -------
    The true dI/dV relates to the raw LIA signal by:

        true dI/dV(V) = a · LIA(V) + b

    Integrating both sides:

        I(V) = a · X(V) + b · V + c

    where X(V) is the cumulative integral of the experimental LIA signal
    (cumulative trapezoidal rule from the lowest-energy sample) and c the
    constant of integration.  The constant carries where the integral
    starts and the level of the current, so no sample is privileged, no
    current is subtracted, and the fit does not depend on where X starts
    (a shift of X by a constant is absorbed by c).  For each spectrum j the
    least squares of

        S_j(a, b, c) = Σₙ [I_j(Vₙ) − a·X_j(Vₙ) − b·Vₙ − c]²

    gives (a_j, b_j, c_j), the normal equations ∂S/∂a = ∂S/∂b = ∂S/∂c = 0,
    i.e. [[ΣX², ΣXV, ΣX], [ΣXV, ΣV², ΣV], [ΣX, ΣV, n]] · [a; b; c] =
    [ΣIX; ΣIV; ΣI].  The relation holds on any bias window, crossing 0 V
    or not, and in either sweep direction (the energies are sorted).  The
    gain from the modulation amplitude, 1 / V_mod, is carried in the fits
    (a_mod) as the check beside a_j; it is not used.

    The gate (gate_sweeps, the module docstring) then runs on the per-sweep
    a and b in the columns' order: a replaced sweep's calibrated curve uses
    the pair it took.  The calibrated dI/dV is a_j · LIA_j(V) + b_j with the
    applied pair.

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
        check's source.
    V_bias, I_setpoint, ref
        Accepted for the callers (the setpoint and a sweep view's
        forward-sweep reference) and not used: every sweep is regressed on
        its own data.
    gate : GateSettings, dict, bool or None
        The gate as the window sets it (gate_settings); None the defaults
        (on, Hampel 3,3, Savitzky-Golay 9,2).

    Returns
    -------
    calibrated : ndarray, same shape as *curves*, float64: a_j * LIA_j + b_j
        per column with the applied pair, in I units per V (A/V for pages
        in A)
    fits : list of ColumnFit, one per column (the applied a, b, c, X, model,
        residual, rms and se of the sweep's own fit, a_mod, the own pair,
        the stage that replaced it, the trend's pair, the Gate record; the
        arrays in the original energy ordering)

    A column whose regressors are dependent (an all-zero LIA column: X is
    then zero) raises ValueError naming it; the pipeline passes the
    acquired columns only.
    """
    if iv is None:
        raise ValueError(
            "No I(V) current channel found in this SM4 file. "
            "Cannot perform linear-regression calibration."
        )
    energy = np.asarray(energy, dtype=np.float64)
    curves = np.asarray(curves, dtype=np.float64)
    iv_data = np.asarray(iv, dtype=np.float64)
    if energy.size == 0:
        raise ValueError("no energy samples.")
    sort_idx = np.argsort(energy)
    e_sorted = energy[sort_idx]
    one = np.ones_like(e_sorted)
    g = gain_from_modulation(vmod, 1)
    a_mod = float('nan') if g is None else float(g)
    n_curves = curves.shape[1]
    Xs, Is, own, ses = [], [], [], []
    for j in range(n_curves):
        # X(V) = cumulative integral of the LIA from the lowest-energy sample.
        X = cumulative_trapezoid(curves[sort_idx, j], e_sorted, initial=0)
        I = iv_data[sort_idx, j]
        sol, se, _ = _solve(np.column_stack([X, e_sorted, one]), I, f"column {j}")
        Xs.append(X)
        Is.append(I)
        own.append(sol)
        ses.append(se)
    own = np.array(own) if n_curves else np.zeros((0, 3))

    def refit(j, a_j):
        """The (b, c) of column j with a held at a_j."""
        sol, _, _ = _solve(np.column_stack([e_sorted, one]), Is[j] - a_j * Xs[j], f"column {j}")
        return float(sol[0]), float(sol[1])

    a_app, b_app, c_app, record = gate_sweeps(own[:, 0], own[:, 1], refit, gate) if n_curves else \
        (np.zeros(0), np.zeros(0), [], _NO_GATE)
    calibrated = np.empty_like(curves, dtype=np.float64)
    fits = []
    for j in range(n_curves):
        a_j, b_j = float(a_app[j]), float(b_app[j])
        if c_app[j] is None:
            c_j = own[j, 2] if (a_j == own[j, 0] and b_j == own[j, 1]) else refit_c(Xs[j], Is[j], e_sorted, a_j, b_j)
        else:
            c_j = float(c_app[j])
        calibrated[:, j] = a_j * curves[:, j] + b_j
        X_orig = np.empty_like(Xs[j])
        X_orig[sort_idx] = Xs[j]
        model = own[j, 0] * X_orig + own[j, 1] * energy + own[j, 2]
        residual = iv_data[:, j] - model
        replaced = 'ab' if j in record.replaced_b else ('a' if j in record.replaced_a else '')
        trend = (float(record.trend_a[j]), float(record.trend_b[j])) if record.ran else (float('nan'),) * 2
        fits.append(ColumnFit(a_j, b_j, float(c_j), X_orig, model, residual,
                              float(np.sqrt(np.mean(residual * residual))), tuple(float(s) for s in ses[j]), a_mod,
                              float(own[j, 0]), float(own[j, 1]), float(own[j, 2]), replaced, trend, record))
    return calibrated, fits


def refit_c(X, I, V, a, b):
    """The level c of a sweep for a given pair (a, b): the mean of I − a·X − b·V over the sweep (sorted arrays)."""
    return float(np.mean(I - a * X - b * V))


def calibrate_iets(energy, curves, iv, vmod=None, V_bias=None, I_setpoint=None, ref=None, gate=None):
    """
    Calibrate IETS (second-harmonic lock-in) curves against the co-acquired
    I(V) current channel, the double-derivative counterpart of calibrate:
    per sweep the least squares of a, b, c, d, nothing locked, no gate.

    The true d2I/dV2 relates to the raw second-harmonic signal S by

        true d2I/dV2(V) = a · S(V) + b

    Integrating twice:

        I(V) = a · Z(V) + b · V² / 2 + c · V + d

    where Z(V) is the double cumulative trapezoidal integral of S from the
    lowest-energy sample, c the constant of the first integration (the
    dI/dV level) and d of the second (the current level).  For each
    spectrum j the least squares in (a, b, c, d) over every point of the
    sweep, the 4 by 4 normal equations on the regressors (Z, V²/2, V, 1).
    The calibrated curve is a_j · S_j(V) + b_j (in the Current page's units
    per V²), in the original energy ordering.  The gain the modulation
    amplitude would give, 4 / V_mod², is carried as the check (a_mod).

    Parameters: as calibrate (energy in V, any order; curves the
    second-harmonic page's data as the loader gives it, through the dI/dV
    filters when they are on; iv the recorded current through the
    pipeline's filters, None raises ValueError; vmod the modulation
    amplitude; V_bias, I_setpoint, ref and gate accepted and not used).

    Returns
    -------
    calibrated : ndarray, same shape as *curves*, float64
    fits : list of IetsColumnFit, one per column (a, b, c, d, Z, model,
        residual, rms, se, a_mod; the arrays in the original energy ordering)
    """
    if iv is None:
        raise ValueError(
            "No I(V) current channel found in this SM4 file. "
            "Cannot perform linear-regression calibration."
        )
    energy = np.asarray(energy, dtype=np.float64)
    curves = np.asarray(curves, dtype=np.float64)
    iv_data = np.asarray(iv, dtype=np.float64)
    if energy.size == 0:
        raise ValueError("no energy samples.")
    sort_idx = np.argsort(energy)
    e_sorted = energy[sort_idx]
    one = np.ones_like(e_sorted)
    g = gain_from_modulation(vmod, 2)
    a_mod = float('nan') if g is None else float(g)
    n_curves = curves.shape[1]
    calibrated = np.empty_like(curves, dtype=np.float64)
    fits = []
    for j in range(n_curves):
        # Y(V) = cumulative integral of S from the lowest-energy sample; Z(V) = its integral.
        Y = cumulative_trapezoid(curves[sort_idx, j], e_sorted, initial=0)
        Zs = cumulative_trapezoid(Y, e_sorted, initial=0)
        Is = iv_data[sort_idx, j]
        (a_j, b_j, c_j, d_j), se, _ = _solve(np.column_stack([Zs, 0.5 * e_sorted * e_sorted, e_sorted, one]), Is,
                                             f"column {j}")
        calibrated[:, j] = a_j * curves[:, j] + b_j
        Z_orig = np.empty_like(Zs)
        Z_orig[sort_idx] = Zs
        model = a_j * Z_orig + b_j * 0.5 * energy * energy + c_j * energy + d_j
        residual = iv_data[:, j] - model
        fits.append(IetsColumnFit(float(a_j), float(b_j), float(c_j), float(d_j), Z_orig, model, residual,
                                  float(np.sqrt(np.mean(residual * residual))), tuple(float(s) for s in se), a_mod))
    return calibrated, fits


def gate_from_record(rec):
    """The GateSettings a window record holds ('gate' on / off, 'gate_hampel' [window, n_sigma], 'gate_sgolay' [length, order]); a record without them (older) gives the defaults."""
    try:
        return gate_settings({'on': rec.get('gate', True), 'hampel': rec.get('gate_hampel', GATE_HAMPEL),
                              'sgolay': rec.get('gate_sgolay', GATE_SGOLAY)})
    except ValueError as exc:
        print(f"record: the gate settings are not usable ({exc}); the defaults are used.")
        return GATE_DEFAULT


def gate_record(settings):
    """The record keys of GateSettings: {'gate', 'gate_hampel', 'gate_sgolay'} (gate_from_record reads them)."""
    return {'gate': bool(settings.on), 'gate_hampel': [settings.hampel[0], settings.hampel[1]],
            'gate_sgolay': [settings.sgolay[0], settings.sgolay[1]]}


def gate_text(fits, cols=None):
    """
    One console phrase for a calibrate call's gate: not run and why, or
    the tolerances and the columns replaced on each stage (the file's
    column indices when `cols` maps the call's columns to them).
    """
    if not fits:
        return "no column"
    rec = fits[0].gate
    if not rec.ran:
        return f"gate not run ({rec.note})"
    name = (lambda k: int(cols[k])) if cols is not None else int
    ra = [name(k) for k in rec.replaced_a]
    rb = [name(k) for k in rec.replaced_b]
    return (f"gate {rec.settings.text}, factor {GATE_FACTOR:g}: tolerance ±{rec.tol_a:.3g} on a, "
            f"±{rec.tol_b * 1e12:.3g} pA/V on b; {len(ra)} sweep(s) took the trend's a" + (f" ({ra})" if ra else "")
            + f", {len(rb)} the trend's pair" + (f" ({rb})" if rb else ""))


if __name__ == '__main__':
    # One file's dI/dV calibration by this method drawn, one sweep at a time (the file, the starting
    # sweep, the current filters and the gate set here; '< sweep' / 'sweep >' step through the
    # acquired columns in file order, point, line, or map alike).  Row 1, the calibration's inputs:
    # the raw LIA, its integral X from the lowest-energy sample, the recorded current (every acquired
    # column faint, the shown sweep bold; the filtered current when the filters are on).  Row 2: the
    # fit in the I(V) frame with its parts a·X, b·V, c and the residual (a, b, c the sweep's own least
    # squares); the calibrated a·LIA + b in the browser's dI/dV frame, with the pair the gate applied
    # when it replaced the sweep, and the numerical derivative of the (filtered) current; the
    # least-squares cost S(a, b) around its minimum, c at its optimum for each (a, b), the trend's
    # pair marked.  Parameters of every column printed once; the shown sweep's on every step.
    from sm4_Browser import qtplot as plt
    from sm4_Browser import sm4file, widgets
    path = r'experiment_dir/file.sm4'
    column = 0        # the starting sweep: an index among the acquired columns (0 = the first)
    hampel = None     # the current filters as the windows set them (None = off), e.g. {'window': 4, 'n_sigma': 3}
    sgolay = None     # e.g. {'window_length': 6, 'polyorder': 3}
    gate = None       # the gate as the window sets it: None the defaults, False off, or {'on': True, 'hampel': (3, 3), 'sgolay': (9, 2)}

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
    cal, fits = calibrate(V, lia, iv, f.vmod, gate=gate)
    uL = f.units_lia or 'no units'                    # the LIA page's RHK_Zunits (the loader's data are in it)
    uI = f.units_iv or 'no units'                     # the Current page's RHK_Zunits
    ua = "1/V" if uI == uL else f"{uI}/({uL}·V)"      # the units of the stretch a
    n_sweeps = int(cols.size)
    order = np.argsort(V)
    Vs = V[order]
    a_all = np.array([cf.a for cf in fits])
    print(f"{f.name}: {n_sweeps} acquired column(s), bias {V.min():+.3f} to {V.max():+.3f} V; per sweep the least "
          f"squares of a, b, c, nothing locked"
          + (f"; current filters Hampel {hampel}, SG {sgolay}" if filtered else "; current filters off"))
    print(f"gain a per sweep (applied): {a_all.min():.4g} to {a_all.max():.4g} {ua}, median {np.median(a_all):.4g}; "
          + (f"1 / V_mod = {fits[0].a_mod:.4g} {ua} (V_mod = {f.vmod * 1e3:.3g} mV) as the check"
             if np.isfinite(fits[0].a_mod) else "no modulation amplitude recorded"))
    print(gate_text(fits, cols))
    for j, cf in zip(cols, fits):
        print(f"column {j}: own a = {cf.a_own:.4g} ± {cf.se[0]:.2g} {ua}, b = {sm4file.si_text(cf.b_own, uI + '/V')} ± "
              f"{sm4file.si_text(cf.se[1], uI + '/V')}, c = {sm4file.si_text(cf.c_own, uI)}, "
              f"rms residual = {sm4file.si_text(cf.rms, uI)}"
              + (f"; replaced on the {'b' if cf.replaced == 'ab' else 'a'} stage: a = {cf.a:.4g}, "
                 f"b = {sm4file.si_text(cf.b, uI + '/V')}" if cf.replaced else ""))

    X_all = np.column_stack([c.X for c in fits])
    faint = dict(color='0.6', lw=0.5, alpha=0.5)
    bold = dict(color='C0', lw=1.2)
    pA = 1e12
    to_pA = (lambda y: y * pA, lambda y: y / pA)      # the right-hand axes in pA (a Current page in A)

    def cost_surface(k):
        """
        This method's least-squares cost of sweep k, S(a, b) = min over c of
        Σ (I − a·X − b·V − c)², on a window of ±4 standard errors of a and b
        around the sweep's own minimum, widened to hold the applied pair.
        (a grid, b grid, S[b, a], S at the fit).
        """
        cf = fits[k]
        Xs = cf.X[order]
        Is = iv[order, k]
        sa, sb = cf.se[0], cf.se[1]
        ha = 4 * sa if np.isfinite(sa) and sa > 0 else 0.1 * (abs(cf.a_own) or 1.0)
        hb = 4 * sb if np.isfinite(sb) and sb > 0 else 0.1 * (abs(cf.b_own) or 1.0)
        ha = max(ha, 1.2 * abs(cf.a - cf.a_own))
        hb = max(hb, 1.2 * abs(cf.b - cf.b_own))
        aa = np.linspace(cf.a_own - ha, cf.a_own + ha, 121)
        bb = np.linspace(cf.b_own - hb, cf.b_own + hb, 121)
        S = np.empty((bb.size, aa.size))
        for ia, a_ in enumerate(aa):
            r = (Is - a_ * Xs)[None, :] - bb[:, None] * Vs[None, :]
            r = r - r.mean(axis=1, keepdims=True)     # c at its optimum: the mean of what is left
            S[:, ia] = np.sum(r * r, axis=1)
        return aa, bb, S, float(np.sum(cf.residual * cf.residual))

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
        lia_k = lia[:, k]
        iv_k = iv[:, k]
        how = "the trend's pair" if cf.replaced == 'ab' else "the trend's a, b refitted"
        print(f"column {cols[k]}: own a = {cf.a_own:.4g}, b = {cf.b_own:.4g}, c = {cf.c_own:.4g}"
              + (f"; applied a = {cf.a:.4g}, b = {cf.b:.4g} ({how})" if cf.replaced else ""))
        aa, bb, S, S_min = cost_surface(k)

        axes = fig.subplots(2, 3)
        state['axes'].extend(axes.flat)
        fig.suptitle(f"{f.name}: calibration of column {cols[k]}, the sweep's own least squares of a, b, c"
                     + (f", replaced by the gate on the {'b' if cf.replaced == 'ab' else 'a'} stage" if cf.replaced else "")
                     + f"  (bias {V.min():+.3f} to {V.max():+.3f} V" + (", current filtered" if filtered else "") + ")")

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
        ax.set_ylabel(f"X = ∫ LIA dV ({uL}·V)")
        ax.set_title("the regressor X(V) = ∫ LIA dV from the lowest-energy sample (where it starts is absorbed by c)")

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
        ax.plot(V, cf.model, color='C3', lw=1.2, label="a·X + b·V + c (the sweep's own fit)")
        ax.plot(V, cf.a_own * cf.X, '--', color='C0', lw=1, label="a·X")
        ax.plot(V, cf.b_own * V, ':', color='C2', lw=1, label="b·V")
        ax.axhline(cf.c_own, ls='-.', color='C1', lw=1, label="c")
        ax.set_ylabel(f"I ({uI})")
        ax.set_title("the fit in the I(V) frame: I = a·X + b·V + c")
        ax.legend(loc='upper left', fontsize=8)
        ax.text(0.02, 0.03, f"a = {cf.a_own:.4g} ± {cf.se[0]:.2g} {ua}\nb = {sm4file.si_text(cf.b_own, uI + '/V')}"
                            f"\nc = {sm4file.si_text(cf.c_own, uI)}\nrms = {sm4file.si_text(cf.rms, uI, 3)}"
                            + (f"\napplied: a = {cf.a:.4g}, b = {sm4file.si_text(cf.b, uI + '/V')}" if cf.replaced else ""),
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
        if cf.replaced:
            ax.plot(V, cf.a_own * lia_k + cf.b_own, color='C1', lw=0.9, ls='--', label="a·LIA + b with the sweep's own pair")
        ax.plot(V, cal[:, k], color='C3', lw=1.3, label="a·LIA + b" + (" with the gate's pair" if cf.replaced else ""))
        ax.set_ylabel(f"dI/dV ({uI}/V)")
        ax.set_title("the dI/dV frame: a scales the LIA, b shifts it")
        ax.legend(fontsize=8)
        if uI == 'A':
            ax.secondary_yaxis('right', functions=to_pA).set_ylabel("pA/V")

        ax = axes[1, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            logS = np.log10(S / S_min)
        cs = ax.contourf(aa, bb, logS, levels=30, cmap='viridis')
        cb = fig.colorbar(cs, ax=ax, label="log10(S / S_min)")
        state['axes'].append(cb.ax)
        ax.plot(cf.a_own, cf.b_own, 'o', color='w', mec='k', label=f"the sweep's own fit: a = {cf.a_own:.4g}, b = {cf.b_own:.4g}")
        if cf.replaced:
            ax.plot(cf.a, cf.b, 's', color='C1', mec='k', label=f"applied: a = {cf.a:.4g}, b = {cf.b:.4g}")
        if np.isfinite(cf.trend[0]):
            ax.plot(cf.trend[0], cf.trend[1], '+', color='w', ms=9, label=f"the trend: a = {cf.trend[0]:.4g}, b = {cf.trend[1]:.4g}")
        if np.isfinite(cf.a_mod):
            ax.axvline(cf.a_mod, color='w', lw=0.8, ls='--', label=f"1 / V_mod = {cf.a_mod:.4g}")
        ax.set_xlabel(f"a ({ua})")
        ax.set_ylabel(f"b ({uI}/V)")
        ax.set_title("least-squares cost S(a, b) of this sweep, c at its optimum (±4 standard errors)")
        ax.legend(fontsize=8, loc='upper right')
        for ax in list(axes[0]) + list(axes[1, :2]):
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
