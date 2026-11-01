"""
Calibration of the lock-in channels against the co-acquired I(V) current,
one least squares per sweep, through the pins of the locks that are on,
and the gate on the per-sweep gains and offsets.  The true dI/dV relates
to the lock-in output by a·LIA + b, so the recorded current is

    I(V) = a · X(V) + b · V + c,     X(V) = ∫ LIA dV

the constant c the constant of integration: it carries where the
integral starts and the current's own level.  For each sweep a, b and c
are the least squares of that relation over every point of the sweep
(the normal equations of the sum of squares in a, b, c); the calibrated
dI/dV is a · LIA + b.  The gain the modulation amplitude would give,
1 / V_mod, is carried beside the fitted a as the check and is not used.

The locks (GateSettings.lock0 and lockv, the window's '0-lock' and
'Vbias lock' rows under 'gate', off at open, each on its own and with
the gate in any combination; calibrate): a lock pins the fit to a
sample whose current is known.  The 0-lock pins it at the sample nearest
0 V to the recorded current there; the Vbias lock pins a forward sweep
at the setpoint-bias sample to the setpoint current, and a reverse
sweep, which has no setpoint current, at its own first acquired sample
to the recorded current there.  With one pin the constant follows from
the pin; with both, the offset and the constant follow from the gain.

The gate (gate_sweeps; GateSettings, the window's 'gate' row with its
'H' and 'SG' boxes): one sweep with a disturbed current must not break
the calibration.  Along the acquisition order, a sweep's a is compared
with the mean a of the two sweeps touching it; a departure beyond
GATE_FACTOR times the robust scale of those departures over the file,
confirmed against the touching sweeps that passed, marks the sweep, and
it takes the trend's a with its b and c refitted, keeping the pins.  The
same test on b then marks sweeps that take the trend's pair (c from the
Vbias pin, else the 0-lock pin, else the mean of what is left).  The trend is the
sequence of the per-sweep values through the package's Hampel filter
(filters.hampel, window and n_sigma from the 'H' box) and then a
Savitzky-Golay smoothing (length and order from the 'SG' box): where a
sweep's regression should sit among the sweeps around it.  The gate
needs as many sweeps as the Savitzky-Golay length and runs on dI/dV and
IETS alike.

The IETS (second-harmonic) channel is the double-derivative
counterpart: I(V) = a · Z(V) + b · V² / 2 + c · V + d with Z the double
integral of the second-harmonic signal, four parameters per sweep, the
calibrated d2I/dV2 a · S + b, the same locks (one pin removes d, two
remove c and d) and the gate on a and b.  calibrate and calibrate_iets
return (calibrated, fits), every fit entry carrying the parameters,
their standard errors, X / Z, the model and the residual, the sweep's
own pair beside the applied one, the stage that replaced it, the trend's
pair, the Gate record and the pins.  Plain arrays
in, columns = spectra; the pipeline (sts.STS.process_arrays) calls these
on the acquired columns only, after the channel and the current have
been through their filters and before the Feenstra division.  Run as a
file, the __main__ block draws one file's calibration (its inputs, the
fit in the I(V) frame, the calibrated curve in the dI/dV frame, and the
cost surface), one sweep at a time, with the gate, its H and SG boxes
and both locks to set.
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
# The Vbias lock's setpoint sample: the setpoint bias must lie within this fraction of the energy step
# of a sample (anchor_index).
ANCHOR_TOL = 0.5


class GateSettings(NamedTuple):
    """
    The gate and the locks as a window sets them: the gate on / off, its Hampel (window, n_sigma)
    and Savitzky-Golay (length, order); the '0-lock' (lock0) and the 'Vbias lock' (lockv), each on
    its own (both off at open; see calibrate).
    """
    on: bool = True
    hampel: tuple = GATE_HAMPEL
    sgolay: tuple = GATE_SGOLAY
    lock0: bool = False
    lockv: bool = False

    @property
    def text(self):
        """The short form shown beside the toggle: 'H 3,3 SG 9,2'."""
        return f"H {self.hampel[0]:g},{self.hampel[1]:g} SG {self.sgolay[0]:g},{self.sgolay[1]:g}"

    @property
    def locks_text(self):
        """The locks for the console: 'nothing locked', '0-lock on', 'Vbias lock on' or '0-lock and Vbias lock on'."""
        on = [name for name, flag in (('0-lock', self.lock0), ('Vbias lock', self.lockv)) if flag]
        return " and ".join(on) + " on" if on else "nothing locked"


GATE_DEFAULT = GateSettings()


def gate_settings(gate):
    """
    The GateSettings of `gate`: None (the defaults), a GateSettings, a dict
    with 'on', 'hampel', 'sgolay', 'lock0', 'lockv' (missing keys the
    defaults; the locks off), or a bool (the gate on / off at the defaults,
    the locks off).  Validated: the Hampel window a whole number
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
                         sgolay=tuple(gate.get('sgolay', GATE_SGOLAY)), lock0=bool(gate.get('lock0', False)),
                         lockv=bool(gate.get('lockv', False)))
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
    return GateSettings(bool(g.on), (int(hw), hs), (int(sl), int(so)), bool(g.lock0), bool(g.lockv))


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
    pins: tuple = ()                  # the locks' pins: ((name, V, I), ...), 'Vbias' before '0'; () with nothing locked


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
    se: tuple = (float('nan'),) * 4   # the standard errors of (a, b, c, d); NaN for a parameter a pin fixes
    a_mod: float = float('nan')       # the gain the modulation amplitude would give, 4 / V_mod² (the check)
    a_own: float = float('nan')       # the sweep's own (a, b) (equal to a, b when the gate left it)
    b_own: float = float('nan')
    replaced: str = ''                # '' (its own pair), 'a' (the trend's a, the rest refitted), 'ab' (the trend's pair)
    trend: tuple = (float('nan'),) * 2   # the trend's (a, b) at this column, NaN when the gate did not run
    gate: Gate = _NO_GATE             # the call's gate record
    pins: tuple = ()                  # the locks' pins: ((name, V, I), ...), as ColumnFit's


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


def _sweep_arrays(sweeps, n):
    """The per-column (direction, first acquired sample) of `sweeps` for n columns; None: every column forward."""
    if sweeps is None or sweeps[0] is None:
        return np.ones(n, dtype=int), np.zeros(n, dtype=int)
    d = np.asarray(sweeps[0], dtype=int).reshape(-1)
    f = np.asarray(sweeps[1], dtype=int).reshape(-1)
    if d.size != n or f.size != n:
        raise ValueError(f"the sweep directions hold {d.size} columns, the data {n}.")
    return d, f


def _pins(settings, e_sorted, sort_idx, I, direction, first, V_bias, I_setpoint):
    """
    The pins of one sweep (the energies sorted, I its current on them):
    [(index on the sorted axis, current, name)], the Vbias pin before the
    0-lock pin.  0-lock: the sample nearest 0 V (whichever is closest, not
    exactly 0) at the recorded current there.  Vbias lock: a forward sweep
    (direction +1, the one that left the setpoint) at the setpoint-bias
    sample (anchor_index) at I_setpoint; a reverse sweep (-1, no setpoint
    current) at its own first acquired sample (`first`, an index on the
    column's original order) at the recorded current there; a column that
    averages sweeps of both directions (0) has no single first sample and
    takes no Vbias pin.  A forward sweep without the setpoint is refused
    with anchor_index's message.
    """
    pins = []
    if settings.lockv:
        if direction > 0:
            if I_setpoint is None or not np.isfinite(I_setpoint):
                raise ValueError("no setpoint current (RHK_Current) recorded; the calibration anchors at the setpoint.")
            pins.append((anchor_index(e_sorted, V_bias), float(I_setpoint), 'Vbias'))
        elif direction < 0:
            k = int(np.flatnonzero(sort_idx == int(first))[0])
            pins.append((k, float(I[k]), 'Vbias'))
    if settings.lock0:
        k = int(np.argmin(np.abs(e_sorted)))
        pins.append((k, float(I[k]), '0'))
    return pins


def _pinned_solve(R, I, pins, what):
    """
    The least squares of I against the columns of R (npts, k) with the fit
    forced through every pin (index p, current I_p): with m pins the last m
    parameters follow from the others through the pins, and the rest are
    the least squares of what is left.  One pin on (X, V, 1):
    I − I_p = a (X − X_p) + b (V − V_p), c from the pin; two pins: b and c
    follow from a, one-parameter least squares in a.  Returns (parameters
    (k,), standard errors (k,; NaN for the parameters the pins fix),
    residual of the free fit).  ValueError when the pins leave the
    parameters undetermined.
    """
    k = R.shape[1]
    m = len(pins)
    if m == 0:
        return _solve(R, I, what)
    P = np.array([p for p, _, _ in pins], dtype=int)
    Ip = np.array([v for _, v, _ in pins], dtype=np.float64)
    M = R[P, k - m:]
    if not np.isfinite(np.linalg.cond(M)) or abs(np.linalg.det(M)) == 0.0:
        raise ValueError(f"{what}: the pins do not determine the parameters.")
    Minv = np.linalg.inv(M)
    A = R[P, :k - m]
    theta = np.empty(k)
    se = np.full(k, np.nan)
    if k - m > 0:
        R_eff = R[:, :k - m] - R[:, k - m:] @ Minv @ A
        y_eff = I - R[:, k - m:] @ (Minv @ Ip)
        free, se_free, residual = _solve(R_eff, y_eff, what)
        theta[:k - m] = free
        se[:k - m] = se_free
    else:
        free = np.zeros(0)
        residual = I - R @ np.concatenate([free, Minv @ Ip])
    theta[k - m:] = Minv @ (Ip - A @ free)
    return theta, se, residual


def _pinned_rest(R, I, pins, held, what):
    """
    The parameters after the `held` ones (the first len(held) of R's
    columns, fixed) by the least squares keeping the pins (_pinned_solve on
    what the held ones leave); with more pins than free parameters the
    first pins are kept (the Vbias pin before the 0-lock pin).
    """
    held = np.asarray(held, dtype=np.float64)
    h = held.size
    y = I - R[:, :h] @ held
    keep = pins[:R.shape[1] - h]
    shifted = [(p, v - float(R[p, :h] @ held), n) for p, v, n in keep]
    theta, _, _ = _pinned_solve(R[:, h:], y, shifted, what)
    return theta


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
    order, `refit(j, a_j)` giving the b of sweep j and the rest of its
    parameters (dI/dV: c; IETS: (c, d)) refitted with its a held and the
    pins kept (calibrate's, calibrate_iets').  Stage 1: the touching-sweep test on a
    (_touching_flags); a flagged sweep takes the trend's a (_trend) and
    its b and c refitted.  Stage 2: the test on the b that came out of
    stage 1; a flagged sweep takes the trend's a and b, the trends drawn
    on the sequences after stage 1.  Returns (a, b, the rest or None per
    column (None: the caller fits it for the applied pair), Gate): the
    applied values and the record.  With the gate off, or
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
# V_bias and I_setpoint are the Vbias lock's setpoint; ref is accepted for the callers and not used.
def calibrate(energy, curves, iv, vmod=None, V_bias=None, I_setpoint=None, ref=None, gate=None, sweeps=None):
    """
    Calibrate dI/dV (LIA) curves against the co-acquired I(V) current
    channel: per sweep the least squares of a, b, c, through the pins of
    the locks that are on (none at open), then the gate.

    Physics
    -------
    The true dI/dV relates to the raw LIA signal by:

        true dI/dV(V) = a · LIA(V) + b

    Integrating both sides:

        I(V) = a · X(V) + b · V + c

    where X(V) is the cumulative integral of the experimental LIA signal
    (cumulative trapezoidal rule from the lowest-energy sample) and c the
    constant of integration.  The constant carries where the integral
    starts and the level of the current, so with no lock on no sample is
    privileged, no current is subtracted, and the fit does not depend on
    where X starts (a shift of X by a constant is absorbed by c).  For each spectrum j the
    least squares of

        S_j(a, b, c) = Σₙ [I_j(Vₙ) − a·X_j(Vₙ) − b·Vₙ − c]²

    gives (a_j, b_j, c_j), the normal equations ∂S/∂a = ∂S/∂b = ∂S/∂c = 0,
    i.e. [[ΣX², ΣXV, ΣX], [ΣXV, ΣV², ΣV], [ΣX, ΣV, n]] · [a; b; c] =
    [ΣIX; ΣIV; ΣI].  The relation holds on any bias window, crossing 0 V
    or not, and in either sweep direction (the energies are sorted).  The
    gain from the modulation amplitude, 1 / V_mod, is carried in the fits
    (a_mod) as the check beside a_j; it is not used.

    The locks (GateSettings.lock0, lockv; each on its own, any
    combination with the gate): a pin is a sample the fit must pass
    through at a known current.  '0-lock': the sample nearest 0 V
    (whichever is closest, not exactly 0) at the recorded (filtered)
    current there.  'Vbias lock': a forward sweep (the one that left the
    setpoint) at the setpoint-bias sample (anchor_index, within
    ANCHOR_TOL of the energy step) at I_setpoint; a reverse sweep, which
    has no setpoint current, at its own first acquired sample at the
    recorded current there; a column averaging sweeps of both directions
    takes no Vbias pin.  One pin p: I − I_p = a (X − X_p) + b (V − V_p),
    the least squares of a and b, c from the pin (the exact-pin form of
    the far-anchor regression); two pins: b and c follow from a, a
    one-parameter least squares in a (_pinned_solve).

    The gate (gate_sweeps, the module docstring) then runs on the per-sweep
    a and b in the columns' order: a sweep that takes the trend's a has b
    and c refitted keeping the pins; one that takes the trend's pair has c
    from the Vbias pin when that lock is on, else from the 0-lock pin, else
    the mean of what is left.  A replaced sweep's calibrated curve uses the
    pair it took.  The calibrated dI/dV is a_j · LIA_j(V) + b_j with the
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
    V_bias, I_setpoint : float or None
        The setpoint (RHK_Bias in V, RHK_Current in A): the Vbias lock's
        pin of a forward sweep; a forward sweep without them, or without a
        sample at V_bias, is refused while that lock is on.
    ref
        A sweep view's forward-sweep reference, accepted for the callers
        and not used: every sweep is regressed on its own data.
    gate : GateSettings, dict, bool or None
        The gate and the locks as the window sets them (gate_settings);
        None the defaults (the gate on, Hampel 3,3, Savitzky-Golay 9,2;
        the locks off).
    sweeps : (direction, first) or None
        Per column its sweep direction (+1 forward, -1 reverse, 0 an
        average of both) and the index of its first acquired sample in the
        column's original order (STS.sweep_pins); None: every column a
        forward sweep.

    Returns
    -------
    calibrated : ndarray, same shape as *curves*, float64: a_j * LIA_j + b_j
        per column with the applied pair, in I units per V (A/V for pages
        in A)
    fits : list of ColumnFit, one per column (the applied a, b, c, X, model,
        residual, rms and se of the sweep's own fit, a_mod, the own pair,
        the stage that replaced it, the trend's pair, the Gate record, the
        pins; the arrays in the original energy ordering)

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
    settings = gate_settings(gate)
    direction, first = _sweep_arrays(sweeps, n_curves)
    Xs, Is, Rs, pins, own, ses = [], [], [], [], [], []
    for j in range(n_curves):
        # X(V) = cumulative integral of the LIA from the lowest-energy sample.
        X = cumulative_trapezoid(curves[sort_idx, j], e_sorted, initial=0)
        I = iv_data[sort_idx, j]
        R = np.column_stack([X, e_sorted, one])
        pj = _pins(settings, e_sorted, sort_idx, I, direction[j], first[j], V_bias, I_setpoint)
        sol, se, _ = _pinned_solve(R, I, pj, f"column {j}")
        Xs.append(X)
        Is.append(I)
        Rs.append(R)
        pins.append(pj)
        own.append(sol)
        ses.append(se)
    own = np.array(own) if n_curves else np.zeros((0, 3))

    def refit(j, a_j):
        """The (b, c) of column j with a held at a_j, the pins kept."""
        b_, c_ = _pinned_rest(Rs[j], Is[j], pins[j], [a_j], f"column {j}")
        return float(b_), float(c_)

    a_app, b_app, c_app, record = gate_sweeps(own[:, 0], own[:, 1], refit, settings) if n_curves else \
        (np.zeros(0), np.zeros(0), [], _NO_GATE)
    calibrated = np.empty_like(curves, dtype=np.float64)
    fits = []
    for j in range(n_curves):
        a_j, b_j = float(a_app[j]), float(b_app[j])
        if c_app[j] is None:
            # The trend's pair (or the sweep's own): c from the Vbias pin, else the 0-lock pin, else the
            # mean of what the pair leaves (_pinned_rest keeps the first pin).
            c_j = own[j, 2] if (a_j == own[j, 0] and b_j == own[j, 1]) else \
                float(_pinned_rest(Rs[j], Is[j], pins[j], [a_j, b_j], f"column {j}")[0])
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
                              float(own[j, 0]), float(own[j, 1]), float(own[j, 2]), replaced, trend, record,
                              tuple((n, float(e_sorted[p]), float(v)) for p, v, n in pins[j])))
    return calibrated, fits


def calibrate_iets(energy, curves, iv, vmod=None, V_bias=None, I_setpoint=None, ref=None, gate=None, sweeps=None):
    """
    Calibrate IETS (second-harmonic lock-in) curves against the co-acquired
    I(V) current channel, the double-derivative counterpart of calibrate:
    per sweep the least squares of a, b, c, d through the pins of the
    locks that are on, then the gate on a and b.

    The true d2I/dV2 relates to the raw second-harmonic signal S by

        true d2I/dV2(V) = a · S(V) + b

    Integrating twice:

        I(V) = a · Z(V) + b · V² / 2 + c · V + d

    where Z(V) is the double cumulative trapezoidal integral of S from the
    lowest-energy sample, c the constant of the first integration (the
    dI/dV level) and d of the second (the current level).  For each
    spectrum j the least squares in (a, b, c, d) over every point of the
    sweep, the 4 by 4 normal equations on the regressors (Z, V²/2, V, 1).
    The locks pin it as they pin calibrate's fit (the same pins): one pin
    removes d, two remove c and d (_pinned_solve).  The gate (gate_sweeps)
    then runs on a and b as on dI/dV: a sweep that takes the trend's a has
    b, c and d refitted keeping the pins; one that takes the trend's pair
    has c and d refitted keeping the pins (the Vbias pin first).  The
    calibrated curve is a_j · S_j(V) + b_j with the applied pair (in the
    Current page's units per V²), in the original energy ordering.  The
    gain the modulation amplitude would give, 4 / V_mod², is carried as
    the check (a_mod).

    Parameters: as calibrate (energy in V, any order; curves the
    second-harmonic page's data as the loader gives it, through the dI/dV
    filters when they are on; iv the recorded current through the
    pipeline's filters, None raises ValueError; vmod the modulation
    amplitude; V_bias, I_setpoint the Vbias lock's setpoint; gate the gate
    and the locks; sweeps the columns' directions; ref accepted and not
    used).

    Returns
    -------
    calibrated : ndarray, same shape as *curves*, float64
    fits : list of IetsColumnFit, one per column (the applied a, b, c, d,
        Z, model, residual, rms and se of the sweep's own fit, a_mod, the
        own pair, the stage that replaced it, the trend's pair, the Gate
        record, the pins; the arrays in the original energy ordering)
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
    settings = gate_settings(gate)
    direction, first = _sweep_arrays(sweeps, n_curves)
    Zs_all, Is_all, Rs, pins, own, ses = [], [], [], [], [], []
    for j in range(n_curves):
        # Y(V) = cumulative integral of S from the lowest-energy sample; Z(V) = its integral.
        Y = cumulative_trapezoid(curves[sort_idx, j], e_sorted, initial=0)
        Zs = cumulative_trapezoid(Y, e_sorted, initial=0)
        Is = iv_data[sort_idx, j]
        R = np.column_stack([Zs, 0.5 * e_sorted * e_sorted, e_sorted, one])
        pj = _pins(settings, e_sorted, sort_idx, Is, direction[j], first[j], V_bias, I_setpoint)
        sol, se, _ = _pinned_solve(R, Is, pj, f"column {j}")
        Zs_all.append(Zs)
        Is_all.append(Is)
        Rs.append(R)
        pins.append(pj)
        own.append(sol)
        ses.append(se)
    own = np.array(own) if n_curves else np.zeros((0, 4))

    def refit(j, a_j):
        """The b and the (c, d) of column j with a held at a_j, the pins kept."""
        b_, c_, d_ = _pinned_rest(Rs[j], Is_all[j], pins[j], [a_j], f"column {j}")
        return float(b_), (float(c_), float(d_))

    a_app, b_app, cd_app, record = gate_sweeps(own[:, 0], own[:, 1], refit, settings) if n_curves else \
        (np.zeros(0), np.zeros(0), [], _NO_GATE)
    calibrated = np.empty_like(curves, dtype=np.float64)
    fits = []
    for j in range(n_curves):
        a_j, b_j = float(a_app[j]), float(b_app[j])
        if cd_app[j] is not None:
            c_j, d_j = cd_app[j]
        elif a_j == own[j, 0] and b_j == own[j, 1]:
            c_j, d_j = float(own[j, 2]), float(own[j, 3])
        else:
            c_j, d_j = (float(v) for v in _pinned_rest(Rs[j], Is_all[j], pins[j], [a_j, b_j], f"column {j}"))
        calibrated[:, j] = a_j * curves[:, j] + b_j
        Z_orig = np.empty_like(Zs_all[j])
        Z_orig[sort_idx] = Zs_all[j]
        a_o, b_o, c_o, d_o = (float(v) for v in own[j])
        model = a_o * Z_orig + b_o * 0.5 * energy * energy + c_o * energy + d_o
        residual = iv_data[:, j] - model
        replaced = 'ab' if j in record.replaced_b else ('a' if j in record.replaced_a else '')
        trend = (float(record.trend_a[j]), float(record.trend_b[j])) if record.ran else (float('nan'),) * 2
        fits.append(IetsColumnFit(a_j, b_j, float(c_j), float(d_j), Z_orig, model, residual,
                                  float(np.sqrt(np.mean(residual * residual))), tuple(float(s) for s in ses[j]), a_mod,
                                  a_o, b_o, replaced, trend, record,
                                  tuple((n, float(e_sorted[p]), float(v)) for p, v, n in pins[j])))
    return calibrated, fits


def gate_from_record(rec):
    """The GateSettings a window record holds ('gate' on / off, 'gate_hampel' [window, n_sigma], 'gate_sgolay' [length, order], 'lock0', 'lockv'); a record without them (older) gives the defaults, the locks off."""
    try:
        return gate_settings({'on': rec.get('gate', True), 'hampel': rec.get('gate_hampel', GATE_HAMPEL),
                              'sgolay': rec.get('gate_sgolay', GATE_SGOLAY), 'lock0': rec.get('lock0', False),
                              'lockv': rec.get('lockv', False)})
    except ValueError as exc:
        print(f"record: the gate settings are not usable ({exc}); the defaults are used.")
        return GATE_DEFAULT


def gate_record(settings):
    """The record keys of GateSettings: {'gate', 'gate_hampel', 'gate_sgolay', 'lock0', 'lockv'} (gate_from_record reads them)."""
    return {'gate': bool(settings.on), 'gate_hampel': [settings.hampel[0], settings.hampel[1]],
            'gate_sgolay': [settings.sgolay[0], settings.sgolay[1]], 'lock0': bool(settings.lock0),
            'lockv': bool(settings.lockv)}


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
    unit_b = "pA/V²" if isinstance(fits[0], IetsColumnFit) else "pA/V"
    return (f"gate {rec.settings.text}, factor {GATE_FACTOR:g}: tolerance ±{rec.tol_a:.3g} on a, "
            f"±{rec.tol_b * 1e12:.3g} {unit_b} on b; {len(ra)} sweep(s) took the trend's a" + (f" ({ra})" if ra else "")
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
    from sm4_Browser import sm4file
    path = r'experiment_dir/file.sm4'
    column = 0        # the starting sweep: an index among the acquired columns (0 = the first)
    hampel = None     # the current filters as the windows set them (None = off), e.g. {'window': 4, 'n_sigma': 3}
    sgolay = None     # e.g. {'window_length': 6, 'polyorder': 3}
    gate = None       # the gate and the locks at open: None the defaults (the gate on, the locks off), False the gate off, or
    #                   {'on': True, 'hampel': (3, 3), 'sgolay': (9, 2), 'lock0': False, 'lockv': False}; the window's
    #                   'gate' row, its 'H' and 'SG' boxes and the '0-lock' / 'Vbias lock' rows change them

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
    sweeps = tuple(a[cols] for a in f.spectra.sweep_pins())   # every column's direction (the Vbias lock's pin)
    run = {'gate': gate_settings(gate)}

    def fit_all():
        """The file's calibration with the window's gate and locks: (calibrated, fits)."""
        return calibrate(V, lia, iv, f.vmod, V_bias=f.spectra.V_bias, I_setpoint=f.spectra.I_setpoint,
                         gate=run['gate'], sweeps=sweeps)

    cal, fits = fit_all()
    uL = f.units_lia or 'no units'                    # the LIA page's RHK_Zunits (the loader's data are in it)
    uI = f.units_iv or 'no units'                     # the Current page's RHK_Zunits
    ua = "1/V" if uI == uL else f"{uI}/({uL}·V)"      # the units of the stretch a
    n_sweeps = int(cols.size)
    order = np.argsort(V)
    Vs = V[order]
    a_all = np.array([cf.a for cf in fits])

    def report():
        """The file's calibration on the console: the method, the gains, the gate's doing, every column's parameters and pins."""
        a_all = np.array([cf.a for cf in fits])
        print(f"{f.name}: {n_sweeps} acquired column(s), bias {V.min():+.3f} to {V.max():+.3f} V; per sweep the least "
              f"squares of a, b, c, {run['gate'].locks_text}"
              + (f"; current filters Hampel {hampel}, SG {sgolay}" if filtered else "; current filters off"))
        print(f"gain a per sweep (applied): {a_all.min():.4g} to {a_all.max():.4g} {ua}, median {np.median(a_all):.4g}; "
              + (f"1 / V_mod = {fits[0].a_mod:.4g} {ua} (V_mod = {f.vmod * 1e3:.3g} mV) as the check"
                 if np.isfinite(fits[0].a_mod) else "no modulation amplitude recorded"))
        print(gate_text(fits, cols))
        for j, cf in zip(cols, fits):
            se_b = (f"± {sm4file.si_text(cf.se[1], uI + '/V')}" if np.isfinite(cf.se[1]) else "(from the pins)")
            print(f"column {j}: own a = {cf.a_own:.4g} ± {cf.se[0]:.2g} {ua}, b = {sm4file.si_text(cf.b_own, uI + '/V')} "
                  f"{se_b}, c = {sm4file.si_text(cf.c_own, uI)}, "
                  f"rms residual = {sm4file.si_text(cf.rms, uI)}"
                  + "".join(f"; {'Vbias' if n == 'Vbias' else '0'} pin at {v:+.3f} V, {sm4file.si_text(i_, uI)}"
                            for n, v, i_ in cf.pins)
                  + (f"; replaced on the {'b' if cf.replaced == 'ab' else 'a'} stage: a = {cf.a:.4g}, "
                     f"b = {sm4file.si_text(cf.b, uI + '/V')}" if cf.replaced else ""))

    report()
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
    # automatic layout: the one rule for every window (nothing placed by hand): the six panels in the
    # figure's automatic grid (fig.subplots, rebuilt for every sweep), the sweep buttons and the sweep
    # text in a row under them (fig.layout_root).
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
        fig.suptitle(f"{f.name}: calibration of column {cols[k]}, the sweep's own least squares of a, b, c, "
                     f"{run['gate'].locks_text}"
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

    def on_gate(st):
        """The 'gate' row, its 'H' / 'SG' boxes, '0-lock' or 'Vbias lock': the file calibrated again, the shown sweep redrawn."""
        if refit(gate_settings(st['gate']._asdict())):
            report()
            draw(state['k'])

    def refit(settings):
        """The calibration with new settings; a refused one (a lock the file cannot hold) is said and the old kept."""
        global cal, fits, X_all
        old = run['gate']
        run['gate'] = settings
        try:
            cal, fits = fit_all()
        except ValueError as exc:
            print(f"calibration not changed: {exc}")
            run['gate'] = old
            box.set_state(gate=old._asdict())
            return False
        X_all = np.column_stack([c.X for c in fits])
        return True

    # The sweep buttons bottom centre, the sweep text beside them, the gate and the locks at the right
    # (controls.PipelineBox without its calibrate / normalize rows: the calibration is this window's subject).
    # The settings cross as dicts: run as a file this module is __main__, whose GateSettings is not the
    # package's that the box uses.
    from sm4_Browser import controls
    steps = controls.StepButtons(fig, '< sweep', 'sweep >')
    steps.on_change(on_step)
    sweep_text = controls.Label(fig, 'sweep text', fs=9)
    box = controls.PipelineBox(fig, gate=run['gate']._asdict(), steps=False)
    box.on_change(on_gate)
    fig.layout_root().addLayout(controls.row('stretch', steps, sweep_text, 'stretch', box))
    draw(state['k'])
    plt.show()
