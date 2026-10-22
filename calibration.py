"""
Calibration of the lock-in channels against the co-acquired I(V) current:
one gain for the file from the modulation amplitude (1 / V_mod for the
first harmonic, 4 / V_mod² for the second; the joint fit over the columns
when none is recorded, and printed beside it as a check either way), one
offset per curve from the regression anchored at the setpoint (the sample
at the setpoint bias, whose current is the setpoint current: the feedback
loop was opened there), and its double-derivative counterpart for the
IETS (second-harmonic) channel.  Nothing refers to 0 V.  A sweep
that does not pass the setpoint bias is refused (anchor_index); a pixel's
reverse sweeps, which start at the far end, take the a, b of its forward
sweeps (the `ref` columns).  calibrate and calibrate_iets return
(calibrated, fits), every fit entry carrying a, b, c, rms, X / Z, model
and residual.  Plain arrays in, columns = spectra; the pipeline
(sts.STS.process_arrays) calls these on the acquired columns only, after
the channel and the current have been through their filters and before
the Feenstra division.  Run as a file, the __main__
block draws one file's calibration (its inputs, the fit in the I(V) frame,
the calibrated curve in the dI/dV frame, and the cost surface), one sweep
at a time.
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
def calibrate(energy, curves, iv, vmod=None, V_bias=None, I_setpoint=None, ref=None):
    """
    Calibrate dI/dV (LIA) curves against the co-acquired I(V) current
    channel: one gain for the file, one offset per curve, anchored at the
    setpoint.

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
    it is not calibrated.

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

    Returns
    -------
    calibrated : ndarray, same shape as *curves*, float64: a * LIA_j + b_j
        per column, in I units per V (A/V for pages in A)
    fits : list of ColumnFit, one per column (a the file's gain, b, c =
        I_set, X, model, residual, rms, a_joint the joint fit's gain, gain
        'modulation' or 'joint'; the arrays in the original energy
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
    UU = float(np.dot(U, U))

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
        # The offset: the least squares of ΔI − a·X against U.
        b_j = float(np.dot(dIs[j] - a * Xs[j], U) / UU) if UU > 0 else 0.0
        # Calibrated dI/dV in original energy ordering (the column's own LIA, whichever columns were regressed).
        calibrated[:, j] = a * curves[:, j] + b_j
        X_orig = np.empty_like(Xs[j])
        X_orig[sort_idx] = Xs[j]
        model = a * X_orig + b_j * U_orig + I_set
        residual = srcs[j] - model
        fits.append(ColumnFit(float(a), b_j, I_set, X_orig, model, residual,
                              float(np.sqrt(np.mean(residual * residual))), float(a_joint), source))

    return calibrated, fits


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
    # on).  Row 2: the fit in the I(V) frame with its parts a·X, b·(V − V_bias), c = I_set and the
    # residual (a the file's gain from V_mod, b the sweep's offset); the calibrated a·LIA + b in the
    # browser's dI/dV frame
    # with the numerical derivative of the (filtered) current; the least-squares cost S(a, b)
    # around its minimum.  Parameters of every column printed once; the shown sweep's on every step.
    from sm4_Browser import qtplot as plt
    from sm4_Browser import sm4file, widgets, filters
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
    cal, fits = calibrate(V, lia, iv, f.vmod, V_bias=f.V_bias, I_setpoint=f.I_setpoint, ref=(f.lia_ref, f.iv_ref))
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
    for j, cf in zip(cols, fits):
        print(f"column {j}: b = {sm4file.si_text(cf.b, uI + '/V')}, "
              f"I_set = {sm4file.si_text(cf.c, uI)}, rms residual = {sm4file.si_text(cf.rms, uI)}")

    X_all = np.column_stack([c.X for c in fits])
    faint = dict(color='0.6', lw=0.5, alpha=0.5)
    bold = dict(color='C0', lw=1.2)
    to_pA = (lambda y: y * pA, lambda y: y / pA)      # the right-hand axes in pA (a Current page in A)

    def cost_surface(k):
        """
        This method's least-squares cost S(a, b) = Σ (ΔI − a·X − b·U)² of
        sweep k, ΔI = I − I_set and X referenced at the anchor, U = V − V_bias,
        on a window of ±4 standard errors of a and b around the minimum (from
        rms² · M⁻¹).  (a grid, b grid, S[b, a], S at the fit).
        """
        cf = fits[k]
        Xs = cf.X[order]
        dI = iv[order, k] - cf.c
        M = np.array([[Xs @ Xs, Us @ Xs], [Us @ Xs, Us @ Us]])
        cov = cf.rms ** 2 * np.linalg.inv(M)
        sa, sb = np.sqrt(abs(cov[0, 0])), np.sqrt(abs(cov[1, 1]))
        ha = 4 * sa if np.isfinite(sa) and sa > 0 else 0.1 * (abs(cf.a) or 1.0)
        hb = 4 * sb if np.isfinite(sb) and sb > 0 else 0.1 * (abs(cf.b) or 1.0)
        aa = np.linspace(cf.a - ha, cf.a + ha, 121)
        bb = np.linspace(cf.b - hb, cf.b + hb, 121)
        S = np.empty((bb.size, aa.size))
        for ia, a_ in enumerate(aa):
            r = (dI - a_ * Xs)[None, :] - bb[:, None] * Us[None, :]
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
        print(f"column {cols[k]}: a = {cf.a:.4g} ({cf.gain}), b = {cf.b:.4g}")
        aa, bb, S, S_min = cost_surface(k)

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
        ax.axvline(Vs[i0], color='0.5', lw=0.8, ls=':')
        ax.set_ylabel(f"I ({uI})")
        ax.set_title("the fit in the I(V) frame: I − I_set = a·X + b·(V − V_bias)")
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
        ax.plot(V, cal[:, k], color='C3', lw=1.3, label="a·LIA + b (anchored at the setpoint)")
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
        ax.plot(cf.a, cf.b, 'o', color='w', mec='k', label=f"used: a = {cf.a:.4g} ({cf.gain}), b = {cf.b:.4g}")
        ax.set_xlabel(f"a ({ua})")
        ax.set_ylabel(f"b ({uI}/V)")
        ax.set_title("least-squares cost S(a, b) of this sweep alone (±4 standard errors of a free fit)")
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
