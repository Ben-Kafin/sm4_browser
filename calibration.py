"""
Calibration of the lock-in channels against the co-acquired I(V) current:
the per-curve gain-plus-offset regression anchored at the setpoint (the
sample at the setpoint bias, whose current is the setpoint current: the
feedback loop was opened there), and its double-derivative counterpart
for the IETS (second-harmonic) channel.  Nothing refers to 0 V.  A sweep
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
    a: float               # dI/dV scale: I units per V per LIA unit: 1/V when both are in A
    b: float               # dI/dV offset: I units per V
    c: float               # the setpoint current I_set, I units (not fitted: the anchor)
    X: np.ndarray          # ∫ LIA dV referenced at the anchor: X_raw − X_raw(V_bias) (LIA units·V)
    model: np.ndarray      # a·X + b·(V − V_bias) + c (I units)
    residual: np.ndarray   # I(V) − model (I units)
    rms: float             # rms of the residual (I units)


class IetsColumnFit(NamedTuple):
    """
    One column's IETS regression, I(V) − I_set = a·Z(V) + b·U²/2 + c·U with
    U = V − V_bias and Z referenced at the anchor sample, the one at the
    setpoint bias (calibrate_iets).  The arrays are in the column's
    original energy order (npts,).
    """
    a: float               # d2I/dV2 scale: I units per V² per second-harmonic unit
    b: float               # d2I/dV2 offset: I units per V²
    c: float               # the unknown dI/dV at the anchor (I units per V)
    d: float               # the setpoint current I_set, I units (not fitted: the anchor)
    Z: np.ndarray          # the double integral of the second-harmonic signal, both integrations referenced at the anchor (second-harmonic units·V²)
    model: np.ndarray      # a·Z + b·U²/2 + c·U + d (I units)
    residual: np.ndarray   # I(V) − model (I units)
    rms: float             # rms of the residual (I units)


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


# vmod (the modulation amplitude, V) is accepted for the pipeline's call (STS.process_arrays passes the
# file's RHK_CH1Drive_Amplitude to whichever calibration file is in place) and is not used by this method.
def calibrate(energy, curves, iv, vmod=None, V_bias=None, I_setpoint=None, ref=None):
    """
    Calibrate dI/dV (LIA) curves via per-curve linear regression against
    the co-acquired I(V) current channel, anchored at the setpoint.

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

    For each spectrum j we minimise the least-squares cost, with
    ΔI = I − I_set and U = V − V_bias,

        S(a, b) = Σₙ [ΔI(Vₙ) − a·X(Vₙ) − b·Uₙ]²

    Setting ∂S/∂a = 0 and ∂S/∂b = 0 gives the normal equations:

        ΣΔI·X = a·ΣX² + b·ΣU·X
        ΣΔI·U = a·ΣX·U + b·ΣU²

    In matrix form:

        M · [a; b] = [ΣΔI·X ; ΣΔI·U]

        M = [[ΣX²,  ΣUX],
             [ΣXU,  ΣU²]]

        [a; b] = M⁻¹ · [ΣΔI·X ; ΣΔI·U]

    The calibrated dI/dV is then  a_j · LIA_j(V) + b_j, which
    simultaneously recovers the scale and offset in a single fit.  The
    relation is exact in U, so the fit is unbiased whatever the bias
    window, crossing 0 V or not, and in either sweep direction (the
    energies are sorted); nothing refers to 0 V.  A sweep that does not
    pass V_bias (anchor_index) or a file without the setpoint is refused
    (ValueError): it is not calibrated.

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
        The modulation amplitude (V), accepted for the pipeline's call and
        not used by this method (see the comment above).
    V_bias, I_setpoint : float
        The setpoint (RHK_Bias in V, RHK_Current in A): the anchor.
    ref : (lia_ref, iv_ref) or None
        Columns aligned with `curves` whose sweeps did start at the
        setpoint: a pixel's forward sweeps (their raw mean), for the views
        of its reverse sweeps or of all its sweeps (sts.STS.sweep_view).
        When given, the regression of column j runs on ref column j and
        its a, b are applied to curves[:, j]; a column whose reference is
        not finite (a pixel without a forward sweep) is regressed on
        itself.

    Returns
    -------
    calibrated : ndarray, same shape as *curves*, float64: a_j * LIA_j + b_j
        per column, in I units per V (A/V for pages in A)
    fits : list of ColumnFit, one per column (a, b, c = I_set, X, model,
        residual, rms; the arrays in the original energy ordering; X, the
        model and the residual of the regressed columns, the reference's
        when one was used)

    A column whose X and U are linearly dependent (for instance an all-zero
    LIA column, X = 0 everywhere) makes M singular and numpy.linalg.solve
    raises LinAlgError; the pipeline passes the acquired columns only.
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
    calibrated = np.empty_like(curves, dtype=np.float64)
    fits = []

    for j in range(n_curves):
        rc = _ref_column(ref, j, curves.shape[0])
        src_lia = curves[:, j] if rc is None else rc[0]
        src_iv = iv_data[:, j] if rc is None else rc[1]
        # Sort LIA and I(V) to match ascending energy.
        lia_sorted = src_lia[sort_idx]
        iv_sorted = src_iv[sort_idx]

        # X(V) = cumulative integral of LIA, referenced at the anchor sample (V_bias).
        X_raw = cumulative_trapezoid(lia_sorted, e_sorted, initial=0)
        X = X_raw - X_raw[i_a]

        # ΔI(V) = I(V) − I_set: the loop was opened at V_bias with the current at the setpoint.
        delta_I = iv_sorted - I_set

        # Assemble normal equations:  M · [a; b] = rhs
        sum_X2 = np.dot(X, X)
        sum_U2 = np.dot(U, U)
        sum_UX = np.dot(U, X)
        sum_IX = np.dot(delta_I, X)
        sum_IU = np.dot(delta_I, U)

        M = np.array([[sum_X2, sum_UX],
                      [sum_UX, sum_U2]])
        rhs = np.array([sum_IX, sum_IU])

        # Solve for [a, b].
        ab = np.linalg.solve(M, rhs)
        a_j = ab[0]
        b_j = ab[1]

        # Calibrated dI/dV in original energy ordering (the column's own LIA, whichever columns were regressed).
        calibrated[:, j] = a_j * curves[:, j] + b_j

        # The fit entry (the common contract of the calibration files): the referenced integral
        # and the model in the original ordering, c the setpoint current.
        X_orig = np.empty_like(X)
        X_orig[sort_idx] = X
        U_orig = np.empty_like(U)
        U_orig[sort_idx] = U
        model = a_j * X_orig + b_j * U_orig + I_set
        residual = src_iv - model
        fits.append(ColumnFit(float(a_j), float(b_j), I_set, X_orig, model, residual,
                              float(np.sqrt(np.mean(residual * residual)))))

    return calibrated, fits


# vmod: accepted for the pipeline's call and not used by this method (see calibrate).
def calibrate_iets(energy, curves, iv, vmod=None, V_bias=None, I_setpoint=None, ref=None):
    """
    Calibrate IETS (second-harmonic lock-in) curves via per-curve linear
    regression against the co-acquired I(V) current channel, the
    double-derivative counterpart of calibrate, anchored at the setpoint.

    The true d2I/dV2 relates to the raw second-harmonic signal S by

        true d2I/dV2(V) = a · S(V) + b

    Integrating twice from the anchor sample (the setpoint bias V_bias,
    where the current is I_set), with U = V − V_bias:

        I(V) − I_set = a · Z(V) + b · U² / 2 + c · U

    where Z(V) is the double cumulative trapezoidal integral of S, both
    integrations referenced at the anchor, and c absorbs the unknown dI/dV
    at the anchor.  For each spectrum j the least-squares cost in (a, b,
    c) gives the 3 by 3 normal equations

        M · [a; b; c] = [ΣΔI·Z ; ΣΔI·U²/2 ; ΣΔI·U],
        M = [[ΣZ², ΣZ·U²/2, ΣZ·U], [ΣZ·U²/2, Σ(U²/2)², ΣU²/2·U], [ΣZ·U, ΣU²/2·U, ΣU²]]

    solved with numpy.linalg.solve as calibrate solves its 2 by 2 system.
    The calibrated curve is a_j · S_j(V) + b_j (in the Current page's units
    per V²), in the original energy ordering.  As in calibrate the relation
    is exact in U, so nothing refers to 0 V; the three regressors are
    nearly collinear, so the anchor must be right, and a sweep that does
    not pass V_bias is refused (anchor_index).

    Parameters: as calibrate (energy in V, any order; curves the
    second-harmonic page's data as the loader gives it, through the dI/dV
    filters when they are on; iv the recorded current through the
    pipeline's filters, None raises ValueError; vmod accepted and not used;
    V_bias, I_setpoint the setpoint; ref the reference columns of the
    sweeps that started at the setpoint).

    Returns
    -------
    calibrated : ndarray, same shape as *curves*, float64
    fits : list of IetsColumnFit, one per column (a, b, c, d = I_set, Z,
        model, residual, rms; the arrays in the original energy ordering)
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
    calibrated = np.empty_like(curves, dtype=np.float64)
    fits = []
    Q = 0.5 * U * U

    for j in range(n_curves):
        rc = _ref_column(ref, j, curves.shape[0])
        src_s = curves[:, j] if rc is None else rc[0]
        src_iv = iv_data[:, j] if rc is None else rc[1]
        s_sorted = src_s[sort_idx]
        iv_sorted = src_iv[sort_idx]

        # Y(V) = cumulative integral of S, referenced at the anchor; Z(V) = its integral, referenced too.
        Y_raw = cumulative_trapezoid(s_sorted, e_sorted, initial=0)
        Y = Y_raw - Y_raw[i_a]
        Z_raw = cumulative_trapezoid(Y, e_sorted, initial=0)
        Z = Z_raw - Z_raw[i_a]

        delta_I = iv_sorted - I_set

        M = np.array([[np.dot(Z, Z), np.dot(Z, Q), np.dot(Z, U)],
                      [np.dot(Q, Z), np.dot(Q, Q), np.dot(Q, U)],
                      [np.dot(U, Z), np.dot(U, Q), np.dot(U, U)]])
        rhs = np.array([np.dot(delta_I, Z), np.dot(delta_I, Q), np.dot(delta_I, U)])

        abc = np.linalg.solve(M, rhs)
        a_j, b_j, c_j = abc[0], abc[1], abc[2]

        calibrated[:, j] = a_j * curves[:, j] + b_j

        # The fit entry: the referenced double integral and the model in the original ordering,
        # d the setpoint current.
        Z_orig = np.empty_like(Z)
        Z_orig[sort_idx] = Z
        U_orig = np.empty_like(U)
        U_orig[sort_idx] = U
        model = a_j * Z_orig + b_j * 0.5 * U_orig * U_orig + c_j * U_orig + I_set
        residual = src_iv - model
        fits.append(IetsColumnFit(float(a_j), float(b_j), float(c_j), I_set, Z_orig, model, residual,
                                  float(np.sqrt(np.mean(residual * residual)))))

    return calibrated, fits


if __name__ == '__main__':
    # One file's dI/dV calibration by this method drawn, one sweep at a time (the file, the starting
    # sweep, and the current filters set here; '< sweep' / 'sweep >' step through the acquired columns
    # in file order, point, line, or map alike).  Row 1, the calibration's inputs: the raw LIA, its
    # integral X referenced at the anchor (the sample at the setpoint bias), the recorded current
    # (every acquired column faint, the shown sweep bold; the filtered current when the filters are
    # on).  Row 2: the fit in the I(V) frame with its parts a·X, b·(V − V_bias), c = I_set and the
    # residual; the calibrated a·LIA + b in the browser's dI/dV frame
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
    for j, cf in zip(cols, fits):
        print(f"column {j}: a = {cf.a:.4g} {ua}, b = {sm4file.si_text(cf.b, uI + '/V')}, "
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
        print(f"column {cols[k]}: a = {cf.a:.4g}, b = {cf.b:.4g}")
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
        ax.set_ylabel(f"X = ∫ LIA dV, referenced at the anchor ({uL}·V)")
        ax.set_title(f"integrated raw dI/dV: X(V) − X(V_bias), V_bias = {Vs[i0]:+.3f} V")

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
        ax.plot(cf.a, cf.b, 'o', color='w', mec='k', label=f"minimum: a = {cf.a:.4g}, b = {cf.b:.4g}")
        ax.set_xlabel(f"a ({ua})")
        ax.set_ylabel(f"b ({uI}/V)")
        ax.set_title("least-squares cost S(a, b), anchored at the setpoint (±4 standard errors)")
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
