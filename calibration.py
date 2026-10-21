"""
Calibration of the lock-in channels against the co-acquired I(V) current:
the per-curve gain-plus-offset regression referenced at the sample nearest
0 V (the method in use before 2026-10-16), and its double-derivative
counterpart for the IETS (second-harmonic) channel.  This file was written
as one of three interchangeable calibration files (calibration_zeroref,
this one; the free-intercept calibration_gainoffset; the scale-only
calibration_stretch), of which the one renamed to calibration.py is the
package's calibration; the repository carries only this one, and the
other two methods appear in the __main__ block for comparison.  The
contract they share: calibrate and calibrate_iets return (calibrated,
fits), every fit entry carrying at least a, c, rms, with X / Z, model,
residual, and b where the method has one.  Plain arrays in, columns =
spectra; the pipeline (sts.STS.process_arrays) calls these on the acquired
columns only, after the current has been through the filters and before
the Feenstra division.  Run as a file, the __main__ block draws one file's
calibration (its inputs, the fit in the I(V) frame, the calibrated curve
in the dI/dV frame with the other two methods for comparison, and the cost
surface), one sweep at a time.
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
    One column's dI/dV regression, I(V) − I(V_ref) = a·X(V) + b·V with X
    and I referenced at the sample nearest 0 V (calibrate).  The arrays are
    in the column's original energy order (npts,).
    """
    a: float               # dI/dV scale: I units per V per LIA unit: 1/V when both are in A
    b: float               # dI/dV offset: I units per V
    c: float               # the current at the reference sample I(V_ref), I units (not fitted)
    X: np.ndarray          # ∫ LIA dV referenced at V_ref: X_raw − X_raw(V_ref) (LIA units·V)
    model: np.ndarray      # a·X + b·V + c (I units)
    residual: np.ndarray   # I(V) − model (I units)
    rms: float             # rms of the residual (I units)


class IetsColumnFit(NamedTuple):
    """
    One column's IETS regression, I(V) − I(V_ref) = a·Z(V) + b·V²/2 + c·V
    with Z and I referenced at the sample nearest 0 V (calibrate_iets).
    The arrays are in the column's original energy order (npts,).
    """
    a: float               # d2I/dV2 scale: I units per V² per second-harmonic unit
    b: float               # d2I/dV2 offset: I units per V²
    c: float               # the unknown dI/dV at the reference (I units per V)
    d: float               # the current at the reference sample I(V_ref), I units (not fitted)
    Z: np.ndarray          # the double integral of the second-harmonic signal, both integrations referenced at V_ref (second-harmonic units·V²)
    model: np.ndarray      # a·Z + b·V²/2 + c·V + d (I units)
    residual: np.ndarray   # I(V) − model (I units)
    rms: float             # rms of the residual (I units)


# vmod (the modulation amplitude, V) is accepted for the pipeline's call (STS.process_arrays passes the
# file's RHK_CH1Drive_Amplitude to whichever calibration file is in place) and is not used by this method.
def calibrate(energy, curves, iv, vmod=None):
    """
    Calibrate dI/dV (LIA) curves via per-curve linear regression against
    the co-acquired I(V) current channel.

    Physics
    -------
    The true dI/dV relates to the raw LIA signal by:

        true dI/dV(V) = a · LIA(V) + b

    Integrating both sides from 0 V to V:

        I(V) − I(0) = a · X(V) + b · V

    where X(V) ≡ ∫₀ᵛ LIA(V') dV'  is the cumulative integral of the
    experimental LIA signal (computed via cumulative trapezoidal rule).

    For each spectrum j we minimise the least-squares cost

        S(a, b) = Σₙ [ΔI(Vₙ) − a·X(Vₙ) − b·Vₙ]²

    Setting ∂S/∂a = 0 and ∂S/∂b = 0 gives the normal equations:

        ΣΔI·X = a·ΣX² + b·ΣV·X
        ΣΔI·V = a·ΣX·V + b·ΣV²

    In matrix form:

        M · [a; b] = [ΣΔI·X ; ΣΔI·V]

        M = [[ΣX²,  ΣVX],
             [ΣXV,  ΣV²]]

        [a; b] = M⁻¹ · [ΣΔI·X ; ΣΔI·V]

    The calibrated dI/dV is then  a_j · LIA_j(V) + b_j, which
    simultaneously recovers the scale and offset in a single fit.
    X and ΔI are referenced at the data point nearest 0 V; V is the
    energy axis itself.  (The method in use before 2026-10-16, unchanged;
    when the bias range does not contain 0 V the reference is the sweep
    end nearest 0 V and the b·V term is then not b·(V − V_ref), which
    biases the fit.)

    Parameters
    ----------
    energy : ndarray, shape (npts,)
        The energy axis of the columns (V), any order.
    curves : ndarray, shape (npts, N)
        The LIA data as the loader gives it (physical units, the page's
        RHK_Zunits; never filtered) in the file's column ordering, i.e.
        *before* the spatial-reversal / transpose done for plotting.
    iv : ndarray or None
        Matching I(V) data with the same shape and column ordering, in the
        Current page's units (A): the recorded current, through the
        pipeline's filters when they are on.  None raises ValueError.
    vmod : float or None
        The modulation amplitude (V), accepted for the pipeline's call and
        not used by this method (see the comment above).

    Returns
    -------
    calibrated : ndarray, same shape as *curves*, float64: a_j * LIA_j + b_j
        per column, in I units per V (A/V for pages in A)
    fits : list of ColumnFit, one per column (a, b, c = I(V_ref), X, model,
        residual, rms; the arrays in the original energy ordering)

    A column whose X and V are linearly dependent (for instance an all-zero
    LIA column, X = 0 everywhere) makes M singular and numpy.linalg.solve
    raises LinAlgError; the pipeline passes the acquired columns only.
    """
    if iv is None:
        raise ValueError(
            "No I(V) current channel found in this SM4 file. "
            "Cannot perform linear-regression calibration."
        )
    energy = np.asarray(energy, dtype=np.float64)
    iv_data = iv

    # Sort energy ascending for cumulative integration.
    sort_idx = np.argsort(energy)
    e_sorted = energy[sort_idx]
    idx_0V = int(np.argmin(np.abs(e_sorted)))

    n_curves = curves.shape[1]
    calibrated = np.empty_like(curves, dtype=np.float64)
    fits = []

    for j in range(n_curves):
        # Sort LIA and I(V) to match ascending energy.
        lia_sorted = curves[sort_idx, j]
        iv_sorted = iv_data[sort_idx, j]

        # X(V) = cumulative integral of LIA, referenced to 0 V.
        X_raw = cumulative_trapezoid(lia_sorted, e_sorted, initial=0)
        X = X_raw - X_raw[idx_0V]

        # ΔI(V) = I(V) − I(0 V)
        delta_I = iv_sorted - iv_sorted[idx_0V]

        # Voltage array.
        V = e_sorted

        # Assemble normal equations:  M · [a; b] = rhs
        sum_X2 = np.dot(X, X)
        sum_V2 = np.dot(V, V)
        sum_VX = np.dot(V, X)
        sum_IX = np.dot(delta_I, X)
        sum_IV = np.dot(delta_I, V)

        M = np.array([[sum_X2, sum_VX],
                      [sum_VX, sum_V2]])
        rhs = np.array([sum_IX, sum_IV])

        # Solve for [a, b].
        ab = np.linalg.solve(M, rhs)
        a_j = ab[0]
        b_j = ab[1]

        # Calibrated dI/dV in original energy ordering.
        calibrated[:, j] = a_j * curves[:, j] + b_j

        # The fit entry (the common contract of the calibration files): the referenced integral
        # and the model in the original ordering, c the current at the reference sample.
        X_orig = np.empty_like(X)
        X_orig[sort_idx] = X
        c_j = float(iv_sorted[idx_0V])
        model = a_j * X_orig + b_j * energy + c_j
        residual = iv_data[:, j] - model
        fits.append(ColumnFit(float(a_j), float(b_j), c_j, X_orig, model, residual,
                              float(np.sqrt(np.mean(residual * residual)))))

    return calibrated, fits


# vmod: accepted for the pipeline's call and not used by this method (see calibrate).
def calibrate_iets(energy, curves, iv, vmod=None):
    """
    Calibrate IETS (second-harmonic lock-in) curves via per-curve linear
    regression against the co-acquired I(V) current channel, the
    double-derivative counterpart of calibrate.

    The true d2I/dV2 relates to the raw second-harmonic signal S by

        true d2I/dV2(V) = a · S(V) + b

    Integrating twice from the reference point V_ref (the data point
    nearest 0 V):

        I(V) − I(V_ref) = a · Z(V) + b · V² / 2 + c · V

    where Z(V) is the double cumulative trapezoidal integral of S, both
    integrations referenced at V_ref, and c absorbs the unknown dI/dV at
    the reference.  For each spectrum j the least-squares cost in (a, b,
    c) gives the 3 by 3 normal equations

        M · [a; b; c] = [ΣΔI·Z ; ΣΔI·V²/2 ; ΣΔI·V],
        M = [[ΣZ², ΣZ·V²/2, ΣZ·V], [ΣZ·V²/2, Σ(V²/2)², ΣV²/2·V], [ΣZ·V, ΣV²/2·V, ΣV²]]

    solved with numpy.linalg.solve as calibrate solves its 2 by 2 system.
    The calibrated curve is a_j · S_j(V) + b_j (in the Current page's units
    per V²), in the original energy ordering.  (The method in use before
    2026-10-16, unchanged.)  As in calibrate, V is the energy axis itself,
    not V − V_ref, so a window without 0 V biases the fit.

    Parameters: as calibrate (energy in V, any order; curves the
    second-harmonic page's data as the loader gives it, never filtered; iv
    the recorded current through the pipeline's filters, None raises
    ValueError; vmod accepted and not used).

    Returns
    -------
    calibrated : ndarray, same shape as *curves*, float64
    fits : list of IetsColumnFit, one per column (a, b, c, d = I(V_ref), Z,
        model, residual, rms; the arrays in the original energy ordering)
    """
    if iv is None:
        raise ValueError(
            "No I(V) current channel found in this SM4 file. "
            "Cannot perform linear-regression calibration."
        )
    energy = np.asarray(energy, dtype=np.float64)
    iv_data = iv

    sort_idx = np.argsort(energy)
    e_sorted = energy[sort_idx]
    idx_0V = int(np.argmin(np.abs(e_sorted)))

    n_curves = curves.shape[1]
    calibrated = np.empty_like(curves, dtype=np.float64)
    fits = []

    for j in range(n_curves):
        s_sorted = curves[sort_idx, j]
        iv_sorted = iv_data[sort_idx, j]

        # Y(V) = cumulative integral of S, referenced at V_ref; Z(V) = its integral, referenced too.
        Y_raw = cumulative_trapezoid(s_sorted, e_sorted, initial=0)
        Y = Y_raw - Y_raw[idx_0V]
        Z_raw = cumulative_trapezoid(Y, e_sorted, initial=0)
        Z = Z_raw - Z_raw[idx_0V]

        delta_I = iv_sorted - iv_sorted[idx_0V]

        V = e_sorted
        Q = 0.5 * V * V

        M = np.array([[np.dot(Z, Z), np.dot(Z, Q), np.dot(Z, V)],
                      [np.dot(Q, Z), np.dot(Q, Q), np.dot(Q, V)],
                      [np.dot(V, Z), np.dot(V, Q), np.dot(V, V)]])
        rhs = np.array([np.dot(delta_I, Z), np.dot(delta_I, Q), np.dot(delta_I, V)])

        abc = np.linalg.solve(M, rhs)
        a_j, b_j, c_j = abc[0], abc[1], abc[2]

        calibrated[:, j] = a_j * curves[:, j] + b_j

        # The fit entry: the referenced double integral and the model in the original ordering,
        # d the current at the reference sample.
        Z_orig = np.empty_like(Z)
        Z_orig[sort_idx] = Z
        d_j = float(iv_sorted[idx_0V])
        model = a_j * Z_orig + b_j * 0.5 * energy * energy + c_j * energy + d_j
        residual = iv_data[:, j] - model
        fits.append(IetsColumnFit(float(a_j), float(b_j), float(c_j), d_j, Z_orig, model, residual,
                                  float(np.sqrt(np.mean(residual * residual)))))

    return calibrated, fits


if __name__ == '__main__':
    # One file's dI/dV calibration by this method drawn, one sweep at a time (the file, the starting
    # sweep, and the current filters set here; '< sweep' / 'sweep >' step through the acquired columns
    # in file order, point, line, or map alike).  Row 1, the calibration's inputs: the raw LIA, its
    # referenced integral X, the recorded current (every acquired column faint, the shown sweep bold;
    # the filtered current when the filters are on).  Row 2: the fit in the I(V) frame with its parts
    # a·X, b·V, c = I(V_ref) and the residual; the calibrated a·LIA + b in the browser's dI/dV frame
    # with the numerical derivative of the (filtered) current and the other two methods dashed; the
    # least-squares cost S(a, b) around its minimum.  Parameters of every column printed once; the
    # shown sweep's three methods on every step.
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
    cal, fits = calibrate(V, lia, iv, f.vmod)
    uL = f.units_lia or 'no units'                    # the LIA page's RHK_Zunits (the loader's data are in it)
    uI = f.units_iv or 'no units'                     # the Current page's RHK_Zunits
    ua = "1/V" if uI == uL else f"{uI}/({uL}·V)"      # the units of the stretch a
    pA = 1e12
    has_zero = V.min() <= 0.0 <= V.max()
    n_sweeps = int(cols.size)
    order = np.argsort(V)
    Vs = V[order]
    i0 = int(np.argmin(np.abs(Vs)))
    print(f"{f.name}: {n_sweeps} acquired column(s), bias {V.min():+.3f} to {V.max():+.3f} V"
          + ("" if has_zero else " (0 V outside the range)") + f"; reference sample at {Vs[i0]:+.4f} V"
          + (f"; current filters Hampel {hampel}, SG {sgolay}" if filtered else "; current filters off"))
    for j, cf in zip(cols, fits):
        print(f"column {j}: a = {cf.a:.4g} {ua}, b = {sm4file.si_text(cf.b, uI + '/V')}, "
              f"I(V_ref) = {sm4file.si_text(cf.c, uI)}, rms residual = {sm4file.si_text(cf.rms, uI)}")

    X_all = np.column_stack([c.X for c in fits])
    faint = dict(color='0.6', lw=0.5, alpha=0.5)
    bold = dict(color='C0', lw=1.2)
    to_pA = (lambda y: y * pA, lambda y: y / pA)      # the right-hand axes in pA (a Current page in A)

    def x_from_start(k):
        """The integral of sweep k's LIA from the first sample of the ascending axis (sorted order)."""
        return cumulative_trapezoid(lia[order, k], Vs, initial=0)

    def gainoffset(k):
        """The free-intercept regression of sweep k: I = a·X + b·V + c, 3 by 3 normal equations.  (a, b, c)."""
        A = np.column_stack([x_from_start(k), Vs, np.ones_like(Vs)])
        a, b, c = np.linalg.solve(A.T @ A, A.T @ iv[order, k])
        return float(a), float(b), float(c)

    def stretch(k):
        """The scale-only calibration of sweep k: a the least-squares slope of I on X, c the equal-integral offset.  (a, c)."""
        X = x_from_start(k)
        Is = iv[order, k]
        dX = X - X.mean()
        a = float(np.dot(dX, Is - Is.mean()) / np.dot(dX, dX))
        c = float((np.trapezoid(Is, Vs) - a * np.trapezoid(X, Vs)) / (Vs[-1] - Vs[0]))
        return a, c

    def cost_surface(k):
        """
        This method's least-squares cost S(a, b) = Σ (ΔI − a·X − b·V)² of
        sweep k, X and ΔI referenced at the sample nearest 0 V, on a window
        of ±4 standard errors of a and b around the minimum (from
        rms² · M⁻¹).  (a grid, b grid, S[b, a], S at the fit).
        """
        cf = fits[k]
        Xs = cf.X[order]
        dI = iv[order, k] - iv[order, k][i0]
        M = np.array([[Xs @ Xs, Vs @ Xs], [Vs @ Xs, Vs @ Vs]])
        cov = cf.rms ** 2 * np.linalg.inv(M)
        sa, sb = np.sqrt(abs(cov[0, 0])), np.sqrt(abs(cov[1, 1]))
        ha = 4 * sa if np.isfinite(sa) and sa > 0 else 0.1 * (abs(cf.a) or 1.0)
        hb = 4 * sb if np.isfinite(sb) and sb > 0 else 0.1 * (abs(cf.b) or 1.0)
        aa = np.linspace(cf.a - ha, cf.a + ha, 121)
        bb = np.linspace(cf.b - hb, cf.b + hb, 121)
        S = np.empty((bb.size, aa.size))
        for ia, a_ in enumerate(aa):
            r = (dI - a_ * Xs)[None, :] - bb[:, None] * Vs[None, :]
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
        a_go, b_go, c_go = gainoffset(k)
        a_st, c_st = stretch(k)
        print(f"column {cols[k]}: this method (0 V-referenced) a = {cf.a:.4g}, b = {cf.b:.4g}; "
              f"gain-plus-offset a = {a_go:.4g}, b = {b_go:.4g}, c = {c_go:.4g}; stretch a = {a_st:.4g}, c = {c_st:.4g}")
        aa, bb, S, S_min = cost_surface(k)

        axes = fig.subplots(2, 3)
        state['axes'].extend(axes.flat)
        fig.suptitle(f"{f.name}: 0 V-referenced calibration of column {cols[k]}  (bias {V.min():+.3f} to {V.max():+.3f} V"
                     + ("" if has_zero else ", 0 V outside the range")
                     + (", current filtered" if filtered else "") + ")")

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
        ax.set_ylabel(f"X = ∫ LIA dV, referenced at V_ref ({uL}·V)")
        ax.set_title(f"integrated raw dI/dV: X(V) − X(V_ref), V_ref = {Vs[i0]:+.3f} V")

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
        ax.plot(V, cf.model, color='C3', lw=1.2, label="a·X + b·V + I(V_ref)")
        ax.plot(V, cf.a * cf.X, '--', color='C0', lw=1, label="a·X")
        ax.plot(V, cf.b * V, ':', color='C2', lw=1, label="b·V")
        ax.axhline(cf.c, ls='-.', color='C1', lw=1, label="c = I(V_ref)")
        ax.axvline(Vs[i0], color='0.5', lw=0.8, ls=':')
        ax.set_ylabel(f"I ({uI})")
        ax.set_title("the fit in the I(V) frame: I − I(V_ref) = a·X + b·V")
        ax.legend(loc='upper left', fontsize=8)
        ax.text(0.02, 0.03, f"a = {cf.a:.4g} {ua}\nb = {sm4file.si_text(cf.b, uI + '/V')}\n"
                            f"I(V_ref) = {sm4file.si_text(cf.c, uI)}\nrms = {sm4file.si_text(cf.rms, uI, 3)}",
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
        ax.plot(V, cal[:, k], color='C3', lw=1.3, label="a·LIA + b (this method, 0 V-referenced)")
        ax.plot(V, a_go * lia_k + b_go, '--', color='0.35', lw=1,
                label=f"gain-plus-offset regression (a = {a_go:.3g}, b = {b_go:.3g})")
        ax.plot(V, a_st * lia_k, '--', color='C0', lw=1, label=f"stretch only (a = {a_st:.3g})")
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
        ax.set_title("least-squares cost S(a, b), 0 V-referenced (±4 standard errors)")
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
