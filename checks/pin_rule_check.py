"""
The unshifted pin rule of calibration.calibrate / calibrate_iets (2026-11-03) checked on synthetic
sweeps against the equations it states; no .sm4 file needed.

The rule: a lock's pin holds through the integrated signal and the constant only, never through
the bias term, so the bias axis enters every fit as recorded.  One pin: c = I_p − a·X_p and a, b
the least squares of I − I_p against (X − X_p, V), which with the 0-lock is the original line
tool's calibrate_didv_curves (X and ΔI referenced to the sample nearest 0 V, V as recorded).  Two
pins: a = (I_1 − I_0) / (X_1 − X_0), c = I_0 − a·X_0, b the least squares of what is left on V.
IETS: the same with Z, (V²/2, V) as the bias terms and d the constant.

Checks, on three windows (−2.0..−0.8 V without 0 V, ±0.5 V, −0.6..+0.6 V with a sample at 0 V):
1. 0-lock, gate off: calibrate equals the line tool's calibrate_didv_curves (transcribed here).
2. Vbias lock, gate off: the one-pin equations, forward sweeps (I_setpoint) and reverse sweeps
   (the first acquired sample, its recorded current).
3. Both locks, gate off: the two-pin equations; a reverse sweep whose first sample is the 0-lock
   sample puts both pins on one X and is refused (ValueError), as before this rule.
4. Gate on with a lock and one disturbed sweep: the replaced sweep's c = I_p − a·X_p with the
   applied a and the first pin, b the least squares of the remainder on V.
5. IETS: the one-pin and two-pin equations with Z and d.
6. The per-sweep gain's scatter on the −2.0..−0.8 V window with 7 % shared gap jitter, every lock
   state (what the rule buys on a window that never reaches 0 V).

Spyder: F5; or python -m sm4_Browser.checks.pin_rule_check.  Exit status 1 on any failure.
"""

if __package__ in (None, ''):
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import sys
import numpy as np
from scipy.integrate import cumulative_trapezoid
from sm4_Browser import calibration

fails = 0


def check(label, ok, detail=''):
    global fails
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f": {detail}" if detail else ''))
    fails += 0 if ok else 1


def line_tool(energy, curves, iv_data):
    """calibrate_didv_curves of dIdV_line_topo_1d_new_norm_final_perfected.py, transcribed: the two-parameter
    normal equations with X and ΔI referenced to the sample nearest 0 V and V as recorded."""
    sort_idx = np.argsort(energy)
    e_sorted = energy[sort_idx]
    idx_0V = int(np.argmin(np.abs(e_sorted)))
    calibrated = np.empty_like(curves, dtype=np.float64)
    for j in range(curves.shape[1]):
        lia_sorted = curves[sort_idx, j]
        iv_sorted = iv_data[sort_idx, j]
        X_raw = cumulative_trapezoid(lia_sorted, e_sorted, initial=0)
        X = X_raw - X_raw[idx_0V]
        delta_I = iv_sorted - iv_sorted[idx_0V]
        V = e_sorted
        M = np.array([[np.dot(X, X), np.dot(V, X)], [np.dot(V, X), np.dot(V, V)]])
        rhs = np.array([np.dot(delta_I, X), np.dot(delta_I, V)])
        ab = np.linalg.solve(M, rhs)
        calibrated[:, j] = ab[0] * curves[:, j] + ab[1]
    return calibrated


def synth(V, Vb, didv_fn, n, rng, I_left, jitter=0.07, reverse_every=0, outlier=None):
    """n sweeps on the axis V: LIA = dI/dV / 40 and I its integral, both times one shared per-sample gap factor;
    every reverse_every-th sweep reverse (its first sample at the far end from V_bias); `outlier` a sweep whose
    current is 3x off (the gate's case).  Returns (lia, iv, (direction, first))."""
    lia = np.empty((V.size, n)); iv = np.empty((V.size, n))
    direction = np.ones(n, dtype=int); first = np.zeros(n, dtype=int)
    order = np.argsort(V)
    for j in range(n):
        didv = didv_fn(V) * (1 + 0.03 * rng.standard_normal())
        g = 1 + jitter * rng.standard_normal(V.size)
        I0 = cumulative_trapezoid(didv[order], V[order], initial=0.0) + I_left
        I = np.empty_like(I0); I[order] = I0
        lia[:, j] = didv / 40.0 * g + 2e-4 * rng.standard_normal(V.size)
        iv[:, j] = I * g
        if reverse_every and j % reverse_every == 1:
            direction[j] = -1; first[j] = int(np.argmax(V)) if Vb < 0 else int(np.argmin(V))
        else:
            first[j] = int(np.argmin(np.abs(V - Vb)))
    if outlier is not None:
        iv[:, outlier] = iv[:, outlier] * 3.0
    return lia, iv, (direction, first)


def lstsq(y, cols):
    return np.linalg.lstsq(np.column_stack(cols), y, rcond=None)[0]


def X_of(energy, lia):
    order = np.argsort(energy); X = np.empty_like(lia)
    for j in range(lia.shape[1]):
        X[order, j] = cumulative_trapezoid(lia[order, j], energy[order], initial=0.0)
    return X


def Z_of(energy, s):
    order = np.argsort(energy); Z = np.empty_like(s)
    for j in range(s.shape[1]):
        Y = cumulative_trapezoid(s[order, j], energy[order], initial=0.0)
        Z[order, j] = cumulative_trapezoid(Y, energy[order], initial=0.0)
    return Z


def maxrel(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b) / np.maximum(np.max(np.abs(b)), 1e-300)))


def close(x, y, tol=1e-9):
    return abs(x - y) <= tol * max(abs(y), 1e-300)


def pin_of(j, sweeps, pb, Iset, iv):
    """The Vbias pin of column j: (sample index, current)."""
    if sweeps[0][j] > 0:
        return pb, Iset
    k = int(sweeps[1][j])
    return k, iv[k, j]


def run():
    rng = np.random.default_rng(11)
    V1 = np.linspace(-2.0, -0.8, 121); d1 = lambda V: 0.8 + 0.2 * (V + 0.8) / (-1.2) * -1
    V2 = np.linspace(-0.5, 0.5, 101); d2 = lambda V: 1.0 + 0.3 * V + 0.4 * np.exp(-((V + 0.3) / 0.08) ** 2)
    V3 = np.linspace(-0.6, 0.6, 61)
    I2 = -cumulative_trapezoid(d2(V2), V2, initial=0.0)[50]
    cases = [('-2.0..-0.8 V, V_bias -2.0 (no 0 V)', V1, -2.0, d1, -1.8), ('-0.5..+0.5 V, V_bias +0.5', V2, 0.5, d2, I2),
             ('-0.6..+0.6 V (a sample at 0 V), V_bias -0.6', V3, -0.6, d2, 0.0)]
    GS = calibration.GateSettings
    N = 12
    for label, V, Vb, dfn, I_left in cases:
        print(f"\n== {label} ==")
        lia, iv, sweeps = synth(V, Vb, dfn, N, rng, I_left, reverse_every=4)
        o = np.argsort(V)
        Iset = float(np.interp(Vb, V[o], cumulative_trapezoid(dfn(V)[o], V[o], initial=0.0) + I_left))
        X = X_of(V, lia)
        pb = int(np.argmin(np.abs(V - Vb))); p0 = int(np.argmin(np.abs(V)))

        # 1. 0-lock, gate off: the line tool
        cal, fits = calibration.calibrate(V, lia, iv, 0.025, V_bias=Vb, I_setpoint=Iset, gate=GS(on=False, lock0=True), sweeps=sweeps)
        ref = line_tool(V, lia, iv)
        check("0-lock, gate off: equals the line tool's calibrate_didv_curves", maxrel(cal, ref) < 1e-9, f"max rel diff {maxrel(cal, ref):.1e}")
        check("0-lock: c = I_0 - a X_0 on every column", all(close(iv[p0, j] - cf.a * X[p0, j], cf.c) for j, cf in enumerate(fits)))

        # 2. Vbias lock, gate off
        cal, fits = calibration.calibrate(V, lia, iv, 0.025, V_bias=Vb, I_setpoint=Iset, gate=GS(on=False, lockv=True), sweeps=sweeps)
        ok = True
        for j, cf in enumerate(fits):
            p, Ip = pin_of(j, sweeps, pb, Iset, iv)
            ab = lstsq(iv[:, j] - Ip, [X[:, j] - X[p, j], V])
            ok &= close(ab[0], cf.a) and close(ab[1], cf.b) and close(Ip - cf.a * X[p, j], cf.c) and cf.pins[0][0] == 'Vbias'
        check("Vbias lock, gate off: a, b = lstsq(I - I_p on X - X_p, V), c = I_p - a X_p; forward and reverse sweeps", ok)

        # 3. both locks, gate off
        clash = [j for j in range(N) if sweeps[0][j] < 0 and int(sweeps[1][j]) == p0]
        if clash:
            try:
                calibration.calibrate(V, lia, iv, 0.025, V_bias=Vb, I_setpoint=Iset, gate=GS(on=False, lock0=True, lockv=True), sweeps=sweeps)
                check(f"both locks, a reverse sweep starting at the 0-lock sample (columns {clash}): refused", False)
            except ValueError as exc:
                check(f"both locks, a reverse sweep starting at the 0-lock sample (columns {clash}): refused", True, str(exc)[:60])
        keep = [j for j in range(N) if j not in clash]
        sw = (sweeps[0][keep], sweeps[1][keep])
        cal, fits = calibration.calibrate(V, lia[:, keep], iv[:, keep], 0.025, V_bias=Vb, I_setpoint=Iset, gate=GS(on=False, lock0=True, lockv=True), sweeps=sw)
        ok = True
        for jj, cf in enumerate(fits):
            j = keep[jj]
            p, Ip = pin_of(j, sweeps, pb, Iset, iv)
            a_r = (Ip - iv[p0, j]) / (X[p, j] - X[p0, j]); c_r = iv[p0, j] - a_r * X[p0, j]
            b_r = lstsq(iv[:, j] - a_r * X[:, j] - c_r, [V])[0]
            ok &= close(a_r, cf.a) and close(c_r, cf.c) and close(b_r, cf.b) and np.isnan(cf.se[0]) and np.isnan(cf.se[2]) and np.isfinite(cf.se[1])
        check(f"both locks, gate off ({len(keep)} columns): a = dI/dX of the two pins, c from the 0-pin, b = lstsq on V; se NaN for a and c", ok)

        # 4. gate on, a lock, one disturbed sweep
        lia_o, iv_o, sweeps_o = synth(V, Vb, dfn, N, rng, I_left, outlier=5)
        X_o = X_of(V, lia_o)
        for locks, name in ((dict(lock0=True), '0-lock'), (dict(lockv=True), 'Vbias lock'), (dict(lock0=True, lockv=True), 'both locks')):
            cal, fits = calibration.calibrate(V, lia_o, iv_o, 0.025, V_bias=Vb, I_setpoint=Iset, gate=GS(on=True, **locks), sweeps=sweeps_o)
            rep = [j for j in range(N) if fits[j].replaced]
            ok = len(rep) > 0 and fits[0].gate.ran
            for j in rep:
                cf = fits[j]
                nm, vp, Ip = cf.pins[0]; p = int(np.argmin(np.abs(V - vp)))
                c_r = Ip - cf.a * X_o[p, j]
                ok &= close(c_r, cf.c)
                if cf.replaced == 'a':
                    ok &= close(lstsq(iv_o[:, j] - cf.a * X_o[:, j] - c_r, [V])[0], cf.b)
            check(f"gate on, {name}: replaced {rep} {[fits[j].replaced for j in rep]}: c = I_p - a X_p with the first pin "
                  f"({fits[rep[0]].pins[0][0] if rep else '-'}), b = lstsq on V", ok)

        # 5. IETS
        S = np.empty_like(lia)
        for j in range(N):
            S[o, j] = np.gradient(lia[o, j], V[o])
        Z = Z_of(V, S)
        cal, fits = calibration.calibrate_iets(V, S, iv, 0.025, V_bias=Vb, I_setpoint=Iset, gate=GS(on=False, lockv=True), sweeps=sweeps)
        ok = True
        for j, cf in enumerate(fits):
            p, Ip = pin_of(j, sweeps, pb, Iset, iv)
            abc = lstsq(iv[:, j] - Ip, [Z[:, j] - Z[p, j], 0.5 * V * V, V])
            ok &= np.allclose(abc, [cf.a, cf.b, cf.c], rtol=1e-9, atol=0) and close(Ip - cf.a * Z[p, j], cf.d)
        check("IETS Vbias lock, gate off: a, b, c = lstsq(I - I_p on Z - Z_p, V^2/2, V), d = I_p - a Z_p", ok)
        cal, fits = calibration.calibrate_iets(V, S[:, keep], iv[:, keep], 0.025, V_bias=Vb, I_setpoint=Iset, gate=GS(on=False, lock0=True, lockv=True), sweeps=sw)
        ok = True
        for jj, cf in enumerate(fits):
            j = keep[jj]
            p, Ip = pin_of(j, sweeps, pb, Iset, iv)
            a_r = (Ip - iv[p0, j]) / (Z[p, j] - Z[p0, j]); d_r = iv[p0, j] - a_r * Z[p0, j]
            bc = lstsq(iv[:, j] - a_r * Z[:, j] - d_r, [0.5 * V * V, V])
            ok &= close(a_r, cf.a) and close(d_r, cf.d) and np.allclose(bc, [cf.b, cf.c], rtol=1e-9, atol=0)
        check("IETS both locks, gate off: a = dI/dZ of the two pins, d from the 0-pin, b, c = lstsq on (V^2/2, V)", ok)

    # 6. the gain scatter on the window without 0 V
    print("\n== -2.0..-0.8 V, 7 % shared gap jitter, 300 sweeps: a / a_true, 16th to 84th percentile ==")
    lia, iv, sweeps = synth(V1, -2.0, d1, 300, rng, -1.8)
    for locks, name in ((dict(), 'locks off'), (dict(lock0=True), '0-lock'), (dict(lockv=True), 'Vbias lock'), (dict(lock0=True, lockv=True), 'both')):
        _, fits = calibration.calibrate(V1, lia, iv, 0.025, V_bias=-2.0, I_setpoint=-1.8, gate=GS(on=False, **locks), sweeps=sweeps)
        a = np.array([cf.a for cf in fits]) / 40.0
        print(f"  {name:11s} [{np.percentile(a, 16):6.2f}, {np.percentile(a, 84):6.2f}]")
    print(f"\n{fails} failure(s)")
    return fails


if __name__ == '__main__':
    sys.exit(1 if run() else 0)
