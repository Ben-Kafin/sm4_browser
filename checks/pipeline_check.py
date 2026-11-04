"""
The one pipeline path (sts.STS.stages / STS.view, sts.PipelineState) checked on synthetic sweeps,
numpy only, no .sm4 file: the invariants of the stage-2 document, section 3.3.

1. view.shown == process_arrays with the state's arguments, exactly (numpy.array_equal, NaN-aware;
   the areas too), for every setting combination: channel, each filter on / off, denoise,
   calibrate, normalize, divide_iv, iv_zero_sub, iv_broadened, the gate on / off, each lock, the
   energy crop; the refusal of the division without a current in both.
2. normalize off: view.ratio == view.didv / view.ivv wherever both are finite, to float precision
   (the division and the I/V panel are the same array).
3. stages.lia_f == filters.filter_lia(lia, hampel_didv, sgolay_didv) and the same for iv_f: the
   filters run once.
4. denoise off: g is None, step is None and lia_d is lia_f.  Denoise on, calibrate off, normalize
   off, the plain I/V: the ratio with denoise on equals the ratio with denoise off except for the
   term I(0 V) (1/g - 1) of the 0 V subtraction; on sweeps with I(0 V) = 0 exactly, equal to 1e-9
   relative.  Denoise on: stages.step carries the step's own pieces (gap.DenoiseStep): cal is
   calibration.calibrate on the filtered pair with the gate and locks (the step's regression, run
   here once as the check), Isub0 the filtered current with its value at 0 V subtracted,
   gap.gap_factor on Fcal and Isub0 gives stages.g and I_rebuilt again, |Isub0| / I_rebuilt is g
   where rebuilt, and the never-acquired columns are NaN (g 1).
5. The field table: each PipelineState field, when changed, changes the view fields it reaches and
   no other (the one place the toggle semantics are written down).

Three windows (-2.0..-0.8 V without 0 V, +-0.5 V, -0.6..+0.6 V with a sample at 0 V), forward and
reverse sweeps, dI/dV and IETS.  Spyder: F5; or python -m sm4_Browser.checks.pipeline_check.
Exit status 1 on any failure.
"""

if __package__ in (None, ''):
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import contextlib
import io
import itertools
import sys
import numpy as np
from scipy.integrate import cumulative_trapezoid
from sm4_Browser import filters, calibration, feenstra, gap
from sm4_Browser.sts import STS, PipelineState

fails = 0


def check(label, ok, detail=''):
    global fails
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f": {detail}" if detail else ''))
    fails += 0 if ok else 1


def synth(V, Vb, didv_fn, n, rng, jitter=0.07, reverse_every=4, zero_current=False):
    """
    n sweeps on the axis V: LIA = dI/dV / 40 and I its integral from the lowest energy (with
    zero_current, shifted so that I(0 V) = 0 exactly when 0 V is a sample), both times one shared
    per-sample gap factor; every reverse_every-th sweep reverse (its first sample at the far end
    from V_bias).  Returns (lia, iv, (direction, first)).
    """
    lia = np.empty((V.size, n)); iv = np.empty((V.size, n))
    direction = np.ones(n, dtype=int); first = np.zeros(n, dtype=int)
    order = np.argsort(V)
    k0 = int(np.argmin(np.abs(V[order])))
    for j in range(n):
        didv = didv_fn(V) * (1 + 0.03 * rng.standard_normal())
        g = 1 + jitter * rng.standard_normal(V.size)
        I0 = cumulative_trapezoid(didv[order], V[order], initial=0.0)
        I0 = I0 - I0[k0] if zero_current else I0 - 1.8
        I = np.empty_like(I0); I[order] = I0
        lia[:, j] = didv / 40.0 * g + (0.0 if zero_current else 2e-4 * rng.standard_normal(V.size))
        iv[:, j] = I * g
        if reverse_every and j % reverse_every == 1:
            direction[j] = -1; first[j] = int(np.argmax(V)) if Vb < 0 else int(np.argmin(V))
        else:
            first[j] = int(np.argmin(np.abs(V - Vb)))
    return lia, iv, (direction, first)


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


def rel_close(a, b, tol):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        return True
    return float(np.max(np.abs(a[ok] - b[ok]))) <= tol * max(float(np.max(np.abs(b[ok]))), 1e-300)


HP = {'window': 2, 'n_sigma': 3}
SP = {'window_length': 5, 'polyorder': 2}


def run():
    global fails
    fails = 0
    rng = np.random.default_rng(5)
    V1 = np.linspace(-2.0, -0.8, 49); d1 = lambda V: 0.8 + 0.2 * (V + 0.8) / 1.2
    V2 = np.linspace(-0.5, 0.5, 41); d2 = lambda V: 1.0 + 0.3 * V + 0.4 * np.exp(-((V + 0.3) / 0.08) ** 2)
    V3 = np.linspace(0.6, -0.6, 37)             # a descending axis with a sample at 0 V
    cases = [('-2.0..-0.8 V, V_bias -2.0 (no 0 V)', V1, -2.0, d1), ('-0.5..+0.5 V, V_bias +0.5', V2, 0.5, d2),
             ('+0.6..-0.6 V (descending, a sample at 0 V), V_bias -0.6', V3, -0.6, d2)]
    GS = calibration.GateSettings
    N = 8
    for label, V, Vb, dfn in cases:
        print(f"\n== {label} ==")
        lia, iv, sweeps = synth(V, Vb, dfn, N, rng)
        o = np.argsort(V)
        Iset = float(np.interp(Vb, V[o], cumulative_trapezoid(dfn(V)[o], V[o], initial=0.0) - 1.8))
        S = np.empty_like(lia)
        for j in range(N):
            S[o, j] = np.gradient(lia[o, j], V[o])
        acquired = np.ones(N, dtype=bool); acquired[6] = False
        lia[:, 6] = 0.0; iv[:, 6] = 0.0; S[:, 6] = 0.0
        ref = (lia * 1.01, iv * 0.99)
        crop = (float(V[o][4]), float(V[o][-5]))
        data = dict(acquired=acquired, vmod=0.025, V_bias=Vb, I_setpoint=Iset, ref=ref, name='synthetic', sweeps=sweeps)

        # 1. view.shown == process_arrays, every combination
        n_comb, n_bad, n_refused = 0, 0, 0
        # The pipeline rows, the division, the gate and the locks in full (every combination, both
        # channels, with and without a current; the filters off, 0Vsub on, broadened on); the filters,
        # the Feenstra settings and the crop in full around one setting of the rows.
        rows = [(c, None, None, None, None, den, cal, nrm, div, True, True, gon, l0, lv, False, has_iv)
                for c, den, cal, nrm, div, gon, l0, lv, has_iv in itertools.product(
                    ('didv', 'iets'), (False, True), (False, True), (False, True), (False, True), (False, True),
                    (False, True), (False, True), (True, False))]
        rows += [('didv', hp, sp, hd, sd, True, True, True, True, zs, brd, True, False, False, cr, True)
                 for hp, sp, hd, sd, zs, brd, cr in itertools.product(
                     (None, HP), (None, SP), (None, HP), (None, SP), (False, True), (False, True), (False, True))]
        for channel, hp, sp, hd, sd, den, cal, nrm, div, zs, brd, gon, l0, lv, cr, has_iv in rows:
            page = S if channel == 'iets' else lia
            gate = GS(on=gon, lock0=l0, lockv=lv, denoise=den, hampel=(3, 3), sgolay=(5, 2))
            state = PipelineState(channel=channel, hampel=PipelineState.items(hp), sgolay=PipelineState.items(sp),
                                  hampel_didv=PipelineState.items(hd), sgolay_didv=PipelineState.items(sd), gate=gate,
                                  calibrate=cal, normalize=nrm, divide_iv=div, iv_zero_sub=zs, iv_broadened=brd,
                                  iv_broadening=0.15, norm_range=crop if cr else None)
            cur = iv if has_iv else None
            STS._notes_said.clear()
            said = io.StringIO()
            try:
                with contextlib.redirect_stdout(said):            # the earlier interface prints its notes once
                    c, a = STS.process_arrays(V, page, cur, hp, sp, div, state.iv_width, return_areas=True,
                                              channel=channel, iv_zero_sub=zs, normalize=nrm, calibrate=cal,
                                              norm_range=state.norm_range, hampel_didv=hd, sgolay_didv=sd, gate=gate,
                                              **data)
                exc_p = None
            except ValueError as exc:
                c, a, exc_p = None, None, str(exc)
            try:
                v = STS.view(V, page, cur, state, **data)
                s = STS.stages(V, page, cur, state, **data)
                exc_v = None
            except ValueError as exc:
                v, s, exc_v = None, None, str(exc)
            n_comb += 1
            if exc_p is not None or exc_v is not None:
                n_refused += 1
                n_bad += (exc_p != exc_v)
                continue
            n_bad += not (same(v.shown, c) and same(s.areas, a) and said.getvalue().splitlines() == list(v.notes))
        check(f"1. view.shown == process_arrays exactly, areas and console lines too, over {n_comb} setting "
              f"combinations ({n_refused} refused alike)", n_bad == 0, f"{n_bad} differ")

        # 2. normalize off: ratio == didv / ivv
        ok = True
        for brd, zs, cal, den in itertools.product((False, True), (False, True), (False, True), (False, True)):
            st = PipelineState(channel='didv', gate=GS(denoise=den), calibrate=cal, normalize=False, divide_iv=True,
                               iv_zero_sub=zs, iv_broadened=brd, iv_broadening=0.15)
            v = STS.view(V, lia, iv, st, **data)
            with np.errstate(divide='ignore', invalid='ignore'):
                q = v.didv / v.ivv
            fin = np.isfinite(q) & np.isfinite(v.ratio)
            ok &= bool(np.all(q[fin] == v.ratio[fin]))
        check("2. normalize off: view.ratio == view.didv / view.ivv wherever both are finite, to float precision", ok)

        # 3. the filters run once
        st = PipelineState(channel='didv', hampel=PipelineState.items(HP), sgolay=PipelineState.items(SP),
                           hampel_didv=PipelineState.items(SP and HP), sgolay_didv=PipelineState.items(SP),
                           gate=GS(denoise=True), calibrate=True, normalize=True, divide_iv=True)
        s = STS.stages(V, lia, iv, st, **data)
        check("3. stages.lia_f == filter_lia(lia, hampel_didv, sgolay_didv) and iv_f == filter_lia(iv, hampel, sgolay)",
              same(s.lia_f, filters.filter_lia(lia, HP, SP)) and same(s.iv_f, filters.filter_lia(iv, HP, SP)))

        # 4. denoise off: g None, lia_d is lia_f; denoise on: the ratio invariance on sweeps with I(0 V) = 0
        s_off = STS.stages(V, lia, iv, PipelineState(channel='didv', calibrate=True), **data)
        check("4a. denoise off: g is None, step is None and lia_d is lia_f",
              s_off.g is None and s_off.step is None and s_off.lia_d is s_off.lia_f)
        s_on = STS.stages(V, lia, iv, PipelineState(channel='didv', gate=GS(denoise=True), calibrate=True), **data)
        step = s_on.step
        ok = step is not None
        if ok:
            cols = np.flatnonzero(acquired)
            sw = (sweeps[0][cols], sweeps[1][cols])
            cal_ref, fits_ref = calibration.calibrate(V, s_on.lia_f[:, cols], s_on.iv_f[:, cols], 0.025, V_bias=Vb,
                                                      I_setpoint=Iset, gate=GS(denoise=True), sweeps=sw)
            g_ref, kept_ref, reb_ref = gap.gap_factor(V, step.Fcal[:, cols], step.Isub0[:, cols])
            with np.errstate(divide='ignore', invalid='ignore'):
                g_from = np.abs(step.Isub0) / step.I_rebuilt
            fin = np.isfinite(g_from)
            ok &= (same(step.cal[:, cols], cal_ref) and len(step.fits) == cols.size
                   and all(a.a == b.a and a.b == b.b for a, b in zip(step.fits, fits_ref))
                   and same(step.Isub0[:, cols], feenstra.iv_corrected(V, s_on.iv_f[:, cols], zero_sub=True))
                   and same(g_ref, s_on.g[:, cols]) and same(step.g, s_on.g) and same(reb_ref, step.I_rebuilt[:, cols])
                   and same(kept_ref, step.kept[cols])
                   and bool(np.allclose(g_from[fin], s_on.g[fin], rtol=1e-12, atol=0))
                   and bool(np.all(np.isnan(step.cal[:, ~acquired]))) and bool(np.all(s_on.g[:, ~acquired] == 1.0)))
        check("4c. denoise on: stages.step carries cal, fits, Fcal, Isub0, I_rebuilt, g and kept; cal is calibrate on the "
              "filtered pair; gap_factor(Fcal, Isub0) gives stages.g and I_rebuilt; |Isub0| / I_rebuilt is g where rebuilt; "
              "never-acquired columns NaN", ok)
        lia0, iv0, sw0 = synth(V, Vb, dfn, N, rng, zero_current=True)
        d0 = dict(data, sweeps=sw0, acquired=np.ones(N, dtype=bool))
        base = PipelineState(channel='didv', calibrate=False, normalize=False, divide_iv=True, iv_zero_sub=True,
                             iv_broadened=False)
        v_off = STS.view(V, lia0, iv0, base, **d0)
        v_on = STS.view(V, lia0, iv0, base.with_(gate=GS(denoise=True)), **d0)
        if v_on.g is None:
            check("4b. denoise on, calibrate off, normalize off, plain I/V: the ratio equals the ratio with denoise off"
                  " (I(0 V) = 0 exactly)", False, "the denoise step was not applied")
        else:
            recorded = np.abs(v_on.g - 1.0) < 1e-12                    # the points left as recorded, g = 1 (the point at 0 V)
            r_on, r_off = v_on.ratio.copy(), v_off.ratio.copy()
            r_on[recorded] = np.nan; r_off[recorded] = np.nan
            check("4b. denoise on, calibrate off, normalize off, plain I/V: the ratio equals the ratio with denoise off"
                  " at every point not left as recorded (I(0 V) = 0 exactly), to 1e-9 relative", rel_close(r_on, r_off, 1e-9))

        # 5. the field table
        base = PipelineState(channel='didv', hampel=None, sgolay=None, hampel_didv=None, sgolay_didv=None,
                             gate=GS(on=True, lock0=False, lockv=False, denoise=False), calibrate=True, normalize=False,
                             divide_iv=False, iv_zero_sub=True, iv_broadened=False, iv_broadening=0.15, norm_range=None)
        FIELDS = ('didv', 'current', 'ivv', 'ratio', 'shown')
        # (field, new value, the base to change from, the view fields that change)
        with_cal_off = base.with_(calibrate=False)
        table = [
            ('hampel', HP, with_cal_off, {'current', 'ivv', 'ratio'}),
            ('sgolay', SP, with_cal_off, {'current', 'ivv', 'ratio'}),
            ('hampel', HP, base, {'current', 'ivv', 'ratio', 'didv', 'shown'}),      # through the calibration
            ('hampel_didv', HP, base, {'didv', 'ratio', 'shown'}),
            ('sgolay_didv', SP, base, {'didv', 'ratio', 'shown'}),
            ('gate', GS(denoise=True), base, {'didv', 'current', 'ivv', 'ratio', 'shown'}),
            ('gate', GS(on=False), base, {'didv', 'ratio', 'shown'}),
            ('gate', GS(lock0=True), base, {'didv', 'ratio', 'shown'}),
            ('gate', GS(lockv=True), base, {'didv', 'ratio', 'shown'}),
            ('gate', GS(on=False), with_cal_off, set()),
            ('gate', GS(on=False), base.with_(gate=GS(denoise=True)), {'didv', 'ratio', 'shown', 'current', 'ivv'}),
            ('calibrate', False, base, {'didv', 'ratio', 'shown'}),
            ('normalize', True, base, {'didv', 'current', 'ivv', 'ratio', 'shown'}),
            ('divide_iv', True, base, {'shown'}),
            ('iv_zero_sub', False, base, {'current', 'ivv', 'ratio'}),
            ('iv_broadened', True, base, {'ivv', 'ratio'}),
            ('iv_broadening', 0.3, base.with_(iv_broadened=True), {'ivv', 'ratio'}),
            ('norm_range', crop, base.with_(normalize=True), {'didv', 'current', 'ivv', 'ratio', 'shown'}),
            ('norm_range', crop, base, set()),
        ]
        if not (V.min() <= 0.0 <= V.max()):
            # Without 0 V in range the 0 V subtraction has nothing to subtract, and the gate on this window
            # replaces nothing with the Vbias lock alone: those rows are not expected to change anything.
            table = [(f, val, b, (set() if f == 'iv_zero_sub' else fields)) for f, val, b, fields in table]
        bad = []
        for field, value, b, expect in table:
            vb = STS.view(V, lia, iv, b, **data)
            va = STS.view(V, lia, iv, b.with_(**{field: value}), **data)
            changed = {k for k in FIELDS if not same(getattr(vb, k), getattr(va, k))}
            if field == 'gate' and value == GS(lockv=True) and not changed:
                continue                                      # a lock that pins nothing new on these sweeps
            if field == 'gate' and value in (GS(on=False), GS(lock0=True)) and expect and not changed:
                continue                                      # the gate replaced nothing on these sweeps
            if changed != expect:
                bad.append(f"{field}={value!r} from calibrate {b.calibrate}: changed {sorted(changed)}, expected {sorted(expect)}")
        check(f"5. the field table: each state field changes the view fields it reaches and no other ({len(table)} rows)",
              not bad, "; ".join(bad[:3]))
    print(f"\n{fails} failure(s)")
    return fails


if __name__ == '__main__':
    sys.exit(1 if run() else 0)
