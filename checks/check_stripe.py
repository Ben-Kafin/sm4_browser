"""
Stripe matching against known truth (synthetic experiments, checks/synth.py):

  1. the forward compression of every line recovered from the matched extrema (true b_f known);
  2. the slow axis carries the velocity only: (corrected - recorded) . slow = -(d(t) - d(t0)) . slow,
     exactly, and zero with drift none;
  3. drift none + stripe match: line starts still offset along the fast axis, factors still measured;
  4. the factors differ between drift settings (the lines overlap other extrema);
  5. a grid half off the reference topo: the lines on it are still placed and fitted from the part
     on the topo;
  6. a grid across two topos: lines matched against both (named in the report).

    python -m sm4_Browser.checks.check_stripe
"""
import io
import os
import sys
import contextlib
import tempfile

if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from sm4_Browser.checks import synth

GRID = dict(n_fast=32, n_lines=12, size_m=8e-9, npts=11, ramp_s=0.25)
RESULTS = []


def ok(name, cond, detail=''):
    RESULTS.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ''))


def experiment(tmp, v, b_true, grid_centre=(0.0, 0.0), second_topo=None):
    exp = synth.Experiment(tmp, v=v)
    t = synth.T_REF
    exp.topo(t, center=(0.0, 0.0), size_m=24e-9, n=128, period=5e-4)
    t += 128 * 128 * 5e-4 * 2 + 20
    if second_topo is not None:
        exp.topo(t, center=second_topo, size_m=14e-9, n=96, period=5e-4)
        t += 96 * 96 * 5e-4 * 2 + 20
    else:
        exp.topo(t, center=(0.3e-9, -0.2e-9), size_m=24e-9, n=128, period=5e-4)
        t += 128 * 128 * 5e-4 * 2 + 20
    g = exp.grid(t, center=grid_centre, stripe_b=b_true, **GRID)
    t += GRID['n_fast'] * GRID['n_lines'] * GRID['ramp_s'] + 60
    exp.topo(t, center=(0.0, 0.3e-9), size_m=24e-9, n=128, period=5e-4)
    return exp, g


def drift_for(exp):
    from sm4_Browser import sm4file
    from sm4_Browser.drift import Drift, DRIFT_PARAMS
    files = sm4file.list_sm4_files(exp.directory)
    heads = sm4file.header_cache(exp.directory, files, progress=False)
    kinds = {p: sm4file.kind_from_types(heads[p]['types'], heads[p]['line_types']) for p in files}
    geoms = {p: heads[p]['geom'] for p in files if kinds[p] == 'topo'}
    loaded = {}

    def load(p):
        if p not in loaded:
            loaded[p] = sm4file.Sm4File(p)
        return loaded[p]
    d = Drift(exp.directory, files, kinds, geoms, dict(DRIFT_PARAMS), load)
    d.build()
    return d, load


def run():
    with contextlib.redirect_stdout(io.StringIO()) as log:
        tmp = tempfile.mkdtemp(prefix='sm4stripe_')
        v = np.array([12e-12, -6e-12])
        exp, gpath = experiment(tmp, v, b_true=(0.92, 1.08))
        synth.install(exp)
        d, load = drift_for(exp)
        g = load(gpath)
        res = {}
        for name, (auto, manual) in (('auto', (True, False)), ('none', (False, False))):
            res[name] = d.result(g, None, True, auto, manual)
    text = log.getvalue()
    a = res['auto']
    ok("stripe result made (auto)", a is not None and a['lines'], a and a['report'][0])
    bf = np.array([ln['b_f'] for ln in a['lines'] if ln['n_fit'] >= 3])
    # The measured factor includes the surface's drift along the line during the pass (the fast axis is
    # the stripes'): b = b_true - v_fast T_f / L.
    fast0, _ = g.grid_axes()
    T_f = GRID['n_fast'] * GRID['ramp_s']
    L0 = (GRID['n_fast'] - 1) * GRID['size_m'] / GRID['n_fast']
    b_exp = 0.92 - float(v @ fast0) * T_f / L0
    ok("1. forward compression recovered from matched extrema", len(bf) >= 6 and abs(np.median(bf) - b_exp) < 0.01,
       f"{len(bf)} lines fitted from >= 3 pairs, median b_f {np.median(bf) if len(bf) else float('nan'):.4f}, "
       f"expected {b_exp:.4f} (true 0.920 with the in-pass drift)")
    # 2. slow axis = velocity only
    fast, slow = g.grid_axes()
    xy = np.asarray(g.spec_xy_m)
    t = g.spectrum_times()
    t0 = float(np.nanmin(t))
    okm = np.all(np.isfinite(a['xy_corr']), axis=1) & np.isfinite(t)
    model = a['drift']
    want = -((model.displacement(t[okm]) - model.displacement(t0)) @ slow)
    got = (a['xy_corr'][okm] - xy[okm]) @ slow
    ok("2. slow axis carries the velocity only (auto)", np.max(np.abs(got - want)) < 1e-15,
       f"max deviation {np.max(np.abs(got - want)):.1e} m")
    n = res['none']
    ok("3a. drift none + stripe: a result is made", n is not None and n['drift'].rule.startswith('none'),
       n and n['source'])
    got_n = (n['xy_corr'][okm] - xy[okm]) @ slow
    ok("3b. drift none: no slow-axis offset", np.max(np.abs(got_n)) < 1e-15, f"max {np.max(np.abs(got_n)):.1e} m")
    starts = [ln['c'] for ln in n['lines']][1:]
    rec_starts = [float((xy[ln['i'] * g.n_fast] - xy[0]) @ fast) for ln in n['lines']][1:]
    off = np.abs(np.array(starts) - np.array(rec_starts))
    ok("3c. drift none: line starts still offset along the fast axis", np.max(off) > 0.05e-10,
       f"largest start offset {np.max(off) * 1e10:.2f} A")
    bf_n = np.array([ln['b_f'] for ln in n['lines']])
    ok("3d. drift none: forward factors still measured", np.sum([ln['b_f_measured'] for ln in n['lines']]) >= 6)
    bf_a = np.array([ln['b_f'] for ln in a['lines']])
    m = min(len(bf_a), len(bf_n))
    ok("4. factors differ between drift settings", m and np.max(np.abs(bf_a[:m] - bf_n[:m])) > 1e-4,
       f"largest difference {np.max(np.abs(bf_a[:m] - bf_n[:m])):.4f}")

    # 7. without drift the line starts (the chained offsets along the fast axis) are recovered
    with contextlib.redirect_stdout(io.StringIO()):
        tmp4 = tempfile.mkdtemp(prefix='sm4stripe4_')
        exp4, g4path = experiment(tmp4, np.zeros(2), b_true=(0.96, 1.03))
        truth = exp4.true_starts
        synth.install(exp4)
        d4, load4 = drift_for(exp4)
        g4 = load4(g4path)
        r4 = d4.result(g4, None, True, False, False)
    fast4, _ = g4.grid_axes()
    xy4 = np.asarray(g4.spec_xy_m)
    est = np.array([ln['c'] - float((xy4[ln['i'] * g4.n_fast] - xy4[0]) @ fast4) for ln in r4['lines']])
    err = np.abs(est - truth[:len(est)])
    step = float(np.linalg.norm(xy4[1] - xy4[0]))
    ok("7. line starts recovered (no drift, chained truth)", np.max(err) < 0.5 * step,
       f"largest error {np.max(err) * 1e10:.2f} A, spectrum step {step * 1e10:.2f} A, true walk "
       f"{(truth[-1] - truth[0]) * 1e10:.1f} A")

    # 5. a grid half off the reference topo
    with contextlib.redirect_stdout(io.StringIO()):
        tmp2 = tempfile.mkdtemp(prefix='sm4stripe2_')
        exp2, g2path = experiment(tmp2, v, b_true=(0.92, 1.08), grid_centre=(11.5e-9, 0.0))
        synth.install(exp2)
        d2, load2 = drift_for(exp2)
        g2 = load2(g2path)
        r2 = d2.result(g2, None, True, True, False)
    fitted = [ln for ln in r2['lines'] if ln['n_fit'] >= 3]
    bf2 = np.array([ln['b_f'] for ln in fitted])
    ok("5. grid half off the topo: lines fitted from the part on it", len(fitted) >= 4 and abs(np.median(bf2) - 0.92) < 0.04,
       f"{len(fitted)} lines, median b_f {np.median(bf2) if len(bf2) else float('nan'):.3f}")

    # 6. a grid across two topos (the second, later and smaller, covers part of the grid)
    with contextlib.redirect_stdout(io.StringIO()):
        tmp3 = tempfile.mkdtemp(prefix='sm4stripe3_')
        exp3, g3path = experiment(tmp3, v, b_true=(0.92, 1.08), grid_centre=(0.0, 0.0), second_topo=(-6e-9, 0.0))
        synth.install(exp3)
        d3, load3 = drift_for(exp3)
        g3 = load3(g3path)
        r3 = d3.result(g3, None, True, True, False)
    names = [os.path.basename(p) for p in sorted(exp3.files)][:2]
    both = [line for line in r3['report'] if names[0] in line and names[1] in line and line.startswith('line ')]
    ok("6. grid across two topos: lines matched against both", len(both) >= 1,
       f"{len(both)} line(s) report both topos; reference: {r3['report'][0][:120]}")
    print("ALL PASS" if all(RESULTS) else "SOME FAILED")
    return all(RESULTS)


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
