"""
Several sweeps per pixel (sweeps.py, STS._group_sweeps, Sm4File views) on synthetic grids:
forward + reverse sweeps stored (a) per pixel in one page, the reverse sweep in acquisition order
against the page's axis, (b) on pages of their own, the reverse page's axis running the other
way, and (c) 'paired', the instrument's layout, every sweep in the page's axis order and the
pairing read from the tip height; plus a stopped reverse sweep, a three-sweep file, four sweeps
per pixel, and a plain one-sweep grid (unchanged).  Then single-point files: 20 sweeps in
forward / reverse pairs (the views, the clock, the calibration's pins), three forward sweeps
(no reverse view), and a file without the tip height (taken as forward).

    python -m sm4_Browser.checks.check_sweeps
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

RESULTS = []


def ok(name, cond, detail=''):
    """Record and print one result: PASS or FAIL, the name, the detail in parentheses."""
    RESULTS.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ''))


def make(sweeps, layout, stop_after=None):
    """A 6 x 5 synthetic grid without drift with the given sweeps and layout (synth.Experiment.grid), loaded as Sm4File; returns (file, the console text of the load)."""
    tmp = tempfile.mkdtemp(prefix='sm4sw_')
    exp = synth.Experiment(tmp, v=(0, 0))
    g = exp.grid(synth.T_REF, n_fast=6, n_lines=5, size_m=4e-9, npts=21, sweeps=sweeps, layout=layout,
                 stop_after=stop_after)
    synth.install(exp)
    from sm4_Browser.sm4file import Sm4File
    with contextlib.redirect_stdout(io.StringIO()) as log:
        f = Sm4File(g)
    return f, log.getvalue()


def close_rel(a, b):
    """Whether two curve arrays agree to a relative 1e-6 (the raw curves are ~1e-11 A: np.allclose's absolute tolerance would call everything equal)."""
    return np.allclose(a, b, rtol=1e-6, atol=1e-18)


def make_point(n_sweeps, pairs, heights=True):
    """A synthetic single-point file (synth.Experiment.point) loaded as Sm4File; returns (file, the console text of the load)."""
    tmp = tempfile.mkdtemp(prefix='sm4swp_')
    exp = synth.Experiment(tmp, v=(0, 0))
    p = exp.point(synth.T_REF, xy=(1e-9, 1e-9), n_sweeps=n_sweeps, pairs=pairs, heights=heights)
    synth.install(exp)
    from sm4_Browser.sm4file import Sm4File
    with contextlib.redirect_stdout(io.StringIO()) as log:
        f = Sm4File(p)
    return f, log.getvalue()


def run():
    """The checks of the module docstring, each printed; returns True when all passed."""
    for layout in ('per_pixel', 'pages'):
        f, log = make(('forward', 'reverse'), layout)
        info = f.sweep_info
        ok(f"[{layout}] two sweeps per pixel found", info is not None and info['k'] == 2 and info['n_pix'] == 30,
           log.strip().splitlines()[-1][:160] if log.strip() else '')
        ok(f"[{layout}] the second sweep is the reverse one", info is not None and list(info['dir'][:2]) == [1, -1])
        ok(f"[{layout}] views offered", f.sweep_modes()[:3] == ('forward sweeps', 'reverse sweeps', 'all sweeps'), str(f.sweep_modes()))
        fw, rv = f, f.with_sweeps('reverse sweeps')
        same = np.allclose(fw.spectra.lia, rv.spectra.lia)
        ok(f"[{layout}] the reverse sweep lands on the forward axis (same values)", same and f.n_spectra == 30)
        ok(f"[{layout}] one column per pixel, kept", fw.spectra.n_acquired == 30 and rv.spectra.n_acquired == 30)
        et_f, et_r = fw.energy_times(), rv.energy_times()
        ok(f"[{layout}] forward view's clock runs up the axis, reverse view's down",
           np.all(np.diff(et_f[0]) > 0) and np.all(np.diff(et_r[0]) < 0))
        ok(f"[{layout}] reverse sweep follows the forward sweep at each pixel", et_r[0].min() >= et_f[0].max() - 1e-9)
        tm = fw.grid_timing()
        ok(f"[{layout}] the tip arrives at each pixel with its first sweep", np.allclose(tm['arrive'], tm['ramp_start']))
        av = f.with_sweeps('all sweeps')
        ok(f"[{layout}] the average view averages both sweeps' clocks",
           np.allclose(av.energy_times()[0], 0.5 * (et_f[0] + et_r[0])))
        z = f.spectra_z()
        ok(f"[{layout}] the Topography channel per pixel", z is not None and len(z) == 30 and np.isfinite(z).all())
        c = f.curves(channel='didv', normalize=False, calibrate=True)
        ok(f"[{layout}] the pipeline runs on the view", c.shape == (21, 30) and np.isfinite(c).all())
        # The calibration's forward-sweep reference: none on the forward view, the forward sweeps' mean
        # raw columns on the reverse and all-sweeps views (calibration.calibrate's `ref`, handed over and
        # not used: every sweep is regressed on its own data).
        from sm4_Browser import calibration
        ok(f"[{layout}] the forward view is its own reference", fw.spectra.lia_ref is None and fw.spectra.iv_ref is None)
        ok(f"[{layout}] the reverse and all-sweeps views carry the forward sweeps' columns",
           rv.spectra.lia_ref is not None and np.allclose(rv.spectra.lia_ref, fw.spectra.lia)
           and np.allclose(rv.spectra.iv_ref, fw.spectra.iv) and av.spectra.lia_ref is not None
           and np.allclose(av.spectra.lia_ref, fw.spectra.lia))
        _, fits = calibration.calibrate(rv.energy, rv.spectra.lia, rv.spectra.iv, rv.vmod)
        cr = rv.curves(channel='didv', normalize=False, calibrate=True)
        exp = np.column_stack([cf.a * rv.spectra.lia[:, j] + cf.b for j, cf in enumerate(fits)])
        ok(f"[{layout}] the reverse view is calibrated on its own sweeps", np.allclose(cr, exp))
        try:
            img = f.topo_raw('forward')
            ok(f"[{layout}] the grid's topo pages read", img.shape == (5, 6))
        except Exception as exc:
            ok(f"[{layout}] the grid's topo pages read", False, str(exc))
    # a stopped reverse sweep (per pixel, acquisition order reversed onto the axis)
    f, log = make(('forward', 'reverse'), 'pages', stop_after=17)
    ok("[stopped] views count kept pixels", f.spectra.n_acquired == 17 and f.with_sweeps('reverse sweeps').spectra.n_acquired == 17,
       f"forward {f.spectra.n_acquired}, reverse {f.with_sweeps('reverse sweeps').spectra.n_acquired} of 30")
    # three sweeps
    f, log = make(('forward', 'reverse', 'forward'), 'per_pixel')
    ok("[three] three sweeps, directions +1 -1 +1", f.sweep_info is not None and list(f.sweep_info['dir'][:3]) == [1, -1, 1],
       str(f.sweep_info and list(f.sweep_info['dir'][:3])))
    ok("[three] 'forward' averages sweeps 1 and 3", np.allclose(f.with_sweeps('forward sweeps').spectra.lia,
                                                              f.with_sweeps('sweep 1').spectra.lia))
    # one sweep per pixel: unchanged
    f, log = make(('forward',), 'per_pixel')
    ok("[one sweep] no sweep grouping, nothing changed", f.sweep_info is None and f.spectra_all is None and f.n_sweeps == 1)
    # the instrument's layout: both sweeps stored in the axis order, paired by the tip height
    f, log = make(('forward', 'reverse'), 'paired')
    info = f.sweep_info
    ok("[paired] two sweeps per pixel, the second reverse by the tip height",
       info is not None and info['k'] == 2 and list(info['dir'][:2]) == [1, -1],
       log.strip().splitlines()[-1][:200] if log.strip() else '')
    ok("[paired] the console names the feedback period and the current's report",
       '2 sweeps per feedback period' in log and 'forward sweeps start a median' in log)
    ok("[paired] views offered", f.sweep_modes()[:3] == ('forward sweeps', 'reverse sweeps', 'all sweeps'), str(f.sweep_modes()))
    fw, rv = f, f.with_sweeps('reverse sweeps')
    ok("[paired] the forward view is sweep 1, the reverse view sweep 2, and they differ",
       close_rel(fw.spectra.lia, f.with_sweeps('sweep 1').spectra.lia)
       and close_rel(rv.spectra.lia, f.with_sweeps('sweep 2').spectra.lia)
       and not close_rel(fw.spectra.lia, rv.spectra.lia))
    ok("[paired] the reverse sweep's data stay as stored (not reversed onto the axis)",
       not close_rel(rv.spectra.lia, fw.spectra.lia[::-1]) and np.all(np.asarray(info['acq_desc'])[1::2]))
    et_f, et_r = fw.energy_times(), rv.energy_times()
    ok("[paired] forward view's clock runs up the axis, reverse view's down",
       np.all(np.diff(et_f[0]) > 0) and np.all(np.diff(et_r[0]) < 0))
    ok("[paired] reverse sweep follows the forward sweep at each pixel", et_r[0].min() >= et_f[0].max() - 1e-9)
    rows = dict(f.measurement_info())
    ok("[paired] the information panel's sweeps row", rows.get('sweeps per location', '').startswith('2: 1 forward, 1 reverse'),
       rows.get('sweeps per location', ''))
    c = rv.curves(channel='didv', normalize=False, calibrate=True)
    ok("[paired] the reverse view runs the pipeline", c.shape == (21, 30) and np.isfinite(c).all())
    f, log = make(('forward', 'reverse'), 'paired', stop_after=17)
    ok("[paired, stopped] directions from the acquired pixels; views count kept pixels",
       f.sweep_info is not None and list(f.sweep_info['dir'][:2]) == [1, -1] and f.spectra.n_acquired == 17
       and f.with_sweeps('reverse sweeps').spectra.n_acquired == 17)
    f, log = make(('forward', 'reverse', 'forward', 'reverse'), 'paired')
    ok("[paired, four] one feedback period of four: +1 -1 +1 -1",
       f.sweep_info is not None and list(f.sweep_info['dir'][:4]) == [1, -1, 1, -1], str(f.sweep_info and list(f.sweep_info['dir'][:4])))
    # a single point: 20 sweeps in forward / reverse pairs
    f, log = make_point(20, pairs=True)
    info = f.sweep_info
    ok("[point] 20 sweeps in 10 forward / reverse pairs",
       f.kind == 'point' and f.n_sweeps == 20 and info is not None and list(info['dir'][:4]) == [1, -1, 1, -1]
       and int(np.sum(info['dir'] < 0)) == 10, log.strip().splitlines()[-1][:200] if log.strip() else '')
    ok("[point] the file as loaded holds every sweep (the all-sweeps view)",
       f.sweep_mode == 'all sweeps' and f.n_spectra == 20 and f.with_sweeps('all sweeps') is f)
    fw, rv = f.with_sweeps('forward sweeps'), f.with_sweeps('reverse sweeps')
    ok("[point] the forward view holds sweeps 1, 3, ..., the reverse 2, 4, ...",
       list(fw.point_cols) == list(range(0, 20, 2)) and list(rv.point_cols) == list(range(1, 20, 2))
       and fw.n_spectra == 10 and rv.n_spectra == 10)
    ok("[point] the views hold the sweeps' own data",
       close_rel(fw.spectra.lia, f.spectra.lia[:, 0::2]) and close_rel(rv.spectra.lia, f.spectra.lia[:, 1::2])
       and not close_rel(fw.spectra.lia, rv.spectra.lia))
    pt, pf, pr = f.point_timing(), fw.point_timing(), rv.point_timing()
    ok("[point] one clock for every view (the span spent on all sweeps)",
       pt['dwell'] == pf['dwell'] == pr['dwell'] and np.allclose(pf['start'], pt['start'][0::2])
       and np.allclose(pr['start'], pt['start'][1::2]))
    et = f.energy_times()
    ok("[point] a forward sweep's points run up the axis, a reverse sweep's down, one after the other",
       np.all(np.diff(et[0]) > 0) and np.all(np.diff(et[1]) < 0) and et[1].min() > et[0].max())
    ee = f.energy_edge_times()
    ok("[point] the edge times follow", ee.shape == (20, len(f.energy) + 1) and np.all(np.diff(ee[1]) < 0))
    d, first = f.spectra.sweep_pins()
    ok("[point] the calibration's pins: a reverse sweep starts at the axis's last sample",
       list(d[:2]) == [1, -1] and list(first[:2]) == [0, len(f.energy) - 1])
    rows = dict(f.measurement_info())
    ok("[point] the information panel's sweeps row", rows.get('sweeps per location', '') == '20: 10 forward, 10 reverse',
       rows.get('sweeps per location', ''))
    c = rv.curves(channel='didv', normalize=False, calibrate=True)
    ok("[point] the reverse view runs the pipeline", c.shape == (len(f.energy), 10) and np.isfinite(c).all())
    # forward only: every sweep its own feedback period
    f, log = make_point(3, pairs=False)
    ok("[point, forward only] three forward sweeps, no reverse view",
       f.n_sweeps == 3 and list(f.sweep_info['dir'][:3]) == [1, 1, 1] and 'reverse sweeps' not in f.sweep_modes()
       and 'every sweep its own feedback period' in log, log.strip().splitlines()[-1][:200] if log.strip() else '')
    try:
        f.with_sweeps('reverse sweeps')
        refused = False
    except ValueError:
        refused = True
    ok("[point, forward only] the reverse view is refused", refused)
    # no 'Topography' page: nothing marks a direction, taken as forward
    f, log = make_point(4, pairs=True, heights=False)
    ok("[point, no tip height] taken as forward with the console line",
       list(f.sweep_info['dir'][:4]) == [1, 1, 1, 1] and 'nothing in the storage marks a direction' in log,
       log.strip().splitlines()[-1][:200] if log.strip() else '')
    print("ALL PASS" if all(RESULTS) else "SOME FAILED")
    return all(RESULTS)


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
