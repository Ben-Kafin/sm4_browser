"""
Several sweeps per pixel (sweeps.py, STS._group_sweeps, Sm4File views) on synthetic grids:
forward + reverse sweeps stored (a) per pixel in one page, the reverse sweep in acquisition order
against the page's axis, and (b) on pages of their own, the reverse page's axis running the other
way; plus a stopped reverse sweep, a three-sweep file, and a plain one-sweep grid (unchanged).

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
    RESULTS.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ''))


def make(sweeps, layout, stop_after=None):
    tmp = tempfile.mkdtemp(prefix='sm4sw_')
    exp = synth.Experiment(tmp, v=(0, 0))
    g = exp.grid(synth.T_REF, n_fast=6, n_lines=5, size_m=4e-9, npts=21, sweeps=sweeps, layout=layout,
                 stop_after=stop_after)
    synth.install(exp)
    from sm4_Browser.sm4file import Sm4File
    with contextlib.redirect_stdout(io.StringIO()) as log:
        f = Sm4File(g)
    return f, log.getvalue()


def run():
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
    print("ALL PASS" if all(RESULTS) else "SOME FAILED")
    return all(RESULTS)


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
