"""
The scan layer's drift transform (scanlayer.affine_matrix) against the per-pixel resampling it
replaces (drift.Drift.corrected_topo), on a synthetic topo with a large known drift (several pixels
over the image), forward and reverse passes.  Also shows the comparison is discriminating: the
uncorrected placement differs from the resampled image by much more.

    python -m sm4_Browser.checks.check_affine
"""
import os
import sys
import tempfile

if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from scipy.ndimage import map_coordinates
from sm4_Browser.checks import synth
from sm4_Browser.drift import PairDrift


def run():
    tmp = tempfile.mkdtemp(prefix='sm4affine_')
    v = np.array([40e-12, -25e-12])                       # m/s: ~ 0.6 nm over a 16 s image (several px)
    exp = synth.Experiment(tmp, v=v)
    path = exp.topo(synth.T_REF, center=(1e-9, -2e-9), size_m=16e-9, n=64, angle=17.0)
    synth.install(exp)
    from sm4_Browser.sm4file import Sm4File
    from sm4_Browser.scanlayer import display_record, affine_matrix
    from sm4_Browser import topo as topo_mod
    from sm4_Browser.topo import processed_topo
    from sm4_Browser.drift import Drift
    none = [False] * len(topo_mod.processor_defaults())
    f = Sm4File(path)
    model = PairDrift(v, f.start_time, 'test', 'known')
    d = Drift.__new__(Drift)
    ok_all = True
    for direction in ('forward', 'reverse'):
        img = processed_topo(f, direction, none)
        ref = d._corrected_pass(f, direction, none, model, topo_mod)
        rec = display_record(f, direction, none)
        q = f.pixel_xy_m(direction).reshape(-1, 2)
        errs = {}
        for name, mdl in (('affine', model), ('uncorrected', None)):
            M = affine_matrix(rec, 1.0, direction, mdl)
            uv = (np.column_stack([q, np.ones(len(q))]) @ np.linalg.inv(M).T)[:, :2]
            j = uv[:, 0] / rec['xscale_m'] + (rec['nx'] - 1) / 2.0
            i = uv[:, 1] / rec['yscale_m'] + (rec['ny'] - 1) / 2.0
            got = map_coordinates(img, [i, j], order=1, mode='constant', cval=np.nan).reshape(img.shape)
            ok = np.isfinite(ref) & np.isfinite(got)
            ok[:4, :] = ok[-4:, :] = ok[:, :4] = ok[:, -4:] = False
            errs[name] = float(np.nanmax(np.abs(ref[ok] - got[ok])) / (np.nanmax(img) - np.nanmin(img)))
        shift_px = float(np.linalg.norm(v) * (f.end_time - f.start_time) / abs(f.xscale_m))
        print(f"{direction}: drift over the image {shift_px:.1f} px; max difference to the resampled image, "
              f"affine {errs['affine']:.4f}, uncorrected {errs['uncorrected']:.4f} (fraction of the height range)")
        ok_all &= errs['affine'] < 0.02 and errs['uncorrected'] > 5 * errs['affine']
    print("PASS" if ok_all else "FAIL")
    return ok_all


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
