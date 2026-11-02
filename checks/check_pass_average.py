"""
The average of a topography's two passes (topo.average_passes, the rule of every averaged
topography: topo images, the Map window's embedded topo and its contour slice, the scan area's
bands, the Topo window's stripes), numpy only, instant:

    python -m sm4_Browser.checks.check_pass_average

Identical passes come out unchanged (minus the minimum); a pass three times taller carries the
same weight as the other (the least-squares share of each pass's standardized shape in the
average is one half each); a pixel finite in one pass only keeps that pass; passes of different
shapes give the forward alone.  With placed positions: a reverse row recorded compressed and
displaced along the line (its positions, as the stripe matching places them) is resampled onto
the forward row's positions, so the average of the same surface equals the forward within the
interpolation's error where the reverse span covers and keeps the forward alone beyond it; a
rotated scan frame gives the same; a reverse row with one placed pixel contributes nothing;
positions of another shape are ignored.  Prints one PASS / FAIL line per point and the verdict;
returns it.
"""
import os
import sys

if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from sm4_Browser.topo import average_passes


def _weights(avg, a, b):
    """The least-squares share of each pass's standardized shape in the average, normalized to sum to one."""
    z = lambda m: ((m - np.nanmean(m)) / np.nanstd(m)).ravel()
    A = np.column_stack([z(a), z(b)])
    w, *_ = np.linalg.lstsq(A, (avg - np.nanmean(avg)).ravel(), rcond=None)
    return w / w.sum()


def run():
    ok = True

    def report(name, cond, detail=''):
        nonlocal ok
        cond = bool(cond)
        ok = ok and cond
        print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ''))

    rng = np.random.default_rng(1)
    n = 48
    ii, jj = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float), indexing='ij')
    s1 = np.sin(2 * np.pi * jj / 12) + 0.4 * np.cos(2 * np.pi * ii / 9)
    s2 = np.cos(2 * np.pi * jj / 7) - 0.3 * np.sin(2 * np.pi * ii / 15)

    # 1. identical passes: unchanged, minus the minimum
    f = s1 + 0.02 * rng.standard_normal((n, n))
    out = average_passes(f, f)
    report("identical passes: unchanged minus the minimum", np.allclose(out, f - f.min()))

    # 2. neither pass dominates: the reverse three times taller, of another shape
    out = average_passes(s1, 3.0 * s2)
    w = _weights(out, s1, s2)
    report("a pass three times taller carries half the weight", np.allclose(w, [0.5, 0.5], atol=0.02), f"weights {np.round(w, 3)}")
    out = average_passes(3.0 * s1, s2)
    w = _weights(out, s1, s2)
    report("... and the other way round", np.allclose(w, [0.5, 0.5], atol=0.02), f"weights {np.round(w, 3)}")

    # 3. a pixel finite in one pass only keeps that pass; different shapes give the forward
    r = s1.copy()
    r[5, 5] = np.nan
    f2 = s1.copy()
    f2[7, 7] = np.nan
    out = average_passes(f2, r)
    report("a NaN in one pass keeps the other pass there", np.isfinite(out[5, 5]) and np.isfinite(out[7, 7]))
    report("passes of different shapes: the forward alone", np.allclose(average_passes(s1, s2[:-1]), s1))

    # 4. placed positions: the reverse row compressed and displaced along the line, the same surface
    def surface(x, y):
        return np.sin(2 * np.pi * x / 11.0) + 0.5 * np.cos(2 * np.pi * y / 13.0)

    pos_f = np.stack([jj, ii], axis=-1)                                  # forward pixel (i, j) at (x = j, y = i)
    xr = 0.8 * jj + 3.0                                                  # the reverse row compressed by 0.8, shifted by 3
    pos_r = np.stack([xr, ii], axis=-1)
    fwd = surface(jj, ii)
    rev = surface(xr, ii)
    out = average_passes(fwd, rev, pos_f, pos_r)
    covered = (jj >= xr.min(axis=1)[:, None]) & (jj <= xr.max(axis=1)[:, None])
    err = np.abs(out - (fwd - fwd.min()))
    report("resampled reverse: the average equals the forward where the reverse span covers",
           np.max(err[covered]) < 0.02, f"max error {np.max(err[covered]):.4f}")
    report("beyond the reverse span the forward alone", np.allclose(out[~covered], (fwd - fwd.min())[~covered]))
    plain = average_passes(fwd, rev)
    report("without positions the compressed reverse would blur the average", np.max(np.abs(plain - (fwd - fwd.min()))) > 0.2)

    # 5. a rotated frame: the same result (the projection follows the row's own direction)
    th = np.deg2rad(31.0)
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    out_rot = average_passes(fwd, rev, pos_f @ Rm.T + 5.0, pos_r @ Rm.T + 5.0)
    report("a rotated frame gives the same average", np.allclose(out_rot, out, equal_nan=True))

    # 6. a reverse row with one placed pixel contributes nothing; positions of another shape are ignored
    pos_r2 = pos_r.copy()
    pos_r2[10, 1:] = np.nan
    out2 = average_passes(fwd, rev, pos_f, pos_r2)
    report("a reverse row with one placed pixel: the forward row alone", np.allclose(out2[10], (fwd - fwd.min())[10]))
    report("positions of another shape are ignored", np.allclose(average_passes(fwd, rev, pos_f[:-1], pos_r[:-1]), plain))
    print("ALL PASS" if ok else "SOME FAILED")
    return ok


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
