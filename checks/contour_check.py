"""
The contour slice's marching squares (contour.iso_segments) and placement (contour.to_frame) on
known surfaces, numpy only, instant:

    python -m sm4_Browser.checks.contour_check

A plane (every crossing at the known column, one segment per row of cells), a cone (every vertex
at the known radius within the linear interpolation's error; the loop closed: every vertex shared
by two segments; nothing on the image edge), the two saddle cases split by the cell's centre (the
segments cut off the right corners), never-acquired pixels (no vertex in a NaN block's cells; a
level under everything draws nothing, so no line runs along the acquired edge), nothing crossing
(empty), and the placement in the frame (the pixel centres of a reversed extent; a constant shift, a
linear shift read between the pixels, a NaN shift as 0, a shift of another shape ignored).  Prints
one [ok] / [FAIL] line per point and the number of failures; returns it.
"""
import os
import sys

if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from sm4_Browser.contour import iso_segments, to_frame


def _vertices(segs):
    return np.asarray(segs).reshape(-1, 2)


def _around(seg, corner):
    """Both vertices of `seg` lie on an edge touching the cell corner (row, col): the segment cuts that corner off."""
    return all(abs(v[0] - corner[0]) < 1e-12 or abs(v[1] - corner[1]) < 1e-12 for v in seg)


def run():
    fails = 0

    def report(name, cond, detail=''):
        nonlocal fails
        cond = bool(cond)
        fails += 0 if cond else 1
        print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ''))

    # 1. a plane rising along the columns: the level 3.25 crosses every row of cells at column 3.25
    ny, nx = 6, 9
    plane = np.tile(np.arange(nx, dtype=float), (ny, 1))
    segs = iso_segments(plane, 3.25)
    report("plane: one segment per row of cells", segs.shape == (ny - 1, 2, 2), str(segs.shape))
    report("plane: every vertex at column 3.25", np.allclose(_vertices(segs)[:, 1], 3.25))
    report("plane: every segment spans one row", np.allclose(np.abs(segs[:, 1, 0] - segs[:, 0, 0]), 1.0))

    # 2. a cone: the level r0 is a circle of radius r0 about the centre
    n = 64
    ii, jj = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float), indexing='ij')
    ci, cj = 31.3, 30.6
    cone = np.hypot(ii - ci, jj - cj)
    r0 = 10.0
    segs = iso_segments(cone, r0)
    v = _vertices(segs)
    vr = np.hypot(v[:, 0] - ci, v[:, 1] - cj)
    report("cone: every vertex within 0.05 px of the radius", len(segs) and np.max(np.abs(vr - r0)) < 0.05,
           f"{len(segs)} segments, max |r - r0| = {np.max(np.abs(vr - r0)):.4f}")
    key = np.round(v, 6)
    _, counts = np.unique(key, axis=0, return_counts=True)
    report("cone: a closed loop (every vertex shared by two segments)", np.all(counts == 2), str(sorted(set(counts))))
    report("cone: no vertex on the image edge", np.all((key > 0) & (key < n - 1)))

    # 3. the saddles: opposite corners below the level; the cell's centre decides which corners the segments cut off
    a = np.array([[0.0, 0.8], [0.8, 0.0]])                 # v0, v2 below 0.5; the centre 0.4 below: they join
    segs = iso_segments(a, 0.5)
    report("saddle v0 v2, centre below: the segments cut off v1 and v3", segs.shape == (2, 2, 2)
           and sorted(_around(sg, (0, 1)) + 2 * _around(sg, (1, 0)) for sg in segs) == [1, 2])
    a = np.array([[0.0, 1.2], [1.2, 0.0]])                 # the centre 0.6 not below: the above corners join
    segs = iso_segments(a, 0.5)
    report("saddle v0 v2, centre above: the segments cut off v0 and v2", segs.shape == (2, 2, 2)
           and sorted(_around(sg, (0, 0)) + 2 * _around(sg, (1, 1)) for sg in segs) == [1, 2])
    a = np.array([[0.8, 0.0], [0.0, 0.8]])                 # v1, v3 below; the centre 0.4 below: they join
    segs = iso_segments(a, 0.5)
    report("saddle v1 v3, centre below: the segments cut off v0 and v2", segs.shape == (2, 2, 2)
           and sorted(_around(sg, (0, 0)) + 2 * _around(sg, (1, 1)) for sg in segs) == [1, 2])
    a = np.array([[1.2, 0.0], [0.0, 1.2]])                 # the centre 0.6 not below: the above corners join
    segs = iso_segments(a, 0.5)
    report("saddle v1 v3, centre above: the segments cut off v1 and v3", segs.shape == (2, 2, 2)
           and sorted(_around(sg, (0, 1)) + 2 * _around(sg, (1, 0)) for sg in segs) == [1, 2])

    # 4. never-acquired pixels: a NaN block across the cone's circle
    holed = cone.copy()
    holed[20:30, 28:40] = np.nan
    segs = iso_segments(holed, r0)
    v = _vertices(segs)
    inside = (v[:, 0] > 19.0) & (v[:, 0] < 30.0) & (v[:, 1] > 27.0) & (v[:, 1] < 40.0)
    report("NaN block: no vertex inside its cells", len(segs) and not inside.any(), f"{len(segs)} segments")
    report("NaN block: fewer segments than the whole cone", len(segs) < len(iso_segments(cone, r0)))
    report("NaN block: a level under everything draws nothing (no line along the acquired edge)",
           len(iso_segments(holed, -1.0)) == 0)
    report("nothing crossing: empty (0, 2, 2)", iso_segments(cone, 1e9).shape == (0, 2, 2))
    report("a one-row image: empty", iso_segments(np.arange(5.0)[None, :], 2.0).shape == (0, 2, 2))
    report("a level at every corner: nothing", len(iso_segments(np.ones((4, 4)), 1.0)) == 0)

    # 5. to_frame: the pixel centres of a reversed extent; the shifts
    segs = np.array([[[0.0, 0.0], [1.0, 2.0]], [[2.5, 1.5], [3.0, 3.0]]])      # (row, col)
    extent = [10.0, 2.0, -4.0, 4.0]                                          # l > r: x reversed; dx = -2, dy = 2
    shape = (4, 4)
    xy = to_frame(segs, extent, shape)
    want = np.stack([10.0 + (segs[..., 1] + 0.5) * -2.0, -4.0 + (segs[..., 0] + 0.5) * 2.0], axis=-1)
    report("to_frame: the pixel centres of a reversed extent", np.allclose(xy, want))
    shift = np.full((4, 4, 2), [0.3, -0.7])
    report("to_frame: a constant shift adds to every vertex",
           np.allclose(to_frame(segs, extent, shape, shift), want + np.array([0.3, -0.7])))
    lin = np.zeros((4, 4, 2))
    lin[..., 0] = np.arange(4.0)[None, :]                                    # the x shift is the column index
    xy2 = to_frame(segs, extent, shape, lin)
    report("to_frame: a linear shift is read at the fractional column", np.allclose(xy2[..., 0] - want[..., 0], segs[..., 1])
           and np.allclose(xy2[..., 1], want[..., 1]))
    nan_shift = shift.copy()
    nan_shift[0, 0] = np.nan
    report("to_frame: a NaN shift counts as 0", np.allclose(to_frame(segs, extent, shape, nan_shift)[0, 0], want[0, 0]))
    report("to_frame: a shift of another shape is ignored", np.allclose(to_frame(segs, extent, shape, np.zeros((3, 3, 2))), want))
    report("to_frame: empty in, empty out", to_frame(np.zeros((0, 2, 2)), extent, shape, shift).shape == (0, 2, 2))
    print(f"{fails} failure(s)")
    return fails


if __name__ == '__main__':
    sys.exit(1 if run() else 0)
