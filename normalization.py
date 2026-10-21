"""
Unit-area normalization of spectra: normalize (every column divided by the
magnitude of its signed trapezoidal area, so that |area| becomes 1; the
pipeline's last step, sts.STS.process_arrays) and normalize_abs (every
column divided by the area between the curve and zero, the integral of
|curve|; for the current and I/V panels of the Feenstra windows), both
over the whole energy axis or over an energy range (the Line window's
'E range' box; range_mask).  Plain arrays in, columns = spectra; the
energy axis may be in any order (the trapezoids use the values as given,
so an axis running down gives a negative signed area, which the absolute
value removes).
"""

if __package__ in (None, ''):
    # Run as a file (Spyder F5): make the package importable from wherever its folder sits.
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np


def range_mask(energy, energy_range):
    """
    The samples of `energy` inside energy_range (lo, hi), lo <= E <= hi, as
    a boolean mask; every sample with energy_range None.  The energy range
    of the Line window's 'E range' box, as normalize and normalize_abs
    take it.
    """
    energy = np.asarray(energy, dtype=np.float64)
    if energy_range is None:
        return np.ones(len(energy), dtype=bool)
    lo, hi = sorted((float(energy_range[0]), float(energy_range[1])))
    return (energy >= lo) & (energy <= hi)


def normalize(energy, curves, energy_range=None):
    """
    Normalize each spectrum (column) so that the magnitude of its
    integrated area (the trapezoidal rule, numpy.trapezoid) is 1: the
    column divided by |area|, so a column with a negative signed area comes
    back with area -1.  Returns (normalized curves, areas), the per-column
    |area| in the units of the columns times volts, as float64.  A column
    whose area is 0 or NaN (an all-zero column) is returned as it is.  No
    absolute tolerance: the columns are in physical units (a calibrated
    dI/dV in A/V has areas of order 1e-10 A, the lock-in output in A of
    order 1e-12 A V).  With energy_range (lo, hi) the area is taken over
    the samples with lo <= energy <= hi only (the Line window's energy
    range box: the shown range is the normalized range); fewer than two
    such samples give area 0, and the column comes back as it is.
    """
    energy = np.asarray(energy, dtype=np.float64)
    in_range = range_mask(energy, energy_range)
    e_area = energy[in_range]
    norm_curves = np.zeros_like(curves, dtype=np.float64)
    norm_areas = []  # per-column areas (the columns' units times volts), used by "Send to fit"
    for j in range(curves.shape[1]):
        current_curve = curves[:, j]
        area = np.abs(np.trapezoid(current_curve[in_range], x=e_area)) if len(e_area) >= 2 else 0.0
        norm_areas.append(area)
        if np.isfinite(area) and area > 0.0:
            norm_curves[:, j] = current_curve / area
        else:
            norm_curves[:, j] = current_curve
    return norm_curves, np.array(norm_areas, dtype=np.float64)


def normalize_abs(energy, curves, energy_range=None):
    """
    Normalize each column by the area between the curve and zero: the
    trapezoidal integral of |curve| over the energy axis.  Where the
    curve is negative the strip from the curve up to 0 counts, where it
    is positive the strip from 0 up to the curve, so the two sides of a
    0 V crossing add instead of cancelling; for a column that never
    changes sign this is normalize's area.  The Feenstra windows'
    current and I/V panels use it.  Returns (normalized curves, areas),
    the per-column areas in the units of the columns; an all-zero or NaN
    column (area 0 or NaN) is returned as it is.  No absolute tolerance:
    the inputs may be in amps (areas of order 1e-10), and the integral of
    |curve| is zero only for an all-zero column.  energy_range (lo, hi)
    limits the integral to the samples inside that range, as in normalize
    (fewer than two such samples: area 0, the column as it is).
    """
    energy = np.asarray(energy, dtype=np.float64)
    curves = np.asarray(curves, dtype=np.float64)
    in_range = range_mask(energy, energy_range)
    e_area = energy[in_range]
    norm_curves = np.zeros_like(curves, dtype=np.float64)
    areas = []
    for j in range(curves.shape[1]):
        current_curve = curves[:, j]
        area = np.abs(np.trapezoid(np.abs(current_curve[in_range]), x=e_area)) if len(e_area) >= 2 else 0.0
        areas.append(area)
        if np.isfinite(area) and area > 0.0:
            norm_curves[:, j] = current_curve / area
        else:
            norm_curves[:, j] = current_curve
    return norm_curves, np.array(areas, dtype=np.float64)


if __name__ == '__main__':
    # The areas of one file's calibrated dI/dV columns before normalization.
    from sm4_Browser import sm4file, calibration
    path = r'experiment_dir/file.sm4'
    f = sm4file.Sm4File(path)
    cols = np.flatnonzero(f.acquired)
    cal, _ = calibration.calibrate(f.energy, f.lia[:, cols], f.iv[:, cols], f.vmod)
    _, areas = normalize(f.energy, cal)
    for j, a in zip(cols, areas):
        print(f"column {j}: area {a:.4g} (A)")
