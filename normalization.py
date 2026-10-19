"""
Unit-area normalization of spectra: normalize (the signed trapezoidal area
of every column set to 1, the pipeline's last step) and normalize_abs (the
area between the curve and zero, for the current and I/V panels of the
Feenstra windows).  Plain arrays in, columns = spectra.
"""

if __package__ in (None, ''):
    # Run as a file (Spyder F5): make the package importable from wherever its folder sits.
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np


def normalize(energy, curves):
    """
    Normalize each spectrum (column) so its integrated area (via the
    trapezoidal rule) is 1.  Returns (normalized curves, areas), the
    per-column areas in the units of the columns times volts.  A column
    whose area is 0 or NaN (an all-zero column) is returned as it is.  No
    absolute tolerance: the columns are in physical units (a calibrated
    dI/dV in A/V has areas of order 1e-10 A, the lock-in output in A of
    order 1e-12 A V).
    """
    energy = np.asarray(energy, dtype=np.float64)
    norm_curves = np.zeros_like(curves, dtype=np.float64)
    norm_areas = []  # per-column areas (the columns' units times volts), used by "Send to fit"
    for j in range(curves.shape[1]):
        current_curve = curves[:, j]
        area = np.abs(np.trapezoid(current_curve, x=energy))
        norm_areas.append(area)
        if np.isfinite(area) and area > 0.0:
            norm_curves[:, j] = current_curve / area
        else:
            norm_curves[:, j] = current_curve
    return norm_curves, np.array(norm_areas, dtype=np.float64)


def normalize_abs(energy, curves):
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
    |curve| is zero only for an all-zero column.
    """
    energy = np.asarray(energy, dtype=np.float64)
    curves = np.asarray(curves, dtype=np.float64)
    norm_curves = np.zeros_like(curves, dtype=np.float64)
    areas = []
    for j in range(curves.shape[1]):
        current_curve = curves[:, j]
        area = np.abs(np.trapezoid(np.abs(current_curve), x=energy))
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
