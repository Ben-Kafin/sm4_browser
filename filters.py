"""
The spectrum filters of the pipeline: the Hampel filter (spikes) and the
Savitzky-Golay filter (smoothing), each applied to every column (spectrum)
of an array, and filter_lia, which applies the two in that order when their
parameter dicts are given.  The filters act on the recorded current I(V)
alone, once, wherever the pipeline uses it: the calibration against the
current and the Feenstra division (sts.STS.process_arrays filters the
current first and hands the same filtered current to both; the Feenstra
windows' current panels filter through feenstra.iv_corrected); the
channel's columns (dI/dV or IETS) are never filtered.
"""

if __package__ in (None, ''):
    # Run as a file (Spyder F5): make the package importable from wherever its folder sits.
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np
from scipy.signal import savgol_filter


def hampel(curves, window=3, n_sigma=2.5):
    """Apply a Hampel filter to each column (spectrum) in curves."""
    if window != int(window):
        raise ValueError(
            f"Hampel filter 'window' must be a whole number (got {window}). "
            "Window defines the number of neighboring data points on each side "
            "and cannot be fractional."
        )
    filtered = np.zeros_like(curves)
    for j in range(curves.shape[1]):
        current_curve = curves[:, j]
        filtered_curve = current_curve.copy()
        k = int(window)
        for i in range(len(current_curve)):
            lower = max(0, i - k)
            upper = min(len(current_curve), i + k + 1)
            window_data = current_curve[lower:upper]
            med = np.median(window_data)
            mad = np.median(np.abs(window_data - med))
            threshold = n_sigma * 1.4826 * mad if mad != 0 else 0
            if np.abs(current_curve[i] - med) > threshold:
                filtered_curve[i] = med
        filtered[:, j] = filtered_curve
    return filtered


def sgolay(curves, window_length=5, polyorder=3):
    """Apply a Savitzky-Golay filter to each column (spectrum) in curves (an even window is made odd)."""
    if window_length % 2 == 0:
        window_length += 1
    return savgol_filter(curves, window_length, polyorder, axis=0)


def filter_lia(lia, hampel_filter_params, sgolay_filter_params):
    """
    The filters (each when given) on the channel's columns, in the order
    Hampel (spikes first), Savitzky-Golay.  Returns a float64 copy.
    """
    c = np.asarray(lia, dtype=np.float64).copy()
    if hampel_filter_params is not None:
        c = hampel(c, **hampel_filter_params)
    if sgolay_filter_params is not None:
        c = sgolay(c, **sgolay_filter_params)
    return c


if __name__ == '__main__':
    # The filters on the dI/dV columns of one file: the spike and smoothing
    # changes printed per column (nothing is drawn).
    from sm4_Browser import sm4file
    path = r'experiment_dir/file.sm4'
    f = sm4file.Sm4File(path)
    raw = f.lia
    out = filter_lia(raw, {'window': 4, 'n_sigma': 3}, {'window_length': 6, 'polyorder': 3})
    for j in np.flatnonzero(f.acquired):
        print(f"column {j}: largest change {np.max(np.abs(out[:, j] - raw[:, j])):.4g} {f.units_lia}")
