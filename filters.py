"""
The spectrum filters of the pipeline: the Hampel filter (spikes) and the
Savitzky-Golay filter (smoothing), each applied to every column (spectrum)
of an array, and filter_lia, which applies the two in that order when their
parameter dicts are given.  The filters act on the recorded current I(V)
alone, once, wherever the pipeline uses it: the calibration against the
current and the Feenstra division (sts.STS.process_arrays filters the
current first and hands the same filtered current to both; the Feenstra
windows' current panels filter through feenstra.iv_corrected); the
channel's columns (dI/dV or IETS) go through the same two filters under
their own toggles (the dI/dV pair of every filter box).
"""

if __package__ in (None, ''):
    # Run as a file (Spyder F5): make the package importable from wherever its folder sits.
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np
from scipy.signal import savgol_filter


def hampel(curves, window=3, n_sigma=2.5):
    """
    The Hampel filter (a median-based outlier rule) on each column
    (spectrum) of `curves` (npts, N): sample i is compared with the median
    of the samples i - window .. i + window (the window clipped at the
    column's ends); when it lies farther from that median than n_sigma x
    1.4826 x MAD (the median absolute deviation of the same samples from
    their median, scaled to the standard deviation of a Gaussian), it is
    replaced by the median, otherwise kept.  Where the MAD is 0 (every
    neighbour equal) the threshold is 0 and any sample that differs from
    the median is replaced.  `window` is the number of neighbours on each
    side and must be a whole number (ValueError otherwise).  Returns a new
    float64 array of the input's shape; the input is not changed.
    """
    if window != int(window):
        raise ValueError(
            f"Hampel filter 'window' must be a whole number (got {window}). "
            "Window defines the number of neighboring data points on each side "
            "and cannot be fractional."
        )
    filtered = np.zeros_like(curves, dtype=np.float64)
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
    """
    The Savitzky-Golay filter (scipy.signal.savgol_filter, a least-squares
    polynomial of degree `polyorder` fitted over `window_length` samples
    around each sample, evaluated at the sample) on each column (spectrum)
    of `curves` (npts, N), along axis 0.  An even window_length is made odd
    by adding 1; polyorder must be smaller than the window (scipy raises
    otherwise).  Returns a new float array of the input's shape.
    """
    if window_length % 2 == 0:
        window_length += 1
    return savgol_filter(curves, window_length, polyorder, axis=0)


def filter_lia(lia, hampel_filter_params, sgolay_filter_params):
    """
    The filters (each when its parameter dict is given, None = off) on the
    columns of an array (npts, N), in the order Hampel (spikes first),
    Savitzky-Golay: hampel(**hampel_filter_params), then
    sgolay(**sgolay_filter_params).  The name is historical: the pipeline
    (sts.STS.process_arrays, fit.Fit._raw_set_columns, feenstra.iv_corrected)
    calls it on the recorded current I(V) with the I(V) toggles' settings
    and on the dI/dV (or IETS) columns with the dI/dV toggles' (the same
    boxes, each target its own toggles).  Returns a float64 copy (a plain
    copy when both are off); the input is not changed.
    """
    c = np.asarray(lia, dtype=np.float64).copy()
    if hampel_filter_params is not None:
        c = hampel(c, **hampel_filter_params)
    if sgolay_filter_params is not None:
        c = sgolay(c, **sgolay_filter_params)
    return c


if __name__ == '__main__':
    # The filters run on the dI/dV columns of one file as a demonstration of what they do:
    # the largest change per acquired column printed (nothing is drawn).
    from sm4_Browser import sm4file
    path = r'experiment_dir/file.sm4'
    f = sm4file.Sm4File(path)
    raw = f.lia
    out = filter_lia(raw, {'window': 4, 'n_sigma': 3}, {'window_length': 6, 'polyorder': 3})
    for j in np.flatnonzero(f.acquired):
        print(f"column {j}: largest change {np.max(np.abs(out[:, j] - raw[:, j])):.4g} {f.units_lia}")
