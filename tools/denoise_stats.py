"""
The denoise step on one file, off against on, through the pipeline as the browser runs it
(sts.STS.process_arrays with the calibration on, its gate at the defaults, the locks off, no
filters, no Feenstra division, no normalization), on the acquired columns of the dI/dV channel:

- single-sweep roughness: the rms of the second difference of each column over the rms of the
  column, the median over the columns (what the point-to-point noise of one sweep looks like);
- the odd/even noise of the file mean: half the rms difference of the means of the odd and the
  even columns over the rms of the mean, after each column is scaled to unit mean |value| (the
  per-sweep level taken out, as the area normalization does);
- the shape: the median and the spread of the ratio of the denoised mean to the plain mean;
- g itself: the median per-sweep rms of g - 1 and the band left as recorded;

all over the samples outside the band.  The expected result (2026-11-02, the synthetic set and
eight real files): the roughness down by 1.3 to 2.2x on real files (20x on the synthetic), the
mean's noise down 20 to 45 %, the shape within 0.5 % of the plain mean.

Spyder: set `path` below and press F5, or python -m sm4_Browser.tools.denoise_stats <path>.
A .npz of the synthetic set (checks/synthetic_iets) is accepted beside an .sm4.
"""

if __package__ in (None, ''):
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import os
import sys
import numpy as np
from sm4_Browser import calibration, feenstra, gap
from sm4_Browser.sts import STS


def load(path):
    """(energy, lia, iv, acquired, vmod, V_bias, I_setpoint, sweeps, name) of an .sm4 (its spectra) or a synthetic .npz."""
    if path.lower().endswith('.npz'):
        d = np.load(path, allow_pickle=True)
        acq = np.asarray(d['acquired'], dtype=bool) if 'acquired' in d else np.ones(d['lia'].shape[1], dtype=bool)
        return (d['energy'], d['lia'], d['iv'], acq, float(d['vmod']), float(d['V_bias']) if 'V_bias' in d else None,
                float(d['I_setpoint']) if 'I_setpoint' in d else None, None, os.path.basename(path))
    from sm4_Browser import sm4file
    f = sm4file.Sm4File(path)
    s = f.spectra
    return (s.energy, s.lia, s.iv, np.asarray(s.acquired, dtype=bool), getattr(s, 'vmod', None),
            getattr(s, 'V_bias', None), getattr(s, 'I_setpoint', None), s.sweep_pins(), f.name)


def run(energy, lia, iv, acquired, vmod, V_bias, I_setpoint, sweeps, name, denoise):
    """The pipeline on the dI/dV channel with the denoise row off or on; the acquired columns, energies ascending."""
    gate = calibration.GateSettings(denoise=bool(denoise))
    c = STS.process_arrays(energy, lia, iv, None, None, False, 0.0, channel='didv', acquired=acquired,
                           normalize=False, calibrate=True, vmod=vmod, V_bias=V_bias, I_setpoint=I_setpoint,
                           name=name, gate=gate, sweeps=sweeps)
    order = np.argsort(energy)
    return c[order, :][:, acquired]


def band_mask(energy, iv, acquired):
    """The samples outside every acquired column's band (gap.GAP_BAND of its maximum |I - I(0 V)|), energies ascending."""
    order = np.argsort(energy)
    iv0 = feenstra.iv_corrected(energy, np.asarray(iv, dtype=np.float64)[:, acquired], zero_sub=True)[order, :]
    small = np.abs(iv0) < gap.GAP_BAND * np.max(np.abs(iv0), axis=0, keepdims=True)
    return ~np.any(small, axis=1)


def roughness(c, mask):
    d2 = c[2:] - 2 * c[1:-1] + c[:-2]
    m = mask[1:-1]
    return float(np.median(np.sqrt(np.mean(d2[m] ** 2, axis=0)) / np.sqrt(np.mean(c[mask] ** 2, axis=0))))


def odd_even(c, mask):
    if c.shape[1] < 4:
        return float('nan')
    cn = c / np.mean(np.abs(c[mask]), axis=0, keepdims=True)
    a, b = cn[:, 0::2].mean(1), cn[:, 1::2].mean(1)
    return float(np.sqrt(np.mean((a - b)[mask] ** 2)) / 2 / np.sqrt(np.mean(cn.mean(1)[mask] ** 2)))


def check(path):
    energy, lia, iv, acquired, vmod, V_bias, I_setpoint, sweeps, name = load(path)
    if iv is None:
        print(f"{name}: no I(V) current channel; the denoise step needs it.")
        return
    off = run(energy, lia, iv, acquired, vmod, V_bias, I_setpoint, sweeps, name, False)
    on = run(energy, lia, iv, acquired, vmod, V_bias, I_setpoint, sweeps, name, True)
    mask = band_mask(energy, iv, acquired)
    q = on.mean(1)[mask] / off.mean(1)[mask]
    print(f"{name}: {int(acquired.sum())} acquired columns, {len(energy)} samples, {int((~mask).sum())} in the band")
    print(f"  single-sweep roughness: off {roughness(off, mask):.4f}, on {roughness(on, mask):.4f}")
    print(f"  odd/even noise of the mean (per-sweep level taken out): off {odd_even(off, mask):.4f}, on {odd_even(on, mask):.4f}")
    print(f"  shape, denoised mean over plain mean: median {np.median(q):.3f}, spread {np.std(q):.3f}")


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else r'experiment_dir/file.sm4'
    check(path)
