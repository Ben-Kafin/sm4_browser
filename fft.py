"""
Noise spectrum files (pages of type RHK_PAGE_NOISE_POWER_SPECTRUM): the
spectrum the instrument computed from a channel recorded for a period of time
(one page per channel, 'LIA Current', 'LIA-X (2w) IETS', 'Current',
'Topography'), against frequency, in physical units (the loader's data, in
each page's RHK_Zunits); the browser shows
every channel and the recording's resolution and bounds.
"""

if __package__ in (None, ''):
    # Run as a file (Spyder F5): make the package importable from wherever its folder sits.
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import numpy as np
from sm4_Browser import qtplot as plt
from sm4_Browser import sm4file

FFT_PAGE_TYPE = 'RHK_PAGE_NOISE_POWER_SPECTRUM'


class NoiseSpectrum:
    """
    The noise spectra of one file.  Each channel (self.channels, in page
    order) is a dict: 'label' (the page label), 'freq' (the frequency axis:
    the page's x axis, Hz), 'values' (the spectrum in physical units, the
    loader's data), 'units' (the page's RHK_Zunits), and 'peak_hz' (the
    frequency of the largest value).  attrs holds the first page's attributes
    for the measurement information.
    """

    @classmethod
    def from_pages(cls, pages, name=''):
        pages = list(pages)
        spec = [p for p in pages if p.page_type == FFT_PAGE_TYPE]
        if not spec:
            raise ValueError(f"{name} has no noise spectrum page.")
        self = cls.__new__(cls)
        self.name = name
        self.attrs = dict(spec[0].attrs)
        self.channels = []
        for p in spec:
            if p.data is None:
                raise ValueError(f"{name}: the '{p.label}' page has no usable RHK_Zscale, so it has no physical data.")
            values = np.asarray(p.data, dtype=np.float64).ravel()
            if len(p.x) != len(values):
                raise ValueError(f"{name}: the '{p.label}' page's x axis has {len(p.x)} points, its data {len(values)}.")
            freq = np.array(p.x, dtype=np.float64)
            self.channels.append({'label': p.label, 'freq': freq, 'values': values, 'units': p.units,
                                  'peak_hz': float(freq[int(np.argmax(values))]) if len(values) else float('nan')})
        return self

    @classmethod
    def from_file(cls, path):
        return cls.from_pages(sm4file.load(path).pages, name=os.path.basename(path))

    def info(self):
        """
        The measurement information as [(name, text)]: the frequency range
        and resolution, the length of the recording the spectrum came from
        (one-sided bins n give 2 (n - 1) samples at the stored sample time),
        each channel's largest line, and the page's settings.
        """
        rows = []
        c0 = self.channels[0]
        f = c0['freq']
        step = float(np.median(np.diff(f))) if len(f) > 1 else float('nan')
        rows.append(('frequency range', f"{sm4file.si_text(float(f[0]), 'Hz')} to {sm4file.si_text(float(f[-1]), 'Hz')}, "
                                        f"{len(f)} bins, resolution {sm4file.si_text(step, 'Hz')}"))
        period = self.attrs.get('RHK_Period')
        if period is not None and len(f) > 1:
            n_samples = 2 * (len(f) - 1)
            rows.append(('recording', f"{n_samples} samples at {sm4file.si_text(float(period), 's')}, "
                                      f"{sm4file.si_text(n_samples * float(period), 's')} long"))
        for c in self.channels:
            rows.append((f"{c['label']}: largest line", sm4file.si_text(c['peak_hz'], 'Hz')))
        rows.extend(sm4file.measurement_info(self.attrs))
        return rows

    def draw(self, ax):
        """Every channel's spectrum against frequency (Hz) in its units on `ax`, with a legend."""
        for c in self.channels:
            ax.plot(c['freq'], c['values'], lw=0.6, label=f"{c['label']} ({c['units']})" if c['units'] else c['label'])
        ax.set_xlabel("frequency (Hz)")
        units = sorted({c['units'] for c in self.channels if c['units']})
        ax.set_ylabel(f"spectrum ({', '.join(units) if units else 'no units recorded'})")
        ax.legend(loc='best', fontsize=7, frameon=False)

    def show(self, title=None):
        """The spectra in their own figure (draw), with the measurement information printed."""
        fig, ax = plt.subplots(figsize=(10, 5))
        self.draw(ax)
        ax.set_title(title or self.name, fontsize=9)
        for name, text in self.info():
            print(f"{name}: {text}")
        plt.show()
        return fig


if __name__ == '__main__':
    path = r'experiment_dir/file.sm4'
    spectrum = NoiseSpectrum.from_file(path)
    spectrum.show()
