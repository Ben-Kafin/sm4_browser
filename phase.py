"""
Lock-in phase rotation files (pages of line type RHK_LINE_PHASE_ROTATE): the
lock-in output recorded while its reference phase was rotated through -180 to
180 deg, one curve per lock-in (the fundamental, 'Phase Rotate', and the second
harmonic, '2w Phase Rotate', when both were recorded), the phase the procedure
set, and a single-phase sinusoid fitted to every curve whose maximum and
minimum are what the browser marks.  PhaseRotate holds one file's curves;
the browser opens such a file as kind 'phase' (sm4file.Sm4File.phase),
draws the curves into its spectra panel (draw) and lists the fit and the
set phase on its info panel (info, through Sm4File.measurement_info); show
opens the same drawing in a figure of its own when the module is run
stand-alone.
"""

if __package__ in (None, ''):
    # Run as a file (Spyder F5): make the package importable from wherever its folder sits.
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import numpy as np
from sm4_Browser import qtplot as plt
from sm4_Browser import sm4file

PHASE_LINE_TYPE = 'RHK_LINE_PHASE_ROTATE'


def fit_sinusoid(deg, values):
    """
    Least-squares fit (numpy.linalg.lstsq) of a single-phase sinusoid,
    values = A cos(theta) + B sin(theta) + C, to the points (deg, values),
    theta the phase in radians.  Returns a dict: 'amplitude' R = hypot(A, B)
    and 'offset' C (in the units of the values), 'max_deg' = atan2(B, A) and
    'min_deg' = max_deg + 180, the phases of the fitted maximum and minimum
    (deg, wrapped into -180 to 180), 'max_value' = C + R, 'min_value' =
    C - R, and 'rms_residual', the standard deviation of the residual
    (equal to its rms, since the fitted offset makes the residual's mean
    zero); None with fewer than three points.
    """
    deg = np.asarray(deg, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if len(deg) < 3:
        return None
    th = np.radians(deg)
    A = np.column_stack([np.cos(th), np.sin(th), np.ones(len(th))])
    (a, b, c), *_ = np.linalg.lstsq(A, values, rcond=None)
    R = float(np.hypot(a, b))
    wrap = lambda x: (x + 180.0) % 360.0 - 180.0
    th_max = float(np.degrees(np.arctan2(b, a)))
    resid = values - A @ np.array([a, b, c])
    return {'amplitude': R, 'offset': float(c), 'max_deg': wrap(th_max), 'min_deg': wrap(th_max + 180.0),
            'max_value': float(c + R), 'min_value': float(c - R), 'rms_residual': float(np.std(resid))}


class PhaseRotate:
    """
    The phase rotation curves of one file.  Each curve (self.curves, in page
    order) is a dict: 'label' (the page label), 'harmonic' (1 for the
    fundamental, 2 for a '2w' page), 'deg' (the phase axis: the page's x
    axis, deg), 'values' (the lock-in output in physical units, the loader's
    data), 'units' (the page's RHK_Zunits), 'acquired' (points recorded; an
    aborted rotation is zero-filled after its last point, a stored 0 that
    converts to the page's RHK_Zoffset), 'set_deg' (the phase the
    procedure set: RHK_Lockin0_PhaseOffset for the fundamental,
    RHK_Lockin1_PhaseOffset for the second harmonic), and 'fit' (fit_sinusoid
    on the acquired points, or None).  xy_m holds the recorded position (1, 2)
    in metres, or (0, 2) without one; attrs the first page's attributes for
    the measurement information.
    """

    @classmethod
    def from_pages(cls, pages, name=''):
        """
        The phase rotation curves of a loaded file's pages (sm4file.load, in
        physical units): every page of line type PHASE_LINE_TYPE in file
        order, each a curve ('2w' in the label marks the second harmonic).
        ValueError without such a page, for a page without a usable
        RHK_Zscale, or when a page's x axis and data differ in length.
        `name` is the file's base name (messages and the figure title).
        """
        pages = list(pages)
        rot = [p for p in pages if p.line_type == PHASE_LINE_TYPE]
        if not rot:
            raise ValueError(f"{name} has no phase rotation page.")
        self = cls.__new__(cls)
        self.name = name
        self.attrs = dict(rot[0].attrs)
        self.curves = []
        for p in rot:
            a = p.attrs
            if p.data is None:
                raise ValueError(f"{name}: the '{p.label}' page has no usable RHK_Zscale, so it has no physical data.")
            values = np.asarray(p.data, dtype=np.float64).ravel()
            if len(p.x) != len(values):
                raise ValueError(f"{name}: the '{p.label}' page's x axis has {len(p.x)} points, its data {len(values)}.")
            deg = np.array(p.x, dtype=np.float64)
            acquired = ~p.unset(values)
            harmonic = 2 if '2w' in p.label.lower() else 1
            key = 'RHK_Lockin1_PhaseOffset' if harmonic == 2 else 'RHK_Lockin0_PhaseOffset'
            set_deg = float(a[key]) if key in a else None
            self.curves.append({'label': p.label, 'harmonic': harmonic, 'deg': deg,
                                'values': values, 'units': p.units, 'acquired': acquired, 'set_deg': set_deg,
                                'fit': fit_sinusoid(deg[acquired], values[acquired])})
        a = rot[0].attrs
        xs = np.asarray(a.get('RHK_SpecDrift_Xcoord', []), dtype=np.float64)
        ys = np.asarray(a.get('RHK_SpecDrift_Ycoord', []), dtype=np.float64)
        self.xy_m = np.array([[xs[0], ys[0]]]) if len(xs) and len(ys) else np.zeros((0, 2))
        return self

    @classmethod
    def from_file(cls, path):
        """The phase rotation curves of a .sm4 file opened here (sm4file.load); from_pages on its pages."""
        return cls.from_pages(sm4file.load(path).pages, name=os.path.basename(path))

    def info(self):
        """
        The measurement information as [(name, text)] for the info panel:
        the phase range (the first curve's axis: first to last point, the
        number of points, the median step), per curve ('fundamental' or
        'second harmonic') the points acquired, the set rotation when
        recorded, and the fit's maximum, minimum, amplitude and rms residual
        when there is one; the recorded position when there is one; the
        first page's settings (sm4file.measurement_info).
        """
        rows = []
        c0 = self.curves[0]
        step = float(np.median(np.abs(np.diff(c0['deg'])))) if len(c0['deg']) > 1 else float('nan')
        rows.append(('phase range', f"{sm4file.si_text(float(c0['deg'][0]), 'deg')} to "
                                    f"{sm4file.si_text(float(c0['deg'][-1]), 'deg')}, {len(c0['deg'])} points, "
                                    f"step {sm4file.si_text(step, 'deg')}"))
        for c in self.curves:
            tag = 'fundamental' if c['harmonic'] == 1 else 'second harmonic'
            rows.append((f'{tag}: points acquired', f"{int(c['acquired'].sum())} of {len(c['deg'])}"))
            if c['set_deg'] is not None:
                rows.append((f'{tag}: set rotation', sm4file.si_text(c['set_deg'], 'deg')))
            f = c['fit']
            u = c['units']
            if f is not None:
                rows.append((f'{tag}: fitted maximum', f"{sm4file.si_text(f['max_deg'], 'deg')} "
                                                       f"({sm4file.si_text(f['max_value'], u)})"))
                rows.append((f'{tag}: fitted minimum', f"{sm4file.si_text(f['min_deg'], 'deg')} "
                                                       f"({sm4file.si_text(f['min_value'], u)})"))
                rows.append((f'{tag}: fitted amplitude', f"{sm4file.si_text(f['amplitude'], u)}, "
                                                         f"rms residual {sm4file.si_text(f['rms_residual'], u)}"))
        if len(self.xy_m):
            rows.append(('position', f"x {sm4file.si_text(float(self.xy_m[0, 0]), 'm')}, "
                                     f"y {sm4file.si_text(float(self.xy_m[0, 1]), 'm')}"))
        rows.extend(sm4file.measurement_info(self.attrs))
        return rows

    def draw(self, ax):
        """
        The curves against the phase (deg) in their units on `ax`: the acquired
        points of every curve, the fitted sinusoid dashed, its maximum and
        minimum marked, and a vertical line at the set rotation.
        """
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for k, c in enumerate(self.curves):
            col = colors[k % len(colors)]
            m = c['acquired']
            ax.plot(c['deg'][m], c['values'][m], color=col, lw=1.0, label=c['label'])
            f = c['fit']
            if f is not None:
                th = np.radians(c['deg'])
                model = f['offset'] + f['amplitude'] * np.cos(th - np.radians(f['max_deg']))
                ax.plot(c['deg'], model, color=col, lw=0.8, ls='--', alpha=0.7)
                ax.plot([f['max_deg']], [f['max_value']], marker='^', color=col, ms=7,
                        label=f"{c['label']}: fitted max {f['max_deg']:.1f} deg")
                ax.plot([f['min_deg']], [f['min_value']], marker='v', color=col, ms=7,
                        label=f"{c['label']}: fitted min {f['min_deg']:.1f} deg")
            if c['set_deg'] is not None:
                ax.axvline(c['set_deg'], color=col, lw=1.0, ls=':', label=f"{c['label']}: set {c['set_deg']:.1f} deg")
        ax.set_xlabel("lock-in phase (deg)")
        units = sorted({c['units'] for c in self.curves if c['units']})
        ax.set_ylabel(f"lock-in output ({', '.join(units) if units else 'no units recorded'})")
        if self.curves:                                  # the recorded phase range (-180 to 180 deg as a rule)
            ax.set_xlim(min(float(np.min(c['deg'])) for c in self.curves),
                        max(float(np.max(c['deg'])) for c in self.curves))
        ax.legend(loc='best', fontsize=7, frameon=False)

    DESIGN = (9, 5)        # the design size (inches): the text zoom of the window is its size relative to this (qtplot.window_zoom)

    def show(self, title=None):
        """
        The curves in a figure of their own, of size DESIGN (draw), titled
        with `title` or the file name, the measurement information (info)
        printed on the console; stand-alone use (the __main__ block), the
        browser draws into its own panel.  Returns the figure.
        """
        fig, ax = plt.subplots(figsize=self.DESIGN)
        self.draw(ax)
        ax.set_title(title or self.name, fontsize=9)
        for name, text in self.info():
            print(f"{name}: {text}")
        plt.show()
        return fig


if __name__ == '__main__':
    path = r'experiment_dir/file.sm4'
    rotation = PhaseRotate.from_file(path)
    rotation.show()
