"""
Synthetic experiments for testing without instrument files.

A synthetic experiment is a directory of empty .sm4 placeholder files (so list_sm4_files finds and
orders them by their stamps) and a registry of synthetic pages per file.  install() replaces the
package's two file readers, sm4file.load (the pages with their data) and sm4file.header_entry (the
header scan), with readers of that registry, so every other line of the package runs as it does on
a real file: Page conversion to physical units, STS.from_pages, Sm4File, the drift build, the sites,
the browser.

The surface is known in the absolute frame (surface(x, y), metres), and the drift is a known constant
velocity v (m/s): the content recorded at location p at time t is surface(p - v (t - T_REF)).  Topo
pixels are timed as Sm4File.pixel_times times them; grid spectra carry recorded times
(RHK_SpecDrift_Time) laid out as grid_timing reads them.

    exp = Experiment(tmpdir, v=(2e-12, -1e-12))
    exp.topo(start, center=(0, 0), size_m=20e-9, n=64)
    exp.grid(start, center=(0, 0), n_fast=16, n_lines=16, size_m=8e-9, sweeps=('forward',))
    exp.point(start, xy=(1e-9, 2e-9), n_sweeps=3)
    install(exp)
"""

import os
from datetime import datetime, timedelta
import numpy as np

T_REF = datetime(2024, 1, 1, 10, 0, 0).timestamp()
LATTICE_M = 1.2e-9          # period of the synthetic lattice


class FakeReaderPage:
    """What sm4file.Page reads of spym's page object: attrs (the RHK_ attributes), data (the stored values) and _path (None here)."""

    def __init__(self, attrs, data):
        self.attrs = attrs
        self.data = data
        self._path = None


def default_surface(xy):
    """Height (m) at absolute positions xy (..., 2): a square lattice plus a few large blobs (features for the tracker)."""
    xy = np.asarray(xy, dtype=np.float64)
    x, y = xy[..., 0], xy[..., 1]
    k = 2 * np.pi / LATTICE_M
    z = 0.05e-9 * (np.cos(k * x) + np.cos(k * y))
    rng = np.random.default_rng(7)
    for cx, cy, r, h in zip(rng.uniform(-40e-9, 40e-9, 60), rng.uniform(-40e-9, 40e-9, 60),
                            rng.uniform(1e-9, 3e-9, 60), rng.uniform(0.1e-9, 0.4e-9, 60)):
        z = z + h * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * r * r))
    return z


def _stamp_name(prefix, t_end):
    """A file name '<prefix>_YYYY_MM_DD_HH_MM_SS_mmm.sm4' stamped with the file's end time (s since the epoch, local time)."""
    d = datetime.fromtimestamp(t_end)
    return f"{prefix}_{d:%Y_%m_%d_%H_%M_%S}_{d.microsecond // 1000:03d}.sm4"


def _dt_text(t):
    """A time (s since the epoch) as RHK_DateTime text 'YYYY-MM-DDTHH:MM:SS' (local time)."""
    return datetime.fromtimestamp(t).strftime('%Y-%m-%dT%H:%M:%S')


class Experiment:
    """
    A synthetic experiment: a directory of placeholder .sm4 files and, per file, the list of
    (attrs, stored data) pairs that stand for its pages.  `v` is the drift velocity (m/s), `surface`
    the height function of the absolute frame, `prefix` the file names' stem.  topo, grid and point
    add files; install serves them.  A grid built with stripe_b records the true line starts in
    true_starts (checks/check_stripe.py).
    """

    def __init__(self, directory, v=(0.0, 0.0), surface=default_surface, prefix='Synth_CT'):
        self.directory = directory
        self.v = np.asarray(v, dtype=np.float64)
        self.surface = surface
        self.prefix = prefix
        self.files = {}           # path -> (pages attrs+data list)
        os.makedirs(directory, exist_ok=True)

    # --- helpers ---

    def z_at(self, p, t):
        """The height recorded at absolute positions p (..., 2; metres) at times t (...; s): the surface drifted by v (t - T_REF)."""
        p = np.asarray(p, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        return self.surface(p - self.v * (t - T_REF)[..., None])

    def _add(self, t_end, pages):
        """Register a file ending at t_end: an empty placeholder written under its stamped name, its pages kept; returns the path."""
        path = os.path.join(self.directory, _stamp_name(self.prefix, t_end))
        open(path, 'wb').close()
        self.files[path] = pages
        return path

    @staticmethod
    def _image_page(label, z, center, angle, xs, ys, t0, period, completed=None, zoffset=0.0):
        """
        A topographic image page (attrs, stored) from heights z (ny, nx; metres, NaN never written):
        stored = (z - zoffset) / zscale rounded, a 0 bumped to 1 so that only a never-written pixel
        is 0 (the instrument's convention), with the frame (center, angle, xs, ys), RHK_DateTime t0,
        RHK_Period `period` (the dwell per pixel) and RHK_CompletedLineCount when given.
        """
        ny, nx = z.shape
        zscale = 1e-13
        stored = np.round((np.nan_to_num(z, nan=zoffset) - zoffset) / zscale)
        stored[stored == 0] = 1                       # 0 is the instrument's 'never written'
        stored[~np.isfinite(z)] = 0
        a = {'RHK_PageTypeName': 'RHK_PAGE_TOPOGRAPHIC', 'RHK_PageDataTypeName': 'RHK_DATA_IMAGE',
             'RHK_LineTypeName': 'RHK_LINE_NOT_A_LINE', 'RHK_ScanTypeName': 'RHK_SCAN_RIGHT',
             'RHK_Label': label, 'RHK_Xsize': nx, 'RHK_Ysize': ny, 'RHK_Xoffset': center[0],
             'RHK_Yoffset': center[1], 'RHK_Xscale': xs, 'RHK_Yscale': ys, 'RHK_Angle': angle,
             'RHK_Zscale': zscale, 'RHK_Zoffset': zoffset, 'RHK_Zunits': 'm', 'RHK_Xunits': 'm',
             'RHK_Yunits': 'm', 'RHK_DateTime': _dt_text(t0), 'RHK_Period': period,
             'RHK_Bias': 1.0, 'RHK_Current': 1e-10}
        if completed is not None:
            a['RHK_CompletedLineCount'] = completed
        return (a, stored)

    @staticmethod
    def pixel_xy(center, angle, xs, ys, nx, ny):
        """The absolute position (ny, nx, 2; metres) of every pixel, the convention of Sm4File.pixel_xy_m."""
        jj, ii = np.meshgrid(np.arange(nx, dtype=np.float64), np.arange(ny, dtype=np.float64))
        u = (jj - (nx - 1) / 2.0) * xs
        v = (ii - (ny - 1) / 2.0) * ys
        th = np.deg2rad(angle)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        return np.stack([u, v], axis=-1) @ R.T + np.asarray(center, dtype=np.float64)

    # --- file kinds ---

    def topo(self, start, center=(0.0, 0.0), size_m=20e-9, n=64, angle=0.0, period=2e-3, reverse=True):
        """
        A topography file starting at `start` (s since the epoch): n x n pixels over size_m (metres)
        at `center`, rotated by `angle` (deg), RHK_Period `period` (s per pixel), the forward page
        scanned row by row and, with `reverse`, a reverse page scanned back along each row after
        the forward pass (the timing of Sm4File.pixel_times); every pixel holds the surface drifted
        to its own time (z_at).  The file ends after n lines of n x period x passes.  Returns its path.
        """
        xs, ys = size_m / n, -size_m / n
        p_xy = self.pixel_xy(center, angle, xs, ys, n, n)
        passes = 2 if reverse else 1
        ii, jj = np.meshgrid(np.arange(n, dtype=np.float64), np.arange(n, dtype=np.float64), indexing='ij')
        line_time = n * period * passes
        t_f = start + ii * line_time + jj * period
        pages = [self._image_page('Topography', self.z_at(p_xy, t_f), center, angle, xs, ys, start, period)]
        if reverse:
            t_r = start + ii * line_time + n * period + (n - 1 - jj) * period
            pages.append(self._image_page('Topography', self.z_at(p_xy, t_r), center, angle, xs, ys, start, period))
        t_end = start + n * line_time
        return self._add(t_end, pages)

    def grid(self, start, center=(0.0, 0.0), n_fast=16, n_lines=16, size_m=8e-9, angle=0.0, npts=41,
             v_range=(-1.0, 1.0), ramp_s=0.5, period=2e-3, sweeps=('forward',), layout='per_pixel',
             stop_after=None, stripe_b=(1.0, 1.0)):
        """
        A grid file.  sweeps: the direction of each sweep recorded at every pixel ('forward' runs
        v_range[0] -> v_range[1], 'reverse' the other way); layout 'per_pixel' stores a pixel's
        sweeps as consecutive columns, 'pages' stores each direction on its own set of pages (one
        page per channel per direction), the reverse page's x axis running the other way.
        stop_after: the number of spectra (pixel visits) acquired before the stop (None: all).
        stripe_b: the true (forward, reverse) compression of the lines' topo stripes along the fast
        axis relative to the recorded geometry (1, 1: none), applied to what each pass records.
        """
        xs, ys = size_m / n_fast, -size_m / n_lines
        p_xy = self.pixel_xy(center, angle, xs, ys, n_fast, n_lines)
        k = len(sweeps)
        N_pix = n_fast * n_lines
        n_acq = N_pix if stop_after is None else int(stop_after)
        T_rev = n_fast * period
        # Recorded times: every pixel visit takes k ramps; a return pass after every line.
        t_pix = np.full((n_lines, n_fast), np.nan)
        t = start
        for i in range(n_lines):
            for j in range(n_fast):
                if i * n_fast + j < n_acq:
                    t_pix[i, j] = t
                t += k * ramp_s
            t += T_rev
        t_end = np.nanmax(t_pix) + k * ramp_s
        # The tip's true along-line position under the stripe compression, chained as the tip moves:
        # line i's forward pass runs from its start C_i over b_f x the recorded length, the reverse
        # pass from that forward end E_i back over b_r x the recorded length to the return point
        # R_i = E_i - b_r L, and the next line starts there (C_{i+1} = R_i; line 0 at its recorded
        # start).  self.true_starts records C_i (metres along the fast axis from the recorded start).
        fast = (p_xy[0, 1] - p_xy[0, 0]) / np.linalg.norm(p_xy[0, 1] - p_xy[0, 0])
        s_rec = (p_xy - p_xy[:, :1]) @ fast                      # (ny, nx) along-line distance from the line start
        L = float(s_rec[0, -1])
        bf, br = stripe_b
        C = np.zeros(n_lines)
        for i in range(1, n_lines):
            C[i] = C[i - 1] + bf * L - br * L
        self.true_starts = C
        true_f = p_xy + (C[:, None] + (bf - 1.0) * s_rec)[..., None] * fast
        E = C + bf * L
        true_r = p_xy + ((E[:, None] - br * (L - s_rec)) - s_rec)[..., None] * fast
        z_f = self.z_at(true_f, t_pix)
        t_rev = np.full((n_lines, n_fast), np.nan)
        for i in range(n_lines):
            if (i + 1) * n_fast <= n_acq:
                t_rev[i] = t_pix[i, -1] + k * ramp_s + (n_fast - 1 - np.arange(n_fast)) * period
        z_r = self.z_at(true_r, t_rev)
        pages = [self._image_page('Topography', z_f, center, angle, xs, ys, start, period,
                                  completed=n_acq // n_fast),
                 self._image_page('Topography', z_r, center, angle, xs, ys, start, period,
                                  completed=n_acq // n_fast)]
        # The spectra: dI/dV with a peak whose position follows the local height; I(V) its integral.
        V_fwd = np.linspace(v_range[0], v_range[1], npts)
        zz = z_f.ravel()
        zn = (zz - np.nanmin(zz)) / max(np.nanmax(zz) - np.nanmin(zz), 1e-30)
        zn = np.nan_to_num(zn, nan=0.5)
        peak = -0.5 + zn                                        # V
        didv = (1.0 + 2.0 * np.exp(-((V_fwd[:, None] - peak[None, :]) ** 2) / 0.02)) * 1e-11   # A (lock-in)
        cur = np.cumsum(didv, axis=0) * (V_fwd[1] - V_fwd[0]) * 10.0
        cur -= cur[np.argmin(np.abs(V_fwd)), :][None, :]
        cols = {'forward': (didv, cur, V_fwd), 'reverse': (didv[::-1], cur[::-1], V_fwd[::-1])}
        xy_flat = p_xy.reshape(-1, 2)
        t_flat = t_pix.ravel()

        def spec_page(label, data, V, xs_list, ys_list, times, units):
            zscale = 1e-15
            stored = np.round(np.nan_to_num(data, nan=0.0) / zscale)
            stored[stored == 0] = 1                       # a written sample is never the 'never written' 0
            stored[~np.isfinite(data)] = 0                # never written
            a = {'RHK_PageTypeName': 'RHK_PAGE_INTERACTIVE_SPECTRA', 'RHK_PageDataTypeName': 'RHK_DATA_LINE',
                 'RHK_LineTypeName': 'RHK_LINE_IV_SPECTRUM', 'RHK_ScanTypeName': 'RHK_SCAN_RIGHT',
                 'RHK_Label': label, 'RHK_Xsize': len(V), 'RHK_Ysize': data.shape[1],
                 'RHK_Xoffset': float(V[0]), 'RHK_Xscale': float(V[1] - V[0]) if len(V) > 1 else 1.0, 'RHK_Yoffset': 0.0,
                 'RHK_Yscale': 1.0, 'RHK_Xunits': 'V', 'RHK_Yunits': '', 'RHK_Zunits': units,
                 'RHK_Zscale': zscale, 'RHK_Zoffset': 0.0, 'RHK_Bias': 1.0, 'RHK_Current': 1e-10,
                 'RHK_GridXsize': n_fast, 'RHK_GridYsize': n_lines, 'RHK_DateTime': _dt_text(start),
                 'RHK_Period': 1e-4, 'RHK_CH1Drive_Amplitude': 0.01,
                 'RHK_SpecDrift_Xcoord': list(xs_list), 'RHK_SpecDrift_Ycoord': list(ys_list),
                 'RHK_SpecDrift_Time': list(times)}
            return (a, stored)

        def unset_after(arr, n_cols_done):
            arr = np.array(arr, dtype=np.float64, copy=True)
            arr[:, n_cols_done:] = np.nan                 # never written
            return arr

        spec_pages = []
        if layout == 'per_pixel':
            order = [(pix, s) for pix in range(N_pix) for s in range(k)]
            L, I, X, Y, T = [], [], [], [], []
            for pix, s in order:
                d, c, _ = cols[sweeps[s]]
                L.append(d[:, pix]); I.append(c[:, pix])
                X.append(xy_flat[pix, 0]); Y.append(xy_flat[pix, 1])
                T.append((t_flat[pix] - start + s * ramp_s) if np.isfinite(t_flat[pix]) else 0.0)
            L, I = np.array(L).T, np.array(I).T
            done = n_acq * k
            V = cols[sweeps[0]][2]
            spec_pages += [spec_page('LIA Current', unset_after(L, done), V, X, Y, T, 'A'),
                           spec_page('Current', unset_after(I, done), V, X, Y, T, 'A')]
            topo_ch = np.array([[zz[pix] if np.isfinite(zz[pix]) else 0.0 for pix, s in order]])
            spec_pages.append(spec_page('Topography', topo_ch, V[:1], X, Y, T, 'm'))
        else:
            for s, direction in enumerate(sweeps):
                d, c, V = cols[direction]
                T = [(tt - start + s * ramp_s) if np.isfinite(tt) else 0.0 for tt in t_flat]
                a1 = spec_page('LIA Current', unset_after(d, n_acq), V, xy_flat[:, 0], xy_flat[:, 1], T, 'A')
                a2 = spec_page('Current', unset_after(c, n_acq), V, xy_flat[:, 0], xy_flat[:, 1], T, 'A')
                scan = 'RHK_SCAN_RIGHT' if direction == 'forward' else 'RHK_SCAN_LEFT'
                a1[0]['RHK_ScanTypeName'] = a2[0]['RHK_ScanTypeName'] = scan
                spec_pages += [a1, a2]
            spec_pages.append(spec_page('Topography', np.array([np.nan_to_num(zz)]), V_fwd[:1],
                                        xy_flat[:, 0], xy_flat[:, 1],
                                        [(tt - start) if np.isfinite(tt) else 0.0 for tt in t_flat], 'm'))
        return self._add(t_end, pages + spec_pages)

    def point(self, start, xy=(0.0, 0.0), n_sweeps=3, npts=41, v_range=(-1.0, 1.0), ramp_s=0.5):
        """
        A single-point file with n_sweeps identical sweeps at xy (metres), starting at `start` and
        ending n_sweeps x ramp_s later: a dI/dV with one Gaussian at 0 V and I(V) its integral
        (both pages in A), no recorded sweep times (RHK_SpecDrift_Time 0).  Returns its path.
        """
        V = np.linspace(v_range[0], v_range[1], npts)
        didv = np.tile(((1.0 + np.exp(-(V ** 2) / 0.05)) * 1e-11)[:, None], (1, n_sweeps))
        cur = np.cumsum(didv, axis=0) * (V[1] - V[0]) * 10.0
        cur -= cur[np.argmin(np.abs(V)), :][None, :]

        def page(label, data):
            zscale = 1e-15
            stored = np.round(data / zscale)
            stored[stored == 0] = 1
            a = {'RHK_PageTypeName': 'RHK_PAGE_RAMP_SPECTROSCOPY_RP', 'RHK_PageDataTypeName': 'RHK_DATA_LINE',
                 'RHK_LineTypeName': 'RHK_LINE_IV_SPECTRUM', 'RHK_Label': label, 'RHK_Xsize': npts,
                 'RHK_Ysize': n_sweeps, 'RHK_Xoffset': float(V[0]), 'RHK_Xscale': float(V[1] - V[0]),
                 'RHK_Xunits': 'V', 'RHK_Zunits': 'A', 'RHK_Zscale': zscale, 'RHK_Zoffset': 0.0,
                 'RHK_Bias': 1.0, 'RHK_Current': 1e-10, 'RHK_DateTime': _dt_text(start), 'RHK_Period': 1e-4,
                 'RHK_SpecDrift_Xcoord': [xy[0]] * n_sweeps, 'RHK_SpecDrift_Ycoord': [xy[1]] * n_sweeps,
                 'RHK_SpecDrift_Time': [0.0] * n_sweeps}
            return (a, stored)
        return self._add(start + n_sweeps * ramp_s, [page('LIA Current', didv), page('Current', cur)])


_ORIG = {}


def install(exp):
    """
    Serve the experiment's files through sm4file.load and sm4file.header_entry (other paths: the real
    readers).  The browser's worker processes cannot see this patch, so they are switched off.
    """
    from sm4_Browser import sm4file
    os.environ['SM4_BROWSER_WORKERS'] = '0'
    try:
        from sm4_Browser import browser as _b
        _b._BrowserCore.BG_WORKERS = 0
    except Exception:
        pass
    if not _ORIG:
        _ORIG['load'] = sm4file.load
        _ORIG['header_entry'] = sm4file.header_entry

    def pages_of(path):
        return [sm4file.Page(FakeReaderPage(dict(a), np.array(d, dtype=np.float64))) for a, d in exp.files[path]]

    def load(path):
        if path in exp.files:
            return sm4file.Sm4Data(path, pages_of(path), 'synthetic', None)
        return _ORIG['load'](path)

    def header_entry(path):
        if path in exp.files:
            attrs = [dict(a) for a, _ in exp.files[path]]
            types, geom, line_types = sm4file.header_info_from_attrs(attrs)
            return types, geom, line_types, sm4file.datetime_seconds(attrs[0].get('RHK_DateTime', ''))
        return _ORIG['header_entry'](path)

    sm4file.load = load
    sm4file.header_entry = header_entry


def install_files(directory, files):
    """install() from the files of an experiment (a worker process's initializer: picklable arguments)."""
    exp = Experiment.__new__(Experiment)
    exp.directory, exp.files = directory, files
    install(exp)


def standard(directory, v=(3e-12, -2e-12), grid_sweeps=('forward',), grid_layout='per_pixel', stripe_b=(1.0, 1.0)):
    """A small standard experiment: topo, topo, grid, point, topo (drift v in m/s); returns the Experiment."""
    exp = Experiment(directory, v=v)
    t = T_REF
    exp.topo(t, center=(0.0, 0.0), size_m=24e-9, n=64)
    t += 64 * 64 * 2e-3 * 2 + 30
    exp.topo(t, center=(0.5e-9, 0.0), size_m=24e-9, n=64)
    t += 64 * 64 * 2e-3 * 2 + 30
    exp.grid(t, center=(0.0, 0.0), n_fast=12, n_lines=12, size_m=8e-9, sweeps=grid_sweeps, layout=grid_layout,
             stripe_b=stripe_b)
    t += 12 * 12 * 0.5 * len(grid_sweeps) + 12 * 12 * 2e-3 + 60
    exp.point(t, xy=(2e-9, 1e-9), n_sweeps=3)
    t += 60
    exp.topo(t, center=(0.0, 0.5e-9), size_m=24e-9, n=64)
    return exp
