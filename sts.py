"""
Scanning tunnelling spectroscopy of one .sm4 file: the dI/dV and IETS channels
with the I(V) page (STS, loaded by page label), and the pipeline driver
(STS.process_arrays / STS.process) that runs the steps of the other modules
in order: filters (filters.py), calibration (calibration.py), the Feenstra
division (feenstra.py), unit-area normalization (normalization.py).
"""

if __package__ in (None, ''):
    # Run as a file (Spyder F5): make the package importable from wherever its folder sits.
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import numpy as np
from sm4_Browser import filters, calibration, normalization, feenstra


def _normalized_label(label):
    """A page label lowercased with everything but letters and digits removed."""
    return ''.join(ch for ch in str(label).lower() if ch.isalnum())


def curve_units(channel, calibrate, normalize, units_lia=None, units_iets=None, units_iv=None):
    """
    The units of a channel's curves out of the pipeline (process_arrays):
    'arb. units' when normalized (unit area); calibrated (a Current page
    present, units_iv given), the Current page's units per volt (dI/dV) or
    per volt squared (IETS); otherwise the channel page's own units
    (RHK_Zunits).  'no units recorded' when the page gives none.
    """
    if normalize:
        return "arb. units"
    if calibrate and units_iv:
        return f"{units_iv}/V\u00b2" if channel == 'iets' else f"{units_iv}/V"
    u = units_iets if channel == 'iets' else units_lia
    return u if u else "no units recorded"


def curve_units_of(entries, channel, calibrate, normalize):
    """
    curve_units over several files (anything with units_lia, units_iets,
    units_iv): the one text when they agree, else the distinct texts joined
    with ', '.
    """
    texts = []
    for f in entries:
        t = curve_units(channel, calibrate, normalize, getattr(f, 'units_lia', None),
                        getattr(f, 'units_iets', None), getattr(f, 'units_iv', None))
        if t not in texts:
            texts.append(t)
    return ", ".join(texts) if texts else curve_units(channel, calibrate, normalize)


class AlignmentError(ValueError):
    """A spectroscopy page whose recorded locations cannot be aligned onto the dI/dV page's (STS.page_perm)."""


class STS:
    """
    One file's scanning tunnelling spectroscopy in physical units, from
    the loader's pages (sm4file.load: every page's data is stored x
    RHK_Zscale + RHK_Zoffset in the page's RHK_Zunits): the dI/dV channel
    (the lock-in output at the modulation frequency, page 'LIA Current'),
    the IETS channel (the second-harmonic output, proportional to d2I/dV2,
    page 'LIA-X (2w) IETS'), and the I(V) page ('Current'), with the energy
    axis (the channel page's x axis), the setpoint, the acquired mask, and
    the recorded positions and times.  Either channel may be absent; the
    I(V) page is optional here and required by the fit (Fit).  Each page is
    found by its label within the file's spectroscopy page type (grid
    files: interactive spectra; point files: ramp spectroscopy), never by
    page index; the I(V) page is the 'Current' page whose RHK_Xunits is V
    (a 'Current' page over another axis is not the I(V)); the three
    locators never find each other's pages, and two pages for one locator
    are an error.  A Current page whose energy axis differs from the
    channel page's is refused (AlignmentError).  Each page's units
    (units_lia, units_iets, units_iv: RHK_Zunits) are kept, with the
    Current page's RHK_Zscale and RHK_Zoffset (zscale_iv, zoffset_iv) as a
    record (the data are already in physical units), the modulation
    amplitude (vmod: RHK_CH1Drive_Amplitude, None when not recorded), and
    the recorded grid size (grid_x, grid_y: RHK_GridXsize, RHK_GridYsize,
    None when not recorded).

    The kept spectra (acquired): a spectrum is kept only when none of its
    columns (the dI/dV or IETS channel, and the Current page when present)
    is never written or ends in never-written samples (kept_mask).  A
    stored 0 is a sample never written, and converts to exactly the page's
    RHK_Zoffset, so the tests compare each page's columns with its own
    offset.  A column never written was never acquired (an aborted grid or
    point file is zero-filled); a column that ends in never-written samples
    is a sweep stopped before its end (the stop zero-fills the remainder).
    Both are left out of every calculation, every plot, and every average,
    and are never drawn; their positions are still recorded.  The pipeline
    keeps a zero column in their place.

    The pipeline (process / process_arrays): the channel's columns through
    the dI/dV Hampel and Savitzky-Golay filters and the recorded current
    through the I(V) ones, each pair once (filters; the two targets have
    their own toggles, the same settings), calibration against that
    current (calibration.calibrate / calibrate_iets: per sweep the least
    squares of the gain, the offset and the constant of integration of
    calibration.py, nothing locked; every sweep on its own data; then the
    gate on the dI/dV gains and offsets as the window's 'gate' row sets
    it, calibration.gate_sweeps), the division by the broadened I/V of
    the same current when
    divide_iv is on (feenstra; dI/dV only), and unit-area normalization
    (normalization).  Every step takes plain arrays, so the Line window
    (one chunk), the fit (its raw sets), the Map, and the averaging windows
    (their files) use the same code; calibration and normalization are
    each switchable.
    """

    DIDV_LABELS = ('liacurrent', 'lia', 'didv')     # normalized label equal to one of these
    IETS_MARKS = ('iets', '2w')                      # normalized label containing one of these
    IV_LABELS = ('current',)                         # normalized label equal to one of these
    CHANNELS = ('didv', 'iets')
    CHANNEL_NAMES = {'didv': 'dI/dV', 'iets': 'IETS'}

    # --- loading ---

    @classmethod
    def _locate(cls, pages, page_type, channel, name):
        """
        Index of the page of `page_type` whose label the locator of `channel`
        ('didv', 'iets', 'iv') accepts, or None; two such pages are an error.
        """
        found = []
        for i, p in enumerate(pages):
            if p.page_type != page_type:
                continue
            lab = _normalized_label(p.label)
            if channel == 'didv':
                hit = lab in cls.DIDV_LABELS and not any(m in lab for m in cls.IETS_MARKS)
            elif channel == 'iets':
                hit = any(m in lab for m in cls.IETS_MARKS)
            else:
                # The I(V): a 'Current' page over the bias axis (RHK_Xunits V) only.
                hit = lab in cls.IV_LABELS and p.xunits == 'V'
            if hit:
                found.append(i)
        if len(found) > 1:
            labels = ', '.join(repr(pages[i].label) for i in found)
            raise ValueError(f"{name}: {len(found)} pages match the {channel} channel ({labels}); "
                             f"a file must hold one.")
        return found[0] if found else None

    @classmethod
    def _locate_all(cls, pages, page_type, channel):
        """Every page index of `page_type` whose label the locator of `channel` accepts (file order)."""
        out = []
        for i, p in enumerate(pages):
            if p.page_type != page_type:
                continue
            lab = _normalized_label(p.label)
            if channel == 'didv':
                hit = lab in cls.DIDV_LABELS and not any(m in lab for m in cls.IETS_MARKS)
            elif channel == 'iets':
                hit = any(m in lab for m in cls.IETS_MARKS)
            else:
                hit = lab in cls.IV_LABELS and p.xunits == 'V'
            if hit:
                out.append(i)
        return out

    @classmethod
    def _from_sweep_pages(cls, pages, page_type, name):
        """
        A file holding one page set per sweep direction (several dI/dV or IETS pages): every set
        (its dI/dV, IETS, and Current pages, paired in file order) read as a file of its own, the
        sets' energy axes compared with the first set's (the same axis: the same direction; the
        axis reversed: the set's columns reversed onto the first axis, its sweeps running the other
        way), the columns merged and grouped into pixels and sweeps (_group_sweeps).  A set whose
        axis is neither is left out with a console line.
        """
        D = cls._locate_all(pages, page_type, 'didv')
        I = cls._locate_all(pages, page_type, 'iets')
        V = cls._locate_all(pages, page_type, 'iv')
        n = max(len(D), len(I))
        for lst, what in ((D, 'dI/dV'), (I, 'IETS'), (V, 'Current')):
            if lst and len(lst) != n:
                raise ValueError(f"{name}: {len(lst)} {what} pages for {n} sweep page set(s); the sets cannot be paired.")
        everything = set(D) | set(I) | set(V)
        sets = []
        for j in range(n):
            keep = {lst[j] for lst in (D, I, V) if lst}
            sub = [p for i, p in enumerate(pages) if i not in everything or i in keep]
            sets.append(cls.from_pages(sub, page_type, name=name, _group=False))
        base = sets[0]
        if base is None:
            return None
        E0 = base.energy
        parts, desc = [base], [np.zeros(base.n_spectra, dtype=bool)]
        notes = []
        for j, sj in enumerate(sets[1:], start=1):
            if sj is None:
                continue
            if len(sj.energy) == len(E0) and np.allclose(sj.energy, E0, rtol=0, atol=1e-9 * max(np.ptp(E0), 1e-12)):
                d = False
            elif len(sj.energy) == len(E0) and np.allclose(sj.energy[::-1], E0, rtol=0,
                                                           atol=1e-9 * max(np.ptp(E0), 1e-12)):
                d = True
                for attr in ('lia', 'iets', 'iv'):
                    a = getattr(sj, attr)
                    if a is not None:
                        setattr(sj, attr, a[::-1, :].copy())
            else:
                print(f"{name}: sweep page set {j + 1}'s energy axis is not the first set's (or its reverse); "
                      f"that set is left out.")
                continue
            parts.append(sj)
            desc.append(np.full(sj.n_spectra, d, dtype=bool))
            notes.append(f"page set {j + 1}: its energy axis runs {'the other way' if d else 'the same way'} "
                         f"as page set 1's")
        if len(parts) == 1:
            base._group_sweeps(np.zeros(base.n_spectra, dtype=bool), one_page=True)
            return base
        s = base
        for attr in ('lia', 'iets', 'iv'):
            arrs = [getattr(pt, attr) for pt in parts]
            setattr(s, attr, None if any(a is None for a in arrs) else np.hstack(arrs))
        s.spec_xy_m = np.vstack([pt.spec_xy_m for pt in parts])
        s.spec_time_s = (np.concatenate([pt.spec_time_s for pt in parts])
                         if all(len(pt.spec_time_s) == pt.n_spectra for pt in parts) else np.zeros(0))
        s.n_spectra = int(sum(pt.n_spectra for pt in parts))
        s._sweep_notes = notes
        s._group_sweeps(np.concatenate(desc), one_page=False)
        return s

    def _acq_ordered(self, a, desc):
        """A column array with the columns acquired down the axis (desc) reversed: the acquisition order."""
        if a is None:
            return None
        out = np.array(a, dtype=np.float64, copy=True)
        if np.any(desc):
            out[:, desc] = out[::-1, desc]
        return out

    def _sweep_masks(self, desc):
        """The kept / ever / points masks of every column in acquisition order (desc: acquired down the axis)."""
        lead = self.lia if self.lia is not None else self.iets
        lead_unset = self.zoffset_lia if self.lia is not None else self.zoffset_iets
        lead_unset = 0.0 if lead_unset is None else lead_unset
        acq = [self._acq_ordered(a, desc) for a in (self.lia, self.iets, self.iv)]
        self.ever = np.any(lead != lead_unset, axis=0)
        self.acquired = self.kept_mask(*acq, unset=(self.zoffset_lia, self.zoffset_iets, self.zoffset_iv))
        self.n_acquired = int(np.sum(self.acquired))
        self.n_stopped = int(np.sum(self.ever & ~self.acquired))
        self.points_acquired = self.points_mask(self._acq_ordered(lead, desc), lead_unset)

    def _group_sweeps(self, desc, one_page):
        """
        Group the columns into pixels and sweeps by their recorded locations (sweeps.py); with k > 1
        sweeps per pixel, the direction of every sweep found (one_page: from stopped sweeps and the
        reversed-storage test; else from the pages, `desc` per column), the reversed columns put
        onto the axis, the columns reordered pixel after pixel (each pixel's sweeps in acquisition
        order), the masks recomputed in acquisition order, and self.sweep_info set; with one sweep
        per pixel nothing changes (sweep_info None).
        """
        from sm4_Browser import sweeps as sw_mod
        self.sweep_info = None
        N = self.n_spectra
        xy = np.asarray(self.spec_xy_m, dtype=np.float64)
        desc = np.asarray(desc, dtype=bool).copy()
        notes = list(getattr(self, '_sweep_notes', []))
        if len(xy) != N or N < 2:
            if not one_page:
                self._sweep_masks(desc)
            return
        d = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
        d = d[d > 0]
        tol = max(self.LOCATION_TOL_FRAC * float(np.min(d)) if d.size else 0.0, self.LOCATION_TOL_MIN_M)
        pixel, locs = sw_mod.group_locations(xy, tol)
        counts = np.bincount(pixel)
        if counts.max() <= 1:
            if not one_page:
                self._sweep_masks(desc)
            return
        if not np.all(counts == counts[0]):
            print(f"{self.name}: the spectra lie {counts.min()} to {counts.max()} per location (not the same "
                  f"number everywhere); taken as one sweep per column.")
            if not one_page:
                self._sweep_masks(desc)
            return
        k, n_pix = int(counts[0]), len(locs)
        gx, gy = getattr(self, 'grid_x', None), getattr(self, 'grid_y', None)
        if gx and gy and gx * gy != n_pix:
            print(f"{self.name}: {n_pix} distinct locations with {k} spectra each, but the grid is {gx} x {gy}; "
                  f"taken as one sweep per column.")
            if not one_page:
                self._sweep_masks(desc)
            return
        rec = np.asarray(self.spec_time_s, dtype=np.float64)
        timed = len(rec) == N and bool(np.any(rec != 0))
        cols_of = []
        for p in range(n_pix):
            c = np.flatnonzero(pixel == p)
            if timed:
                c = c[np.lexsort((c, rec[c]))]
            cols_of.append(c)
        sweep = np.empty(N, dtype=int)
        for c in cols_of:
            sweep[c] = np.arange(k)
        if one_page:
            # Columns sharing a page: the direction from stopped sweeps (never-written samples at the start
            # of the axis) and from storage in acquisition order (the I(V) matches the first sweep reversed).
            lead = self.lia if self.lia is not None else self.iets
            u = self.zoffset_lia if self.lia is not None else self.zoffset_iets
            u = 0.0 if u is None else u
            ref_arr = self.iv if self.iv is not None else lead
            firsts = np.array([c[0] for c in cols_of])
            for s_i in range(1, k):
                cs = np.array([c[s_i] for c in cols_of])
                stopped_start = (lead[0, cs] == u) & (lead[-1, cs] != u) & np.any(lead[:, cs] != u, axis=0)
                if np.any(stopped_start):
                    desc[sweep == s_i] = True
                    notes.append(f"sweep {s_i + 1}: a stopped sweep's never-written samples lie at the start of the "
                                 f"axis, so it runs down the axis (reverse)")
                    continue
                ok = np.any(lead[:, cs] != u, axis=0) & np.any(lead[:, firsts] != u, axis=0)
                if ok.sum() == 0:
                    continue
                as_is = np.nanmedian([sw_mod._corr(ref_arr[:, a], ref_arr[:, b]) for a, b in zip(cs[ok], firsts[ok])])
                flipped = np.nanmedian([sw_mod._corr(ref_arr[::-1, a], ref_arr[:, b]) for a, b in zip(cs[ok], firsts[ok])])
                if flipped > sw_mod.SWEEP_FLIP_CORR and flipped > as_is + sw_mod.SWEEP_FLIP_MARGIN:
                    m = sweep == s_i
                    for attr in ('lia', 'iets', 'iv'):
                        a = getattr(self, attr)
                        if a is not None:
                            a[:, m] = a[::-1, m]
                    desc[m] = True
                    notes.append(f"sweep {s_i + 1}: stored in acquisition order against the axis (its current matches "
                                 f"sweep 1 reversed, r = {flipped:.3f} against {as_is:.3f} as stored); reversed onto "
                                 f"the axis, a reverse sweep")
                else:
                    notes.append(f"sweep {s_i + 1}: the file does not record its bias direction; taken as running "
                                 f"like the page")
        # Pixel after pixel, each pixel's sweeps in acquisition order.
        order = np.concatenate(cols_of)
        for attr in ('lia', 'iets', 'iv'):
            a = getattr(self, attr)
            if a is not None:
                setattr(self, attr, a[:, order])
        self.spec_xy_m = xy[order]
        if len(rec) == N:
            self.spec_time_s = rec[order]
        desc = desc[order]
        pixel = pixel[order]
        sweep = sweep[order]
        new_cols = [np.arange(p * k, (p + 1) * k) for p in range(n_pix)]
        first_desc = desc[np.arange(n_pix) * k]
        direction = np.where(desc == np.repeat(first_desc, k), 1, -1)
        self._sweep_masks(desc)
        self.sweep_info = {'k': k, 'n_pix': n_pix, 'pixel': pixel, 'sweep': sweep, 'dir': direction,
                           'acq_desc': desc, 'cols_of': new_cols, 'locs': locs, 'notes': notes}
        n_rev = int(np.sum(direction[:k] < 0))
        print(f"{self.name}: {k} sweeps per pixel at {n_pix} locations ({k - n_rev} forward, {n_rev} reverse"
              + (", order by recorded time" if timed else ", order as stored") + ")"
              + ("; " + "; ".join(notes) if notes else ""))

    def sweep_view(self, mode):
        """
        One column per pixel (grid order) from the chosen sweeps of every pixel (`mode` one of
        sweeps.sweep_modes; sweeps.chosen_columns): the raw columns averaged over the chosen kept
        sweeps; a pixel with no chosen
        kept sweep keeps its first chosen column (never acquired or stopped).  The view knows its
        parent (the all-sweeps STS) and its chosen columns for the clock (Sm4File.energy_times).
        """
        from sm4_Browser import sweeps as sw_mod
        info = self.sweep_info
        if info is None:
            return self
        chosen = sw_mod.chosen_columns(info, mode)
        n_pix = info['n_pix']
        s = STS.__new__(STS)
        for attr in ('name', 'attrs', 'zscale_iv', 'zoffset_iv', 'zoffset_lia', 'zoffset_iets', 'units_lia',
                     'units_iets', 'units_iv', 'vmod', 'grid_x', 'grid_y', 'iv_dropped', 'energy', 'energy_units',
                     'adc_period', 'V_bias', 'I_setpoint', 'iv_realigned'):
            setattr(s, attr, getattr(self, attr, None))
        npts = len(self.energy)
        acq_v = np.zeros(n_pix, dtype=bool)
        ever_v = np.zeros(n_pix, dtype=bool)
        pts_v = np.zeros(n_pix, dtype=int)
        for attr, u in (('lia', self.zoffset_lia), ('iets', self.zoffset_iets), ('iv', self.zoffset_iv)):
            a = getattr(self, attr)
            if a is None:
                setattr(s, attr, None)
                continue
            out = np.full((npts, n_pix), 0.0 if u is None else float(u))
            for p, c in enumerate(chosen):
                if not len(c):
                    continue
                kc = c[self.acquired[c]]
                out[:, p] = a[:, kc].mean(axis=1) if len(kc) else a[:, c[0]]
            setattr(s, attr, out)
        for p, c in enumerate(chosen):
            if len(c):
                acq_v[p] = bool(np.any(self.acquired[c]))
                ever_v[p] = bool(np.any(self.ever[c]))
                pts_v[p] = npts if acq_v[p] else int(np.max(self.points_acquired[c]))
        s.n_spectra = n_pix
        s.acquired = acq_v
        s.ever = ever_v
        s.n_acquired = int(acq_v.sum())
        s.n_stopped = int(np.sum(ever_v & ~acq_v))
        s.points_acquired = pts_v
        s.spec_xy_m = np.asarray(info['locs'], dtype=np.float64).copy()
        rec = self.spec_time_s
        s.spec_time_s = (np.array([rec[c[0]] if len(c) else 0.0 for c in chosen])
                         if len(rec) == self.n_spectra else np.zeros(0))
        s.sweep_info = None
        s.view_mode = mode
        s.view_cols = chosen
        s.parent = self
        # The calibration's reference (calibration.calibrate's `ref`) for a view that is not the
        # forward one: per pixel the mean raw columns of its kept forward sweeps, the sweeps that left
        # the setpoint; NaN for a pixel without one.  The forward view regresses on its own columns.
        s.lia_ref = s.iets_ref = s.iv_ref = None
        if mode != sw_mod.FORWARD:
            fwd = sw_mod.chosen_columns(info, sw_mod.FORWARD)
            for attr in ('lia', 'iets', 'iv'):
                a = getattr(self, attr)
                if a is None:
                    continue
                out = np.full((npts, n_pix), np.nan)
                for p, c in enumerate(fwd):
                    kc = c[self.acquired[c]] if len(c) else c
                    if len(kc):
                        out[:, p] = a[:, kc].mean(axis=1)
                setattr(s, attr + '_ref', out)
        return s

    @classmethod
    def from_pages(cls, pages, page_type, name='', _group=True):
        """
        The spectroscopy of a loaded file (sm4file.load's pages, in physical
        units) from its pages of `page_type`: the dI/dV and IETS channels by
        label, the optional I(V) of matching shape, the energy axis (the
        channel page's x axis), V_bias, I_setpoint, the units, the
        modulation amplitude, the grid size, the acquired mask, the recorded
        positions and times.  None when neither channel exists.  The I(V)
        page's columns (and the IETS page's) are aligned onto the dI/dV page's
        recorded locations by the Current page's own location list
        (page_perm): identical lists leave the columns as stored, another
        order is realigned with a console line, an unmatched location raises
        AlignmentError (Sm4File then shows the file without spectra), as does
        a Current page whose energy axis differs from the channel page's.
        """
        pages = list(pages)
        if _group:
            # Several pages for one channel: the sweeps of every pixel stored one page per sweep
            # direction (sweeps.py); each set read on its own and merged onto the first set's axis.
            D = cls._locate_all(pages, page_type, 'didv')
            I = cls._locate_all(pages, page_type, 'iets')
            if max(len(D), len(I)) > 1:
                return cls._from_sweep_pages(pages, page_type, name)
        didv_idx = cls._locate(pages, page_type, 'didv', name)
        iets_idx = cls._locate(pages, page_type, 'iets', name)
        iv_idx = cls._locate(pages, page_type, 'iv', name)
        if didv_idx is None and iets_idx is None:
            return None
        self = cls.__new__(cls)
        self.name = name
        ref = pages[didv_idx if didv_idx is not None else iets_idx]
        self.attrs = dict(ref.attrs)

        def column_array(page):
            """A page's data (physical units) as a (npts, N) float64 array, a single spectrum as one column."""
            if page.data is None:
                raise ValueError(f"{name}: the '{page.label}' page has no usable RHK_Zscale, so it has no "
                                 f"physical data.")
            a = np.array(page.data, dtype=np.float64)
            return a[:, None] if a.ndim == 1 else a

        self.lia = column_array(pages[didv_idx]) if didv_idx is not None else None
        self.iets = column_array(pages[iets_idx]) if iets_idx is not None else None
        lead = self.lia if self.lia is not None else self.iets
        if self.lia is not None and self.iets is not None and self.iets.shape != self.lia.shape:
            raise ValueError(f"{name}: the dI/dV page {self.lia.shape} and the IETS page {self.iets.shape} "
                             f"differ in shape.")
        ref_xy = cls._page_locations(ref)
        # The IETS page's columns aligned onto the dI/dV page's recorded locations (page_perm).
        if self.lia is not None and self.iets is not None:
            perm = cls.page_perm(ref_xy, cls._page_locations(pages[iets_idx]), name, 'IETS')
            if perm is not None:
                self.iets = self.iets[:, perm]
        npts = lead.shape[0]
        # The energy axis: the channel page's x axis (RHK_Xoffset + i x RHK_Xscale, in RHK_Xunits).
        if len(ref.x) != npts:
            raise ValueError(f"{name}: the '{ref.label}' page's x axis has {len(ref.x)} points, its data "
                             f"{npts}.")
        self.energy = np.array(ref.x, dtype=np.float64)
        self.energy_units = ref.xunits
        self.units_lia = pages[didv_idx].units if didv_idx is not None else None
        self.units_iets = pages[iets_idx].units if iets_idx is not None else None
        # Each channel page's RHK_Zoffset: the value of a never-written sample (records, see unset_value).
        self.zoffset_lia = pages[didv_idx].zoffset if didv_idx is not None else None
        self.zoffset_iets = pages[iets_idx].zoffset if iets_idx is not None else None
        self.iv = None
        self.units_iv = None
        self.zscale_iv = None                    # the Current page's RHK_Zscale, a record (the data are in units_iv)
        self.zoffset_iv = None                   # the Current page's RHK_Zoffset, a record
        self.iv_dropped = None                   # shape of an I(V) page left out for not matching
        self.iv_realigned = False                # True when the I(V) columns were reordered by location
        iv_unset = None
        if iv_idx is not None:
            iv_page = pages[iv_idx]
            iv = column_array(iv_page)
            if iv.shape == lead.shape:
                if not np.array_equal(iv_page.x, ref.x):
                    raise AlignmentError(f"{name}: the 'Current' page's energy axis differs from the "
                                         f"'{ref.label}' page's; the pages cannot be aligned.")
                # The I(V) page's columns aligned onto the ref page's recorded locations: the
                # Current page carries its own location list, which is read and matched here.
                perm = cls.page_perm(ref_xy, cls._page_locations(iv_page), name, 'Current',
                                     'dI/dV' if didv_idx is not None else 'IETS')
                if perm is not None:
                    iv = iv[:, perm]
                    self.iv_realigned = True
                self.iv = iv
                self.units_iv = iv_page.units
                self.zscale_iv = iv_page.zscale
                self.zoffset_iv = iv_page.zoffset
                iv_unset = iv_page.zoffset
            else:
                self.iv_dropped = iv.shape
        # The modulation amplitude (V) and the recorded grid size, None when not recorded.
        self.vmod = cls._float_attr(ref.attrs, 'RHK_CH1Drive_Amplitude')
        self.lia_ref = self.iets_ref = self.iv_ref = None     # a sweep view's forward-sweep reference (sweep_view)
        self.grid_x = cls._int_attr(ref.attrs, 'RHK_GridXsize')
        self.grid_y = cls._int_attr(ref.attrs, 'RHK_GridYsize')
        self.V_bias = float(ref.attrs['RHK_Bias'])
        self.I_setpoint = float(ref.attrs['RHK_Current'])
        # The spectroscopy page's RHK_Period (s): the digitizing period of the LIA channel (1e-4 s,
        # 10 kHz, the same on every spectroscopy-type page of an experiment, the noise spectra
        # included), not the time per energy point; that time is derived from the file span
        # (Sm4File.grid_timing / dwell_s).  Shown on the info panel, used for nothing else.
        try:
            self.adc_period = float(ref.attrs.get('RHK_Period', 0.0)) or None
        except (TypeError, ValueError):
            self.adc_period = None
        self.n_spectra = lead.shape[1]
        # The kept spectra: no column never written (never acquired) or ending in never-written
        # samples (a stopped sweep); a never-written sample is the page's RHK_Zoffset.
        lead_unset = ref.zoffset
        self.ever = np.any(lead != lead_unset, axis=0)
        self.acquired = self.kept_mask(self.lia, self.iets, self.iv,
                                       unset=(self.zoffset_lia, self.zoffset_iets, iv_unset))
        self.n_acquired = int(np.sum(self.acquired))
        self.n_stopped = int(np.sum(self.ever & ~self.acquired))
        # The points each sweep acquired: the last written sample's index plus one in the acquisition
        # order (npts for a complete sweep, fewer for a stopped one, 0 for one never started); the
        # stopped sweep's share of the file span in Sm4File.grid_timing.
        self.points_acquired = self.points_mask(lead, lead_unset)
        if self.n_stopped:
            print(f"{name}: {self.n_stopped} sweep(s) end in zeros (stopped before the end) and are left out; "
                  f"{self.n_acquired} of {self.n_spectra} spectra kept.")
        xs = np.asarray(ref.attrs.get('RHK_SpecDrift_Xcoord', []), dtype=np.float64)
        ys = np.asarray(ref.attrs.get('RHK_SpecDrift_Ycoord', []), dtype=np.float64)
        n = min(len(xs), len(ys), self.n_spectra)
        self.spec_xy_m = np.column_stack([xs[:n], ys[:n]]) if n else np.zeros((0, 2))
        self.spec_time_s = np.asarray(ref.attrs.get('RHK_SpecDrift_Time', []), dtype=np.float64)[:self.n_spectra]
        self.sweep_info = None
        if _group:
            self._group_sweeps(np.zeros(self.n_spectra, dtype=bool), one_page=True)
        return self

    @staticmethod
    def _float_attr(attrs, key):
        """A recorded attribute as a finite float; None when absent or unreadable."""
        try:
            v = float(attrs[key])
        except (KeyError, TypeError, ValueError):
            return None
        return v if np.isfinite(v) else None

    @staticmethod
    def _int_attr(attrs, key):
        """A recorded attribute as a positive int; None when absent, unreadable, or not positive."""
        try:
            v = int(attrs[key])
        except (KeyError, TypeError, ValueError):
            return None
        # 0xFFFFFFFF (the instrument's 'not set', e.g. a grid stopped before it began) is not a value.
        return v if 0 < v < 2 ** 31 - 1 else None

    @staticmethod
    def points_mask(columns, unset=0.0):
        """
        (N,) the points each column acquired: the last written sample's index
        plus one (0 for a column never written); `unset` is the value of a
        never-written sample (the page's RHK_Zoffset).
        """
        c = np.asarray(columns, dtype=np.float64)
        if c.ndim != 2 or c.shape[0] == 0:
            return np.zeros(c.shape[1] if c.ndim == 2 else 0, dtype=int)
        nz = c != unset
        last = c.shape[0] - 1 - np.argmax(nz[::-1, :], axis=0)
        return np.where(nz.any(axis=0), last + 1, 0).astype(int)

    @staticmethod
    def kept_mask(*columns, unset=None):
        """
        The kept spectra of a file: for every column array given (npts, N)
        (None skipped), a spectrum is dropped when its column is never
        written (never acquired) or when its last sample, in the file's
        column order (the acquisition order), is never written (a sweep
        stopped before its end is zero-filled from the stop).  `unset` gives,
        in the order of `columns`, the value of a never-written sample of
        each array (its page's RHK_Zoffset: a stored 0 converts to exactly
        that); None, or a None entry, is 0.  Returns the (N,) bool mask.
        """
        if unset is None:
            unset = (None,) * len(columns)
        if len(unset) != len(columns):
            raise ValueError(f"kept_mask: {len(columns)} column arrays and {len(unset)} unset values.")
        mask = None
        for c, u in zip(columns, unset):
            if c is None:
                continue
            u = 0.0 if u is None else float(u)
            c = np.asarray(c, dtype=np.float64)
            ok = np.any(c != u, axis=0) & (c[-1, :] != u)
            mask = ok if mask is None else (mask & ok)
        return mask if mask is not None else np.zeros(0, dtype=bool)

    @staticmethod
    def _page_locations(page):
        """The recorded location (x, y in metres) of every spectrum of a spectroscopy page, (n, 2); (0, 2) when absent."""
        xs = np.asarray(page.attrs.get('RHK_SpecDrift_Xcoord', []), dtype=np.float64)
        ys = np.asarray(page.attrs.get('RHK_SpecDrift_Ycoord', []), dtype=np.float64)
        n = min(len(xs), len(ys))
        return np.column_stack([xs[:n], ys[:n]]) if n else np.zeros((0, 2))

    LOCATION_TOL_FRAC = 1e-3     # of the smallest step between distinct recorded locations: two locations within it are the same
    LOCATION_TOL_MIN_M = 1e-14   # metres: the floor of that tolerance

    @classmethod
    def page_perm(cls, ref_xy, page_xy, name, label, ref_label='dI/dV'):
        """
        The column order that aligns another spectroscopy page onto the
        reference (dI/dV) page's recorded locations: perm[j] is the page's
        column recorded at the reference's location j.  None when the two
        location lists are identical in order (nothing to reorder) or when
        either page carries no locations.  Columns recorded at one and the
        same location (repeated sweeps at a point) keep their relative order.
        AlignmentError when the lists differ in length or when a reference
        location has no unused spectrum of the page within the tolerance
        (LOCATION_TOL_FRAC of the smallest step between distinct reference
        locations, at least LOCATION_TOL_MIN_M).
        """
        ref_xy = np.asarray(ref_xy, dtype=np.float64)
        page_xy = np.asarray(page_xy, dtype=np.float64)
        if len(ref_xy) == 0 or len(page_xy) == 0:
            return None
        if len(ref_xy) != len(page_xy):
            raise AlignmentError(f"{name}: the '{label}' page records {len(page_xy)} locations, the {ref_label} page "
                                 f"{len(ref_xy)}; the pages cannot be aligned.")
        if np.array_equal(ref_xy, page_xy):
            return None
        # Tolerance from the smallest step between distinct reference locations.
        d = np.hypot(np.diff(ref_xy[:, 0]), np.diff(ref_xy[:, 1]))
        d = d[d > 0]
        tol = max(cls.LOCATION_TOL_FRAC * float(np.min(d)) if d.size else 0.0, cls.LOCATION_TOL_MIN_M)
        if np.max(np.hypot(ref_xy[:, 0] - page_xy[:, 0], ref_xy[:, 1] - page_xy[:, 1])) <= tol:
            return None                                       # the same list up to rounding
        used = np.zeros(len(page_xy), dtype=bool)
        perm = np.empty(len(ref_xy), dtype=int)
        for j, (x, y) in enumerate(ref_xy):
            dist = np.hypot(page_xy[:, 0] - x, page_xy[:, 1] - y)
            cands = np.flatnonzero((dist <= tol) & ~used)     # in page order: repeated sweeps keep their order
            if cands.size == 0:
                raise AlignmentError(f"{name}: no '{label}' spectrum is recorded at the location of {ref_label} "
                                     f"spectrum {j + 1} ({x * 1e9:.4f}, {y * 1e9:.4f} nm); the pages cannot be aligned.")
            perm[j] = cands[0]
            used[cands[0]] = True
        n_moved = int(np.sum(perm != np.arange(len(perm))))
        print(f"{name}: the '{label}' page's spectra are recorded in another order than the {ref_label} page's; "
              f"{n_moved} of {len(perm)} columns realigned by their recorded locations.")
        return perm

    @classmethod
    def from_file(cls, path, page_type='RHK_PAGE_RAMP_SPECTROSCOPY_RP'):
        """
        The spectroscopy of a .sm4 file opened here, from its pages of
        `page_type` (ramp spectroscopy, the single-point files, unless given);
        None when the file holds no dI/dV or IETS page of that type.
        """
        from sm4_Browser import sm4file            # here: sm4file imports this module
        f = sm4file.load(path)
        return cls.from_pages(f.pages, page_type, name=os.path.basename(path))

    @staticmethod
    def from_arrays(energy, lia, iv, units_lia=None, units_iv=None, vmod=None, V_bias=None, I_setpoint=None,
                    lia_ref=None, iv_ref=None):
        """
        A raw set dict: energy (npts,), lia and iv (npts, n) in physical units
        (the loader's data), the units of each (RHK_Zunits; None when not
        known), the modulation amplitude vmod (V; None when not recorded),
        the setpoint V_bias (V) and I_setpoint (A) (recorded; the calibration
        does not use them), and the forward-sweep reference columns lia_ref /
        iv_ref (npts, n; None when the columns are their own reference; the
        calibration does not use them either).
        """
        energy = np.asarray(energy, dtype=np.float64)
        lia = np.asarray(lia, dtype=np.float64)
        iv = np.asarray(iv, dtype=np.float64)
        if lia.ndim == 1:
            lia = lia[:, None]
        if iv.ndim == 1:
            iv = iv[:, None]
        if lia.shape != iv.shape or lia.shape[0] != len(energy):
            raise ValueError(f"raw set: lia {lia.shape}, iv {iv.shape} and energy ({len(energy)}) must agree.")
        return {'energy': energy, 'lia': lia, 'iv': iv, 'units_lia': units_lia, 'units_iv': units_iv,
                'vmod': None if vmod is None else float(vmod),
                'V_bias': None if V_bias is None else float(V_bias),
                'I_setpoint': None if I_setpoint is None else float(I_setpoint),
                'lia_ref': None if lia_ref is None else np.asarray(lia_ref, dtype=np.float64),
                'iv_ref': None if iv_ref is None else np.asarray(iv_ref, dtype=np.float64)}

    def unset_value(self, name):
        """
        The value of a never-written sample of 'didv', 'iets', or 'iv' (its
        page's RHK_Zoffset: a stored 0 converts to exactly that); 0.0 for a
        page this file does not hold.
        """
        u = {'didv': getattr(self, 'zoffset_lia', None), 'iets': getattr(self, 'zoffset_iets', None),
             'iv': getattr(self, 'zoffset_iv', None)}[name]
        return 0.0 if u is None else float(u)

    def channel(self, name):
        """The raw columns (npts, N) of channel 'didv' or 'iets'; an error when the file has none."""
        cols = {'didv': self.lia, 'iets': self.iets}.get(name)
        if cols is None:
            raise ValueError(f"{self.name} has no {self.CHANNEL_NAMES.get(name, name)} channel.")
        return cols

    def channels(self):
        """The channels this file holds, in CHANNELS order."""
        return tuple(c for c in self.CHANNELS if {'didv': self.lia, 'iets': self.iets}[c] is not None)

    def subset(self, cols):
        """
        A new STS of the chosen columns (global indices of this file's
        spectra, in the order given): lia, iets, iv, acquired, positions and
        times restricted to them, n_spectra and n_acquired recomputed; the
        energy axis, V_bias, I_setpoint, the units, zscale_iv, zoffset_iv,
        vmod, the grid size, attrs, and name shared.
        Every pipeline step is per column, so a column processed in a subset
        equals the same column processed with its whole file.
        """
        cols = np.asarray(cols, dtype=int)
        if cols.ndim != 1 or cols.size == 0:
            raise ValueError(f"{self.name}: a subset needs at least one column index.")
        if np.any(cols < 0) or np.any(cols >= self.n_spectra):
            raise ValueError(f"{self.name}: column index out of range 0..{self.n_spectra - 1}.")
        s = STS.__new__(STS)
        s.name = self.name
        s.attrs = self.attrs
        s.lia = None if self.lia is None else self.lia[:, cols].copy()
        s.iets = None if self.iets is None else self.iets[:, cols].copy()
        s.iv = None if self.iv is None else self.iv[:, cols].copy()
        for attr in ('lia_ref', 'iets_ref', 'iv_ref'):           # the forward-sweep reference, the same columns
            a = getattr(self, attr, None)
            setattr(s, attr, None if a is None else a[:, cols].copy())
        s.zscale_iv = self.zscale_iv
        s.zoffset_iv = getattr(self, 'zoffset_iv', None)
        s.zoffset_lia = getattr(self, 'zoffset_lia', None)
        s.zoffset_iets = getattr(self, 'zoffset_iets', None)
        s.units_lia = getattr(self, 'units_lia', None)
        s.units_iets = getattr(self, 'units_iets', None)
        s.units_iv = getattr(self, 'units_iv', None)
        s.vmod = getattr(self, 'vmod', None)
        s.grid_x = getattr(self, 'grid_x', None)
        s.grid_y = getattr(self, 'grid_y', None)
        s.iv_dropped = self.iv_dropped
        s.energy = self.energy
        s.energy_units = getattr(self, 'energy_units', None)
        s.adc_period = getattr(self, 'adc_period', None)
        s.V_bias = self.V_bias
        s.I_setpoint = self.I_setpoint
        s.n_spectra = int(cols.size)
        s.acquired = self.acquired[cols].copy()
        s.n_acquired = int(np.sum(s.acquired))
        s.ever = self.ever[cols].copy() if len(getattr(self, 'ever', [])) == len(self.acquired) else s.acquired.copy()
        s.n_stopped = int(np.sum(s.ever & ~s.acquired))
        pa = getattr(self, 'points_acquired', None)
        s.points_acquired = (pa[cols].copy() if pa is not None and len(pa) == len(self.acquired)
                             else np.where(s.acquired, len(self.energy), 0))
        s.spec_xy_m = self.spec_xy_m[cols].copy() if len(self.spec_xy_m) == len(self.acquired) else np.zeros((0, 2))
        s.spec_time_s = (self.spec_time_s[cols].copy() if len(self.spec_time_s) == len(self.acquired)
                         else np.zeros(0))
        s.sweep_info = None
        s.view_mode = getattr(self, 'view_mode', None)
        return s

    # --- the pipeline (the steps live in filters, calibration, feenstra, normalization) ---

    _notes_said = set()        # the console lines of a refused calibration and of a file's gain, each said once

    @classmethod
    def _note_once(cls, text):
        if text not in cls._notes_said:
            cls._notes_said.add(text)
            print(text)

    @staticmethod
    def _gain_text(name, fits, vmod, channel, cols=None):
        """One console line: the file's fitted gains (per sweep: their range and median), the gate's doing (dI/dV), and the modulation amplitude's check."""
        unit = "/V²" if channel == 'iets' else "/V"
        a = np.array([ft.a for ft in fits], dtype=np.float64)
        text = (f"{name}: calibration per sweep (a, b" + (", c, d" if channel == 'iets' else ", c")
                + f" by least squares, nothing locked); gain a from {a.min():.4g} to {a.max():.4g} {unit}, "
                f"median {np.median(a):.4g} {unit}")
        if channel != 'iets':
            text += "; " + calibration.gate_text(fits, cols)
        a_mod = fits[0].a_mod
        if np.isfinite(a_mod):
            rel = "4 / V_mod²" if channel == 'iets' else "1 / V_mod"
            return text + f"; {rel} = {a_mod:.4g} {unit} (V_mod = {float(vmod) * 1e3:.3g} mV) as the check."
        return text + "; no modulation amplitude recorded."

    @classmethod
    def process_arrays(cls, energy, lia, iv, hampel=None, sgolay=None, divide_iv=False, iv_broadening=0.0,
                       return_areas=False, channel='didv', acquired=None, iv_zero_sub=False, normalize=True,
                       calibrate=True, vmod=None, norm_range=None, V_bias=None, I_setpoint=None, ref=None,
                       hampel_didv=None, sgolay_didv=None, name=None, gate=None):
        """
        Per column: the channel's columns through the dI/dV Hampel and
        Savitzky-Golay filters (hampel_didv, sgolay_didv: None = off) and
        the recorded current through the I(V) ones (hampel, sgolay), each
        once (filters.filter_lia), calibration of the channel's columns
        against that current (when present and `calibrate` is on;
        calibration.calibrate for dI/dV, calibrate_iets for IETS: per sweep
        the least squares of the gain, the offset and the constant of
        integration, nothing locked, then the gate on the dI/dV gains and
        offsets as `gate` sets it, calibration.GateSettings or None for the
        defaults (the window's 'gate' row with its 'H' and 'SG' boxes:
        calibration.gate_sweeps); V_bias / I_setpoint and `ref` =
        (lia_ref, iv_ref), the forward-sweep reference columns of a
        reverse-sweep view through the same filters, are handed over and
        not used; off, the columns stay in the channel page's units; a
        column the regression cannot determine refuses the file, said once
        on the console; with `name` the file's, the gains, the gate's
        doing and the check from the modulation amplitude `vmod` are said
        once per file), the division by the broadened I/V of the same current when
        divide_iv is on (dI/dV only; needs I(V); iv_broadening is the
        broadening width in volts, 0 the plain I/V; iv_zero_sub subtracts
        each column's current at 0 V before I/V is formed, see
        feenstra.iv_corrected; the division is told not to filter again),
        unit-area normalization (normalization.normalize; skipped with
        `normalize` off: the areas are then NaN; norm_range (lo, hi) limits
        the area to the samples inside that energy range).  The filters reach every
        use of the current here, the calibration included.  Columns flagged
        as never acquired in `acquired` run through the division with the
        others (see the comment below), are left out of the calibration,
        and come back as zero columns (area 0).  With return_areas, returns
        (curves, areas).  vmod (the file's modulation amplitude,
        RHK_CH1Drive_Amplitude) is handed to the calibration for the check
        it carries beside the fitted gains (calibration.py's regression
        does not use it).
        """
        lia = np.asarray(lia, dtype=np.float64)
        if acquired is None:
            acquired = np.ones(lia.shape[1], dtype=bool)
        acquired = np.asarray(acquired, dtype=bool)
        cols = np.flatnonzero(acquired)
        # The division runs on the array's full width (a never-acquired column holds its page's offset),
        # because its matrix products give the last bits of a column differently for another
        # width; only the regression, column by column, leaves the never-acquired columns out.
        # The channel's columns through the dI/dV filters once (a plain copy when both are off).
        c = filters.filter_lia(lia, hampel_didv, sgolay_didv)
        if iv is not None:
            # The recorded current through the I(V) filters once (a copy when both are off): the
            # calibration and the division both work on this current.
            iv = filters.filter_lia(np.asarray(iv, dtype=np.float64), hampel, sgolay)
        if calibrate and iv is not None:
            ref_f = None
            if ref is not None and ref[0] is not None and ref[1] is not None:
                # The reference columns through the same filters as the columns they stand for.
                ref_f = (filters.filter_lia(np.asarray(ref[0], dtype=np.float64), hampel_didv, sgolay_didv)[:, cols],
                         filters.filter_lia(np.asarray(ref[1], dtype=np.float64), hampel, sgolay)[:, cols])
            fn = calibration.calibrate_iets if channel == 'iets' else calibration.calibrate
            try:
                cal, fits = fn(energy, c[:, cols], iv[:, cols], vmod, V_bias=V_bias, I_setpoint=I_setpoint, ref=ref_f,
                               gate=gate)
                c[:, cols] = cal
                if name and fits:
                    cls._note_once(cls._gain_text(name, fits, vmod, channel, cols))
            except ValueError as exc:
                cls._note_once(f"calibration not applied: {exc}")
        if divide_iv and channel != 'iets':
            if iv is None:
                raise ValueError("Feenstra needs the 'Current' channel, which this file does not have.")
            with np.errstate(divide='ignore', invalid='ignore'):
                c = feenstra.divide_by_iv(energy, c, iv, iv_broadening, iv_zero_sub, None, None)
        if normalize:
            with np.errstate(invalid='ignore'):
                c, areas = normalization.normalize(energy, c, energy_range=norm_range)
        else:
            areas = np.full(c.shape[1], np.nan)
        c[:, ~acquired] = 0.0
        areas[~acquired] = 0.0
        return (c, areas) if return_areas else c

    def process(self, channel='didv', hampel=None, sgolay=None, divide_iv=False, iv_broadening=0.0,
                return_areas=False, iv_zero_sub=False, normalize=True, calibrate=True,
                hampel_didv=None, sgolay_didv=None, gate=None):
        """Every spectrum of the channel through the pipeline (npts, N), original energy order (process_arrays: the setpoint and a sweep view's forward reference from the file; `gate` the window's gate settings)."""
        ref = (getattr(self, 'iets_ref' if channel == 'iets' else 'lia_ref', None), getattr(self, 'iv_ref', None))
        return self.process_arrays(self.energy, self.channel(channel), self.iv, hampel, sgolay, divide_iv,
                                   iv_broadening, return_areas=return_areas, channel=channel,
                                   acquired=self.acquired, iv_zero_sub=iv_zero_sub, normalize=normalize,
                                   calibrate=calibrate, vmod=getattr(self, 'vmod', None),
                                   V_bias=getattr(self, 'V_bias', None), I_setpoint=getattr(self, 'I_setpoint', None),
                                   ref=ref, hampel_didv=hampel_didv, sgolay_didv=sgolay_didv,
                                   name=getattr(self, 'name', None), gate=gate)


if __name__ == '__main__':
    # One file's spectroscopy through the pipeline: the channels found, the shapes, and the
    # per-column areas before normalization (nothing is drawn).
    path = r'experiment_dir/file.sm4'
    s = STS.from_file(path)
    print(s.name, 'channels', s.channels(), 'spectra', s.n_spectra, 'acquired', s.n_acquired,
          'I(V)' if s.iv is not None else 'no I(V)')
    c, areas = s.process('didv', return_areas=True)
    print('processed', c.shape, 'areas', np.array2string(areas[s.acquired], precision=4))
