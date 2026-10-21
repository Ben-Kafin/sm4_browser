# -*- coding: utf-8 -*-
"""
Created on Fri May  2 14:01:51 2025

@author: Benjamin Kafin

The Line tool: a grid line's dI/dV or IETS against position.  Data
extraction from the file (the channel's columns and the I(V) of one line,
the topography rows of the stripe), and the Line window: the topo stripe,
the heatmap with its 'γ' box, the marker curve panel, the marker groups
(widgets.MarkerGroupControls), the window's own 'Feenstra' toggle with its
Feenstra companion (feenstra.FeenstraWindow), the 'calibrate' /
'normalize' box (widgets.PipelineBox, top right as in every window), the
'line' box with its '< line' / 'line >' arrows, the topo-row controls,
'save' (the marker record averages/<file stem>_line<k>_markers.json),
'line → new avg' / 'line → existing avg' (this line into a line average,
lineavg.LineAverages), 'add new' / 'add existing' per marker group (the
averaging windows), and 'Send to fit'.  Every control is built through
widgets.py; every drawing goes through plots.py.  The pipeline is the
package's one pipeline (STS.process_arrays).
"""

if __package__ in (None, ''):
    # Run as a file (Spyder F5): make the package importable from wherever its folder sits.
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from datetime import datetime
import numpy as np
from sm4_Browser import qtplot as plt
from sm4_Browser.qtplot import gridspec
from sm4_Browser import fit, topo, sm4file, widgets, plots, records
from sm4_Browser.drift import DriftWindow
from sm4_Browser.sts import STS, curve_units
from sm4_Browser.averaging import Averages, Selection
from sm4_Browser.feenstra import (FeenstraWindow, feenstra_width, feenstra_titles, energy_has_zero, iv_corrected,
                                  iv_broadened)
from sm4_Browser.normalization import normalize_abs


TOPO_MODES = ('forward', 'reverse')


def fast_page_of(pages, name=''):
    """The first topographic page of a file (by page type); the fast axis and the position axis come from it."""
    for page in pages:
        if page.attrs.get('RHK_PageTypeName') == 'RHK_PAGE_TOPOGRAPHIC':
            return page
    raise ValueError(f"{name}: no topographic page (the fast axis of the grid).")


def line_positions(fast, scalefactor):
    """
    The position axis of a grid line in Å (float64) from the fast page's
    attributes: RHK_Xoffset + i * RHK_Xscale over shape[0] of the page,
    divided by the scale factor, reversed and shifted so that the first
    value is zero.  The fallback of line_axis for a file that records no
    spectrum positions.
    """
    size = int(np.shape(fast.data)[0])
    pos = np.array(
        [fast.attrs['RHK_Xoffset'] + i * fast.attrs['RHK_Xscale']
         for i in range(size)], dtype=np.float64
    ) * 1.0e10  # convert position to Å
    pos = (pos / scalefactor) - (pos[0] / scalefactor)
    # Reverse and shift so that the first value is zero.
    return pos[::-1] - pos[::-1][0]


def line_axis(xy_m, n_fast, line, scalefactor, fast_page=None):
    """
    The position axis of one line of a grid (Å after the scale factor): the
    along-line distance of every spectrum of the line from the line's first
    spectrum, from the recorded spectrum positions xy_m (N, 2; metres,
    RHK_SpecDrift_Xcoord / Ycoord), the line direction the recorded vector
    from the grid's first spectrum to its second.  Spectrum j of the line
    sits at pos[j]; so does pixel j of the line's forward and reverse topo
    rows (Sm4File.topo_raw, raw pixel order: the image pixel (i, j) is
    recorded at the position of spectrum i x n_fast + j).  Without recorded
    positions the fast page's axis (line_positions) is used.
    """
    xy = np.asarray(xy_m, dtype=np.float64)
    n_fast = int(n_fast)
    cols = np.arange(line * n_fast, min((line + 1) * n_fast, len(xy)))
    if len(xy) < 2 or len(cols) < 2:
        if fast_page is not None:
            return line_positions(fast_page, scalefactor)
        return np.arange(max(n_fast, 1), dtype=np.float64)
    fast = xy[1] - xy[0]
    n = np.linalg.norm(fast)
    fast = fast / n if n > 0 else fast
    s = (xy[cols] - xy[cols[0]]) @ fast * 1e10 / float(scalefactor)
    if len(s) < n_fast:                                    # a short last line: the nominal spacing continues
        step = float(np.linalg.norm(xy[1] - xy[0])) * 1e10 / float(scalefactor)
        s = np.concatenate([s, s[-1] + step * np.arange(1, n_fast - len(s) + 1)])
    return np.asarray(s, dtype=np.float64)


def stripe_rows(sm4, topo_lines, line=None, scalefactor=1.0, kept_only=True):
    """
    The raw row profiles of (row, mode) pairs from a grid file's topographic
    pages, every pixel at its own recorded location: a page row's pixels
    (Sm4File.topo_raw, raw pixel order) with the along-line coordinate of
    each, its recorded location from the page geometry
    (Sm4File.pixel_xy_m) projected on the grid's fast axis from the first
    spectrum of `line` (the window's line; the row's own line when None),
    in Å after the scale factor.  A forward pixel is kept only when the
    spectrum recorded at its location (Sm4File.spectrum_pixels, by
    location, never by index) is kept; a forward pixel with no spectrum at
    its location is NaN and counted; the reverse page has no spectroscopy
    and its acquired pixels are all kept.  With kept_only False (the raw
    profile window) every acquired forward pixel is kept whether or not its
    spectrum is.  A row of None means the middle
    row.  Returns ([profile, ...], [positions, ...], [(row, mode), ...]
    resolved), each profile and its positions sorted by position.  Raises
    on a bad row or mode.
    """
    profiles, positions, resolved = [], [], []
    imgs, locs = {}, {}
    fast, _ = sm4.grid_axes()
    xy = np.asarray(sm4.spec_xy_m, dtype=np.float64)
    if fast is None or len(xy) == 0:
        raise ValueError(f"{sm4.name}: no recorded spectrum positions; the stripe cannot be placed.")
    idx_f, _ = sm4.spectrum_pixels('forward')
    acq = np.asarray(sm4.acquired, dtype=bool)[:len(idx_f)]
    pix_kept = None
    unpaired = int(np.sum(acq & (idx_f[:, 0] < 0)))
    if unpaired:
        print(f"{sm4.name}: {unpaired} kept spectra have no forward pixel at their location "
              f"(deviation above {sm4file.PIXEL_PAIR_TOL_PX:g} px); no stripe cell for them.")
    for r, mode in topo_lines:
        if mode not in TOPO_MODES:
            raise ValueError(f"topo mode must be one of {TOPO_MODES} (got '{mode}').")
        if mode not in imgs:
            if mode == 'reverse' and len(sm4.topo_pages) < 2:
                raise ValueError("File does not contain both forward and reverse topo pages.")
            imgs[mode] = sm4.topo_raw(mode)
            locs[mode] = sm4.pixel_xy_m(mode)
            if mode == 'forward':
                pix_kept = np.zeros(imgs[mode].shape, dtype=bool)
                for k in np.flatnonzero(acq & (idx_f[:, 0] >= 0)):
                    pix_kept[idx_f[k, 0], idx_f[k, 1]] = True
        td = imgs[mode]
        row = r if r is not None else td.shape[0] // 2
        if row < 0 or row >= td.shape[0]:
            raise ValueError(f"topo row {row} out of range 0..{td.shape[0] - 1}.")
        ref_line = int(line) if line is not None else int(row)
        k0 = min(ref_line * int(sm4.n_fast), len(xy) - 1)
        s = (locs[mode][row] - xy[k0]) @ fast * 1e10 / float(scalefactor)
        prof = np.array(td[row, :], dtype=np.float64) * 1e10          # Å
        if mode == 'forward' and pix_kept is not None and kept_only:
            prof[~pix_kept[row]] = np.nan
        order = np.argsort(s)
        profiles.append(prof[order])
        positions.append(np.asarray(s[order], dtype=np.float64))
        resolved.append((int(row), mode))
    if not profiles:
        raise ValueError("at least one topo row is needed.")
    return profiles, positions, resolved


def common_stripe_axis(profiles, positions):
    """
    One position axis for several row profiles: the first profile's; a
    profile recorded on another set of positions is interpolated onto it
    over its finite samples (NaN outside them).  Returns (positions,
    [profile, ...]).
    """
    base = np.asarray(positions[0], dtype=np.float64)
    out = []
    for prof, pos in zip(profiles, positions):
        pos = np.asarray(pos, dtype=np.float64)
        if len(pos) == len(base) and np.allclose(pos, base, atol=1e-6):
            out.append(np.asarray(prof, dtype=np.float64))
            continue
        ok = np.isfinite(prof)
        if ok.sum() < 2:
            out.append(np.full(len(base), np.nan))
            continue
        y = np.interp(base, pos[ok], np.asarray(prof)[ok], left=np.nan, right=np.nan)
        out.append(y)
    return base, out


class Line:
    """
    A grid line's dI/dV or IETS against position: the data of one line of an
    SM4 grid file (extract_data) and the interactive window
    (plot_line_with_topo_interactive).
    """

    # The Feenstra companion shows heatmaps (its own 'γ' boxes and pipeline box).
    feenstra_heatmaps = True
    M_COLORS = widgets.MarkerGroupControls.M_COLORS
    STACK_OFFSET_FRAC = 0.25   # stack: offset per curve inside a block, fraction of the largest curve range
    MAX_GROUPS = 10
    # The 'Markers' slider stack (widgets.MarkerGroupControls): slider k from the bottom at
    # SLIDER_Y0 + k * SLIDER_PITCH; the panels' bottom stays at PANEL_BOTTOM_MIN until the
    # stack's top plus PANEL_CLEARANCE would reach it, then follows the stack.
    SLIDER_X, SLIDER_Y0, SLIDER_PITCH = 0.255, 0.02, 0.025
    AVGS_XY = (0.60, 0.02)
    PANEL_BOTTOM_MIN = 0.20
    PANEL_CLEARANCE = 0.06
    TOPO_CBAR = [0.94, 0.35, 0.01, 0.35]        # the page's topo colour bar (plots.draw_topo_cbar)
    # Topo row controls: two columns of TOPO_ROWS_PER_COLUMN rows each (rows 1 to 4 at
    # TOPO_ROW_X[0], 5 to 8 at TOPO_ROW_X[1]), a 'rows' box and an 'f' button per row, filled
    # from the bottom; more than TOPO_ROWS_IN_FIGURE rows live in the 'topo rows' window.
    TOPO_ROW_X = (0.848, 0.924)
    TOPO_ROW_GAP = 0.002
    TOPO_ROW_Y0, TOPO_ROW_PITCH = 0.105, 0.035
    TOPO_ROWS_PER_COLUMN = 4
    TOPO_ROWS_IN_FIGURE = 8
    TOPO_ROW_CLEARANCE = 0.03       # the panels' bottom stays this far above the row stack
    TOPO_PASS_LETTER = {'forward': 'f', 'reverse': 'r'}
    # Several sweeps per pixel (set by __init__; a subclass that builds its own data keeps None).
    spectra_all = None
    sweep_mode = None

    def __init__(self, ifile, chunk_index=0, cmap='jet', scalefactor=2.75, scan_mode='single', combine_scans=False,
                 sweeps=None):
        """
        Initialize the Line tool (a grid line's dI/dV or IETS against position).
          ifile         : Path to an SM4 file.
          chunk_index   : Which full line (if more than one exists) for the dI/dV data.
          cmap          : Colormap for plotting.
          scalefactor   : Scale factor for converting positions.
          scan_mode     : 'single' (one spectrum per position) or 'combined' (forward/backward interleaved).
          combine_scans : Only used when scan_mode='combined'. If True, average forward and backward scans.
                          If False, keep forward and backward separate for side-by-side plotting.
          sweeps        : a file with several sweeps per pixel (sweeps.py): the sweep view to open on
                          ('forward sweeps', 'reverse sweeps', 'all sweeps', 'sweep n'); None: the file's default.
                          The window's 'sweeps' radio changes it.
        """
        self.ifile = ifile
        self.chunk_index = chunk_index
        self.cmap = cmap
        self.scalefactor = scalefactor
        self._user_scan_mode = scan_mode
        self.combine_scans = combine_scans

        # Load the SM4 file and extract the spectroscopy data (the first channel the file holds
        # to start with: the dI/dV channel when it has one, else the IETS channel).
        self.f = sm4file.load(self.ifile)
        self.spectra = STS.from_pages(self.f.pages, 'RHK_PAGE_INTERACTIVE_SPECTRA',
                                      name=os.path.basename(self.ifile))
        if self.spectra is None:
            raise ValueError(f"{os.path.basename(self.ifile)}: no dI/dV or IETS spectroscopy page found.")
        # Several sweeps per pixel: the window works on a view (one column per pixel).
        self.spectra_all = None
        self.sweep_mode = None
        if getattr(self.spectra, 'sweep_info', None) is not None:
            from sm4_Browser import sweeps as sweeps_mod
            self.spectra_all = self.spectra
            modes = sweeps_mod.sweep_modes(self.spectra_all.sweep_info)
            self.sweep_mode = sweeps if sweeps in modes else modes[0]
            self.spectra = self.spectra_all.sweep_view(self.sweep_mode)
        self.channel = self.spectra.channels()[0]
        self.extract_data()

    def _fast_page(self):
        """The first topographic page of the file (by page type); the fast axis and the position axis come from it."""
        return fast_page_of(self.f.pages, os.path.basename(self.ifile))

    def extract_data(self):
        """
        Extract the channel's columns (LIAcurrent: the dI/dV or, with
        self.channel 'iets', the IETS page) and I(V) current data, energy, and
        position, in physical units (the loader's data, float64), with their
        units, the modulation amplitude, and the value of a never-written
        sample of the channel (its page's RHK_Zoffset).  The spectra per line
        are the recorded grid size RHK_GridXsize (STS.grid_x); a file that
        does not record it is refused.
        """
        fast = self._fast_page()
        size = self._line_size()
        # The position axis in Å: every spectrum of the line at its recorded position along the
        # line, from the line's first spectrum (line_axis; the fast page's axis without positions).
        self.pos = line_axis(self.spectra.spec_xy_m, size, self.chunk_index, self.scalefactor, fast_page=fast)

        self.energy = self.spectra.energy.copy()
        self.LIAcurrent = self.spectra.channel(self.channel).copy()  # shape: (npts, total_scans)
        total_scans = self.LIAcurrent.shape[1]
        # The kept spectra of the file (STS.acquired: no column all zeros or ending in zeros).
        kept_all = np.asarray(self.spectra.acquired, dtype=bool)

        # --- The I(V) current spectroscopy channel (STS: present when its shape matches) ---
        if self.spectra.iv is not None:
            self.iv_raw = self.spectra.iv.copy()  # same shape as LIAcurrent
        else:
            self.iv_raw = None
        self.units_lia = self.spectra.units_lia
        self.units_iets = self.spectra.units_iets
        self.units_iv = self.spectra.units_iv
        self.vmod = self.spectra.vmod
        self._unset = self.spectra.unset_value(self.channel)   # a never-written sample of the channel

        # --- Store bias / current setpoint from metadata ---
        self.V_bias = self.spectra.V_bias
        self.I_setpoint = self.spectra.I_setpoint

        # --- Chunk the LIA data (and I(V) in parallel) ---
        if self._user_scan_mode == 'single':
            if total_scans % size != 0:
                raise ValueError(f"scan_mode='single' but total_scans ({total_scans}) is not a multiple of size ({size}).")
            n_chunks = total_scans // size
            if n_chunks == 1:
                self.chunk = self.LIAcurrent.copy()
                self.iv_chunk = self.iv_raw.copy() if self.iv_raw is not None else None
                self.kept = kept_all.copy()
            else:
                if self.chunk_index < 0 or self.chunk_index >= n_chunks:
                    raise ValueError(f"chunk_index {self.chunk_index} out of range. Only {n_chunks} chunks available.")
                sl = slice(self.chunk_index * size, (self.chunk_index + 1) * size)
                self.chunk = self.LIAcurrent[:, sl]
                self.iv_chunk = self.iv_raw[:, sl] if self.iv_raw is not None else None
                self.kept = kept_all[sl].copy()
            self.scan_mode = 'single'
        elif self._user_scan_mode == 'combined':
            expected_scans = 2 * size
            if total_scans % expected_scans != 0:
                raise ValueError(f"scan_mode='combined' but total_scans ({total_scans}) is not a multiple of 2*size ({expected_scans}).")
            n_chunks = total_scans // expected_scans
            if self.chunk_index < 0 or self.chunk_index >= n_chunks:
                raise ValueError(f"chunk_index {self.chunk_index} out of range. Only {n_chunks} chunks available.")
            sl = slice(self.chunk_index * expected_scans, (self.chunk_index + 1) * expected_scans)
            lia_chunk = self.LIAcurrent[:, sl]
            iv_chunk_raw = self.iv_raw[:, sl] if self.iv_raw is not None else None
            kc = kept_all[sl]
            self.kept = kc[0::2] & kc[1::2]
            if self.combine_scans:
                self.chunk = (lia_chunk[:, 0::2] + lia_chunk[:, 1::2]) / 2.0
                self.iv_chunk = ((iv_chunk_raw[:, 0::2] + iv_chunk_raw[:, 1::2]) / 2.0
                                 if iv_chunk_raw is not None else None)
                self.scan_mode = 'combined'
            else:
                self.forward = lia_chunk[:, 0::2]
                self.backward = lia_chunk[:, 1::2]
                if iv_chunk_raw is not None:
                    self.iv_forward = iv_chunk_raw[:, 0::2]
                    self.iv_backward = iv_chunk_raw[:, 1::2]
                else:
                    self.iv_forward = None
                    self.iv_backward = None
                self.scan_mode = 'separate'
                self.chunk = None
                self.iv_chunk = None
        else:
            raise ValueError(f"scan_mode must be 'single' or 'combined' (got '{self._user_scan_mode}').")

    def _line_size(self):
        """The spectra per line: the recorded grid size RHK_GridXsize (STS.grid_x); refused when not recorded."""
        size = self.spectra.grid_x
        if size is None:
            raise ValueError(f"{os.path.basename(self.ifile)}: the file records no grid size (RHK_GridXsize).")
        return int(size)

    def _pipeline(self, raw, iv, hampel_filter_params, sgolay_filter_params, apply_calibration,
                  divide_iv, iv_broadening, apply_normalize, return_areas=False, iv_zero_sub=False,
                  norm_range=None):
        """
        The package's one pipeline (STS.process_arrays) on this line's
        columns (npts, N), each step when switched on: the I(V) through the
        Hampel and Savitzky-Golay filters once (the columns themselves are
        never filtered), the calibration against that current
        (calibration.calibrate for dI/dV, calibrate_iets for IETS, of
        whichever calibration file is in place: calibration_zeroref,
        calibration_gainoffset, or calibration_stretch), the
        division by the broadened I/V of the same current (dI/dV only;
        iv_broadening 0 the plain I/V; before iv_zero_sub subtracts its
        value at 0 V, feenstra.iv_corrected), unit-area normalization (over
        the energy range norm_range (lo, hi) when one is given: the window's
        'E range' box; else the whole axis).  A column that is
        never written (every sample the page's offset, self._unset) was never acquired: it is left out of the regression and
        comes back as NaN (never drawn); a line with no acquired column at
        all is refused, as is the calibration of a file without the I(V).
        With return_areas, returns (curves, areas), the areas of the
        normalization (NaN for never-acquired columns).  The kept columns are
        the file's (STS.acquired, this line's slice in self.kept: no column
        never written or ending in never-written samples); for columns of
        another width the never-written test alone.
        """
        raw = np.asarray(raw, dtype=np.float64)
        kept = getattr(self, 'kept', None)
        if kept is not None and len(kept) == raw.shape[1]:
            acquired = np.asarray(kept, dtype=bool)
        else:
            acquired = np.any(raw != self._unset, axis=0)
        if not acquired.any():
            raise ValueError("no kept spectrum on this line.")
        if apply_calibration and iv is None:
            raise ValueError(
                "No I(V) current channel found in this SM4 file. "
                "Cannot perform linear-regression calibration."
            )
        curves, areas = STS.process_arrays(self.energy, raw, iv, hampel_filter_params, sgolay_filter_params,
                                           divide_iv, iv_broadening, return_areas=True, channel=self.channel,
                                           acquired=acquired, iv_zero_sub=iv_zero_sub,
                                           normalize=apply_normalize, calibrate=apply_calibration,
                                           vmod=getattr(self, 'vmod', None), norm_range=norm_range)
        curves = np.array(curves, dtype=np.float64)
        areas = np.array(areas, dtype=np.float64)
        curves[:, ~acquired] = np.nan
        areas[~acquired] = np.nan
        return (curves, areas) if return_areas else curves

    def _topo_profiles(self, topo_lines):
        """
        The raw profiles of (row, mode) pairs with every pixel at its own
        recorded location (stripe_rows on the file's Sm4File: the pixels'
        along-line coordinates from this line's first spectrum, a forward
        pixel kept only when the spectrum at its location is kept) and the
        resolved (row, mode) list.  Returns (topo_x, [profile, ...],
        resolved), topo_x the pixels' position axis (the first row's; other
        rows interpolated onto it when their pixels sit elsewhere,
        common_stripe_axis).  Raises on a bad row or mode.
        """
        profiles, positions, resolved = stripe_rows(self._int_sm4(), topo_lines, line=self.chunk_index,
                                                    scalefactor=self.scalefactor)
        topo_x, profiles = common_stripe_axis(profiles, positions)
        return np.asarray(topo_x, dtype=np.float64), profiles, resolved

    # ------------------------------------------------------------------ #
    #  The displayed line                                                  #
    # ------------------------------------------------------------------ #

    def _int_load_source(self):
        """The raw LIA and I(V) columns of the displayed line (the pipeline's input) from the current slicing."""
        if self.scan_mode in ['single', 'combined']:
            self._int_raw = self.chunk.copy()
            self._int_iv = self.iv_chunk
        elif self.scan_mode == 'separate':
            self._int_raw = (self.forward + self.backward) / 2.0
            self._int_iv = ((self.iv_forward + self.iv_backward) / 2.0
                            if self.iv_forward is not None and self.iv_backward is not None else None)
        else:
            raise ValueError("Unknown scan mode.")

    def _int_orient(self, c):
        """Columns (npts, N) of the line oriented like the heatmap (row j = spectrum j at self.pos[j], energy ascending)."""
        c = np.asarray(c, dtype=np.float64).T
        if self.energy[0] > self.energy[-1]:
            c = c[:, ::-1]
        return c

    def _int_pipeline(self, hampel_filter_params, sgolay_filter_params):
        """
        The displayed line's curves for given filter settings (None = off):
        the pipeline (_pipeline) with the window's calibrate / normalize
        flags, the division when the window's 'Feenstra' toggle is on, and
        the Feenstra settings, the normalization over the 'E range' box's
        range, oriented like the heatmap.  Returns an
        (n_positions, n_energies) array.
        """
        curves = self._pipeline(self._int_raw, self._int_iv, hampel_filter_params, sgolay_filter_params,
                                self._int_apply_calibration, self._int_divide_iv, self._int_iv_width(),
                                self._int_apply_normalize, iv_zero_sub=self._int_iv_zero_sub,
                                norm_range=self._int_e_range)
        return self._int_orient(curves)

    def _int_source_label(self):
        """The window's source for titles: '<file>: line k'."""
        return f"{os.path.basename(self.ifile)}: line {self.chunk_index}"

    def _int_heat_title(self):
        return self._int_quantity() + f", line {self.chunk_index}"

    # --- the Feenstra settings (the companion's '0Vsub', 'broadened', and width boxes) ---

    def _int_iv_width(self):
        """The broadening width the division uses (feenstra.feenstra_width)."""
        return feenstra_width(self._int_iv_broadening, self._int_iv_broadened)

    def feenstra_settings(self):
        """The window's Feenstra settings: {'zero_sub', 'broadened', 'broadening'}."""
        return {'zero_sub': self._int_iv_zero_sub, 'broadened': self._int_iv_broadened,
                'broadening': self._int_iv_broadening}

    def set_feenstra_settings(self, zero_sub=None, broadened=None, broadening=None):
        """
        A Feenstra setting changed (the companion's boxes; None = unchanged):
        the heatmap and the marker curves are recomputed with it (the
        division when the window's toggle is on), the companion refreshed,
        and every fit sent from this window receives it
        (Fit.set_feenstra).  Returns False (nothing changed) when the line
        cannot be processed with it.
        """
        new = (self._int_iv_zero_sub if zero_sub is None else bool(zero_sub),
               self._int_iv_broadened if broadened is None else bool(broadened),
               self._int_iv_broadening if broadening is None else float(broadening))
        old = (self._int_iv_zero_sub, self._int_iv_broadened, self._int_iv_broadening)
        if new == old:
            return True
        self._int_iv_zero_sub, self._int_iv_broadened, self._int_iv_broadening = new
        note = ("0 V subtraction " + ('on' if new[0] else 'off') if new[0] != old[0]
                else "broadening " + ('on' if new[1] else 'off') if new[1] != old[1]
                else f"broadening {new[2]:g} V")
        if not self._int_apply_feenstra(note):
            self._int_iv_zero_sub, self._int_iv_broadened, self._int_iv_broadening = old
            return False
        for an in list(self._int_fits):
            try:
                an.set_feenstra(self._int_iv_width(), self._int_iv_zero_sub)
            except Exception as exc:
                print(f"{self._int_tag()}: a fit did not take the Feenstra settings: {exc}")
        return True

    def _int_tag(self):
        return "Line window"

    def _int_apply_feenstra(self, note):
        """
        Recompute the heatmap with the current Feenstra state, Feenstra
        settings, and pipeline flags and redraw the marker curves (and the
        companion).  Returns False, with the reason printed, when the line
        cannot be processed (nothing changed).
        """
        try:
            curves = self._int_pipeline(self._int_hampel_params, self._int_sgolay_params)
        except Exception as exc:
            print(f"{note} not applied: {exc}")
            return False
        self._int_curves = curves
        self._int_draw_heatmap()
        q = self._int_quantity()
        t = feenstra_titles(self._int_iv_zero_sub, self._int_iv_broadened, self._int_iv_broadening,
                            energy_has_zero(self.energy))
        if self.channel == 'iets':
            print(f"{note}: the division applies to dI/dV only; the {self._int_tag()} shows {q}.")
        else:
            print(f"{note}: the {self._int_tag()} shows " + (t['ratio'] if self._int_divide_iv else "dI/dV")
                  + ("" if self._int_apply_calibration else ", uncalibrated")
                  + ("" if self._int_apply_normalize else ", not normalized") + ".")
        self._redraw_curves()
        return True

    def _int_on_pipeline_change(self, state):
        """The 'calibrate' / 'normalize' box: recomputed through the Feenstra path; refused, the box goes back."""
        old = (self._int_apply_calibration, self._int_apply_normalize)
        self._int_apply_calibration, self._int_apply_normalize = state['calibrate'], state['normalize']
        note = (f"calibration {'on' if state['calibrate'] else 'off'}, "
                f"normalization {'on' if state['normalize'] else 'off'}")
        if not self._int_apply_feenstra(note):
            self._int_apply_calibration, self._int_apply_normalize = old
            self._int_pipeline_box.set_state(*old)
            self._int_fig.canvas.draw_idle()

    # --- the heatmap ---

    def _int_units(self):
        """The y-label units of the displayed quantity."""
        if self.channel != 'iets' and self._int_divide_iv:
            return "arb. units"
        return curve_units(self.channel, self._int_apply_calibration, self._int_apply_normalize,
                           self.units_lia, self.units_iets, self.units_iv)

    def _int_cbar_label(self):
        return self._int_quantity() + f" ({self._int_units()})"

    # --- the kept rows and the drift (the window's own 'auto drift' / 'manual drift' toggles, the drift window) ---

    def _int_kept_mask(self):
        """(n_pos,) bool: the kept spectra of the displayed line (self.kept; all when the widths differ)."""
        kept = getattr(self, 'kept', None)
        if kept is not None and len(kept) == len(self._int_pos):
            return np.asarray(kept, dtype=bool)
        return np.ones(len(self._int_pos), dtype=bool)

    def _int_kept_pos(self):
        """The positions (Å) of the kept spectra, the rows of the heatmap and the cells of the stripe."""
        return np.asarray(self._int_pos, dtype=np.float64)[self._int_kept_mask()]

    def _int_drift_active(self):
        return self._int_drift is not None and (self._int_auto_on or self._int_manual_on)

    def _int_shear(self):
        """
        The total skew of every kept row relative to the topo path, with a
        drift source on or the grid's fast-axis placement set (the stripe
        window): the row's along-line offset (its spectrum's placed
        position minus the recorded one, projected on the line: the stripe
        matching's placement and the slow-axis drift's along-line part,
        Drift.result with the stored fast-axis state) plus the within-sweep
        shear (Drift.row_shear: the along-line drift at the energy-cell
        edges of each sweep, the line's own forward compression where the
        placement measured one, the velocity otherwise), converted to the
        position axis's units and ordered as the heatmap's energy axis,
        (n_kept, n_e + 1); None with nothing to apply.  The stripe itself is
        never moved: the rows move relative to it.  The velocity's source is
        kept for the stripe lines' colour (_int_drift_source).
        """
        if self._int_drift is None:
            return None
        if self._int_shear_cache is None:
            sh = None
            try:
                sm4 = self._int_sm4()
                stripe = sm4.kind == 'grid' and self._int_drift.fast_state(sm4, None, self._int_auto_on, self._int_manual_on) != 'off'
                if self._int_drift_active() or stripe:
                    cols = np.flatnonzero(self._int_kept_mask()) + self.chunk_index * len(self._int_pos)
                    sh = self._int_drift.row_shear(sm4, cols, None, None, self._int_auto_on, self._int_manual_on)
                    res = self._int_drift.result(sm4, None, None, self._int_auto_on, self._int_manual_on)
                    self._int_drift_source = res.get('source') if res is not None else None
                    if res is not None:
                        # Every row's along-line offset: where its spectrum was placed against where it was recorded.
                        fast, _ = sm4.grid_axes()
                        xy = np.asarray(sm4.spec_xy_m, dtype=np.float64)
                        corr = np.asarray(res['xy_corr'], dtype=np.float64)
                        off = np.full(len(cols), np.nan)
                        ok = cols < min(len(xy), len(corr))
                        off[ok] = (corr[cols[ok]] - xy[cols[ok]]) @ fast
                        off = np.where(np.isfinite(off), off, 0.0)
                        if sh is None:
                            sh = np.zeros((len(cols), len(self.energy) + 1))
                        sh = np.asarray(sh, dtype=np.float64) + off[:, None]
            except Exception as exc:
                print(f"{self._int_tag()}: drift shear not available ({exc}); rows unsheared.")
                sh = None
            if sh is not None:
                sh = np.asarray(sh, dtype=np.float64) * 1e10 / self.scalefactor
                if self.energy[0] > self.energy[-1]:
                    sh = sh[:, ::-1]
            self._int_shear_cache = sh
        return self._int_shear_cache

    def _int_apply_drift_toggle(self):
        """The toggles' states applied: the shear recomputed, the heatmap rebuilt, the stripe redrawn, the companion refreshed."""
        self._int_shear_cache = None
        if self._int_im is not None:
            self._int_im.set_shear(self._int_shear())
            self._int_set_heat_limits()
        self._redraw_topo_stripe()

    def _int_on_drift_toggle(self, which, flag):
        """The window's 'auto drift' / 'manual drift' toggles: on only with a drift model at hand (else the toggle goes back)."""
        flag = bool(flag)
        chk = {'auto': self._int_chk_auto, 'manual': self._int_chk_manual}[which]
        if flag and self._int_drift is None:
            print(f"{self._int_tag()}: no drift model at hand; the toggle stays off.")
            chk.set(False)
            self._int_fig.canvas.draw_idle()
            return
        current = {'auto': self._int_auto_on, 'manual': self._int_manual_on}[which]
        if flag == current:
            return
        if which == 'auto':
            self._int_auto_on = flag
        else:
            self._int_manual_on = flag
        self._int_apply_drift_toggle()
        print(f"{self._int_tag()}: {which} drift {'on' if flag else 'off'}"
              + (" (rows sheared by the drift during their sweeps; the stripe untouched)."
                 if self._int_drift_active() else " (rows plain)."))

    def _int_on_drift_window(self):
        """'drift window': the file's before and after topos with marker pairs (drift.DriftWindow, seeding from this window's 'auto drift'); a change applied there reaches every subscriber of the Drift, this window included."""
        if self._int_drift is None:
            print(f"{self._int_tag()}: no drift model at hand; no drift window.")
            return
        try:
            w = DriftWindow(self._int_drift, self._int_sm4(), self.scalefactor, list(self._int_topo_active),
                            auto_on=self._int_auto_on)
        except ValueError as exc:
            print(f"{self._int_tag()}: drift window not opened ({exc}).")
            return
        self._int_drift_windows.append(w)

    def _int_on_topo_row_profile(self, k):
        """
        Row k's 'profile' button: the raw forward and reverse passes of that
        row's line in one profile window (topo.ProfileWindow): every
        acquired pixel of each pass at its recorded position along the line
        from the line's first spectrum (stripe_rows with kept_only False,
        nm after the scale factor), heights raw (in metres as the loader gives them, no
        processor, no normalization), forward and reverse in their own
        colours.
        """
        if k >= len(self._int_topo_lines):
            return
        row = int(self._int_topo_lines[k][0])
        sm4 = self._int_sm4()
        try:
            modes = [(row, 'forward')] + ([(row, 'reverse')] if len(sm4.topo_pages) > 1 else [])
            profiles, positions, resolved = stripe_rows(sm4, modes, line=row, scalefactor=self.scalefactor, kept_only=False)
        except Exception as exc:
            print(f"{self._int_tag()}: profile of line {row} not available ({exc}).")
            return
        curves = []
        pair = {'label': f"{os.path.basename(self.ifile)}, line {row}, row {row}", 'forward': None, 'reverse': None}
        for prof, pos, (r, mode) in zip(profiles, positions, resolved):
            color = plots.PROFILE_FORWARD_COLOR if mode == 'forward' else plots.PROFILE_REVERSE_COLOR
            x, z = np.asarray(pos) / 10.0, np.asarray(prof) / 10.0
            curves.append((mode, x, z, color))
            pair[mode] = (x, z)
        title = f"Topography profile: {os.path.basename(self.ifile)}, line {row}, raw, forward and reverse"
        try:
            w = topo.ProfileWindow(curves, title, xlabel="position along the line (nm)", ylabel="height (nm, raw)",
                                   pair=pair, registry=self._int_profile_plots)
        except Exception as exc:
            print(f"{self._int_tag()}: profile window not opened ({exc}).")
            return
        self._int_profile_windows.append(w)

    # --- the heatmap ---

    def _int_set_heat_limits(self):
        """
        The heatmap's limits: the energy axis and the kept rows' edges (the
        shear included), or, where typed, the 'E range' box's energy range
        and the 'x range' box's position range (the stripe and the energy
        slice share the position axis).
        """
        if self._int_im is None:
            return
        x0, x1, y0, y1 = self._int_im.extent()
        if self._int_e_range is not None:
            x0, x1 = self._int_e_range
        if self._int_x_range is not None:
            y0, y1 = self._int_x_range
        self._int_ax_didv.set_xlim(x0, x1)
        self._int_ax_didv.set_ylim(y0, y1)

    def _int_draw_heatmap(self):
        """
        The heatmap from self._int_curves: the kept spectra's rows at their
        positions (plots.RowHeatmap; the colour scale from the kept rows,
        inside the 'E range' box's range when one is typed
        (plots.energy_range_scale), the 'γ' box's exponent; with the drift toggle on each row sheared
        along the position axis by the drift during its sweep), the colour
        bar rebuilt, the title and label set.
        """
        kept = self._int_kept_mask()
        img = np.asarray(self._int_curves, dtype=np.float64)[kept]
        lo, hi = plots.energy_range_scale(img, self._int_energy, self._int_e_range)
        if self._int_im is None:
            self._int_im = plots.RowHeatmap(self._int_ax_didv, img, self._int_energy, self._int_kept_pos(),
                                            self.cmap, gamma=self._int_gamma_box.value, shear=self._int_shear(),
                                            vmin=lo, vmax=hi)
            self._int_set_heat_limits()
            self._int_ax_didv.set_xlabel("Energy (eV)")
        else:
            plots.update_heatmap(self._int_im, img, gamma=self._int_gamma_box.value, vmin=lo, vmax=hi)
        plots.draw_colorbar(self._int_fig, self._int_im, self._int_cax_didv, label=self._int_cbar_label())
        self._int_ax_didv.set_title(self._int_heat_title())

    def _int_on_gamma(self, value):
        """The heatmap's 'γ' box: the colour map's exponent (plots.heat_norm), over the 'E range' box's cells."""
        lo, hi = plots.energy_range_scale(self._int_im.get_array(), self._int_energy, self._int_e_range)
        plots.update_heatmap(self._int_im, gamma=value, vmin=lo, vmax=hi)
        plots.draw_colorbar(self._int_fig, self._int_im, self._int_cax_didv, label=self._int_cbar_label())
        self._int_fig.canvas.draw_idle()

    def _int_on_e_range(self, rng):
        """
        The 'E range' box (widgets.RangeBox; (lo, hi) in V, None the full
        axis): the shown energy range of the heatmap and the curve panel,
        the normalization range (the pipeline rerun: _int_apply_feenstra,
        the companion with it), and the colour scale's.  A range holding
        fewer than two recorded energies is refused (False: the box goes
        back), as is a range the line cannot be processed with.
        """
        if rng is not None:
            e = self._int_e_axis()
            if int(np.count_nonzero((e >= rng[0]) & (e <= rng[1]))) < 2:
                print(f"E range: {rng[0]:g}, {rng[1]:g} V holds fewer than two recorded energies; the box is put back.")
                return False
        old = self._int_e_range
        self._int_e_range = rng
        if not self._int_apply_feenstra("E range"):
            self._int_e_range = old
            return False
        self._int_set_heat_limits()
        self._int_fig.canvas.draw_idle()
        return True

    def _int_on_x_range(self, rng):
        """The 'x range' box ((lo, hi) in Å, None the kept rows): the shown position range of the stripe, the heatmap, and the energy slice; nothing is recomputed."""
        self._int_x_range = rng
        self._int_set_heat_limits()
        self._int_fig.canvas.draw_idle()

    # --- topo stripe source: rows (line index, forward / reverse) ---

    def _int_set_topo_lines(self, topo_lines):
        """
        Extract the given (row, mode) profiles into self._int_topo_profiles
        (every pixel at its own recorded location, self._int_topo_x its
        position axis; the pixels of spectra not kept NaN) and
        self._int_topo_lines; a row of None means the middle row.  Raises on a bad row or mode, leaving the
        previous rows in place.  The stripe is built from them by
        _redraw_topo_stripe (per-row processing, range matching, mean).
        """
        topo_x, profiles, resolved = self._topo_profiles(topo_lines)
        self._int_topo_x = np.asarray(topo_x, dtype=np.float64)
        self._int_topo_profiles = [p.copy() for p in profiles]
        self._int_topo_lines = resolved

    def _int_max_topo_rows(self):
        """The hard maximum of rows: 2 x the lines recorded in the file (a forward and a reverse pass each)."""
        return 2 * self._int_n_chunks()

    def _int_apply_topo_rows(self, rows, note):
        """
        Make `rows` the stripe's rows: on success the controls follow and the
        stripe is redrawn; on failure the reason is printed and the controls
        go back to the current rows.
        """
        try:
            self._int_set_topo_lines(list(rows))
        except Exception as exc:
            print(f"Topo rows ({note}) not applied: {exc}")
            self._int_refresh_topo_rows()
            return False
        self._int_refresh_topo_rows()
        self._redraw_topo_stripe()
        return True

    def _int_on_topo_row_edit(self, k, text):
        """Row k's line-index box."""
        if k >= len(self._int_topo_lines):
            return
        try:
            row = int(float(str(text).strip()))
        except ValueError:
            print(f"Topo row {k + 1}: '{text}' is not a row index.")
            self._int_refresh_topo_rows()
            return
        rows = list(self._int_topo_lines)
        rows[k] = (row, rows[k][1])
        self._int_apply_topo_rows(rows, f"row {k + 1} = {row}")

    def _int_on_topo_row_toggle(self, k):
        """Row k's f/r button: the other pass."""
        if k >= len(self._int_topo_lines):
            return
        rows = list(self._int_topo_lines)
        r, mode = rows[k]
        rows[k] = (r, 'reverse' if mode == 'forward' else 'forward')
        self._int_apply_topo_rows(rows, f"row {k + 1} {self.TOPO_PASS_LETTER[rows[k][1]]}")

    def _int_on_topo_add(self):
        """'+': one more row (the current line, forward), up to 2 x the lines in the file."""
        n_max = self._int_max_topo_rows()
        if len(self._int_topo_lines) >= n_max:
            print(f"Topo rows: at most {n_max} rows (2 x {self._int_n_chunks()} lines in the file).")
            return
        rows = list(self._int_topo_lines) + [(self.chunk_index, 'forward')]
        if self._int_apply_topo_rows(rows, f"+ row {len(rows)}") and len(rows) > self.TOPO_ROWS_IN_FIGURE:
            self._int_topo_rows_window()

    def _int_on_topo_remove(self):
        """'-': the last row goes; one row always stays."""
        if len(self._int_topo_lines) <= 1:
            print("Topo rows: one row stays.")
            return
        rows = list(self._int_topo_lines)[:-1]
        if self._int_apply_topo_rows(rows, f"- row {len(rows) + 1}") and len(rows) > self.TOPO_ROWS_IN_FIGURE:
            self._int_topo_rows_window()

    def _int_set_row_widgets(self, box, btn, k):
        """Put row k's values into a (box, button) pair without firing callbacks."""
        row, mode = self._int_topo_lines[k]
        box.set(str(row))
        btn.set_text(self.TOPO_PASS_LETTER[mode])

    def _int_refresh_topo_rows(self):
        """
        The row controls follow self._int_topo_lines: up to TOPO_ROWS_IN_FIGURE
        rows shown in the figure (the others hidden); with more rows every
        in-figure control is hidden, a note names the 'topo rows' window, and
        that window, when open, is brought up to date.  Raises the panels when
        the row stack would run into them (_int_place_panels).
        """
        row_widgets = getattr(self, '_int_topo_row_widgets', None)
        if row_widgets is None:
            return
        n = len(self._int_topo_lines)
        many = n > self.TOPO_ROWS_IN_FIGURE
        for k, (box, btn, prof) in enumerate(row_widgets):
            shown = (k < n) and not many
            box.show(shown)
            btn.show(shown)
            prof.show(shown)
            if shown:
                self._int_set_row_widgets(box, btn, k)
        self._int_topo_rows_note.set_visible(many)
        if many:
            self._int_topo_rows_note.set_text(f"{n} rows: see the 'topo rows' window")
            self._int_sync_rows_window()
        else:
            self._int_close_rows_window()
        self._int_place_panels()
        self._int_fig.canvas.draw_idle()

    def _int_topo_rows_window(self):
        """
        The separate 'topo rows' window: every row with its line-index box and
        f/r button (the same callbacks as the in-figure controls), used while
        more than TOPO_ROWS_IN_FIGURE rows exist.  Opened by '+' / '-' when
        needed; closed by hand it stays closed until the next '+' / '-'.
        """
        self._int_close_rows_window()
        n = len(self._int_topo_lines)
        fig = plt.figure(figsize=(self.ROWS_DESIGN[0], max(self.ROWS_DESIGN[1], self.ROWS_ROW_IN * n + self.ROWS_PAD_IN)))
        try:
            fig.canvas.manager.set_window_title('topo rows')
        except Exception:
            pass
        fig.text(0.05, 0.96, f"{os.path.basename(self.ifile)}: {n} topo rows (line index, f/r)",
                 fontsize=8, ha='left', va='top')
        pitch = min(0.9 / max(n, 1), 0.08)
        row_widgets = []
        w_box, _ = widgets.size_of(fig, 'rows')
        w_f, _ = widgets.size_of(fig, 'f')
        for k in range(n):
            y = 0.90 - (k + 1) * pitch
            fig.text(0.40, y, f"row {k + 1} ", fontsize=8, ha='right', va='bottom')
            box = widgets.make(fig, 'rows', 0.42, y, value=str(self._int_topo_lines[k][0]))
            btn = widgets.make(fig, 'f', 0.42 + w_box + 0.03, y)
            btn.set_text(self.TOPO_PASS_LETTER[self._int_topo_lines[k][1]])
            prof = widgets.make(fig, 'profile', 0.42 + w_box + 0.03 + w_f + 0.03, y)
            box.on_change(lambda text, k=k: self._int_on_topo_row_edit(k, text))
            btn.on_change(lambda k=k: self._int_on_topo_row_toggle(k))
            prof.on_change(lambda k=k: self._int_on_topo_row_profile(k))
            row_widgets.append((box, btn, prof))
        self._int_rows_fig = fig
        self._int_rows_widgets = row_widgets
        fig.canvas.draw_idle()
        plt.show(block=False)

    def _int_sync_rows_window(self):
        """Bring the open 'topo rows' window up to date (rebuilt when the row count changed)."""
        fig = getattr(self, '_int_rows_fig', None)
        if fig is None or not widgets.figure_is_open(fig):
            return
        if len(self._int_rows_widgets) != len(self._int_topo_lines):
            self._int_topo_rows_window()
            return
        for k, (box, btn, _prof) in enumerate(self._int_rows_widgets):
            self._int_set_row_widgets(box, btn, k)
        fig.canvas.draw_idle()

    def _int_close_rows_window(self):
        fig = getattr(self, '_int_rows_fig', None)
        if fig is not None and widgets.figure_is_open(fig):
            plt.close(fig)
        self._int_rows_fig = None
        self._int_rows_widgets = []

    def _int_rows_stack_top(self):
        """The top of the in-figure topo row stack (0 when hidden or absent)."""
        n = len(getattr(self, '_int_topo_lines', []))
        if not getattr(self, '_int_topo_row_widgets', None):
            return 0.0
        n_rows = 0 if n > self.TOPO_ROWS_IN_FIGURE else min(n, self.TOPO_ROWS_PER_COLUMN)   # hidden stack: no height
        if n_rows == 0:
            return 0.0
        _, h = widgets.size_of(self._int_fig, 'rows')
        return self.TOPO_ROW_Y0 + (n_rows - 1) * self.TOPO_ROW_PITCH + h

    def _int_place_panels(self):
        """
        The panels' bottom: PANEL_BOTTOM_MIN, or higher when the 'Markers'
        slider stack (plus PANEL_CLEARANCE) or the topo row stack (plus
        TOPO_ROW_CLEARANCE) would otherwise run into the panels.
        """
        slider_top = self._int_groups_ctl.stack_top()
        rows_top = self._int_rows_stack_top()
        bottom = max(self.PANEL_BOTTOM_MIN, slider_top + self.PANEL_CLEARANCE,
                     rows_top + self.TOPO_ROW_CLEARANCE if rows_top > 0 else 0.0)
        if abs(bottom - getattr(self, '_int_panel_bottom', self.PANEL_BOTTOM_MIN)) > 1e-9:
            self._int_panel_bottom = bottom
            self._int_fig.subplots_adjust(bottom=bottom)
            self._int_gamma_box.attach(self._int_ax_didv)

    def _int_n_chunks(self):
        """Number of lines (chunks) in the file for the current scan mode."""
        size = self._line_size()
        total = self.LIAcurrent.shape[1]
        per = size if self._user_scan_mode == 'single' else 2 * size
        return max(total // per, 1)

    def _int_set_line(self, new, text):
        """
        Show another line (chunk index): re-slice the file at it, recompute
        the heatmap and the marker curves with the current settings, and
        move the topo rows along when they were the line's own rows.  A bad
        index is refused (the reason printed; the box put back).
        """
        old = self.chunk_index
        try:
            new = int(new)
            n = self._int_n_chunks()
            if new < 0 or new >= n:
                raise ValueError(f"line must be 0..{n - 1}")
            self.chunk_index = new
            self.extract_data()
            self._int_load_source()
            self._int_curves = self._int_pipeline(self._int_hampel_params, self._int_sgolay_params)
        except Exception as exc:
            print(f"Line '{text}' not applied: {exc}")
            self.chunk_index = old
            self.extract_data()
            self._int_load_source()
            self._int_box_line.set(str(self.chunk_index))
            self._int_fig.canvas.draw_idle()
            return
        # The new line's own positions and kept rows: the heatmap is rebuilt (its rows change).
        self._int_pos = np.asarray(self.pos, dtype=np.float64)
        self._int_shear_cache = None
        if self._int_im is not None:
            self._int_im.remove()
            self._int_im = None
        lo, hi = self._int_pos_bounds()
        self._int_marker_pos = [float(np.clip(p, lo, hi)) for p in self._int_marker_pos]
        self._int_draw_heatmap()
        # Topo rows that were the previous line's own rows follow the new line.
        if all(r == old for r, _ in self._int_topo_lines):
            try:
                self._int_set_topo_lines([(new, m) for _, m in self._int_topo_lines])
            except Exception as exc:
                print(f"Topo rows for line {new} not applied: {exc}")
            self._int_refresh_topo_rows()
        self._int_box_line.set(str(self.chunk_index))
        self._redraw_topo_stripe()
        cb = getattr(self, '_int_line_callback', None)
        if cb is not None:
            try:
                cb(self.chunk_index)
            except Exception as exc:
                print(f"line change callback: {exc}")

    def _int_on_line_index(self, text):
        """The 'line' box: the entered chunk index."""
        try:
            new = int(float(text))
        except ValueError:
            print(f"Line '{text}' not applied: not a line index.")
            self._int_box_line.set(str(self.chunk_index))
            self._int_fig.canvas.draw_idle()
            return
        self._int_set_line(new, text)

    def _int_on_line_step(self, d):
        """'< line' / 'line >': the previous / next line (the box follows and stays editable)."""
        new = self.chunk_index + int(d)
        n = self._int_n_chunks()
        if new < 0 or new >= n:
            print(f"line {new}: the file has lines 0..{n - 1}.")
            return
        self._int_set_line(new, str(new))

    def _int_on_filter_change(self):
        """A filter toggle or edit: recompute the heatmap and the marker curves."""
        hp, sp = self._int_filters.params()
        try:
            curves = self._int_pipeline(hp, sp)
        except Exception as exc:
            print(f"Filter controls: settings not applied ({exc}).")
            return
        self._int_hampel_params = hp
        self._int_sgolay_params = sp
        self._int_curves = curves
        self._int_draw_heatmap()
        self._redraw_curves()

    def _int_quantity(self):
        """Name of the displayed quantity: dI/dV, (dI/dV)/(I/V), or d2I/dV2 (IETS), with the flags that are off."""
        if self.channel == 'iets':
            q = "d2I/dV2 (IETS)"
        else:
            q = "(dI/dV)/(I/V)" if self._int_divide_iv else "dI/dV"
        return q + ("" if self._int_apply_calibration else ", uncalibrated") \
            + ("" if self._int_apply_normalize else ", not normalized")

    def _int_on_channel(self, label):
        """
        The 'channel' radio: the line is re-sliced from the chosen channel
        (dI/dV or IETS) and the heatmap and the marker curves recomputed with
        the current settings; a file without the channel puts the radio back.
        """
        channel = {v: k for k, v in STS.CHANNEL_NAMES.items()}.get(label, 'didv')
        if channel == self.channel:
            return
        old = self.channel
        try:
            self.channel = channel
            self.extract_data()
            self._int_load_source()
            curves = self._int_pipeline(self._int_hampel_params, self._int_sgolay_params)
        except Exception as exc:
            print(f"channel {label} not applied: {exc}")
            self.channel = old
            self.extract_data()
            self._int_load_source()                 # the raw columns go back to the old channel with it
            self._int_rad_channel.set(STS.CHANNEL_NAMES[old])
            self._int_fig.canvas.draw_idle()
            return
        self._int_curves = curves
        self._int_draw_heatmap()
        print(f"channel {label}: the {self._int_tag()} shows {self._int_quantity()}.")
        self._redraw_curves()

    def _redraw_topo_stripe(self):
        """
        Recompute the stripe from the raw row profiles (each through the
        active processor, range matched, averaged, min-shifted:
        topo.stripe_profile) and redraw it (plots.draw_topo_stripe).
        """
        profile = topo.stripe_profile(self._int_topo_x, self._int_topo_profiles, self._int_topo_active)
        self._int_stripe_profile = profile          # the companion draws the same stripe
        plots.draw_topo_stripe(self._int_ax_topo, self._int_topo_x, profile)    # cells at the pixels' own positions
        self._int_set_heat_limits()                 # the stripe shares the y axis: the kept rows' range
        # The stripe is never moved by the drift; with the toggle on every kept row's drift line is
        # drawn on it: from the row's anchor through the along-line drift of its sweep at every
        # energy-cell edge (the same shear the row's pixels get).
        sh = self._int_shear()
        if sh is not None and len(sh):
            plots.draw_stripe_drift_lines(self._int_ax_topo, self._int_kept_pos(), sh,
                                          color=plots.drift_color(getattr(self, '_int_drift_source', None)))
        self._int_ax_topo.set_ylabel("Position (Å)")
        # Marker overlays are drawn exclusively by _redraw_curves()
        # so they are tracked in _int_marker_artists for clean removal.
        self._redraw_curves()

    # ------------------------------------------------------------------ #
    #  Marker groups                                                      #
    # ------------------------------------------------------------------ #

    @property
    def _int_avg_on(self):
        return self._int_stacks.average

    @property
    def _int_stack_on(self):
        return self._int_stacks.stack

    @property
    def _int_substack_on(self):
        return self._int_stacks.substack

    def _int_pos_step(self):
        """Signed spacing of the spectra along the position axis (Å)."""
        p = self._int_pos
        return float((p[-1] - p[0]) / max(len(p) - 1, 1))

    def _int_pos_bounds(self):
        """The position range of the kept spectra (the drawn rows)."""
        p = self._int_kept_pos()
        if len(p) == 0:
            p = self._int_pos
        return float(np.min(p)), float(np.max(p))

    def _int_layout_centers(self, n):
        """n group centres evenly from 10% to 90% of the kept part of the line."""
        p0, p1 = self._int_pos_bounds()
        rng = p1 - p0
        return list(np.linspace(p0 + 0.1 * rng, p1 - 0.1 * rng, n)) if n > 1 else [float(p0 + 0.5 * rng)]

    def _int_place_group(self, center, m):
        """m marker positions one spectrum step apart around `center`, clipped to the line."""
        lo, hi = self._int_pos_bounds()
        step = self._int_pos_step()
        return [float(np.clip(center + (k - (m - 1) / 2.0) * step, lo, hi)) for k in range(m)]

    def _int_groups(self):
        """Marker indices per group (membership by index, widgets.MarkerGroupControls.groups)."""
        return widgets.MarkerGroupControls.groups(self._int_group_sizes)

    def _int_group_of(self, i):
        return widgets.MarkerGroupControls.group_of(self._int_group_sizes, i)

    def _int_group_order(self):
        """
        Group indices ordered as the groups' average locations sit on the
        stripe and the heatmap, highest first (the order of the stack panel
        and of the 'Markers' slider stack, top slider first).
        """
        groups = self._int_groups()
        return sorted(range(len(groups)), key=lambda g: (-self._int_group_center(groups[g], g), g))

    DEFAULT_GROUP_SIZE = 2     # markers of a group added while there is no group to copy the size from

    def _int_group_center(self, members, g=None):
        """
        A group's centre: the mean of its markers; a group without markers (its
        'Markers' slider at 0) keeps the centre it last had (_int_group_ctr).
        """
        if members:
            return float(np.mean([self._int_marker_pos[i] for i in members]))
        ctr = getattr(self, '_int_group_ctr', [])
        if g is not None and g < len(ctr) and ctr[g] is not None:
            return float(ctr[g])
        lo, hi = self._int_pos_bounds()
        return 0.5 * (lo + hi)

    def _int_sync_centres(self):
        """The stored group centres follow the groups that hold markers (_int_group_center of an empty group)."""
        groups = self._int_groups()
        ctr = list(getattr(self, '_int_group_ctr', []))
        ctr = (ctr + [None] * len(groups))[:len(groups)]
        for g, members in enumerate(groups):
            ctr[g] = self._int_group_center(members, g)
        self._int_group_ctr = ctr

    def _int_marker_color(self, i):
        """Average off: one color per marker; Average on: the group's color."""
        k = self._int_group_of(i) if self._int_avg_on else i
        return self.M_COLORS[k % len(self.M_COLORS)]

    def _int_row_of(self, pos_val):
        """Row of the displayed curves (row j = spectrum j at self.pos[j]) nearest a position."""
        return int(np.argmin(np.abs(self._int_pos - pos_val)))

    def _int_col_of(self, pos_val):
        """Original column of the line (along self.pos) nearest a position; _int_row_of is its row."""
        return int(np.argmin(np.abs(self._int_pos - pos_val)))

    def _int_acquired(self, j):
        """Whether original column j holds a kept spectrum (STS.acquired: no column never written or ending in never-written samples)."""
        kept = getattr(self, 'kept', None)
        if kept is not None and j < len(kept) and len(kept) == self._int_raw.shape[1]:
            return bool(kept[j])
        return bool(np.any(self._int_raw[:, j] != self._unset) and self._int_raw[-1, j] != self._unset)

    def _int_rows_average(self, rows):
        """The plain mean of curves given as rows (n_curves, n_energies): Averages.mean_rows."""
        return Averages.mean_rows(rows)

    def _int_group_label(self, g, members):
        if not members:
            return f"avg {g + 1}: no markers"
        p = [self._int_marker_pos[i] for i in members]
        return f"avg {g + 1}: {min(p):.1f}–{max(p):.1f} Å"

    def _redraw_curves(self):
        """Redraw marker overlay lines on heatmap/topo and 1D curve panel."""
        # --- Remove old marker artists (Line2D only) ---
        for art in self._int_marker_artists:
            try:
                art.remove()
            except (ValueError, NotImplementedError):
                pass
        self._int_marker_artists = []

        self._int_sync_centres()
        groups = self._int_groups()

        # --- Draw marker lines on heatmap and topo stripe (per-marker or per-group color) ---
        # Picking: on the topo stripe a single marker is grabbed; on the heatmap, with
        # Average on, a group's average line is grabbed (and moves the whole group),
        # the member lines there being unpickable; with Average off the heatmap grabs
        # single markers as the stripe does.
        # (A Line2D is unpickable when no picker is given; picker=None is rejected.)
        for ax in (self._int_ax_didv, self._int_ax_topo):
            on_heatmap = ax is self._int_ax_didv
            marker_pick = {} if (self._int_avg_on and on_heatmap) else {'picker': 5}
            for i, pos_val in enumerate(self._int_marker_pos):
                h = ax.axhline(y=pos_val, color=self._int_marker_color(i), ls='--', lw=2, alpha=0.7,
                               label=f'marker_{i}', **marker_pick)
                self._int_marker_artists.append(h)
            # Average on: each group's average location as a thick black line
            # (draggable on the heatmap only).
            if self._int_avg_on:
                avg_pick = {'picker': 5} if on_heatmap else {}
                for g, members in enumerate(groups):
                    if not members:
                        continue                         # a group without markers has no average line
                    h = ax.axhline(y=self._int_group_center(members, g), color='black', ls='-', lw=2.5,
                                   alpha=0.9, label=f'_avg_{g}', **avg_pick)
                    self._int_marker_artists.append(h)

        # --- Redraw 1D curve panel (plots.draw_marker_curves, shared with the Feenstra companion) ---
        markers, group_items = self._int_curve_items(self._int_curves)
        plots.draw_marker_curves(self._int_ax_curve, self._int_energy, markers, group_items, self._int_avg_on,
                                 self._int_stack_on, self._int_substack_on, self._int_curve_texts(),
                                 offset_frac=self.STACK_OFFSET_FRAC)
        if self._int_e_range is not None:
            self._int_ax_curve.set_xlim(*self._int_e_range)       # the 'E range' box (the panel was cleared above)

        # The 'Markers' slider stack follows the groups' order on the stripe.
        self._int_refresh_group_sliders()
        self._int_redraw_e_markers(draw=False)
        self._int_fig.canvas.draw_idle()
        self._int_feenstra_refresh()

    # ------------------------------------------------------------------ #
    #  E-markers (energy markers) and the energy slice                     #
    # ------------------------------------------------------------------ #

    E_MAX = 8

    def _int_e_axis(self):
        return np.asarray(self._int_energy, dtype=np.float64)

    def _int_e_snap(self, x):
        """The recorded energy nearest x (the heatmap's ascending energy axis)."""
        e = self._int_e_axis()
        return float(e[int(np.argmin(np.abs(e - float(x))))])

    def _int_e_index(self, energy):
        e = self._int_e_axis()
        return int(np.argmin(np.abs(e - float(energy))))

    def _int_on_e_count(self, n):
        """
        The 'E-markers' slider (0 to 8; 'clear E-markers' sets 0): growing adds each new E-marker at the
        free slot of an even layout over the energy range farthest from the others; shrinking drops the
        last ones.
        """
        n = int(np.clip(n, 0, self.E_MAX))
        cur = list(self._int_e_markers)
        if n < len(cur):
            cur = cur[:n]
        elif n > len(cur):
            e = self._int_e_axis()
            lo, hi = float(e.min()), float(e.max())
            slots = [self._int_e_snap(v) for v in (np.linspace(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo), n)
                                                   if n > 1 else [0.5 * (lo + hi)])]
            dist = lambda a, b: abs(float(a) - float(b))
            cur.extend(widgets.MarkerGroupControls.grow_groups(n, cur, slots, dist))
        self._int_e_markers = [float(v) for v in cur]
        self._int_e_selected = min(self._int_e_selected, max(len(cur) - 1, 0))
        self._int_sld_e.set(len(cur))
        self._int_redraw_e_markers()

    def _int_on_key(self, event):
        """Left / Right: the selected E-marker one recorded energy down / up; Tab: the next E-marker selected."""
        if not self._int_e_markers or event.key not in ('left', 'right', 'tab'):
            return
        if event.key == 'tab':
            self._int_e_selected = (self._int_e_selected + 1) % len(self._int_e_markers)
        else:
            k = self._int_e_selected
            e = self._int_e_axis()
            i = int(np.clip(self._int_e_index(self._int_e_markers[k]) + (1 if event.key == 'right' else -1),
                            0, len(e) - 1))
            self._int_e_markers[k] = float(e[i])
        event.handled = True
        self._int_redraw_e_markers()

    def _int_e_readout(self):
        if not self._int_e_markers:
            return "E-markers: none"
        parts = [f"E{k + 1} {e:+.3f} V" + (" *" if k == self._int_e_selected else "")
                 for k, e in enumerate(self._int_e_markers)]
        d = [f"ΔE{k + 1}{k} {self._int_e_markers[k] - self._int_e_markers[k - 1]:+.3f} V"
             for k in range(1, len(self._int_e_markers))]
        return "   ".join(parts) + (("     " + "   ".join(d)) if d else "")

    def _int_redraw_e_markers(self, draw=True):
        """
        The E-markers as vertical dash-dot lines on the heatmap and the curve panel (their own colours,
        labelled E1, E2, ...; the selected one thicker), and the energy slice: for every E-marker the
        heatmap's quantity against position (the kept rows at their recorded positions), with the
        position markers as horizontal ticks in their colours.
        """
        for a in self._int_e_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._int_e_artists = []
        ax_s = self._int_ax_slice
        ax_s.clear()
        ax_s.tick_params(axis='y', left=False, labelleft=False)
        curves = getattr(self, '_int_curves', None)
        if curves is None:
            return
        rows = np.asarray(curves, dtype=np.float64)          # oriented like the heatmap: rows follow self.pos
        e_axis = self._int_e_axis()
        for k, e in enumerate(self._int_e_markers):
            col = plots.e_marker_color(k)
            lw = 2.4 if k == self._int_e_selected else 1.4
            for ax in (self._int_ax_didv, self._int_ax_curve):
                h = ax.axvline(x=e, color=col, ls='-.', lw=lw, alpha=0.95, label=f'_emarker_{k}', picker=5)
                self._int_e_artists.append(h)
            y0, y1 = self._int_ax_didv.get_ylim()
            t = self._int_ax_didv.text(e, max(y0, y1), f"E{k + 1}", color=col, fontsize=8, ha='center', va='bottom')
            self._int_e_artists.append(t)
        pos = np.asarray(self._int_pos, dtype=np.float64)
        kept = self._int_kept_mask()
        if rows.shape[0] == len(pos) and self._int_e_markers:
            for k, e in enumerate(self._int_e_markers):
                j = int(np.argmin(np.abs(e_axis - e)))
                prof = np.array(rows[:, j], dtype=np.float64)
                prof[~kept] = np.nan
                ax_s.plot(prof, pos, color=plots.e_marker_color(k), lw=1.3, marker='o', ms=2.5,
                          label=f"E{k + 1} {e:+.3f} V")
            for i, p in enumerate(self._int_marker_pos):
                ax_s.axhline(y=p, color=self._int_marker_color(i), ls='--', lw=0.8, alpha=0.8)
            ax_s.set_title("Energy slice: " + self._int_quantity(), fontsize=9)
            ax_s.set_xlabel(f"{self._int_quantity()}", fontsize=8)
        else:
            ax_s.text(0.5, 0.5, "no E-markers\n(raise 'E-markers')", ha='center', va='center',
                      transform=ax_s.transAxes, fontsize=8)
            ax_s.set_title("Energy slice", fontsize=9)
        ax_s.tick_params(labelsize=7)
        self._int_txt_e.set_text(self._int_e_readout())
        if draw:
            self._int_fig.canvas.draw_idle()

    def _int_curve_texts(self, quantity=None):
        """The texts of plots.draw_marker_curves for the window's quantity (or another)."""
        if quantity is None:
            qn = self._int_quantity()
            q = f"{qn} ({self._int_units()})"
        else:
            qn = q = quantity
        return {'ylabel': q, 'ylabel_offset': q + " (offset)", 'title_markers': "Marker spectra: " + qn,
                'stack_where': ", blocks as on the stripe (highest at the top)"}

    def _int_curve_items(self, curves):
        """
        The plots.draw_marker_curves items of the current marker state for a
        curves array oriented like the heatmap (rows follow self.pos): per
        marker its row of `curves` (NaN when never acquired), colour, label,
        position; per group its members, the plain mean of its acquired
        members' rows (NaN when none), colour, label, centre.
        """
        markers = []
        for i, pos_val in enumerate(self._int_marker_pos):
            j = self._int_col_of(pos_val)
            curve = (curves[self._int_row_of(pos_val), :] if self._int_acquired(j)
                     else np.full(curves.shape[1], np.nan))
            markers.append({'curve': curve, 'color': self._int_marker_color(i), 'label': f'{pos_val:.1f} Å',
                            'key': float(pos_val)})
        groups = []
        for g, members in enumerate(self._int_groups()):
            rows = [markers[i]['curve'] for i in members if self._int_acquired(self._int_col_of(self._int_marker_pos[i]))]
            avg = self._int_rows_average(np.vstack(rows)) if rows else np.full(curves.shape[1], np.nan)
            groups.append({'members': list(members), 'average': avg,
                           'color': self.M_COLORS[g % len(self.M_COLORS)],
                           'label': self._int_group_label(g, members), 'center': self._int_group_center(members, g)})
        return markers, groups

    # ------------------------------------------------------------------ #
    #  Feenstra companion                                                  #
    # ------------------------------------------------------------------ #

    def _int_feenstra_refresh(self):
        w = getattr(self, '_int_feenstra', None)
        if w is not None and w.is_open():
            w.refresh()

    def _int_feenstra_versions(self, calibrate, normalize):
        """
        The four versions of the displayed line for the companion
        (feenstra_view): a dict key -> {'title', 'ylabel', 'top' or 'note'}
        with the companion's own calibrate / normalize flags.
        """
        hp, sp = self._int_hampel_params, self._int_sgolay_params
        raw, iv = self._int_raw, self._int_iv
        # The four versions depend on the settings below, never on the markers: computed once per
        # setting (a marker drag refreshes the companion without rerunning the pipeline).
        memo_key = (bool(calibrate), bool(normalize), repr(hp), repr(sp), self._int_iv_zero_sub,
                    self._int_iv_broadened, self._int_iv_broadening, self.channel, self.chunk_index,
                    id(raw), id(iv), len(self._int_pos), self._int_e_range)
        memo = getattr(self, '_int_versions_memo', None)
        if memo is not None and memo[0] == memo_key:
            return {k: dict(v) for k, v in memo[1].items()}, memo[2]
        versions, q = self._int_feenstra_versions_compute(calibrate, normalize, hp, sp, raw, iv)
        self._int_versions_memo = (memo_key, versions, q)
        return {k: dict(v) for k, v in versions.items()}, q

    def _int_feenstra_versions_compute(self, calibrate, normalize, hp, sp, raw, iv):
        """The four versions of _int_feenstra_versions, computed (every normalization over the 'E range' box's range)."""
        q = ("d2I/dV2 (IETS)" if self.channel == 'iets' else "dI/dV") + ("" if calibrate else ", uncalibrated")
        nn = "" if normalize else ", not normalized"
        zs, width = self._int_iv_zero_sub, self._int_iv_width()
        t = feenstra_titles(zs, self._int_iv_broadened, self._int_iv_broadening, energy_has_zero(self.energy))
        units = curve_units(self.channel, calibrate, normalize, self.units_lia, self.units_iets, self.units_iv)
        versions = {
            'didv': {'title': f"{q} (division off){nn} line {self.chunk_index}", 'ylabel': f"{q} ({units})"},
            'iv': {'title': t['iv'], 'ylabel': "I (unit area)" if normalize else "I (A)"},
            'ivv': {'title': t['ivv'], 'ylabel': "I/V (unit area)" if normalize else "I/V (A/V)"},
            'ratio': {'title': t['ratio'] + f" (division on){nn}", 'ylabel': "(dI/dV)/(I/V) (arb. units)"},
        }
        versions['didv']['top'] = self._int_orient(self._pipeline(raw, iv, hp, sp, calibrate, False, 0.0, normalize,
                                                                  norm_range=self._int_e_range))
        if iv is None:
            for key in ('iv', 'ivv', 'ratio'):
                versions[key]['note'] = "no Current channel in this file"
        else:
            acquired = self._int_kept_mask() if np.asarray(raw).shape[1] == len(self._int_pos) \
                else np.any(np.asarray(raw) != self._unset, axis=0)
            iv_a = iv_corrected(self.energy, iv, zs, hp, sp)
            if normalize:
                iv_a = normalize_abs(self.energy, iv_a, energy_range=self._int_e_range)[0]
            iv_a[:, ~acquired] = np.nan
            versions['iv']['top'] = self._int_orient(iv_a)
            ivv = iv_broadened(self.energy, iv, width, zs, hp, sp)
            if normalize:
                ivv = normalize_abs(self.energy, ivv, energy_range=self._int_e_range)[0]
            ivv[:, ~acquired] = np.nan
            versions['ivv']['top'] = self._int_orient(ivv)
            if self.channel == 'iets':
                versions['ratio']['note'] = "the division applies to dI/dV only"
            else:
                versions['ratio']['top'] = self._int_orient(self._pipeline(raw, iv, hp, sp, calibrate, True, width,
                                                                           normalize, iv_zero_sub=zs,
                                                                           norm_range=self._int_e_range))
        return versions, q

    def feenstra_view(self, calibrate=True, normalize=True):
        """
        The companion's view (feenstra.FeenstraWindow): the displayed line in
        four versions, the dI/dV of the window's pipeline with the division
        off, the current in amps as the division works on it (its value at
        0 V subtracted with '0Vsub'), its I/V in A/V as the division divides
        by it (broadened with 'broadened'), and the ratio with the division
        on, each oriented like the heatmap with the marker curves in the
        current marker state, the current stripe, and the marker lines; a
        line without the Current channel gives notes for the current, I/V,
        and ratio panels, the IETS channel a note for the ratio panel.
        `calibrate` and `normalize` are the companion's own flags: the dI/dV
        and the ratio through the pipeline with them (calibrate off: the
        lock-in output as recorded, and the ratio of that output by the filtered I/V);
        with `normalize` on the current and the I/V are unit area per
        location (normalization.normalize_abs); off, no panel is normalized.
        Every normalization is over the 'E range' box's range, handed over
        as 'e_range' ((lo, hi) or None): the companion shows that range and
        scales its colours over it.
        """
        versions, q = self._int_feenstra_versions(calibrate, normalize)
        for key, v in versions.items():
            if 'top' in v:
                markers, groups = self._int_curve_items(v['top'])
                v.update({'markers': markers, 'groups': groups})
        e_left = min(self.energy[0], self.energy[-1])
        e_right = max(self.energy[0], self.energy[-1])
        sh = self._int_shear()
        return {'kind': 'line',
                'title': self._int_source_label(),
                'energy': self._int_energy,
                'e_range': self._int_e_range,
                'versions': versions,
                'texts': self._int_curve_texts(q),
                'avg_on': self._int_avg_on,
                'extent': [e_left, e_right, self._int_pos[0], self._int_pos[-1]],
                'pos': self._int_pos,
                'kept': self._int_kept_mask(),                 # the rows drawn (kept spectra), at pos[kept]
                'shear': sh,                                   # (n_kept, n_e + 1) with the drift toggle on, else None
                'stripe': self._int_stripe_profile,
                'stripe_pos': np.array(self._int_topo_x, dtype=np.float64),
                'marker_lines': [(float(p), self._int_marker_color(i)) for i, p in enumerate(self._int_marker_pos)],
                'centers': [self._int_group_center(m, g) for g, m in enumerate(self._int_groups())],
                'cmap': self.cmap}

    # ------------------------------------------------------------------ #
    #  Event handlers                                                     #
    # ------------------------------------------------------------------ #

    def _int_on_pick(self, event):
        """Identify which marker (or, on the heatmap, which group's average line) was clicked."""
        artist = event.artist
        label = getattr(artist, 'get_label', lambda: '')()
        if not isinstance(label, str):
            return
        if label.startswith('_emarker_'):                     # (a leading '_' keeps it out of the legends)
            k = int(label.split('_')[2])
            if k < len(self._int_e_markers):
                self._int_active_obj = ('emarker', k)
                self._int_e_selected = k
        elif label.startswith('marker_'):
            idx = int(label.split('_')[1])
            if idx < len(self._int_marker_pos):
                self._int_active_obj = ('marker', idx)
        elif label.startswith('_avg_'):
            g = int(label.split('_')[2])
            if g < len(self._int_group_sizes):
                self._int_active_obj = ('avg', g)

    def _int_on_motion(self, event):
        """
        Drag the active object vertically: a single marker, or a group's
        average line, which moves every member of the group by the same
        amount (rigid) and stops when its first or last marker reaches the
        start or the end of the line.
        """
        if self._int_active_obj is None or event.ydata is None:
            return
        if self._int_active_obj[0] == 'emarker':
            # An E-marker follows the cursor along the energy axis (heatmap or curve panel), snapped to
            # the recorded energies.
            if event.inaxes in (self._int_ax_didv, self._int_ax_curve) and event.xdata is not None:
                k = self._int_active_obj[1]
                e = self._int_e_snap(event.xdata)
                if e != self._int_e_markers[k]:
                    self._int_e_markers[k] = e
                    self._int_redraw_e_markers()
            return
        if event.inaxes not in [self._int_ax_didv, self._int_ax_topo]:
            return
        kind, idx = self._int_active_obj
        if kind == 'marker':
            pos_min, pos_max = self._int_pos_bounds()
            self._int_marker_pos[idx] = float(np.clip(event.ydata, pos_min, pos_max))
        else:
            lo, hi = self._int_pos_bounds()
            members = self._int_groups()[idx]
            if not members:
                return
            pos = [self._int_marker_pos[i] for i in members]
            delta = float(event.ydata) - self._int_group_center(members, idx)
            delta = float(np.clip(delta, lo - min(pos), hi - max(pos)))
            for i in members:
                self._int_marker_pos[i] = float(self._int_marker_pos[i] + delta)
        self._redraw_curves()

    def _int_on_release(self, event):
        """End drag."""
        self._int_active_obj = None

    def _int_on_group_slider(self, g, m):
        """
        A 'Markers' slider: the number of markers of the group it shows (the
        stack is ordered as the groups' average locations sit on the stripe,
        highest at the top).  That group keeps its centre (the mean of its
        current markers) and gets the new number of markers one spectrum
        step apart around it; every other group's markers stay where they are.
        """
        if m == int(self._int_group_sizes[g]):
            return
        groups = self._int_groups()
        center = self._int_group_center(groups[g], g)
        new_pos = []
        for gi, members in enumerate(groups):
            if gi == g:
                new_pos.extend(self._int_place_group(center, m))
            else:
                new_pos.extend(self._int_marker_pos[i] for i in members)
        self._int_group_sizes[g] = m
        self._int_marker_pos = new_pos
        self._int_sync_centres()
        if m == 0:
            self._int_group_ctr[g] = center               # an emptied group keeps its place
        self._redraw_curves()

    def _int_on_avgs_change(self, n):
        """
        'avgs' slider: number of groups.  Growing keeps the existing groups and
        adds each new group at the free slot of the even layout (the slot
        farthest from every existing group centre) with the size of the last
        group; shrinking drops the last groups (their markers and sizes).
        """
        groups = self._int_groups()
        if n == len(groups):
            return
        self._int_sync_centres()
        if n < len(groups):
            keep = sum(max(0, int(s)) for s in self._int_group_sizes[:n])
            self._int_marker_pos = self._int_marker_pos[:keep]
            self._int_group_sizes = self._int_group_sizes[:n]
            self._int_group_ctr = self._int_group_ctr[:n]
        else:
            # A new group takes the last group's size (the default size when there is no group, or
            # when the last one is empty).
            m = int(self._int_group_sizes[-1]) if self._int_group_sizes else 0
            m = m if m > 0 else self.DEFAULT_GROUP_SIZE
            centers = [self._int_group_center(members, g) for g, members in enumerate(groups)]
            dist = lambda a, b: abs(float(a) - float(b))
            for best in widgets.MarkerGroupControls.grow_groups(n, centers, self._int_layout_centers(n), dist):
                self._int_marker_pos.extend(self._int_place_group(best, m))
                self._int_group_sizes.append(m)
                self._int_group_ctr.append(float(best))
        self._redraw_curves()

    def _int_refresh_group_sliders(self):
        """
        One 'Markers' slider per group with its 'add new' / 'add existing'
        buttons, stacked bottom to top in the order of the groups' average
        locations (widgets.MarkerGroupControls.refresh); the panels raised
        when the stack would otherwise run into them (_int_place_panels).
        """
        self._int_groups_ctl.refresh(self._int_group_order(), self._int_group_sizes)
        self._int_place_panels()

    # ------------------------------------------------------------------ #
    #  Add to averages                                                     #
    # ------------------------------------------------------------------ #

    def _int_averages(self):
        """
        The Averages registry the 'add' buttons use: the one handed over by
        the caller (the Browser's), else one made here at the first add on the
        file's folder with the filter boxes' values, the window's
        fit_settings, Feenstra on with the window's broadening, and the
        window's channel.
        """
        if self._int_avg_registry is None:
            self._int_avg_registry = Averages(self._int_directory(),
                                              self._int_box_hampel, self._int_box_sgolay,
                                              fit_settings=self._int_fit_settings,
                                              divide_iv=True, iv_broadening=self._int_iv_broadening,
                                              channel=self.channel, iv_zero_sub=self._int_iv_zero_sub,
                                              iv_broadened=self._int_iv_broadened)
        return self._int_avg_registry

    def _int_directory(self):
        """The experiment directory (the file's folder)."""
        return os.path.dirname(os.path.abspath(self.ifile))

    def _int_sm4(self):
        """The file's Sm4File (handed over by the caller, else loaded here at the first use), seen through the window's sweep view."""
        if self._int_sm4_file is None:
            self._int_sm4_file = sm4file.Sm4File(self.ifile)
        if self.sweep_mode is not None and getattr(self._int_sm4_file, 'sweep_mode', None) != self.sweep_mode:
            self._int_sm4_file = self._int_sm4_file.with_sweeps(self.sweep_mode)
        return self._int_sm4_file

    def _int_on_sweeps(self, label):
        """
        The 'sweeps' radio (a file with several sweeps per pixel): the line re-sliced from the chosen
        sweep view ('forward sweeps', 'reverse sweeps', 'all sweeps', 'sweep n'), the heatmap, the marker curves, the
        energy slice, the drift shear (each sweep at its own time), and the companion recomputed.
        """
        if self.spectra_all is None or label == self.sweep_mode:
            return
        old = self.sweep_mode
        try:
            self.sweep_mode = label
            self.spectra = self.spectra_all.sweep_view(label)
            self.extract_data()
            self._int_load_source()
            curves = self._int_pipeline(self._int_hampel_params, self._int_sgolay_params)
        except Exception as exc:
            print(f"sweeps {label} not applied: {exc}")
            self.sweep_mode = old
            self.spectra = self.spectra_all.sweep_view(old)
            self.extract_data()
            self._int_load_source()
            self._int_rad_sweeps.set(old)
            self._int_fig.canvas.draw_idle()
            return
        self._int_curves = curves
        self._int_shear_cache = None
        self._int_draw_heatmap()
        print(f"sweeps {label}: the {self._int_tag()} shows the {label} sweep(s) of every pixel.")
        self._redraw_curves()

    def _int_line_columns(self):
        """The global column indices of the displayed line's spectra (scan_mode 'single' only)."""
        if self.scan_mode != 'single':
            raise ValueError("this needs scan_mode 'single' (one spectrum per position).")
        n_cols = self._int_raw.shape[1]
        return [self.chunk_index * n_cols + j for j in range(n_cols)]

    def _int_group_selection(self, g):
        """
        The Selection of group g: its markers' columns as global indices of the
        file's spectra (chunk_index x line length + column), markers on
        never-acquired spectra left out with a console line; None when no
        marker remains.  Only scan_mode 'single' maps a marker to one column of
        the file.
        """
        if self.scan_mode != 'single':
            raise ValueError("add to averages needs scan_mode 'single' (one spectrum per position).")
        members = self._int_groups()[g]
        n_cols = self._int_raw.shape[1]
        cols, positions, skipped = [], [], []
        for i in members:
            pos_val = self._int_marker_pos[i]
            j = self._int_col_of(pos_val)
            if not self._int_acquired(j):
                skipped.append(pos_val)
                continue
            cols.append(self.chunk_index * n_cols + j)
            positions.append(float(pos_val))
        if skipped:
            print("add: marker(s) at " + ", ".join(f"{p:.1f}" for p in skipped)
                  + " A sit on never-acquired spectra and are left out.")
        if not cols:
            return None
        return Selection(self._int_sm4(), cols, {'line': int(self.chunk_index), 'positions': positions})

    def _int_group_selection_checked(self, g):
        """The Selection of a group, or None (the reason printed)."""
        try:
            selection = self._int_group_selection(g)
        except Exception as exc:
            print(f"add: {exc}")
            return None
        if selection is None:
            print(f"add: avg {g + 1} has no marker on an acquired spectrum; nothing added.")
        return selection

    def _int_on_add_new(self, g):
        """'add new' of a slider row: the group shown by that slider starts a new average (each marker one sweep)."""
        self._int_picker().close()
        selection = self._int_group_selection_checked(g)
        if selection is None:
            return
        try:
            w = self._int_averages().add('', selection, channel=self.channel)
        except Exception as exc:
            print(f"add new: {exc}")
            return
        self._int_avg_windows.append(w)            # the window stays referenced so its widgets stay live
        self._int_fig.canvas.draw_idle()

    def _int_on_add_existing(self, g, anchor_ax):
        """'add existing' of a slider row: the list of the open averages; a row picked adds the group to it."""
        picker = self._int_picker()
        if picker.is_open():
            picker.close()
            return
        selection = self._int_group_selection_checked(g)
        if selection is None:
            return

        def add_to(index):
            w = self._int_averages().open().get(index)
            if w is None:
                return
            if w.has(selection):
                print(f"{w.name} already holds {selection.entry_label} ({selection.name}).")
                return
            try:
                w.add(selection)
            except Exception as exc:
                print(f"add to {w.name}: {exc}")
                return
            print(f"{w.name}: added {selection.entry_label} ({selection.name}; {len(w.entries)} entries).")
            self._int_avg_windows.append(w)
            self._int_fig.canvas.draw_idle()

        picker.open(anchor_ax, selection, add_to)

    def _int_picker(self):
        """The pop-up list of the open averages (widgets.AveragePicker), made at the first use."""
        if self._int_avg_picker is None:
            self._int_avg_picker = widgets.AveragePicker(self._int_fig, self._int_averages())
        return self._int_avg_picker

    # ------------------------------------------------------------------ #
    #  This line into a line average (lineavg.LineAverages)               #
    # ------------------------------------------------------------------ #

    def _int_line_averages(self):
        """The LineAverages registry: the caller's (the Browser's), else one made here on the file's folder."""
        if self._int_line_registry is None:
            from sm4_Browser import lineavg
            self._int_line_registry = lineavg.LineAverages(self._int_directory(), self._int_box_hampel,
                                                           self._int_box_sgolay,
                                                           fit_settings=self._int_fit_settings,
                                                           divide_iv=True, iv_broadening=self._int_iv_broadening,
                                                           channel=self.channel, iv_zero_sub=self._int_iv_zero_sub,
                                                           iv_broadened=self._int_iv_broadened, cmap=self.cmap)
        return self._int_line_registry

    def line_entry(self):
        """
        This line as a member of a line average: the file's spectra of the
        line (a Selection of its columns, which carries the raw LIA and I(V)
        columns, the energy axis, V_bias, I_setpoint, and the current's
        scale), the line index, the positions (Å), the stripe rows as the
        window has them (the raw profiles and their (row, pass) list), and
        the window's marker positions and group centres.  scan_mode 'single'
        only.
        """
        cols = self._int_line_columns()
        selection = Selection(self._int_sm4(), cols, {'line': int(self.chunk_index),
                                                      'positions': [float(p) for p in self._int_pos]})
        return {'selection': selection, 'file': selection.file, 'path': selection.path, 'name': selection.name,
                'line': int(self.chunk_index), 'pos': np.array(self._int_pos, dtype=np.float64),
                'topo_x': np.array(self._int_topo_x, dtype=np.float64),
                'topo_profiles': [np.array(p, dtype=np.float64) for p in self._int_topo_profiles],
                'topo_lines': [(int(r), m) for r, m in self._int_topo_lines],
                'markers': [float(p) for p in self._int_marker_pos],
                'centers': [self._int_group_center(m, g) for g, m in enumerate(self._int_groups())],
                'scalefactor': float(self.scalefactor)}

    def _int_line_entry_checked(self):
        try:
            return self.line_entry()
        except Exception as exc:
            print(f"line → avg: {exc}")
            return None

    def _int_on_line_new(self):
        """'line → new avg': this line starts a new line average (lineavg.LineAverages.add)."""
        self._int_line_picker().close()
        entry = self._int_line_entry_checked()
        if entry is None:
            return
        try:
            w = self._int_line_averages().add('', entry, channel=self.channel)
        except Exception as exc:
            print(f"line → new avg: {exc}")
            return
        self._int_avg_windows.append(w)
        self._int_fig.canvas.draw_idle()

    def _int_on_line_existing(self):
        """'line → existing avg': the list of the open line averages; a row picked adds this line to it."""
        picker = self._int_line_picker()
        if picker.is_open():
            picker.close()
            return
        entry = self._int_line_entry_checked()
        if entry is None:
            return

        def add_to(index):
            w = self._int_line_averages().open().get(index)
            if w is None:
                return
            if w.has(entry):
                print(f"{w.name} already holds {entry['name']}: line {entry['line']}.")
                return
            try:
                w.add(entry)
            except Exception as exc:
                print(f"line → {w.name}: {exc}")
                return
            self._int_avg_windows.append(w)
            self._int_fig.canvas.draw_idle()

        picker.open(self._int_btn_line_exist.ax, entry, add_to)

    def _int_line_picker(self):
        """The pop-up list of the open line averages (widgets.AveragePicker), made at the first use."""
        if self._int_line_avg_picker is None:
            self._int_line_avg_picker = widgets.AveragePicker(self._int_fig, self._int_line_averages())
        return self._int_line_avg_picker

    # ------------------------------------------------------------------ #
    #  The marker record ('save')                                          #
    # ------------------------------------------------------------------ #

    def marker_record(self):
        """
        The marker record of this file and line (records.LINE_RECORD_FORMAT):
        the file (path and name), the line index, the channel, the sweep
        view (None for one sweep per pixel), the filter boxes and their on /
        off states, the calibrate / normalize / Feenstra states, the
        Feenstra settings, the gamma, the 'E range' and 'x range' boxes
        (None = full), the topo rows and the active processor, the group
        sizes, their centres and the marker positions (Å), the E-markers
        (V), the Average / stack / substack states, the scale factor the
        window was made with ('scalefactor': the positions above are in
        Å after it; the browser's loader opens the window at its
        timeline's scale for the file and notes a difference on the
        console, browser._record_scale), and the time of writing.
        """
        hp, sp = self._int_filters.values()
        h_on, s_on = self._int_filters.states()
        return {'format': records.LINE_RECORD_FORMAT,
                'path': os.path.abspath(self.ifile),
                'name': os.path.basename(self.ifile),
                'line': int(self.chunk_index),
                'scalefactor': float(self.scalefactor),
                'channel': self.channel,
                'sweeps': self.sweep_mode,
                'hampel': hp, 'sgolay': sp,
                'hampel_on': bool(h_on), 'sgolay_on': bool(s_on),
                'calibrate': bool(self._int_apply_calibration),
                'normalize': bool(self._int_apply_normalize),
                'divide_iv': bool(self._int_divide_iv),
                'iv_broadening_V': float(self._int_iv_broadening),
                'iv_broadened': bool(self._int_iv_broadened),
                'iv_zero_sub': bool(self._int_iv_zero_sub),
                'gamma': float(self._int_gamma_box.value),
                'energy_range': None if self._int_e_range is None else [float(v) for v in self._int_e_range],
                'pos_range': None if self._int_x_range is None else [float(v) for v in self._int_x_range],
                'topo_lines': [[int(r), m] for r, m in self._int_topo_lines],
                'topo_active': [bool(a) for a in self._int_topo_active],
                'group_sizes': [int(s) for s in self._int_group_sizes],
                'group_centers': [float(c) for c in (self._int_sync_centres() or self._int_group_ctr)],
                'markers': [float(p) for p in self._int_marker_pos],
                'energy_markers': [float(e) for e in self._int_e_markers],
                'average': bool(self._int_avg_on),
                'stack': bool(self._int_stack_on),
                'substack': bool(self._int_substack_on),
                'saved': datetime.now().isoformat(timespec='seconds')}

    def _int_on_save(self):
        """'save': the marker record averages/<file stem>_line<k>_markers.json (one per file and line, overwritten)."""
        try:
            path = records.write_json(records.line_record_path(self._int_directory(), self.ifile, self.chunk_index),
                                      self.marker_record())
        except Exception as exc:
            print(f"save: {exc}")
            return
        print(f"{self._int_tag()}: marker record written: {path}")

    def restore_markers(self, rec):
        """
        Put a marker record's state into the open window (the browser's
        loader, after plot_line_with_topo_interactive on the record's file
        and line): the filter boxes and states, the calibrate / normalize /
        Feenstra states and settings, the gamma, the 'E range' and 'x range'
        boxes, the topo rows and processor,
        the group sizes and marker positions, Average / stack / substack.
        Every part that cannot be applied is reported and skipped.
        """
        hp, sp = rec.get('hampel'), rec.get('sgolay')
        try:
            self._int_filters.set_params(hp if rec.get('hampel_on') else None, sp if rec.get('sgolay_on') else None)
            if hp is not None and not rec.get('hampel_on'):
                self._int_filters._hp.update(hp)
            if sp is not None and not rec.get('sgolay_on'):
                self._int_filters._sp.update(sp)
            self._int_box_hampel = dict(self._int_filters.values()[0])
            self._int_box_sgolay = dict(self._int_filters.values()[1])
            self._int_hampel_params, self._int_sgolay_params = self._int_filters.params()
        except Exception as exc:
            print(f"restore markers: filter settings not applied ({exc}).")
        self._int_iv_zero_sub = bool(rec.get('iv_zero_sub', self._int_iv_zero_sub))
        self._int_iv_broadened = bool(rec.get('iv_broadened', self._int_iv_broadened))
        self._int_iv_broadening = float(rec.get('iv_broadening_V', self._int_iv_broadening))
        self._int_divide_iv = bool(rec.get('divide_iv', self._int_divide_iv))
        self._int_chk_feenstra.set(self._int_divide_iv)
        self._int_apply_calibration = bool(rec.get('calibrate', self._int_apply_calibration))
        self._int_apply_normalize = bool(rec.get('normalize', self._int_apply_normalize))
        self._int_pipeline_box.set_state(self._int_apply_calibration, self._int_apply_normalize)
        try:
            self._int_gamma_box.set(float(rec.get('gamma', 1.0)))
        except Exception:
            pass
        if 'energy_range' in rec:
            r = rec.get('energy_range')
            rng = None if r is None else (float(min(r)), float(max(r)))
            e = self._int_e_axis()
            if rng is not None and int(np.count_nonzero((e >= rng[0]) & (e <= rng[1]))) < 2:
                print("restore markers: the record's energy range holds fewer than two recorded energies; "
                      "the window keeps its own.")
            else:
                self._int_e_range = rng
                self._int_e_range_box.set(rng)
        if 'pos_range' in rec:
            r = rec.get('pos_range')
            self._int_x_range = None if r is None else (float(min(r)), float(max(r)))
            self._int_x_range_box.set(self._int_x_range)
        rows = rec.get('topo_lines')
        if rows:
            try:
                self._int_set_topo_lines([(int(r), m) for r, m in rows])
            except Exception as exc:
                print(f"restore markers: topo rows not applied ({exc}).")
        active = rec.get('topo_active')
        if active is not None and len(active) == len(self._int_topo_active):
            self._int_topo_active = [bool(a) for a in active]
            self._int_topo_box.set_active(self._int_topo_active)
        sizes = [max(0, int(s)) for s in rec.get('group_sizes', [])]
        markers = [float(p) for p in rec.get('markers', [])]
        if 'group_sizes' in rec and sum(sizes) == len(markers) and len(sizes) <= self.MAX_GROUPS:
            lo, hi = self._int_pos_bounds()
            self._int_group_sizes = sizes
            self._int_marker_pos = [float(np.clip(p, lo, hi)) for p in markers]
            ctr = rec.get('group_centers')
            self._int_group_ctr = ([float(c) for c in ctr] if ctr is not None and len(ctr) == len(sizes)
                                   else [None] * len(sizes))
            self._int_sync_centres()
        elif sizes or markers:
            print("restore markers: the group sizes and the marker positions do not match; markers kept.")
        if 'energy_markers' in rec:
            em = [self._int_e_snap(e) for e in rec.get('energy_markers') or []][:self.E_MAX]
            self._int_e_markers = em
            self._int_e_selected = 0
            self._int_sld_e.set(len(em))
        self._int_stacks.chk_average.set(bool(rec.get('average', False)))
        self._int_stacks.chk_stack.set(bool(rec.get('stack', False)))
        self._int_stacks.chk_substack.set(bool(rec.get('substack', False)))
        if not self._int_apply_feenstra("marker record"):
            print("restore markers: the record's pipeline state could not be applied; the window keeps its own.")
        self._int_refresh_topo_rows()
        self._redraw_topo_stripe()

    # ------------------------------------------------------------------ #
    #  Toggles                                                             #
    # ------------------------------------------------------------------ #

    def _int_on_topo_toggle(self, active):
        """The topo processor box (widgets.TopoProcessorBox): the stripe redrawn with the new flags."""
        self._int_topo_active = list(active)
        self._redraw_topo_stripe()

    def _int_on_feenstra_toggle(self, flag):
        """
        The window's own 'Feenstra' toggle: on, the heatmap, the marker curves,
        and the group averages are (dI/dV)/(I/V) (the division by the
        broadened I/V after the calibration); off, the plain dI/dV of the same
        pipeline.  Send to fit sends what is shown.  A file without the
        Current channel cannot be divided: the toggle is put back and the
        reason printed.
        """
        flag = bool(flag)
        if flag == self._int_divide_iv:
            return
        old = self._int_divide_iv
        self._int_divide_iv = flag
        if not self._int_apply_feenstra(f"Feenstra {'on' if flag else 'off'}"):
            self._int_divide_iv = old
            self._int_chk_feenstra.set(old)
            self._int_fig.canvas.draw_idle()

    # ------------------------------------------------------------------ #
    #  Send to fit                                                      #
    # ------------------------------------------------------------------ #

    def _int_on_send(self):
        try:
            self._int_send_to_fit()
        except Exception as exc:
            print(f"Send to fit: {exc}")

    def _int_send_to_fit(self):
        """
        Send the marker spectra (Average off: one fit per marker) or
        the group averages (Average on: one fit per group, labeled
        with that group's marker positions) to the single-point
        fit.  The sent curves are always rebuilt with calibration and
        normalization on, using the display's filter settings (Hampel, SG on
        the LIA) and the 'E range' box's normalization range (the full
        curves, their areas over that range; the fit refits over the same
        range); a console line says so when either is off in the window.
        A group average is the plain mean of its members.  The unsmoothed
        counterpart of each sent curve goes along for the noise estimate.  With divide_by_iv
        on, the curves are (dI/dV)/(I/V) (division after the calibration), the
        send carries quantity 'ratio' and the broadening (the fit
        then has no current anchor), and the label ends in ' over I/V' so the
        ratio's fit histories are separate from the dI/dV ones.
        """
        settings = dict(self._int_fit_settings)
        if self.channel != 'didv':
            raise ValueError("the fit models dI/dV; the window shows IETS.")
        if self._int_iv is None:
            raise ValueError("no 'Current' spectroscopy page in this file.")
        switched = []
        if not self._int_apply_calibration:
            switched.append('calibration')
        if not self._int_apply_normalize:
            switched.append('normalization')
        if switched:
            print("Send to fit: " + " and ".join(switched) + " switched on for the sent curves.")

        # Raw LIA and I(V) columns for the displayed line (same branches as the display).
        raw, iv = self._int_raw, self._int_iv
        if iv is None:
            raise ValueError("no I(V) current channel in this file; calibration is required.")

        hp, sp = self._int_hampel_params, self._int_sgolay_params

        # As displayed: calibrate against the raw I(V), (divide by the filtered I/V), normalize.
        div = self._int_divide_iv
        quantity = 'ratio' if div else 'didv'
        width, zs = self._int_iv_width(), self._int_iv_zero_sub
        cf, norm_areas = self._pipeline(raw, iv, hp, sp, True, div, width, True, return_areas=True, iv_zero_sub=zs,
                                        norm_range=self._int_e_range)
        areas_f = np.array(norm_areas)                       # per column, amps (ratio: arb. units x V)
        # The same with every filter off (the 'unfiltered' curve of the record).
        cu = self._pipeline(raw.copy(), iv, None, None, True, div, width, True, iv_zero_sub=zs,
                            norm_range=self._int_e_range)
        suffix = " over I/V" if div else ""                       # separate fit histories for the ratio

        # Orient like the display: rows follow self.pos, energy ascending.
        cf_rows, cu_rows = self._int_orient(cf), self._int_orient(cu)
        V = np.sort(self.energy)

        # Measured I(V_bias) - I(V_ref) per column (amps, raw current), V_ref = data
        # point nearest 0 V.  Used by the fit as the anchor when the window
        # does not contain 0 V.
        e_order = np.argsort(self.energy)
        iv_sorted = np.asarray(iv, dtype=np.float64)[e_order, :]
        i_ref = int(np.argmin(np.abs(V)))
        i_bias = int(np.argmin(np.abs(V - self.V_bias)))
        delta_I_cols = iv_sorted[i_bias, :] - iv_sorted[i_ref, :]

        # Each send also carries the raw LIA and I(V) columns it was built from (one
        # raw set on this line's energy axis), so that a refit with other filter
        # settings in the fit window reruns the same pipeline.
        sends = []
        if self._int_avg_on:
            # One send per group: the plain mean of its markers' displayed curves
            # (_int_rows_average).
            for members in self._int_groups():
                cols, positions, skipped = [], [], []
                for i in members:
                    pos_val = self._int_marker_pos[i]
                    j = self._int_col_of(pos_val)                          # original column
                    if not self._int_acquired(j):                          # never acquired: not sent
                        skipped.append(pos_val)
                        continue
                    cols.append(j)
                    positions.append(pos_val)
                if skipped:
                    print("Send to fit: marker(s) at " + ", ".join(f"{p:.1f}" for p in skipped)
                          + " A sit on never-acquired spectra and are left out of the average.")
                if not cols:
                    continue
                rows = list(cols)                                       # row j = spectrum j
                data = self._int_rows_average(cf_rows[rows, :])
                unf = self._int_rows_average(cu_rows[rows, :])
                label = (f"{self.ifile} | average of {len(cols)} markers at "
                         + ", ".join(f"{p:.1f}" for p in positions) + " A" + suffix)
                sends.append((data, unf, areas_f[cols], delta_I_cols[cols], label,
                              raw[:, cols], iv[:, cols], 'average'))
        else:
            for pos_val in self._int_marker_pos:
                j = self._int_col_of(pos_val)
                if not self._int_acquired(j):                              # never acquired: not sent
                    print(f"Send to fit: the marker at {pos_val:.1f} A sits on a never-acquired "
                          "spectrum and is not sent.")
                    continue
                r = j                                                   # row j = spectrum j
                data = cf_rows[r, :]          # displayed row (the filters on its LIA)
                unf = cu_rows[r, :]           # same position, every filter off
                sends.append((data, unf, areas_f[[j]], delta_I_cols[[j]],
                              f"{self.ifile} | marker at {pos_val:.1f} A" + suffix,
                              raw[:, [j]], iv[:, [j]], 'column'))

        # Fit history cache per sent curve, as in the fit module's __main__: with
        # use_cache on and a readable <date_time>_<label>_fits.pkl present in this
        # file's folder, the figure opens from it; otherwise a fresh sweep runs and
        # writes the cache.
        use_cache = settings.pop('use_cache', True)
        for data, data_unfiltered, areas, delta_I, label, lia_raw, iv_raw, stage in sends:
            print("\n" + "=" * 70)
            if use_cache and fit.Fit.has_cache(label):
                an = fit.Fit.from_cache(fit.Fit.cache_path(label))
            else:
                an = fit.Fit.from_curve(V, data, data_unfiltered, self.V_bias, self.I_setpoint,
                                        areas, label,
                                        hampel_filter_params=hp, sgolay_filter_params=sp,
                                        delta_I_window_A=delta_I,
                                        lia_raw=lia_raw, iv_raw=iv_raw,
                                        units_lia=self.units_lia, units_iv=self.units_iv, vmod=self.vmod,
                                        energy_raw=self.energy,
                                        filter_stage=stage,
                                        quantity=quantity, iv_broadening=width, iv_zero_sub=zs,
                                        norm_range=self._int_e_range)
                an.select_model(**settings)
            an.plot()
            self._int_fits.append(an)

    # ------------------------------------------------------------------ #
    #  The figure                                                          #
    # ------------------------------------------------------------------ #

    def _int_build_axes(self):
        """The panels: the topo stripe, the heatmap with its colour bar, the curve panel, the topo colour bar."""
        fig = self._int_fig
        fig.subplots_adjust(bottom=self.PANEL_BOTTOM_MIN)
        gs_main = gridspec.GridSpec(2, 1, height_ratios=[1, 0.001], hspace=0.3)
        gs_top = gridspec.GridSpecFromSubplotSpec(
            1, 7, subplot_spec=gs_main[0],
            width_ratios=[0.175, 3, 0.15, 0.1, 0.9, 0.3, 1.5], wspace=0.05)
        self._int_ax_topo = fig.add_subplot(gs_top[0])
        self._int_ax_didv = fig.add_subplot(gs_top[1], sharey=self._int_ax_topo)
        self._int_cax_didv = fig.add_subplot(gs_top[2])
        ax_spacer = fig.add_subplot(gs_top[3])
        ax_spacer.axis('off')
        # The energy slice (E-markers): the quantity against position at each E-marker, on the heatmap's
        # position axis (shared y), as MATLAB's line script showed it beside its heatmaps.
        self._int_ax_slice = fig.add_subplot(gs_top[4], sharey=self._int_ax_topo)
        self._int_ax_slice.tick_params(axis='y', left=False, labelleft=False)
        ax_spacer2 = fig.add_subplot(gs_top[5])
        ax_spacer2.axis('off')
        self._int_ax_curve = fig.add_subplot(gs_top[6])
        # The heatmap's positions are the stripe's (shared y): no y ticks, values, or label of its own.
        self._int_ax_didv.tick_params(axis='y', left=False, labelleft=False)
        # The slice and the marker curve panel never run into the heatmap colour bar's ticks and labels.
        self._int_ax_slice.keep_clear_of(self._int_cax_didv)
        self._int_e_artists = []
        self._int_ax_topo_cbar = plots.draw_topo_cbar(fig, self.TOPO_CBAR)
        self._int_im = None

    def _int_build_controls(self):
        """Every control of the window (widgets.py), at the window's positions."""
        fig = self._int_fig
        # 'Markers' sliders: one per group with 'add new' / 'add existing', the stack right of
        # the filter box (row 0 at the bottom), shown as 'avgs' says.
        self._int_groups_ctl = widgets.MarkerGroupControls(fig, self.SLIDER_X, self.SLIDER_Y0, self.SLIDER_PITCH,
                                                           self.AVGS_XY, max_groups=self.MAX_GROUPS,
                                                           n_groups=len(self._int_group_sizes))
        self._int_groups_ctl.on_avgs(self._int_on_avgs_change)
        self._int_groups_ctl.on_markers(self._int_on_group_slider)
        self._int_groups_ctl.on_add_new(self._int_on_add_new)
        self._int_groups_ctl.on_add_existing(self._int_on_add_existing)
        # The window's own 'Feenstra' toggle, below the filter box, and the channel radio beside it.
        self._int_chk_feenstra = widgets.make(fig, 'Feenstra', 0.03, 0.01, value=self._int_divide_iv)
        self._int_chk_feenstra.on_change(self._int_on_feenstra_toggle)
        # The window's own 'auto drift' and 'manual drift' toggles (set from the caller's states at
        # opening, independent after) and the 'drift window' button, one row left of the button column.
        fw = fig.get_size_inches()[0]
        x_auto = 0.53
        x_manual = x_auto + widgets.SPECS['auto drift']['w'] / fw + 0.003
        x_win = x_manual + widgets.SPECS['manual drift']['w'] / fw + 0.003
        self._int_chk_auto = widgets.make(fig, 'auto drift', x_auto, 0.125, value=self._int_auto_on)
        self._int_chk_auto.on_change(lambda flag: self._int_on_drift_toggle('auto', flag))
        self._int_chk_manual = widgets.make(fig, 'manual drift', x_manual, 0.125, value=self._int_manual_on)
        self._int_chk_manual.on_change(lambda flag: self._int_on_drift_toggle('manual', flag))
        self._int_btn_drift_win = widgets.make(fig, 'drift window', x_win, 0.127)
        self._int_btn_drift_win.on_change(self._int_on_drift_window)
        self._int_rad_channel = widgets.make(fig, 'channel', 0.12, 0.005, value=STS.CHANNEL_NAMES[self.channel])
        # The E-markers (energy markers, vertical lines at an energy; the position markers are unchanged):
        # their own row above the panels: the count, 'clear E-markers', the readout.
        self._int_sld_e = widgets.make(fig, 'E-markers', 0.17, 0.925, value=len(self._int_e_markers))
        self._int_sld_e.on_change(lambda val: self._int_on_e_count(int(round(float(val)))))
        self._int_btn_e_clear = widgets.make(fig, 'clear E-markers', 0.30, 0.918)
        self._int_btn_e_clear.on_change(lambda: self._int_on_e_count(0))
        self._int_txt_e = fig.text(0.375, 0.932, '', fontsize=8, ha='left', va='center')
        # The 'x range' and 'E range' boxes (widgets.RangeBox, "lo, hi", blank = full), right of the
        # readout: the shown position range, and the shown energy range, which is also the
        # normalization range and the colour scale's (_int_on_x_range, _int_on_e_range).
        self._int_x_range_box = widgets.RangeBox(fig, 'x range', 0.72, 0.925, value=self._int_x_range)
        self._int_x_range_box.on_change(self._int_on_x_range)
        self._int_e_range_box = widgets.RangeBox(fig, 'E range', 0.84, 0.925, value=self._int_e_range)
        self._int_e_range_box.on_change(self._int_on_e_range)
        self._int_rad_sweeps = None
        if self.spectra_all is not None:
            from sm4_Browser import sweeps as sweeps_mod
            self._int_rad_sweeps = widgets.make(fig, 'sweeps', 0.005, 0.80,
                                                labels=list(sweeps_mod.sweep_modes(self.spectra_all.sweep_info)),
                                                value=self.sweep_mode)
            self._int_rad_sweeps.on_change(self._int_on_sweeps)
        self._int_rad_channel.on_change(self._int_on_channel)
        # The topo processor box: 'Slope Sub' and 'Δz sub', one active at most.
        self._int_topo_box = widgets.TopoProcessorBox(fig, 0.44, 0.01, topo.processor_labels(), self._int_topo_active)
        self._int_topo_box.on_change(self._int_on_topo_toggle)
        # Average, stack, substack.
        self._int_stacks = widgets.StackControls(fig, (0.69, 0.01), (0.69, 0.065), (0.53, 0.065), needs_average=True)
        self._int_stacks.on_change(self._redraw_curves)
        # 'Send to fit', 'line → new avg', 'line → existing avg', 'save' in one column.
        self._int_btn_send = widgets.make(fig, 'Send to fit', 0.76, 0.01)
        self._int_btn_send.on_change(self._int_on_send)
        self._int_btn_line_new = widgets.make(fig, 'line → new avg', 0.76, 0.065)
        self._int_btn_line_new.on_change(self._int_on_line_new)
        self._int_btn_line_exist = widgets.make(fig, 'line → existing avg', 0.76, 0.10)
        self._int_btn_line_exist.on_change(self._int_on_line_existing)
        self._int_btn_save = widgets.make(fig, 'save', 0.76, 0.135)
        self._int_btn_save.on_change(self._int_on_save)
        # Savitzky-Golay / Hampel controls (every filter off; boxes pre-filled from the call).
        self._int_filters = widgets.FilterControls(fig, [0.03, 0.065, 0.19, 0.13],
                                                   hampel=self._int_box_hampel, sgolay=self._int_box_sgolay,
                                                   hampel_on=False, sgolay_on=False)
        self._int_filters.on_change(self._int_on_filter_change)
        # The 'calibrate' / 'normalize' box (top right, as in every window).
        self._int_pipeline_box = widgets.PipelineBox(fig, calibrate=self._int_apply_calibration,
                                                     normalize=self._int_apply_normalize)
        self._int_pipeline_box.on_change(self._int_on_pipeline_change)
        # The heatmap's 'γ' box, just below the heatmap's bottom-right corner.
        self._int_gamma_box = widgets.GammaBox(fig, gamma=self._int_gamma0)
        self._int_gamma_box.on_change(self._int_on_gamma)
        self._int_gamma_box.attach(self._int_ax_didv)
        # Topo stripe source: one row of controls per averaged profile (a 'rows' box and an
        # 'f' button), TOPO_ROWS_IN_FIGURE of them in two columns filled from the bottom,
        # hidden until used (_int_refresh_topo_rows); '+' / '-' add and remove rows.
        self._int_topo_row_widgets = []
        w_box, _ = widgets.size_of(fig, 'rows')
        w_f, _ = widgets.size_of(fig, 'f')
        for k in range(self.TOPO_ROWS_IN_FIGURE):
            col, r = divmod(k, self.TOPO_ROWS_PER_COLUMN)
            x = self.TOPO_ROW_X[col]
            y = self.TOPO_ROW_Y0 + r * self.TOPO_ROW_PITCH
            box = widgets.make(fig, 'rows', x, y, value='')
            btn = widgets.make(fig, 'f', x + w_box + self.TOPO_ROW_GAP, y)
            prof = widgets.make(fig, 'profile', x + w_box + self.TOPO_ROW_GAP + w_f + self.TOPO_ROW_GAP, y)
            box.show(False)
            btn.show(False)
            prof.show(False)
            box.on_change(lambda text, k=k: self._int_on_topo_row_edit(k, text))
            btn.on_change(lambda k=k: self._int_on_topo_row_toggle(k))
            prof.on_change(lambda k=k: self._int_on_topo_row_profile(k))
            self._int_topo_row_widgets.append((box, btn, prof))
        self._int_topo_rows_label = fig.text(self.TOPO_ROW_X[0] - 0.004, self.TOPO_ROW_Y0, 'rows', fontsize=8,
                                             ha='right', va='bottom')
        self._int_topo_rows_note = fig.text(self.TOPO_ROW_X[0], self.TOPO_ROW_Y0 + 0.01, '',
                                            fontsize=8, ha='left', va='bottom', visible=False)
        self._int_btn_topo_add = widgets.make(fig, '+', 0.915, 0.055)
        self._int_btn_topo_add.on_change(self._int_on_topo_add)
        self._int_btn_topo_remove = widgets.make(fig, '-', 0.945, 0.055)
        self._int_btn_topo_remove.on_change(self._int_on_topo_remove)
        # Line (chunk) selection, below the topo rows: the box re-slices the file's spectra,
        # the arrows step through the lines (the box follows and stays editable).
        self._int_box_line = widgets.make(fig, 'line', 0.865, 0.055, value=str(self.chunk_index))
        self._int_box_line.on_change(self._int_on_line_index)
        self._int_line_steps = widgets.StepButtons(fig, '< line', 'line >', 0.865, 0.01)
        self._int_line_steps.on_change(self._int_on_line_step)

    def _int_connect(self):
        fig = self._int_fig
        fig.canvas.mpl_connect('pick_event', self._int_on_pick)
        fig.canvas.mpl_connect('motion_notify_event', self._int_on_motion)
        fig.canvas.mpl_connect('button_release_event', self._int_on_release)
        fig.canvas.mpl_connect('key_press_event', self._int_on_key)
        # Every manual change of the drift (a span applied or cleared in a drift window), made
        # in any drift window, re-shears this window's rows (Drift.subscribe); the subscription
        # ends with the window.
        if self._int_drift is not None:
            drift = self._int_drift
            drift.subscribe(self._int_apply_drift_toggle, alive=lambda: plt.fignum_exists(fig))
            fig.canvas.mpl_connect('close_event', lambda event: drift.unsubscribe(self._int_apply_drift_toggle))

    # ------------------------------------------------------------------ #
    #  Main interactive entry point                                       #
    # ------------------------------------------------------------------ #

    DESIGN = (16, 7.5)     # the design size (inches): the text zoom of the window is its size relative to this (qtplot.window_zoom)
    ROWS_DESIGN = (3.6, 2.0)       # the topo-rows pop-up: its width and its least height (inches)
    ROWS_ROW_IN, ROWS_PAD_IN = 0.42, 0.8   # ... its height grows by one row's height per topo row, plus the padding

    def plot_line_with_topo_interactive(self, hampel_filter_params=None,
                                        sgolay_filter_params=None,
                                        apply_calibration=False,
                                        apply_normalize=False,
                                        marker_positions=None,
                                        topo_lines=None,
                                        fit_settings=None,
                                        divide_by_iv=True,
                                        iv_broadening_V=0.0,
                                        averages=None,
                                        sm4=None,
                                        iv_zero_sub=True,
                                        iv_broadened=False,
                                        line_averages=None,
                                        gamma=1.0,
                                        drift=None,
                                        auto_on=False,
                                        manual_on=False,
                                        profile_plots=None):
        """
        The Line window.

        Placement: every row of the heatmap is one kept spectrum of the line
        (STS.acquired: no column all zeros or ending in zeros) at its
        recorded position along the line (line_axis), and every cell of the
        topo stripe is a page pixel at the position its page geometry
        records for it (stripe_rows, Sm4File.pixel_xy_m), a forward pixel
        drawn only when the spectrum recorded at its location (found by
        location, Sm4File.spectrum_pixels, never by index) is kept, so the
        stripe and the heatmap share one recorded position axis and nothing
        is paired by order; an unfinished line is drawn up to its last kept
        spectrum, and the colour scale comes from the kept rows.  The
        window's own 'auto drift' and 'manual drift' toggles (drift,
        auto_on, manual_on: the directory's Drift and the starting states,
        the Browser's): with a source on, every row keeps its anchor at its
        own topo pixel (the height was recorded before the sweep) and is
        sheared along the position axis by the drift during its own sweep
        (Drift.row_shear: pixel by pixel in time from the file's velocity,
        the before/after topo pair's, auto or manual; the stripe matching
        never enters here; nothing along the energy axis), the stripe is
        never moved, and each row's drift line is drawn on the stripe from
        its anchor through the along-line drift of its sweep, in the auto
        or the manual colour; with both sources off, the rows are plain.
        'drift window' opens the file's before and after topos for marker
        pairs (drift.DriftWindow), whose span refits the rows at once.
        Every topo row has a 'profile' button opening its line's raw
        forward and reverse passes in a profile window
        (_int_on_topo_row_profile).

        Static panels: topo stripe + dI/dV heatmap (its 'γ' box just below
        its bottom-right corner: the colour map's exponent, plots.heat_norm).
        Interactive: draggable position markers whose 1D spectra are shown in
        a third panel.  The 'avgs' slider (1 to 10) is the number of average
        groups; every group has its own 'Markers' slider (1 to 10 markers),
        the sliders stacked bottom to top in the order of the groups' average
        locations on the stripe (highest group at the top, as in the stack
        panel), each filled with its group's colour; the stack grows and
        shrinks with 'avgs', and the panels move up when the stack would
        otherwise run into them.  Group membership is by index (the first
        group's markers, then the next group's, ...), so a dragged marker
        stays in its group wherever it is moved.  When a group's count
        changes, its markers are placed one spectrum step apart around the
        group's centre, the other groups untouched (a new group starts with
        the last group's size at the free slot of an even layout along the
        line; fewer groups drops the last groups).  With Average off every
        marker has its own color and its own curve.  With Average on the
        markers are colored by group, each group's average location (mean of
        its marker positions) is drawn as a thick black line on the heatmap
        and the stripe, and the curve panel shows one average per group
        (black, thick): the plain mean of its members' displayed curves.
        Dragging: a press on the topo stripe moves one marker; a press on the
        heatmap with Average on grabs the group's average line and moves the
        whole group rigidly, stopping when its first or last marker reaches
        the start or the end of the line (with Average off the heatmap moves
        single markers too).  The 'stack' toggle (with Average on) turns the
        curve panel into one stack: per group a block with the group's
        average at the bottom and, with the 'substack' toggle on, its member
        spectra above it (filtered as displayed, in the group's color)
        ordered by position with the highest at the top ('substack' off, the
        default: the average alone); the blocks ordered as the groups'
        average locations sit on the stripe, highest at the top; offset
        STACK_OFFSET_FRAC of the largest curve range per step inside a block
        and one full range between blocks.  The topo processor box holds
        'Slope Sub' and 'Δz sub' (one active at most, Δz sub on at open; a
        click on the active one turns it off).  'add new' and 'add existing'
        beside every 'Markers' slider add that group to the averages as the
        browser's buttons add a single point: a Selection of the group's
        marker columns (each marker one sweep) goes to a new average, or to
        the open average picked from the pop-up list (widgets.AveragePicker).
        'line → new avg' / 'line → existing avg' send the whole line (its
        raw columns, positions, stripe rows, and markers) to a new line
        average or to the open one picked from the list (lineavg).  'save'
        writes the marker record averages/<file stem>_line<k>_markers.json
        (marker_record), reloaded from the saved averages window.  A
        Feenstra companion window (feenstra.FeenstraWindow) opens with the
        window and follows it: the line in four versions, the dI/dV with the
        division off, the current and its I/V as the division uses them
        (its '0Vsub', 'broadened', and width boxes are this window's Feenstra
        settings), and the ratio, with the marker curves under each; its own
        'calibrate' / 'normalize' box and 'γ' boxes.  The 'calibrate' /
        'normalize' box (top right) switches the two pipeline steps of this
        window.  The Savitzky-Golay and Hampel filters are in-window controls
        (widgets.FilterControls): every filter starts off, each switchable on
        its own, their boxes at hampel_filter_params / sgolay_filter_params
        (or the control defaults when None); they act on the current alone,
        inside the Feenstra division (Hampel, SG in that order, before the 0 V
        subtraction and the broadening; the channel's columns are never
        filtered, so with the division off they change nothing), and every
        toggle or edit recomputes the heatmap and the marker curves through
        the same pipeline (calibration against the raw I(V), division by the
        filtered I/V, normalization; a group average is the plain mean).
        A "Send to fit" button sends the marker spectra (or, when the
        Average toggle is on, one average per group, each labeled with its
        own marker positions) to the single-point fit (Fit) with
        the on-screen filter settings (dI/dV only).  The 'channel' radio
        switches the line between the dI/dV and the IETS channel of the file
        (STS; the IETS is calibrated with the double-derivative regression
        and normalized, never divided by I/V).  The topo stripe source is
        changeable in the window: one row of controls per averaged topo
        profile, a line-index box and an f/r button (forward / reverse), '+'
        adds a row (the current line, forward; at most 2 x the lines in the
        file), '-' removes the last (one stays); up to TOPO_ROWS_IN_FIGURE
        rows sit in the figure in two columns, more rows move to the separate
        'topo rows' window.  The stripe is the mean of the rows after each is
        processed and range matched to the row with the largest range
        (topo.stripe_profile); topo_lines only sets the initial rows.  The
        'line' box re-slices the file at another chunk index and '< line' /
        'line >' step through the lines (the topo rows follow when they were
        the line's own rows); an optional attribute
        _int_line_callback(chunk_index), set by a caller such as the
        Browser, is called after a change.

        Parameters
        ----------
        hampel_filter_params, sgolay_filter_params : dict or None
            Pre-fill of the in-window filter controls (the filters start off).
        apply_calibration, apply_normalize : bool
            Starting states of the 'calibrate' / 'normalize' box.
        marker_positions : list of float or None
            Initial marker positions in Å along the line axis (one group of
            that many markers).  If None, one group of two markers one
            spectrum step apart at the middle of the line.
        topo_lines : list of (row, 'forward' | 'reverse') or None
            The initial rows; None gives (chunk_index, forward), (chunk_index, reverse).
        fit_settings : dict or None
            Keyword arguments for Fit.select_model
            (known_features, n_jobs, max_width_ratio, noise_z), plus
            'use_cache' (default True): open a sent curve's figure from its
            fit history cache when that file exists (see the fit module)
            instead of running a fresh sweep.
        divide_by_iv : bool
            Starting state of the window's own 'Feenstra' toggle (default
            on).  On: after the calibration every curve is divided point by
            point by its own broadened I/V (feenstra.divide_by_iv); the
            heatmap, the marker curves, and the group averages show
            (dI/dV)/(I/V), and Send to fit sends the ratio (quantity
            'ratio').  Off: the plain dI/dV of the same pipeline.  The toggle
            is independent of the Browser's.
        iv_broadening_V : float
            Broadening width (V) of I/V for divide_by_iv (the companion's width box).
        iv_zero_sub, iv_broadened : bool
            Starting states of the companion's '0Vsub' (on by default) and
            'broadened' (off by default) boxes: the current's value at 0 V
            subtracted before I/V is formed (feenstra.iv_corrected), and I/V
            broadened by iv_broadening_V (off: the plain I/V).  Both act on
            the division of this window and of its fits
            (set_feenstra_settings).
        averages : averaging.Averages or None
            The registry of open averaging windows the 'add' buttons use (the
            Browser's); None makes one on the file's folder at the first add.
        sm4 : sm4file.Sm4File or None
            The file as the Browser has it loaded; None loads it at the first use.
        line_averages : lineavg.LineAverages or None
            The registry of open line-average windows the 'line → ...'
            buttons use (the Browser's); None makes one at the first use.
        gamma : float
            Starting value of the heatmap's 'γ' box.
        profile_plots : topo.ProfilePlots or None
            The Browser's profile plots: the 'profile' windows' 'add new' /
            'add existing' buttons add their forward/reverse pair there
            (None: the profile windows have no 'add' buttons).
        """
        self._int_fit_settings = fit_settings if fit_settings is not None else {}
        self._int_divide_iv = bool(divide_by_iv)
        self._int_iv_broadening = float(iv_broadening_V)
        self._int_iv_zero_sub = bool(iv_zero_sub)
        self._int_iv_broadened = bool(iv_broadened)
        self._int_fits = []       # sent fits stay referenced so their widgets stay live
        self._int_avg_registry = averages
        self._int_line_registry = line_averages
        self._int_sm4_file = sm4
        self._int_avg_windows = []            # averaging windows opened from here stay referenced
        self._int_avg_picker = None           # the pop-up list of the open averages (made at the first use)
        self._int_line_avg_picker = None      # the pop-up list of the open line averages
        self._int_feenstra = None             # the Feenstra companion (opened at the end)
        self._int_apply_calibration = bool(apply_calibration)
        self._int_apply_normalize = bool(apply_normalize)
        self._int_gamma0 = float(gamma)
        self._int_e_range = None              # the 'E range' box: (lo, hi) V shown and normalized over, None = all
        self._int_x_range = None              # the 'x range' box: (lo, hi) Å shown, None = the kept rows
        self._int_drift = drift
        self._int_auto_on = bool(auto_on and drift is not None)
        self._int_manual_on = bool(manual_on and drift is not None)
        self._int_drift_source = None
        self._int_shear_cache = None
        self._int_profile_windows = []        # profile windows opened from the topo rows stay referenced
        self._int_profile_plots = profile_plots   # the browser's profile plots (topo.ProfilePlots) for their 'add' buttons
        self._int_drift_windows = []          # drift windows opened from here stay referenced

        # --- Raw LIA and I(V) columns of the displayed line (pipeline input) ---
        self._int_load_source()

        # --- In-window filter settings: the boxes at the call arguments (or the control ---
        # --- defaults); every filter starts off; the filters act on the current (feenstra). ---
        self._int_box_hampel = (dict(hampel_filter_params) if hampel_filter_params is not None
                                else dict(widgets.FilterControls.HAMPEL_DEFAULT))
        self._int_box_sgolay = (dict(sgolay_filter_params) if sgolay_filter_params is not None
                                else dict(widgets.FilterControls.SGOLAY_DEFAULT))
        self._int_hampel_params = None
        self._int_sgolay_params = None

        # --- The curves (the pipeline of _int_pipeline with the starting settings) ---
        self._int_curves = self._int_pipeline(self._int_hampel_params, self._int_sgolay_params)
        e_left = min(self.energy[0], self.energy[-1])
        e_right = max(self.energy[0], self.energy[-1])
        self._int_energy = np.linspace(e_left, e_right, self._int_curves.shape[1])
        self._int_pos = self.pos                             # position axis in Å

        # --- Topography raw profiles (rows changeable in the window) ---
        if topo_lines is None:
            topo_lines = [(self.chunk_index, 'forward')]          # the line's forward row; the row controls add more
        self._int_rows_fig = None
        self._int_rows_widgets = []
        try:
            self._int_set_topo_lines(list(topo_lines))
        except Exception as e:
            print("Failed to extract topography:", e)
            return
        # Topo processing registry (one active at a time, Δz sub at open).
        self._int_topo_active = topo.processor_defaults()

        # --- Marker state: one group at the start, the given positions, or two markers one ---
        # --- spectrum step apart at the middle of the kept part of the line. ---
        pos_min, pos_max = self._int_pos_bounds()
        if marker_positions is not None:
            self._int_marker_pos = [float(np.clip(p, pos_min, pos_max)) for p in marker_positions]
            self._int_group_sizes = [len(self._int_marker_pos)]
        else:
            self._int_group_sizes = [2]
            self._int_marker_pos = self._int_place_group(self._int_layout_centers(1)[0], 2)
        self._int_group_ctr = []
        self._int_sync_centres()
        # E-markers: one at the middle of the energy range to start with (the 'E-markers' slider: 0 to 8).
        e_axis = np.sort(np.asarray(self.energy, dtype=np.float64))
        self._int_e_markers = [float(e_axis[len(e_axis) // 2])] if len(e_axis) else []
        self._int_e_selected = 0
        self._int_active_obj = None
        self._int_marker_artists = []
        self._int_panel_bottom = self.PANEL_BOTTOM_MIN

        # --- The figure, its panels and controls ---
        self._int_fig = plt.figure(figsize=self.DESIGN)
        self._int_build_axes()
        self._int_build_controls()
        self._int_draw_heatmap()
        # _redraw_topo_stripe chains to _redraw_curves at the end; the row controls follow the rows.
        self._redraw_topo_stripe()
        self._int_refresh_topo_rows()
        self._int_connect()

        self._int_open_companion()
        plt.show()

    def _int_open_companion(self):
        """The Feenstra companion: the four versions of this line, opened whatever the toggle's state."""
        self._int_feenstra = FeenstraWindow(self, self._int_source_label())

# Sample run code:
if __name__ == '__main__':
    # Replace 'example.sm4' with the correct path.
    #analyzer = Line('experiment_dir/file.sm4', chunk_index=0, cmap='jet', scalefactor=2.75, scan_mode='single', combine_scans=False)
    analyzer = Line('experiment_dir/file.sm4', chunk_index=1, cmap='jet', scalefactor=2.75, scan_mode='single', combine_scans=False)    
    # Static plot (original).
    '''
    analyzer.plot_line_with_topo(hampel_filter_params={'window': 3, 'n_sigma': 2.5},
                                  sgolay_filter_params={'window_length': 5, 'polyorder': 3},
                                  apply_calibration=True,
                                  apply_normalize=False,
                                  topo_lines=[(0, 'forward')])  # or e.g. [(0, 'forward'), (1, 'reverse')]
    '''
    
    # Settings for "Send to fit" (passed to Fit.select_model).
    fit_settings = {
        'known_features': [],                              # see fit.py for the format
        'n_jobs': -1,                                      # -1 = all CPU cores (pip install loky)
        'max_width_ratio': 3,
        'noise_z': 2.5,                                      # candidate threshold = noise_z x noise level
        'use_cache': True,                                   # open from <date_time>_<marker>_fits.pkl when present
    }

    # Interactive plot with draggable markers.
    analyzer.plot_line_with_topo_interactive(
        hampel_filter_params={'window': 4, 'n_sigma': 3},
        sgolay_filter_params={'window_length': 6, 'polyorder': 3},
        apply_calibration=True,
        apply_normalize=True,
        marker_positions=None,              # or e.g. [5.0, 15.0, 25.0]
        #topo_lines=[(0, 'forward'),(0, 'reverse')])
        topo_lines=[(0, 'forward'),(1, 'forward')],        # or e.g. [(0, 'forward'), (1, 'reverse')]
        fit_settings=fit_settings)
