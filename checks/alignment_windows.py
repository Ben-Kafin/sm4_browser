# -*- coding: utf-8 -*-
"""
alignment_windows: the within-file alignment of a grid line's forward topo
stripe with its dI/dV heatmap, side by side for several files, under the two
placements:

  window 1, "current alignment": the package as it is now (line.py,
      sm4file.py).  Every heatmap row is a kept spectrum (STS.acquired: no
      column all zeros or ending in zeros) at its recorded along-line position
      (line.line_axis, from RHK_SpecDrift_Xcoord / Ycoord); every stripe cell
      is a page pixel at the position its page geometry gives it
      (line.stripe_rows, Sm4File.pixel_xy_m), a forward pixel drawn only when
      the spectrum recorded at its location (Sm4File.spectrum_pixels, by
      location) is kept.  Rows at their own positions (plots.RowHeatmap), the
      stripe cells at theirs (plots.draw_topo_stripe).  No drift correction.

  window 2, "old alignment (zip)": the placement of the earlier line.py
      (the zip of 2026-10-11), copied here verbatim where it matters:
      line_positions (the fast page's RHK_Xoffset + i * RHK_Xscale, reversed
      and shifted to start at zero), topography (spym's page array as it
      comes, its row `line`, the reverse page mirrored along the fast axis),
      the profile reversed when V_bias > 0, the stripe drawn as cells spread
      evenly from pos[0] to pos[-1] (the old plots.draw_topo_stripe), the
      heatmap as imshow with extent [e_left, e_right, pos[0], pos[-1]] and
      origin 'upper' over the columns reversed and transposed (the old
      _int_orient), ylim (pos[0], pos[-1]).  The net rule of that drawing:
      spectrum k of the line sits in the k-th cell from pos[0], and so does
      profile index k.  Acquired there means any non-zero sample (the old
      _pipeline).
      One change, asked for on 2026-10-15: an unfinished line (its image row
      has unacquired pixels, or one of its spectra is all zeros or ends in
      zeros) throws out its last spectrum with data, and its stripe uses the
      same indexing as every other line (pixel m in the m-th cell from
      pos[0], the same start and the same cell size, the V_bias reversal as
      for a finished line) with the pixels the scan never took left blank
      instead of drawn as zeros; the rows of the dropped and the
      never-acquired spectra are not drawn.  The panel title and the console
      say which cells the taken pixels and the kept rows occupy and how many
      overlap.

Each window holds one panel per file: the forward topo stripe (left) and the
heatmap (right), '< line' / 'line >' buttons for files with several lines,
and the panel's own 'Feenstra', 'calibrate', 'normalize' toggles (the
package's pipeline, STS.process_arrays: calibration against the raw I(V), the
division by I/V with the current's 0 V value subtracted and no broadening,
unit-area normalization; filters off).  No markers, no reverse stripe.

Run from Spyder on the files set in the __main__ block.  Nothing is written.
"""

if __package__ in (None, ''):
    # checks/ sits inside sm4_Browser: the package's parent folder must be importable.
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import os
import numpy as np
from sm4_Browser import qtplot as plt
from sm4_Browser import qtplot as mcolors

from sm4_Browser import line as line_mod, topo, plots, widgets
from sm4_Browser.sm4file import Sm4File
from sm4_Browser.sts import STS


# ---------------------------------------------------------------------- #
#  The old placement (copied from the zip's line.py and plots.py)          #
# ---------------------------------------------------------------------- #

def old_line_positions(fast, scalefactor):
    """zip line.py, line_positions: RHK_Xoffset + i * RHK_Xscale over shape[0] of the page, reversed, from zero."""
    size = int(np.shape(fast.data)[0])
    pos = np.array(
        [fast.attrs['RHK_Xoffset'] + i * fast.attrs['RHK_Xscale']
         for i in range(size)], dtype=np.float64
    ) * 1.0e10  # convert position to Å
    pos = (pos / scalefactor) - (pos[0] / scalefactor)
    # Reverse and shift so that the first value is zero.
    return pos[::-1] - pos[::-1][0]


def old_topography(pages, scalefactor, topo_mode='forward', name=''):
    """zip line.py, topography: spym's page array as it comes (the reverse page mirrored), the axes from the attributes."""
    topo_pages = [p for p in pages if p.attrs.get('RHK_PageTypeName') == 'RHK_PAGE_TOPOGRAPHIC']
    if not topo_pages:
        raise ValueError(f"{name}: no topographic page.")
    if topo_mode == 'forward':
        page = topo_pages[0]
    elif topo_mode == 'reverse':
        if len(topo_pages) < 2:
            raise ValueError("File does not contain both forward and reverse topo pages.")
        page = topo_pages[1]
    else:
        raise ValueError("topo_mode must be 'forward' or 'reverse'.")

    # The shape of the topography data (an image page).
    ny, nx = np.shape(page.data)

    # Compute coordinate arrays from the attributes.
    topo_x = np.array([page.attrs['RHK_Xoffset'] + i * page.attrs['RHK_Xscale']
                       for i in range(nx)]) * 1.0e10  # in Å
    topo_y = np.array([page.attrs['RHK_Yoffset'] + i * page.attrs['RHK_Yscale']
                       for i in range(ny)]) * 1.0e10
    topo_x = (topo_x / scalefactor) - (topo_x[0] / scalefactor)
    topo_y = (topo_y / scalefactor) - (topo_y[0] / scalefactor)

    # Get topo_data.
    if topo_mode == 'forward':
        topo_data = np.array(page.data)
    else:
        topo_data = np.array(page.data)[:, ::-1]                 # mirrored, see above

    # The page is in metres already (the loader applied RHK_Zscale and RHK_Zoffset).
    topo_data = topo_data * 1E10
    return topo_x, topo_y, topo_data


def old_orient(curves, energy):
    """zip line.py, _int_orient: columns (npts, N) reversed and transposed, energy ascending."""
    c = np.asarray(curves, dtype=np.float64)[:, ::-1].T
    if energy[0] > energy[-1]:
        c = c[:, ::-1]
    return c


def old_draw_stripe(ax, pos, profile, n_full):
    """
    zip plots.py, draw_topo_stripe: the profile as cells spread evenly from
    pos[0] to pos[-1] (n_full cells, pixel m in the m-th cell from pos[0]);
    NaN cells (pixels an unfinished line never took) are not drawn.
    """
    ax.clear()
    prof = np.asarray(profile, dtype=np.float64)
    step = (pos[-1] - pos[0]) / float(n_full)
    y_edges = pos[0] + step * np.arange(len(prof) + 1)
    x_edges = np.array([0.0, 1.0])
    ok = np.isfinite(prof)
    lo, hi = (float(np.min(prof[ok])), float(np.max(prof[ok]))) if ok.any() else (0.0, 0.0)
    norm = (mcolors.TwoSlopeNorm(vmin=lo, vcenter=(lo + hi) / 2, vmax=hi) if hi > lo else None)
    ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(prof)[:, None],
                  cmap=plots.TOPO_CMAP, norm=norm, shading='flat')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_title("Topo")


# ---------------------------------------------------------------------- #
#  One file in a panel                                                    #
# ---------------------------------------------------------------------- #

class FilePanel:
    """
    One file's line in one window: the data of the chosen line under the
    window's placement ('current' or 'old'), the panel's toggles, the
    drawing.  The file is loaded once (Sm4File) and shared by both windows.
    """

    def __init__(self, sm4, mode, settings):
        self.sm4 = sm4
        self.mode = mode
        self.s = settings
        self.name = sm4.name
        self.line = 0
        self.feenstra = bool(settings['feenstra'])
        self.calibrate = bool(settings['calibrate'])
        self.normalize = bool(settings['normalize'])
        self.fast_page = line_mod.fast_page_of(sm4.f.pages, sm4.name)
        self.n_fast = int(sm4.n_fast)
        self.n_lines = int(sm4.n_lines)
        self.energy = np.asarray(sm4.energy, dtype=np.float64)
        self.energy_asc = np.sort(self.energy)
        self.ax_topo = self.ax_heat = None
        self.im = None
        self.widgets = []

    # --- the line's columns ---

    def _cols(self):
        """The global column indices of the chosen line (clipped to the spectra the file holds)."""
        return np.arange(self.line * self.n_fast, min((self.line + 1) * self.n_fast, self.sm4.n_spectra))

    def _curves(self, cols, acquired):
        """The pipeline on the line's columns with the panel's toggles (npts, n); NaN for columns not acquired."""
        lia = np.asarray(self.sm4.lia, dtype=np.float64)[:, cols]
        iv = None if self.sm4.iv is None else np.asarray(self.sm4.iv, dtype=np.float64)[:, cols]
        divide = self.feenstra
        if divide and iv is None:
            print(f"{self.name}: no Current channel; the Feenstra division is not available.")
            divide = False
        calibrate = self.calibrate and iv is not None
        c = STS.process_arrays(self.energy, lia, iv, None, None, divide, float(self.s['iv_broadening']),
                               channel='didv', acquired=acquired, iv_zero_sub=bool(self.s['iv_zero_sub']),
                               normalize=self.normalize, calibrate=calibrate, vmod=self.sm4.vmod)
        c = np.array(c, dtype=np.float64)
        c[:, ~np.asarray(acquired, dtype=bool)] = np.nan
        return c

    # --- the current placement ---

    def _draw_current(self):
        """Window 1 (the module docstring): the kept rows at their recorded positions (plots.RowHeatmap), the forward stripe cells at theirs."""
        sm4, i, sf = self.sm4, self.line, float(self.s['scalefactor'])
        cols = self._cols()
        kept = np.asarray(sm4.acquired, dtype=bool)[cols]
        pos = line_mod.line_axis(sm4.spec_xy_m, self.n_fast, i, sf, fast_page=self.fast_page)[:len(cols)]
        curves = self._curves(cols, kept)
        img = curves.T                                           # row j = spectrum j at pos[j]
        if self.energy[0] > self.energy[-1]:
            img = img[:, ::-1]
        img = img[kept]
        rows = pos[kept]
        # The stripe: the line's forward row, every pixel at its own recorded location.
        profiles, positions, _ = line_mod.stripe_rows(sm4, [(i, 'forward')], line=i, scalefactor=sf)
        topo_x, profiles = line_mod.common_stripe_axis(profiles, positions)
        profile = topo.stripe_profile(topo_x, profiles, topo.processor_defaults())
        self.ax_heat.clear()
        if len(rows) == 0:
            self.ax_heat.set_title(f"{self._short()}  line {i}: no kept spectrum")
            self.ax_topo.clear()
            return
        self.im = plots.RowHeatmap(self.ax_heat, img, self.energy_asc, rows, self.s['cmap'], gamma=1.0)
        x0, x1, y0, y1 = self.im.extent()
        self.ax_heat.set_xlim(x0, x1)
        self.ax_heat.set_ylim(y0, y1)
        plots.draw_topo_stripe(self.ax_topo, topo_x, profile)
        self.ax_topo.set_ylim(y0, y1)
        n_cells = int(np.isfinite(profile).sum())
        self.ax_heat.set_title(f"{self._short()}  line {i} of {self.n_lines - 1}  V_bias {sm4.V_bias:+.2f} V\n"
                               f"{int(kept.sum())} of {len(cols)} spectra kept, {n_cells} stripe cells; "
                               + self._toggle_text(), fontsize=9)
        print(f"[current] {self.name} line {i}: {int(kept.sum())} of {len(cols)} spectra kept at their recorded "
              f"positions ({rows.min():.1f} to {rows.max():.1f} A), {n_cells} forward pixels at their recorded "
              f"positions ({np.nanmin(topo_x[np.isfinite(profile)]):.1f} to "
              f"{np.nanmax(topo_x[np.isfinite(profile)]):.1f} A)" if n_cells else
              f"[current] {self.name} line {i}: {int(kept.sum())} spectra kept, no stripe cell")

    # --- the old placement ---

    def _old_raw_row(self, page, spym_row):
        """
        The pixels the scan took on the raw row behind spym's row (the old
        code's td[line, :]): spym mirrors the row along the fast axis when
        RHK_Xscale < 0, so the acquisition order is the row read back the
        other way in that case.  Returns (the row in acquisition order, the
        taken pixels: the non-zero ones, in that order).
        """
        acq = spym_row[::-1] if float(page.attrs['RHK_Xscale']) < 0 else spym_row.copy()
        nz = np.flatnonzero(acq != 0)
        taken = acq[nz] if len(nz) else acq[:0]
        if len(nz) and (nz[-1] - nz[0] + 1) != len(nz):
            print(f"{self.name}: the taken pixels of this row are not one contiguous block "
                  f"({len(nz)} non-zero of {len(acq)}); drawn in acquisition order anyway.")
        return acq, taken

    def _draw_old(self):
        """Window 2 (the module docstring): the zip's placement, spectrum k and pixel k in the k-th cell from pos[0], with the unfinished-line change."""
        sm4, i, sf = self.sm4, self.line, float(self.s['scalefactor'])
        pages = sm4.f.pages
        cols = self._cols()
        N = len(cols)
        pos = old_line_positions(self.fast_page, sf)             # the full line's axis
        n_full = len(pos)
        lia = np.asarray(sm4.lia, dtype=np.float64)[:, cols]
        iv = None if sm4.iv is None else np.asarray(sm4.iv, dtype=np.float64)[:, cols]
        acquired = np.any(lia != 0, axis=0)                      # the old rule: any non-zero sample
        # The old stripe row: spym's array, row `line`, V_bias flip.
        topo_x_old, _, td = old_topography(pages, sf, 'forward', sm4.name)
        if i >= td.shape[0]:
            self.ax_heat.clear()
            self.ax_topo.clear()
            self.ax_heat.set_title(f"{self._short()}  line {i}: no row {i} in the page ({td.shape[0]} rows)")
            return
        spym_row = np.array(td[i, :], dtype=np.float64)
        topo_page = [p for p in pages if p.attrs.get('RHK_PageTypeName') == 'RHK_PAGE_TOPOGRAPHIC'][0]
        acq_row, taken = self._old_raw_row(topo_page, spym_row)
        # Unfinished line: unacquired pixels on the row, or a spectrum all zeros or ending in zeros.
        ends_zero = np.any(lia != 0, axis=0) & (lia[-1, :] == 0)
        if iv is not None:
            ends_zero |= np.any(iv != 0, axis=0) & (iv[-1, :] == 0)
        unfinished = bool(np.any(acq_row == 0)) or bool(np.any(~acquired)) or bool(np.any(ends_zero))
        note = ""
        # The old indexing for every line, finished or not: the row as spym gives it (the fast axis
        # mirrored by spym when RHK_Xscale < 0), reversed when V_bias > 0, pixel m in the m-th cell
        # from pos[0] over the full line.
        row_spym = spym_row.astype(np.float64).copy()
        n_fill = 0
        if unfinished:
            with_data = np.flatnonzero(acquired)
            if len(with_data):
                acquired = acquired.copy()
                acquired[with_data[-1]] = False               # the last spectrum with data is thrown out
            row_spym[row_spym == 0] = np.nan                  # pixels the image page does not hold: blank
            # The image page holds the completed lines only; the pixels an unfinished line took are
            # the Z recorded at its spectra (the embedded Topography channel), in spectrum order,
            # which is the raw row's order (pixel j of raw row i is spectrum i x N + j).  They are
            # put into spym's orientation (mirrored when RHK_Xscale < 0, as spym mirrors the image
            # row) and used where the image row holds nothing, for the kept spectra only.
            z = sm4.spectra_z()
            if z is not None:
                n = min(len(cols), len(row_spym))
                vals = np.asarray(z, dtype=np.float64)[cols[:n]] * 1e10
                vals[~acquired[:n]] = np.nan
                zrow = np.full(len(row_spym), np.nan)
                zrow[:n] = vals
                if float(topo_page.attrs['RHK_Xscale']) < 0:
                    zrow = zrow[::-1]
                fill = np.isnan(row_spym) & np.isfinite(zrow)
                row_spym[fill] = zrow[fill]
                n_fill = int(fill.sum())
        profile_raw = row_spym[::-1] if sm4.V_bias > 0 else row_spym
        if unfinished:
            cells = np.flatnonzero(np.isfinite(profile_raw))  # the cells (from pos[0]) the taken pixels occupy
            rows_kept = np.flatnonzero(acquired)              # the cells the kept spectra occupy
            overlap = np.intersect1d(cells, rows_kept)
            note = (f"unfinished: last spectrum (column {with_data[-1] if len(with_data) else '-'}) thrown out; "
                    f"{len(cells)} pixels ({len(taken)} from the image row, {n_fill} from the Z channel) in cells "
                    f"{cells.min() if len(cells) else '-'}..{cells.max() if len(cells) else '-'}, kept rows in cells "
                    f"{rows_kept.min() if len(rows_kept) else '-'}..{rows_kept.max() if len(rows_kept) else '-'}; "
                    f"{len(overlap)} cell(s) overlap")
        curves = self._curves(cols, acquired)
        # The old heatmap: columns reversed and transposed, imshow with origin 'upper' from pos[0] to pos[-1]
        # (spectrum k in the k-th cell from pos[0]); an unfinished line's drawn rows are its kept ones only.
        self.ax_heat.clear()
        e_left, e_right = float(self.energy_asc[0]), float(self.energy_asc[-1])
        if unfinished:
            keep = np.flatnonzero(acquired)
            if len(keep) == 0:
                self.ax_heat.set_title(f"{self._short()}  line {i}: no kept spectrum")
                self.ax_topo.clear()
                return
            n_draw = int(keep[-1]) + 1                        # the kept spectra are the first n_draw columns
            img = old_orient(curves[:, :n_draw], self.energy)
            y_end = pos[0] + (pos[-1] - pos[0]) * n_draw / float(n_full)
        else:
            img = old_orient(curves, self.energy)
            y_end = pos[-1]
        self.im = plots.draw_heatmap(self.ax_heat, img, [e_left, e_right, float(pos[0]), float(y_end)],
                                     self.s['cmap'], gamma=1.0)
        self.ax_heat.set_xlim(e_left, e_right)
        self.ax_heat.set_ylim(float(pos[0]), float(pos[-1]))
        # The old stripe: the profile through the stripe processing, cells from pos[0] in full-line steps.
        x_proc = topo_x_old[:len(profile_raw)] - np.min(topo_x_old)
        profile = topo.stripe_profile(x_proc, [np.asarray(profile_raw, dtype=np.float64)], topo.processor_defaults())
        old_draw_stripe(self.ax_topo, pos, profile, n_full)
        self.ax_topo.set_ylim(float(pos[0]), float(pos[-1]))
        self.ax_heat.set_title(f"{self._short()}  line {i} of {self.n_lines - 1}  V_bias {sm4.V_bias:+.2f} V\n"
                               f"{int(acquired.sum())} of {N} spectra drawn; " + self._toggle_text()
                               + (f"\n{note}" if note else ""), fontsize=9)
        print(f"[old] {self.name} line {i}: spym row {i} of the page ({td.shape[0]} x {td.shape[1]}), "
              f"RHK_Xscale {float(topo_page.attrs['RHK_Xscale']):+.3e}, RHK_Yscale "
              f"{float(topo_page.attrs['RHK_Yscale']):+.3e}, V_bias {sm4.V_bias:+.2f} V -> profile "
              + ("reversed (V_bias > 0)" if sm4.V_bias > 0 else "as spym gives it")
              + f"; pos from {pos[0]:.1f} to {pos[-1]:.1f} A; {int(acquired.sum())} of {N} spectra drawn"
              + (f"; {note}" if note else ""))

    # --- the panel ---

    def _short(self):
        """The file's stem for the panel title, the middle elided when longer than 34 characters."""
        stem = os.path.splitext(self.name)[0]
        return stem if len(stem) <= 34 else stem[:16] + "…" + stem[-16:]

    def _toggle_text(self):
        """The three toggles' states for the panel title."""
        return (f"Feenstra {'on' if self.feenstra else 'off'}, calibrate {'on' if self.calibrate else 'off'}, "
                f"normalize {'on' if self.normalize else 'off'}")

    def draw(self):
        """Draw the panel under its window's placement; an exception becomes the panel's title and a console line."""
        try:
            if self.mode == 'current':
                self._draw_current()
            else:
                self._draw_old()
        except Exception as exc:
            self.ax_heat.clear()
            self.ax_topo.clear()
            self.ax_heat.set_title(f"{self._short()}  line {self.line}: {exc}", fontsize=8)
            print(f"[{self.mode}] {self.name} line {self.line}: {exc}")
        self.ax_heat.set_xlabel("Energy (eV)")
        self.ax_topo.set_ylabel("Position (Å)")

    def build(self, fig, left, width):
        """The panel's axes and controls in the figure over [left, left + width] of its width."""
        gap = 0.012
        w_topo = 0.045
        self.ax_topo = fig.add_axes([left, 0.30, w_topo, 0.58])
        self.ax_heat = fig.add_axes([left + w_topo + gap, 0.30, width - w_topo - gap - 0.01, 0.58])
        # Controls: line buttons (when the file has several lines) and the three toggles.
        y = 0.12
        if self.n_lines > 1:
            steps = widgets.StepButtons(fig, '< line', 'line >', left, y)
            steps.on_change(self._step)
            self.widgets.append(steps)
        chk = widgets.make(fig, 'pipeline', left + 0.12, 0.04, labels=['Feenstra', 'calibrate', 'normalize'],
                           values=[self.feenstra, self.calibrate, self.normalize])
        chk.on_change(lambda label, states: self._toggle(label))
        self.widgets.append(chk)
        self.fig = fig

    def _step(self, d):
        """'< line' / 'line >': the previous / next line of the file (the ends are not passed; a console line)."""
        new = self.line + int(d)
        if new < 0 or new >= self.n_lines:
            print(f"{self.name}: line {new} is out of range 0..{self.n_lines - 1}.")
            return
        self.line = new
        self.draw()
        self.fig.canvas.draw_idle()

    def _toggle(self, label):
        """One of the panel's three toggles clicked: its state flipped, the panel redrawn."""
        if label == 'Feenstra':
            self.feenstra = not self.feenstra
        elif label == 'calibrate':
            self.calibrate = not self.calibrate
        elif label == 'normalize':
            self.normalize = not self.normalize
        self.draw()
        self.fig.canvas.draw_idle()


# ---------------------------------------------------------------------- #
#  A window: one panel per file                                           #
# ---------------------------------------------------------------------- #

class AlignmentWindow:
    """One window of placement `mode` ('current' or 'old'): a FilePanel per loaded file side by side, 6.2 in wide each."""

    def __init__(self, files, mode, settings):
        title = {'current': "current alignment (the package as it is)",
                 'old': "old alignment (the zip's line.py; unfinished line: last spectrum out, same indexing, untaken pixels blank)"}[mode]
        self.fig = plt.figure(figsize=(6.2 * len(files), 7.5))
        try:
            self.fig.canvas.manager.set_window_title(title)
        except Exception:
            pass
        self.fig.suptitle(title, fontsize=11)
        self.panels = []
        n = max(len(files), 1)
        margin, inner = 0.045, 0.03
        width = (1.0 - 2 * margin - inner * (n - 1)) / n
        for k, sm4 in enumerate(files):
            p = FilePanel(sm4, mode, settings)
            p.build(self.fig, margin + k * (width + inner), width)
            p.draw()
            self.panels.append(p)


def load_files(paths):
    """The grid files with spectra among `paths` as Sm4File (the others reported and skipped), each with a console line and its self check."""
    out = []
    for p in paths:
        try:
            f = Sm4File(p)
        except Exception as exc:
            print(f"{os.path.basename(p)}: not loaded ({exc}).")
            continue
        if f.kind != 'grid' or f.spectra is None or f.n_fast < 1:
            print(f"{f.name}: kind '{f.kind}', not a grid with spectra; skipped.")
            continue
        if f.self_check_text:
            print(f.self_check_text)
        print(f"{f.name}: {f.n_lines} line(s) x {f.n_fast} spectra, {f.n_acquired} of {f.n_spectra} kept, "
              f"{len(f.topo_pages)} topo page(s), V_bias {f.V_bias:+.2f} V")
        out.append(f)
    return out


if __name__ == '__main__':
    FOLDER = r'experiment_dir'
    FILES = ['file.sm4',
             'file.sm4',
             'file.sm4']
    SETTINGS = {
        'scalefactor': 2.75,      # the browser's position scale factor
        'cmap': 'jet',
        'feenstra': True,         # the toggles' starting state (the Line window's defaults)
        'calibrate': True,
        'normalize': True,
        'iv_zero_sub': True,      # the Feenstra settings the Line window opens with
        'iv_broadening': 0.0,
    }

    files = load_files([os.path.join(FOLDER, n) for n in FILES])
    if not files:
        raise SystemExit("no grid file loaded.")
    windows = [AlignmentWindow(files, 'current', SETTINGS), AlignmentWindow(files, 'old', SETTINGS)]
    plt.show()
