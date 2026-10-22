"""
Line averages: several grid lines (from Line windows, 'line → new avg' /
'line → existing avg') aligned along position and averaged.  LineAverages
is the registry of the open line-average windows of an experiment (the
browser owns one and hands it to the Line windows; a stand-alone Line
makes its own); LineAverage is the window, built on the Line window's
marker toolset over arrays (align.py does the alignment and the
averaging); AlignmentEditor is the 'edit alignment' window.

The average, per run (LineAverage.run):
  - every member through the pipeline on its own raw columns (filters,
    calibration, the division by its own location's broadened I/V with
    'Feenstra' on, normalization, each as toggled: the Feenstra ratio is
    formed per location before any averaging);
  - the alignment: the DTW of every member against the main (the
    'reference' box of the editor, the first member at open) on the dI/dV
    with the division off and the current with 0Vsub as toggled, the
    stretch equalized over every member (align.equalize_stretch); after
    'apply' in the editor the marker-section maps instead ('auto' returns
    to the DTW);
  - the common grids (the finest position step; the finest energy step
    over the common energy range), every member's heatmap and stripe
    resampled onto them, the mean over the members covering each pixel
    ('overlapping' off) or over the pixels every member covers
    ('overlapping' on).
A filter, pipeline, Feenstra, channel, overlapping, crop, or alignment change needs
'run' again (the title says so); 'apply' in the editor runs; 'remove
last' does not; 'restore' reruns when a run exists.  The Feenstra
companion opens after the first run with the averaged dI/dV (division
off), the averaged current, the averaged broadened I/V, and the averaged
ratio, every one an average of per-location quantities through the same
maps, under its own 'calibrate' / 'normalize' flags.

Positions are in Å after each member's own scale factor (the scale the
Line window was opened with, line.line_axis), kept with the member
('scalefactor') and in the record; on reload the browser rebuilds every
member at the scale its timeline gives for the file and notes a
difference on the console (browser._record_scale).  Records are
lineavg<N>_average.json (or <name>_average.json after a rename) in the
'averages' subfolder (records.py).
"""

if __package__ in (None, ''):
    # Run as a file (Spyder F5): make the package importable from wherever its folder sits.
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from datetime import datetime
import numpy as np
from sm4_Browser import qtplot as plt
from sm4_Browser import topo, widgets, plots, records, align
from sm4_Browser.sts import STS, curve_units_of
from sm4_Browser.averaging import Averages, Selection, send_sets_to_fit
from sm4_Browser.feenstra import FeenstraWindow, feenstra_titles, energy_has_zero, iv_corrected, iv_broadened
from sm4_Browser.normalization import normalize_abs
from sm4_Browser.line import Line, fast_page_of, line_axis, stripe_rows, common_stripe_axis


def member_from_file(sm4, line, scalefactor, topo_lines=None, markers=None, centers=None):
    """
    A member entry from a loaded grid file (the record loader,
    browser.load_lineavg_record), as Line.line_entry builds one: the
    Selection of the line's columns (line x n_fast to the next line), the
    positions (line.line_axis: every spectrum at its recorded position
    along the line, Å after `scalefactor`), the stripe rows ((row, pass)
    pairs with every pixel at its own recorded location, line.stripe_rows,
    'topo_x' their common position axis, the pixels of spectra not kept
    NaN; the line's forward row alone when None, as the Line window opens),
    the Line
    window's marker positions and group centres when given (Å), and the
    scale factor.  ValueError for a file that is not a grid with spectra or
    a line index outside 0..n_lines - 1.
    """
    if sm4.kind != 'grid' or sm4.spectra is None:
        raise ValueError(f"{sm4.name} is not a grid file with spectra.")
    line = int(line)
    if line < 0 or line >= sm4.n_lines:
        raise ValueError(f"{sm4.name} has lines 0..{sm4.n_lines - 1}, not {line}.")
    cols = list(range(line * sm4.n_fast, (line + 1) * sm4.n_fast))
    pages = sm4.f.pages
    pos = line_axis(sm4.spec_xy_m, sm4.n_fast, line, scalefactor, fast_page=fast_page_of(pages, sm4.name))
    selection = Selection(sm4, cols, {'line': line, 'positions': [float(p) for p in pos]})
    if topo_lines is None:
        topo_lines = [(line, 'forward')]
    profiles, positions, resolved = stripe_rows(sm4, topo_lines, line=line, scalefactor=scalefactor)
    topo_x, profiles = common_stripe_axis(profiles, positions)
    return {'selection': selection, 'file': sm4, 'path': sm4.path, 'name': sm4.name, 'line': line,
            'pos': np.asarray(pos, dtype=np.float64),
            'topo_x': np.asarray(topo_x, dtype=np.float64),
            'topo_profiles': [p.copy() for p in profiles],
            'topo_lines': resolved,
            'markers': [float(p) for p in (markers or [])],
            'centers': [float(p) for p in (centers or [])],
            'scalefactor': float(scalefactor)}


class LineAverages:
    """
    The registry of an experiment's open line-average windows (by index)
    and the defaults handed to new ones (the browser's: the filter box
    values, fit_settings, the Feenstra state and settings, the channel,
    the colour map, the Averages registry for the 'add' buttons).  The
    index is unique per directory: 1 + the highest among the open windows
    and the saved lineavg<N> records.
    """

    def __init__(self, directory, hampel_defaults, sgolay_defaults, fit_settings=None,
                 divide_iv=True, iv_broadening=0.0, channel='didv', iv_zero_sub=True, iv_broadened=False,
                 cmap='jet', averages=None):
        self.directory = os.path.abspath(directory)
        self.hampel_defaults = hampel_defaults
        self.sgolay_defaults = sgolay_defaults
        self.fit_settings = dict(fit_settings) if fit_settings else {}
        self.divide_iv = bool(divide_iv)
        self.iv_broadening = float(iv_broadening)
        self.iv_zero_sub = bool(iv_zero_sub)
        self.iv_broadened = bool(iv_broadened)
        self.channel = channel if channel in STS.CHANNELS else 'didv'
        self.cmap = cmap
        self.averages = averages
        self.windows = {}

    def open(self):
        """index -> window of the open line-average windows (closed ones are dropped)."""
        self.windows = {i: w for i, w in self.windows.items() if w.is_open()}
        return self.windows

    def saved_indices(self):
        """The 'index' of every line-average record in the directory's 'averages' subfolder."""
        return records.saved_lineavg_indices_of(self.directory)

    def next_index(self):
        """1 + the highest index among the open windows and the saved records (1 when there is none)."""
        return 1 + max([0] + list(self.open()) + self.saved_indices())

    def by_name(self, name):
        """The open window of that name, or None."""
        for w in self.open().values():
            if w.name == name:
                return w
        return None

    def names(self):
        """The names of the open windows in index order."""
        return [w.name for _, w in sorted(self.open().items())]

    def new(self, first, index=None, name=None, divide_iv=None, iv_broadening=None, channel=None,
            iv_zero_sub=None, iv_broadened=None):
        """Open a line-average window on its first member (a Line.line_entry dict) with the registry's defaults (each overridable; the index next_index unless given, the name lineavg<index> unless given), register it, return it."""
        index = self.next_index() if index is None else int(index)
        w = LineAverage(index, self.directory, first, self.hampel_defaults, self.sgolay_defaults,
                        fit_settings=self.fit_settings,
                        divide_iv=self.divide_iv if divide_iv is None else divide_iv,
                        iv_broadening=self.iv_broadening if iv_broadening is None else iv_broadening,
                        channel=self.channel if channel is None else channel,
                        name=name,
                        iv_zero_sub=self.iv_zero_sub if iv_zero_sub is None else iv_zero_sub,
                        iv_broadened=self.iv_broadened if iv_broadened is None else iv_broadened,
                        cmap=self.cmap, averages=self.averages)
        self.windows[index] = w
        return w

    def add(self, target, entry, channel=None):
        """
        A line (a Line.line_entry dict) into a line average: `target` None
        or '' opens a new window on it (on `channel` when the line holds
        it); a name adds to the open window of that name (ValueError when
        none is open).  A window that already holds the line says so.
        Returns the window.
        """
        target = (target or '').strip()
        if target:
            w = self.by_name(target)
            if w is None:
                raise ValueError(f"no open line average named '{target}' (open: {', '.join(self.names()) or 'none'}).")
            if w.has(entry):
                print(f"{w.name} already holds {entry['name']}: line {entry['line']}.")
                return w
            w.add(entry)
            return w
        held = entry['selection'].channels()
        ch = channel if channel in held else (self.channel if self.channel in held else held[0])
        return self.new(entry, channel=ch)

    def holding(self, path):
        """Indices of the open windows holding a line of that file path."""
        return sorted(i for i, w in self.open().items() if any(m['path'] == path for m in w.members))


class LineAverage(Line):
    """
    One indexed line average (lineavg<N>, or the name in its name box): the
    Line window's marker toolset (groups, 'Markers' sliders with 'add new'
    / 'add existing', Average, stack, substack, dragging, stripe, curve
    panel, 'γ', the window's own 'Feenstra' toggle and companion, the
    'calibrate' / 'normalize' box, the 'channel' radio, the filter box)
    over the aligned average of its members (see the module docstring),
    plus 'name', 'run', 'edit alignment', 'overlapping', 'save', 'remove last',
    'restore', the members line, and 'Send to fit'.  Before the first
    run the panels show the main member's own line (unaligned) and the
    title says run needed; after a run the heatmap's rows are the common
    positions (Å) and its columns the common energies.  Hidden Line
    window controls: the 'line' box and its arrows, 'line → new avg' /
    'line → existing avg', the Line window's 'save' (this window's own
    'save' writes the line-average record), the topo-row controls ('rows'
    boxes, f/r and 'profile' buttons, '+' / '-').  The crop cursors with
    'crop' / 'uncrop' work as in the Line window: the view at once, the
    normalization over the energy crop at the next run (every member's
    columns, current and I/V, and the sent curves).  The 'calibrate' /
    'normalize' box starts with both off.
    The drift toggles and 'drift window' are built but have no Drift (none
    is handed over): they report "no drift model at hand" and stay off.
    The E-markers and the energy slice work on the displayed array.
    """

    RECORD_FORMAT = records.LINEAVG_RECORD_FORMAT
    NAME_XY = (0.03, 0.95)
    MEMBERS_CHARS = 150       # the members line is cut at this many characters

    def __init__(self, index, directory, first, hampel_defaults, sgolay_defaults, fit_settings=None,
                 divide_iv=True, iv_broadening=0.0, channel='didv', name=None, iv_zero_sub=True,
                 iv_broadened=False, cmap='jet', averages=None):
        self.index = int(index)
        self.directory = os.path.abspath(directory)
        self.avg_dir = records.avg_dir_of(self.directory)
        self.ifile = os.path.join(self.directory, f"lineavg{self.index}")   # a stand-in for the Line window's file path:
        #                                                                     the Line methods that read it (titles,
        #                                                                     the directory, the file) are overridden here
        self.cmap = cmap
        self.scalefactor = float(first.get('scalefactor', 2.75))
        self.scan_mode = 'single'
        self.chunk_index = 0
        self.members = []
        self.removed = []                  # removed members, the last removed last
        self.reference = 0                 # the main member (the DTW target)
        self.mode = 'auto'                 # 'auto' (DTW maps) or 'markers' (marker-section maps)
        self.overlapping = False           # 'overlapping': average only where every member covers a pixel
        self.run_needed = True
        self.run_state = None              # the last run (grids, maps, the averaged quantities)
        self._saved_stem = None
        self.name = ''
        self._set_name((name or '').strip() or f"lineavg{self.index}")
        self.editor = None
        self._add_member(first)
        held = first['selection'].channels()
        self.channel = channel if channel in held else held[0]
        self._load_member_axes(0)
        self.plot_line_with_topo_interactive(hampel_filter_params=hampel_defaults,
                                             sgolay_filter_params=sgolay_defaults,
                                             apply_calibration=False, apply_normalize=False,
                                             fit_settings=fit_settings, divide_by_iv=divide_iv,
                                             iv_broadening_V=iv_broadening, averages=averages,
                                             iv_zero_sub=iv_zero_sub, iv_broadened=iv_broadened)

    # --- identity ---

    def is_open(self):
        """True while the window's figure exists (False before it is built)."""
        fig = getattr(self, '_int_fig', None)
        return fig is not None and widgets.figure_is_open(fig)

    @property
    def entries(self):
        """The members (the pop-up list counts them)."""
        return self.members

    def _set_name(self, name):
        """The name set: the console tag, the record stem (records.stem_of), the send label '<averages folder> | <name>', and, when built, the window title, the heatmap title and the name box follow."""
        self.name = str(name)
        self.tag = self.name
        self.stem = records.stem_of(self.name)
        self.label = f"{self.avg_dir} | {self.name}"
        fig = getattr(self, '_int_fig', None)
        if fig is not None:
            try:
                fig.canvas.manager.set_window_title(self.name)
            except Exception:
                pass
            self._int_ax_didv.set_title(self._int_heat_title())
        box = getattr(self, 'box_name', None)
        if box is not None:
            box.set(self.name)

    def _on_rename(self, text):
        """The name box: a new name (its stem non-empty and not another saved record's, else the box is put back); when this window's record was written under another stem, the record and its fit histories are renamed (records.rename_record)."""
        new = str(text).strip()
        if new == self.name:
            return
        stem = records.stem_of(new)
        if not stem:
            print(f"{self.tag}: '{new}' leaves no file name (letters, digits, '.', '_', '-'); name kept.")
            self._set_name(self.name)
            return
        new_rec = records.record_path(self.directory, stem)
        if os.path.isfile(new_rec) and stem != self._saved_stem:
            print(f"{self.tag}: a saved average already uses the file name {os.path.basename(new_rec)}; name kept.")
            self._set_name(self.name)
            return
        old_name, old_stem = self.name, self._saved_stem
        self._set_name(new)
        if old_stem is None or old_stem == stem:
            print(f"{old_name} renamed {self.name}.")
            self._int_fig.canvas.draw_idle()
            return
        renamed = records.rename_record(self.directory, old_stem, stem, self.name, self.label, self.tag)
        self._saved_stem = stem
        print(f"{old_name} renamed {self.name}" + (f"; renamed: {', '.join(renamed)}." if renamed else "."))
        self._int_fig.canvas.draw_idle()

    def _int_tag(self):
        """The console tag: the window's name."""
        return self.name

    def _int_source_label(self):
        """The source for titles (the companion's): '<name>: line average of N line(s)'."""
        return f"{self.name}: line average of {len(self.members)} line(s)"

    def _int_heat_title(self):
        """The heatmap title: the quantity, the name, the member count, ' aligned (DTW|markers to <reference>)' after a run, ', overlapping', ' (run needed)'."""
        return (f"{self._int_quantity()}, {self.name}, {len(self.members)} line(s)"
                + (f" aligned ({'DTW' if self.mode == 'auto' else 'markers'} to {self.reference + 1})"
                   if self.run_state is not None else "")
                + (", overlapping" if self.overlapping else "")
                + (" (run needed)" if self.run_needed else ""))

    def _int_directory(self):
        """The experiment directory (the registry's, not a file's folder)."""
        return self.directory

    def member_label(self, k):
        """Member k as '<file name>:<line>'."""
        m = self.members[k]
        return f"{m['name']}:{m['line']}"

    def members_text(self):
        """The members line: 'k: file:line' per member, cut at MEMBERS_CHARS characters ('+n more')."""
        items = [f"{k + 1}: {self.member_label(k)}" for k in range(len(self.members))]
        text, shown = "members: ", 0
        for k, item in enumerate(items):
            add = ("" if k == 0 else ", ") + item
            if len(text) + len(add) > self.MEMBERS_CHARS and k > 0:
                break
            text += add
            shown += 1
        if shown < len(items):
            text += f" +{len(items) - shown} more"
        return text

    def has(self, entry):
        """Whether a line (by file path and line index) is already a member."""
        return any(m['path'] == entry['path'] and m['line'] == entry['line'] for m in self.members)

    # --- members ---

    def _add_member(self, entry):
        """A copy of the entry dict appended to the members with the per-run keys (a run also sets 'dtw_map' and 'cost')."""
        m = dict(entry)
        m['align_markers'] = None          # the editor's markers in the member's own axis (set by a run)
        m['map'] = None                    # main-frame map (a run)
        m['common'] = None                 # common positions (a run)
        self.members.append(m)

    def add(self, entry):
        """A line (a Line.line_entry dict) joins the average: refused with ValueError when it lacks the window's channel or shares no energy range with the members (align.common_energy); a line already held is only reported.  Run needed; the members line follows."""
        if self.has(entry):
            print(f"{self.name} already holds {entry['name']}: line {entry['line']}.")
            return
        sel = entry['selection']
        if self.channel not in sel.channels():
            raise ValueError(f"{entry['name']} has no {STS.CHANNEL_NAMES[self.channel]} channel.")
        align.common_energy([m['selection'].energy for m in self.members] + [sel.energy])
        self._add_member(entry)
        self.run_needed = True
        print(f"{self.name}: added {entry['name']}: line {entry['line']} ({len(self.members)} lines); run needed.")
        self._refresh_members_line()

    def _on_remove_last(self):
        """'remove last': the last member leaves (one stays; it goes to `removed` for 'restore'); the reference moves to the last member when it was removed; no rerun (the title says run needed)."""
        if len(self.members) <= 1:
            print(f"{self.name}: one line stays.")
            return
        m = self.members.pop()
        self.removed.append(m)
        if self.reference >= len(self.members):
            self.reference = len(self.members) - 1
        self.run_needed = True
        print(f"{self.name}: removed {m['name']}: line {m['line']} ({len(self.members)} lines); run needed.")
        self._refresh_members_line()

    def _on_restore(self):
        """'restore': the last removed member back; the average rerun when a run exists."""
        if not self.removed:
            print(f"{self.name}: nothing removed.")
            return
        m = self.removed.pop()
        self.members.append(m)
        print(f"{self.name}: restored {m['name']}: line {m['line']} ({len(self.members)} lines).")
        self.run_needed = True
        self._refresh_members_line()
        if self.run_state is not None:
            self.run()

    def _refresh_members_line(self):
        """The members line and the heatmap title redrawn."""
        txt = getattr(self, 'txt_members', None)
        if txt is not None:
            txt.set_text(self.members_text())
        self._int_ax_didv.set_title(self._int_heat_title())
        self._int_fig.canvas.draw_idle()

    # --- the Line window's source hooks ---

    def _load_member_axes(self, k):
        """Member k's own line (its positions, energy axis, setpoint, units, modulation, the channel's raw columns and the I(V)) as the display's source, the attributes the Line window reads (energy, pos, chunk, iv_chunk, _int_raw, _int_iv, ...); used before any run for the main member."""
        m = self.members[k]
        sel = m['selection']
        self.energy = np.asarray(sel.energy, dtype=np.float64).copy()
        self.pos = np.asarray(m['pos'], dtype=np.float64)
        self.V_bias = sel.V_bias
        self.I_setpoint = sel.I_setpoint
        self.units_lia, self.units_iets, self.units_iv = sel.units_lia, sel.units_iets, sel.units_iv
        self.vmod = sel.vmod
        self._unset = sel.spectra.unset_value(self.channel)     # a never-written sample of the channel
        self._int_raw = np.asarray(sel.spectra.channel(self.channel), dtype=np.float64).copy()
        self._int_iv = None if sel.iv is None else np.asarray(sel.iv, dtype=np.float64)
        self.chunk = self._int_raw
        self.iv_chunk = self._int_iv

    def _int_load_source(self):
        """The display's raw columns: the main member's own (before a run); after a run the average stands."""
        if self.run_state is None:
            self._load_member_axes(self.reference)

    def _int_n_chunks(self):
        """One line: the window is not sliced (the Line window's line controls are hidden)."""
        return 1

    def _int_set_topo_lines(self, topo_lines):
        """The stripe rows: the main member's own raw profiles (the rows came with the line)."""
        m = self.members[self.reference]
        self._int_topo_x = np.asarray(m['topo_x'], dtype=np.float64)
        self._int_topo_profiles = [np.asarray(p, dtype=np.float64) for p in m['topo_profiles']]
        self._int_topo_lines = list(m['topo_lines'])

    def _int_refresh_topo_rows(self):
        """The topo-row controls stay hidden; the panels follow the slider stack only."""
        self._int_place_panels()
        self._int_fig.canvas.draw_idle()

    def _int_rows_stack_top(self):
        """No topo row stack: the panels' bottom follows the slider stack alone."""
        return 0.0

    def _int_acquired(self, j):
        """Whether row j of the displayed array (a common position after a run; a spectrum of the main member before) holds any finite value."""
        r = int(j)                                             # row j = common position j
        return bool(0 <= r < self._int_curves.shape[0] and np.any(np.isfinite(self._int_curves[r, :])))

    def _int_pipeline(self, hampel_filter_params, sgolay_filter_params, hampel_didv=None, sgolay_didv=None):
        """
        Before a run: the main member's own line through the pipeline (the
        display's source).  After a run the average stands until the next
        run: the displayed array is returned as it is.
        """
        if self.run_state is None:
            return self._int_orient(self._member_columns(self.members[self.reference], self._int_divide_iv,
                                                         self._int_apply_calibration, self._int_apply_normalize,
                                                         hampel_filter_params, sgolay_filter_params,
                                                         hampel_didv=hampel_didv, sgolay_didv=sgolay_didv))
        return self._int_curves

    def _int_apply_feenstra(self, note):
        """A pipeline, Feenstra, or filter change: run needed (the display keeps the last run)."""
        if self.run_state is None:
            return Line._int_apply_feenstra(self, note)
        self.run_needed = True
        print(f"{note}: run needed.")
        self._int_ax_didv.set_title(self._int_heat_title())
        self._int_draw_heatmap()
        self._redraw_curves()
        return True

    def _int_on_filter_change(self):
        """A filter toggle or edit: the settings kept; before a run the display recomputed, after one run needed."""
        hp, sp = self._int_filters.params()
        self._int_hampel_params, self._int_sgolay_params = hp, sp
        self._int_hampel_didv, self._int_sgolay_didv = self._int_filters.params_didv()
        self._int_apply_feenstra("filters")

    def _int_on_channel(self, label):
        """The 'channel' radio: every member must hold the channel; run needed."""
        channel = {v: k for k, v in STS.CHANNEL_NAMES.items()}.get(label, 'didv')
        if channel == self.channel:
            return
        lacking = [self.member_label(k) for k, m in enumerate(self.members) if channel not in m['selection'].channels()]
        if lacking:
            print(f"channel {label} not applied: no such channel in " + ", ".join(lacking) + ".")
            self._int_rad_channel.set(STS.CHANNEL_NAMES[self.channel])
            self._int_fig.canvas.draw_idle()
            return
        self.channel = channel
        self._unset = self.members[self.reference]['selection'].spectra.unset_value(channel)
        if self.run_state is None:
            self._load_member_axes(self.reference)
        self._int_apply_feenstra(f"channel {label}")

    def _on_overlapping(self, flag):
        """The 'overlapping' toggle: the next run averages only where every member covers a pixel (align.average_stack, common_positions); run needed."""
        self.overlapping = bool(flag)
        self.run_needed = True
        print(f"overlapping {'on' if self.overlapping else 'off'}: run needed.")
        self._int_ax_didv.set_title(self._int_heat_title())
        self._int_fig.canvas.draw_idle()

    # --- the pipeline per member ---

    def _member_columns(self, m, divide_iv, calibrate, normalize, hampel, sgolay, channel=None,
                        hampel_didv=None, sgolay_didv=None):
        """A member's columns (npts, N, the member's own energy order) through the pipeline (STS.process_arrays with the given division, calibrate / normalize flags and filter settings (the I(V) pair, then the dI/dV pair; the window's dI/dV settings when the latter are not given), the member's setpoint and forward-sweep reference, the window's Feenstra settings, the channel, and the normalization over the window's energy crop; never-acquired columns NaN).  ValueError for a member without an acquired spectrum, or without the I(V) when the calibration or the division needs it."""
        sel = m['selection']
        channel = self.channel if channel is None else channel
        if hampel_didv is None and sgolay_didv is None:
            hampel_didv, sgolay_didv = getattr(self, '_int_hampel_didv', None), getattr(self, '_int_sgolay_didv', None)
        raw = np.asarray(sel.spectra.channel(channel), dtype=np.float64)
        acquired = np.asarray(sel.acquired, dtype=bool)
        if not acquired.any():
            raise ValueError(f"{self.member_label(self.members.index(m))}: no acquired spectrum.")
        if calibrate and sel.iv is None:
            raise ValueError(f"{m['name']}: no I(V) current channel; the calibration cannot be applied.")
        if divide_iv and channel != 'iets' and sel.iv is None:
            raise ValueError(f"{m['name']}: no I(V) current channel; the division cannot be applied.")
        ref = (getattr(sel.spectra, 'iets_ref' if channel == 'iets' else 'lia_ref', None),
               getattr(sel.spectra, 'iv_ref', None))
        c = STS.process_arrays(sel.energy, raw, sel.iv, hampel, sgolay, divide_iv, self._int_iv_width(),
                               channel=channel, acquired=acquired, iv_zero_sub=self._int_iv_zero_sub,
                               normalize=normalize, calibrate=calibrate, vmod=sel.vmod,
                               norm_range=self._int_e_range, V_bias=sel.V_bias, I_setpoint=sel.I_setpoint,
                               ref=ref, hampel_didv=hampel_didv, sgolay_didv=sgolay_didv, name=m['name'])
        c = np.array(c, dtype=np.float64)
        c[:, ~acquired] = np.nan
        return c

    def _member_current(self, m, normalize):
        """A member's current in amps (filters and 0Vsub as set), unit area per location with `normalize`; None without the channel."""
        sel = m['selection']
        if sel.iv is None:
            return None
        iv = iv_corrected(sel.energy, sel.iv, self._int_iv_zero_sub,
                          self._int_hampel_params, self._int_sgolay_params)
        if normalize:
            iv = normalize_abs(sel.energy, iv, energy_range=self._int_e_range)[0]
        iv = np.array(iv, dtype=np.float64)
        iv[:, ~np.asarray(sel.acquired, dtype=bool)] = np.nan
        return iv

    def _member_ivv(self, m, normalize):
        """A member's broadened I/V in A/V (the division's denominator); None without the channel."""
        sel = m['selection']
        if sel.iv is None:
            return None
        ivv = iv_broadened(sel.energy, sel.iv, self._int_iv_width(), self._int_iv_zero_sub,
                           self._int_hampel_params, self._int_sgolay_params)
        if normalize:
            ivv = normalize_abs(sel.energy, ivv, energy_range=self._int_e_range)[0]
        ivv = np.array(ivv, dtype=np.float64)
        ivv[:, ~np.asarray(sel.acquired, dtype=bool)] = np.nan
        return ivv

    # --- the run ---

    def _align(self, hp, sp):
        """
        The maps of every member (the DTW against the main on the dI/dV
        with the division off and the current with 0Vsub as set, both on
        the common energy grid, align.align_to_main; in 'markers' mode the
        marker sections instead, align.marker_map; a member, or a main,
        without editor markers keeps its DTW map, and a marker mismatch is
        reported with the DTW map kept), the equalized
        common positions (align.equalize_stretch), and the editor's
        markers (auto mode: the master's markers through the maps,
        _place_align_markers).  Sets m['dtw_map'], m['cost'] (the DTW
        total; 0 for the main), m['map'], m['common'], m['align_markers'].
        Returns the common energy grid.
        """
        ref = self.members[self.reference]
        grid_e = align.common_energy([m['selection'].energy for m in self.members])
        didv = [align.on_energy_grid(self._member_columns(m, False, self._int_apply_calibration,
                                                          self._int_apply_normalize, hp, sp),
                                     m['selection'].energy, grid_e) for m in self.members]
        ivs = []
        for m in self.members:
            iv = self._member_current(m, False)
            ivs.append(None if iv is None else align.on_energy_grid(iv, m['selection'].energy, grid_e))
        r = self.reference
        acq_ref = np.asarray(ref['selection'].acquired, dtype=bool)
        maps = []
        for k, m in enumerate(self.members):
            if k == r:
                maps.append(np.asarray(m['pos'], dtype=np.float64).copy())
                m['cost'] = 0.0
                continue
            mp, total = align.align_to_main(didv[k], ivs[k], m['pos'], np.asarray(m['selection'].acquired, dtype=bool),
                                            didv[r], ivs[r], ref['pos'], acq_ref)
            m['dtw_map'] = mp
            m['cost'] = total
            maps.append(mp)
        ref['dtw_map'] = maps[r]
        if self.mode == 'markers':
            for k, m in enumerate(self.members):
                if k == r or not m.get('align_markers') or not ref.get('align_markers'):
                    continue
                try:
                    maps[k] = align.marker_map(m['pos'], m['dtw_map'], m['align_markers'], ref['align_markers'])
                except ValueError as exc:
                    print(f"{self.name}: {self.member_label(k)} keeps its DTW map ({exc}).")
        positions = [np.asarray(m['pos'], dtype=np.float64) for m in self.members]
        common, axis, Q = align.equalize_stretch(maps, positions, r)
        for k, m in enumerate(self.members):
            m['map'] = maps[k]
            m['common'] = common[k]
        if self.mode == 'auto':
            self._place_align_markers()
        return grid_e

    def _place_align_markers(self):
        """
        The editor's markers in auto mode: the master set is the member with
        the most markers (its Line window's markers plus group centres),
        put into every member at the mapped positions (the master's common
        positions, each member's own position there), its own discarded.
        """
        counts = [len(m.get('markers', [])) + len(m.get('centers', [])) for m in self.members]
        master = int(np.argmax(counts)) if any(counts) else self.reference
        mm = self.members[master]
        own = sorted(list(mm.get('markers', [])) + list(mm.get('centers', [])))
        if not own:
            p = np.asarray(mm['pos'], dtype=np.float64)
            own = [float(p[0] + 0.5 * (p[-1] - p[0]))]
        common = align.map_at(np.asarray(own), mm['pos'], mm['common'])
        for m in self.members:
            m['align_markers'] = [float(v) for v in align.inverse_map_at(common, m['pos'], m['common'])]

    def _average_of(self, member_arrays, state):
        """Member columns (one (npts, N) array per member on its own energy axis) resampled onto the run's grids through the maps (align.resample_columns) and averaged (align.average_stack with 'overlapping'), (n_grid_e, n_grid_pos); None when any member's array is None."""
        stack = []
        for m, c in zip(self.members, member_arrays):
            if c is None:
                return None
            stack.append(align.resample_columns(c, m['selection'].energy, state['grid_e'], m['common'], state['grid_pos']))
        return align.average_stack(np.stack(stack), self.overlapping)

    def _stripe_of(self, state):
        """The members' stripes through the maps (each the Line window's stripe rule, topo.stripe_profile with the window's processor) resampled onto the position grid, range matched (topo.match_ranges), averaged where any covers, minus the minimum."""
        profiles = []
        for m in self.members:
            prof = topo.stripe_profile(m['topo_x'], m['topo_profiles'], self._int_topo_active)
            # The stripe's cells sit at the pixels' own recorded positions (topo_x) when the
            # member carries them one per cell; else spread over the line as drawn.
            sp = np.asarray(m['topo_x'], dtype=np.float64) if len(m['topo_x']) == len(prof) else align.stripe_positions(m['pos'], len(prof))
            common = align.map_at(sp, m['pos'], m['common'])
            profiles.append(align.resample_profile(prof, common, state['grid_pos']))
        matched = topo.match_ranges(profiles)
        with np.errstate(invalid='ignore'):
            profile = np.nanmean(np.vstack(matched), axis=0)
        if np.any(np.isfinite(profile)):
            profile = profile - np.nanmin(profile)
        return profile

    def run(self):
        """
        'run': the alignment (_align) and the average with the current
        settings (the module docstring): the position grid
        (align.common_positions with 'overlapping'), the average of every member's
        pipeline columns (_average_of); then the display replaced by the
        average (energy = the common energy grid, pos = the position grid,
        the array transposed so row j = common position j), the markers
        clipped to the grid, the combined stripe, the heatmap rebuilt, the
        console line with every member's DTW cost, the curves redrawn, the
        companion opened at the first run (refreshed after), the editor
        refreshed.  Returns True; False with the reason printed when a
        member cannot be processed or aligned (the display unchanged).
        """
        hp, sp = self._int_filters.params()
        self._int_hampel_params, self._int_sgolay_params = hp, sp
        self._int_hampel_didv, self._int_sgolay_didv = self._int_filters.params_didv()
        try:
            grid_e = self._align(hp, sp)
            positions = [np.asarray(m['pos'], dtype=np.float64) for m in self.members]
            grid_pos = align.common_positions([m['common'] for m in self.members], positions, self.overlapping)
            state = {'grid_e': grid_e, 'grid_pos': grid_pos}
            avg = self._average_of([self._member_columns(m, self._int_divide_iv, self._int_apply_calibration,
                                                         self._int_apply_normalize, hp, sp)
                                    for m in self.members], state)
        except Exception as exc:
            print(f"{self.name} run: {exc}")
            return False
        self.run_state = state
        self.run_needed = False
        # The display: the average on the common grids, oriented like the Line window's heatmap.
        self._int_energy = grid_e
        self._int_pos = grid_pos
        self.energy = grid_e
        self._int_curves = avg.T                              # row j = common position j
        lo, hi = float(grid_pos[0]), float(grid_pos[-1])
        self._int_marker_pos = [float(np.clip(p, lo, hi)) for p in self._int_marker_pos]
        self._int_stripe_profile = self._stripe_of(state)
        self._int_reset_heatmap()
        self._int_draw_heatmap()
        plots.draw_topo_stripe(self._int_ax_topo, self._int_pos, self._int_stripe_profile)
        self._int_ax_topo.set_ylabel("Position (Å)")
        costs = ", ".join(f"{self.member_label(k)}: {m.get('cost', 0.0):.3g}" for k, m in enumerate(self.members))
        print(f"{self.name}: {len(self.members)} lines aligned ({'DTW' if self.mode == 'auto' else 'markers'} to "
              f"{self.member_label(self.reference)}; costs {costs}); grid {len(grid_pos)} positions x "
              f"{len(grid_e)} energies{', overlapping pixels only' if self.overlapping else ''}.")
        self._redraw_curves()
        if self._int_feenstra is None:
            self._int_feenstra = FeenstraWindow(self, self._int_source_label())
        else:
            self._int_feenstra_refresh()
        ed = self.editor
        if ed is not None and ed.is_open():
            ed.refresh()
        return True

    def _int_reset_heatmap(self):
        """Drop the image (the grid changed) so that the next draw makes a new one on the new extent."""
        im = getattr(self, '_int_im', None)
        if im is not None:
            try:
                im.remove()
            except Exception:
                pass
        self._int_im = None

    def _redraw_topo_stripe(self):
        """The stripe: the main member's own (before a run), or the run's combined stripe."""
        if self.run_state is None:
            Line._redraw_topo_stripe(self)
            return
        self._int_stripe_profile = self._stripe_of(self.run_state)
        plots.draw_topo_stripe(self._int_ax_topo, self._int_pos, self._int_stripe_profile)
        self._int_ax_topo.set_ylabel("Position (Å)")
        self._redraw_curves()

    def _int_on_topo_toggle(self, active):
        """The topo processor box: the stripe (the main member's or the combined one) redrawn with the new flags."""
        self._int_topo_active = list(active)
        self._redraw_topo_stripe()

    # --- the companion ---

    def _int_feenstra_versions(self, calibrate, normalize):
        """The four averaged versions through the run's maps (the companion's own flags); before a run, the main member's."""
        if self.run_state is None:
            return Line._int_feenstra_versions(self, calibrate, normalize)
        hp, sp = self._int_hampel_params, self._int_sgolay_params
        state = self.run_state
        q = ("d2I/dV2 (IETS)" if self.channel == 'iets' else "dI/dV") + ("" if calibrate else ", uncalibrated")
        nn = "" if normalize else ", not normalized"
        zs = self._int_iv_zero_sub
        t = feenstra_titles(zs, self._int_iv_broadened, self._int_iv_broadening, energy_has_zero(state['grid_e']))
        units = curve_units_of([m['selection'] for m in self.members], self.channel, calibrate, normalize)
        versions = {
            'didv': {'title': f"{q} (division off){nn}, averaged", 'ylabel': f"{q} ({units})"},
            'iv': {'title': t['iv'] + ", averaged", 'ylabel': "I (unit area)" if normalize else "I (A)"},
            'ivv': {'title': t['ivv'] + ", averaged", 'ylabel': "I/V (unit area)" if normalize else "I/V (A/V)"},
            'ratio': {'title': t['ratio'] + f" (division on){nn}, averaged", 'ylabel': "(dI/dV)/(I/V) (arb. units)"},
        }

        def orient(a):
            return a.T                                          # row j = common position j

        versions['didv']['top'] = orient(self._average_of(
            [self._member_columns(m, False, calibrate, normalize, hp, sp) for m in self.members], state))
        iv_avg = self._average_of([self._member_current(m, normalize) for m in self.members], state)
        ivv_avg = self._average_of([self._member_ivv(m, normalize) for m in self.members], state)
        if iv_avg is None or ivv_avg is None:
            for key in ('iv', 'ivv', 'ratio'):
                versions[key]['note'] = "a member has no Current channel"
        else:
            versions['iv']['top'] = orient(iv_avg)
            versions['ivv']['top'] = orient(ivv_avg)
            if self.channel == 'iets':
                versions['ratio']['note'] = "the division applies to dI/dV only"
            else:
                versions['ratio']['top'] = orient(self._average_of(
                    [self._member_columns(m, True, calibrate, normalize, hp, sp) for m in self.members], state))
        return versions, q

    # --- marker groups into the averages, and the send ---

    def _member_column_at(self, m, common_pos):
        """A member's column nearest a common position (own position through the inverse map), or None when never acquired."""
        own = float(align.inverse_map_at(np.array([common_pos]), m['pos'], m['common'])[0])
        j = int(np.argmin(np.abs(np.asarray(m['pos']) - own)))
        return j if bool(m['selection'].acquired[j]) else None

    def _group_selections(self, g):
        """One Selection per member for marker group g (each marker to that member's column nearest its common position)."""
        if self.run_state is None:
            raise ValueError("run the average first.")
        members = self._int_groups()[g]
        out = []
        for m in self.members:
            cols, positions = [], []
            for i in members:
                j = self._member_column_at(m, self._int_marker_pos[i])
                if j is None:
                    continue
                cols.append(m['selection'].cols[j])
                positions.append(float(m['pos'][j]))
            if cols:
                out.append(Selection(m['file'], cols, {'line': int(m['line']), 'positions': positions}))
        if not out:
            print(f"add: avg {g + 1} has no marker on an acquired spectrum; nothing added.")
            return None
        return out

    def _int_on_add_new(self, g):
        """'add new': the group's columns of every member into a new averaging window (one Selection per member)."""
        self._int_picker().close()
        try:
            sels = self._group_selections(g)
        except Exception as exc:
            print(f"add new: {exc}")
            return
        if sels is None:
            return
        try:
            w = self._int_averages().add('', sels[0], channel=self.channel)
            for s in sels[1:]:
                if not w.has(s):
                    w.add(s)
                    print(f"{w.name}: added {s.entry_label} ({s.name}; {len(w.entries)} entries).")
        except Exception as exc:
            print(f"add new: {exc}")
            return
        self._int_avg_windows.append(w)
        self._int_fig.canvas.draw_idle()

    def _int_on_add_existing(self, g, anchor_ax):
        """'add existing': the list of the open averages; a row picked adds every member's Selection of the group to it (those it already holds are reported)."""
        picker = self._int_picker()
        if picker.is_open():
            picker.close()
            return
        try:
            sels = self._group_selections(g)
        except Exception as exc:
            print(f"add existing: {exc}")
            return
        if sels is None:
            return

        def add_to(index):
            w = self._int_averages().open().get(index)
            if w is None:
                return
            for s in sels:
                if w.has(s):
                    print(f"{w.name} already holds {s.entry_label} ({s.name}).")
                    continue
                try:
                    w.add(s)
                except Exception as exc:
                    print(f"add to {w.name}: {exc}")
                    return
                print(f"{w.name}: added {s.entry_label} ({s.name}; {len(w.entries)} entries).")
            self._int_avg_windows.append(w)
            self._int_fig.canvas.draw_idle()

        picker.open(anchor_ax, sels[0], add_to)

    def _int_send_to_fit(self):
        """
        'Send to fit': the marker spectra (Average off: one per marker) or
        the group averages (Average on: one per group) of the averaged line,
        each through send_sets_to_fit with one raw set per member (the
        member's columns nearest the markers' common positions) and the
        avgN setpoint rule; the sent curve is the average rebuilt with
        calibration and normalization on through the run's maps (a console
        line says so when the window's boxes are off), its unfiltered
        counterpart the same with every filter off.  The record is written
        first (save), so the fit history and the record share the stem.  A
        marker on a row without data is left out; an average with a gap
        (NaN) at the marker is not sent.  The fit label is '<averages
        folder> | <name> | marker at <p> A' or '... | average of n markers
        at ...' (' over I/V' with Feenstra on), its fit history
        records.cache_path_of of that stem.  dI/dV only; needs a run.
        """
        if self.run_state is None:
            raise ValueError("run the average first.")
        if self.channel != 'didv':
            raise ValueError("the fit models dI/dV; the window shows IETS.")
        switched = [n for n, on in (('calibration', self._int_apply_calibration),
                                    ('normalization', self._int_apply_normalize)) if not on]
        if switched:
            print("Send to fit: " + " and ".join(switched) + " switched on for the sent curves.")
        hp, sp = self._int_hampel_params, self._int_sgolay_params
        div = self._int_divide_iv
        state = self.run_state
        cf = self._average_of([self._member_columns(m, div, True, True, hp, sp) for m in self.members], state)
        cu = self._average_of([self._member_columns(m, div, True, True, None, None, hampel_didv=None, sgolay_didv=None)
                               for m in self.members], state)
        cf_rows, cu_rows = cf.T, cu.T                          # row j = common position j
        suffix = " over I/V" if div else ""
        groups = self._int_groups() if self._int_avg_on else [[i] for i in range(len(self._int_marker_pos))]
        print(f"{self.name}: record written: {self.save()}")
        for members in groups:
            rows, places = [], []
            for i in members:
                p = self._int_marker_pos[i]
                j = self._int_col_of(p)
                if not self._int_acquired(j):
                    print(f"Send to fit: the marker at {p:.1f} A has no data and is left out.")
                    continue
                rows.append(j)
                places.append(p)
            if not rows:
                continue
            data = Averages.mean_rows(cf_rows[rows, :])
            unf = Averages.mean_rows(cu_rows[rows, :])
            if np.any(~np.isfinite(data)):
                print("Send to fit: the average has gaps at this marker ('overlapping' or the gaps' members); not sent.")
                continue
            parts = []
            for m in self.members:
                cols = []
                for p in places:
                    j = self._member_column_at(m, p)
                    if j is not None:
                        cols.append(j)
                parts.append({'file': m['selection'], 'cols': np.array(sorted(set(cols)), dtype=int)})
            parts = [p for p in parts if p['cols'].size]
            if not parts:
                continue
            where = ", ".join(f"{p:.1f}" for p in places)
            rest = (f"{self.name} | average of {len(places)} markers at {where} A" if self._int_avg_on
                    else f"{self.name} | marker at {where} A") + suffix
            label = f"{self.avg_dir} | {rest}"
            cache = records.cache_path_of(self.directory, records.stem_of(rest))
            describe = (f"{len(self.members)} lines aligned, markers at {where} A"
                        + (", the group's plain mean" if self._int_avg_on else ""))
            an = send_sets_to_fit(label, state['grid_e'], data, unf, parts, hp, sp, div, self._int_iv_width(),
                                     self._int_iv_zero_sub, 'interp', self._int_fit_settings, cache, self.tag,
                                     describe, norm_range=self._int_e_range)
            if an is not None:
                self._int_fits.append(an)

    # --- the record ---

    def record(self):
        """
        The line-average record (records.LINEAVG_RECORD_FORMAT): the index,
        name, send label and directory, the members (path, name, line,
        stripe rows, the Line window's markers and centres, the editor's
        markers in the member's own axis, the scale factor the member was
        made with, the sweep view), the reference, the alignment mode,
        overlapping, the channel, the filter boxes and states, the energy and
        position crops (None = full), the calibrate /
        normalize / Feenstra states and settings, the gamma, the topo
        processor flags, the averaged line's group sizes, marker positions
        and E-markers, Average / stack / substack, the last run's grid
        sizes (None before a run), and the time of writing.
        """
        hp, sp = self._int_filters.values()
        h_on, s_on = self._int_filters.states()
        hd_on, sd_on = self._int_filters.states_didv()
        members = [{'path': os.path.abspath(m['path']), 'name': m['name'], 'line': int(m['line']),
                    'topo_lines': [[int(r), md] for r, md in m['topo_lines']],
                    'markers': [float(p) for p in m.get('markers', [])],
                    'centers': [float(p) for p in m.get('centers', [])],
                    'align_markers': ([float(p) for p in m['align_markers']] if m.get('align_markers') else None),
                    'scalefactor': float(m.get('scalefactor', self.scalefactor)),
                    'sweeps': getattr(m.get('file'), 'sweep_mode', None)}
                   for m in self.members]
        return {'format': self.RECORD_FORMAT,
                'index': self.index,
                'name': self.name,
                'label': self.label,
                'directory': self.directory,
                'members': members,
                'reference': int(self.reference),
                'mode': self.mode,
                'overlapping': bool(self.overlapping),
                'channel': self.channel,
                'energy_range': None if self._int_e_range is None else [float(v) for v in self._int_e_range],
                'pos_range': None if self._int_x_range is None else [float(v) for v in self._int_x_range],
                'hampel': hp, 'sgolay': sp,
                'hampel_on': bool(h_on), 'sgolay_on': bool(s_on),
                'hampel_didv_on': bool(hd_on), 'sgolay_didv_on': bool(sd_on),
                'calibrate': bool(self._int_apply_calibration),
                'normalize': bool(self._int_apply_normalize),
                'divide_iv': bool(self._int_divide_iv),
                'iv_broadening_V': float(self._int_iv_broadening),
                'iv_broadened': bool(self._int_iv_broadened),
                'iv_zero_sub': bool(self._int_iv_zero_sub),
                'gamma': float(self._int_gamma_box.value),
                'topo_active': [bool(a) for a in self._int_topo_active],
                'group_sizes': [int(s) for s in self._int_group_sizes],
                'markers': [float(p) for p in self._int_marker_pos],
                'energy_markers': [float(e) for e in self._int_e_markers],
                'average': bool(self._int_avg_on),
                'stack': bool(self._int_stack_on),
                'substack': bool(self._int_substack_on),
                'run': None if self.run_state is None else {
                    'n_positions': int(len(self.run_state['grid_pos'])),
                    'n_energies': int(len(self.run_state['grid_e']))},
                'saved': datetime.now().isoformat(timespec='seconds')}

    def save(self):
        """The record written to <stem>_average.json in the 'averages' subfolder (records.write_json); returns the path."""
        path = records.write_json(records.record_path(self.directory, self.stem), self.record())
        self._saved_stem = self.stem
        return path

    def _int_on_save(self):
        """'save': the record written and its path printed."""
        print(f"{self.name}: record written: {self.save()}")

    def restore(self, rec):
        """
        Put a record's state into the window (the browser's loader, after
        the members are added): the reference, the mode and the editor's
        markers, overlapping (an older record's 'crop'), then the Line window's record keys through
        Line.restore_markers (the filter boxes and states, the calibrate /
        normalize / Feenstra states and settings, the gamma, the topo
        processor, the group sizes and marker positions, the E-markers,
        Average / stack / substack), then run when no run exists yet (the
        case at loading).
        """
        self.reference = int(np.clip(int(rec.get('reference', 0)), 0, len(self.members) - 1))
        self.mode = 'markers' if rec.get('mode') == 'markers' else 'auto'
        for m, rm in zip(self.members, rec.get('members', [])):
            am = rm.get('align_markers')
            m['align_markers'] = [float(p) for p in am] if am else None
        self.overlapping = bool(rec.get('overlapping', rec.get('crop', False)))
        self.chk_overlapping.set(self.overlapping)
        Line.restore_markers(self, rec)
        if self.run_state is None:
            self.run()

    # --- the alignment editor ---

    def _on_edit_alignment(self):
        """'edit alignment': the AlignmentEditor opened (refreshed when open); needs a run."""
        if self.run_state is None:
            print(f"{self.name}: run the average first ('edit alignment' needs the maps).")
            return
        if self.editor is not None and self.editor.is_open():
            self.editor.refresh()
            return
        self.editor = AlignmentEditor(self)

    def set_reference(self, k):
        """The editor's 'reference' box: member k (0-based) becomes the main; run needed."""
        k = int(k)
        if k < 0 or k >= len(self.members):
            raise ValueError(f"reference must be 1..{len(self.members)}")
        if k == self.reference:
            return
        self.reference = k
        self.run_needed = True
        print(f"{self.name}: reference {self.member_label(k)}; run needed.")
        self._int_ax_didv.set_title(self._int_heat_title())
        self._int_fig.canvas.draw_idle()

    def apply_markers(self):
        """The editor's 'apply': the marker-section maps become the alignment and the average runs."""
        self.mode = 'markers'
        return self.run()

    def auto_markers(self):
        """The editor's 'auto': back to the DTW maps, the markers re-placed, the average run."""
        self.mode = 'auto'
        return self.run()

    def set_align_marker_count(self, n):
        """
        The editor's 'Markers' slider: n markers in every member.  Growing
        adds markers at the free slots of the even layout of each member
        (10% to 90% of its line, the slot farthest from its markers;
        widgets.MarkerGroupControls.grow_groups); shrinking removes from
        the end.  The mode becomes 'markers' and run is needed; nothing
        runs until the editor's 'apply'.
        """
        n = int(n)
        for m in self.members:
            cur = list(m.get('align_markers') or [])
            p = np.asarray(m['pos'], dtype=np.float64)
            lo, hi = float(p[0]), float(p[-1])
            if n < len(cur):
                cur = cur[:n]
            elif n > len(cur):
                slots = list(np.linspace(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo), n)) if n > 1 else [0.5 * (lo + hi)]
                dist = lambda a, b: abs(float(a) - float(b))
                cur.extend(widgets.MarkerGroupControls.grow_groups(n, cur, slots, dist))
            m['align_markers'] = [float(v) for v in cur]
        self.mode = 'markers'
        self.run_needed = True

    # --- the figure ---

    def _int_build_controls(self):
        """The Line window's controls, the Line-only ones hidden (the 'line' box and arrows, the 'line → ...' buttons, 'save', '+' / '-', the topo row controls and their labels), plus this window's own: the name box, the members line, 'run', 'edit alignment', 'save', 'remove last', 'restore' (widgets.RunControls) and 'overlapping'; the window title is the name."""
        Line._int_build_controls(self)
        fig = self._int_fig
        for w in (self._int_box_line, self._int_btn_line_new, self._int_btn_line_exist, self._int_btn_save,
                  self._int_btn_topo_add, self._int_btn_topo_remove):
            w.show(False)
        self._int_line_steps.show(False)
        for row in self._int_topo_row_widgets:            # (box, pass button, profile button) per row
            for wdg in row:
                wdg.show(False)
        self._int_topo_rows_label.set_visible(False)
        self._int_topo_rows_note.set_visible(False)
        self.box_name = widgets.NameBox(fig, *self.NAME_XY, name=self.name)
        self.box_name.on_change(self._on_rename)
        self.txt_members = fig.text(0.17, 0.965, self.members_text(), fontsize=8, ha='left', va='center')
        self.buttons = widgets.RunControls(fig, [('run', 0.76, 0.065), ('edit alignment', 0.76, 0.12),
                                                 ('save', 0.865, 0.01), ('remove last', 0.865, 0.065),
                                                 ('restore', 0.865, 0.12)])
        self.buttons.on('run', self.run)
        self.buttons.on('edit alignment', self._on_edit_alignment)
        self.buttons.on('save', self._int_on_save)
        self.buttons.on('remove last', self._on_remove_last)
        self.buttons.on('restore', self._on_restore)
        self.chk_overlapping = widgets.make(fig, 'overlapping', 0.943, 0.01, value=self.overlapping)
        self.chk_overlapping.on_change(self._on_overlapping)
        try:
            fig.canvas.manager.set_window_title(self.name)
        except Exception:
            pass

    def plot_line_with_topo_interactive(self, **kw):
        """The window (Line.plot_line_with_topo_interactive on the main member's own line, its stripe rows unless given); no Drift is handed over."""
        kw.setdefault('topo_lines', self.members[self.reference]['topo_lines'])
        Line.plot_line_with_topo_interactive(self, **kw)

    def _int_open_companion(self):
        """The companion opens after the first run (run), not with the window."""
        self._int_feenstra = None

    def _int_orient_member(self, arr, m):
        """A member's columns (npts, N) as the editor draws them: rows = the member's locations in reverse order, columns = energy ascending (plots.draw_heatmap's origin 'upper' then puts the last position at the top of the extent)."""
        c = np.asarray(arr, dtype=np.float64)
        e = np.asarray(m['selection'].energy, dtype=np.float64)
        order = np.argsort(e)
        return c[order, :][:, ::-1].T


class AlignmentEditor:
    """
    The 'edit alignment' window of a line average: one row per member with
    its stripe at the left, its dI/dV heatmap (division off) and its
    current heatmap (each with its own colour bar and 'γ' box, through the
    window's filters, pipeline flags, and Feenstra settings; "no Current
    channel" without one), over the member's own positions (Å), and M
    horizontal draggable marker lines on both heatmaps of the member
    (marker j in colour j in every member, clipped to the member's line
    when dragged); one 'Markers' slider (0 to 10, on the first figure)
    that sets the marker count of every member
    (LineAverage.set_align_marker_count), a 'reference' box (the member
    number, from 1, that is the main; a bad number is put back), 'apply'
    (the marker sections become the alignment and the average runs),
    'auto' (back to the DTW maps, the markers re-placed, the average run),
    'close' (every figure closed); one topo colour bar per figure.
    PER_FIGURE (6) members per figure; more open further figures, the
    controls on the first.  Built again when the member count changes.
    """

    PER_FIGURE = 6
    ROW_H, ROW_PITCH, ROW_Y0 = 0.105, 0.145, 0.07
    STRIPE_X, STRIPE_W = 0.035, 0.012
    DIDV_X, IMG_W, CBAR_W, GAP = 0.055, 0.34, 0.008, 0.006
    IV_X = 0.49
    TOPO_CBAR = [0.94, 0.35, 0.01, 0.35]
    M_COLORS = widgets.MarkerGroupControls.M_COLORS

    def __init__(self, parent):
        self.parent = parent
        self.figs = []
        self.rows = {}                     # member index -> row dict (axes, images, gamma boxes)
        self._artists = {}                 # member index -> marker Line2D list
        self._active = None                # (member index, marker index) while dragging
        self._quiet = False
        self.controls = None
        self.build()

    def is_open(self):
        """True while any of the editor's figures exists."""
        return any(widgets.figure_is_open(f) for f in self.figs)

    def close(self):
        """Every figure of the editor closed."""
        for f in self.figs:
            if widgets.figure_is_open(f):
                plt.close(f)
        self.figs = []

    DESIGN = (16, 10)      # the design size (inches): the text zoom of the window is its size relative to this (qtplot.window_zoom)

    def build(self):
        """
        The figures made from scratch (the old ones closed): PER_FIGURE
        rows per figure from the top, each row the stripe, the dI/dV and
        current axes (sharing the stripe's y), their colour bars and γ
        boxes; the controls on the first figure; then every member drawn
        (refresh).  Row k is self.rows[k] (a dict of its axes, images and
        boxes).
        """
        self.close()
        self.rows = {}
        self._artists = {}
        p = self.parent
        n = len(p.members)
        n_figs = int(np.ceil(n / self.PER_FIGURE)) or 1
        for fi in range(n_figs):
            fig = plt.figure(figsize=self.DESIGN)
            try:
                fig.canvas.manager.set_window_title(f"{p.name}: alignment" + (f" ({fi + 1} of {n_figs})" if n_figs > 1 else ""))
            except Exception:
                pass
            self.figs.append(fig)
            plots.draw_topo_cbar(fig, self.TOPO_CBAR)
            for r in range(self.PER_FIGURE):
                k = fi * self.PER_FIGURE + r
                if k >= n:
                    break
                y = self.ROW_Y0 + (self.PER_FIGURE - 1 - r) * self.ROW_PITCH
                d = {'fig': fig}
                d['stripe'] = fig.add_axes([self.STRIPE_X, y, self.STRIPE_W, self.ROW_H])
                d['didv'] = fig.add_axes([self.DIDV_X, y, self.IMG_W, self.ROW_H], sharey=d['stripe'])
                d['didv_cbar'] = fig.add_axes([self.DIDV_X + self.IMG_W + self.GAP, y, self.CBAR_W, self.ROW_H])
                d['iv'] = fig.add_axes([self.IV_X, y, self.IMG_W, self.ROW_H], sharey=d['stripe'])
                d['iv_cbar'] = fig.add_axes([self.IV_X + self.IMG_W + self.GAP, y, self.CBAR_W, self.ROW_H])
                d['g_didv'] = widgets.GammaBox(fig)
                d['g_didv'].attach(d['didv'], gap_in=0.02)
                d['g_didv'].on_change(lambda value, k=k: self.draw_member(k))
                d['g_iv'] = widgets.GammaBox(fig)
                d['g_iv'].attach(d['iv'], gap_in=0.02)
                d['g_iv'].on_change(lambda value, k=k: self.draw_member(k))
                d['im_didv'] = None
                d['im_iv'] = None
                self.rows[k] = d
            fig.canvas.mpl_connect('pick_event', self._on_pick)
            fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
            fig.canvas.mpl_connect('button_release_event', self._on_release)
            if fi == 0:
                m0 = p.members[0].get('align_markers') or []
                self.s_markers = widgets.make(fig, 'Markers', 0.10, 0.015, value=len(m0))
                self.s_markers.on_change(self._on_markers)
                self.box_ref = widgets.make(fig, 'reference', 0.33, 0.012, value=str(p.reference + 1))
                self.box_ref.on_change(self._on_reference)
                self.buttons = widgets.RunControls(fig, [('apply', 0.45, 0.01), ('auto', 0.52, 0.01), ('close', 0.59, 0.01)])
                self.buttons.on('apply', self._on_apply)
                self.buttons.on('auto', self._on_auto)
                self.buttons.on('close', self.close)
        self.refresh()
        for f in self.figs:
            f.canvas.draw_idle()
        plt.show(block=False)

    def refresh(self):
        """Every member redrawn (the editor rebuilt when the member count changed); the 'reference' box and the 'Markers' slider follow the parent without firing."""
        p = self.parent
        if len(self.rows) != len(p.members):
            self.build()
            return
        for k in range(len(p.members)):
            self.draw_member(k)
        self._quiet = True
        try:
            self.box_ref.set(str(p.reference + 1))
            m0 = p.members[0].get('align_markers') or []
            if m0:
                self.s_markers.set(len(m0))
        finally:
            self._quiet = False
        for f in self.figs:
            f.canvas.draw_idle()

    def draw_member(self, k):
        """
        Member k's row: its dI/dV (division off, the parent's calibrate /
        normalize flags and filters) and its current (0Vsub and the
        filters as set, unit area with the parent's normalize flag) as
        heatmaps over [energy, own positions] (_int_orient_member, the
        parent's colour map, the row's γ boxes), each with its colour bar,
        or the note "no Current channel"; the title 'Line k: file:line
        <quantity>' with '[reference]' for the main; the member's stripe
        (topo.stripe_profile of its rows with the parent's processor) at
        the pixels' positions; then the marker lines (_draw_markers).
        """
        p = self.parent
        m = p.members[k]
        d = self.rows[k]
        hp, sp = p._int_hampel_params, p._int_sgolay_params
        pos = np.asarray(m['pos'], dtype=np.float64)
        energy = np.sort(np.asarray(m['selection'].energy, dtype=np.float64))
        extent = [float(energy[0]), float(energy[-1]), float(pos[0]), float(pos[-1])]
        for key, cbar_key, g_key, arr, label in (
                ('didv', 'didv_cbar', 'g_didv',
                 p._member_columns(m, False, p._int_apply_calibration, p._int_apply_normalize, hp, sp),
                 ("d2I/dV2 (IETS)" if p.channel == 'iets' else "dI/dV") + " (division off)"),
                ('iv', 'iv_cbar', 'g_iv', p._member_current(m, p._int_apply_normalize), "current")):
            ax = d[key]
            ax.clear()
            if arr is None:
                ax.text(0.5, 0.5, "no Current channel", ha='center', va='center', transform=ax.transAxes, fontsize=8)
                d['im_' + key] = None
            else:
                img = p._int_orient_member(arr, m)
                d['im_' + key] = plots.draw_heatmap(ax, img, extent, p.cmap, gamma=d[g_key].value)
                plots.draw_colorbar(d['fig'], d['im_' + key], d[cbar_key], labelsize=6)
            ax.tick_params(labelsize=6)
            ax.tick_params(labelleft=False)
            ax.set_title(f"Line {k + 1}: {p.member_label(k)}  {label}" + ("  [reference]" if k == p.reference else ""),
                         fontsize=7)
        prof = topo.stripe_profile(m['topo_x'], m['topo_profiles'], p._int_topo_active)
        tx = np.asarray(m['topo_x'], dtype=np.float64)
        plots.draw_topo_stripe(d['stripe'], tx if len(tx) == len(prof) else pos, prof)
        d['stripe'].set_title('')
        d['stripe'].tick_params(labelsize=6)
        d['stripe'].set_ylabel("Position (Å)", fontsize=7)
        self._draw_markers(k)

    def _draw_markers(self, k):
        """Member k's marker lines redrawn: a dashed, pickable horizontal line per editor marker (colour j) on its dI/dV and current axes."""
        p = self.parent
        m = p.members[k]
        d = self.rows[k]
        for art in self._artists.get(k, []):
            try:
                art.remove()
            except Exception:
                pass
        arts = []
        for j, pos_val in enumerate(m.get('align_markers') or []):
            color = self.M_COLORS[j % len(self.M_COLORS)]
            for ax in (d['didv'], d['iv']):
                h = ax.axhline(y=pos_val, color=color, ls='--', lw=1.8, alpha=0.8, label=f'marker_{k}_{j}', picker=5)
                arts.append(h)
        self._artists[k] = arts
        d['fig'].canvas.draw_idle()

    # --- events ---

    def _on_pick(self, event):
        """A press on a marker line ('marker_<member>_<j>'): it becomes the dragged marker."""
        label = getattr(event.artist, 'get_label', lambda: '')()
        if isinstance(label, str) and label.startswith('marker_'):
            _, k, j = label.split('_')
            self._active = (int(k), int(j))

    def _on_motion(self, event):
        """Drag: the marker follows the cursor's position on its member's heatmaps, clipped to the member's line; the member's lines redrawn (no run)."""
        if self._active is None or event.ydata is None:
            return
        k, j = self._active
        d = self.rows.get(k)
        if d is None or event.inaxes not in (d['didv'], d['iv']):
            return
        p = self.parent
        m = p.members[k]
        pos = np.asarray(m['pos'], dtype=np.float64)
        markers = list(m['align_markers'])
        markers[j] = float(np.clip(event.ydata, min(pos[0], pos[-1]), max(pos[0], pos[-1])))
        m['align_markers'] = markers
        self._draw_markers(k)

    def _on_release(self, event):
        """End of a drag."""
        self._active = None

    def _on_markers(self, val):
        """The 'Markers' slider: the count set in every member (the parent) and the lines redrawn; ignored while refresh sets the slider."""
        if self._quiet:
            return
        self.parent.set_align_marker_count(int(val))
        for k in range(len(self.parent.members)):
            self._draw_markers(k)

    def _on_reference(self, text):
        """The 'reference' box: member number (from 1) to the parent (set_reference); a bad one is reported and the box put back."""
        if self._quiet:
            return
        try:
            self.parent.set_reference(int(float(text)) - 1)
        except ValueError as exc:
            print(f"reference: {exc}")
            self.box_ref.set(str(self.parent.reference + 1))

    def _on_apply(self):
        """'apply': the parent runs with the marker-section maps; the editor refreshed on success."""
        if self.parent.apply_markers():
            self.refresh()

    def _on_auto(self):
        """'auto': the parent runs with the DTW maps; the editor refreshed on success."""
        if self.parent.auto_markers():
            self.refresh()


if __name__ == '__main__':
    # Two lines of one grid file into a line average (run from the registry, as the Line window does it).
    from sm4_Browser import sm4file
    path = r'experiment_dir/file.sm4'
    f = sm4file.Sm4File(path)
    registry = LineAverages(os.path.dirname(path), {'window': 4, 'n_sigma': 3}, {'window_length': 6, 'polyorder': 3})
    first = member_from_file(f, 1, 2.75)
    w = registry.new(first)
    w.add(member_from_file(f, 2, 2.75))
    w.run()
