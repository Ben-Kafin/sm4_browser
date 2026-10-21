"""
The saved records of an experiment directory, all in its 'averages'
subfolder (AVG_SUBDIR), which also holds the fit histories
(<stem>_fits.pkl and its numbered family) and the saved fits:
  - the averaging windows' records <stem>_average.json (RECORD_FORMAT 5:
    the index, name and send label, the files or selections in added
    order, the removed sweeps, the channel, the filters, the resampling,
    the Feenstra state and settings, the calibrate / normalize flags, the
    last run's range and grid, the time of saving; averaging.
    MultiAverageWindow.record);
  - the Line window's marker records <file stem>_line<k>_markers.json
    (LINE_RECORD_FORMAT 1: the file and line, the channel and sweep view,
    the filters with their on / off states, the calibrate / normalize
    flags, the Feenstra state and settings, the gamma, the energy and
    position ranges, the topo rows, the marker groups, markers and
    E-markers, the Average / stack / substack states, the scale factor the
    window was made with ('scalefactor'), the time of saving;
    line.Line.marker_record);
  - the line-average records lineavg<N>_average.json (LINEAVG_RECORD_FORMAT
    1: the members, each with its line, markers and the scale factor it
    was made with, and the alignment; lineavg.py); a record is a line
    average by its 'members' key, not by its name (is_lineavg_record).
A record that carries a scale factor keeps the one it was made with; on
reload the browser applies the scale timeline's and reports a difference
(browser._record_scale).  Here live the paths, the name-to-stem rule
(stem_of), the atomic JSON writer (write_json), the fit-history reader
(read_cache), the rename of a record with its fit history (rename_record),
and the saved-records window (SavedAveragesWindow) that lists all three
kinds and hands the chosen entry to the browser (Browser.load_saved_average).
"""

if __package__ in (None, ''):
    # Run as a file (Spyder F5): make the package importable from wherever its folder sits.
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json
import os
import pickle
import re
from datetime import datetime
import numpy as np
from sm4_Browser import qtplot as plt
from sm4_Browser import sm4file, widgets

AVG_SUBDIR = 'averages'    # subfolder of the experiment directory holding records, fit histories, saved fits
RECORD_FORMAT = 5          # 1: removed sweeps indexed among all of a file's sweeps (the old browser); 2: among the
#                            acquired ones; 3: name and entry kinds; 4: the Feenstra settings; 5: calibrate / normalize
LINE_RECORD_FORMAT = 1     # the Line window's marker records
LINEAVG_RECORD_FORMAT = 1  # the line-average records

RECORD_RE = re.compile(r'^(.+)_average\.json$')          # group 1: the stem (the name cleaned as a file name)
LINE_RECORD_RE = re.compile(r'^(.+)_line(\d+)_markers\.json$')
LINEAVG_RECORD_RE = re.compile(r'^(lineavg\d+.*)_average\.json$')   # the default names; a record's 'members' key decides
OLD_CACHE_RE = re.compile(r'^(.*_average_of_.*)_fits(?:_\d+)?\.pkl$')   # group 1: the family's stem (the numbered members too)


# --- paths and names ---

def avg_dir_of(directory):
    """The 'averages' subfolder of an experiment directory (may not exist yet)."""
    return os.path.join(os.path.abspath(directory), AVG_SUBDIR)


def stem_of(name):
    """
    The file stem of a name: the cleaning rule of Fit._stem_of (letters,
    digits, '.', '_', '-' kept, any other run replaced by '_', outer
    underscores stripped), which is the stem the fit history gets for the
    label '<averages folder> | <name>'; '' for a name with nothing left.
    """
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(name)).strip('_')


def record_path(directory, stem):
    """An averaging window's record (also a line-average window's: their stems start with 'lineavg')."""
    return os.path.join(avg_dir_of(directory), f"{stem}_average.json")


def cache_path_of(directory, stem):
    """
    The fit history path, as Fit.cache_path gives it for the label once the
    subfolder exists: the newest up-to-date file of <stem>_fits.pkl's family
    in the 'averages' subfolder, or the path a new one is written to
    (Fit.cache_file).
    """
    from sm4_Browser import fit
    return fit.Fit.cache_file(avg_dir_of(directory), stem)


def cache_files_of(directory, stem):
    """Every file of the fit history family of a stem (sm4file.cache_family), [(n, path)] in ascending number."""
    return sm4file.cache_family(os.path.join(avg_dir_of(directory), f"{stem}_fits.pkl"))


def line_record_path(directory, file_path, line):
    """The Line window's marker record of one file and line: <file stem>_line<k>_markers.json."""
    stem = os.path.splitext(os.path.basename(file_path))[0]
    return os.path.join(avg_dir_of(directory), f"{stem}_line{int(line)}_markers.json")


# --- reading and writing ---

def write_json(path, obj):
    """Write a JSON record (to a temporary file, then replaced in place; the subfolder created)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)
    return path


def read_json(path):
    """A JSON record read whole (the exceptions of open / json.load propagate)."""
    with open(path) as fh:
        return json.load(fh)


def read_cache(path, up_to_date=True):
    """
    The plain state dict of a fit history (the pickle Fit._write_cache
    writes), or None when the file cannot be unpickled or is not a dict of
    a readable format (Fit._CACHE_FORMATS_READ: 2 or 3).  With up_to_date
    (the default) only a history written from this loader is returned
    (Fit.cache_up_to_date: its 'loader_format' is sm4file.LOADER_FORMAT);
    with up_to_date False any readable one, for its label, not its data.
    """
    from sm4_Browser import fit
    try:
        with open(path, 'rb') as fh:
            state = pickle.load(fh)
    except Exception:
        return None
    if not isinstance(state, dict) or state.get('format') not in fit.Fit._CACHE_FORMATS_READ:
        return None
    if up_to_date and not fit.Fit.cache_up_to_date(state):
        return None
    return state


def saved_indices_of(directory):
    """The 'index' of every averaging record *_average.json in the directory's 'averages' subfolder."""
    out = []
    avg_dir = avg_dir_of(directory)
    if os.path.isdir(avg_dir):
        for name in os.listdir(avg_dir):
            if not RECORD_RE.match(name):
                continue
            try:
                rec = read_json(os.path.join(avg_dir, name))
                if is_lineavg_record(rec):
                    continue
                out.append(int(rec['index']))
            except Exception:
                continue
    return out


def saved_lineavg_indices_of(directory):
    """The 'index' of every line-average record lineavg<N>*_average.json of the directory."""
    out = []
    avg_dir = avg_dir_of(directory)
    if os.path.isdir(avg_dir):
        for name in os.listdir(avg_dir):
            if not RECORD_RE.match(name):
                continue
            try:
                rec = read_json(os.path.join(avg_dir, name))
                if not is_lineavg_record(rec):
                    continue
                out.append(int(rec['index']))
            except Exception:
                continue
    return out


def is_lineavg_record(rec):
    """A line-average record holds 'members' (an averaging record holds 'files'); the file name alone does not say."""
    return isinstance(rec, dict) and 'members' in rec


def rename_record(directory, old_stem, new_stem, name, label, tag=''):
    """
    Rename a record (and every file of the old stem's fit history family,
    each keeping its number: <stem>_fits.pkl, <stem>_fits_2.pkl, ...) to a
    new stem, rewriting the record's 'name' and 'label' and each fit
    history's stored label (the pickle format kept).  Returns the list of
    the renamed file names; failures are reported on the console.
    """
    renamed = []
    old_rec = record_path(directory, old_stem)
    new_rec = record_path(directory, new_stem)
    if os.path.isfile(old_rec):
        try:
            rec = read_json(old_rec)
            rec['name'] = name
            rec['label'] = label
            write_json(new_rec, rec)
            os.remove(old_rec)
            renamed.append(os.path.basename(new_rec))
        except Exception as exc:
            print(f"{tag}: record not renamed ({exc}).")
    for n, old_cache in cache_files_of(directory, old_stem):
        new_cache = os.path.join(avg_dir_of(directory), f"{new_stem}_fits.pkl" if n == 1 else f"{new_stem}_fits_{n}.pkl")
        try:
            with open(old_cache, 'rb') as fh:
                state = pickle.load(fh)
            if isinstance(state, dict) and 'ifile' in state:
                state['ifile'] = label
            tmp = new_cache + '.tmp'
            with open(tmp, 'wb') as fh:
                pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)   # as Fit._write_cache writes it
            os.replace(tmp, new_cache)
            os.remove(old_cache)
            renamed.append(os.path.basename(new_cache))
        except Exception as exc:
            print(f"{tag}: fit history not renamed ({exc}).")
    return renamed


# --- the saved-records window ---

class SavedAveragesWindow:
    """
    The saved records of the experiment directory, one row each, in four
    sections: the averaging records <stem>_average.json of its 'averages'
    subfolder by the index stored in the record, descending (name, entries
    and sweeps, removed sweeps, the files' times, the filter settings, the
    number of fits in the newest up-to-date file of <stem>_fits.pkl's family
    when there is one, 'fit history out of date' when files of the family
    exist but none was written from this loader, 'no fit history' without
    any, the time of saving); then every fit history of the earlier
    label-named form (<stamp>_average_of_..._fits.pkl in the directory
    itself) by modification time, newest first; the Line window's marker
    records <file stem>_line<k>_markers.json (file, line, groups and
    markers, channel, Feenstra, the time of saving) by modification time;
    and the line-average records lineavg<N>_average.json (name, the first
    four members, channel, Feenstra, the time of saving) by index,
    descending.  A record that cannot be read is reported on the console
    and left out.  A click on a row selects it (highlighted); 'load' hands
    the selected entry to the browser (on_load(entry),
    Browser.load_saved_average); 'rescan' re-reads the directory.  The
    window is a figure of size DESIGN with the rows drawn as text in one
    axes (ROW_H of the figure per row, at most 0.78 in all).

    Entry dicts: kind ('record', 'old', 'line', 'lineavg'), text, and for a
    record 'record' (the JSON dict), 'index', 'name', 'stem', 'cache' (the
    path of the up-to-date fit history, or None), 'n_fits'; an old entry
    carries 'label' and its parsed form 'parsed' (see parse_label),
    'cache', 'n_fits', 'hampel', 'sgolay', 'mtime'; a line entry 'record',
    'path' and 'mtime'; a lineavg entry 'record', 'index', 'name', 'stem'.
    """

    LABEL_RE = re.compile(r'^(?P<path>.*?) \| average of (?P<n>\d+) sweeps from (?P<k>\d+) files '
                          r'(?P<times>.*?)(?: minus (?P<minus>.*))?$')
    TIME_RE = re.compile(r'^(\d{2}):(\d{2}):(\d{2})\.(\d{3})$')
    ROW_H = 0.045          # figure fraction per row
    MAX_TIMES = 8          # file times shown per row before '+k more'

    DESIGN = (16, 6)       # the design size (inches): the text zoom of the window is its size relative to this (qtplot.window_zoom)

    def __init__(self, directory, on_load):
        self.directory = os.path.abspath(directory)
        self.avg_dir = avg_dir_of(self.directory)
        self.on_load = on_load
        self.entries = []
        self.selected = 0
        self.ax_list = None
        self._highlight = None
        self.fig = plt.figure(figsize=self.DESIGN)
        try:
            self.fig.canvas.manager.set_window_title(f"saved averages: {self.directory}")
        except Exception:
            pass
        self.txt = self.fig.text(0.03, 0.96, '', fontsize=9, ha='left', va='top')
        self.buttons = widgets.RunControls(self.fig, [('load', 0.03, 0.03), ('rescan', 0.13, 0.03)])
        self.buttons.on('load', self._on_load)
        self.buttons.on('rescan', self.scan)
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.scan()
        plt.show()

    def is_open(self):
        """True while the window's figure is open (widgets.figure_is_open: the Figure object itself, not its number)."""
        return widgets.figure_is_open(self.fig)

    # --- reading ---

    @classmethod
    def parse_label(cls, label):
        """
        The parts of an old send label '<first file path> | average of N sweeps
        from K files HH:MM:SS.mmm, ... [minus i:j ...]': dict(path, n, k, times
        [(h, m, s, ms) per file], removed [(entry, sweep), 0-based, as sorted in
        the label]), or None when the label has another form.
        """
        m = cls.LABEL_RE.match(str(label).strip())
        if m is None:
            return None
        times = []
        for t in m.group('times').split(', '):
            tm = cls.TIME_RE.match(t.strip())
            if tm is None:
                return None
            times.append(tuple(int(g) for g in tm.groups()))
        removed = []
        if m.group('minus'):
            for pair in m.group('minus').split():
                i, _, j = pair.partition(':')
                if not (i.isdigit() and j.isdigit()):
                    return None
                removed.append((int(i) - 1, int(j) - 1))
        return {'path': m.group('path').strip(), 'n': int(m.group('n')), 'k': int(m.group('k')),
                'times': times, 'removed': removed}

    @staticmethod
    def _filters_text(hp, sp, resample=None, channel='didv'):
        """'IETS, ' when the channel is IETS, then 'Hampel window/n_sigma' or 'Hampel off', 'SG window/order' or 'SG off', and the resampling when set."""
        h = f"Hampel {hp['window']}/{hp['n_sigma']}" if hp else "Hampel off"
        s = f"SG {sp['window_length']}/{sp['polyorder']}" if sp else "SG off"
        c = "IETS, " if channel == 'iets' else ""
        return c + h + ", " + s + (f", {resample}" if resample else "")

    @classmethod
    def _times_text(cls, times):
        """The files' times of day 'HH:MM:SS.mmm', the first MAX_TIMES then '+k more'."""
        shown = [f"{h:02d}:{mi:02d}:{s:02d}.{ms:03d}" for (h, mi, s, ms) in times[:cls.MAX_TIMES]]
        more = len(times) - len(shown)
        return ", ".join(shown) + (f" +{more} more" if more > 0 else "")

    def _record_entry(self, name, stem):
        """
        The entry of an averaging record (file `name`, its stem): the record
        read, the files' times from their name stamps ((0, 0, 0, 0) for a
        name without one), the sweeps and marker-group selections counted,
        the up-to-date fit history looked up (cache_path_of, read_cache);
        None (a console line) when the record cannot be read or has no index.
        """
        path = os.path.join(self.avg_dir, name)
        try:
            rec = read_json(path)
            index = int(rec['index'])
        except Exception as exc:
            print(f"saved averages: {name} not read ({exc}).")
            return None
        avg_name = str(rec.get('name') or f"avg{index}")
        files = rec.get('files', [])
        times = []
        for fi in files:
            try:
                times.append(sm4file.filename_stamp(fi.get('name', ''))[3:])
            except ValueError:
                times.append((0, 0, 0, 0))
        n_sweeps = sum(int(fi.get('n_sweeps', 0)) for fi in files)
        n_sel = sum(1 for fi in files if fi.get('kind') == 'selection')
        cache = cache_path_of(self.directory, stem)
        state = read_cache(cache) if os.path.isfile(cache) else None
        n_fits = len(state.get('fits', [])) if state is not None else None
        stale = state is None and bool(cache_files_of(self.directory, stem))   # only out-of-date histories
        what = (f"{len(files)} entries ({len(files) - n_sel} files, {n_sel} marker groups)" if n_sel
                else f"{len(files)} files")
        text = (f"{avg_name} | {what}, {n_sweeps} sweeps, {len(rec.get('removed', []))} removed | "
                f"{self._times_text(times)} | "
                f"{self._filters_text(rec.get('hampel'), rec.get('sgolay'), rec.get('resample'), rec.get('channel', 'didv'))} | "
                + (f"{n_fits} fits" if n_fits is not None
                   else "fit history out of date" if stale else "no fit history")
                + f" | saved {rec.get('saved', '?')}")
        return {'kind': 'record', 'index': index, 'name': avg_name, 'stem': stem, 'record': rec,
                'cache': cache if state is not None else None, 'n_fits': n_fits, 'text': text}

    @staticmethod
    def old_family_stems(names):
        """The stems of the old label-named fit-history families among `names` (OLD_CACHE_RE: <stem>_fits.pkl and its numbered members <stem>_fits_<n>.pkl), each family once, in name order."""
        out = []
        for name in sorted(names):
            m = OLD_CACHE_RE.search(name)
            if m and m.group(1) not in out:
                out.append(m.group(1))
        return out

    def _old_entry(self, stem):
        """
        The entry of a fit-history family of the earlier label-named form in
        the directory itself (OLD_CACHE_RE; `stem` from old_family_stems):
        its newest member (sm4file.cache_family) read whatever its loader
        format (listed by its label, which parse_label must accept, else
        None); the text carries the files, sweeps and removed sweeps of the
        label, its filter settings and fit count, and the file's
        modification time.
        """
        family = sm4file.cache_family(os.path.join(self.directory, f"{stem}_fits.pkl"))
        if not family:
            return None
        cache = family[-1][1]
        state = read_cache(cache, up_to_date=False)      # listed by its label; every such history predates the loader format
        if state is None:
            return None
        parsed = self.parse_label(state.get('ifile'))
        if parsed is None:
            return None
        n_fits = len(state.get('fits', []))
        hp, sp = state.get('hampel_filter_params'), state.get('sgolay_filter_params')
        when = datetime.fromtimestamp(os.path.getmtime(cache)).isoformat(timespec='seconds')
        text = (f"old {stem} | {parsed['k']} files, {parsed['n']} sweeps, "
                f"{len(parsed['removed'])} removed | {self._times_text(parsed['times'])} | "
                f"{self._filters_text(hp, sp)} | {n_fits} fits | written {when}")
        return {'kind': 'old', 'label': state.get('ifile'), 'parsed': parsed, 'cache': cache,
                'n_fits': n_fits, 'hampel': hp, 'sgolay': sp, 'text': text, 'mtime': os.path.getmtime(cache)}

    def _line_entry(self, name):
        """The entry of a Line window's marker record (file, line, groups and markers, channel, Feenstra, time of saving); None when unreadable."""
        path = os.path.join(self.avg_dir, name)
        try:
            rec = read_json(path)
        except Exception as exc:
            print(f"saved averages: {name} not read ({exc}).")
            return None
        groups = rec.get('group_sizes', [])
        text = (f"line markers | {rec.get('name', '?')} line {rec.get('line', '?')} | {len(groups)} group(s), "
                f"{sum(int(g) for g in groups)} markers | {rec.get('channel', 'didv')}"
                + (", Feenstra" if rec.get('divide_iv') else "")
                + f" | saved {rec.get('saved', '?')}")
        return {'kind': 'line', 'record': rec, 'path': path, 'text': text,
                'mtime': os.path.getmtime(path)}

    def _lineavg_entry(self, name, stem):
        """The entry of a line-average record (name, the first four members as name:line, channel, Feenstra, time of saving); None when unreadable or without an index."""
        path = os.path.join(self.avg_dir, name)
        try:
            rec = read_json(path)
            index = int(rec['index'])
        except Exception as exc:
            print(f"saved averages: {name} not read ({exc}).")
            return None
        members = rec.get('members', [])
        text = (f"line average {rec.get('name', stem)} | {len(members)} lines: "
                + ", ".join(f"{m.get('name', '?')}:{m.get('line', '?')}" for m in members[:4])
                + (f" +{len(members) - 4} more" if len(members) > 4 else "")
                + f" | {rec.get('channel', 'didv')}" + (", Feenstra" if rec.get('divide_iv') else "")
                + f" | saved {rec.get('saved', '?')}")
        return {'kind': 'lineavg', 'record': rec, 'index': index, 'name': str(rec.get('name') or stem),
                'stem': stem, 'text': text}

    def scan(self):
        """
        Re-read the directory and redraw the list: the 'averages' subfolder
        for the three record kinds (a *_average.json is a line average when
        its content has 'members', else by its name when unreadable), the
        directory itself for the old label-named fit histories; the sections
        sorted as the class docstring says; the selection back to the first
        row.
        """
        records, old, lines, lineavgs = [], [], [], []
        if os.path.isdir(self.avg_dir):
            for name in sorted(os.listdir(self.avg_dir)):
                if LINE_RECORD_RE.match(name):
                    e = self._line_entry(name)
                    if e is not None:
                        lines.append(e)
                    continue
                m = RECORD_RE.match(name)
                if m:
                    try:
                        lineavg = is_lineavg_record(read_json(os.path.join(self.avg_dir, name)))
                    except Exception:
                        lineavg = bool(LINEAVG_RECORD_RE.match(name))
                    e = self._lineavg_entry(name, m.group(1)) if lineavg else self._record_entry(name, m.group(1))
                    if e is not None:
                        (lineavgs if lineavg else records).append(e)
        for stem in self.old_family_stems(os.listdir(self.directory)):
            e = self._old_entry(stem)
            if e is not None:
                old.append(e)
        records.sort(key=lambda e: -e['index'])
        old.sort(key=lambda e: -e['mtime'])
        lines.sort(key=lambda e: -e['mtime'])
        lineavgs.sort(key=lambda e: -e['index'])
        self.entries = records + old + lines + lineavgs
        self.selected = 0
        self._draw_list()

    # --- figure ---

    def _draw_list(self):
        """Rebuild the list axes: the header text, one text row per entry (row 0 at the top), the separators, the highlight."""
        if self.ax_list is not None:
            self.ax_list.remove()
            self.ax_list = None
            self._highlight = None
        n = len(self.entries)
        self.txt.set_text(f"{n} saved record(s) in {self.avg_dir}   "
                          f"(averages by index, old label-named fit histories of the directory itself, "
                          f"line markers, line averages; click a row, then 'load')")
        height = min(0.78, self.ROW_H * n) if n else 0.78
        ax = self.fig.add_axes([0.03, 0.90 - height, 0.94, height])
        ax.set_navigate(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.patch.set_facecolor('0.97')
        self.ax_list = ax
        if n == 0:
            ax.text(0.5, 0.5, "no saved records ('save' in an average, Line, or line-average window)",
                    ha='center', va='center', transform=ax.transAxes)
            self.fig.canvas.draw_idle()
            return
        ax.set_xlim(0, 1)
        ax.set_ylim(0, n)
        for r, e in enumerate(self.entries):                 # row 0 at the top
            ax.text(0.006, n - r - 0.5, e['text'], fontsize=8, ha='left', va='center')
            if r:
                ax.axhline(n - r, color='0.85', lw=0.5)
        self._draw_highlight()
        self.fig.canvas.draw_idle()

    def _draw_highlight(self):
        """The light blue band behind the selected row (the previous band removed)."""
        if self._highlight is not None:
            try:
                self._highlight.remove()
            except Exception:
                pass
            self._highlight = None
        n = len(self.entries)
        if n:
            self._highlight = self.ax_list.axhspan(n - self.selected - 1, n - self.selected,
                                                   color='#cfe2f3', zorder=0)

    def _on_press(self, event):
        """A left click on a row of the list selects it (the row under the click's y, clipped to the list)."""
        if (event.inaxes is not self.ax_list or not self.entries or event.button != 1
                or event.ydata is None):
            return
        n = len(self.entries)
        self.selected = int(np.clip(np.floor(n - event.ydata), 0, n - 1))
        self._draw_highlight()
        self.fig.canvas.draw_idle()

    def _on_load(self):
        """'load': the selected entry dict to the browser's on_load (a console line when the list is empty)."""
        if not self.entries:
            print("saved averages: nothing to load.")
            return
        self.on_load(self.entries[self.selected])


if __name__ == '__main__':
    # The saved records of one experiment directory listed on the console (no window).
    directory = r'experiment_dir'
    avg_dir = avg_dir_of(directory)
    if os.path.isdir(avg_dir):
        for name in sorted(os.listdir(avg_dir)):
            if name.endswith('.json'):
                rec = read_json(os.path.join(avg_dir, name))
                print(name, 'format', rec.get('format'), 'name', rec.get('name'), 'saved', rec.get('saved'))
    else:
        print(f"no 'averages' subfolder in {directory}")
