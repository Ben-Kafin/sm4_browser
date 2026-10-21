# -*- coding: utf-8 -*-
"""
The scale timeline of an experiment: the x/y scale factors of the piezo,
one per temperature regime, each anchored at the time of the file it was
set on, so that every file of the experiment is drawn and measured at the
scale in force at its own time.

Nominal and calibrated.  A file records positions and sizes in metres as
the piezo calibration of the recording software had them: the nominal
values.  The true distances are the nominal ones divided by the scale
factor in force when the file was taken: the calibrated values, the units
of the scan area and of every window (nm after the scale factor).  The
scale factor changes with the temperature, so one experiment may need
several: the timeline.

Rules (set with the owner; PLAN.md item 2):
  * An entry is anchored at the time of the file it was set on (the file's
    date-and-time stamp, file_time).  A file takes the latest entry at or
    before its own time.
  * The first entry set covers the whole experiment: a file earlier than
    every entry takes the earliest one.  Without any entry the base scale
    applies (the browser's constructor argument, default 2.75).
  * Setting the scale anywhere (the browser's 'scale factor' box, the
    section features' 'set scale' window) adds or replaces the entry at the
    current file's time: the files from that time on, up to the next entry,
    follow; the files before keep theirs.
  * The timeline is the 'scales' section of the settings store
    (caches.Settings, settings_<stamp>.json), written on every change and
    never discarded; an entry naming a file no longer in the directory is
    kept and reported.
  * Nothing computed depends on the scale: it is applied at drawing and
    readout time (the browser's scale_of / k_of, the windows' scalefactor
    argument, drift.PairDrift.text's calibrated form).  Records saved in
    the 'averages' folder keep the scale they were made with; on reload
    the timeline's scale applies, with a console note when the two differ.

Scales(base, settings, files): at(path or time) gives a file's scale,
k(path or time) its drawing factor 1e9 / scale (recorded metres to
calibrated nm), set_from(path, scale) sets the timeline from that file's
time on, remove(path) drops the entry at that file's time, describe()
prints the timeline.
"""

import os

from sm4_Browser import sm4file


SCALES_FORMAT = 1


def file_time(path):
    """The time of a file, seconds since the epoch, from the date-and-time stamp of its name; None without a stamp."""
    try:
        return sm4file.stamp_datetime(sm4file.filename_stamp(path)).timestamp()
    except (ValueError, TypeError):
        return None


class Scales:
    """The timeline (the module docstring)."""

    def __init__(self, base, settings=None, files=()):
        self.base = float(base)
        self.settings = settings
        self.entries = []                 # [{'file': name, 'time': seconds, 'scale': factor}] in time order
        sec = (settings.get('scales') if settings is not None else None) or {}
        for e in sec.get('entries', []):
            try:
                t, s = float(e['time']), float(e['scale'])
            except (KeyError, TypeError, ValueError):
                print(f"scale timeline: an entry without a time or a scale is left out: {e}")
                continue
            if s <= 0:
                print(f"scale timeline: an entry with a scale of {s} is left out (it must be positive).")
                continue
            self.entries.append({'file': str(e.get('file', '')), 'time': t, 'scale': s})
        self._sort()
        names = {os.path.basename(p) for p in files}
        for e in self.entries:
            if e['file'] and e['file'] not in names:
                print(f"scale timeline: the entry of {e['file']} ({e['scale']:g}) belongs to no file in the directory; kept.")

    def _sort(self):
        self.entries.sort(key=lambda e: e['time'])

    @staticmethod
    def _time(path_or_time):
        if isinstance(path_or_time, str):
            return file_time(path_or_time)
        return None if path_or_time is None else float(path_or_time)

    def at(self, path_or_time):
        """The scale factor in force for a file (its path) or a time: the latest entry at or before it, the earliest when it is before every entry, the base without entries."""
        if not self.entries:
            return self.base
        t = self._time(path_or_time)
        if t is None:
            return self.entries[0]['scale']
        current = None
        for e in self.entries:
            if e['time'] <= t:
                current = e
            else:
                break
        return (current or self.entries[0])['scale']

    def k(self, path_or_time):
        """The drawing factor of a file or a time: recorded metres times k are calibrated nanometres."""
        return 1e9 / self.at(path_or_time)

    def entry_of(self, path):
        """The entry anchored at this file's time, or None."""
        t = file_time(path)
        return next((e for e in self.entries if t is not None and abs(e['time'] - t) < 1e-6), None)

    def set_from(self, path, scale):
        """The timeline from this file's time on: an entry at its time (replacing one there) with `scale` (> 0); written at once."""
        scale = float(scale)
        if scale <= 0:
            raise ValueError("the scale factor must be > 0")
        t = file_time(path)
        if t is None:
            raise ValueError(f"{os.path.basename(path)} has no date-and-time stamp to anchor the scale at")
        e = self.entry_of(path)
        if e is None:
            e = {'file': os.path.basename(path), 'time': t, 'scale': scale}
            self.entries.append(e)
            self._sort()
        else:
            e['file'], e['scale'] = os.path.basename(path), scale
        self._save()
        return e

    def remove(self, path):
        """The entry at this file's time dropped (nothing without one); written at once."""
        e = self.entry_of(path)
        if e is not None:
            self.entries.remove(e)
            self._save()
        return e

    def record(self):
        return {'entries': [{'file': e['file'], 'time': float(e['time']), 'scale': float(e['scale'])} for e in self.entries]}

    def _save(self):
        if self.settings is not None:
            self.settings.put('scales', self.record())

    def describe(self):
        """The timeline on the console."""
        if not self.entries:
            print(f"scale timeline: no entry; the base scale factor {self.base:g} applies to every file.")
            return
        print(f"scale timeline ({len(self.entries)} entries; the first covers the files before it):")
        for e in self.entries:
            print(f"  from {e['file'] or 'time ' + str(e['time'])}: scale factor {e['scale']:g}")
