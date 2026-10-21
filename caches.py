# -*- coding: utf-8 -*-
"""
The two on-disk stores of an experiment directory, split by origin, beside
the .sm4 files:

  settings_<stamp>.json : everything set by hand and never recomputed.  The
      manual drift spans with their markers, kinds and velocities and the
      stripe fast-axis states and points (section 'drift', written by
      drift.Drift), the scale timeline (section 'scales', scales.Scales),
      the site flags set in the browser's 'site' radio (section 'flags',
      sites.Sites).  Written at once on every change (Settings.put).  Never
      invalidated by a format change: a newer package reads an older file
      and carries everything over.
  computed_<stamp>.json : everything computed from the .sm4 files alone and
      reused at the next startup.  The header scan (section 'headers',
      sm4file.header_cache), the once-over metrics, positions and ohmic
      tests (section 'metrics', sites.Sites), the tracked drift pairs and
      the topo times (section 'pairs', drift.Drift).  Every section carries
      its own format number and the loader format it was computed with
      (sm4file.LOADER_FORMAT): a section of another format or loader is
      ignored and rebuilt by its owner, the other sections stay.  Every
      entry is keyed by the file's name, with the file's size and
      modification time in the header scan and its modification time in
      the metrics, so a new or changed file is recomputed alone (a tracked
      pair is keyed by its two topos' names only).  Written once at the end
      of each computation stage (Computed.write), never while stepping.  Deleting the file loses
      nothing: it is rebuilt at the next startup.
  scan_cache_<stamp>/   : the processed images, one array file each
      (scanlayer.py), keyed by the file's name, size and time, the pass, the
      processors and the loader format; it stays a folder because images do
      not belong in a JSON file.

<stamp> is the date and time of the directory's first file
(YYYY_MM_DD_HH_MM_SS, stamp_of), as the earlier caches used it; the scan
cache's folder (scanlayer.cache_dir) also carries the millisecond field of
that file's name stamp.

The rules the stores enforce (PLAN.md, item 2):
  1. Storage: the computed file holds only what is computed from the files
     alone; a setting never enters a stored computed value.  A cached result
     that depends on a setting carries the setting in its key (the image
     cache carries the processor toggles), so a changed setting selects
     another entry and never overwrites one.
  2. Use: a setting is applied at the moment a value is used, on top of the
     computed value, through one function per kind: a file's velocity
     (drift.Drift.result: the narrowest manual span covering it, else the
     tracker's), a file's flag (sites.Sites.flag: the hand-set value feeding
     the region and clean chain), a file's scale (scales.Scales.at).
     Everything derived from such a value lives in memory and is rebuilt
     whenever the setting changes.
  3. Startup: the computed store is loaded or rebuilt first, then the
     settings are applied on top.  A setting that names a file no longer in
     the directory is kept in the file and reported, never dropped.
  4. Records saved in the 'averages' folder keep the scale they were made
     with; the timeline's scale applies when they are reloaded, with a
     console note when the two differ.

Carry-over (open_stores): when a store's file is absent and the earlier
caches exist (drift_cache_<stamp>[_n].json, sites_<stamp>[_n].json,
headers_<stamp>.json, the newest readable of each family), their contents
are read once into the absent store, which is written, and the old files
are renamed with a '.old' suffix so that nothing reads them again.  The
hand-set parts (the manual section, the flags) go to the settings whatever
the old file's loader format; the computed parts keep the old file's format
numbers, so a section from another loader is rebuilt.

Every write goes to a temporary file in the directory, then os.replace,
so a program that dies mid-write leaves the previous file intact.
"""

import json
import os
from datetime import datetime

from sm4_Browser import sm4file


SETTINGS_FORMAT = 1       # the settings file's own format (migrated, never discarded)
COMPUTED_FORMAT = 1       # the computed file's envelope; every section carries its own number as well


def stamp_of(files):
    """The experiment's stamp, 'YYYY_MM_DD_HH_MM_SS' of the first file (the earlier caches' stamp), or 'empty'."""
    if not files:
        return 'empty'
    y, mo, d, h, mi, sec, _ = sm4file.filename_stamp(files[0])
    return f"{y:04d}_{mo:02d}_{d:02d}_{h:02d}_{mi:02d}_{sec:02d}"


def _read_json(path):
    """A JSON file as a dict, or None when it is missing or unreadable (the reason printed)."""
    try:
        with open(path, encoding='utf-8') as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        print(f"{os.path.basename(path)}: not readable ({exc}).")
        return None
    return data if isinstance(data, dict) else None


def _write_json(path, data):
    """`data` written to `path` through a temporary file and os.replace; False (and a console line) when it fails."""
    tmp = path + '.tmp'
    try:
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, indent=1)
        os.replace(tmp, path)
    except OSError as exc:
        print(f"{os.path.basename(path)}: not written ({exc}).")
        return False
    return True


class _Store:
    """One JSON file beside the .sm4 files: its path, its sections (self.data), read and write."""

    KIND = ''
    FORMAT = 1

    def __init__(self, directory, stamp):
        self.directory = directory
        self.stamp = stamp
        self.path = os.path.join(directory, f"{self.KIND}_{stamp}.json")
        self.data = {}
        self.loaded = False        # True when the file was read at startup (not carried over or new)

    @property
    def name(self):
        """The file's base name (the console lines)."""
        return os.path.basename(self.path)

    def read(self):
        """
        The file read into self.data (its 'format' and 'written' keys left out); True when it
        existed, was readable and was of this store's format, or when the store takes another
        format (_accept_other_format: Settings carries it over, Computed does not).
        """
        data = _read_json(self.path)
        if data is None:
            return False
        if int(data.get('format', 0)) != self.FORMAT:
            print(f"{self.name}: format {data.get('format')} is not {self.FORMAT}; {self._on_format_mismatch()}")
            return self._accept_other_format(data)
        self.data = {k: v for k, v in data.items() if k not in ('format', 'written')}
        self.loaded = True
        return True

    def _on_format_mismatch(self):
        """The console's wording of what read() does with a file of another format."""
        return "ignored."

    def _accept_other_format(self, data):
        """What read() does with a file of another format: nothing here (False, not loaded); Settings takes it over."""
        return False

    def write(self):
        """The store written whole: its sections under 'format' (this store's) and 'written' (the time of writing); a temporary file, then os.replace.  True when written."""
        out = {'format': self.FORMAT, 'written': datetime.now().isoformat(timespec='seconds')}
        out.update(self.data)
        return _write_json(self.path, out)


class Settings(_Store):
    """
    settings_<stamp>.json: the hand-set state of the experiment (the module docstring).  get(section)
    gives a section's stored value (None when absent), put(section, value) stores it and writes the
    file at once.  An older format is read and carried over: nothing set by hand is ever discarded.
    """

    KIND = 'settings'
    FORMAT = SETTINGS_FORMAT

    def _on_format_mismatch(self):
        return "read and carried over as it is (settings are never discarded)."

    def _accept_other_format(self, data):
        self.data = {k: v for k, v in data.items() if k not in ('format', 'written')}
        self.loaded = True
        self.write()
        return True

    def get(self, section, default=None):
        """A section's stored value, `default` (None) when absent."""
        return self.data.get(section, default)

    def put(self, section, value):
        """The section replaced by `value` and the file written at once."""
        self.data[section] = value
        self.write()


class Computed(_Store):
    """
    computed_<stamp>.json: everything computed from the files alone (the module docstring), in
    sections that each carry their own format number and loader format.  section(name, fmt) gives a
    section's content when both match, else {} (the owner then rebuilds it); put(name, fmt, content)
    replaces a section in memory; write() writes the file (once per computation stage).
    """

    KIND = 'computed'
    FORMAT = COMPUTED_FORMAT

    def section(self, name, fmt):
        """The content of a section when its format is `fmt` and its loader format is sm4file.LOADER_FORMAT, else {} (a console line says which section is computed again)."""
        sec = self.data.get(name)
        if not isinstance(sec, dict):
            return {}
        if int(sec.get('format', -1)) != int(fmt) or sec.get('loader_format') != sm4file.LOADER_FORMAT:
            print(f"{self.name}: section '{name}' is of format {sec.get('format')} / loader "
                  f"{sec.get('loader_format')}, not {fmt} / {sm4file.LOADER_FORMAT}; it is computed again.")
            return {}
        content = sec.get('content')
        return content if isinstance(content, dict) else {}

    def put(self, name, fmt, content):
        """A section replaced in memory by `content` under format `fmt` and the loader format in force; write() puts it on disk."""
        self.data[name] = {'format': int(fmt), 'loader_format': sm4file.LOADER_FORMAT, 'content': content}


# ---------------------------------------------------------------------- #
#  The carry-over of the earlier caches                                    #
# ---------------------------------------------------------------------- #

OLD_DRIFT_FORMAT = 5      # the drift cache's last format: pairs, topos, manual (for reference; carry_over keeps the old file's own number)
OLD_SITES_FORMAT = 1
OLD_HEADER_FORMAT = 1


def _newest_readable(base):
    """(path, data) of the newest readable file of an old cache family (base, base_2, ...), or (None, None)."""
    for _, path in reversed(sm4file.cache_family(base)):
        data = _read_json(path)
        if data is not None:
            return path, data
    return None, None


def carry_over(directory, stamp, settings=None, computed=None):
    """
    The earlier caches read once into the stores given (the absent ones; None for a store that was
    read from its own file), which are then written, and every old file of the three families
    renamed with '.old', read or not.  `settings` takes the drift cache's 'manual' section (the
    spans and the stripe sets, as the 'drift' section) and the sites cache's flags ('flags');
    `computed` takes the drift cache's pairs and topos ('pairs'), the sites cache's topos and
    spectra ('metrics') and the header cache's files ('headers'), each under the old file's own
    format and loader numbers.  Returns the list of files renamed.
    """
    old = []
    drift_base = os.path.join(directory, f"drift_cache_{stamp}.json")
    sites_base = os.path.join(directory, f"sites_{stamp}.json")
    header_path = os.path.join(directory, f"headers_{stamp}.json")
    d_path, d_data = _newest_readable(drift_base)
    s_path, s_data = _newest_readable(sites_base)
    h_data = _read_json(header_path) if os.path.isfile(header_path) else None
    if settings is not None:
        if d_data is not None:
            man = d_data.get('manual') or {}
            settings.data['drift'] = {'spans': list(man.get('spans', [])), 'stripe': dict(man.get('stripe') or {})}
        if s_data is not None:
            settings.data['flags'] = dict(s_data.get('flags') or {})
        if d_data is not None or s_data is not None:
            settings.write()
            print(f"{settings.name}: made from "
                  + " and ".join(os.path.basename(p) for p in (d_path, s_path) if p) + ".")
    if computed is not None:
        if d_data is not None:
            computed.data['pairs'] = {'format': int(d_data.get('format', 0)), 'loader_format': d_data.get('loader_format'),
                                      'content': {'pairs': dict(d_data.get('pairs') or {}),
                                                  'topos': dict(d_data.get('topos') or {})}}
        if s_data is not None:
            computed.data['metrics'] = {'format': int(s_data.get('format', OLD_SITES_FORMAT)),
                                        'loader_format': s_data.get('loader_format'),
                                        'content': {'topos': dict(s_data.get('topos') or {}),
                                                    'spectra': dict(s_data.get('spectra') or {})}}
        if h_data is not None:
            computed.data['headers'] = {'format': int(h_data.get('format', OLD_HEADER_FORMAT)),
                                        'loader_format': h_data.get('loader_format'),
                                        'content': {'files': dict(h_data.get('files') or {})}}
        if d_data is not None or s_data is not None or h_data is not None:
            computed.write()
            print(f"{computed.name}: made from "
                  + " and ".join(os.path.basename(p) for p in (d_path, s_path, header_path if h_data else None) if p) + ".")
    for _, path in sm4file.cache_family(drift_base) + sm4file.cache_family(sites_base) + \
            ([(1, header_path)] if os.path.isfile(header_path) else []):
        try:
            os.replace(path, path + '.old')
            old.append(path)
        except OSError as exc:
            print(f"{os.path.basename(path)}: not renamed to .old ({exc}).")
    if old:
        print("earlier caches renamed with .old: " + ", ".join(os.path.basename(p) for p in old))
    return old


def open_stores(directory, files):
    """
    The experiment's (Settings, Computed): each read from its file; the absent ones (or unreadable,
    or for Computed of another format) made by the carry-over of the earlier caches when those
    exist, else empty.  The settings file is written at once when it does not exist yet, so it is
    there from the first startup on; the console names the sections each store loaded.  `files`
    is the directory's stamp-sorted .sm4 list (its first file gives the stamp).  The browser opens
    the stores once and hands them to the header scan (sm4file.header_cache), the drift tracker
    (drift.Drift) and the sites (sites.Sites); header_cache and Drift open their own when called
    without.
    """
    stamp = stamp_of(files)
    settings, computed = Settings(directory, stamp), Computed(directory, stamp)
    s_ok, c_ok = settings.read(), computed.read()
    if not s_ok or not c_ok:
        carry_over(directory, stamp, None if s_ok else settings, None if c_ok else computed)
    if not os.path.isfile(settings.path):
        settings.write()               # the settings file exists from the first startup on, empty until something is set
    if s_ok:
        print(f"settings loaded: {settings.name} ({', '.join(sorted(settings.data)) or 'empty'})")
    if c_ok:
        print(f"computed loaded: {computed.name} ({', '.join(sorted(computed.data)) or 'empty'})")
    return settings, computed
