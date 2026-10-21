"""
One .sm4 file: its kind, the file-name stamp and times, the image frame and raw
topography, the spectroscopy (STS), phase rotate (PhaseRotate), or noise spectrum
(NoiseSpectrum) data, and the header scan the browser uses at startup.

The loader (load, Page, Sm4Data).  A file is read whole once: every page with
every attribute the reader parses (the page header, its string block, the
drift, piezo, drive, lock-in, controller, and filter blocks), the file's
signature, the instrument's original path of the file, and each page's
recorded filetime.  Every page's stored values are converted on load to
physical units, data = stored x RHK_Zscale + RHK_Zoffset in the page's
RHK_Zunits, and every page carries its x axis (RHK_Xoffset + i x RHK_Xscale
over RHK_Xsize, in RHK_Xunits) and y axis (likewise with Y).  Nothing past the
loader sees a stored value: "raw" data in the package is the loader's data in
physical units.  A stored 0 (never written: an aborted page is zero-filled)
converts to exactly RHK_Zoffset, so the never-acquired and stopped-sweep tests
compare with the page's offset.

The sweep clock.  A spectroscopy page's RHK_Period is the digitizing period of
the LIA channel (1e-4 s on every spectroscopy-type page of an experiment, the
noise spectra included), not the time per energy point, and the file holds no
other per-point time.  The time per energy point is derived from the file span:
RHK_DateTime of the first page (the start) to the file-name stamp, which is
when the run was stopped (at the end of its scheduled time or by the stop
button), to the millisecond, and the moment the last acquired data point (the
last point before the zeros) was taken.  A grid spends its span on the sweeps,
points x sample time per started spectrum, with a return pass (n_fast x the
image page's RHK_Period) after every completed line; so the sample time is
(span - return passes) over the points of the completed spectra plus the
points a stopped spectrum got (grid_timing).  A single-point file spends its
span on the points of its sweeps the same way: the sample time is the span
over the points acquired by every started sweep, a stopped sweep counting its
points up to its last acquired one (point_timing).  Every energy point's time
(energy_times), the drift lines, the heatmap shear, and the map slices follow
from that sample time (dwell_s).
"""

if __package__ in (None, ''):
    # Run as a file (Spyder F5): make the package importable from wherever its folder sits.
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import re
from datetime import datetime
import numpy as np
from spym.io.rhksm4._sm4 import RHKsm4, RHKFileHeader, RHKPageIndexHeader, RHKPage, RHKPageHeader
from sm4_Browser import sts, phase, fft, sweeps

# The loader's format, recorded in everything the package computes from the files and keeps (the
# fit histories, every section of the computed store): 2 is the loader that converts every page to
# physical units on load (data = stored x RHK_Zscale + RHK_Zoffset); a record without it, or with
# another number, was written from the stored values (1) and is out of date.
LOADER_FORMAT = 2


def cache_family(base):
    """
    The existing files of one cache: the base path <stem><ext> (number 1)
    and its numbered successors <stem>_<n><ext> (n >= 2), written beside an
    out-of-date cache instead of over it; [(n, path)] in ascending number.
    """
    directory, name = os.path.split(base)
    stem, ext = os.path.splitext(name)
    out = [(1, base)] if os.path.isfile(base) else []
    pat = re.compile(r'^' + re.escape(stem) + r'_(\d+)' + re.escape(ext) + r'$')
    try:
        names = os.listdir(directory or '.')
    except OSError:
        names = []
    for n in names:
        m = pat.match(n)
        if m and int(m.group(1)) >= 2:
            out.append((int(m.group(1)), os.path.join(directory, n)))
    return sorted(out)


def cache_new_path(base):
    """The path of a new cache file of the family: the base when none exists, else the next number."""
    fam = cache_family(base)
    if not fam:
        return base
    directory, name = os.path.split(base)
    stem, ext = os.path.splitext(name)
    return os.path.join(directory, f"{stem}_{fam[-1][0] + 1}{ext}")

# --- file names and times ---

STAMP_RE = re.compile(r'(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{3})\s*$')


def filename_stamp(path):
    """(year, month, day, hour, minute, second, millisecond) from the end of the base name."""
    base = os.path.splitext(os.path.basename(path))[0]
    m = STAMP_RE.search(base)
    if m is None:
        raise ValueError(f"No _YYYY_MM_DD_HH_MM_SS_mmm stamp at the end of '{base}'.")
    return tuple(int(g) for g in m.groups())


def stamp_text(stamp):
    """A stamp tuple (filename_stamp) as 'YYYY-MM-DD HH:MM:SS.mmm'; its [11:] is the time of day the windows label entries with."""
    y, mo, d, h, mi, s, ms = stamp
    return f"{y:04d}-{mo:02d}-{d:02d} {h:02d}:{mi:02d}:{s:02d}.{ms:03d}"


def stamp_datetime(stamp):
    """A stamp tuple (filename_stamp) as a naive local datetime (the millisecond as microseconds)."""
    y, mo, d, h, mi, s, ms = stamp
    return datetime(y, mo, d, h, mi, s, ms * 1000)


def list_sm4_files(directory):
    """Every .sm4 file of the directory, sorted by its file-name stamp."""
    names = [n for n in os.listdir(directory) if n.lower().endswith('.sm4')]
    files = []
    for n in names:
        p = os.path.join(directory, n)
        try:
            files.append((filename_stamp(p), p))
        except ValueError as exc:
            print(f"Skipped: {exc}")
    files.sort(key=lambda t: t[0])
    return [p for _, p in files]


# --- the loader: the file read whole, every page in physical units ---

# Windows FILETIME (100 ns intervals since 1601-01-01 UTC) of the Unix epoch 1970-01-01 UTC.
_FILETIME_EPOCH = 116444736000000000


def filetime_seconds(value):
    """A recorded Windows FILETIME as seconds since the epoch (UTC); None for 0 or an unreadable value."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    return (v - _FILETIME_EPOCH) / 1e7


def _axis(attrs, size_key, offset_key, scale_key):
    """
    A page axis from its attributes, offset + i x scale over the size, the
    arithmetic the package has always used for the energy axis (the stored
    float32 attributes combined as the reader returns them, then float64);
    an empty array when the page lacks the attributes.
    """
    try:
        n = int(attrs[size_key])
        off, sc = attrs[offset_key], attrs[scale_key]
    except (KeyError, TypeError, ValueError):
        return np.zeros(0)
    return np.array([off + i * sc for i in range(n)], dtype=np.float64)


class Page:
    """
    One page of a .sm4 file as the loader keeps it:
      attrs      : every attribute the reader parsed, complete (a copy of the reader's dict)
      label, page_type, line_type, data_type, scan : RHK_Label, RHK_PageTypeName,
                   RHK_LineTypeName, RHK_PageDataTypeName, RHK_ScanTypeName
      data       : the stored values in physical units, stored x RHK_Zscale + RHK_Zoffset
                   (float64, the reader's array shape and order); None for a page without a
                   usable RHK_Zscale (absent, zero, or not finite: a sequential-data page
                   carries none), which cannot be put in physical units and is refused by
                   every use
      zscale, zoffset : RHK_Zscale and RHK_Zoffset (None with data None)
      units, xunits, yunits : RHK_Zunits, RHK_Xunits, RHK_Yunits (stripped)
      x, y       : the page's axes, RHK_Xoffset + i x RHK_Xscale over RHK_Xsize (likewise y)
      filetime_s : the filetime recorded with the page (RHK_ImageDrift_Filetime, else
                   RHK_SpecDrift_Filetime) in seconds since the epoch, None when not recorded
      original_path : the instrument's original path of the file (the reader replaces
                   RHK_FileName with the local path; this is the one the file holds)
    """

    def __init__(self, reader_page):
        a = dict(reader_page.attrs)
        self.attrs = a
        self.label = str(a.get('RHK_Label', ''))
        self.page_type = str(a.get('RHK_PageTypeName', ''))
        self.line_type = str(a.get('RHK_LineTypeName', ''))
        self.data_type = str(a.get('RHK_PageDataTypeName', ''))
        self.scan = str(a.get('RHK_ScanTypeName', ''))
        self.units = str(a.get('RHK_Zunits', '')).strip()
        self.xunits = str(a.get('RHK_Xunits', '')).strip()
        self.yunits = str(a.get('RHK_Yunits', '')).strip()
        self.original_path = getattr(reader_page, '_path', None)
        self.x = _axis(a, 'RHK_Xsize', 'RHK_Xoffset', 'RHK_Xscale')
        self.y = _axis(a, 'RHK_Ysize', 'RHK_Yoffset', 'RHK_Yscale')
        ft = filetime_seconds(a.get('RHK_ImageDrift_Filetime', 0))
        self.filetime_s = ft if ft is not None else filetime_seconds(a.get('RHK_SpecDrift_Filetime', 0))
        zs, zo = a.get('RHK_Zscale'), a.get('RHK_Zoffset', 0.0)
        try:
            zs = float(zs)
            zo = float(zo if zo is not None else 0.0)
        except (TypeError, ValueError):
            zs = None
        if zs is None or not np.isfinite(zs) or zs == 0.0 or not np.isfinite(zo):
            self.zscale = self.zoffset = None
            self.data = None
        else:
            self.zscale, self.zoffset = zs, zo
            self.data = np.asarray(reader_page.data).astype(np.float64) * zs + zo

    def unset(self, values=None):
        """
        The mask of values never written (stored 0, which converts to exactly
        RHK_Zoffset): over the page's data, or over `values` taken from it
        (a slice or a reshaped copy).
        """
        if self.data is None:
            raise ValueError(f"page '{self.label}' has no usable RHK_Zscale; it has no physical data.")
        v = self.data if values is None else np.asarray(values)
        return v == self.zoffset


class Sm4Data:
    """A .sm4 file read whole (load): its pages (Page, in file order), the file signature, the instrument's original path."""

    def __init__(self, path, pages, signature, original_path):
        self.path = path
        self.pages = pages
        self.signature = signature
        self.original_path = original_path


def load(path):
    """
    Read a .sm4 file whole (spym's reader: every page, every object it
    parses) and return it as Sm4Data with every page in physical units
    (Page).  The only loader of the package: everything that needs a file's
    pages goes through it.
    """
    reader = RHKsm4(path)
    pages = [Page(p) for p in reader._pages]
    original = next((p.original_path for p in pages if p.original_path), None)
    return Sm4Data(path, pages, str(getattr(reader._header, 'signature', '')), original)


# --- the header scan: kinds, scan centres, and recorded times without the page data ---

def sm4_header_info(path):
    """
    (page types, image geometry, line types) of a .sm4 file from its headers
    and string blocks: the file header, the page index, each page header,
    and each page's string block (labels, units, RHK_DateTime,
    RHK_CompletedLineCount) are read (spym's reader classes); the page data,
    the other page objects, and the thumbnails are not.  A page whose string
    block cannot be read keeps its header attributes alone.
    The page types are the RHK_PageTypeName of every page and the line types
    the RHK_LineTypeName of every page (the phase rotate pages are told by
    theirs); the geometry is that of the first
    RHK_PAGE_TOPOGRAPHIC page (else the first RHK_DATA_IMAGE page): a dict
    with 'center_m' (RHK_Xoffset, RHK_Yoffset: the recorded scan centre,
    absolute metres), 'angle_deg', 'xscale_m', 'yscale_m', 'nx', 'ny',
    'period_s' (RHK_Period of that page, the dwell per pixel, None when the
    page lacks it or holds 0), 'completed_lines' (RHK_CompletedLineCount,
    None when absent), 'n_topo_pages' (the topographic pages: 2 or more
    means a reverse pass is stored), and 'start_time' (RHK_DateTime of the
    first page as seconds since the epoch, None when unreadable); None
    without an image page.  Used by the browser at startup to learn every
    file's type and every topo's scan centre (the long-range moves) without
    loading any file, and by the drift build for the topos' times.
    """
    types, geom, line_types, _ = header_entry(path)
    return types, geom, line_types


def header_entry(path):
    """
    The header scan of one file (see sm4_header_info): (page types, image
    geometry, line types, start time), the start time RHK_DateTime of the
    first page in seconds since the epoch (None when unreadable), also for a
    file without an image page.  The one function that opens a file for its
    headers.
    """
    sm4 = RHKsm4.__new__(RHKsm4)
    sm4._file = open(path, 'rb')
    try:
        pages = _read_header_pages(sm4)
    finally:
        sm4._file.close()
    types, geom, line_types = header_info_from_attrs([p.attrs for p in pages])
    start = datetime_seconds(pages[0].attrs.get('RHK_DateTime', '')) if pages else None
    return types, geom, line_types, start


def _read_header_pages(sm4):
    """The page headers and string blocks of an open reader (sm4_header_info); the page data are not read."""
    sm4._header = RHKFileHeader(sm4)
    sm4._header._read_object_list(sm4)
    sm4._page_index_header = RHKPageIndexHeader(sm4)
    sm4._page_index_header._read_object_list(sm4)
    sm4._seek(sm4._page_index_header._get_offset('RHK_OBJECT_PAGE_INDEX_ARRAY'), 0)
    pages = []
    for _ in range(sm4._page_index_header.page_count):
        page = RHKPage(sm4)
        pages.append(page)
        page._read_object_list(sm4)
    for page in pages:
        hdr = RHKPageHeader(page, sm4)          # seeks to the page header; sets RHK_PageTypeName and the geometry
        try:
            # The page header's object list follows the header; of its objects only the string
            # block is read (it holds RHK_DateTime and RHK_CompletedLineCount).
            hdr._read_object_list(sm4)
            for obj in hdr._object_list:
                if obj.name == 'RHK_OBJECT_STRING_DATA':
                    page._read_object_content(obj)
        except Exception:
            pass
    return pages


def header_info_from_attrs(attrs_list):
    """
    (page types, image geometry, line types) from every page's attributes,
    in file order: the rule of sm4_header_info, shared with the header
    cache (header_cache) and with anything that already holds the pages.
    """
    types = []
    line_types = []
    geom = None
    n_topo = 0
    start_time = None
    for a in attrs_list:
        types.append(a.get('RHK_PageTypeName', ''))
        line_types.append(a.get('RHK_LineTypeName', ''))
        is_topo = a.get('RHK_PageTypeName', '') == 'RHK_PAGE_TOPOGRAPHIC'
        is_image = a.get('RHK_PageDataTypeName') == 'RHK_DATA_IMAGE'
        n_topo += int(is_topo)
        if start_time is None:
            start_time = datetime_seconds(a.get('RHK_DateTime', ''))
        if (is_topo or (is_image and geom is None)) and (geom is None or not geom.get('_topo')):
            try:
                geom = {'center_m': (float(a['RHK_Xoffset']), float(a['RHK_Yoffset'])),
                        'angle_deg': float(a.get('RHK_Angle', 0.0)),
                        'xscale_m': float(a['RHK_Xscale']), 'yscale_m': float(a['RHK_Yscale']),
                        'nx': int(a['RHK_Xsize']), 'ny': int(a['RHK_Ysize']), '_topo': is_topo,
                        'period_s': _period_of(a), 'completed_lines': _int_or_none(a.get('RHK_CompletedLineCount'))}
            except (KeyError, TypeError, ValueError):
                pass
    if geom is not None:
        geom.pop('_topo', None)
        geom['n_topo_pages'] = n_topo
        geom['start_time'] = start_time
    return types, geom, line_types


HEADER_CACHE_FORMAT = 1


def _file_sig(path):
    """[size, mtime] of a file (a list, as JSON stores it), the signature of its header-cache entry; None when it cannot be stat'ed."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return [int(st.st_size), float(st.st_mtime)]


def header_cache(directory, files, progress=True, computed=None):
    """
    The header scan of every file (sm4_header_info) through the 'headers'
    section of the experiment's computed store (caches.Computed; opened
    here when the caller gives none):
    {path: {'types', 'geom', 'line_types', 'start_time', 'end_time'}}.  The
    section holds one entry per file name with the file's size and
    modification time; an entry whose file changed (size or time) or that is
    missing is read anew, the rest are taken as stored, so a later startup
    reads only new or changed files.  start_time is RHK_DateTime of the first
    page (seconds since the epoch, None when unreadable), end_time the
    file-name stamp; their difference is the file's duration (the browser's
    step filter).  A file whose headers cannot be read is reported and
    entered with no page types (kind 'other').  The section is replaced and
    the store written once when any file was read from disk or the set of
    file names changed; a section of another format or loader is ignored
    (every file read, the section rebuilt).  With `progress` a line is
    printed every 50 files read and a summary at the end.
    """
    if computed is None:
        from sm4_Browser.caches import open_stores
        _, computed = open_stores(directory, files)
    stored = computed.section('headers', HEADER_CACHE_FORMAT).get('files', {})
    out, new_entries, n_read = {}, {}, 0
    for k, p in enumerate(files):
        name = os.path.basename(p)
        sig = _file_sig(p)
        e = stored.get(name)
        if e is None or sig is None or e.get('sig') != sig:
            try:
                types, geom, line_types, start = header_entry(p)
            except Exception as exc:
                print(f"\n{name}: header not readable ({exc}); taken as 'other'.")
                types, geom, line_types, start = [], None, [], None
            e = {'sig': sig, 'types': list(types), 'line_types': list(line_types),
                 'geom': geom, 'start_time': start}
            n_read += 1
            if progress and (n_read % 50 == 0):
                print(f"\rheader scan: {k + 1} / {len(files)} files ({n_read} read)", end='', flush=True)
        new_entries[name] = e
        g = e.get('geom')
        if g is not None and isinstance(g.get('center_m'), list):
            g['center_m'] = tuple(g['center_m'])
        try:
            end = stamp_datetime(filename_stamp(p)).timestamp()
        except ValueError:
            end = None
        out[p] = {'types': e['types'], 'line_types': e['line_types'], 'geom': g,
                  'start_time': e.get('start_time'), 'end_time': end}
    if progress and n_read >= 50:
        print()
    if n_read or set(new_entries) != set(stored):
        computed.put('headers', HEADER_CACHE_FORMAT, {'files': new_entries})
        computed.write()
    if progress:
        print(f"header scan: {len(files)} files, {n_read} read from disk, {len(files) - n_read} from the store "
              f"({computed.name}).")
    return out


def file_duration_s(entry):
    """A header-cache entry's duration (s): the file-name stamp minus RHK_DateTime; None when either is unknown."""
    if not entry or entry.get('start_time') is None or entry.get('end_time') is None:
        return None
    return max(float(entry['end_time']) - float(entry['start_time']), 0.0)


def datetime_seconds(text):
    """RHK_DateTime ('YYYY-MM-DDTHH:MM:SS...') as seconds since the epoch; None when unreadable."""
    try:
        return datetime.strptime(str(text)[:19], '%Y-%m-%dT%H:%M:%S').timestamp()
    except (TypeError, ValueError):
        return None


def _period_of(attrs):
    """RHK_Period of a page as a positive float (seconds per point or per pixel), None when absent or 0."""
    try:
        p = float(attrs.get('RHK_Period', 0.0))
    except (TypeError, ValueError):
        return None
    return p if p > 0 else None


def _int_or_none(value):
    """An attribute as int, None when absent or not a number."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


SPECTRA_PAGE_TYPE = {'grid': 'RHK_PAGE_INTERACTIVE_SPECTRA', 'point': 'RHK_PAGE_RAMP_SPECTROSCOPY_RP'}
# A spectrum and a topo pixel are the same point when the spectrum's recorded location, taken into the
# page's pixel frame, lies within this many pixels of the pixel's centre on both axes
# (Sm4File.spectrum_pixels).  The recorded spectrum coordinates and the page geometry agree to about
# 1e-4 px on the files measured, so 0.05 px never fails on a correct file and reports a convention
# error (a half-pixel offset, a mirrored axis, a scale or centre mismatch) instead of pairing across it.
PIXEL_PAIR_TOL_PX = 0.05
PHASE_LINE_TYPE = 'RHK_LINE_PHASE_ROTATE'
FFT_PAGE_TYPE = 'RHK_PAGE_NOISE_POWER_SPECTRUM'


def kind_from_types(types, line_types=()):
    """
    A file's type from its page types (and line types): 'grid'
    (RHK_PAGE_INTERACTIVE_SPECTRA pages), else 'point'
    (RHK_PAGE_RAMP_SPECTROSCOPY_RP pages, unless every one of them is a phase
    rotation, line type RHK_LINE_PHASE_ROTATE: 'phase'), else 'fft'
    (RHK_PAGE_NOISE_POWER_SPECTRUM), else 'topo' (RHK_PAGE_TOPOGRAPHIC), else
    'other'.  Sm4File and sm4_kind share this rule.
    """
    if 'RHK_PAGE_INTERACTIVE_SPECTRA' in types:
        return 'grid'
    if 'RHK_PAGE_RAMP_SPECTROSCOPY_RP' in types:
        ramp = [lt for t, lt in zip(types, line_types) if t == 'RHK_PAGE_RAMP_SPECTROSCOPY_RP']
        if ramp and all(lt == PHASE_LINE_TYPE for lt in ramp):
            return 'phase'
        return 'point'
    if FFT_PAGE_TYPE in types:
        return 'fft'
    if 'RHK_PAGE_TOPOGRAPHIC' in types:
        return 'topo'
    return 'other'


def sm4_kind(path):
    """A .sm4 file's type ('grid', 'point', 'phase', 'fft', 'topo', 'other') from its headers alone."""
    types, _, line_types = sm4_header_info(path)
    return kind_from_types(types, line_types)


_SI_PREFIXES = ((1e-15, 'f'), (1e-12, 'p'), (1e-9, 'n'), (1e-6, 'u'), (1e-3, 'm'), (1.0, ''),
                (1e3, 'k'), (1e6, 'M'), (1e9, 'G'))
_SI_UNITS = ('V', 'A', 'Hz', 's', 'm', 'm/A', 'm/As')      # the units that take an SI prefix


def si_text(value, unit='', digits=4):
    """
    A number as text for the info panel: with a unit that takes an SI prefix
    (_SI_UNITS), scaled to the prefix that puts it between 1 and 1000 (400 pA,
    25 mV, 100 us, 600 Hz); with another unit (%, deg), without a unit, or
    outside the prefix range, in scientific notation.  Integers stay
    integers; None, text, and NaN are returned as they are.
    """
    if value is None or isinstance(value, str):
        return value
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(v):
        return str(v)
    if unit in _SI_UNITS and v != 0.0:
        a = abs(v)
        for scale, prefix in reversed(_SI_PREFIXES):      # the scales are 1000 apart: the first match
            if a >= scale * (1.0 - 5.0 * 10.0 ** -digits):  # is below 1000 of it (0.9999 V reads 1 V)
                return f"{v / scale:.{digits}g} {prefix}{unit}"
        return f"{v:.{digits}e} {unit}"                       # below the smallest prefix
    if unit:
        return f"{v:.{digits}g} {unit}" if v == 0.0 or 1e-3 <= abs(v) < 1e4 else f"{v:.{digits}e} {unit}"
    if v == int(v) and abs(v) < 1e6:
        return str(int(v))
    return f"{v:.{digits}g}" if 1e-3 <= abs(v) < 1e4 else f"{v:.{digits}e}"


def measurement_info(attrs):
    """
    The measurement information a page's attributes hold, as [(name, text)]
    for the info panel: the setpoint, the lock-in and modulation settings,
    the filters, the sample time, and the operator's notes.  Nothing that
    describes the file's storage (page types, labels, scales, units) is
    included; what a page does not hold is left out.
    """
    a = attrs
    rows = []

    def put(name, key, unit='', digits=4, convert=None):
        """One row (name, text) for attribute `key` when the page holds it and it is not empty: converted, then si_text with the unit (text as it is)."""
        v = a.get(key)
        if v is None or (isinstance(v, str) and v == ''):
            return
        if convert is not None:
            v = convert(v)
        rows.append((name, si_text(v, unit, digits) if unit or not isinstance(v, str) else str(v)))

    put('time', 'RHK_DateTime', convert=lambda t: str(t).replace('T', ' ')[:19])
    put('bias', 'RHK_Bias', 'V')
    put('setpoint', 'RHK_Current', 'A')
    put('modulation frequency', 'RHK_CH1Drive_Frequency', 'Hz')
    put('modulation amplitude (energy resolution)', 'RHK_CH1Drive_Amplitude', 'V')
    put('modulation harmonic', 'RHK_CH1Drive_HarmonicFactor')
    put('lock-in 1 harmonic', 'RHK_Lockin0_HarmonicFactor')
    put('lock-in 1 phase', 'RHK_Lockin0_PhaseOffset', 'deg')
    put('lock-in 1 filter', 'RHK_Lockin0_FilterCutoffFrequency')
    put('lock-in 2 harmonic', 'RHK_Lockin1_HarmonicFactor')
    put('lock-in 2 phase', 'RHK_Lockin1_PhaseOffset', 'deg')
    put('lock-in 2 filter', 'RHK_Lockin1_FilterCutoffFrequency')
    put('low-pass filter 1', 'RHK_LowPassFilter1_CutoffFrequency', 'Hz')
    # RHK_Period: the dwell per pixel on an image page; on a spectroscopy page (RHK_DATA_LINE) the
    # digitizing period of the LIA channel, not the time per energy point (the module docstring).
    spec_page = str(a.get('RHK_PageDataTypeName', '')) == 'RHK_DATA_LINE' or a.get('RHK_PageDataType') == 1
    put('ADC sample period' if spec_page else 'sample time', 'RHK_Period', 's')
    put('oversampling', 'RHK_OverSamplingCount')
    put('feedback', 'RHK_ZPI_FeedbackType')
    put('feedback proportional gain', 'RHK_ZPI_ProportionalGain', 'm/A')
    put('feedback integral gain', 'RHK_ZPI_IntegralGain', 'm/As')
    put('x slope compensation', 'RHK_ScanProcessor_XSlopeCompensation', '%')
    put('y slope compensation', 'RHK_ScanProcessor_YSlopeCompensation', '%')
    put('scan angle', 'RHK_Angle', 'deg')
    put('session notes', 'RHK_SessionText')
    put('user notes', 'RHK_UserText')
    return rows


class Sm4File:
    """
    One .sm4 file: type (kind: 'grid', 'point', 'phase', 'fft', 'topo',
    'other', kind_from_types), image geometry, raw topography, spectra
    (STS) and their absolute positions, or a phase rotation (PhaseRotate)
    or a noise spectrum (NoiseSpectrum).  Lengths in metres as recorded,
    the nominal values: the piezo scale factor of the timeline (scales.py)
    is applied by the browser and the windows when they draw or measure
    (recorded metres x 1e9 / scale factor = calibrated nm), never here.
    Every data array is float64.  A grid file with several sweeps per pixel
    (sweeps.py) keeps the all-sweeps spectroscopy in spectra_all and works
    on a view (spectra, one column per pixel) chosen by sweep_mode.
    """

    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        self.stamp = filename_stamp(path)
        self.f = load(path)                    # the file read whole, every page in physical units (Sm4Data)
        pages = self.f.pages
        types = [p.page_type for p in pages]
        line_types = [p.line_type for p in pages]

        self.topo_pages = [i for i, t in enumerate(types) if t == 'RHK_PAGE_TOPOGRAPHIC']
        self.image_pages = [i for i, p in enumerate(pages)
                            if p.data_type == 'RHK_DATA_IMAGE'
                            or (p.data is not None and np.ndim(p.data) == 2 and p.attrs.get('RHK_Xunits') == 'm')]

        # The file's content by kind: spectra (STS) for a grid or point file (None when no
        # page matches the channel labels), the phase rotation, or the noise spectrum.
        self.spectra = None
        self.phase = None
        self.fft = None
        self.kind = kind_from_types(types, line_types)
        self.spectra_all = None
        self.sweep_info = None
        self.sweep_mode = None
        if self.kind in SPECTRA_PAGE_TYPE:
            try:
                self.spectra = sts.STS.from_pages(pages, SPECTRA_PAGE_TYPE[self.kind], name=self.name)
                if self.spectra is None:
                    print(f"{self.name}: no page matches the dI/dV or IETS channel; the file has no spectra.")
                elif self.kind == 'grid' and getattr(self.spectra, 'sweep_info', None) is not None:
                    # Several sweeps per pixel (sweeps.py): the windows work on a view, one column per pixel.
                    self.spectra_all = self.spectra
                    self.sweep_info = self.spectra.sweep_info
                    self.sweep_mode = sweeps.default_mode(self.sweep_info)
                    self.spectra = self.spectra_all.sweep_view(self.sweep_mode)
            except sts.AlignmentError as exc:
                # The Current (or IETS) page's recorded locations cannot be aligned onto the dI/dV
                # page's: the pairing of the pages is unknown, so the file shows no spectra.
                print(f"{exc}  The file has no spectra to show.")
                self.spectra = None
        elif self.kind == 'phase':
            self.phase = phase.PhaseRotate.from_pages(pages, name=self.name)
        elif self.kind == 'fft':
            self.fft = fft.NoiseSpectrum.from_pages(pages, name=self.name)

        # Image geometry from the first topography page (or the first image page).
        geo = self.topo_pages[0] if self.topo_pages else (self.image_pages[0] if self.image_pages else None)
        if geo is not None:
            a = pages[geo].attrs
            self.center_m = (float(a['RHK_Xoffset']), float(a['RHK_Yoffset']))
            self.angle_deg = float(a.get('RHK_Angle', 0.0))
            self.xscale_m = float(a['RHK_Xscale'])
            self.yscale_m = float(a['RHK_Yscale'])
            self.nx = int(a['RHK_Xsize'])
            self.ny = int(a['RHK_Ysize'])
            # The scan timing of the image page: RHK_Period is the dwell per pixel (s), so the raw
            # scan speed is |RHK_Xscale| / RHK_Period (the scale factor is not applied here);
            # RHK_CompletedLineCount the lines the scan completed (an aborted scan or grid).
            self.scan_period = _period_of(a)
            self.scan_speed = abs(self.xscale_m) / self.scan_period if self.scan_period else None
            self.completed_lines = _int_or_none(a.get('RHK_CompletedLineCount'))
        else:
            self.center_m = None
            self.angle_deg = 0.0
            self.xscale_m = self.yscale_m = None
            self.nx = self.ny = None
            self.scan_period = None
            self.scan_speed = None
            self.completed_lines = None
        self._timing = None            # grid_timing, computed once

        if self.kind == 'grid':
            # The grid as recorded: RHK_GridXsize spectra per line (the fast axis) and
            # RHK_GridYsize lines, from the grid's spectroscopy pages (every one carries the same
            # values).  A grid file that records no grid size has no spectra to show.
            sp = next((p for p in pages if p.page_type == SPECTRA_PAGE_TYPE['grid']), None)
            gx = _int_or_none(sp.attrs.get('RHK_GridXsize')) if sp is not None else None
            gy = _int_or_none(sp.attrs.get('RHK_GridYsize')) if sp is not None else None
            n_sp = int(self.spectra.n_spectra) if self.spectra is not None else 0
            # 0xFFFFFFFF (4294967295) is the instrument's 'not set': a grid stopped before the tip
            # reached its first point.  A size that is not set, or that cannot hold the recorded
            # spectra (more lines than spectra), is not a grid size.
            unset = [v for v in (gx, gy) if v is not None and (v >= 2 ** 31 - 1 or (n_sp and v > max(n_sp, 1) * 4))]
            if unset and self.spectra is not None:
                print(f"{self.name}: the grid size is not set (RHK_GridXsize {gx}, RHK_GridYsize {gy}: the grid was "
                      f"stopped before it began); its {n_sp} recorded spectra are shown as one line.")
                self.spectra.grid_x = n_sp or None
                self.spectra.grid_y = 1
                gx, gy = n_sp, 1
            elif unset:
                gx = gy = None
            if gx and gy and gx > 0 and gy > 0:
                self.n_fast, self.n_lines = int(gx), int(gy)
            else:
                if self.spectra is not None:
                    print(f"{self.name}: no recorded grid size (RHK_GridXsize {gx}, RHK_GridYsize {gy}); "
                          f"the file has no spectra to show.")
                self.spectra = None
                self.n_fast = self.n_lines = 0
            self.self_check_text = self._self_check(pages, types)
        else:
            self.n_fast = self.n_spectra if self.kind == 'point' else 0
            self.n_lines = 1
            self.self_check_text = None

    # --- identity for the averages (an entry of an avgN window) ---

    @property
    def key(self):
        """What identifies this entry in an averaging window: the file path (a Selection adds its columns)."""
        return self.path

    @property
    def entry_label(self):
        """The entry's label in an averaging window: the file's time of day."""
        return stamp_text(self.stamp)[11:]

    # --- spectra (pass-throughs to self.spectra; a file without spectra gives the empty values) ---

    @property
    def lia(self):
        """The dI/dV channel (npts, N) in physical units (units_lia), or None."""
        return None if self.spectra is None else self.spectra.lia

    @property
    def iets(self):
        """The IETS channel (npts, N) in physical units (units_iets), or None."""
        return None if self.spectra is None else self.spectra.iets

    @property
    def iv(self):
        """The I(V) page (npts, N) in physical units (units_iv), or None."""
        return None if self.spectra is None else self.spectra.iv

    @property
    def zscale_iv(self):
        """The Current page's RHK_Zscale, a record (the I(V) is already in units_iv), or None."""
        return None if self.spectra is None else self.spectra.zscale_iv

    @property
    def units_lia(self):
        """The dI/dV page's RHK_Zunits, or None."""
        return None if self.spectra is None else self.spectra.units_lia

    @property
    def units_iets(self):
        """The IETS page's RHK_Zunits, or None."""
        return None if self.spectra is None else self.spectra.units_iets

    @property
    def units_iv(self):
        """The Current page's RHK_Zunits, or None."""
        return None if self.spectra is None else self.spectra.units_iv

    @property
    def vmod(self):
        """The modulation amplitude RHK_CH1Drive_Amplitude (V), or None when not recorded."""
        return None if self.spectra is None else self.spectra.vmod

    @property
    def energy(self):
        """The energy axis of the spectra (V, the channel page's x axis, as stored), or None."""
        return None if self.spectra is None else self.spectra.energy

    @property
    def V_bias(self):
        """The setpoint bias RHK_Bias (V), or None."""
        return None if self.spectra is None else self.spectra.V_bias

    @property
    def I_setpoint(self):
        """The setpoint current RHK_Current (A), or None."""
        return None if self.spectra is None else self.spectra.I_setpoint

    @property
    def n_spectra(self):
        """The number of spectrum columns (of the view, for several sweeps per pixel); 0 without spectra."""
        return 0 if self.spectra is None else self.spectra.n_spectra

    @property
    def n_acquired(self):
        """The number of kept spectra (STS.acquired); 0 without spectra."""
        return 0 if self.spectra is None else self.spectra.n_acquired

    @property
    def acquired(self):
        """(N,) bool, the kept spectra (STS.kept_mask); empty without spectra."""
        return np.zeros(0, dtype=bool) if self.spectra is None else self.spectra.acquired

    @property
    def spec_xy_m(self):
        """Recorded positions (N, 2), metres: of the spectra, or of a phase rotation's single point."""
        if self.spectra is not None:
            return self.spectra.spec_xy_m
        if self.phase is not None:
            return self.phase.xy_m
        return np.zeros((0, 2))

    @property
    def spec_time_s(self):
        """RHK_SpecDrift_Time per spectrum (s), zeros when not recorded."""
        return np.zeros(0) if self.spectra is None else self.spectra.spec_time_s

    def channels(self):
        """The spectroscopy channels the file holds ('didv', 'iets'), () without spectra."""
        return () if self.spectra is None else self.spectra.channels()

    def measurement_info(self):
        """
        The measurement information of the file as [(name, text)] for the
        info panel (measurement_info of the page the kind is read from, plus
        the kind's own numbers: energy range and step, spectra and sweeps,
        position, image size), nothing about the file's storage.
        """
        rows = []
        pages = self.f.pages
        if self.spectra is not None:
            s = self.spectra
            e = s.energy
            step = float(np.median(np.abs(np.diff(e)))) if len(e) > 1 else float('nan')
            rows.append(('channels', ', '.join(sts.STS.CHANNEL_NAMES[c] for c in s.channels())
                         + (' with I(V)' if s.iv is not None else ', no I(V) page')))
            rows.append(('energy range', f"{si_text(float(np.min(e)), 'V')} to {si_text(float(np.max(e)), 'V')}, "
                                         f"{len(e)} points, step {si_text(step, 'V')}"))
            stopped = f", {s.n_stopped} stopped" if getattr(s, 'n_stopped', 0) else ""
            if self.kind == 'grid':
                rows.append(('spectra', f"{s.n_spectra} = {self.n_lines} lines x {self.n_fast}, "
                                        f"{s.n_acquired} kept{stopped}"))
                tm = self.grid_timing()
                rows.append(('sweep timing', (f"recorded per spectrum; " if tm['recorded'] else "")
                                             + f"sample time "
                                             f"{si_text(tm['dwell'], 's')} per point and sweep {si_text(tm['ramp'], 's')} "
                                             + ("(from the recorded times)" if tm['recorded'] else
                                                f"(derived from the file span, {tm['n_started']} sweep(s) started, "
                                                f"{tm['lines_done']} return pass(es))")
                                             + (f", return pass {si_text(tm['T_rev'], 's')}" if tm['T_rev'] else
                                                "; no return pass time (no RHK_Period on the image page)")))
            else:
                rows.append(('sweeps', f"{s.n_spectra}, {s.n_acquired} kept{stopped}"))
                if self.kind == 'point':
                    pt = self.point_timing()
                    rows.append(('sweep timing', f"sample time {si_text(pt['dwell'], 's')} per point and sweep "
                                                 f"{si_text(pt['period'], 's')} (derived from the file span over the "
                                                 f"points acquired, {pt['n_started']} sweep(s) started)"))
            if len(s.spec_xy_m):
                rows.append(('position', f"x {si_text(float(s.spec_xy_m[0, 0]), 'm')}, "
                                         f"y {si_text(float(s.spec_xy_m[0, 1]), 'm')}"))
            rows.extend(measurement_info(s.attrs))
        elif self.phase is not None:
            rows.extend(self.phase.info())
        elif self.fft is not None:
            rows.extend(self.fft.info())
        if self.topo_pages:
            a = pages[self.topo_pages[0]].attrs
            rows.append(('image', f"{self.nx} x {self.ny} px, "
                                  f"{si_text(self.nx * abs(self.xscale_m), 'm')} x {si_text(self.ny * abs(self.yscale_m), 'm')}, "
                                  f"{len(self.topo_pages)} topography pages"))
            rows.append(('scan centre', f"x {si_text(self.center_m[0], 'm')}, y {si_text(self.center_m[1], 'm')}"))
            if self.scan_speed is not None:
                rows.append(('scan speed (raw)', f"{self.scan_speed * 1e9:.3g} nm/s ({si_text(self.scan_period, 's')} per pixel)"))
            if self.completed_lines is not None:
                rows.append(('completed lines', f"{self.completed_lines} of {self.ny}"))
            if self.spectra is None and self.phase is None and self.fft is None:
                rows.extend(measurement_info(a))
        elif self.image_pages and self.spectra is None and self.phase is None and self.fft is None:
            rows.extend(measurement_info(pages[self.image_pages[0]].attrs))
        return rows

    def line_columns(self, line_index):
        """Columns of one line of a grid (the Line tool's chunk_index slicing)."""
        return slice(line_index * self.n_fast, (line_index + 1) * self.n_fast)

    # --- several sweeps per pixel (sweeps.py) ---

    @property
    def n_sweeps(self):
        """The sweeps recorded at every pixel (1 for one sweep per pixel)."""
        return int(self.sweep_info['k']) if self.sweep_info is not None else 1

    def sweep_modes(self):
        """The sweep views this file offers ('forward sweeps', 'reverse sweeps', 'all sweeps', 'sweep n'); () for one sweep per pixel."""
        return sweeps.sweep_modes(self.sweep_info)

    def with_sweeps(self, mode):
        """
        The file seen through another sweep view (a shallow copy sharing the pages and the
        all-sweeps spectroscopy; its own view, timing, and caches); the file itself for one sweep
        per pixel, None, or the view it already shows.
        """
        if self.sweep_info is None or mode is None or mode == self.sweep_mode:
            return self
        if mode not in self.sweep_modes():
            raise ValueError(f"{self.name}: no sweep view '{mode}' (it offers {', '.join(self.sweep_modes())}).")
        import copy
        g = copy.copy(self)
        g.spectra = self.spectra_all.sweep_view(mode)
        g.sweep_mode = mode
        g._timing = None
        return g

    # --- times (seconds since the epoch) ---

    @property
    def end_time(self):
        """The file-name stamp, to the millisecond: when the run was stopped, the moment the last acquired point was taken."""
        return stamp_datetime(self.stamp).timestamp()

    @property
    def start_time(self):
        """RHK_DateTime of the first page (start of the acquisition); the end time when unreadable."""
        txt = self.f.pages[0].attrs.get('RHK_DateTime', '') if self.f.pages else ''
        try:
            t = datetime.strptime(str(txt)[:19], '%Y-%m-%dT%H:%M:%S').timestamp()
        except ValueError:
            return self.end_time
        return t if t <= self.end_time else self.end_time

    @property
    def mid_time(self):
        """The middle of the file's span (s since the epoch)."""
        return 0.5 * (self.start_time + self.end_time)

    # --- the sweep timing ---

    @property
    def adc_period(self):
        """The spectroscopy page's RHK_Period (s): the LIA channel's digitizing period (the module docstring); None without one."""
        return None if self.spectra is None else getattr(self.spectra, 'adc_period', None)

    @property
    def dwell_s(self):
        """
        The time per energy point (s), derived from the file span (the module
        docstring): a grid's grid_timing 'dwell', a single point's period over
        its points (point_timing); None for other kinds or without spectra.
        """
        if self.spectra is None or self.n_spectra == 0:
            return None
        if self.kind == 'grid':
            return self.grid_timing()['dwell']
        if self.kind == 'point':
            return self.point_timing()['dwell']
        return None

    @property
    def ramp_duration(self):
        """The duration of one complete sweep (s): points x dwell_s; None without a dwell."""
        d = self.dwell_s
        return None if d is None else float(len(self.energy)) * d

    def point_timing(self):
        """
        The timing of a single-point file (the module docstring, "The sweep
        clock"): the span from the start (RHK_DateTime) to the file-name
        stamp (when the run was stopped, the moment the last acquired point
        was taken) is spent on the points acquired, sweep after sweep in file
        order.  A dict with
          'dwell'     : the sample time per point (s): the span over the points of every
                        started sweep, a stopped sweep counting its points up to its last
                        acquired one (points_acquired)
          'period'    : a complete sweep, points x dwell (s)
          'n_started' : the sweeps started (kept plus stopped)
          'start', 'end' : (N,) every sweep's first point slot and the end of its last
                        acquired point's slot (s since the epoch); a sweep never started
                        has start = end, the moment it would have begun
        The last acquired point's slot ends at the stamp.
        """
        N = self.n_spectra
        kept = np.asarray(self.acquired, dtype=bool)
        ever = np.asarray(getattr(self.spectra, 'ever', kept), dtype=bool)[:N] if self.spectra is not None else kept
        started = kept | ever
        pts = np.asarray(self.points_acquired, dtype=np.float64)[:N]
        pts = np.where(started, pts, 0.0)
        n_points = float(pts.sum())
        t0 = float(self.start_time)
        span = max(float(self.end_time) - t0, 0.0)
        dwell = span / n_points if n_points > 0 else 0.0
        npts = max(len(self.energy), 1)
        ends = t0 + dwell * np.cumsum(pts)
        starts = ends - dwell * pts
        return {'dwell': dwell, 'period': dwell * npts, 'n_started': int(started.sum()),
                'start': starts, 'end': ends}

    @property
    def points_acquired(self):
        """(N,) the points each sweep acquired (STS.points_acquired): all for a kept sweep, fewer for a stopped one, 0 for one never started."""
        if self.spectra is None:
            return np.zeros(0, dtype=int)
        pa = getattr(self.spectra, 'points_acquired', None)
        if pa is None or len(pa) != self.n_spectra:
            return np.where(np.asarray(self.acquired, dtype=bool), len(self.energy), 0)
        return np.asarray(pa, dtype=int)

    def grid_timing(self):
        """
        The timing of a grid file's spectra and return passes (computed once;
        the module docstring, "The sweep clock"): a dict with
          'recorded'   : True when the file records a time per spectrum (RHK_SpecDrift_Time)
          'dwell'      : the sample time per energy point (s): (the span, RHK_DateTime of the
                         first page to the file-name stamp, minus the completed lines' return
                         passes) over the points of the completed spectra plus the points a
                         stopped spectrum got; with recorded times the median gap between
                         spectra over the points
          'ramp'       : the sweep of a complete spectrum, points x dwell (s)
          'T_rev'      : the return pass of a line, n_fast x RHK_Period of the image page
                         (the raw scan speed |RHK_Xscale| / RHK_Period); None without the field
          'lines_done' : the lines whose return pass was made (RHK_CompletedLineCount, else the
                         lines whose spectra are all kept)
          'n_started'  : the sweeps started (kept plus stopped)
          'ramp_start', 'ramp_end' : (N,) the sweep of every spectrum (NaN for one not kept);
                         the ramp starts at the point's arrival, right after the previous
                         sweep (or return pass) ends
          'rev_start', 'rev_end'   : (n_lines,) the return pass of every line (NaN for a line
                         without one)
          'arrive'     : (N,) for a view of several sweeps per pixel only (_view_timing): the
                         tip's arrival at every pixel, the start of its first sweep
        With recorded times the ramp of spectrum k starts at RHK_DateTime + its recorded
        time; the return pass of a completed line starts at its last kept spectrum's ramp
        end.  Without them the points run from RHK_DateTime in file order, sweep after
        sweep, a return pass after every completed line, the stopped sweep taking its
        partial sweep at its place (its ramp is not reported).
        """
        if self._timing is not None:
            return self._timing
        if self.sweep_info is not None:
            self._timing = self._view_timing()
            return self._timing
        N = self.n_spectra
        n_fast = max(int(self.n_fast), 1) if self.kind == 'grid' else max(N, 1)
        n_lines = max(int(self.n_lines), 1)
        kept = np.asarray(self.acquired, dtype=bool)
        npts = max(len(self.energy), 1)
        T_rev = (n_fast * self.scan_period) if (self.kind == 'grid' and self.scan_period) else None
        t0, t1 = float(self.start_time), float(self.end_time)
        ramp_start = np.full(N, np.nan)
        ramp_end = np.full(N, np.nan)
        rev_start = np.full(n_lines, np.nan)
        rev_end = np.full(n_lines, np.nan)
        full = [bool(np.all(kept[i * n_fast:(i + 1) * n_fast])) and (i + 1) * n_fast <= N for i in range(n_lines)]
        if self.completed_lines is not None:
            lines_done = int(min(max(self.completed_lines, 0), n_lines))
        else:
            lines_done = sum(1 for i in range(n_lines) if full[i])
        rec = np.asarray(self.spec_time_s, dtype=np.float64)
        recorded = len(rec) == N and bool(np.any(rec != 0))
        started = kept | (np.asarray(getattr(self.spectra, 'ever', kept), dtype=bool)[:N] if self.spectra is not None else kept)
        n_started = int(started.sum())
        pts = np.asarray(self.points_acquired, dtype=np.float64)[:N]
        if recorded:
            idx = np.flatnonzero(kept)
            ramp_start[idx] = t0 + rec[idx]
            gaps = np.diff(ramp_start[idx]) if len(idx) > 1 else np.zeros(0)
            r = max(float(np.median(gaps)), 0.0) if len(gaps) else 0.0
            dwell = r / npts
            ramp_end[idx] = ramp_start[idx] + r
            for i in range(n_lines):
                if i < lines_done:
                    cols = idx[(idx >= i * n_fast) & (idx < (i + 1) * n_fast)]
                    if len(cols):
                        rev_start[i] = ramp_end[cols[-1]]
                        rev_end[i] = rev_start[i] + (T_rev or 0.0)
        else:
            W = max(t1 - t0, 0.0)
            den = float(pts[started].sum())                      # the points of every started sweep (the stopped one partial)
            dwell = max(W - lines_done * (T_rev or 0.0), 0.0) / den if den > 0 else 0.0
            t = t0
            for i in range(n_lines):
                for k in range(i * n_fast, min((i + 1) * n_fast, N)):
                    if not started[k]:
                        continue
                    t_ramp = t
                    sweep_k = dwell * pts[k]
                    if kept[k]:
                        ramp_start[k] = t_ramp
                        ramp_end[k] = t_ramp + sweep_k
                    t = t_ramp + sweep_k
                if i < lines_done:
                    rev_start[i] = t
                    t += (T_rev or 0.0)
                    rev_end[i] = t
        ramp = dwell * npts
        self._timing = {'recorded': recorded, 'dwell': dwell, 'ramp': ramp,
                        'T_rev': T_rev, 'lines_done': lines_done, 'n_started': n_started,
                        'ramp_start': ramp_start, 'ramp_end': ramp_end, 'rev_start': rev_start, 'rev_end': rev_end}
        return self._timing

    def _all_timing(self):
        """
        grid_timing over every column of a file with several sweeps per pixel (the columns pixel after
        pixel, each pixel's sweeps in acquisition order: a line is n_fast x k consecutive columns, its
        return pass after its last pixel's last sweep); shared by every view (kept on the all-sweeps
        spectroscopy).
        """
        allv = self.spectra_all
        cached = getattr(allv, '_timing_all', None)
        if cached is not None:
            return cached
        k = self.n_sweeps
        import copy
        g = copy.copy(self)
        g.sweep_info = None
        g.spectra = allv
        g.n_fast = int(self.n_fast) * k
        g._timing = None
        g.scan_period = (self.scan_period / k) if self.scan_period else self.scan_period   # T_rev = n_fast x period kept
        tm = g.grid_timing()
        allv._timing_all = tm
        return tm

    def _view_timing(self):
        """grid_timing of a sweep view: every pixel's ramp from its chosen sweeps, its arrival from its first sweep."""
        tm_all = self._all_timing()
        chosen = self.spectra.view_cols
        info = self.sweep_info
        n = len(chosen)
        rs, re_, arrive = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
        for p, c in enumerate(chosen):
            kc = c[self.spectra_all.acquired[c]] if len(c) else c
            if len(kc):
                rs[p] = np.nanmin(tm_all['ramp_start'][kc])
                re_[p] = np.nanmax(tm_all['ramp_end'][kc])
            allc = info['cols_of'][p]
            t = tm_all['ramp_start'][allc]
            if np.isfinite(t).any():
                arrive[p] = np.nanmin(t)
        out = dict(tm_all)
        out.update({'ramp_start': rs, 'ramp_end': re_, 'arrive': arrive})
        return out

    def _column_point_times(self, cols, edges=False):
        """
        (len(cols), npts) the time of every energy point of all-sweeps columns (s since the epoch;
        NaN for a column not kept), in the axis order, from the column's ramp start and its
        direction (acquired down the axis: the last point first); edges=True: the npts + 1 cell edges.
        """
        tm_all = self._all_timing()
        npts = len(self.energy)
        dwell = float(tm_all['dwell'])
        desc = self.sweep_info['acq_desc'][cols]
        n = npts + 1 if edges else npts
        idx = np.arange(n, dtype=np.float64)
        rank = np.where(desc[:, None], (npts if edges else npts - 1) - idx[None, :], idx[None, :])
        t0 = tm_all['ramp_start'][cols]
        out = t0[:, None] + rank * dwell
        out[~self.spectra_all.acquired[cols]] = np.nan
        return out

    def energy_edge_times(self):
        """
        (N, npts + 1): the time of every energy-cell edge of every spectrum, in the axis order (the
        Line window's row shear): the sweep window spread over the points, the first edge at the
        start; for a view of several sweeps per pixel, the mean over its chosen sweeps, each running
        its own way.
        """
        npts = len(self.energy) if self.spectra is not None else 0
        if self.sweep_info is not None and self.spectra is not None:
            out = np.full((self.n_spectra, npts + 1), np.nan)
            for p, c in enumerate(self.spectra.view_cols):
                if len(c):
                    t = self._column_point_times(c, edges=True)
                    if np.isfinite(t).any():
                        out[p] = np.nanmean(t, axis=0)
            return out
        win = self.spectrum_windows()
        edges = np.arange(npts + 1, dtype=np.float64)
        return win[:, :1] + (win[:, 1:] - win[:, :1]) * edges[None, :] / max(npts, 1)

    def spectrum_times(self):
        """
        Acquisition time of every spectrum (s since the epoch; NaN for a
        spectrum not kept): the start of its ramp (grid_timing) for a grid
        file, the sweep centres (sweep_times) for a single point, and the
        linear spread from the start to the end of the file over the kept
        spectra for any other kind.
        """
        if self.kind == 'grid' and self.spectra is not None:
            return self.grid_timing()['ramp_start'].copy()
        if self.kind == 'point' and self.spectra is not None:
            return self.sweep_times()
        t = np.full(self.n_spectra, np.nan)
        idx = np.where(self.acquired)[0]
        if len(idx) == 0:
            return t
        if len(idx) == 1:
            t[idx] = self.end_time
            return t
        t[idx] = self.start_time + (self.end_time - self.start_time) * np.arange(len(idx)) / (len(idx) - 1)
        return t

    def spectrum_windows(self):
        """
        (N, 2): the start and end of every spectrum's sweep (s since the
        epoch; NaN for a spectrum not kept): grid_timing for a grid; for a
        single point the window of its acquired points (point_timing), or,
        when the file recorded a time per sweep, that time plus and minus half
        a complete sweep; the linear spread for other kinds.
        """
        N = self.n_spectra
        win = np.full((N, 2), np.nan)
        if N == 0:
            return win
        if self.kind == 'grid' and self.spectra is not None:
            tm = self.grid_timing()
            win[:, 0], win[:, 1] = tm['ramp_start'], tm['ramp_end']
            return win
        t = self.spectrum_times()
        if self.kind == 'point':
            rec = self.spec_time_s
            if len(rec) == N and np.any(rec != 0):
                r = self.point_timing()['period']                  # recorded sweep times: centred windows
                win[:, 0], win[:, 1] = t - 0.5 * r, t + 0.5 * r
            else:
                pt = self.point_timing()                           # each sweep's acquired points (point_timing)
                win[:, 0], win[:, 1] = pt['start'], pt['end']
        else:
            win[:, 0] = t
            win[:, 1] = t
        win[~np.asarray(self.acquired, dtype=bool)] = np.nan
        return win

    def energy_times(self):
        """
        (N, npts): the acquisition time of every energy point of every
        spectrum, in the file's column order (the acquisition order): the
        sweep's start plus the point's index times the sweep's sample time
        (the sweep window, spectrum_windows, over the points: the derived
        dwell_s).  NaN for a spectrum not kept.
        """
        npts = len(self.energy) if self.spectra is not None else 0
        if npts == 0:
            return np.zeros((self.n_spectra, 0))
        if self.sweep_info is not None:
            # A view of several sweeps per pixel: every point at the mean time of its chosen sweeps,
            # each sweep running its own way along the axis.
            out = np.full((self.n_spectra, npts), np.nan)
            for p, c in enumerate(self.spectra.view_cols):
                if len(c):
                    t = self._column_point_times(c)
                    if np.isfinite(t).any():
                        out[p] = np.nanmean(t, axis=0)
            return out
        win = self.spectrum_windows()
        dwell = (win[:, 1:] - win[:, :1]) / npts
        return win[:, :1] + np.arange(npts, dtype=np.float64)[None, :] * dwell

    def pixel_times(self, direction='forward'):
        """
        (ny, nx): the acquisition time of every pixel of the forward or
        reverse topography page, raw pixel order (s since the epoch).  A
        grid: the forward pixel of spectrum k is its ramp start (the tip
        stops there for the spectrum), the reverse pixel j of line i lies on
        the return pass at rev_start + (n_fast - 1 - j) x RHK_Period (the
        return runs from the line's end back to its start); NaN where the
        file has no spectrum or no return pass.  A topo image: scanned row
        by row from the first page's RHK_DateTime, each row forward then
        back when a reverse page exists, RHK_Period per pixel (the file span
        over the pixels when the page has no period); the first pixel of the
        image is at the start time.
        """
        if self.nx is None or self.ny is None:
            raise ValueError(f"{self.name} has no image page.")
        nx, ny = int(self.nx), int(self.ny)
        if direction not in ('forward', 'reverse'):
            raise ValueError("pixel_times: direction must be 'forward' or 'reverse'.")
        if self.kind == 'grid' and self.spectra is not None:
            tm = self.grid_timing()
            n_fast, n_lines = max(int(self.n_fast), 1), max(int(self.n_lines), 1)
            out = np.full((ny, nx), np.nan)
            if direction == 'forward':
                n = min(n_lines * n_fast, self.n_spectra, ny * nx)
                flat = out.ravel()
                flat[:n] = tm.get('arrive', tm['ramp_start'])[:n]      # the tip's arrival at the pixel
                return flat.reshape(ny, nx) if (ny, nx) == (n_lines, n_fast) else out
            if (ny, nx) != (n_lines, n_fast):
                return out
            p = self.scan_period or 0.0
            for i in range(min(n_lines, ny)):
                if np.isfinite(tm['rev_start'][i]):
                    out[i, :] = tm['rev_start'][i] + (n_fast - 1 - np.arange(n_fast)) * p
            return out
        passes = 2 if len(self.topo_pages) > 1 else 1
        t0 = float(self.start_time)
        p = self.scan_period
        if p is None:
            span = max(float(self.end_time) - t0, 0.0)
            p = span / max(ny * nx * passes, 1)
        line_time = nx * p * passes
        ii, jj = np.meshgrid(np.arange(ny, dtype=np.float64), np.arange(nx, dtype=np.float64), indexing='ij')
        if direction == 'forward':
            return t0 + ii * line_time + jj * p
        return t0 + ii * line_time + nx * p + (nx - 1 - jj) * p

    def sweep_times(self):
        """
        Acquisition time of every sweep of a single-point file (s since the
        epoch).  From the file when it recorded one per sweep
        (RHK_SpecDrift_Time, taken as seconds after the page's start time
        RHK_DateTime); otherwise computed: the centre of the sweep's acquired
        points (point_timing: the span from the start time to the file-name
        stamp spent on the points acquired, sweep after sweep).
        """
        n = self.n_spectra
        if n == 0:
            return np.zeros(0)
        rec = self.spec_time_s
        if len(rec) == n and np.any(rec != 0):
            return self.start_time + rec
        pt = self.point_timing()
        return 0.5 * (pt['start'] + pt['end'])

    def grid_axes(self):
        """
        Unit vectors (absolute frame, metres) of the grid's fast axis (along a
        line, in the direction the spectra were taken) and slow axis (from one
        line to the next), from the recorded positions; (None, None) without them.
        """
        xy = self.spec_xy_m
        if len(xy) < 2:
            return None, None
        fast = xy[1] - xy[0]
        nf = np.linalg.norm(fast)
        if nf == 0:
            return None, None
        fast = fast / nf
        if len(xy) > self.n_fast:
            slow = xy[self.n_fast] - xy[0]
            ns = np.linalg.norm(slow)
            slow = slow / ns if ns > 0 else np.array([-fast[1], fast[0]])
        else:
            slow = np.array([-fast[1], fast[0]])
        return fast, slow

    def curves(self, hampel=None, sgolay=None, divide_iv=False, iv_broadening=0.0, channel='didv',
               iv_zero_sub=False, normalize=True, calibrate=True):
        """
        Every spectrum of the channel ('didv' or 'iets') through the pipeline
        (npts, N), original energy order (STS.process; iv_zero_sub: the
        current's value at 0 V subtracted before the division's I/V;
        normalize and calibrate: the two switchable steps); never-acquired
        spectra come back as zero columns.
        """
        if self.spectra is None:
            raise ValueError(f"{self.name} has no spectra.")
        if self.iv is None and calibrate:
            print(f"{self.name}: no 'Current' channel, curves shown without calibration.")
        return self.spectra.process(channel, hampel, sgolay, divide_iv, iv_broadening, iv_zero_sub=iv_zero_sub,
                                    normalize=normalize, calibrate=calibrate)

    # --- images ---

    def topo_raw(self, direction='forward'):
        """
        Topography of the first ('forward') or second ('reverse') topographic
        page, in metres, raw file pixel order: rows = scan lines, columns =
        fast axis.  spym's sign flips are undone and the array is reshaped
        (RHK_Ysize, RHK_Xsize) from the flat file order.  The page's data is
        the loader's, already in metres (stored x RHK_Zscale + RHK_Zoffset).
        Pixels never written (stored 0, exactly the page's RHK_Zoffset after
        the conversion: an aborted scan or grid is zero-filled) were never
        acquired and are returned as NaN; they are left out of the
        processing and drawn transparent.  The forward/reverse 'average' is
        built from the two processed passes by topo.processed_topo (range
        matched, then averaged), not here: 'average' is refused.
        """
        if not self.topo_pages:
            raise ValueError(f"{self.name} has no topography page.")
        if direction not in ('forward', 'reverse'):
            raise ValueError(f"topo_raw: direction must be 'forward' or 'reverse' (got '{direction}'); "
                             "the 'average' is topo.processed_topo.")
        idx = self.topo_pages[0] if direction == 'forward' or len(self.topo_pages) < 2 else self.topo_pages[1]
        page = self.f.pages[idx]
        if page.data is None:
            raise ValueError(f"{self.name}: the topography page has no usable RHK_Zscale; no heights in metres.")
        a = np.asarray(page.data)
        if page.attrs['RHK_Xscale'] < 0:
            a = np.flip(a, axis=1)
        if page.attrs['RHK_Yscale'] > 0:
            a = np.flip(a, axis=0)
        img = a.ravel().reshape(int(page.attrs['RHK_Ysize']), int(page.attrs['RHK_Xsize']))
        out = np.array(img, dtype=np.float64)
        out[page.unset(img)] = np.nan
        if direction == 'forward' and self.kind == 'grid':
            # The image page holds the completed lines only; the pixel of a kept spectrum of an
            # unfinished line comes from the embedded Topography spectra channel (the Z recorded at
            # that spectrum, the same value: the self check), placed in the pixel recorded at the
            # spectrum's location (spectrum_pixels), so the line is drawn to its last kept spectrum.
            z = self.spectra_z()
            if z is not None:
                idx, _ = self.spectrum_pixels('forward')
                for k in np.flatnonzero(np.isfinite(z) & (idx[:, 0] >= 0)):
                    i, j = idx[k]
                    if not np.isfinite(out[i, j]):
                        out[i, j] = z[k]
        return out

    def topo_spectra_z(self):
        """
        The embedded Topography spectra channel of a grid (the Z recorded at
        every spectrum) as an (n_lines, n_fast) array in metres, raw spectrum
        order, NaN where the value was never written (the page's
        RHK_Zoffset: not acquired) or the spectrum is not kept; None without
        the channel or when the grid's shape is unknown.  The page's data is
        the loader's, in metres.
        """
        if self.kind != 'grid' or self.spectra is None or not self.n_fast or not self.n_lines:
            return None
        pages = self.f.pages
        tz = next((i for i, p in enumerate(pages) if p.page_type == 'RHK_PAGE_INTERACTIVE_SPECTRA'
                   and p.label == 'Topography'), None)
        if tz is None:
            return None
        try:
            zpage = pages[tz]
            n = int(self.n_lines) * int(self.n_fast)
            z = self._z_per_spectrum(zpage)[:n]
            if z.size < n:
                return None
            zg = z.reshape(int(self.n_lines), int(self.n_fast))
            kept = np.asarray(self.acquired, dtype=bool)[:n]
            if len(kept) == n:
                zg[~kept.reshape(int(self.n_lines), int(self.n_fast))] = np.nan
            return zg
        except Exception:
            return None

    def rotation(self):
        """The 2 x 2 rotation of the image's scan frame into the absolute frame (RHK_Angle, counter-clockwise), applied as R @ uv."""
        th = np.deg2rad(self.angle_deg)
        return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    def frame_extent_m(self):
        """[left, right, bottom, top] of the image in its own scan frame (pixel edges, metres)."""
        u0 = -(self.nx - 1) / 2.0 * self.xscale_m
        u1 = (self.nx - 1) / 2.0 * self.xscale_m
        v0 = -(self.ny - 1) / 2.0 * self.yscale_m
        v1 = (self.ny - 1) / 2.0 * self.yscale_m
        return [u0 - self.xscale_m / 2.0, u1 + self.xscale_m / 2.0,
                v0 - self.yscale_m / 2.0, v1 + self.yscale_m / 2.0]

    def corners_m(self):
        """The four absolute corner positions of the image (metres), shape (4, 2)."""
        l, r, b, t = self.frame_extent_m()
        uv = np.array([[l, b], [r, b], [r, t], [l, t]])
        return uv @ self.rotation().T + np.asarray(self.center_m)

    def to_frame_m(self, xy_m):
        """Absolute positions (N, 2) into the image's scan frame (u, v), metres."""
        xy = np.atleast_2d(np.asarray(xy_m, dtype=np.float64)) - np.asarray(self.center_m)
        return xy @ self.rotation()           # R^-1 = R^T, applied as rows

    def geometry(self):
        """Footprint record (center, angle, frame extent) kept in the browser's index; None without an image."""
        if self.center_m is None or not self.topo_pages:
            return None
        return {'center_m': tuple(self.center_m), 'angle_deg': self.angle_deg,
                'extent_m': list(self.frame_extent_m())}

    def contains_m(self, xy_m):
        """True when the absolute position xy_m (metres) lies inside this file's image footprint."""
        return geom_contains(self.geometry(), xy_m)

    # --- pixel locations and the pairing of spectra with pixels ---

    def page_geometry(self, direction='forward'):
        """
        The recorded geometry of the forward (first) or reverse (second)
        topographic page from that page's own attributes: {'center_m',
        'angle_deg', 'xscale_m', 'yscale_m', 'nx', 'ny'}; the forward page
        when there is no reverse one.  Raises without a topographic page.
        """
        if not self.topo_pages:
            raise ValueError(f"{self.name} has no topography page.")
        idx = self.topo_pages[0] if direction == 'forward' or len(self.topo_pages) < 2 else self.topo_pages[1]
        a = self.f.pages[idx].attrs
        return {'center_m': (float(a['RHK_Xoffset']), float(a['RHK_Yoffset'])),
                'angle_deg': float(a.get('RHK_Angle', 0.0)),
                'xscale_m': float(a['RHK_Xscale']), 'yscale_m': float(a['RHK_Yscale']),
                'nx': int(a['RHK_Xsize']), 'ny': int(a['RHK_Ysize'])}

    @staticmethod
    def _rotation_of(angle_deg):
        """The rotation matrix of rotation() for any angle (a page's own RHK_Angle)."""
        th = np.deg2rad(angle_deg)
        return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    def pixel_xy_m(self, direction='forward'):
        """
        The recorded absolute location (metres) of every pixel of the
        forward or reverse topographic page, (ny, nx, 2) in the raw pixel
        order of topo_raw: pixel (i, j) at the page's scan centre plus the
        rotation of ((j - (nx - 1) / 2) xscale, (i - (ny - 1) / 2) yscale),
        the page's own signed scales, the convention of to_frame_m and
        frame_extent_m.  A pixel is a point, its centre; its drawn footprint
        belongs to the drawing.
        """
        g = self.page_geometry(direction)
        jj, ii = np.meshgrid(np.arange(g['nx'], dtype=np.float64), np.arange(g['ny'], dtype=np.float64))
        u = (jj - (g['nx'] - 1) / 2.0) * g['xscale_m']
        v = (ii - (g['ny'] - 1) / 2.0) * g['yscale_m']
        uv = np.stack([u, v], axis=-1)
        return uv @ self._rotation_of(g['angle_deg']).T + np.asarray(g['center_m'], dtype=np.float64)

    def spectrum_pixels(self, direction='forward', tol_px=None):
        """
        The topo pixel recorded at every spectrum's location, by location:
        each spectrum's recorded (x, y) is taken into the page's pixel frame
        (the inverse of pixel_xy_m) as a fractional (row, column) index; the
        spectrum is paired with the pixel at the nearest integer index when
        the fractional index lies within tol_px (PIXEL_PAIR_TOL_PX) of it on
        both axes and inside the page, else with none.  Returns (idx, dev):
        idx (N, 2) int, (row, column) or (-1, -1); dev (N,) the larger of the
        two fractional deviations in pixels (inf without positions).  No
        index of a spectrum is ever assumed: several sweeps recorded at one
        location (forward and backward sweeps, averages) all pair with the
        same pixel.
        """
        tol = float(PIXEL_PAIR_TOL_PX if tol_px is None else tol_px)
        xy = np.asarray(self.spec_xy_m, dtype=np.float64)
        N = len(xy)
        idx = np.full((N, 2), -1, dtype=int)
        dev = np.full(N, np.inf)
        if N == 0 or not self.topo_pages:
            return idx, dev
        g = self.page_geometry(direction)
        uv = (xy - np.asarray(g['center_m'], dtype=np.float64)) @ self._rotation_of(g['angle_deg'])
        if not g['xscale_m'] or not g['yscale_m']:
            return idx, dev                    # a page without a pixel size: no pixel can be paired
        fj = uv[:, 0] / g['xscale_m'] + (g['nx'] - 1) / 2.0
        fi = uv[:, 1] / g['yscale_m'] + (g['ny'] - 1) / 2.0
        i = np.rint(fi).astype(int)
        j = np.rint(fj).astype(int)
        dev = np.maximum(np.abs(fi - i), np.abs(fj - j))
        ok = (dev <= tol) & (i >= 0) & (i < g['ny']) & (j >= 0) & (j < g['nx'])
        idx[ok, 0] = i[ok]
        idx[ok, 1] = j[ok]
        return idx, dev

    def pairing_text(self, direction='forward', tol_px=None):
        """One console line: how many spectra found their pixel on the page and the largest deviation seen."""
        idx, dev = self.spectrum_pixels(direction, tol_px)
        if len(idx) == 0:
            return f"{self.name}: no spectrum positions; no pixel pairing."
        paired = idx[:, 0] >= 0
        finite = np.isfinite(dev)
        return (f"{self.name}: {int(paired.sum())} of {len(idx)} spectra paired with a {direction} pixel by location "
                f"(largest deviation {np.max(dev[finite]) if finite.any() else float('nan'):.2e} px, "
                f"tolerance {float(PIXEL_PAIR_TOL_PX if tol_px is None else tol_px):g} px)"
                + (f"; {int((~paired).sum())} spectra have no pixel at their location" if not paired.all() else ""))

    def _z_page(self):
        """The grid's embedded Topography spectra page (the Z recorded at every spectrum), or None.  Not called anywhere at present (topo_spectra_z, spectra_z and _self_check find the page themselves)."""
        pages = self.f.pages
        return next((p for p in pages if p.page_type == 'RHK_PAGE_INTERACTIVE_SPECTRA' and p.label == 'Topography'), None)

    def _z_per_spectrum(self, zpage):
        """
        The Topography channel's value for every spectrum of the file as shown (N,), NaN where never
        written: as stored for one sweep per pixel; for a view of several sweeps per pixel, the
        value recorded at each pixel's location (the page's own recorded locations matched to the
        pixels; without them, the first value of each pixel's columns).
        """
        z = np.array(zpage.data, dtype=np.float64)[0]
        unset = zpage.unset(z)
        z[unset] = np.nan
        if self.sweep_info is None:
            return z[:self.n_spectra]
        info = self.sweep_info
        locs = np.asarray(info['locs'], dtype=np.float64)
        zxy = sts.STS._page_locations(zpage)
        out = np.full(info['n_pix'], np.nan)
        if len(zxy) == len(z) and len(locs):
            from scipy.spatial import cKDTree
            dist, idx = cKDTree(locs).query(zxy)
            step = np.min(np.hypot(*np.diff(locs, axis=0).T)) if len(locs) > 1 else np.inf
            for c in range(len(z)):
                p = idx[c]
                if dist[c] <= 0.25 * step and not np.isfinite(out[p]) and np.isfinite(z[c]):
                    out[p] = z[c]
            return out
        if len(z) == info['n_pix'] * info['k']:
            return np.array([z[c[0]] for c in info['cols_of']])
        return z[:info['n_pix']] if len(z) >= info['n_pix'] else out

    def spectra_z(self):
        """
        The embedded Topography spectra channel of a grid, the Z recorded at
        every spectrum, (N,) metres in spectrum order (the loader's data);
        NaN where the value was never written (the page's RHK_Zoffset: not
        acquired) or the spectrum is not kept; None without the channel.
        """
        if self.kind != 'grid' or self.spectra is None:
            return None
        pages = self.f.pages
        tz = next((i for i, p in enumerate(pages) if p.page_type == 'RHK_PAGE_INTERACTIVE_SPECTRA'
                   and p.label == 'Topography'), None)
        if tz is None:
            return None
        try:
            zpage = pages[tz]
            z = self._z_per_spectrum(zpage)
            kept = np.asarray(self.acquired, dtype=bool)[:len(z)]
            if len(kept) == len(z):
                z[~kept] = np.nan
            return z
        except Exception:
            return None

    def topo_at_spectra(self, direction='forward', tol_px=None):
        """
        The topo height (metres) at every spectrum from the forward or
        reverse page through the location pairing (spectrum_pixels): NaN for
        a spectrum with no pixel at its location, a pixel never acquired, or
        a spectrum not kept.  (N,).
        """
        img = self.topo_raw(direction)
        idx, _ = self.spectrum_pixels(direction, tol_px)
        out = np.full(len(idx), np.nan)
        ok = idx[:, 0] >= 0
        out[ok] = img[idx[ok, 0], idx[ok, 1]]
        kept = np.asarray(self.acquired, dtype=bool)[:len(out)]
        if len(kept) == len(out):
            out[~kept] = np.nan
        return out

    # --- self check ---

    def _self_check(self, pages, types):
        """
        One console line (self_check_text) on the orientation of a grid: the
        embedded Topography spectra channel reshaped to (n_lines, n_fast)
        against the forward image page (topo_raw), each with a fitted plane
        removed, correlated as is and mirrored (fliplr, flipud, rot180) over
        the pixels finite in both.  'consistent with the raw pixel order'
        when as-is is the best match or within 0.01 of it (noting when a
        mirrored order fits equally well), else the best match named;
        'skipped' without the channel, an image page or a grid size, or
        when the shapes differ; 'failed' with the exception's text.
        """
        tz = next((i for i, t in enumerate(types) if t == 'RHK_PAGE_INTERACTIVE_SPECTRA'
                   and pages[i].label == 'Topography'), None)
        if tz is None or not self.topo_pages or not self.n_fast or not self.n_lines:
            return f"{self.name}: self check skipped (no Topography spectra channel, image page, or grid size)."
        try:
            zpage = pages[tz]
            z = self._z_per_spectrum(zpage)[:self.n_lines * self.n_fast]
            zg = z.reshape(self.n_lines, self.n_fast)                       # never written: NaN
            img = self.topo_raw('forward')
            if img.shape != zg.shape:
                return (f"{self.name}: self check skipped (image {img.shape} and grid "
                        f"{zg.shape} differ in shape).")

            def deplane(a):
                """The image minus the least-squares plane through its finite pixels."""
                ny, nx = a.shape
                J, I = np.meshgrid(np.arange(nx), np.arange(ny))
                ok = np.isfinite(a).ravel()
                A = np.c_[J.ravel(), I.ravel(), np.ones(nx * ny)]
                c, *_ = np.linalg.lstsq(A[ok], a.ravel()[ok], rcond=None)
                return a - (A @ c).reshape(ny, nx)

            d_img = deplane(img)
            cands = {'as-is': zg, 'fliplr': np.fliplr(zg), 'flipud': np.flipud(zg),
                     'rot180': np.flipud(np.fliplr(zg))}
            corr = {}
            for k, v in cands.items():
                d_z = deplane(v)
                ok = np.isfinite(d_img) & np.isfinite(d_z)
                corr[k] = (float(np.corrcoef(d_img[ok], d_z[ok])[0, 1]) if np.sum(ok) > 3 else np.nan)
            n_ok = int(np.sum(np.isfinite(d_img) & np.isfinite(zg)))
            best = max(corr, key=lambda k: -np.inf if np.isnan(corr[k]) else corr[k])
            txt = ", ".join(f"{k} {v:+.3f}" for k, v in corr.items())
            if best == 'as-is' or corr[best] - corr['as-is'] < 0.01:
                verdict = "consistent with the raw pixel order"
                if best != 'as-is' or any(abs(corr[k] - corr['as-is']) < 0.01 for k in corr if k != 'as-is'):
                    verdict += " (a mirrored order fits equally well: this image cannot tell them apart)"
            else:
                verdict = f"best match is '{best}', check the orientation"
            return (f"{self.name}: self check (embedded Topography channel vs image page, "
                    f"{n_ok} acquired pixels): {txt}; {verdict}.")
        except Exception as exc:
            return f"{self.name}: self check failed ({exc})."


def geom_contains(geom, xy_m):
    """True when the absolute position xy_m (metres) lies inside a topo footprint record (see Sm4File.geometry)."""
    if geom is None:
        return False
    th = np.deg2rad(geom['angle_deg'])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    u, v = (np.asarray(xy_m, dtype=np.float64) - np.asarray(geom['center_m'])) @ R
    l, r, b, t = geom['extent_m']
    return (min(l, r) <= u <= max(l, r)) and (min(b, t) <= v <= max(b, t))



if __name__ == '__main__':
    # One file: its kind from the headers alone, its page types, its image frame, then the file
    # loaded whole with its measurement information and the grid's self check (nothing is drawn).
    path = r'experiment_dir/file.sm4'
    types, geom, line_types = sm4_header_info(path)
    print(f"{os.path.basename(path)}: kind {kind_from_types(types, line_types)} (from the headers)")
    for i, (t, lt) in enumerate(zip(types, line_types)):
        print(f"  page {i}: {t}" + (f", {lt}" if lt and lt != 'RHK_LINE_NOT_A_LINE' else ''))
    print(f"  image frame: {geom}")
    f = Sm4File(path)
    print(f"loaded: kind {f.kind}, {len(f.topo_pages)} topography pages, image {f.nx} x {f.ny} px, "
          f"centre {f.center_m}, angle {f.angle_deg} deg, scales {f.xscale_m}, {f.yscale_m} m/px")
    for name, text in f.measurement_info():
        print(f"  {name}: {text}")
    if f.self_check_text:
        print(f.self_check_text)
