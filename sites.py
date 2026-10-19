"""
Sites of an experiment directory: the regions of every long-range-movement
set, which files were taken while pulsing the tip (a silver mound, high
absolute Z relief), which spectra file ends such a run with a clean tip, and
the navigation by experiment region.

Metrics (once over the directory at the first startup, cached in
sites_<stamp>.json beside the drift cache, per file with its modification
time; later startups read the cache and compute only files not in it).  The
cache records the loader format it was computed from
(sm4file.LOADER_FORMAT); one from another loader is out of date and left as
it is, and the metrics go into a new file beside it, sites_<stamp>_2.json
(the next number, sm4file.cache_family), with the flags set by hand carried
over from it; the newest up-to-date file is the one read and rewritten:
  topo    : the 2nd and 98th percentiles and the median of the absolute
            forward topography (metres)
  spectra : the first spectrum's absolute position and the ohmic test
            (grid and point files)

Regions (Sites.build_regions): within every segment (the topos between two
long-range moves, the drift index's segments) the topos are grouped by
footprint overlap at their recorded absolute positions, before any drift
correction (a correction never changes a region or a flag): any topo whose
footprint overlaps a member's joins the region, so a region is a connected
set of overlapping footprints.  A topo overlapping no member but whose
centre lies closer to a region's centre than that region's radius joins it
too, unless its own height range is significantly larger than the region's:
its own range (z98 - z2) lies above the split threshold below while the
region's total range does not.  A region's centre is the midpoint of the
farthest pair of its footprints' corners and its radius half their
distance.  A topo overlapping nothing and near no region is a region of
one.  Regions are numbered in time order of their first topo.

Classification (Sites.classify; TIP_PARAMS holds the ohmic settings):
  - per region the absolute Z range is max(z98) - min(z2) over its topos;
    the regions of the directory are split into pulsing and experiment by
    Otsu's threshold on the logarithm of their ranges (otsu_threshold; no
    height number in the code; the threshold and the ranges are reported on
    the console).  Every topo of a pulsing region is pulsing, every other
    topo experiment; one region alone cannot be split: everything is
    experiment.  No per-image rule: a zoomed-in image takes its region's
    class.
  - a grid or point takes the flag and the region of the most recent
    preceding topo (anywhere in the directory, header geometry) whose
    footprint contains its position; inside no footprint the ohmic test
    decides pulsing: R² at or above ohmic_r2 for the current's linear fit
    through its 0 V value and the calibrated dI/dV's standard deviation at
    most ohmic_flat times its mean (an ohmic, featureless spectrum); such a
    file is in no region.
  - a run of consecutive pulsing files whose last spectra file is followed
    by experiment files gets that last spectra file flagged clean.
  - phase rotations and noise spectra take the preceding file's flag and
    region.
  - every flag is the auto value until it is changed in the browser (the
    'site' radio); a change is written to the sites file.
Navigation (Sites.neighbour_region): 'next region' from a file is the
first spectra file ahead whose region is an experiment region other than
the file's own (a pulsing file has no region, so from it the first
experiment spectra file ahead); 'prev region' the same backwards, landing
on the first spectra file of that visit of the region (the walk is in
time order, so a region visited twice at separate times is two stops).
"""

if __package__ in (None, ''):
    # Run as a file (Spyder F5): make the package importable from wherever its folder sits.
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import json
from datetime import datetime
import numpy as np
from sm4_Browser import sm4file

TIP_PARAMS = {
    'ohmic_r2': 0.98,      # the current's linear fit through its 0 V value: R² at or above this ...
    'ohmic_flat': 0.15,    # ... and the calibrated dI/dV's standard deviation at most this fraction of its mean
    #                        make a spectrum ohmic (a spectra file inside no topo footprint is then pulsing)
}

FLAGS = ('pulsing', 'clean', 'experiment')
SITES_FORMAT = 1


def otsu_threshold(values):
    """
    Otsu's threshold of a list of values (the split between two classes
    that maximizes the between-class variance; every split between
    consecutive sorted values is tried).  Returns (threshold, separability)
    with the threshold halfway between the two classes' nearest members and
    the separability the between-class variance over the total variance
    (0 to 1); (None, 0.0) with fewer than two distinct values.
    """
    v = np.sort(np.asarray(values, dtype=np.float64))
    v = v[np.isfinite(v)]
    if len(v) < 2 or v[-1] == v[0]:
        return None, 0.0
    total = float(np.var(v))
    best, best_k = -1.0, None
    for k in range(1, len(v)):
        a, b = v[:k], v[k:]
        wa, wb = len(a) / len(v), len(b) / len(v)
        between = wa * wb * (float(np.mean(a)) - float(np.mean(b))) ** 2
        if between > best:
            best, best_k = between, k
    thr = 0.5 * (v[best_k - 1] + v[best_k])
    return float(thr), float(best / total) if total > 0 else 0.0


def topo_metrics(sm4):
    """The 2nd and 98th percentiles and the median of a file's absolute forward topography (metres, finite pixels)."""
    img = sm4.topo_raw('forward')
    ok = np.isfinite(img)
    if not ok.any():
        return {'z2': None, 'z98': None, 'median': None}
    z = img[ok]
    return {'z2': float(np.percentile(z, 2)), 'z98': float(np.percentile(z, 98)), 'median': float(np.median(z))}


def ohmic_test(sm4, params=TIP_PARAMS):
    """
    The ohmic test of a spectra file: the mean current over the kept
    spectra fitted linearly through its value at 0 V (a plain linear fit
    when 0 V is not in range), R² at or above ohmic_r2, and the mean
    calibrated dI/dV's standard deviation at most ohmic_flat times the
    absolute value of its mean.  Returns (ohmic, r2, flatness); a file
    without the Current channel or without kept spectra is not ohmic.
    """
    s = sm4.spectra
    if s is None or s.iv is None or not np.any(s.acquired):
        return False, None, None
    acq = np.asarray(s.acquired, dtype=bool)
    V = np.asarray(s.energy, dtype=np.float64)
    order = np.argsort(V)
    V = V[order]
    I = np.asarray(s.iv, dtype=np.float64)[order, :][:, acq].mean(axis=1)
    if V[0] <= 0.0 <= V[-1]:
        i0 = int(np.argmin(np.abs(V)))
        x, y = V - V[i0], I - I[i0]
        slope = float(np.sum(x * y) / np.sum(x * x)) if np.sum(x * x) > 0 else 0.0
        fit = I[i0] + slope * x
    else:
        p = np.polyfit(V, I, 1)
        fit = np.polyval(p, V)
    ss_res = float(np.sum((I - fit) ** 2))
    ss_tot = float(np.sum((I - np.mean(I)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    try:
        c = sm4.curves(None, None, False, 0.0, channel='didv', normalize=False, calibrate=True)
    except Exception:
        return False, r2, None
    d = np.asarray(c, dtype=np.float64)[order, :][:, acq].mean(axis=1)
    mean = float(np.mean(d))
    flat = float(np.std(d) / abs(mean)) if mean != 0 else np.inf
    ohmic = bool(r2 >= params['ohmic_r2'] and flat <= params['ohmic_flat'])
    return ohmic, r2, flat


def head_footprint(geom):
    """A topo's footprint record (center_m, angle_deg, extent_m: sm4file.geom_contains) from its header geometry."""
    if geom is None:
        return None
    nx, ny = int(geom['nx']), int(geom['ny'])
    xs, ys = float(geom['xscale_m']), float(geom['yscale_m'])
    u0, u1 = -(nx - 1) / 2.0 * xs, (nx - 1) / 2.0 * xs
    v0, v1 = -(ny - 1) / 2.0 * ys, (ny - 1) / 2.0 * ys
    return {'center_m': tuple(geom['center_m']), 'angle_deg': float(geom.get('angle_deg', 0.0)),
            'extent_m': [u0 - xs / 2.0, u1 + xs / 2.0, v0 - ys / 2.0, v1 + ys / 2.0]}


def footprint_corners(fp):
    """The four absolute corners (4, 2; metres) of a footprint record."""
    l, r, b, t = fp['extent_m']
    th = np.deg2rad(fp['angle_deg'])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    uv = np.array([[l, b], [r, b], [r, t], [l, t]], dtype=np.float64)
    return uv @ R.T + np.asarray(fp['center_m'], dtype=np.float64)


def footprints_overlap(ca, cb):
    """
    True when two footprints (their corners, (4, 2) each, convex) overlap:
    the separating-axis test over the four edge normals; touching edges count
    as overlapping.
    """
    for poly in (ca, cb):
        for k in range(4):
            edge = poly[(k + 1) % 4] - poly[k]
            n = np.array([-edge[1], edge[0]])
            pa = ca @ n
            pb = cb @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


class Sites:
    """
    The sites of a directory (the module docstring): the metrics cache, the
    regions of every segment, the flags (auto, with the user's overrides),
    and the navigation by experiment region.

    directory, files, kinds, head_geoms : as the browser has them (files in
        time order; kinds from the headers; the topos' header geometries)
    seg_first : topo path -> the first topo of its segment (Drift.seg_first)
    loader    : loader(path) -> Sm4File (the browser's; files loaded here for
        their metrics are released at once)
    """

    def __init__(self, directory, files, kinds, head_geoms, seg_first, loader, params=None):
        self.directory = directory
        self.files = list(files)
        self.kinds = dict(kinds)
        self.head_geoms = dict(head_geoms)
        self.seg_first = dict(seg_first)
        self._load = loader
        self.params = dict(TIP_PARAMS)
        self.params.update(params or {})
        y, mo, d, h, mi, sec, _ = sm4file.filename_stamp(self.files[0])
        self.base_path = os.path.join(directory, f"sites_{y:04d}_{mo:02d}_{d:02d}_{h:02d}_{mi:02d}_{sec:02d}.json")
        self.path = self.base_path   # the file read and written (_read: the newest up-to-date one, or a new one)
        self.topos = {}          # file name -> {'z2', 'z98', 'median', 'mtime'}
        self.spectra = {}        # file name -> {'xy_m': [x, y] or None, 'ohmic', 'r2', 'flat', 'mtime'}
        self.overrides = {}      # file name -> flag set by the user
        self._read()
        self._once_over()
        self.auto = {}           # path -> auto flag
        self.flags = {}          # path -> flag (the override when there is one)
        self.regions = []        # [{'index', 'segment', 'first', 'topos', 'span_m', 'center_m', 'radius_m', 'flag'}]
        self.region_index = {}   # path -> region index (1-based), None for a file in no region
        self.threshold_m = None
        self.separability = 0.0
        self.classify()

    # --- the cache ---

    def _read(self):
        """
        The newest up-to-date file of the cache's family (its loader format
        this loader's) is read and becomes self.path.  Without one, self.path
        is a new file beside the others (sm4file.cache_new_path; an
        out-of-date file is left as it is) and the flags set by hand are
        carried over from the newest readable out-of-date file.
        """
        stale = None
        for _, path in reversed(sm4file.cache_family(self.base_path)):
            try:
                with open(path) as fh:
                    data = json.load(fh)
            except (OSError, ValueError) as exc:
                print(f"sites cache {path}: not readable ({exc}).")
                continue
            if data.get('loader_format') == sm4file.LOADER_FORMAT:
                self.path = path
                self.topos = dict(data.get('topos', {}))
                self.spectra = dict(data.get('spectra', {}))
                self.overrides = {k: v for k, v in data.get('flags', {}).items() if v in FLAGS}
                print(f"sites cache loaded: {os.path.basename(self.path)} ({len(self.topos)} topos, "
                      f"{len(self.spectra)} spectra files, {len(self.overrides)} flags set by hand)")
                return
            if stale is None:
                stale = (path, data)
        self.path = sm4file.cache_new_path(self.base_path)
        if stale is not None:
            path, data = stale
            self.overrides = {k: v for k, v in data.get('flags', {}).items() if v in FLAGS}
            print(f"sites cache {os.path.basename(path)}: written from another loader (loader format "
                  f"{data.get('loader_format', 1)}, this loader {sm4file.LOADER_FORMAT}); left as it is. The "
                  f"metrics are computed again into {os.path.basename(self.path)}, with its "
                  f"{len(self.overrides)} flag(s) set by hand carried over.")

    def save(self):
        data = {'format': SITES_FORMAT, 'loader_format': sm4file.LOADER_FORMAT,
                'written': datetime.now().isoformat(timespec='seconds'),
                'topos': self.topos, 'spectra': self.spectra, 'flags': self.overrides}
        tmp = self.path + '.tmp'
        try:
            with open(tmp, 'w') as fh:
                json.dump(data, fh, indent=1)
            os.replace(tmp, self.path)
        except OSError as exc:
            print(f"sites cache {self.path}: not written ({exc}).")

    def _once_over(self):
        """Compute the metrics of every file not in the cache (or changed since), loading each once."""
        todo = []
        for p in self.files:
            name = os.path.basename(p)
            kind = self.kinds.get(p)
            try:
                mtime = os.path.getmtime(p)
            except OSError:
                continue
            if kind == 'topo':
                rec = self.topos.get(name)
                if rec is None or abs(float(rec.get('mtime', -1)) - mtime) > 1e-6:
                    todo.append((p, name, kind, mtime))
            elif kind in ('grid', 'point'):
                rec = self.spectra.get(name)
                if rec is None or abs(float(rec.get('mtime', -1)) - mtime) > 1e-6:
                    todo.append((p, name, kind, mtime))
        if not todo:
            return
        print(f"sites: once over {len(todo)} file(s) for their height metrics and positions (cached in "
              f"{os.path.basename(self.path)})")
        for n, (p, name, kind, mtime) in enumerate(todo, 1):
            try:
                f = self._load(p)
                if kind == 'topo':
                    rec = topo_metrics(f)
                else:
                    xy = f.spec_xy_m[0] if len(f.spec_xy_m) else None
                    ohmic, r2, flat = ohmic_test(f, self.params)
                    rec = {'xy_m': None if xy is None else [float(xy[0]), float(xy[1])], 'ohmic': bool(ohmic),
                           'r2': r2, 'flat': flat}
            except Exception as exc:
                print(f"sites: {name} not read ({exc}).")
                continue
            rec['mtime'] = mtime
            (self.topos if kind == 'topo' else self.spectra)[name] = rec
            if n % 10 == 0 or n == len(todo):
                print(f"sites: {n} / {len(todo)}")
        self.save()

    # --- the regions ---

    def _own_range(self, p):
        rec = self.topos.get(os.path.basename(p), {})
        if rec.get('z98') is None or rec.get('z2') is None:
            return None
        return float(rec['z98'] - rec['z2'])

    def _span(self, members):
        z2 = [self.topos[os.path.basename(p)]['z2'] for p in members
              if os.path.basename(p) in self.topos and self.topos[os.path.basename(p)].get('z2') is not None]
        z98 = [self.topos[os.path.basename(p)]['z98'] for p in members
               if os.path.basename(p) in self.topos and self.topos[os.path.basename(p)].get('z98') is not None]
        return (max(z98) - min(z2)) if z2 and z98 else None

    @staticmethod
    def _bounding(corners):
        """(centre, radius) of a set of points: the midpoint of the farthest pair and half their distance."""
        pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        if len(pts) == 1:
            return pts[0], 0.0
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        i, j = np.unravel_index(int(np.argmax(d)), d.shape)
        return 0.5 * (pts[i] + pts[j]), 0.5 * float(d[i, j])

    def build_regions(self, topo_seq, footprints):
        """
        The regions of every segment (the module docstring): connected sets
        of overlapping footprints, then the nearby topos absorbed under the
        radius rule with the range exclusion, the threshold for the
        exclusion found over the overlap regions (Otsu on the log of their
        ranges).  Sets self.regions (numbered in time order of their first
        topo) and self.region_index for the topos.
        """
        corners = {p: footprint_corners(footprints[p]) for p in topo_seq if footprints.get(p) is not None}
        regions = []
        for key in dict.fromkeys(self.seg_first.get(p, p) for p in topo_seq):
            members = [p for p in topo_seq if self.seg_first.get(p, p) == key and p in corners]
            # Connected sets of overlapping footprints (union-find).
            parent = {p: p for p in members}

            def find(p):
                while parent[p] != p:
                    parent[p] = parent[parent[p]]
                    p = parent[p]
                return p

            for a_i, a in enumerate(members):
                for b in members[a_i + 1:]:
                    if find(a) != find(b) and footprints_overlap(corners[a], corners[b]):
                        parent[find(b)] = find(a)
            groups = {}
            for p in members:
                groups.setdefault(find(p), []).append(p)
            segs = [{'segment': key, 'topos': list(g)} for g in groups.values()]
            # The nearby topos: a region of one whose centre lies within another region's radius joins it,
            # unless its own range lies above the split threshold while the region's total range does not.
            spans = [self._span(r['topos']) for r in segs]
            thr_log, _ = otsu_threshold(np.log([s for s in spans if s is not None and s > 0])) \
                if sum(1 for s in spans if s is not None and s > 0) >= 2 else (None, 0.0)
            thr = None if thr_log is None else float(np.exp(thr_log))
            changed = True
            while changed and len(segs) > 1:
                changed = False
                segs.sort(key=lambda r: -len(r['topos']))
                for r in segs:
                    if r.get('_gone'):
                        continue
                    c_r, rad_r = self._bounding(np.vstack([corners[p] for p in r['topos']]))
                    span_r = self._span(r['topos'])
                    for s in segs:
                        if s is r or s.get('_gone') or len(s['topos']) != 1:
                            continue
                        p = s['topos'][0]
                        c_p = np.mean(corners[p], axis=0)
                        if float(np.linalg.norm(c_p - c_r)) >= rad_r:
                            continue
                        own = self._own_range(p)
                        if (thr is not None and own is not None and span_r is not None
                                and own > thr and span_r <= thr):
                            continue                                   # significantly larger: stays out
                        r['topos'].append(p)
                        s['_gone'] = True
                        changed = True
                        c_r, rad_r = self._bounding(np.vstack([corners[q] for q in r['topos']]))
                        span_r = self._span(r['topos'])
                segs = [r for r in segs if not r.get('_gone')]
            regions.extend(segs)
        order = {p: i for i, p in enumerate(topo_seq)}
        for r in regions:
            r['topos'].sort(key=lambda p: order[p])
            r['first'] = r['topos'][0]
            r['span_m'] = self._span(r['topos'])
            c, rad = self._bounding(np.vstack([corners[p] for p in r['topos']]))
            r['center_m'], r['radius_m'] = (float(c[0]), float(c[1])), float(rad)
            r['flag'] = 'experiment'
        regions.sort(key=lambda r: order[r['first']])
        for k, r in enumerate(regions, 1):
            r['index'] = k
        self.regions = regions
        self.region_index = {p: r['index'] for r in regions for p in r['topos']}

    # --- classification ---

    def classify(self):
        """The regions, their ranges and the threshold, every file's auto flag and region, the overrides."""
        topo_seq = [p for p in self.files if self.kinds.get(p) == 'topo']
        footprints = {p: head_footprint(self.head_geoms.get(p)) for p in topo_seq}
        self.build_regions(topo_seq, footprints)
        spans = [r['span_m'] for r in self.regions if r['span_m'] is not None and r['span_m'] > 0]
        thr_log, sep = otsu_threshold(np.log(spans)) if len(spans) >= 2 else (None, 0.0)
        self.threshold_m = None if thr_log is None else float(np.exp(thr_log))
        self.separability = sep
        auto = {}
        for r in self.regions:
            pulsing = self.threshold_m is not None and r['span_m'] is not None and r['span_m'] > self.threshold_m
            r['flag'] = 'pulsing' if pulsing else 'experiment'
            for p in r['topos']:
                auto[p] = r['flag']
        for p in topo_seq:                                   # a topo without a footprint (no geometry): experiment
            auto.setdefault(p, 'experiment')
        # Spectra: the most recent preceding topo whose footprint contains the position; else the ohmic test.
        prev_flag, prev_region = 'experiment', None
        for i, p in enumerate(self.files):
            kind = self.kinds.get(p)
            if kind == 'topo':
                prev_flag, prev_region = auto[p], self.region_index.get(p)
                continue
            if kind in ('grid', 'point'):
                rec = self.spectra.get(os.path.basename(p), {})
                xy = rec.get('xy_m')
                flag, region = None, None
                if xy is not None:
                    for q in reversed(self.files[:i]):
                        if self.kinds.get(q) == 'topo' and sm4file.geom_contains(footprints.get(q), xy):
                            flag, region = auto[q], self.region_index.get(q)
                            break
                if flag is None:
                    flag = 'pulsing' if rec.get('ohmic') else 'experiment'
                auto[p] = flag
                self.region_index[p] = region
            else:
                auto[p] = prev_flag                      # phase rotations and noise spectra
                self.region_index[p] = prev_region
            prev_flag, prev_region = auto[p], self.region_index.get(p)
        # Clean: the last spectra file of a pulsing run followed by experiment files.
        i = 0
        n = len(self.files)
        while i < n:
            if auto[self.files[i]] != 'pulsing':
                i += 1
                continue
            j = i
            while j < n and auto[self.files[j]] == 'pulsing':
                j += 1
            if j < n and auto[self.files[j]] == 'experiment':
                for k in range(j - 1, i - 1, -1):
                    if self.kinds.get(self.files[k]) in ('grid', 'point'):
                        auto[self.files[k]] = 'clean'
                        break
            i = j
        self.auto = auto
        self.flags = {p: self.overrides.get(os.path.basename(p), auto[p]) for p in self.files}

    # --- queries and edits ---

    def flag(self, path):
        return self.flags.get(path, 'experiment')

    def region_of(self, path):
        """The region index (1-based) of a file, or None (a pulsing-by-ohmic spectra file, or no topos)."""
        return self.region_index.get(path)

    def region(self, k):
        """The region record of index k, or None."""
        for r in self.regions:
            if r['index'] == k:
                return r
        return None

    def region_text(self, path):
        """'region k (n topos, flag)' for the info panel, or 'none'."""
        k = self.region_of(path)
        r = self.region(k) if k is not None else None
        if r is None:
            return 'none'
        return f"region {k} ({len(r['topos'])} topos, {r['flag']})"

    def set_flag(self, path, flag):
        """A flag set by hand (the browser's radio): kept in the sites file; the flags derived again."""
        if flag not in FLAGS:
            raise ValueError(f"flag must be one of {FLAGS}")
        name = os.path.basename(path)
        if flag == self.auto.get(path):
            self.overrides.pop(name, None)
        else:
            self.overrides[name] = flag
        self.flags = {p: self.overrides.get(os.path.basename(p), self.auto[p]) for p in self.files}
        self.save()

    def _is_experiment_region(self, k):
        r = self.region(k) if k is not None else None
        return r is not None and r['flag'] == 'experiment'

    def neighbour_region(self, path, direction):
        """
        The file index of the next (+1) or previous (-1) stop at an
        experiment region (the module docstring), or None when there is none.
        """
        i = self.files.index(path)
        current = self.region_of(path)
        if direction > 0:
            for j in range(i + 1, len(self.files)):
                q = self.files[j]
                if self.kinds.get(q) not in ('grid', 'point'):
                    continue
                k = self.region_of(q)
                if k is not None and k != current and self._is_experiment_region(k):
                    return j
            return None
        for j in range(i - 1, -1, -1):
            q = self.files[j]
            if self.kinds.get(q) not in ('grid', 'point'):
                continue
            k = self.region_of(q)
            if k is not None and k != current and self._is_experiment_region(k):
                # Back to the first spectra file of this visit of the region.
                first = j
                for m in range(j - 1, -1, -1):
                    r = self.files[m]
                    if self.kinds.get(r) not in ('grid', 'point'):
                        continue
                    if self.region_of(r) == k:
                        first = m
                    else:
                        break
                return first
        return None

    def describe(self):
        """The console report: the regions' ranges, the threshold, the flag counts."""
        thr = ("none (one region: everything experiment)" if self.threshold_m is None
               else f"{self.threshold_m * 1e9:.2f} nm (Otsu on the log ranges, separability {self.separability:.2f})")
        n_seg = len(set(r['segment'] for r in self.regions))
        print(f"sites: {len(self.regions)} region(s) in {n_seg} segment(s); pulsing threshold {thr}")
        for r in self.regions:
            span = r['span_m']
            print(f"  region {r['index']}: {len(r['topos'])} topos from {os.path.basename(r['first'])}, Z range "
                  + (f"{span * 1e9:.2f} nm" if span is not None else "unknown")
                  + f", radius {r['radius_m'] * 1e9:.1f} nm, {r['flag']}")
        counts = {f: sum(1 for p in self.files if self.flags[p] == f) for f in FLAGS}
        print("sites: " + ", ".join(f"{counts[f]} {f}" for f in FLAGS)
              + (f"; {len(self.overrides)} flag(s) set by hand" if self.overrides else ""))


if __name__ == '__main__':
    # The sites of one directory on the console (the drift index's segments, as the browser builds them).
    from sm4_Browser.drift import DRIFT_PARAMS, Drift
    directory = r'experiment_dir'
    files = sm4file.list_sm4_files(directory)
    kinds, head_geoms = {}, {}
    for p in files:
        types, geom, line_types = sm4file.sm4_header_info(p)
        kinds[p] = sm4file.kind_from_types(types, line_types)
        if kinds[p] == 'topo' and geom is not None:
            head_geoms[p] = geom
    loader = sm4file.Sm4File
    drift = Drift(directory, files, kinds, head_geoms, dict(DRIFT_PARAMS), loader)
    sites = Sites(directory, files, kinds, head_geoms, drift.seg_first, loader)
    sites.describe()
    for p in files:
        print(f"  {os.path.basename(p)}: {kinds[p]}, {sites.flag(p)}, {sites.region_text(p)}")
