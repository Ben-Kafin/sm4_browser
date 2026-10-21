"""
The persistent layer of the browser's scan area: every file of the directory placed once and kept
(the MATLAB browser's model), shown or hidden by the 'show' mode as the current file changes, and
moved, never re-imaged, when the drift changes.

Display records (display_record): what the scan area needs of a file, kept for every file of the
directory after it was read once, so a file released from memory is still drawn:
  topo (and a grid's own topo): the processed image ('img', nm, float32) for one (pass, topo
      processors) key ('key'), whether every pixel of it is finite ('complete': only a complete
      image hides the images under it), the frame (geometry_record: extent, centre, angle, pixel
      scales, sizes, start time) and the pixel clock as two coefficients per pass (tcoef: t(i, j)
      - t(0, 0) = a_i i + a_j j, fitted to Sm4File.pixel_times, exact for a topo image);
  point / phase: the recorded position ('xy_m');
  every record: 'path', 'kind', 'name'.

Drift as a transform (affine_matrix): with one constant velocity v per file (drift.PairDrift), the
corrected position of pixel (i, j) is its recorded position - v (t(i, j) - t(0, 0)), and t is linear
in (i, j), so the whole correction is one affine map of the image's frame: the image item is drawn
with that map instead of being resampled.  A velocity change recomputes a 3 x 3 matrix and nothing
else; no image is reprocessed or saved.  For the 'average' pass the two passes' clocks are averaged
(their difference within a line is one line's return, far below a pixel's drift).

ScanLayer: one image item per topo (and per grid, drawn as its own topo while it is not the current
file) created when the file first becomes visible, z-ordered by time (later on top, every image
below the overlays); show modes:
  'up to current' : every file up to the current one (MATLAB's hide^),
  'all'           : every file of the directory,
  'last 25'       : the browser's loaded set (the current file and the 24 before it, plus the
                    extra topos loaded with them: the nearest earlier topo when those 25 hold
                    none, the topo covering a single point; browser._ensure_window), the
                    behaviour before the layer existed.
Files not yet read are read by the browser's worker processes (read_record; in the browser's
thread when they cannot start), their records written to the disk cache below and placed as they
arrive, the window staying usable meanwhile.
"""

import hashlib
import json
import os
import tempfile

import numpy as np

from sm4_Browser import qtplot as plt
from sm4_Browser.qtplot import Affine2D, QtWidgets

SHOW_MODES = ('up to current', 'all', 'last 25')
CACHE_FORMAT = 1
DEFAULT_SHOW_MODE = 'up to current'


def _pass_clock(f, direction):
    """(a_i, a_j): t(i, j) - t(0, 0) = a_i i + a_j j fitted to the pass's pixel times (s per row, s per column); None without times."""
    try:
        t = np.asarray(f.pixel_times(direction), dtype=np.float64)
    except Exception:
        return None
    ok = np.isfinite(t)
    if ok.sum() < 3:
        return None
    ii, jj = np.nonzero(ok)
    t0 = t[0, 0] if np.isfinite(t[0, 0]) else float(np.nanmin(t))
    A = np.column_stack([ii, jj]).astype(np.float64)
    coef, *_ = np.linalg.lstsq(A, t[ok] - t0, rcond=None)
    return float(coef[0]), float(coef[1])


def geometry_record(f):
    """The frame and pixel clock of a loaded file with an image page (affine_matrix's input), without an image; None without one."""
    if not (f.topo_pages and f.nx and f.ny and f.center_m is not None):
        return None
    fwd = _pass_clock(f, 'forward')
    rev = _pass_clock(f, 'reverse') if len(f.topo_pages) > 1 else fwd
    return {'path': f.path, 'kind': f.kind, 'name': f.name,
            'extent_m': [float(v) for v in f.frame_extent_m()],
            'center_m': (float(f.center_m[0]), float(f.center_m[1])),
            'angle_deg': float(f.angle_deg), 'xscale_m': float(f.xscale_m), 'yscale_m': float(f.yscale_m),
            'nx': int(f.nx), 'ny': int(f.ny), 'tcoef': {'forward': fwd, 'reverse': rev},
            'start_time': float(f.start_time)}


def display_record(f, direction, active):
    """
    The display record of a loaded Sm4File (the module docstring) for one pass ('forward',
    'reverse', 'average') and the topo processors `active`; None for a file with nothing to place.
    """
    from sm4_Browser.topo import processed_topo
    rec = {'path': f.path, 'kind': f.kind, 'name': f.name}
    placed = False
    if f.kind in ('topo', 'grid') and f.topo_pages and f.nx and f.ny:
        try:
            img = processed_topo(f, direction, active)
        except Exception as exc:
            print(f"{f.name}: not placed on the scan area ({exc}).")
            img = None
        if img is not None and np.isfinite(img).any():
            rec.update(geometry_record(f))
            rec.update({'img': (np.asarray(img, dtype=np.float64) * 1e9).astype(np.float32),
                        'complete': bool(np.isfinite(img).all()),
                        'key': (direction, tuple(bool(a) for a in active))})
            placed = True
    if f.kind in ('point', 'phase') and len(f.spec_xy_m):
        rec['xy_m'] = (float(f.spec_xy_m[0][0]), float(f.spec_xy_m[0][1]))
        placed = True
    return rec if placed else None


# --- the disk cache of display records (processed images, no drift: the drift is the transform) ---

def cache_dir(directory, files):
    """
    scan_cache_<stamp>/ in the directory, beside the settings and computed stores (caches.py):
    the stamp is every field of the first file's name stamp (sm4file.filename_stamp: year to
    millisecond, each at least two digits), so it also carries the milliseconds the stores'
    stamp leaves out; 'empty' without files.
    """
    from sm4_Browser import sm4file
    stamp = '_'.join(f"{v:02d}" for v in sm4file.filename_stamp(files[0])) if files else 'empty'
    return os.path.join(directory, f"scan_cache_{stamp}")


def _cache_name(path, direction, active):
    """The cache file of one file's record, <stem>_<12 hex>.npz: the hash over its name, size and modification time, the pass, the processors (0/1 flags), the loader format and CACHE_FORMAT; None when the file cannot be stat'ed."""
    from sm4_Browser import sm4file
    try:
        st = os.stat(path)
        sig = f"{st.st_size}_{int(st.st_mtime * 1000)}"
    except OSError:
        return None
    flags = ''.join('1' if a else '0' for a in active)
    h = hashlib.sha1(f"{os.path.basename(path)}|{sig}|{direction}|{flags}|{sm4file.LOADER_FORMAT}|{CACHE_FORMAT}".encode())
    return f"{os.path.splitext(os.path.basename(path))[0]}_{h.hexdigest()[:12]}.npz"


def cache_load(cdir, path, direction, active):
    """A cached record (the file unchanged since), or None."""
    name = _cache_name(path, direction, active)
    if name is None:
        return None
    fp = os.path.join(cdir, name)
    if not os.path.isfile(fp):
        return None
    try:
        with np.load(fp, allow_pickle=False) as z:
            meta = json.loads(str(z['meta']))
            img = z['img'] if 'img' in z.files else None
    except Exception:
        return None
    if meta.get('format') != CACHE_FORMAT:
        return None
    rec = meta['rec']
    rec['path'] = path
    if rec.get('center_m') is not None:
        rec['center_m'] = tuple(rec['center_m'])
    if rec.get('xy_m') is not None:
        rec['xy_m'] = tuple(rec['xy_m'])
    if rec.get('key') is not None:
        rec['key'] = (rec['key'][0], tuple(bool(a) for a in rec['key'][1]))
    if rec.get('tcoef'):
        rec['tcoef'] = {k: (None if v is None else tuple(v)) for k, v in rec['tcoef'].items()}
    if img is not None:
        rec['img'] = img
    return rec


def cache_store(cdir, path, direction, active, rec):
    """A record written to the cache (atomically); errors are reported, never raised."""
    name = _cache_name(path, direction, active)
    if name is None or rec is None:
        return
    try:
        os.makedirs(cdir, exist_ok=True)
        meta = {k: v for k, v in rec.items() if k not in ('img', 'path')}
        if meta.get('key') is not None:
            meta['key'] = [meta['key'][0], [bool(a) for a in meta['key'][1]]]
        arrays = {'meta': np.array(json.dumps({'format': CACHE_FORMAT, 'rec': meta}))}
        if rec.get('img') is not None:
            arrays['img'] = np.asarray(rec['img'], dtype=np.float32)
        fd, tmp = tempfile.mkstemp(dir=cdir, prefix='.tmp_', suffix='.npz')
        with os.fdopen(fd, 'wb') as fh:
            np.savez(fh, **arrays)
        os.replace(tmp, os.path.join(cdir, name))
    except Exception as exc:
        print(f"scan cache: {os.path.basename(path)} not written ({exc}).")


def read_record(path, direction, active):
    """
    A display record read from the file itself (the worker processes' task: picklable in and out;
    the file's console lines are kept out of the browser's console).  Returns (path, kind, geometry,
    record or None, error text or None).
    """
    import contextlib
    import io
    from sm4_Browser.sm4file import Sm4File
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            f = Sm4File(path)
            rec = display_record(f, direction, active)
        return path, f.kind, (f.geometry() if f.kind == 'topo' else None), rec, None
    except Exception as exc:
        return path, None, None, None, str(exc)


def affine_matrix(rec, k, direction, model):
    """
    The 3 x 3 matrix that takes the image's frame coordinates (drawing units, the extent times k)
    to absolute drawing units: rotation and translation to the recorded place, plus the drift
    correction -v (t(i, j) - t(0, 0)) of the file's velocity (model.v, m/s; None: no correction),
    so pixel (0, 0) stays at its recorded place and the rest of the image is sheared by the clock
    (tcoef) of `direction` ('average': the mean of the two passes' coefficients, none without
    both; a pass without a clock takes the forward one; no clock: no correction).  `k` is the
    file's drawing factor (calibrated nm per recorded metre).
    """
    th = np.deg2rad(rec['angle_deg'])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    M = R.copy()
    off = np.asarray(rec['center_m'], dtype=np.float64).copy()
    if model is not None:
        tc = rec['tcoef']
        if direction == 'average':
            a, b = tc.get('forward'), tc.get('reverse')
            coef = None if (a is None or b is None) else (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
        else:
            coef = tc.get(direction) or tc.get('forward')
        if coef is not None:
            a_i, a_j = coef
            v = np.asarray(model.v, dtype=np.float64)
            M[:, 0] = R[:, 0] - v * a_j / rec['xscale_m']
            M[:, 1] = R[:, 1] - v * a_i / rec['yscale_m']
            off = off - v * (a_i * (rec['ny'] - 1) / 2.0 + a_j * (rec['nx'] - 1) / 2.0)
    out = np.eye(3)
    out[:2, :2] = M
    out[:2, 2] = off * k
    return out


def record_corners(rec, k, direction, model):
    """The four corners (drawing units at `k`) of a placed image under its drift correction, (4, 2), in the order (l, b), (r, b), (r, t), (l, t) of the frame."""
    l, r, b, t = [v * k for v in rec['extent_m']]
    uv = np.array([[l, b], [r, b], [r, t], [l, t]])
    m = affine_matrix(rec, k, direction, model)
    return uv @ m[:2, :2].T + m[:2, 2]


def _cross2d(a, b):
    """z component of a x b for 2-D vectors (NumPy 2.5 dropped the 2-D np.cross)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _covers(quads, order, complete):
    """
    For every image, the later complete images whose quadrilateral contains it entirely:
    {path: [paths in time order]} (occlusion: an image so covered is not painted).  quads
    {path: (4, 2)} the corners in drawing units under the drift correction, each file at its own
    drawing factor (_occlusion passes k_of(path)), so the test is the scan area's own geometry
    when files carry different scale factors; order {path: rank},
    complete {path: bool}.  A bounding-box test first, then every corner of the candidate inside
    every edge of the cover, within a small tolerance.
    """
    paths = sorted(quads, key=lambda p: order.get(p, 0))
    if len(paths) < 2:
        return {}
    Q = np.array([quads[p] for p in paths])                     # (n, 4, 2)
    lo, hi = Q.min(axis=1), Q.max(axis=1)
    comp = np.array([bool(complete.get(p)) for p in paths])
    E = np.roll(Q, -1, axis=1) - Q                              # edges (n, 4, 2)
    # The orientation of each quad (its vertices run one way round).
    sgn = np.sign(_cross2d(E[:, 0], E[:, 1]))
    out = {}
    tol = 1e-12
    for a in range(len(paths) - 1):
        later = np.arange(a + 1, len(paths))
        cand = later[comp[later] & np.all(lo[later] <= lo[a] + tol, axis=1) & np.all(hi[later] >= hi[a] - tol, axis=1)]
        if not len(cand):
            continue
        P = Q[a]                                                # (4, 2) corners of the covered candidate
        d = P[None, None, :, :] - Q[cand][:, :, None, :]        # (m, 4 edges, 4 points, 2)
        cr = _cross2d(E[cand][:, :, None, :], d)                # (m, 4, 4)
        inside = np.all(cr * sgn[cand][:, None, None] >= -tol * np.abs(cr).max(initial=1.0), axis=(1, 2))
        if inside.any():
            out[paths[a]] = [paths[j] for j in cand[inside]]
    return out


class ScanLayer:
    """
    The persistent images of the scan area (the module docstring).  The browser hands it the
    records (add), tells it the current state (update: visible paths, the current file, the
    drawing factor, one number or per file, each file's pass and drift model), and asks it which
    artists are its own (artists) so that its redraw of the overlays leaves them.
    """

    Z_BOTTOM, Z_TOP = -1.0, -0.01        # every image between these (later files on top), overlays above
    device_cache = False                 # Qt's device-coordinate cache on every image item (see update)

    def __init__(self, ax, files, cmap):
        self.ax = ax
        self.files = list(files)
        self.rank = {p: i for i, p in enumerate(self.files)}
        self.cmap = cmap
        self.records = {}                # path -> display record
        self.items = {}                  # path -> {'im', 'key', 'mat'}
        self._reported = set()
        self._occ_state = None           # the drift state the occlusion was computed for
        self._occ_dirty = True           # records added since
        self.covers = {}                 # path -> the later complete images that hide it (_covers)

    # --- records ---

    def add(self, rec):
        """A display record (replaces the file's earlier one; its image item is rebuilt on the next update)."""
        if rec is None:
            return
        old = self.records.get(rec['path'])
        self.records[rec['path']] = rec
        if old is not None and old.get('key') != rec.get('key'):
            self._drop_item(rec['path'])
        if 'img' in rec:
            self._occ_dirty = True

    def has(self, path, key=None):
        """True when the file has a record; with `key`, when that record is a point record or an image record made for that (pass, processors) key."""
        r = self.records.get(path)
        return r is not None and (key is None or 'img' not in r or r.get('key') == key)

    def artists(self):
        """The image items of the layer (the browser's overlay redraw leaves them)."""
        return [it['im'] for it in self.items.values()]

    def _drop_item(self, path):
        """The file's image item removed from the axes (the record kept)."""
        it = self.items.pop(path, None)
        if it is not None:
            try:
                it['im'].remove()
            except Exception:
                pass

    def clear_items(self):
        """Every image item removed (the records kept): a new axes, or a full rebuild."""
        for p in list(self.items):
            self._drop_item(p)

    # --- the state ---

    def _occlusion(self, state, direction_of, model_of, wanted, k_of):
        """
        The covers map (_covers) for the drift state `state` (a token that carries the scale
        timeline too, browser._drift_state), recomputed when it or the image records changed, over
        the records whose key is the wanted one; every record's corners in drawing units at its own
        file's factor k_of(path), so an image is hidden only when a later complete image covers it
        as the two are drawn.
        """
        if not self._occ_dirty and state == self._occ_state:
            return
        quads, complete = {}, {}
        for p, rec in self.records.items():
            if 'img' not in rec or (wanted is not None and rec.get('key') != wanted(p)):
                continue
            quads[p] = record_corners(rec, k_of(p), direction_of(p), model_of(p, rec))
            complete[p] = rec.get('complete', False)
        self.covers = _covers(quads, self.rank, complete)
        self._occ_state = state
        self._occ_dirty = False

    def update(self, visible, current_path, k, direction_of, model_of, wanted=None, state=None):
        """
        Show the image of every visible path that has an image record (created on first need),
        hide the rest; the current grid is not drawn as an image (its stripes are the overlay);
        every shown image placed by affine_matrix with its file's pass and drift model.  `k` is
        the drawing factor (calibrated nm per recorded metre, 1e9 / scale factor): one number for
        every file, or k(path), each file at the scale in force at its own time (the browser's
        scale timeline, scales.Scales), so files of different temperature regimes sit at their
        true sizes side by side.  With `wanted(path)` given, a record made for another (pass,
        processors) key is not shown (it waits for its new record).  `direction_of(path)` gives a
        file's pass, `model_of(path, record)` its drift.PairDrift (None: no correction); `state`
        is a token of everything the placed positions depend on (the browser's _drift_state):
        with one, the occlusion is recomputed when it changed and an image wholly under a later
        complete visible image is hidden; with None (files still arriving) no image is hidden.
        Returns the corners (drawing units) of the shown images, for the view's extent.
        """
        k_of = k if callable(k) else (lambda p, kk=float(k): kk)
        n = max(len(self.files), 1)
        corners = []
        shown = set()
        if state is not None:
            self._occlusion(state, direction_of, model_of, wanted, k_of)
        vis_set = set(visible)
        vis_set.discard(current_path if self.records.get(current_path, {}).get('kind') == 'grid' else None)
        for p in visible:
            rec = self.records.get(p)
            if rec is None or 'img' not in rec:
                continue
            if p == current_path and rec['kind'] == 'grid':
                continue
            if wanted is not None and rec.get('key') != wanted(p):
                continue
            if state is not None and any(q in vis_set for q in self.covers.get(p, ())):
                continue                                         # wholly under a later complete image
            direction = direction_of(p)
            model = model_of(p, rec)
            k = k_of(p)
            mat = affine_matrix(rec, k, direction, model)
            it = self.items.get(p)
            if it is None or it['key'] != rec['key']:
                self._drop_item(p)
                l, r, b, t = [v * k for v in rec['extent_m']]
                z = self.Z_BOTTOM + (self.Z_TOP - self.Z_BOTTOM) * self.rank.get(p, 0) / n
                im = self.ax.imshow(rec['img'], origin='lower', extent=[l, r, b, t], interpolation='none',
                                    cmap=self.cmap, zorder=z)
                if self.device_cache:
                    # Painted once into a screen-resolution pixmap and blitted from then on: stepping and
                    # overlay redraws do not redraw the transformed images (a pan or zoom refreshes it).
                    im.item.setCacheMode(QtWidgets.QGraphicsItem.CacheMode.DeviceCoordinateCache)
                it = {'im': im, 'key': rec['key'], 'mat': None, 'k': k}
                self.items[p] = it
            if it.get('k') != k:
                l, r, b, t = [v * k for v in rec['extent_m']]
                it['im'].set_extent([l, r, b, t])
                it['k'] = k
            if it['mat'] is None or not np.array_equal(it['mat'], mat):
                it['im'].set_transform(Affine2D(mat) + self.ax.transData)
                it['mat'] = mat
            it['im'].set_visible(True)
            shown.add(p)
            l, r, b, t = [v * k for v in rec['extent_m']]
            uv = np.array([[l, b], [r, b], [r, t], [l, t]])
            corners.append(uv @ mat[:2, :2].T + mat[:2, 2])
        for p, it in self.items.items():
            if p not in shown:
                it['im'].set_visible(False)
        return corners

    def points(self, visible, current_path):
        """(path, xy_m) of every visible point / phase record other than the current file, in time order."""
        out = []
        for p in visible:
            if p == current_path:
                continue
            rec = self.records.get(p)
            if rec is not None and 'xy_m' in rec:
                out.append((p, rec['xy_m']))
        return out
