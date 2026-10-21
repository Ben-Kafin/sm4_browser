"""
The feature checks of the package on a synthetic experiment (checks/synth.py), offscreen.

    PYTHONDONTWRITEBYTECODE=1 python -m sm4_Browser.checks.smoke_browser [names]

run from the parent directory of a folder named sm4_Browser (the package is imported under that
name; the repository's folder is sm4_browser, so link or copy it to that name first).  The runner
configures Qt's offscreen platform itself with a 4200 x 2600 screen (_offscreen_platform) when
QT_QPA_PLATFORM is unset or plain 'offscreen'; any other setting is left alone.  Every argument
given is a substring: only the checks whose name contains one of them run (e.g. 'K1', 'drift').

Builds the standard experiment (synth.standard: topo, topo, grid, point, topo) in a temporary
directory, opens the Browser once at scale factor 1 (the event loop is not entered: plt.show
returns at once), then runs every @check in order, one line 'PASS  <name>' or 'FAIL  <name>' with
the traceback, and exits non-zero when any failed.  One @check per feature, named by family:

  (first)  step through every file.
  D   the scan area's view: D1 double-click centring at the same zoom, D2 the R key saves the view
      and space returns to it, D3 Home (toolbar and key) returns to the full extent and the scroll
      zooms.
  E   marker groups down to zero and back in the Line window, the Map window and align.marker_map.
  A   the scan layer: A2 images persist across steps and under the show modes, the occlusion rule;
      A3 the drift transform equals the per-pixel resampling, a drift change only moves the images;
      A4 the show modes.
  B   B1 the step filter: kinds and minimum duration.
  C   C1 point colours: grey before the current topo, blue after.
  G   the windows: G1-G5 the drift window (marker pairs, partners seeded both ways, a typed
      velocity, arrows), G5 the stripe window (handles, a drag along the line, the sets per setting,
      the cache), G6 the pass order and the Line window's rows under the placement, G1 the Line
      window's and the line average's crop cursors with 'crop' / 'uncrop', G2 the Map window under the drift, G3 the whole map into
      an average and the figure composer, G4 the drift window's pair stepping, G7 the drift window's
      frames, titles and one centred view.
  H   stripe matching states in the browser and the Topo window.
  F   F1-F3 the E-markers and the energy slice (Line window, line average).
  T   T1 the title rule (a file name shown exactly as named).
  I   item 1: I1 one text-size rule for every window kind at its design size, 2x and 0.5x; I2 short
      labels and stamp titles in portrait and back; I3 the browser window's rule and native controls.
  J   item 2: J1 the settings and computed stores by origin (a setting never changes the computed
      file; deleted, it is rebuilt the same), J2 the carry-over of the earlier caches, J3 the
      hand-set flags feeding the region, clean rule and navigation.
  K   K1 the scale timeline: the first entry covers everything, later ones from their file's time;
      every file and window at its own scale; drift readouts calibrated with the nominal; records
      made with another scale are noted; a marker record stores its scale.
  L   L1 the section path: editing with frozen navigation, add / drag / remove, off keeps the
      points; measure, set scale (two-way, apply), the cross section along the path, clear.
  M   item 4: M1 'anchor to topo' (the browser's toggle and the Topo window's: the file moved by
      the reported rigid shift, off restores it); M2 the Line window opens with 'calibrate' and
      'normalize' off; M3 the console strings and the filter box label; M4 the occlusion tested
      as drawn at each file's scale; M5 tracked pairs with the files' signatures, tracked again
      after a change, written when tracked outside the build; M7 the Map window under the stripe
      matching alone (frame, lines, title, one label per grid cell); M8 RowHeatmap with a short
      shear; M6 the fit history keeps norm_range and the barrier bound is one rule; M9 qtplot
      (a title's size kept, a decimal is not a file name, a numeric aspect in the datalim
      widening); M10 records and helpers (an old fit family listed once, a member's default row,
      the Hampel filter's dtype, zero_crossing's tag).

The checks share one browser and tidy up after themselves (windows closed, settings removed), so
a failed check can leave state behind for the ones after it.  smoke_sweeps.py reuses make_browser,
goto_kind, open_line and open_map.
"""
import os
import sys
import tempfile
import time
import traceback



def _offscreen_platform():
    """
    Qt's offscreen platform with a screen large enough for every window at twice its design size
    (its built-in screen is 800 x 800, which caps the windows): the plugin takes a JSON config file
    with a 'screens' list (offscreen:configfile=<path>).  Used when QT_QPA_PLATFORM is unset or
    plain 'offscreen'; any other setting is left alone.
    """
    cur = os.environ.get('QT_QPA_PLATFORM', '')
    if cur not in ('', 'offscreen'):
        return
    import json
    path = os.path.join(tempfile.gettempdir(), 'sm4_browser_offscreen_screen.json')
    with open(path, 'w') as fh:
        json.dump({'screens': [{'name': 'checks', 'x': 0, 'y': 0, 'width': 4200, 'height': 2600,
                                'logicalDpi': 96, 'logicalBaseDpi': 96, 'dpr': 1.0}]}, fh)
    os.environ['QT_QPA_PLATFORM'] = f"offscreen:configfile={path}"


_offscreen_platform()
if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from sm4_Browser import qtplot as plt
from sm4_Browser.checks import synth

CHECKS = []


def check(name):
    """Decorator: register fn(browser, experiment) as the check `name`, run in registration order."""
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def make_browser(directory, **kw):
    """A Browser on `directory` at scale factor 1 with plt.show returning at once (no event loop), its startup events processed."""
    from sm4_Browser.browser import Browser
    plt._RUNNING[0] = True              # plt.show() returns at once (no event loop in a test)
    b = Browser(directory, scalefactor=1.0, **kw)
    plt._app().processEvents()
    return b


def run(selected=None):
    """
    Build the standard experiment in a temporary directory, open the browser on it and run every
    registered check (those whose name contains one of `selected` when given), printing PASS or
    FAIL with the traceback per check.  Returns True when every check run passed.
    """
    tmp = tempfile.mkdtemp(prefix='sm4synth_')
    exp = synth.standard(tmp)
    synth.install(exp)
    t0 = time.perf_counter()
    b = make_browser(tmp)
    print(f"browser opened on {len(b.files)} synthetic files in {time.perf_counter() - t0:.2f} s")
    ok = True
    for name, fn in CHECKS:
        if selected and not any(s in name for s in selected):
            continue
        try:
            fn(b, exp)
            plt._app().processEvents()
            print(f"PASS  {name}")
        except Exception:
            ok = False
            print(f"FAIL  {name}")
            traceback.print_exc()
    return ok


@check("step through every file")
def _step_all(b, exp):
    for i in range(len(b.files)):
        b._goto(i)
    b._goto(0)


def _axes_centre_global(canvas, ax):
    """The centre of an axes as a global screen point (for the cursor) and in the canvas's pixels."""
    from sm4_Browser.qtplot import QtCore
    x0, y0, w, h = ax._px_rect()
    return canvas.viewport().mapToGlobal(QtCore.QPoint(int(x0 + w / 2), int(y0 + h / 2))), (x0 + w / 2, y0 + h / 2)


@check("D1 double-click centres at the same zoom")
def _dbl(b, exp):
    from sm4_Browser.qtplot import QtCore, QtGui, MouseEvent
    ax, v = b.ax_scan, b.view
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    w0 = x1 - x0
    tx, ty = x0 + 0.8 * (x1 - x0), y0 + 0.3 * (y1 - y0)
    e = MouseEvent('button_press_event', b.canvas_scan, 0, 0, button=1, dblclick=True)
    e.inaxes, e.xdata, e.ydata = ax, tx, ty
    b.canvas_scan._fire('button_press_event', e)
    nx0, nx1 = ax.get_xlim(); ny0, ny1 = ax.get_ylim()
    assert abs((nx1 - nx0) - w0) < 1e-6 * max(w0, 1), (nx1 - nx0, w0)
    assert abs(0.5 * (nx0 + nx1) - tx) < 1e-6 * max(w0, 1) and abs(0.5 * (ny0 + ny1) - ty) < 1e-6 * max(w0, 1)


@check("D2 R saves the view, space returns to it (real Qt key events)")
def _save_restore(b, exp):
    from sm4_Browser.qtplot import QtCore, QtGui, Qt
    from PySide6.QtTest import QTest
    ax, c = b.ax_scan, b.canvas_scan
    g, local = _axes_centre_global(c, ax)
    QtGui.QCursor.setPos(g)
    c.setFocus()
    plt._app().processEvents()
    saved = (ax.get_xlim(), ax.get_ylim())
    QTest.keyClick(c.viewport() if False else c, Qt.Key.Key_R)
    assert b.view.saved is not None, "R did not save (cursor position not over the axes in this platform?)"
    b.view.centre_on(saved[0][0] - 50.0, saved[1][0] - 50.0)
    QTest.keyClick(c, Qt.Key.Key_Space)
    (a0, a1), (c0, c1) = ax.get_xlim(), ax.get_ylim()
    assert abs(a0 - saved[0][0]) < 1e-6 and abs(c1 - saved[1][1]) < 1e-6, ((a0, a1), saved)


@check("D3 Home (toolbar) and Home key return to the full extent; scroll still zooms")
def _home(b, exp):
    from sm4_Browser.qtplot import Qt, MouseEvent
    from PySide6.QtTest import QTest
    ax, v = b.ax_scan, b.view
    full = v.full
    e = MouseEvent('scroll_event', b.canvas_scan, 0, 0, button='up', step=1)
    e.inaxes, e.xdata, e.ydata = ax, 0.5 * (full[0] + full[1]), 0.5 * (full[2] + full[3])
    b.canvas_scan._fire('scroll_event', e)
    zx0, zx1 = ax.get_xlim()
    assert (zx1 - zx0) < (full[1] - full[0]) * 0.99, "scroll did not zoom"
    b.toolbar_scan._home()
    hx0, hx1 = ax.get_xlim()
    assert abs((hx1 - hx0) - (full[1] - full[0])) < 1e-6 * (full[1] - full[0])
    b.view.centre_on(0.0, 0.0)
    QTest.keyClick(b.canvas_scan, Qt.Key.Key_Home)
    hx0, hx1 = ax.get_xlim()
    assert abs((hx1 - hx0) - (full[1] - full[0])) < 1e-6 * (full[1] - full[0])


def goto_kind(b, kind):
    """Step the browser to the first file of `kind` ('topo', 'grid', 'point', ...) and return it (b.current)."""
    i = next(i for i, p in enumerate(b.files) if b.kinds[p] == kind)
    b._goto(i)
    return b.current


def open_line(b):
    """Open a Line window on the first grid (the browser's 'Line' button) and return it."""
    goto_kind(b, 'grid')
    n = len(b.children)
    b._on_line()
    assert len(b.children) == n + 1, "Line window did not open"
    return b.children[-1]


def open_map(b):
    """Open a Map window on the first grid (the browser's 'Map' button) and return it."""
    goto_kind(b, 'grid')
    n = len(b.children)
    b._on_map()
    assert len(b.children) == n + 1, "Map window did not open"
    return b.children[-1]


@check("E1 Line window: groups and markers down to 0 and back")
def _line_zero(b, exp):
    from sm4_Browser import widgets
    assert widgets.SPECS['Markers']['vmin'] == 0 and widgets.SPECS['avgs']['vmin'] == 0
    w = open_line(b)
    w._int_on_group_slider(0, 0)                      # the group's markers to 0
    assert w._int_group_sizes == [0] and w._int_marker_pos == []
    w._int_on_add_new(0)                             # nothing to add: a message, no window
    w._int_on_avgs_change(0)                         # no group at all
    assert w._int_group_sizes == [] and w._int_marker_pos == []
    w._int_stacks.chk_average.set(True); w._redraw_curves()
    w._int_on_avgs_change(2)                         # back: default size
    assert w._int_group_sizes == [2, 2] and len(w._int_marker_pos) == 4, w._int_group_sizes
    w._int_on_group_slider(1, 0)
    assert w._int_group_sizes == [2, 0]
    w._int_stacks.chk_stack.set(True); w._redraw_curves()
    w._int_stacks.chk_substack.set(True); w._redraw_curves()
    w._int_on_group_slider(1, 3)
    assert w._int_group_sizes == [2, 3] and len(w._int_marker_pos) == 5


@check("E1 Map window: groups and markers down to 0 and back")
def _map_zero(b, exp):
    m = open_map(b)
    m._on_group_slider(0, 0)
    assert m.group_sizes == [0] and m.marker_px == []
    m._on_avgs_change(0)
    assert m.group_sizes == [] and m.marker_px == []
    m.stacks.chk_average.set(True)
    m._redraw_markers()
    m._on_avgs_change(1)
    assert m.group_sizes == [2] and len(m.marker_px) == 2


@check("E1 align.marker_map with no markers keeps the DTW map")
def _align_zero(b, exp):
    from sm4_Browser import align
    pos = np.linspace(0, 10, 11)
    out = align.marker_map(pos, pos * 1.1, [], [])
    assert np.allclose(out, pos * 1.1)


def settle(b, max_turns=2000):
    """Let the background placing finish (zero timers run in processEvents)."""
    for _ in range(max_turns):
        plt._app().processEvents()
        if not b._bg_running and not b._bg_queue:
            break


@check("A3 drift transform equals the per-pixel resampling (corrected_topo) inside the frame")
def _affine_equals_resample(b, exp):
    from scipy.ndimage import map_coordinates
    from sm4_Browser.scanlayer import display_record, affine_matrix
    from sm4_Browser.topo import processed_topo
    t = next(f for f in b.loaded.values() if f.kind == 'topo')
    for direction in ('forward', 'reverse'):
        model, _ = b.drift.model_for_topo(t.path, True, True)
        assert model is not None, "the synthetic pair should give a velocity"
        ref = b.drift.corrected_topo(t, direction, b.topo_active, b.loaded, True, True) * 1e9   # nm, own grid
        rec = display_record(t, direction, b.topo_active)
        k = 1.0
        M = affine_matrix(rec, k, direction, model)
        # Every output pixel q (absolute, metres): its frame coordinates through the inverse of M, the
        # processed image sampled there (bilinear) = what the transformed item shows at q.
        q = t.pixel_xy_m(direction).reshape(-1, 2)
        Minv = np.linalg.inv(M)
        uv = (np.column_stack([q, np.ones(len(q))]) @ Minv.T)[:, :2]
        j = uv[:, 0] / rec['xscale_m'] + (rec['nx'] - 1) / 2.0
        i = uv[:, 1] / rec['yscale_m'] + (rec['ny'] - 1) / 2.0
        img = processed_topo(t, direction, b.topo_active) * 1e9
        got = map_coordinates(img, [i, j], order=1, mode='constant', cval=np.nan).reshape(img.shape)
        ok = np.isfinite(ref) & np.isfinite(got)
        ok[:2, :] = ok[-2:, :] = ok[:, :2] = ok[:, -2:] = False
        rng = np.nanmax(img) - np.nanmin(img)
        err = np.nanmax(np.abs(ref[ok] - got[ok])) / rng
        assert ok.sum() > 0.8 * ok.size * 0.9 and err < 0.02, (direction, ok.sum(), err)


@check("A2 images persist across steps (no re-imaging); A4 show modes")
def _persist_and_modes(b, exp):
    b._on_show_mode('all')
    settle(b)
    assert set(b._visible_paths()) == set(b.files)
    ids = {p: id(it['im']) for p, it in b.layer.items.items()}
    for i in range(len(b.files)):
        b._goto(i)
    ids2 = {p: id(it['im']) for p, it in b.layer.items.items()}
    assert all(ids2[p] == ids[p] for p in ids if p in ids2), "an image was rebuilt while stepping"
    topo_paths = [p for p in b.files if b.kinds[p] == 'topo']
    b._goto(0)
    b._on_show_mode('up to current')
    shown = [p for p, it in b.layer.items.items() if it['im'].get_visible()]
    assert shown == [b.files[0]], shown
    b._goto(len(b.files) - 1)
    shown = {p for p, it in b.layer.items.items() if it['im'].get_visible()}
    assert set(topo_paths) <= shown, (shown, topo_paths)
    b._on_show_mode('last 25')
    b._on_show_mode('up to current')


@check("A3 a drift change only moves images (same items, new matrices)")
def _drift_toggle_moves(b, exp):
    b._goto(len(b.files) - 1)
    p = next(p for p in b.files if b.kinds[p] == 'topo')
    im = b.layer.items[p]['im']
    m_on = b.layer.items[p]['mat'].copy()
    b._on_drift_toggle('auto', False)
    b._on_drift_toggle('manual', False)
    assert b.layer.items[p]['im'] is im
    m_off = b.layer.items[p]['mat']
    assert not np.allclose(m_on, m_off), "turning the drift off did not move the image"
    b._on_drift_toggle('auto', True)
    b._on_drift_toggle('manual', True)
    assert np.allclose(b.layer.items[p]['mat'], m_on)


@check("B1 step filter: kinds and minimum duration")
def _step_filter(b, exp):
    b._goto(0)
    b._on_step_kinds('point', [False, True, False, False, False])     # only points
    b._step(+1)
    assert b.kinds[b.files[b.index]] == 'point', b.kinds[b.files[b.index]]
    b._on_step_kinds('topo', [True, True, True, True, True])
    durs = {p: (b.headers[p]['end_time'] - b.headers[p]['start_time']) for p in b.files}
    longest = max(durs.values())
    b._on_min_duration(f"{longest / 60.0 * 0.99:g}")                      # only the longest file passes
    b._goto(0)
    b._step(+1)
    p_long = max(durs, key=durs.get)
    assert b.files[b.index] == p_long or b.index == 0 and b.files[0] == p_long, (b.files[b.index], p_long)
    b._on_min_duration("0")
    j = b._nearest_allowed(2)
    assert j == 2


@check("C1 points grey before the current topo, blue after")
def _point_colours(b, exp):
    b._on_show_mode('all')
    settle(b)
    gi = next(i for i, p in enumerate(b.files) if b.kinds[p] == 'grid')
    b._goto(gi)                                    # the point comes after this grid: blue
    grey, blue = b._point_colours(b._visible_paths())
    assert len(blue) == 1 and len(grey) == 0, (grey, blue)
    b._goto(len(b.files) - 1)                      # the last topo comes after the point: grey
    grey, blue = b._point_colours(b._visible_paths())
    assert len(grey) == 1 and len(blue) == 0, (grey, blue)
    b._on_show_mode('up to current')


@check("A2 occlusion: an image wholly under a later complete image is not painted")
def _occlusion(b, exp):
    from sm4_Browser.scanlayer import _covers
    sq = lambda c, h: np.array([[c - h, c - h], [c + h, c - h], [c + h, c + h], [c - h, c + h]], dtype=float)
    quads = {'a': sq(0, 1), 'b': sq(0, 2), 'c': sq(3, 1), 'd': sq(0, 1.5)}
    order = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    cov = _covers(quads, order, {'a': True, 'b': True, 'c': True, 'd': False})
    assert cov == {'a': ['b']}, cov            # d is incomplete (holes): it hides nothing; c lies elsewhere
    rot = sq(0, 0.5) @ np.array([[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)]]).T
    cov = _covers({'r': rot, 'b': sq(0, 2)}, {'r': 0, 'b': 1}, {'r': True, 'b': True})
    assert cov == {'r': ['b']}, cov


def click(fig, ax, x, y):
    """A press and release at data (x, y) in ax (no motion): mpl events through the canvas."""
    from sm4_Browser.qtplot import MouseEvent
    px, py = ax.transData.transform([[x, y]])[0]
    for name in ('button_press_event', 'button_release_event'):
        e = MouseEvent(name, fig.canvas, px, py, button=1)
        e.inaxes, e.xdata, e.ydata = ax, x, y
        fig.canvas._fire(name, e)


@check("G1-G5 drift window: partners placed at once both ways with the current velocity; arrows on both panels")
def _drift_window(b, exp):
    from sm4_Browser.drift import seed_partner
    goto_kind(b, 'grid')
    b._on_drift_window()
    w = b.drift_window
    assert w is not None and w.markers.n_pairs == 0 and w.sld_markers.value == 0
    v0, src0 = w._seed_velocity()
    assert v0 is not None and src0.startswith('auto'), (v0, src0)
    k = w.k
    # three clicks on the earlier topo: three pairs, partners where the velocity puts them
    a_c = np.asarray(w.a.center_m)
    pts = [a_c + np.array(o) for o in ((-2e-9, -1e-9), (1.5e-9, 2e-9), (3e-9, -3e-9))]
    for p_m in pts:
        click(w.fig, w.ax_a, *(p_m * k))
    A, B = w.markers.markers[w.a.path], w.markers.markers[w.b.path]
    assert len(A) == 3 and len(B) == 3, (len(A), len(B))
    assert w.markers.n_pairs == 3 and w.sld_markers.value == 3
    for ma, mb in zip(A, B):
        exp_xy, _ = seed_partner(w.a, w.b, ma['xy_m'], ma['t'], v0)
        assert np.allclose(mb['xy_m'], exp_xy, atol=1e-15), (mb['xy_m'], exp_xy)
        assert mb.get('seed') is not None
    # arrows on both panels: draw_drift_markers returns markers + texts + arrows per panel
    n_a = sum(1 for a in w.markers._arts if getattr(a, 'axes', None) is w.ax_a)
    n_b = sum(1 for a in w.markers._arts if getattr(a, 'axes', None) is w.ax_b)
    assert n_a >= 3 * 2 and n_b >= 3 * 2, (n_a, n_b)
    # a click on the later topo places its partner on the earlier one
    b_c = np.asarray(w.b.center_m)
    click(w.fig, w.ax_b, *((b_c + np.array([-3e-9, 2.5e-9])) * k))
    assert len(A) == 4 and len(B) == 4 and A[3].get('seed') is not None
    exp_xy, _ = seed_partner(w.b, w.a, B[3]['xy_m'], B[3]['t'], v0)
    assert np.allclose(A[3]['xy_m'], exp_xy, atol=1e-15)
    # a typed velocity applied: new markers use it
    w.box_v.set("50, -40")
    w._on_typed("50, -40")
    w._on_apply()
    v1, src1 = w._seed_velocity()
    assert src1.startswith('manual') and np.allclose(v1 * 3.6e12, [50, -40]), (v1, src1)
    click(w.fig, w.ax_a, *((a_c + np.array([0.5e-9, 0.5e-9])) * k))
    assert len(A) == 5
    exp_xy, _ = seed_partner(w.a, w.b, A[4]['xy_m'], A[4]['t'], v1)
    assert np.allclose(B[4]['xy_m'], exp_xy, atol=1e-15)
    assert B[4].get('seed_source', '').startswith('manual')
    # the count slider back to 0 clears every marker
    w.markers.set_pairs(0)
    assert not w.markers.markers[w.a.path] and not w.markers.markers[w.b.path]
    w.box_v.set("")
    w._on_apply()                                   # the typed span removed again
    plt.close(w.fig)
    plt._app().processEvents()
    assert not w.is_open(), "a closed window still counts as open (qtplot Figure._on_close)"


@check("H stripe matching: the stored fast-axis state in the browser and the Topo window, with drift on and with drift none")
def _stripe_browser(b, exp):
    f = goto_kind(b, 'grid')
    d = b.drift
    assert d.fast_state(f, None, b.auto_on, b.manual_on) == 'off'
    setting = d.slow_setting(f, b.auto_on, b.manual_on)
    assert setting == 'auto', setting
    d.set_fast_state(f, 'auto', 'auto')
    r = d.result(f, b.loaded, None, b.auto_on, b.manual_on, b.anchor_on)
    assert r is not None and r['stripe'] == 'auto' and r['lines']
    b._draw_scan_area()
    # drift none: the 'off' set of the grid places it with a zero velocity once its fast axis is on
    b._on_drift_toggle('auto', False)
    b._on_drift_toggle('manual', False)
    assert d.slow_setting(f, False, False) == 'off'
    assert d.result(f, b.loaded, None, False, False, False) is None          # the 'off' set's fast axis is off
    d.set_fast_state(f, 'off', 'auto')
    d0 = d.result(f, b.loaded, None, False, False, False)
    assert d0 is not None and d0['source'].startswith('none') and d0['stripe'] == 'auto', d0 and d0['source']
    b._on_topo_window()
    w = b.children[-1]
    w.draw()
    assert hasattr(w, 'btn_stripe') and hasattr(w, 'chk_stripe')        # the button and the shared toggle (H2)
    plt.close(w.fig)
    b._on_drift_toggle('auto', True)
    b._on_drift_toggle('manual', True)
    d.set_fast_state(f, 'off', 'off')
    d.set_fast_state(f, 'auto', 'off')


@check("H2 the 'stripe' toggle: one state per file under every drift setting; the browser's, the Topo window's and the Line window's toggles follow; 'stripe match' in the Line window")
def _stripe_toggle(b, exp):
    f = goto_kind(b, 'grid')
    d = b.drift
    for s in d.SLOW_SETTINGS:
        d.set_fast_state(f, s, 'off')
    d.clear_stripe_points(f, 'auto')
    assert not b.chk_stripe.value
    w = open_line(b)
    b._on_topo_window()
    tw = b.children[-1]
    assert not w._int_chk_stripe.value and not tw.chk_stripe.value
    b._on_stripe_toggle(True)                                   # the browser's toggle: on under every setting
    assert all(d.fast_state(f, s) == 'auto' for s in d.SLOW_SETTINGS), [d.fast_state(f, s) for s in d.SLOW_SETTINGS]
    assert b.chk_stripe.value and w._int_chk_stripe.value and tw.chk_stripe.value
    # the stripe window's choice of manual for one setting (a dragged point) keeps every toggle on
    d.set_stripe_point(f, 'auto', 'E:0', 1e-9)
    d.set_fast_state(f, 'auto', 'manual')
    assert b.chk_stripe.value and w._int_chk_stripe.value and tw.chk_stripe.value
    w._int_on_stripe_toggle(False)                              # the Line window's toggle: off everywhere
    assert all(d.fast_state(f, s) == 'off' for s in d.SLOW_SETTINGS)
    assert not b.chk_stripe.value and not w._int_chk_stripe.value and not tw.chk_stripe.value
    tw._on_stripe_toggle(True)                                  # the Topo window's: on; 'auto' holds a point -> manual
    assert d.fast_state(f, 'auto') == 'manual' and d.fast_state(f, 'off') == 'auto' and d.fast_state(f, 'manual') == 'auto'
    assert b.chk_stripe.value and w._int_chk_stripe.value and tw.chk_stripe.value
    # the stripe window's 'off' for the setting in force: the toggles of the windows under it go off
    d.set_fast_state(f, d.slow_setting(f, b.auto_on, b.manual_on), 'off')
    assert not b.chk_stripe.value and not w._int_chk_stripe.value and not tw.chk_stripe.value
    # the Line window's 'stripe match' opens the stripe window once
    w._int_on_stripe_window()
    sw = w._int_stripe_window
    assert sw is not None and sw.is_open() and sw.grid is f
    w._int_on_stripe_window()
    assert w._int_stripe_window is sw
    plt.close(sw.fig)
    for s in d.SLOW_SETTINGS:
        d.set_fast_state(f, s, 'off')
    d.clear_stripe_points(f, 'auto')
    plt.close(tw.fig)
    plt.close(w._int_fig)


@check("G5 stripe window: handles, a drag along the line only, one point moves, the sets per setting, the cache")
def _stripe_window(b, exp):
    f = goto_kind(b, 'grid')
    d = b.drift
    assert b.stripe_window is None
    b._on_stripe_window()
    w = b.stripe_window
    assert w is not None and w.is_open() and w.setting == 'auto' and w.top == 'forward'
    assert d.fast_state(f, 'auto') == 'off' and not w._handles
    w._on_fast('auto')                                              # the matching: handles appear
    assert d.fast_state(f, 'auto') == 'auto' and w._handles, len(w._handles)
    res = w._res
    keys = [h[1] for h in w._handles]
    assert any(k.startswith('E:') for k in keys) and any(k.startswith('R:') for k in keys)
    # a drag: the mouse moves diagonally, the point moves along its line only
    art, key, i, pos0, u = next(h for h in w._handles if h[1].startswith('E:') and h[2] == 2)
    lines0 = {ln['i']: dict(ln) for ln in res['lines']}
    class _E:
        pass
    ev = _E()
    ev.artist = art
    w._on_pick(ev)
    assert w._drag is not None and d.fast_state(f, 'auto') == 'manual'
    pts0 = d.stripe_points(f, 'auto')
    assert len(pts0) == 2 * len(lines0) and abs(pts0[key] - lines0[i]['E']) < 1e-15
    perp = np.array([-u[1], u[0]])
    target = pos0 + 5e-10 * u + 3e-10 * perp                        # 5 A along the line, 3 A across
    mv = _E()
    mv.xdata, mv.ydata, mv.inaxes = target[0] * w.k, target[1] * w.k, w.ax
    w._on_motion(mv)
    w._on_release(mv)
    pts1 = d.stripe_points(f, 'auto')
    fast, _ = f.grid_axes()
    assert abs((pts1[key] - pts0[key]) - 5e-10 * float(np.dot(u, fast))) < 1e-16, (pts1[key] - pts0[key])
    assert all(abs(pts1[k] - pts0[k]) < 1e-18 for k in pts0 if k != key)
    r1 = d.result(f, None, None, True, False)
    l1 = {ln['i']: ln for ln in r1['lines']}
    moved = [j for j in lines0 if abs(l1[j]['E'] - lines0[j]['E']) > 1e-15 or abs(l1[j]['R'] - lines0[j]['R']) > 1e-15
             or abs(l1[j]['c'] - lines0[j]['c']) > 1e-15]
    assert moved == [i], moved
    # the handle sits where the placement puts the point (on the placed line) and moved along the scan
    # axis by the projected amount; the placed direction and the recorded axis differ by the drift's tilt
    h_new = next(h for h in w._handles if h[1] == key)
    disp = h_new[3] - pos0
    assert abs(disp[0] * fast[1] - disp[1] * fast[0]) < 1e-6 * np.linalg.norm(disp), disp
    assert abs(float(np.dot(disp, fast)) - 5e-10 * float(np.dot(u, fast))) < 1e-16
    assert abs(float(np.dot(u, fast))) > 0.999, u
    xy1 = np.asarray(r1['xy_corr'])
    cols_i = np.arange(i * f.n_fast, (i + 1) * f.n_fast)
    kept_i = cols_i[np.asarray(f.acquired, dtype=bool)[cols_i]]
    assert np.allclose(h_new[3], xy1[kept_i[-1]])
    # the cache carries the set; another slow-axis setting has its own, untouched
    rec = d.manual_record()['stripe'][f.name]
    assert rec['auto']['fast'] == 'manual' and len(rec['auto']['points']) == len(pts1)
    assert 'off' not in rec or not rec['off']['points']
    w._on_slow('off')
    assert w.setting == 'off' and d.fast_state(f, 'off') == 'off' and not d.stripe_points(f, 'off')
    w._on_pass('reverse')
    assert w.top == 'reverse'
    w._on_slow('auto')
    w._on_clear()
    assert not d.stripe_points(f, 'auto') and d.fast_state(f, 'auto') == 'manual'
    w._on_fast('off')
    plt.close(w.fig)
    b.stripe_window = None


@check("G6 pass order everywhere; the Line window's forward row default and its rows skewed by the placement")
def _pass_order_and_line(b, exp):
    from sm4_Browser import plots
    f = goto_kind(b, 'grid')
    d = b.drift
    # the browser's pass radio puts the chosen pass on top of the other (both drawn with a velocity)
    b._on_direction('reverse')
    assert b._direction_of(f) == 'reverse'
    b._draw_scan_area()
    b._on_direction('forward')
    # the Map's embedded topo: both passes, the chosen on top
    m = open_map(b)
    m._on_drift_toggle('auto', True)
    m._on_direction('reverse')
    quads = m._topo_maps[id(m.ax_topo)]
    assert len(quads) == 2 and quads[-1].zorder > quads[0].zorder and np.allclose(quads[-1].shift, m._topo_shift('reverse'), equal_nan=True)
    m._on_direction('average')
    quads = m._topo_maps[id(m.ax_topo)]
    assert len(quads) == 2 and np.allclose(quads[-1].shift, m._topo_shift('forward'), equal_nan=True)
    plt.close(m.fig)
    # the Line window opens with the forward row only; its rows sit on their own pixels and carry the within-sweep shear
    w = open_line(b)
    assert w._int_topo_lines == [(w.chunk_index, 'forward')], w._int_topo_lines
    d.set_fast_state(f, 'auto', 'auto')
    w._int_apply_drift_toggle()
    sh = w._int_shear()
    assert sh is not None
    res = d.result(w._int_sm4(), None, None, w._int_auto_on, w._int_manual_on)
    assert res is not None and res['stripe'] == 'auto'
    fast, _ = f.grid_axes()
    cols = np.flatnonzero(w._int_kept_mask()) + w.chunk_index * len(w._int_pos)
    corr = np.asarray(res['xy_corr'])
    pix_f = np.asarray(res['pix']['forward'])
    idx, _ = f.spectrum_pixels('forward')
    # the first edge: the row against its own placed topo pixel only (the seconds between pixel and sweep),
    # never the placement's shift of the spectrum against its recorded position
    off = np.array([((corr[k] - pix_f[idx[k, 0], idx[k, 1]]) @ fast) if idx[k, 0] >= 0 else 0.0
                    for k in cols]) * 1e10 / w.scalefactor
    assert np.allclose(sh[:, 0], off, atol=1e-9), (sh[:3, 0], off[:3])
    step = float(np.linalg.norm(f.spec_xy_m[1] - f.spec_xy_m[0])) * 1e10 / w.scalefactor
    assert np.nanmax(np.abs(off)) < step, (np.nanmax(np.abs(off)), step)
    placed = ((corr[cols] - np.asarray(f.spec_xy_m)[cols]) @ fast) * 1e10 / w.scalefactor
    assert np.nanmax(np.abs(placed)) > step                                       # the placement did move the spectra
    assert np.nanmax(np.abs(sh[:, -1] - sh[:, 0])) > 0                            # the within-sweep shear grows along the sweep
    # the position markers on the heatmap follow their rows' shear (polylines); a drag on the heatmap undoes it
    from sm4_Browser.qtplot import MouseEvent
    p = w._int_marker_pos[0]
    heat = [a for a in w._int_marker_artists if a.get_label() == 'marker_0' and a.axes is w._int_ax_didv][0]
    r = int(np.argmin(np.abs(w._int_kept_pos() - p)))
    ys = np.asarray(heat.get_ydata(), dtype=np.float64)
    assert len(ys) == sh.shape[1] and np.allclose(ys, p + np.nan_to_num(sh[r])), (ys[:3], p, sh[r, :3])
    edges = plots.row_edges(w._int_energy)
    j = sh.shape[1] // 2
    e_mid, s_mid = 0.5 * (edges[j] + edges[j + 1]), 0.5 * (sh[r, j] + sh[r, j + 1])
    class P: pass
    ev = P(); ev.artist = heat
    w._int_on_pick(ev)
    mv = MouseEvent('motion_notify_event', w._int_fig.canvas, 0, 0, button=1)
    mv.inaxes, mv.xdata, mv.ydata = w._int_ax_didv, e_mid, p + s_mid + 0.25 * step
    w._int_on_motion(mv)
    w._int_on_release(mv)
    assert abs(w._int_marker_pos[0] - (p + 0.25 * step)) < 1e-9, (w._int_marker_pos[0], p, step)
    d.set_fast_state(f, 'auto', 'off')
    plt.close(w._int_fig)


@check("F1-F3 E-markers and the energy slice (Line window; line average)")
def _e_markers(b, exp):
    from sm4_Browser.qtplot import Qt, MouseEvent
    from PySide6.QtTest import QTest
    w = open_line(b)
    assert len(w._int_e_markers) == 1 and w._int_sld_e.value == 1
    assert w._int_marker_pos, "position markers present"
    pos_before = list(w._int_marker_pos)
    # count 0..3, the position markers untouched
    w._int_on_e_count(3)
    assert len(w._int_e_markers) == 3 and len(set(w._int_e_markers)) == 3
    w._int_on_e_count(0)
    assert w._int_e_markers == [] and w._int_sld_e.value == 0
    w._int_on_e_count(2)
    assert w._int_marker_pos == pos_before
    # the readout in the information panel (energies and ΔE, no E numbers); the slider under the 'Markers' stack
    info = w._int_fig.info_text()
    assert "E-markers: " in info and "ΔE: " in info and "E1" not in info, info
    assert w._int_sld_e._rect[1] < w._int_groups_ctl.y0, (w._int_sld_e._rect, w._int_groups_ctl.y0)
    # the slice: one curve per E-marker, against position, the value at the marker's energy; its legend
    # names each by its energy, in a column right of the curves (the x range widened for it)
    lines = [a for a in w._int_ax_slice._children if hasattr(a, '_y') and len(getattr(a, '_y', [])) == len(w._int_pos)]
    assert len(lines) == 2, len(lines)
    assert w._int_ax_slice.get_legend() is not None
    xmin = min(float(np.nanmin(l._x)) for l in lines)
    xmax = max(float(np.nanmax(l._x)) for l in lines)
    assert w._int_ax_slice.get_xlim()[1] >= xmax + 0.2 * (xmax - xmin), (w._int_ax_slice.get_xlim(), xmin, xmax)
    j = w._int_e_index(w._int_e_markers[0])
    kept = w._int_kept_mask()
    assert np.allclose(lines[0]._x[kept], w._int_curves[kept, j], equal_nan=True)
    # drag E1 on the heatmap: pick then motion
    e_axis = w._int_e_axis()
    art = next(a for a in w._int_e_artists if getattr(a, 'get_label', lambda: '')() == '_emarker_0')
    class P: pass
    ev = P(); ev.artist = art
    w._int_on_pick(ev)
    target = float(e_axis[3])
    mv = MouseEvent('motion_notify_event', w._int_fig.canvas, 0, 0, button=1)
    mv.inaxes, mv.xdata, mv.ydata = w._int_ax_didv, target + 1e-4, 10.0
    w._int_on_motion(mv)
    w._int_on_release(mv)
    assert w._int_e_markers[0] == target, (w._int_e_markers[0], target)
    # keys step the selected one by one recorded energy (real key events on the canvas)
    c = w._int_fig.canvas
    c.setFocus()
    QTest.keyClick(c, Qt.Key.Key_Right)
    assert w._int_e_markers[0] == float(e_axis[4]), w._int_e_markers
    QTest.keyClick(c, Qt.Key.Key_Tab)
    assert w._int_e_selected == 1
    # the record keeps them
    w._int_on_e_count(1)
    plt.close(w._int_fig)
    # line average: built on the Line window, gets the E-markers too
    w2 = open_line(b)
    n0 = len(b.line_averages.open())
    w2._int_on_line_new()
    la = list(b.line_averages.open().values())
    assert len(la) == n0 + 1, "line average did not open"
    lw = la[-1]
    lw.run()
    assert lw._int_e_markers and lw._int_ax_slice is not None
    lw._int_on_e_count(2)
    rec = lw.record()
    assert len(rec.get('energy_markers', [])) == 2
    plt.close(lw._int_fig)
    plt.close(w2._int_fig)


@check("G1 Line window and line average: the crop cursors, 'crop' / 'uncrop' (limits, normalization, colour scale, record)")
def _line_ranges(b, exp):
    from sm4_Browser import plots
    from sm4_Browser.qtplot import MouseEvent
    w = open_line(b)
    e = w._int_e_axis()
    lo, hi = float(e[len(e) // 4]), float(e[3 * len(e) // 4])
    # the cursors start at the ends: four lines, no shade, the boxes showing the ends
    assert w._int_cursors['e'] == [float(e.min()), float(e.max())], w._int_cursors['e']
    assert len(w._int_cursor_artists) == 4 and not w._int_shade_artists
    assert w._int_cursor_boxes[('e', 0)].value == f"{e.min():.3f}", w._int_cursor_boxes[('e', 0)].value
    # 'crop' with every cursor at its end keeps everything as it is
    w._int_on_crop()
    assert w._int_e_range is None and w._int_x_range is None
    # normalize on: the energy crop is the normalized range
    w._int_pipeline_box.set_state(calibrate=w._int_apply_calibration, normalize=True)
    w._int_on_pipeline_change({'calibrate': w._int_apply_calibration, 'normalize': True})
    assert w._int_apply_normalize
    # a typed value snaps the cursor to the recorded energy; the shade covers the outside; nothing applied yet
    w._int_on_cursor_box('e', 0, f"{lo + 0.3 * (e[1] - e[0]):.6f}")
    w._int_set_cursor('e', 1, hi)
    assert w._int_cursors['e'] == [lo, hi], w._int_cursors['e']
    assert len(w._int_shade_artists) == 2 and w._int_e_range is None
    assert w._int_cursor_boxes[('e', 0)].value == f"{lo:.3f}" and w._int_cursor_boxes[('e', 1)].value == f"{hi:.3f}"
    # the lower cursor never passes the upper: two recorded energies stay between the pair
    w._int_set_cursor('e', 0, hi + 1.0)
    i_hi = int(np.argmin(np.abs(e - hi)))
    assert w._int_cursors['e'][0] == float(e[i_hi - 1]), (w._int_cursors['e'], hi)
    w._int_set_cursor('e', 0, lo)
    w._int_on_crop()
    assert w._int_e_range == (lo, hi), w._int_e_range
    assert w._int_cursors['e'] == [lo, hi] and not w._int_shade_artists          # at the new edges
    assert np.allclose(w._int_ax_didv.get_xlim(), (lo, hi)), w._int_ax_didv.get_xlim()
    assert np.allclose(w._int_ax_curve.get_xlim(), (lo, hi)), w._int_ax_curve.get_xlim()
    m = (e >= lo) & (e <= hi)
    rows = np.asarray(w._int_curves)[w._int_kept_mask()]
    areas = [abs(np.trapezoid(r[m], e[m])) for r in rows if np.isfinite(r).all()]
    assert areas and np.allclose(areas, 1.0), areas[:5]
    # the colour scale from the cells inside the crop only
    vmin, vmax = plots.finite_range(rows[:, m])
    assert np.isclose(w._int_im.norm.vmin, vmin) and np.isclose(w._int_im.norm.vmax, vmax), \
        (w._int_im.norm.vmin, w._int_im.norm.vmax, vmin, vmax)
    w._int_on_gamma(2.0)
    assert np.isclose(w._int_im.norm.vmin, vmin) and np.isclose(w._int_im.norm.vmax, vmax)
    # the companion shows and normalizes over the same range
    fw = w._int_feenstra
    if fw is not None and fw.is_open():
        fw.pipeline.set_state(normalize=True)
        fw.refresh()
        col = fw.cols[0]
        assert np.allclose(col['top'].get_xlim(), (lo, hi)), col['top'].get_xlim()
        assert np.allclose(col['curves'].get_xlim(), (lo, hi)), col['curves'].get_xlim()
        v = w.feenstra_view(calibrate=w._int_apply_calibration, normalize=True)
        top = np.asarray(v['versions']['didv']['top'])[w._int_kept_mask()]
        a2 = [abs(np.trapezoid(r[m], e[m])) for r in top if np.isfinite(r).all()]
        assert a2 and np.allclose(a2, 1.0), a2[:5]
    # the position pair: the view only
    p0, p1 = w._int_pos_bounds()
    q0, q1 = p0 + 0.25 * (p1 - p0), p0 + 0.75 * (p1 - p0)
    curves_before = np.array(w._int_curves)
    w._int_set_cursor('x', 0, q0)
    w._int_set_cursor('x', 1, q1)
    assert np.allclose(w._int_cursors['x'], (q0, q1)) and len(w._int_shade_artists) == 2
    w._int_on_crop()
    assert w._int_x_range == (q0, q1) and w._int_e_range == (lo, hi)
    assert np.allclose(w._int_ax_didv.get_ylim(), (q0, q1)), w._int_ax_didv.get_ylim()
    assert np.allclose(w._int_ax_topo.get_ylim(), (q0, q1)), w._int_ax_topo.get_ylim()
    assert np.allclose(curves_before, w._int_curves, equal_nan=True)
    # the boxes ride with their cursors: the energy ones above the heatmap, the position ones right of it
    bx, by, bw, bh = w._int_ax_didv.get_position().bounds
    r_e = w._int_cursor_boxes[('e', 0)]._rect
    r_x = w._int_cursor_boxes[('x', 1)]._rect
    assert r_e[1] >= by + bh - 1e-9 and abs(r_e[0] + 0.5 * r_e[2] - bx) < 1e-6, (r_e, bx, by, bh)
    assert r_x[0] >= bx + bw - 1e-9 and abs(r_x[1] + 0.5 * r_x[3] - (by + bh)) < 1e-6, (r_x, bx, by, bh)
    # the 'γ' box hangs from the top of the x label's band, right-aligned with the heatmap: under the
    # tick marks and their values (the panel's placements re-place it, qtplot.Axes.on_placed)
    plt._app().processEvents()
    row = w._int_ax_didv.xlabel_row()
    g = w._int_gamma_box.box._rect
    assert row is not None, "the heatmap has no x label"
    assert abs((g[1] + g[3]) - (row[0] + 0.5 * row[1])) < 1e-9, (g, row)
    assert abs(g[0] + g[2] - (bx + bw)) < 1e-9, (g, bx, bw)
    assert g[1] + g[3] < by - 1e-9, (g, by)
    # the record carries both; 'uncrop' restores the full ranges and the cursors; the record restores them
    rec = w.marker_record()
    assert rec['energy_range'] == [lo, hi] and rec['pos_range'] == [q0, q1], (rec['energy_range'], rec['pos_range'])
    w._int_on_uncrop()
    assert w._int_e_range is None and w._int_x_range is None
    x0, x1, y0, y1 = w._int_im.extent()
    assert np.allclose(w._int_ax_didv.get_xlim(), (x0, x1)) and np.allclose(w._int_ax_didv.get_ylim(), (y0, y1))
    assert w._int_cursors['e'] == [float(e.min()), float(e.max())]
    assert np.allclose(w._int_cursors['x'], (min(y0, y1), max(y0, y1))), w._int_cursors['x']
    full = [abs(np.trapezoid(r, e)) for r in np.asarray(w._int_curves)[w._int_kept_mask()] if np.isfinite(r).all()]
    assert np.allclose(full, 1.0)
    w.restore_markers(rec)
    assert w._int_e_range == (lo, hi) and w._int_x_range == (q0, q1), (w._int_e_range, w._int_x_range)
    assert np.allclose(w._int_ax_didv.get_xlim(), (lo, hi)) and np.allclose(w._int_ax_didv.get_ylim(), (q0, q1))
    assert w._int_cursors['e'] == [lo, hi] and np.allclose(w._int_cursors['x'], (q0, q1))
    # a box text that is not a number is refused and the box put back
    w._int_on_cursor_box('e', 0, "abc")
    assert w._int_cursors['e'] == [lo, hi] and w._int_cursor_boxes[('e', 0)].value == f"{lo:.3f}"
    # a cursor dragged on the heatmap: pick (the cursor's before any marker's), motion along its axis, release
    art = next(a for a in w._int_cursor_artists if a.get_label() == '_cursor_e_1')
    class P: pass
    ev = P(); ev.artist = art
    w._int_on_pick(ev)
    assert w._int_active_obj == ('cursor', 'e', 1)
    inside = e[(e >= lo) & (e <= hi)]
    target = float(inside[len(inside) // 2])
    mv = MouseEvent('motion_notify_event', w._int_fig.canvas, 0, 0, button=1)
    mv.inaxes, mv.xdata, mv.ydata = w._int_ax_didv, target + 1e-4, q0
    w._int_on_motion(mv)
    w._int_on_release(mv)
    assert w._int_cursors['e'][1] == target and len(w._int_shade_artists) == 1, w._int_cursors['e']
    w._int_on_uncrop()
    plt.close(w._int_fig)
    # the line average: the same cursors and buttons; the energy crop normalizes every member's columns at the next run
    w2 = open_line(b)
    n0 = len(b.line_averages.open())
    w2._int_on_line_new()
    la = list(b.line_averages.open().values())
    assert len(la) == n0 + 1, "line average did not open"
    lw = la[-1]
    assert lw._int_cursor_boxes[('e', 0)].widget.isVisible() and lw._int_btn_crop.widget.isVisible()
    assert lw.chk_overlapping.label == 'overlapping'
    lw.run()
    ge = lw._int_e_axis()
    glo, ghi = float(ge[len(ge) // 4]), float(ge[3 * len(ge) // 4])
    lw._int_pipeline_box.set_state(calibrate=lw._int_apply_calibration, normalize=True)
    lw._int_on_pipeline_change({'calibrate': lw._int_apply_calibration, 'normalize': True})
    lw._int_set_cursor('e', 0, glo)
    lw._int_set_cursor('e', 1, ghi)
    lw._int_on_crop()
    assert lw._int_e_range == (glo, ghi) and lw.run_needed              # the view at once, the normalization at the next run
    assert np.allclose(lw._int_ax_didv.get_xlim(), (glo, ghi)), lw._int_ax_didv.get_xlim()
    lw.run()
    assert not lw.run_needed
    mem = lw.members[0]
    me = np.asarray(mem['selection'].energy, dtype=np.float64)
    mm = (me >= glo) & (me <= ghi)
    cols = lw._member_columns(mem, False, lw._int_apply_calibration, True, *lw._int_filters.params())
    a3 = [abs(np.trapezoid(cols[mm, j], me[mm])) for j in range(cols.shape[1]) if np.isfinite(cols[:, j]).all()]
    assert a3 and np.allclose(a3, 1.0), a3[:5]
    rec2 = lw.record()
    assert rec2['energy_range'] == [glo, ghi] and 'overlapping' in rec2 and 'crop' not in rec2
    lw._int_on_uncrop()
    assert lw._int_e_range is None
    plt.close(lw._int_fig)
    plt.close(w2._int_fig)


@check("G2 Map window: pixels skewed by the chain plus the sweep drift, the topo unclipped, the grid view's topo")
def _map_drift(b, exp):
    from sm4_Browser import plots
    m = open_map(b)
    assert m.drift is not None and m.has_topo
    l, r, bb, t = m.extent
    frame = (min(l, r), max(l, r), min(bb, t), max(bb, t))
    m._on_drift_toggle('auto', False)                                # the window opens with the browser's states
    m._on_drift_toggle('manual', False)
    assert m._frame_shift(m.e_index) is None                        # both toggles off: no shift
    plt._app().processEvents()                                      # the deferred placement of the box runs
    off_lim = (m.ax_map.get_xlim(), m.ax_map.get_ylim())             # the frame
    m._on_drift_toggle('auto', True)
    assert m.auto_on
    plt._app().processEvents()
    res = m.drift.result(m.grid, None, False, True, False)
    assert res is not None and res.get('shift') is not None, "the synthetic grid should get an auto velocity"
    n = m.n_lines * m.n_fast
    k = 1e9 / m.scalefactor
    chain = np.asarray(res['xy_corr'])[:n] - np.asarray(m.grid.spec_xy_m)[:n]
    assert np.nanmax(np.abs(chain)) > 0, "the chain should move later spectra"
    sh = m._frame_shift(m.e_index)
    assert sh is not None and sh.shape == (m.n_lines, m.n_fast, 2)
    col = int(m.order[m.e_index])
    expect = (chain + res['shift'][:n, col, :]) @ m.grid.rotation() * k
    assert np.allclose(sh.reshape(-1, 2), expect, equal_nan=True)
    # the chain dominates the within-sweep drift: the shift is not the sweep drift alone
    sweep_only = res['shift'][:n, col, :] @ m.grid.rotation() * k
    assert not np.allclose(np.nan_to_num(sh.reshape(-1, 2)), np.nan_to_num(sweep_only))
    # the single view's map carries that shift; the topo is quads with the chain's pixel shifts
    pm = m._pixel_maps['single']
    assert isinstance(pm, plots.PixelMap) and np.allclose(pm.shift, sh, equal_nan=True)
    tmaps = m._topo_maps.get(id(m.ax_topo))
    assert tmaps, "the topo should be drawn as quads with a drift source on"
    tsh = m._topo_shift('forward')
    assert tsh is not None and np.allclose(tmaps[-1].shift, tsh, equal_nan=True)
    # every marker is drawn on its pixel as drawn: on the map at the shown energy's shift, on the topo at the chain's
    px0 = m.marker_px[0]
    arts = {id(a.axes): a for a in m._marker_artists if a.get_label() == 'marker_0'}
    on_map = arts[id(m.ax_map)]
    assert np.allclose((on_map.get_xdata()[0], on_map.get_ydata()[0]), m.marker_xy(px0, True))
    assert np.allclose(m.marker_xy(px0, True), np.asarray(m.pixel_xy(px0)) + sh[px0[0], px0[1]])
    on_topo = arts[id(m.ax_topo)]
    assert np.allclose((on_topo.get_xdata()[0], on_topo.get_ydata()[0]), m.marker_xy(px0, False))
    assert not np.allclose(m.marker_xy(px0, True), m.pixel_xy(px0))
    # a drag lands on the pixel drawn under the cursor
    assert m.xy_pixel_drawn(*m.marker_xy(m.marker_px[1], True), True) == tuple(m.marker_px[1])
    # every drawn quad (map and topo) lies inside the limits, which are exactly the box of everything drawn
    # with every energy counted (the acquired map pixels at every energy's shift, the placed topo pixels,
    # the drift lines): edge to edge, no margin, the recorded frame no longer forced in
    drawn_q = lambda q: q._q.reshape(-1, 4, 2)[np.isfinite(q.img).ravel()].reshape(-1, 2)
    quads = np.vstack([drawn_q(q) for q in tmaps] + [drawn_q(pm)])
    x0, x1 = m.ax_map.get_xlim()
    y0, y1 = m.ax_map.get_ylim()
    assert quads[:, 0].min() >= x0 - 1e-6 and quads[:, 0].max() <= x1 + 1e-6, (quads[:, 0].min(), quads[:, 0].max(), x0, x1)
    assert quads[:, 1].min() >= y0 - 1e-6 and quads[:, 1].max() <= y1 + 1e-6, (quads[:, 1].min(), quads[:, 1].max(), y0, y1)
    assert np.allclose(plots.pixel_bounds(pm.img, m.extent, pm.shift), plots.drawn_box(drawn_q(pm)))   # the helper = the quads drawn
    acq_img = np.where(np.asarray(m.grid.acquired, dtype=bool)[:n].reshape(m.n_lines, m.n_fast), 0.0, np.nan)
    boxes = [plots.pixel_bounds(acq_img, m.extent, m._frame_shift(e)) for e in range(m.n_e)]
    pts = [np.array([[bx[0], bx[2]], [bx[1], bx[3]]]) for bx in boxes if bx is not None] + [drawn_q(q) for q in tmaps]
    exp_box = plots.drawn_box(*pts, m._frame_lines())
    assert np.allclose((x0, x1, y0, y1), exp_box), ((x0, x1, y0, y1), exp_box)
    assert np.allclose(m.ax_topo.get_xlim(), (x0, x1)) and np.allclose(m.ax_topo.get_ylim(), (y0, y1))
    # the limits do not jump while stepping the energy
    m.step(1)
    plt._app().processEvents()
    assert np.allclose(m.ax_map.get_xlim(), (x0, x1)) and np.allclose(m.ax_map.get_ylim(), (y0, y1))
    # the companion gets the same bounds
    v = m.feenstra_view()
    assert v['limits'] is not None and np.allclose(v['limits'], m._drift_bounds())
    # the grid view shows the topo in its first cell, the energy maps after it
    m.toggle_view()
    assert m.view == 'grid' and m.grid_topo_ax.get_visible() and m._topo_maps.get(id(m.grid_topo_ax))
    assert len(m.grid_axes) == m.n_e and all(ax.get_visible() for ax in m.grid_axes)
    gx = m.grid_topo_ax.get_position().bounds
    assert all(gx[0] < ax.get_position().bounds[0] or gx[1] > ax.get_position().bounds[1] for ax in m.grid_axes[:1])
    m.toggle_view()
    assert m.view == 'single' and not m.grid_topo_ax.get_visible()
    # the pass radio with a velocity on: both passes drawn, the chosen one on top ('average' counts as forward)
    m._on_direction('reverse')
    both = m._topo_maps[id(m.ax_topo)]
    assert len(both) == 2 and both[-1].zorder > both[0].zorder
    assert np.allclose(both[-1].shift, m._topo_shift('reverse'), equal_nan=True)
    m._on_direction('average')
    both = m._topo_maps[id(m.ax_topo)]
    assert len(both) == 2 and np.allclose(both[-1].shift, m._topo_shift('forward'), equal_nan=True)
    m._on_direction('forward')
    # drift off: the recorded frame, the topo as an image, no shift
    m._on_drift_toggle('auto', False)
    plt._app().processEvents()
    assert m._frame_shift(m.e_index) is None and not m._topo_maps.get(id(m.ax_topo))
    assert np.allclose(m.ax_map.get_xlim(), off_lim[0]) and np.allclose(m.ax_map.get_ylim(), off_lim[1])
    assert off_lim[0][0] <= frame[0] + 1e-9 and off_lim[0][1] >= frame[1] - 1e-9 and off_lim[1][0] <= frame[2] + 1e-9 \
        and off_lim[1][1] >= frame[3] - 1e-9, "drift off: the recorded frame"
    assert not (np.allclose((x0, x1), off_lim[0]) and np.allclose((y0, y1), off_lim[1])), "the limits follow the shifted pixels"
    assert m.ax_topo.images, "the topo should be an image with the drift off"
    plt.close(m.fig)


@check("G3 Map window: the whole map into an average; the figure composer (order, symbols, arrows, delete, new plot)")
def _map_composer(b, exp):
    m = open_map(b)
    # the whole map as one entry: every acquired pixel one sweep
    sel = m.map_selection()
    n = m.n_lines * m.n_fast
    assert sel is not None and len(sel.cols) == int(np.sum(np.asarray(m.grid.acquired, dtype=bool)[:n]))
    assert sel.where.get('map') and len(sel.where['positions']) == len(sel.cols)
    k0 = len(m._avg_windows)
    m._on_map_new()
    assert len(m._avg_windows) == k0 + 1, "map → new avg should open an averaging window"
    w = m._avg_windows[-1]
    assert w.has(sel) and len(w.entries) == 1
    # the composer: a map, the topo, another map at a lower energy, the spectra
    m._on_add_to_plot('map')
    m._on_add_to_plot('topo')
    e0 = m.e_index
    m.step(-1)
    assert m.e_index != e0
    m._on_add_to_plot('map')
    m._on_add_to_plot('spectra')
    c = m.composer
    assert c is not None and c.is_open()
    kinds = [it['kind'] for it in c.ordered()]
    n_sp = len(m.marker_px)
    assert kinds == ['topo', 'map', 'map'] + ['spectrum'] * n_sp, kinds
    biases = [it['bias'] for it in c.ordered() if it['kind'] == 'map']
    assert biases == sorted(biases) and np.isclose(biases[0], m.V[m.e_index]) and np.isclose(biases[1], m.V[e0])
    assert len(c.panels) == len(kinds)
    for ax, it, info in c.panels:
        if it['kind'] in ('map', 'topo'):
            assert info['symbols'] == n_sp
        else:
            assert info['arrows'] == 2
    # the map item carries the shown slice and its colour limits; the spectra their positions
    it_map = [it for it in c.ordered() if it['kind'] == 'map'][1]
    assert np.allclose(it_map['img'], m.maps[e0], equal_nan=True) and it_map['vmin'] is not None
    pos = [it['position'] for it in c.ordered() if it['kind'] == 'spectrum']
    assert np.allclose(pos, [m.marker_xy(px, True) for px in m.marker_px])      # the markers as drawn on the map
    # Delete over the topo panel removes it; 'new plot' empties the list
    class _E:
        pass
    ev = _E()
    ev.key, ev.inaxes = 'delete', c.panels[0][0]
    c._on_key(ev)
    assert [it['kind'] for it in c.ordered()] == ['map', 'map'] + ['spectrum'] * n_sp
    m._on_new_plot()
    assert c.items == [] and c.panels == [] and c.is_open()
    m._on_add_to_plot('map')
    assert len(c.panels) == 1
    plt.close(c.fig)
    plt.close(w.fig)
    plt.close(m.fig)


@check("G4 drift window: '< before' / 'before >' / '< after' / 'after >' step the pair; the span covers the files between")
def _drift_step(b, exp):
    goto_kind(b, 'grid')
    b._on_drift_window()
    w = b.drift_window
    paths = list(w.members)
    assert len(paths) >= 3, paths
    a0, b0 = w.a.path, w.b.path
    ia, ib = paths.index(a0), paths.index(b0)
    assert ia < ib
    sp0 = b.drift.span_between(a0, b0)
    # beyond the segment's end, or the later onto the earlier: refused, nothing changes
    if ib == len(paths) - 1:
        w._step('after', +1)
        assert (w.a.path, w.b.path) == (a0, b0)
    if ib - ia == 1:
        w._step('after', -1)
        assert (w.a.path, w.b.path) == (a0, b0)
    assert ia > 0, "the synthetic grid's before topo should not be the segment's first"
    w._step('before', -1)
    assert w.a.path == paths[ia - 1] and w.b.path == b0, (w.a.path, w.b.path)
    assert w.markers.n_pairs == 0 and w.markers.panels[0][1] == w.a.path and w.markers.panels[1][1] == w.b.path
    assert w.box_v.value == '' and w._applied is None
    # a typed velocity applied covers every file between the topos shown, whatever its kind
    w.box_v.set("30, -20")
    w._on_typed("30, -20")
    w._on_apply()
    sp = b.drift.span_between(w.a.path, w.b.path)
    assert sp is not None and np.allclose(np.asarray(sp['v']) * 3.6e12, [30, -20])
    i_a, i_b = b.files.index(w.a.path), b.files.index(w.b.path)
    between = [p for p in b.files if i_a <= b.files.index(p) <= i_b]
    assert len(between) >= 3 and any(b.kinds[p] != 'topo' for p in between)
    for p in between:
        m = b.drift._manual_model(p, 0.0)
        assert m is not None and np.allclose(m.v, sp['v']), p
    # back to the original pair: its own span state comes back
    w._step('before', +1)
    assert (w.a.path, w.b.path) == (a0, b0)
    assert (w._applied is None) == (sp0 is None)
    b.drift.clear_manual_span(paths[ia - 1], b0)
    plt.close(w.fig)


@check("T1 titles: a file name is shown exactly as named (no capitalisation), other titles as before")
def _title_rule(b, exp):
    st = plt.short_title
    assert st("image_12.sm4: later topography, forward pass, 01-01 10:05:01") == "image_12.sm4"
    assert st("synth_ct_2024_01_01_10_05_01_440.SM4, this file") == "synth_ct_2024_01_01_10_05_01_440.SM4"
    assert st("raw dI/dV: 12 curves") == "Raw dI/dV"
    assert st("dI/dV: 12 curves") == "dI/dV"
    assert st("Scan area: 5 files") == "Scan area"


@check("G7 drift window: the other topo's recorded frame on each panel, on top of everything; the titles; one view edge to edge; the step")
def _drift_outline(b, exp):
    from sm4_Browser import plots
    goto_kind(b, 'grid')
    b._on_drift_window()
    w = b.drift_window

    def check_frames():
        plt._app().processEvents()                          # the deferred placement of the boxes runs
        for ax, own, other in ((w.ax_a, w.a, w.b), (w.ax_b, w.b, w.a)):
            arts = w._outlines[ax]
            assert len(arts) == 1, "the outline carries no text"
            line = arts[0]
            x, y = line.get_data()
            c = other.corners_m() * w.k_of(other.path)       # the other topo's frame at its own scale
            assert np.allclose(np.column_stack([x, y]), np.vstack([c, c[:1]])), (own.name, other.name)
            assert line.get_zorder() > max(a.get_zorder() for a in ax.lines if a is not line), own.name
            assert line.get_zorder() > max(a.get_zorder() for a in ax.images), own.name
            assert not np.allclose(c, own.corners_m() * w.k_of(own.path)), "the frame must be the other topo's, not this panel's"
            for art in w.markers._arts:                     # markers and arrows drawn later still sit below it
                if getattr(art, 'axes', None) is ax:
                    assert art.get_zorder() < line.get_zorder(), own.name
            assert ax.get_title() == own.name, (ax.get_title(), own.name)
        # one view on both panels: the same limits, the same box, edge to edge around the two frames and the
        # file's spectra (plots.fit_limits: no margin)
        allc = np.vstack([w.a.corners_m() * w.k_of(w.a.path), w.b.corners_m() * w.k_of(w.b.path)])
        (xa0, xa1), (ya0, ya1) = w.ax_a.get_xlim(), w.ax_a.get_ylim()
        (xb0, xb1), (yb0, yb1) = w.ax_b.get_xlim(), w.ax_b.get_ylim()
        assert np.allclose([xa0, xa1, ya0, ya1], [xb0, xb1, yb0, yb1]), ((xa0, xa1, ya0, ya1), (xb0, xb1, yb0, yb1))
        ga, gb = w.ax_a.vb.geometry(), w.ax_b.vb.geometry()
        ra, rb = (ga.width(), ga.height()), (gb.width(), gb.height())
        assert abs(ra[0] - rb[0]) < 1.0 and abs(ra[1] - rb[1]) < 1.0, (ra, rb)
        ff = w.f
        sp = None
        if ff.kind != 'topo' and len(ff.spec_xy_m):
            xy = np.asarray(ff.spec_xy_m, dtype=np.float64) * w.k_of(ff.path)
            sp = xy[np.asarray(ff.acquired, dtype=bool)[:len(xy)]] if ff.kind == 'grid' else xy
        box = plots.drawn_box(allc, sp)
        tol = 1e-6 * max(xa1 - xa0, ya1 - ya0)
        assert np.allclose([xa0, xa1, ya0, ya1], box, atol=tol, rtol=0), ((xa0, xa1, ya0, ya1), box)
        assert abs((xa1 - xa0) / (ya1 - ya0) - ra[0] / ra[1]) < 0.02, "the view is not at equal aspect in the box"

    check_frames()
    # a marker pair placed after the frames (a click on the earlier topo seeds its partner on the later)
    click(w.fig, w.ax_a, *(np.asarray(w.a.center_m) * w.k_of(w.a.path)))
    assert w.markers.n_pairs == 1 and any(getattr(a, 'axes', None) is w.ax_b for a in w.markers._arts)
    check_frames()
    paths = list(w.members)
    ia = paths.index(w.a.path)
    assert ia > 0
    a0 = w.a.path
    w._step('before', -1)
    assert w.a.path == paths[ia - 1] and w.a.path != a0
    check_frames()
    w._step('before', +1)
    check_frames()
    plt.close(w.fig)


# ---------------------------------------------------------------------- #
#  Item 1: one text-size rule, design sizes, short labels and stamp titles  #
# ---------------------------------------------------------------------- #

def _new_figures(fn):
    """The figures a call opens (qtplot's registry before and after), in number order."""
    before = set(plt._figures)
    fn()
    plt._app().processEvents()
    return [plt._figures[n] for n in sorted(set(plt._figures) - before)]


def _item1_windows(b):
    """One figure window of every kind the synthetic experiment opens: [(kind, fig)]."""
    wins = []
    goto_kind(b, 'grid')
    b._on_topo_window()
    wins.append(('Topo', b.children[-1].fig))
    w = open_line(b)
    wins.append(('Line', w._int_fig))
    if w._int_feenstra is not None and w._int_feenstra.is_open():
        wins.append(('Feenstra', w._int_feenstra.fig))
    rows = _new_figures(w._int_topo_rows_window)
    wins += [('topo rows', f) for f in rows]
    fits = _new_figures(w._int_on_send)
    wins += [(f"fit {i + 1}", f) for i, f in enumerate(fits)]
    m = open_map(b)
    wins.append(('Map', m.fig))
    m._on_add_to_plot('map')
    wins.append(('composer', m.composer.fig))
    b._on_drift_window()
    wins.append(('drift', b.drift_window.fig))
    b._on_stripe_window()
    wins.append(('stripe', b.stripe_window.fig))
    settle(b)
    ax = b.ax_scan
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    b._on_section()
    _path_click(b, x0 + 0.3 * (x1 - x0), 0.5 * (y0 + y1))
    _path_click(b, x0 + 0.7 * (x1 - x0), 0.5 * (y0 + y1))
    b._on_cross_section()
    wins.append(('profile', b.profile_window.fig))
    b._on_clear_path()
    goto_kind(b, 'point')
    b._on_add_new_avg()
    wins.append(('averaging', list(b.averages.open().values())[-1].fig))
    w._int_on_line_new()
    wins.append(('line average', list(b.line_averages.open().values())[-1]._int_fig))
    b._on_load_saved()
    wins.append(('saved averages', b.saved_window.fig))
    plt._app().processEvents()
    return wins


def _text_widgets(c):
    """The Qt widgets with a text of one widgets.Control: (widget, extra width its indicator needs)."""
    from sm4_Browser.qtplot import QtWidgets
    out = []
    for qw in c._qws:
        for w in [qw] + qw.findChildren(QtWidgets.QWidget):
            if not w.isVisible() or not hasattr(w, 'text') or not str(w.text()).strip():
                continue
            if isinstance(w, QtWidgets.QLineEdit):
                continue
            extra = 0
            if isinstance(w, QtWidgets.QCheckBox):
                st = w.style()
                extra = st.pixelMetric(QtWidgets.QStyle.PixelMetric.PM_IndicatorWidth, None, w) \
                    + st.pixelMetric(QtWidgets.QStyle.PixelMetric.PM_CheckBoxLabelSpacing, None, w)
            elif isinstance(w, QtWidgets.QRadioButton):
                st = w.style()
                extra = st.pixelMetric(QtWidgets.QStyle.PixelMetric.PM_ExclusiveIndicatorWidth, None, w) \
                    + st.pixelMetric(QtWidgets.QStyle.PixelMetric.PM_RadioButtonLabelSpacing, None, w)
            out.append((w, extra))
    return out


def _settle_dock(ms=120):
    """Run the event loop for a moment: the windows' info docks open on a 50 ms timer after a title is set."""
    from PySide6.QtTest import QTest
    QTest.qWait(ms)


def _assert_text_rule(kind, fig, factor):
    """The fonts of a figure window follow the one rule at this size factor, and no text is cut."""
    from sm4_Browser.qtplot import QtGui, qfont, window_zoom, ZOOM_MIN, ZOOM_MAX, DPI
    from sm4_Browser import widgets
    W, H = fig._canvas_size()
    W0, H0 = fig._design_px
    z = window_zoom(W, H, W0, H0)
    want = min(max(factor, ZOOM_MIN), ZOOM_MAX)
    assert abs(z - want) < 0.06, (kind, factor, (W, H), (W0, H0), z)
    assert abs(fig.font_zoom - z) <= 0.03 * z + 1e-9, (kind, factor, fig.font_zoom, z)
    z = fig.font_zoom                 # the fonts are at the figure's zoom (updated on a change above 3 %)
    n_axes = 0
    for ax in fig.axes:
        if not ax._visible:
            continue
        n_axes += 1
        for item in (ax.xaxis_item, ax.yaxis_item):
            tf = item.style.get('tickFont')
            assert tf is not None and tf.pixelSize() == qfont(item._tick_pt, zoom=z).pixelSize(), (kind, factor, tf)
        if ax._title is not None:
            got, want_px = ax._title.item.textItem.font().pixelSize(), qfont(ax._title._size, zoom=z).pixelSize()
            assert got == want_px, (kind, factor, ax._title_full[:50], ax._title._size, got, want_px, fig.font_zoom, z)
    n_ctrl = 0
    for c in fig._qt_controls:
        if not isinstance(c, widgets.Control) or not c._visible:
            continue
        base = widgets._font(c.spec['fs'], z).pixelSize()
        for w, extra in _text_widgets(c):
            n_ctrl += 1
            px = w.font().pixelSize()
            assert 0.75 * base - 1.0 <= px <= base + 1.0, (kind, factor, c.name, w.text(), px, base)
            need = QtGui.QFontMetricsF(w.font()).horizontalAdvance(w.text()) + extra
            assert need <= w.width() + 1.0, (kind, factor, c.name, w.text(), need, w.width())
    return n_axes, n_ctrl


def _label_map(fig):
    """{id(widget): text} of every text widget of the figure's controls (buttons, toggles, radios, labels)."""
    from sm4_Browser import widgets
    out = {}
    for c in fig._qt_controls:
        if isinstance(c, widgets.Control):
            for w, _ in _text_widgets(c):
                out[id(w)] = (w, w.text())
    return out


@check("I1 text size: one rule 0.5x to 4x for every window kind at its design size, 2x and 0.5x; no text cut")
def _text_rule(b, exp):
    from sm4_Browser.qtplot import DPI
    wins = _item1_windows(b)
    assert len(wins) >= 12, [k for k, _ in wins]
    counts = {}
    for kind, fig in wins:
        W0, H0 = fig._design_px
        _settle_dock()                  # the info dock a title's details open appears on a short timer
        for factor in (1.0, 2.0, 0.5, 1.0):
            for _ in range(2):          # twice: the dock and the toolbar take window height, the canvas gets the rest
                fig.set_size_inches(W0 / DPI * factor, H0 / DPI * factor)
                plt._app().processEvents()
            counts[kind] = _assert_text_rule(kind, fig, factor)
    assert all(n_axes + n_ctrl >= 1 for n_axes, n_ctrl in counts.values()), counts   # the topo-rows pop-up has controls only
    assert sum(n_axes for n_axes, _ in counts.values()) >= 20 and sum(n_ctrl for _, n_ctrl in counts.values()) > 40, counts
    b._item1_windows = wins


@check("I2 short labels and stamp titles when a window is taller than wide, and back")
def _portrait(b, exp):
    from sm4_Browser.qtplot import DPI, file_stamp, is_file_name
    from sm4_Browser import widgets
    wins = getattr(b, '_item1_windows', None) or _item1_windows(b)
    n_short = n_stamp = 0
    for kind, fig in wins:
        W0, H0 = fig._design_px
        assert not fig.portrait, kind
        full = _label_map(fig)
        titles = [(ax, ax._title_full) for ax in fig.axes if ax._title is not None and ax._title_full]
        # a shape taller than wide for every window, the wide composer strip included
        pw = min(0.3 * W0, 0.5 * H0)
        for _ in range(2):
            fig.set_size_inches(pw / DPI, H0 / DPI)
            plt._app().processEvents()
        assert fig.portrait, (kind, fig._canvas_size())
        _assert_text_rule(kind, fig, pw / W0)
        for w, text in full.values():
            want = widgets.short_form(text)
            assert w.text() == want, (kind, text, w.text(), want)
            n_short += want != text
        for ax, tfull in titles:
            name = plt.short_title(tfull)
            if is_file_name(name):
                assert ax.get_title() == file_stamp(name), (kind, ax.get_title(), name)
                n_stamp += 1
            else:
                assert ax.get_title() == name, (kind, ax.get_title(), name)
        fig.set_size_inches(W0 / DPI, H0 / DPI)
        plt._app().processEvents()
        assert not fig.portrait, kind
        for w, text in full.values():
            assert w.text() == text, (kind, text, w.text())
        for ax, tfull in titles:
            assert ax.get_title() == plt.short_title(tfull), (kind, ax.get_title())
    assert n_short >= 10 and n_stamp >= 2, (n_short, n_stamp)
    for kind, fig in wins:
        plt.close(fig)


@check("I3 the browser window: the one rule, its native controls' labels short in portrait, the embedded figures follow")
def _browser_text(b, exp):
    from sm4_Browser.qtplot import QtGui, QtWidgets, window_zoom, screen_scale, ZOOM_MIN, ZOOM_MAX
    from sm4_Browser.browser import QControl
    from sm4_Browser import widgets
    k = screen_scale()
    W0, H0 = b.DESIGN
    mine = [c for c in list(QControl._all) if b.win.isAncestorOf(c.qt)]
    assert len(mine) > 20, len(mine)

    def texts():
        out = []
        for c in mine:
            for w in [c.qt] + c.qt.findChildren(QtWidgets.QWidget):
                if hasattr(w, 'text') and str(w.text()).strip() and not isinstance(w, QtWidgets.QLineEdit) and w.isVisible():
                    out.append(w)
            if isinstance(c.qt, QtWidgets.QGroupBox) and c.qt.title():
                out.append(c.qt)
        return out

    for factor in (1.0, 2.0, 0.5):
        b.win.resize(int(W0 * k * factor), int(H0 * k * factor))
        plt._app().processEvents()
        z = window_zoom(b.win.width(), b.win.height(), W0 * k, H0 * k) * k
        zb = b._zoomer.zoom
        assert z * b._zoomer.FIT_MIN - 1e-9 <= zb <= z * 1.03 + 1e-9, (factor, zb, z)    # the rule, or fitted below it
        assert abs(b.win.font().pointSizeF() - b._zoomer.base.pointSizeF() * zb / k) < 0.05, factor
        lay = b.win.centralWidget().layout()
        assert lay.minimumSize().width() <= b.win.width() and lay.minimumSize().height() <= b.win.height(), \
            (factor, lay.minimumSize(), b.win.size())
        for w in texts():
            text = w.title() if isinstance(w, QtWidgets.QGroupBox) else w.text()
            need = QtGui.QFontMetricsF(w.font()).horizontalAdvance(text)
            if isinstance(w, (QtWidgets.QCheckBox, QtWidgets.QRadioButton)):
                need += 24
            if not isinstance(w, QtWidgets.QGroupBox):
                assert need <= w.width() + 1.0, (factor, text, need, w.width())
        assert not b.win.height() > b.win.width() or b._zoomer.portrait
    before = {id(w): (w, w.title() if isinstance(w, QtWidgets.QGroupBox) else w.text()) for w in texts()}
    b.win.resize(int(W0 * k * 0.3), int(H0 * k * 1.6))
    plt._app().processEvents()
    assert b.win.height() > b.win.width(), (b.win.width(), b.win.height())
    assert b._zoomer.portrait and b.fig_scan.portrait and b.fig_spec.portrait
    n_short = 0
    for w, text in before.values():
        now = w.title() if isinstance(w, QtWidgets.QGroupBox) else w.text()
        assert now == widgets.short_form(text), (text, now)
        n_short += now != text
    assert n_short >= 10, n_short
    assert b.btn_section.qt.text() == 'section'
    b.win.resize(int(W0 * k), int(H0 * k))
    plt._app().processEvents()
    assert not b._zoomer.portrait and not b.fig_scan.portrait
    for w, text in before.values():
        now = w.title() if isinstance(w, QtWidgets.QGroupBox) else w.text()
        assert now == text, (text, now)


# ---------------------------------------------------------------------- #
#  Item 2: the two stores by origin, the carry-over, the rules; the flags   #
# ---------------------------------------------------------------------- #

def _store_files(directory):
    """(settings path, computed path, stamp) of the directory's two stores (caches.py)."""
    from sm4_Browser.caches import stamp_of
    from sm4_Browser.sm4file import list_sm4_files
    stamp = stamp_of(list_sm4_files(directory))
    return (os.path.join(directory, f"settings_{stamp}.json"), os.path.join(directory, f"computed_{stamp}.json"), stamp)


def _json(path):
    """A JSON file read whole."""
    import json
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


@check("J1 stores: settings and computed files by origin; a setting never changes the computed file; deleted, it is rebuilt the same")
def _stores(b, exp):
    import json
    s_path, c_path, stamp = _store_files(b.directory)
    assert os.path.isfile(s_path) and os.path.isfile(c_path), (s_path, c_path)
    assert not any(n.startswith(('drift_cache_', 'sites_', 'headers_')) for n in os.listdir(b.directory)), os.listdir(b.directory)
    comp = _json(c_path)
    for sec in ('headers', 'metrics', 'pairs'):
        assert sec in comp and comp[sec]['loader_format'] == 2 and comp[sec]['content'], sec
    names = [os.path.basename(p) for p in b.files if b.kinds[p] == 'topo']
    pairs = comp['pairs']['content']['pairs']
    assert all(f"{a}|{c}" in pairs for a, c in zip(names, names[1:])), "the consecutive pairs of the build"
    assert all('sig' in rec for rec in pairs.values()) and len(comp['headers']['content']['files']) == 5
    bytes0 = open(c_path, 'rb').read()
    # settings: a manual span, a stripe state, a flag: the settings file holds them at once, the computed file is untouched
    topos = [p for p in b.files if b.kinds[p] == 'topo']
    grid = next(p for p in b.files if b.kinds[p] == 'grid')
    point = next(p for p in b.files if b.kinds[p] == 'point')
    b.drift.set_manual_span(topos[0], topos[-1], np.zeros((0, 6)), kind='typed', v=np.array([2e-12, -1e-12]))
    b.drift.set_fast_state(b.loaded.get(grid) or b._load(grid), 'auto', 'auto')
    b.sites.set_flag(point, 'pulsing')
    st = _json(s_path)
    assert len(st['drift']['spans']) == 1 and st['drift']['spans'][0]['kind'] == 'typed'
    assert st['drift']['stripe'][os.path.basename(grid)]['auto']['fast'] == 'auto'
    assert st['flags'] == {os.path.basename(point): 'pulsing'}
    assert open(c_path, 'rb').read() == bytes0, "a setting changed the computed file"
    # back to the automatic state: the overrides go, the files follow
    b.drift.clear_manual_span(topos[0], topos[-1])
    b.drift.set_fast_state(b.loaded.get(grid) or b._load(grid), 'auto', 'off')
    b.sites.set_flag(point, 'experiment')
    st = _json(s_path)
    assert st['drift']['spans'] == [] and st['flags'] == {}
    assert open(c_path, 'rb').read() == bytes0
    # the computed file deleted: a second browser rebuilds it with the same content
    os.remove(c_path)
    b2 = make_browser(b.directory)
    try:
        comp2 = _json(c_path)
        for sec in ('headers', 'metrics'):
            assert comp2[sec]['content'] == comp[sec]['content'], sec
        # the build tracks the consecutive pairs again, the same; pairs the session tracked beyond them
        # (the drift window stepped to one) are not part of a build
        p2, p1 = comp2['pairs']['content']['pairs'], comp['pairs']['content']['pairs']
        assert set(p2) <= set(p1) and all(p2[k] == p1[k] for k in p2), (set(p2), set(p1))
        assert comp2['pairs']['content']['topos'] == comp['pairs']['content']['topos']
        assert _json(s_path) == st
    finally:
        b2._bg_shutdown()
        b2.win.close()
        plt._app().processEvents()


@check("J2 stores: the earlier caches are carried over once into the two files and renamed .old")
def _carry_over(b, exp):
    import json
    from sm4_Browser.checks import synth
    s_path, c_path, stamp = _store_files(b.directory)
    comp, sett = _json(c_path), _json(s_path)
    tmp = tempfile.mkdtemp(prefix='sm4synth_old_')
    exp2 = synth.standard(tmp)
    synth.install(exp2)
    try:
        topos = [os.path.basename(p) for p in b.files if b.kinds[p] == 'topo']
        point = os.path.basename(next(p for p in b.files if b.kinds[p] == 'point'))
        old_drift = {'format': 5, 'loader_format': 2, 'written': 'x', 'pairs': comp['pairs']['content']['pairs'],
                     'topos': comp['pairs']['content']['topos'],
                     'manual': {'spans': [{'a': topos[0], 'b': topos[-1], 'kind': 'typed', 'markers': [], 'v': [2e-12, -1e-12]}],
                                'stripe': {}}}
        old_sites = {'format': 1, 'loader_format': 2, 'written': 'x', 'topos': comp['metrics']['content']['topos'],
                     'spectra': comp['metrics']['content']['spectra'], 'flags': {point: 'pulsing'}}
        old_heads = {'format': 1, 'loader_format': 2, 'files': comp['headers']['content']['files']}
        for name, data in ((f"drift_cache_{stamp}.json", old_drift), (f"sites_{stamp}.json", old_sites),
                           (f"headers_{stamp}.json", old_heads)):
            with open(os.path.join(tmp, name), 'w') as fh:
                json.dump(data, fh)
        b2 = make_browser(tmp)
        try:
            names = set(os.listdir(tmp))
            for name in (f"drift_cache_{stamp}.json", f"sites_{stamp}.json", f"headers_{stamp}.json"):
                assert name + '.old' in names and name not in names, (name, names)
            s2, c2 = _json(os.path.join(tmp, f"settings_{stamp}.json")), _json(os.path.join(tmp, f"computed_{stamp}.json"))
            assert s2['drift']['spans'][0]['v'] == [2e-12, -1e-12] and s2['flags'] == {point: 'pulsing'}
            # the old cache's pairs (format 5, no signatures) are of another format: tracked again on the
            # copied files (other signatures), the same results for the consecutive pairs
            strip = lambda d: {k: {kk: v for kk, v in r.items() if kk != 'sig'} for k, r in d.items()}
            c2p, cp = strip(c2['pairs']['content']['pairs']), strip(comp['pairs']['content']['pairs'])
            assert len(c2p) >= 2 and all(c2p[k] == cp[k] for k in c2p), (c2p.keys(), cp.keys())
            # the files of the new directory have other modification times: the header scan and the
            # once-over read them again, keeping the same names
            assert set(c2['headers']['content']['files']) == set(comp['headers']['content']['files'])
            assert set(c2['metrics']['content']['topos']) == set(comp['metrics']['content']['topos'])
            # the carried-over settings are in force: the span governs the files between, the flag shows
            assert len(b2.drift.manual_spans) == 1 and b2.sites.flag(os.path.join(tmp, point)) == 'pulsing'
        finally:
            b2._bg_shutdown()
            b2.win.close()
            plt._app().processEvents()
    finally:
        synth.install(exp)


@check("J3 flags: a hand-set flag feeds the region, the spectra on it, the clean rule and the navigation; back to automatic drops the override")
def _flag_chain(b, exp):
    topos = [p for p in b.files if b.kinds[p] == 'topo']
    point = next(p for p in b.files if b.kinds[p] == 'point')
    grid = next(p for p in b.files if b.kinds[p] == 'grid')
    s = b.sites
    auto0 = dict(s.auto)
    assert all(v == 'experiment' for v in s.flags.values()), s.flags
    k = s.region_of(topos[0])
    assert k is not None and s.region_of(grid) == k and s.region_of(point) == k
    # a topo set to pulsing by hand: its region, every topo of it and the spectra on it are pulsing
    s.set_flag(topos[0], 'pulsing')
    assert s.region(k)['flag'] == 'pulsing' and s.region(k)['auto_flag'] == 'experiment'
    assert all(s.flag(p) == 'pulsing' for p in topos + [grid, point]), s.flags
    assert s.auto == auto0, "the automatic flags never change"
    assert s.neighbour_region(topos[0], +1) is None and s.neighbour_region(point, -1) is None
    assert 'pulsing' in s.region_text(grid)
    # a grid set to experiment by hand within the pulsing region: that file alone, and the clean rule runs on the final flags
    s.set_flag(grid, 'experiment')
    assert s.flag(grid) == 'experiment' and s.flag(point) == 'pulsing' and s.flag(topos[0]) == 'pulsing'
    files_before_grid = [p for p in b.files if b.files.index(p) < b.files.index(grid) and b.kinds[p] in ('grid', 'point')]
    assert not files_before_grid or all(s.flag(p) != 'clean' for p in files_before_grid)
    # the settings file carries exactly the overrides
    st = _json(_store_files(b.directory)[0])
    assert st['flags'] == {os.path.basename(topos[0]): 'pulsing', os.path.basename(grid): 'experiment'}
    s.set_flag(grid, 'pulsing')            # the chain's value without an override of its own: no override
    assert os.path.basename(grid) not in s.overrides and s.flag(grid) == 'pulsing'
    s.set_flag(grid, 'experiment')         # differs from it: stored again
    assert s.overrides == {os.path.basename(topos[0]): 'pulsing', os.path.basename(grid): 'experiment'}
    s.set_flag(topos[0], 'experiment')     # the topo's automatic value: dropped, the region back to experiment
    assert os.path.basename(topos[0]) not in s.overrides and s.region(k)['flag'] == 'experiment'
    s.set_flag(grid, 'experiment')         # now the chain's own value: dropped
    assert s.overrides == {}, s.overrides
    assert all(v == 'experiment' for v in s.flags.values())
    assert _json(_store_files(b.directory)[0])['flags'] == {}


@check("K1 scale timeline: the first entry covers everything, later ones from their file's time; every file and window at its own scale; drift readouts calibrated with the nominal; records note another scale and store theirs")
def _scale_timeline(b, exp):
    import io, contextlib
    from sm4_Browser.scales import file_time
    topos = [p for p in b.files if b.kinds[p] == 'topo']
    grid = next(p for p in b.files if b.kinds[p] == 'grid')
    point = next(p for p in b.files if b.kinds[p] == 'point')
    assert b.scales.entries == [] and all(b.scale_of(p) == 1.0 for p in b.files)
    # set on the grid: the first entry covers every file, the ones before it too
    b._goto(b.files.index(grid))
    assert b.box_scale.value == '1'
    b._on_scale("2.0")
    assert b.scale_of(grid) == 2.0 and b.scale_of(point) == 2.0 and b.scale_of(topos[0]) == 2.0
    assert b.box_scale.value == '2' and b.scalefactor == 2.0
    # set on the later point: from its time on; the grid and the earlier topos keep 2
    b._goto(b.files.index(point))
    b._on_scale("4.0")
    assert b.scale_of(point) == 4.0 and b.scale_of(topos[-1]) == 4.0 and b.scale_of(grid) == 2.0 and b.scale_of(topos[0]) == 2.0
    assert file_time(point) > file_time(grid)
    st = _json(_store_files(b.directory)[0])
    assert [e['scale'] for e in st['scales']['entries']] == [2.0, 4.0]
    # the scan area: every topo image at its own scale, the diamonds too
    b._goto(b.files.index(topos[-1]))
    plt._app().processEvents()
    assert abs(b.layer.items[topos[0]]['k'] - 1e9 / 2.0) < 1e-3 and abs(b.layer.items[topos[-1]]['k'] - 1e9 / 4.0) < 1e-3
    grey, blue = b._point_colours(b._visible_paths())
    assert any(p == point for p, _ in grey + blue)
    arts = b._draw_points(b._visible_paths())
    xy_point = np.asarray(b.layer.records[point]['xy_m']) * 1e9 / 4.0
    assert any(np.allclose(a, xy_point) for arr in arts for a in arr), "the point's diamond is not at its own scale"
    # every window opened for a file takes that file's scale
    w = open_line(b)
    assert w.scalefactor == 2.0
    m = open_map(b)
    assert m.scalefactor == 2.0
    b._goto(b.files.index(point))
    b._on_topo_window()
    assert b.children[-1].scalefactor == 4.0
    if b.drift_window is not None and b.drift_window.is_open():
        plt.close(b.drift_window.fig)              # a window left open by an earlier check would be raised, not replaced
    b._on_drift_window()
    d = b.drift_window
    assert d.scale == 4.0 and d.scale_of(d.a.path) == b.scale_of(d.a.path), (d.scale, d.f.name)
    # the drift window: each topo image and the other's frame at that topo's own scale
    for ax, own, other in ((d.ax_a, d.a, d.b), (d.ax_b, d.b, d.a)):
        x, y = d._outlines[ax][0].get_data()
        assert np.allclose(np.column_stack([x, y])[:-1], other.corners_m() * 1e9 / b.scale_of(other.path))
        im = ax.images[0]
        l, r, bt, tp = im.get_extent()
        assert abs((r - l) - abs(own.frame_extent_m()[1] - own.frame_extent_m()[0]) * 1e9 / b.scale_of(own.path)) < 1e-6
    # the readouts: calibrated with the nominal beside; the box calibrated, stored nominal
    v = np.array([4e-12, -2e-12])
    t = d._v_text(v)
    assert "calibrated (/ 4)" in t and f"({4e-12 * 3.6e12 / 4:+.3f}, {-2e-12 * 3.6e12 / 4:+.3f}) nm/h calibrated" in t and "nominal" in t
    d._fill_box(v, 'typed', None)
    assert d.box_v.value.startswith(f"{4e-12 * 3.6e12 / 4:.4g}")
    d.box_v.set("30, -20")
    d._on_typed("30, -20")
    assert np.allclose(d._parse_box() * 3.6e12, [30 * 4.0, -20 * 4.0]), "the box is calibrated: the stored velocity is nominal"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        d._on_apply()
    sp = b.drift.span_between(d.a.path, d.b.path)
    assert sp is not None and np.allclose(np.asarray(sp['v']) * 3.6e12, [120.0, -80.0])
    assert "calibrated" in buf.getvalue() and "nominal" in buf.getvalue()
    b.drift.clear_manual_span(d.a.path, d.b.path)
    plt.close(d.fig)
    # the console readout of a placed file's velocity carries the nominal and the calibrated value
    model = b._layer_model(topos[0], b.layer.records[topos[0]])
    assert model is not None, "the synthetic topo has an auto velocity"
    t = model.text(b.scale_of(topos[0]))
    assert "nm/h nominal" in t and "calibrated (/ 2)" in t, t
    # a record made with another scale: the timeline's applies, with a note
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sc = b._record_scale(b.loaded.get(grid) or b._load(grid), 2.75)
    assert sc == 2.0 and "2.75" in buf.getvalue() and "applies now" in buf.getvalue()
    # a Line window's marker record keeps the scale it was made with (the loader reads 'scalefactor' back)
    assert w.marker_record().get('scalefactor') == w.scalefactor == 2.0, w.marker_record().get('scalefactor')
    # back to the base scale for the checks that follow
    b.scales.remove(point)
    b.scales.remove(grid)
    assert b.scales.entries == [] and b.scale_of(point) == 1.0
    b._goto(0)
    plt.close(w._int_fig)
    plt.close(m.fig)


def _mouse(b, name, x, y, button=1, key=None):
    """A mouse event on the scan area at data (x, y): the display position filled in, so the path's grab works."""
    from sm4_Browser.qtplot import MouseEvent
    ax, canvas = b.ax_scan, b.canvas_scan
    px, py = ax.transData.transform((x, y))
    e = MouseEvent(name, canvas, float(px), float(py), button=button)
    e.inaxes, e.xdata, e.ydata = ax, float(x), float(y)
    canvas._fire(name, e)
    plt._app().processEvents()
    return e


def _path_click(b, x, y, button=1):
    """A press and release on the scan area at data (x, y): a path point while editing (button 3 removes one)."""
    _mouse(b, 'button_press_event', x, y, button)
    _mouse(b, 'button_release_event', x, y, button)


@check("L1 section path: editing with frozen navigation, add / drag / remove, off keeps the points; measure, set scale (two-way, apply), cross section along the path, clear")
def _section_path(b, exp):
    from sm4_Browser.qtplot import Qt, MouseEvent
    from PySide6.QtTest import QTest
    from sm4_Browser.section import path_lengths
    grid = next(p for p in b.files if b.kinds[p] == 'grid')
    b._goto(b.files.index(grid))
    b._on_scale("2.0")                       # the grid's scale: nominal = calibrated x 2
    settle(b)                                # the background placing may relocate the view once more
    sec, view, tb = b.section, b.view, b.toolbar_scan
    assert not sec.editing and not sec.active() and not view.frozen
    x0, x1 = b.ax_scan.get_xlim(); y0, y1 = b.ax_scan.get_ylim()
    cx, cy, w = 0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.25 * (x1 - x0)
    # editing on: the view is frozen, the toolbar's Home, Pan and Zoom are off
    b._on_section()
    assert sec.editing and view.frozen and not tb.a_home.isEnabled() and not tb.a_pan.isEnabled() and not tb.a_zoom.isEnabled()
    e = MouseEvent('scroll_event', b.canvas_scan, 0, 0, button='up', step=1)
    e.inaxes, e.xdata, e.ydata = b.ax_scan, cx, cy
    b.canvas_scan._fire('scroll_event', e)
    QTest.keyClick(b.canvas_scan, Qt.Key.Key_Home)
    tb._home()
    assert b.ax_scan.get_xlim() == (x0, x1) and b.ax_scan.get_ylim() == (y0, y1), "the view moved while editing"
    # clicks add points, numbered; a drag moves one; a right click removes one
    _path_click(b, cx - w, cy)
    _path_click(b, cx, cy + 0.5 * w)
    _path_click(b, cx + w, cy)
    pts = sec.points()
    assert pts is not None and len(pts) == 3 and np.allclose(pts[1], [cx, cy + 0.5 * w])
    assert sum(1 for a in sec._arts if hasattr(a, 'text')) == 3, "three numbered points"
    _mouse(b, 'button_press_event', cx, cy + 0.5 * w)
    _mouse(b, 'motion_notify_event', cx + 0.1 * w, cy + 0.7 * w)
    _mouse(b, 'button_release_event', cx + 0.1 * w, cy + 0.7 * w)
    pts = sec.points()
    assert len(pts) == 3 and np.allclose(pts[1], [cx + 0.1 * w, cy + 0.7 * w]), pts
    _path_click(b, cx + w, cy, button=3)
    pts = sec.points()
    assert len(pts) == 2 and np.allclose(pts[0], [cx - w, cy]), pts
    _path_click(b, cx + w, cy - 0.3 * w)
    assert len(sec.points()) == 3
    assert b.ax_scan.get_xlim() == (x0, x1), "a click on the scan area is a point, never a pan"
    # editing off: the points stay, the view moves again; on again: the same points
    kept = sec.points().copy()
    b._on_section()
    assert not sec.editing and not view.frozen and tb.a_home.isEnabled() and tb.a_pan.isEnabled()
    b.canvas_scan._fire('scroll_event', e)
    assert b.ax_scan.get_xlim() != (x0, x1), "the scroll zooms again"
    b.view.keep(x0, x1, y0, y1); b.view.view = (x0, x1, y0, y1); b.view._apply()
    assert np.allclose(sec.points(), kept)
    b._on_section()
    assert sec.editing and np.allclose(sec.points(), kept)
    # measure: nominal = calibrated x 2, the lengths too; using it ends the editing
    b._on_measure()
    assert not sec.editing and b.measure_window is not None and b.measure_window.is_open()
    lines = b.measure_window.lines
    seg, total = path_lengths(kept)
    assert any(f"{kept[0][0] * 2:12.3f}" in l and f"{kept[0][0]:12.3f}" in l for l in lines), lines
    assert any(l.strip().startswith('total') and f"{total * 2:14.3f}" in l and f"{total:16.3f}" in l for l in lines), lines
    # set scale: the nominal length fixed, the two boxes recompute each other, apply sets the timeline and rescales the path
    b._on_set_scale()
    sw = b.scale_window
    assert sw is not None and sw.is_open() and abs(sw.nominal - total * 2) < 1e-9
    assert sw.box_scale.value == '2' and abs(float(sw.box_cal.value) - total) < 1e-3
    sw._on_scale("4")
    assert abs(float(sw.box_cal.value) - total * 2 / 4) < 1e-3
    sw._on_cal(f"{total * 2 / 8:.6f}")
    assert abs(float(sw.box_scale.value) - 8.0) < 1e-3
    sw._on_apply()
    assert abs(b.scale_of(grid) - 8.0) < 1e-3 and b.box_scale.value.startswith('8')
    assert np.allclose(sec.points(), kept * 2 / 8), "the path keeps its nominal place when the scale changes"
    assert abs(sw.nominal - total * 2) < 1e-6 and b.measure_window.lines[0].find('scale factor 8') > 0
    # the cross section along the whole path
    b._on_section()
    b._on_cross_section()
    assert not sec.editing and b.profile_window is not None and b.profile_window.is_open()
    x = b.profile_window.pair['forward'][0]
    assert abs(x[-1] - path_lengths(sec.points())[1]) < 1e-6 and len(x) > 10
    # clear: the path goes, the editing state stays
    b._on_clear_path()
    assert not sec.active() and not sec.editing and b.measure_window.lines[0].startswith('no path')
    for win in (b.measure_window, b.scale_window, b.profile_window):
        plt.close(win.fig)
    b.scales.remove(grid)
    assert b.scale_of(grid) == 1.0
    b._goto(0)


@check("M1 anchor to topo: the browser's toggle and the Topo window's; the file moved rigidly by the reported shift, off restores it")
def _anchor_to_topo(b, exp):
    import io, contextlib
    point = goto_kind(b, 'point')
    assert b.anchor_on is False and b.chk_anchor.value is False, "the toggle starts off"
    d_off = b.drift.result(point, b.loaded, None, b.auto_on, b.manual_on, False)
    assert d_off is not None and not d_off['anchor'] and np.allclose(d_off['anchor_shift'], 0)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        b.chk_anchor.set(True)
        b._on_drift_toggle('anchor', True)
    assert b.anchor_on is True and "anchor to topo on." in buf.getvalue(), buf.getvalue()
    d_on = b.drift.result(point, b.loaded, None, b.auto_on, b.manual_on, True)
    sh = np.asarray(d_on['anchor_shift'])
    assert d_on['anchor'] and np.linalg.norm(sh) > 0, "a velocity and an earlier topo: the rigid shift is not zero"
    assert np.allclose(d_on['xy_corr'] - d_off['xy_corr'], sh[None, :]), "every position moved by the one shift"
    assert any("anchor to topo: the file moved rigidly" in line for line in d_on['report']), d_on['report']
    # the Topo window takes the browser's state and has its own toggle
    b._on_topo_window()
    w = b.children[-1]
    assert w.anchor_on is True and w.chk_anchor.value is True
    first = lambda: next(np.array([ln.get_data()[0][0], ln.get_data()[1][0]]) for ln in w.ax.lines if ln.get_label() == 'first sweep')
    p_on = first()
    w.chk_anchor.set(False)
    w._on_drift_toggle('anchor', False)
    assert w.anchor_on is False and b.anchor_on is True, "the window's toggle is independent of the browser's"
    p_off = first()
    k = 1e9 / w.scalefactor
    expect = (w.topo.to_frame_m(d_on['xy_corr'][:1]) - w.topo.to_frame_m(d_off['xy_corr'][:1]))[0] * k
    assert np.allclose(p_on - p_off, expect, atol=1e-6), (p_on - p_off, expect)
    plt.close(w.fig)
    # off again in the browser: the recorded anchor stands
    b.chk_anchor.set(False)
    b._on_drift_toggle('anchor', False)
    assert b.anchor_on is False
    b._goto(0)


@check("M2 the Line window opens with 'calibrate' and 'normalize' off, like every other window")
def _line_flags_off(b, exp):
    import inspect
    from sm4_Browser import browser as browser_mod
    src = inspect.getsource(browser_mod.main)
    assert "'apply_calibration': False" in src and "'apply_normalize': False" in src, "main() opens the Line window with both off"
    assert b.line_window_kwargs.get('apply_calibration', False) is False and b.line_window_kwargs.get('apply_normalize', False) is False
    w = open_line(b)
    assert w._int_apply_calibration is False and w._int_apply_normalize is False
    assert w._int_pipeline_box.state() == {'calibrate': False, 'normalize': False}
    plt.close(w._int_fig)
    b._goto(0)


@check("M3 console strings: the settings file, the calibrated pair readout, the Line window's toggle line, the averaging filter line, the filter box label")
def _console_strings(b, exp):
    import io, contextlib
    from sm4_Browser.drift import marker_pair_text
    assert b.filters._title == 'filters' and b.filters.qt.title() == 'filters'
    # the site flag line names the settings file
    grid = goto_kind(b, 'grid')
    was = b.sites.flag(grid.path)
    other = 'experiment' if was != 'experiment' else 'pulsing'
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        b._on_site(other)
    assert "kept in the settings file" in buf.getvalue() and "sites file" not in buf.getvalue(), buf.getvalue()
    b.sites.set_flag(grid.path, was)
    assert b.sites.flag(grid.path) == was
    # the drift window prints every pair with the calibrated value beside the nominal
    b._on_scale("2.0")
    if b.drift_window is not None and b.drift_window.is_open():
        plt.close(b.drift_window.fig)
    b._on_drift_window()
    d = b.drift_window
    rec = {'dx': 1e-9, 'dy': 2e-9, 'dL': 2.236e-9, 'dt': 3600.0, 'used': True, 'vx': 1e-9 / 3600, 'vy': 2e-9 / 3600,
           'vL': 2.236e-9 / 3600, 'alpha': 63.4, 'rx': 0.0, 'ry': 0.0}
    assert "[+0.500 cal]" in marker_pair_text(0, rec, d.scale), marker_pair_text(0, rec, d.scale)
    a_c = 0.5 * (d.a.corners_m()[0] + d.a.corners_m()[2])
    k = 1e9 / b.scale_of(d.a.path)
    click(d.fig, d.ax_a, *(a_c * k))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        d._on_set_from_markers()
    assert "pair 1:" in buf.getvalue() and " cal]" in buf.getvalue(), buf.getvalue()
    d.markers.set_pairs(0)
    plt.close(d.fig)
    b.scales.remove(grid.path)
    # the Line window's toggle line: plain rows without a placement, the stripe matching named with one
    w = open_line(b)
    f = w._int_sm4()
    setting = b.drift.slow_setting(f, False, False)
    assert b.drift.fast_state(f, setting, False, False) == 'off'
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        w._int_on_drift_toggle('auto', False)
        w._int_on_drift_toggle('manual', False)
    assert "(rows plain)." in buf.getvalue(), buf.getvalue()
    b.drift.set_fast_state(f, setting, 'auto')
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        w._int_on_drift_toggle('auto', True)
        w._int_on_drift_toggle('auto', False)
    assert "(rows placed by the stripe matching alone)." in buf.getvalue(), buf.getvalue()
    b.drift.set_fast_state(f, setting, 'off')
    plt.close(w._int_fig)
    # the averaging window's filter line names each filter's targets (every toggle off at open)
    goto_kind(b, 'point')
    b._on_add_new_avg()
    aw = list(b.averages.open().values())[-1]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        aw._on_filter_change()
    assert "Hampel off; SG off" in buf.getvalue(), buf.getvalue()
    aw.filters.set_params(None, None, {'window': 4, 'n_sigma': 3}, None)   # Hampel on dI/dV only
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        aw._on_filter_change()
    assert "Hampel window 4, n_sigma 3 on dI/dV; SG off" in buf.getvalue(), buf.getvalue()
    aw.filters.set_params(None, None)
    plt.close(aw.fig)
    b._goto(0)


@check("M4 occlusion at per-file scales: an image is hidden only when a later complete image covers it as the two are drawn")
def _occlusion_scales(b, exp):
    topos = [p for p in b.files if b.kinds[p] == 'topo']
    last = b.files.index(topos[-1])
    b._goto(last)
    settle(b)
    assert b.layer.covers.get(topos[0], []) == [], "at one scale no synthetic topo lies wholly inside another"
    assert b.layer.items[topos[0]]['im'].get_visible()
    # the first topo drawn at scale 2 (half size) under the second at scale 1: covered as drawn, not nominally
    b._goto(b.files.index(topos[0]))
    b._on_scale("2.0")
    b._goto(b.files.index(topos[1]))
    b._on_scale("1.0")
    b._goto(last)
    settle(b)
    assert b.scale_of(topos[0]) == 2.0 and b.scale_of(topos[1]) == 1.0
    assert topos[1] in b.layer.covers.get(topos[0], []), b.layer.covers
    assert not b.layer.items[topos[0]]['im'].get_visible(), "the covered image is not painted"
    # one scale again: nothing covered, the image back
    b.scales.remove(topos[0])
    b.scales.remove(topos[1])
    b._layer_refresh()
    settle(b)
    assert b.layer.covers.get(topos[0], []) == [] and b.layer.items[topos[0]]['im'].get_visible()
    b._goto(0)


@check("M5 tracked pairs: stored with the files' signatures, tracked again when a file changed, written at once when tracked outside the build")
def _tracked_pairs(b, exp):
    import io, contextlib, time
    from sm4_Browser import sm4file
    topos = [p for p in b.files if b.kinds[p] == 'topo']
    d = b.drift
    names = [os.path.basename(p) for p in topos]
    key = d.tracked.pair_key(names[0], names[1])
    assert d.tracked.pairs[key].get('sig') == [sm4file._file_sig(topos[0]), sm4file._file_sig(topos[1])], d.tracked.pairs[key].get('sig')
    # the later file changed (another modification time): the stored pair no longer counts, a new session tracks it again
    now = time.time() + 5
    os.utime(topos[1], (now, now))
    sig1 = sm4file._file_sig(topos[1])
    assert d.tracked.get_pair(names[0], names[1]) is not None, "without signatures the stored pair is returned"
    assert d.tracked.get_pair(names[0], names[1], sm4file._file_sig(topos[0]), sig1) is None
    d.pair_cache.clear()                                   # a new session's memo
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        est = d.pair(d.refs[topos[0]], d.refs[topos[1]])
    assert est.get('v') is not None and "drift:" in buf.getvalue(), buf.getvalue()
    assert d.tracked.pairs[key]['sig'] == [sm4file._file_sig(topos[0]), sig1]
    comp = _json(_store_files(b.directory)[1])
    assert comp['pairs']['content']['pairs'][key]['sig'][1] == sig1, "tracked outside the build: written at once"
    # a pair never tracked before (not consecutive): tracked now and written
    key2 = d.tracked.pair_key(names[0], names[2])
    d.tracked.pairs.pop(key2, None)                       # as if never tracked (an earlier check may have stepped to it)
    d.pair_cache.pop((topos[0], topos[2]), None)
    with contextlib.redirect_stdout(io.StringIO()):
        d.pair(d.refs[topos[0]], d.refs[topos[2]])
    assert key2 in _json(_store_files(b.directory)[1])['pairs']['content']['pairs']


@check("M7 Map window: the stripe matching alone places the pixels, widens the frame and draws the lines, the titles say 'stripe matched'; one energy label per grid cell")
def _map_stripe_only(b, exp):
    grid = goto_kind(b, 'grid')
    m = open_map(b)
    m._on_drift_toggle('auto', False)
    m._on_drift_toggle('manual', False)
    assert m._drift_result() is None and m._frame_lines() is None
    assert 'drift corrected' not in m.ax_map._title_full and 'stripe matched' not in m.ax_map._title_full
    setting = b.drift.slow_setting(grid, False, False)
    b.drift.set_fast_state(grid, setting, 'auto')          # the Map window is a subscriber: _refresh_drift
    plt._app().processEvents()
    res = m._drift_result()
    assert res is not None and str(res.get('source', '')).startswith('none'), res and res.get('source')
    assert m._frame_lines() is not None and m._bounds is not None, "the lines and the widened frame under the stripe matching alone"
    t = m.ax_map._title_full                                 # the panel shows the short title; the details are in the full one
    assert 'stripe matched' in t and 'drift corrected' not in t, t
    assert 'stripe matched' in m.ax_topo._title_full, m.ax_topo._title_full
    # the grid view keeps one energy label per cell however often it is redrawn
    m.toggle_view()
    n0 = [len(ax.texts) for ax in m.grid_axes]
    assert all(n == 1 for n in n0), n0
    m._refresh_drift()
    m._on_drift_toggle('auto', True)
    m._on_drift_toggle('auto', False)
    assert [len(ax.texts) for ax in m.grid_axes] == n0, [len(ax.texts) for ax in m.grid_axes]
    b.drift.set_fast_state(grid, setting, 'off')
    plt.close(m.fig)
    b._goto(0)


@check("M8 RowHeatmap: a shear with fewer rows than the image draws and reports its extent as the drawing pads it")
def _row_heatmap_short_shear(b, exp):
    from sm4_Browser import plots
    fig, ax = plt.subplots()
    img = np.arange(20, dtype=float).reshape(4, 5)
    energy = np.linspace(-1.0, 1.0, 5)
    rows = np.arange(4, dtype=float)
    short = np.full((2, 6), -0.25)
    hm = plots.RowHeatmap(ax, img, energy, rows, 'jet', shear=short)
    x0, x1, y0, y1 = hm.extent()
    full = plots.RowHeatmap(ax, img, energy, rows, 'jet', shear=np.vstack([short, np.zeros((2, 6))]))
    assert np.allclose((x0, x1, y0, y1), full.extent()), (hm.extent(), full.extent())
    assert abs(y0 - (float(np.min(hm.y_edges)) - 0.25)) < 1e-12 and abs(y1 - float(np.max(hm.y_edges))) < 1e-12
    assert np.allclose(hm._quads, full._quads)
    plt.close(fig)


@check("M6 the fit: norm_range is kept in the fit history; the barrier height's lower bound is one rule for the fit and the edit panel")
def _fit_history(b, exp):
    import inspect
    from sm4_Browser.fit import Fit
    V = np.linspace(-1.0, 1.0, 101)
    data = np.exp(-((V - 0.2) / 0.1) ** 2) + 0.2
    f = Fit.from_curve(V, data, data, -1.0, 1e-10, np.array([1e-10]), 'synthetic curve', norm_range=(-0.5, 0.5))
    f._shown, f._back, f._fwd = None, [], []            # as a run leaves them before a fit is accepted
    state = f._cache_state()
    assert state['norm_range'] == (-0.5, 0.5) and state['format'] == Fit._CACHE_FORMAT, state.get('norm_range')
    assert f._bg_bounds()[0][0] == f._phi_min() == 0.5 * float(np.max(np.abs(f.V))) + 0.05
    assert "self._phi_min()" in inspect.getsource(Fit._edit_on_param), "the edit panel refuses by the same rule"
    assert Fit._CACHE_FORMATS_READ == (3,) and not hasattr(Fit, '_migrate_format_2')


@check("M9 qtplot: a title re-set without a size keeps the size last set; a decimal is not a file name; the datalim widening applies a numeric aspect")
def _qtplot_rules(b, exp):
    from sm4_Browser.qtplot import is_file_name, short_title
    assert not is_file_name('gamma 0.5') and short_title('gamma 0.5') == 'Gamma 0.5'
    assert is_file_name('Synth_CT_2024_01_01_10_05_01_440.sm4') and is_file_name('x.tif') and is_file_name('a.b1')
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title('one', fontsize=9)
    ax.set_title('two')
    assert ax._title_size == 9.0 and ax._title._size == 9.0, (ax._title_size, ax._title._size)
    ax.title.set_fontsize(7)
    ax.set_title('three')
    assert ax._title_size == 7.0 and ax._title._size == 7.0
    # datalim with aspect 2: pyqtgraph's lock makes the y units per pixel twice the x units per pixel;
    # both requested ranges are held, the short one widened about its centre
    ax.set_aspect(2, adjustable='datalim')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    plt._app().processEvents()
    x0, y0, w, h = ax._px_rect()
    (xa, xb), (ya, yb) = ax.vb.viewRange()
    assert abs((xb - xa) / (yb - ya) - (w / h) / 2.0) < 1e-6 * (w / h), ((xb - xa) / (yb - ya), (w / h) / 2.0)
    assert xa <= 0 + 1e-9 and xb >= 10 - 1e-9 and ya <= 0 + 1e-9 and yb >= 10 - 1e-9, (xa, xb, ya, yb)
    assert abs((xa + xb) - 10) < 1e-9 and abs((ya + yb) - 10) < 1e-9, "widened about the centres"
    plt.close(fig)


@check("M10 records and helpers: an old fit-history family listed once by its stem; a line-average member defaults to the forward row; the Hampel filter returns float64; zero_crossing's tag rule")
def _records_helpers(b, exp):
    import ast
    from sm4_Browser.records import SavedAveragesWindow, OLD_CACHE_RE
    from sm4_Browser.lineavg import member_from_file
    from sm4_Browser.filters import hampel
    names = ['2024_01_01_10_00_00_average_of_a_b_fits.pkl', '2024_01_01_10_00_00_average_of_a_b_fits_2.pkl',
             'other.pkl', 'x_average_of_c_fits_3.pkl', 'avg1_average.json']
    assert all(OLD_CACHE_RE.search(n) for n in (names[0], names[1], names[3])) and not OLD_CACHE_RE.search(names[2])
    assert SavedAveragesWindow.old_family_stems(names) == ['2024_01_01_10_00_00_average_of_a_b', 'x_average_of_c']
    grid = goto_kind(b, 'grid')
    m = member_from_file(grid, 0, 1.0)
    assert [tuple(x) for x in m['topo_lines']] == [(0, 'forward')], m['topo_lines']
    out = hampel(np.arange(12).reshape(12, 1), window=2)
    assert out.dtype == np.float64 and out.shape == (12, 1)
    # zero_crossing is a script over real data: its tag rule compiled alone from the file
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'zero_crossing.py')
    tree = ast.parse(open(path).read())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'tag_of')
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), path, 'exec'), ns)
    assert ns['tag_of']('A_CT_2024_01_01_10_00_00_000.sm4') == '2024_01_01_10_00_00_000'
    assert ns['tag_of']('plain_2024.sm4') == 'plain_2024'
    b._goto(0)


def _drawn_pts(ax):
    """Every finite point drawn on ax as the windows draw: the lines' data and the collections' quads (an image is passed by its corners)."""
    pts = []
    for ln in ax.lines:
        x, y = ln.get_xdata(), ln.get_ydata()
        if len(x) == len(y):
            pts.append(np.column_stack([x, y]))
    for c in ax.collections:
        if hasattr(c, '_segments'):                      # a LineCollection (the drift lines)
            pts.extend(np.asarray(seg, dtype=np.float64).reshape(-1, 2) for seg in c._segments)
        else:                                            # a PolyCollection (the grid's stripes)
            pts.append(np.asarray(c._verts, dtype=np.float64).reshape(-1, 2))
    return pts


def _assert_edge_to_edge(ax, *extra, what=''):
    """The limits of ax are exactly the bounding box of everything drawn (the lines, the collections, `extra`): nothing outside, no margin."""
    from sm4_Browser import plots
    plt._app().processEvents()                      # the deferred placement of the box runs (an equal aspect)
    box = plots.drawn_box(*(_drawn_pts(ax) + list(extra)))
    assert box is not None, what
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    tol = 1e-6 * max(x1 - x0, y1 - y0, 1e-12)
    assert np.allclose((x0, x1, y0, y1), box, atol=tol, rtol=0), (what, (x0, x1, y0, y1), box)
    return box


@check("N1 limits: every spatial panel edge to edge around everything drawn, no margin (the helpers; the Topo window, the scan area, the stripe window, the drift window; the Map window in G2)")
def _edge_to_edge(b, exp):
    from sm4_Browser import plots
    from sm4_Browser.scanlayer import record_corners
    # the helpers: the box of points and quads, non-finite skipped; the limits set to it exactly, a flat box widened
    box = plots.drawn_box(np.array([[1.0, 2.0], [np.nan, 0.0]]), None, np.zeros((0, 2)), np.array([[[3.0, -1.0]] * 4]))
    assert box == (1.0, 3.0, -1.0, 2.0), box
    assert plots.drawn_box(None, np.full((2, 2), np.nan)) is None
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    ax.plot([0.0, 1.0], [0.0, 1.0])
    assert plots.fit_limits(ax, np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([[5.0, -2.0]])) == (0.0, 5.0, -2.0, 1.0)
    assert np.allclose(ax.get_xlim(), (0.0, 5.0)) and np.allclose(ax.get_ylim(), (-2.0, 1.0))
    assert plots.fit_limits(ax, np.array([[2.0, 3.0]])) == (1.5, 2.5, 2.5, 3.5)
    assert plots.fit_limits(ax) is None and np.allclose(ax.get_xlim(), (1.5, 2.5))
    plt.close(fig)
    # pixel_bounds: every finite pixel's cell plus its shift (NaN as 0), a NaN pixel not counted; a cube of shifts at once
    img = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]])
    assert plots.pixel_bounds(img, [0, 3, 0, 2]) == (0.0, 3.0, 0.0, 2.0)
    assert plots.pixel_bounds(np.array([[1.0, np.nan], [np.nan, np.nan]]), [0, 2, 0, 2]) == (0.0, 1.0, 0.0, 1.0)
    sh = np.zeros((2, 3, 2))
    sh[0, 0] = (-0.5, -0.25)
    sh[0, 2] = (10.0, 10.0)                                      # the NaN pixel: not drawn, not counted
    sh[1, 2] = (np.nan, 0.5)
    assert plots.pixel_bounds(img, [0, 3, 0, 2], sh) == (-0.5, 3.0, -0.25, 2.5)
    assert plots.pixel_bounds(img, [3, 0, 2, 0], None) == (0.0, 3.0, 0.0, 2.0)               # a reversed extent
    cube = np.stack([sh, np.zeros((2, 3, 2)) + (0.0, 1.0)])
    assert plots.pixel_bounds(img, [0, 3, 0, 2], cube) == (-0.5, 3.0, -0.25, 3.0)
    assert plots.pixel_bounds(np.full((2, 3), np.nan), [0, 3, 0, 2]) is None
    # the windows, with a velocity on and the grid's stripe matching placed (lines and stripes drawn)
    f = goto_kind(b, 'grid')
    d = b.drift
    auto0, manual0 = b.auto_on, b.manual_on
    b._on_drift_toggle('auto', True)
    b._on_drift_toggle('manual', False)
    setting = d.slow_setting(f, b.auto_on, b.manual_on)
    fast0 = d.fast_state(f, setting)
    d.set_fast_state(f, setting, 'auto')
    # the Topo window: the frame, the corrected spectra, their drift lines and the grid's pixels, exactly
    b._on_topo_window()
    w = b.children[-1]
    k = 1e9 / w.scalefactor
    l, r, bb, t = [v * k for v in w.topo.frame_extent_m()]
    frame = np.array([[l, bb], [r, t]])
    _assert_edge_to_edge(w.ax, frame, what='Topo window, drift on')
    assert w.ax.collections and any(len(ln.get_xdata()) > 2 for ln in w.ax.lines), "the stripes and the drift lines are drawn"
    w._on_drift_toggle('auto', False)
    _assert_edge_to_edge(w.ax, frame, what='Topo window, drift off')
    plt.close(w.fig)
    # the scan area's full extent: the shown topos' corners, every position, the drift lines, the stripes, then the box's aspect
    b._draw_scan_area()
    corners = [record_corners(b.layer.records[p], b.k_of(p), b._direction_of_path(p), b._layer_model(p, b.layer.records[p]))
               for p, it in b.layer.items.items() if it['im'].get_visible() and p in b.layer.records]
    assert corners, "topos shown on the scan area"
    full_box = plots.drawn_box(*(_drawn_pts(b.ax_scan) + corners))
    assert np.allclose(b.view.full, b.view.fit_to_box(full_box)), (b.view.full, full_box)
    fx0, fx1, fy0, fy1 = b.view.full
    assert (abs(fx0 - full_box[0]) < 1e-9 and abs(fx1 - full_box[1]) < 1e-9) or (abs(fy0 - full_box[2]) < 1e-9 and abs(fy1 - full_box[3]) < 1e-9), \
        "one axis of the full extent is the drawn box exactly (the other widened to the box's aspect)"
    # the stripe window: the reference topos' frames, the stripes, the spectra, the start and the handles
    if b.stripe_window is not None:                  # an earlier check's window, closed or not: a fresh one
        try:
            plt.close(b.stripe_window.fig)
        except Exception:
            pass
        b.stripe_window = None
    b._on_stripe_window()
    w = b.stripe_window
    assert w is not None and w._handles, "handles under the stripe matching"
    ref = d.stripe_reference(f, *w.flags())
    ref_corners = [e['f'].corners_m() * w.k for e in (ref.entries if ref else [])]
    _assert_edge_to_edge(w.ax, *ref_corners, what='stripe window')
    plt.close(w.fig)
    b.stripe_window = None
    # the drift window: one view edge to edge around the two frames and the file's spectra (G7 checks the rest)
    b._on_drift_window()
    w = b.drift_window
    plt._app().processEvents()
    allc = np.vstack([w.a.corners_m() * w.k_of(w.a.path), w.b.corners_m() * w.k_of(w.b.path)])
    xy = np.asarray(w.f.spec_xy_m, dtype=np.float64) * w.k_of(w.f.path)
    box = plots.drawn_box(allc, xy[np.asarray(w.f.acquired, dtype=bool)[:len(xy)]])
    for ax in (w.ax_a, w.ax_b):
        assert np.allclose(ax.get_xlim() + ax.get_ylim(), box), (ax.get_xlim(), ax.get_ylim(), box)
    plt.close(w.fig)
    d.set_fast_state(f, setting, fast0)
    b._on_drift_toggle('auto', auto0)
    b._on_drift_toggle('manual', manual0)
    b._goto(0)


def _row_heights(axes):
    """The placed heights (px) of the axes' boxes, after the deferred placement."""
    plt._app().processEvents()
    return [ax.vb.geometry().height() for ax in axes]


def _assert_row(axes, what=''):
    """Every axes of a row at one height (within a pixel); every aspect-locked one at its data aspect (within 1 %)."""
    hs = _row_heights(axes)
    assert max(hs) - min(hs) < 1.0, (what, hs)
    for ax in axes:
        want = ax._aspect_want()
        if want is not None:
            g = ax.vb.geometry()
            assert abs(g.height() / g.width() - want) < 0.01 * want, (what, g.width(), g.height(), want)
    return hs[0]


@check("N2 rows: every row of panels at one height, spatial panels keeping their aspect, the rest following (Map single and grid views, its companion, the Topo window, the composer, the drift window); one label between the map's colour bar and the curves")
def _rows(b, exp):
    from sm4_Browser import plots
    m = open_map(b)
    m._on_drift_toggle('auto', False)
    m._on_drift_toggle('manual', False)
    row = [m.ax_topo, m.ax_topo_cbar, m.ax_map, m.cax_map, m.ax_curve]
    assert m.ax_topo._row is row[0]._row and all(ax._row is row[0]._row for ax in row), "one row object"
    h_off = _assert_row(row, 'Map single view, drift off')
    cell_h = m.ax_curve._cell_px()[3]
    assert h_off < cell_h - 1.0, "the colour bars and the curve panel shrink to the square frame"
    assert m.cax_map._colorbar._label and m.ax_curve.get_ylabel() == '', (m.cax_map._colorbar._label, m.ax_curve.get_ylabel())
    # the drift's box changes the map's aspect: the row follows
    m._on_drift_toggle('auto', True)
    h_on = _assert_row(row, 'Map single view, drift on')
    assert abs(h_on - h_off) > 0.5 or np.allclose(m._drift_bounds(), m.extent), (h_on, h_off)
    # the grid view: every row of cells at one height, the colour bars with their maps
    m.toggle_view()
    rows = {}
    for ax in [m.grid_topo_ax] + m.grid_axes + m.grid_caxes:
        rows.setdefault(id(ax._row), ax._row)
    assert len(rows) >= 1 and all(len(r) >= 2 for r in rows.values())
    for r in rows.values():
        if any(ax.get_visible() for ax in r):
            _assert_row([ax for ax in r if ax.get_visible()], 'Map grid view')
    m.toggle_view()
    # the companion of the map: the maps and their colour bars at one height
    fw = m.feenstra
    if fw is not None and fw.is_open() and fw.kind == 'map':
        _assert_row([a for d in fw.cols for a in (d['top'], d['cbar'])], 'Map companion')
    # the composer: an image panel's row sets the height, the spectrum panel takes it
    m._on_add_to_plot('map')
    m._on_add_to_plot('topo')
    m._on_add_to_plot('spectra')
    comp = m.composer
    assert comp is not None and comp.is_open() and len(comp.panels) >= 3, len(comp.panels)
    for r in range(comp._rows):
        _assert_row([ax for i, (ax, _, _) in enumerate(comp.panels) if i // comp.COLUMNS == r], f'composer row {r}')
    plt.close(comp.fig)
    plt.close(m.fig)
    # the Topo window: the height colour bar as tall as the image
    b._on_topo_window()
    w = b.children[-1]
    _assert_row([w.ax, w.cbar.ax], 'Topo window')
    if w.ax._own_height() < w.ax._cell_px()[3] - 1.0:          # the image shrunk in its cell: the bar with it
        assert w.cbar.ax.vb.geometry().height() < w.cbar.ax._cell_px()[3] - 1.0, "the bar follows the shrunk image"
    w._on_drift_toggle('auto', not w.auto_on)              # the bar is remade with the image: still one row
    _assert_row([w.ax, w.cbar.ax], 'Topo window, redrawn')
    plt.close(w.fig)
    # the drift window: both panels at one height
    b._on_drift_window()
    w = b.drift_window
    _assert_row([w.ax_a, w.ax_b], 'drift window')
    plt.close(w.fig)
    b._goto(0)


@check("N3 calibration anchored at the setpoint: exact on every window and sweep direction, the forward reference, the refusals, the pipeline's note; the file's curves")
def _calibration_anchor(b, exp):
    import io
    import contextlib
    from scipy.integrate import cumulative_trapezoid
    from sm4_Browser import calibration
    from sm4_Browser.sts import STS
    a_t, b_t, I_set = 2.0, 0.3, 1e-10

    def synth(V, i_a):
        lia = 0.8 + 0.5 * np.sin(3.0 * V) + 0.3 * V ** 2
        X = cumulative_trapezoid(lia, V, initial=0)
        X -= X[i_a]
        return lia[:, None], (I_set + a_t * X + b_t * (V - V[i_a]))[:, None]

    for V, Vb in ((np.linspace(-1, 1, 201), 1.0), (np.linspace(1, -1, 201), 1.0), (np.linspace(2, 5, 301), 2.0),
                  (np.linspace(-2, -0.8, 121), -0.8), (np.linspace(-1, 1, 201), -1.0)):
        order = np.argsort(V)
        Vs = V[order]
        i_a = calibration.anchor_index(Vs, Vb)
        lia_s, I_s = synth(Vs, i_a)
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        cal, fits = calibration.calibrate(V, lia_s[inv], I_s[inv], V_bias=Vb, I_setpoint=I_set)
        assert abs(fits[0].a - a_t) < 1e-9 and abs(fits[0].b - b_t) < 1e-9 and fits[0].c == I_set, (Vb, fits[0].a, fits[0].b)
        assert np.allclose(cal[:, 0], a_t * lia_s[inv][:, 0] + b_t)
    # the reference: a column of other values takes the reference's a, b, applied to its own LIA
    V = np.linspace(-1, 1, 201)
    i_a = calibration.anchor_index(V, 1.0)
    lia_f, I_f = synth(V, i_a)
    lia_r, I_r = lia_f * 3.0 + 0.1, I_f * 0.5
    cal, fits = calibration.calibrate(V, lia_r, I_r, V_bias=1.0, I_setpoint=I_set, ref=(lia_f, I_f))
    assert abs(fits[0].a - a_t) < 1e-9 and abs(fits[0].b - b_t) < 1e-9 and np.allclose(cal[:, 0], a_t * lia_r[:, 0] + b_t)
    ref_nan = (np.full_like(lia_f, np.nan), I_f)                      # no forward sweep: the column on its own
    _, fits_own = calibration.calibrate(V, lia_f, I_f, V_bias=1.0, I_setpoint=I_set, ref=ref_nan)
    assert abs(fits_own[0].a - a_t) < 1e-9
    # the IETS counterpart on a window away from 0 V
    Vq = np.linspace(2, 5, 301)
    i_q = calibration.anchor_index(Vq, 2.0)
    S = 0.4 + 0.3 * np.cos(2.0 * Vq)
    Y = cumulative_trapezoid(S, Vq, initial=0)
    Y -= Y[i_q]
    Z = cumulative_trapezoid(Y, Vq, initial=0)
    Z -= Z[i_q]
    U = Vq - Vq[i_q]
    Iq = I_set + a_t * Z + b_t * U ** 2 / 2 + 0.7 * U
    _, fq = calibration.calibrate_iets(Vq, S[:, None], Iq[:, None], V_bias=2.0, I_setpoint=I_set)
    assert abs(fq[0].a - a_t) < 1e-6 and abs(fq[0].b - b_t) < 1e-6 and abs(fq[0].c - 0.7) < 1e-6, (fq[0].a, fq[0].b, fq[0].c)
    # the refusals: no sample at the setpoint bias, no setpoint recorded
    for kw in (dict(V_bias=1.5, I_setpoint=I_set), dict(V_bias=None, I_setpoint=I_set), dict(V_bias=1.0, I_setpoint=None)):
        try:
            calibration.calibrate(V, lia_f, I_f, **kw)
            raise AssertionError(f"not refused: {kw}")
        except ValueError as exc:
            assert 'setpoint' in str(exc), str(exc)
    # the pipeline: a refused file is left in its units, said once
    STS._notes_said.clear()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        c1 = STS.process_arrays(V, lia_f, I_f, normalize=False, calibrate=True, V_bias=1.5, I_setpoint=I_set)
        c2 = STS.process_arrays(V, lia_f, I_f, normalize=False, calibrate=True, V_bias=1.5, I_setpoint=I_set)
    assert np.allclose(c1, lia_f) and np.allclose(c2, lia_f)
    assert buf.getvalue().count("calibration not applied") == 1, buf.getvalue()
    # the file's curves: every acquired column a·LIA + b with the a, b of the anchored regression
    g = goto_kind(b, 'grid')
    cols = np.flatnonzero(np.asarray(g.acquired, dtype=bool))
    _, fits = calibration.calibrate(g.energy, g.lia[:, cols], g.iv[:, cols], V_bias=g.V_bias, I_setpoint=g.I_setpoint)
    c = g.curves(normalize=False, calibrate=True)
    expect = np.column_stack([cf.a * g.lia[:, j] + cf.b for j, cf in zip(cols, fits)])
    assert np.allclose(c[:, cols], expect, rtol=1e-9, atol=0), "the pipeline's calibration is the anchored regression"
    assert all(cf.c == g.I_setpoint for cf in fits)
    b._goto(0)


@check("N4 the dI/dV filter toggles: the box's two toggle columns, the pipeline's two targets, the records of the Line and averaging windows")
def _didv_filters(b, exp):
    from sm4_Browser import widgets
    from sm4_Browser.sts import STS
    hp = {'window': 4, 'n_sigma': 3}
    sp = {'window_length': 6, 'polyorder': 3}
    # the figure box: four independent toggles, one set of boxes
    fig = plt.figure(figsize=(4, 3))
    fc = widgets.FilterControls(fig, [0.1, 0.1, 0.5, 0.3], hampel=hp, sgolay=sp)
    assert fc.params() == (None, None) and fc.params_didv() == (None, None)
    fc.set_params(hp, None, None, sp)
    assert fc.params() == (hp, None) and fc.params_didv() == (None, sp)
    assert fc.states() == (True, False) and fc.states_didv() == (False, True)
    fc._toggle(2)
    assert fc.params_didv() == (hp, sp) and fc.params() == (hp, None)
    fc.set_params(None, None)
    assert fc.params() == (None, None) and fc.params_didv() == (None, None)
    assert widgets.filters_text(fc) == "Hampel off; SG off"
    fc.set_params(hp, sp, hp, None)
    assert widgets.filters_text(fc) == "Hampel window 4, n_sigma 3 on dI/dV and I(V); SG window_length 6, polyorder 3 on I(V)"
    plt.close(fig)
    # the browser's native box: the same four toggles
    nb = b.filters
    assert hasattr(nb, 'chk_hd') and hasattr(nb, 'chk_sd') and nb.params_didv() == (None, None)
    nb.set_params(None, None, hp, None)
    assert nb.params_didv() == (hp, None) and nb.params() == (None, None)
    nb.set_params(None, None)
    # the pipeline: the dI/dV toggles act on the channel's columns, the I(V) toggles on the current only
    g = goto_kind(b, 'grid')
    raw = g.curves(normalize=False, calibrate=False)
    only_iv = g.curves(hp, sp, normalize=False, calibrate=False)
    only_didv = g.curves(None, None, normalize=False, calibrate=False, hampel_didv=hp, sgolay_didv=sp)
    assert np.allclose(only_iv, raw, rtol=1e-12, atol=0), "the I(V) filters leave the uncalibrated channel columns alone"
    cols = np.flatnonzero(np.asarray(g.acquired, dtype=bool))
    from sm4_Browser import filters
    assert np.allclose(only_didv[:, cols], filters.filter_lia(g.lia[:, cols], hp, sp), rtol=1e-9, atol=0), \
        "the dI/dV filters act on the channel columns"
    # a spiked column through the pipeline: the dI/dV Hampel removes the spike, the I(V) one does not
    V = np.asarray(g.energy, dtype=np.float64)
    spiky = np.asarray(g.lia[:, cols[:1]], dtype=np.float64).copy()
    spiky[len(V) // 2, 0] += 50.0 * (np.abs(spiky).max() + 1e-12)
    iv1 = np.asarray(g.iv[:, cols[:1]], dtype=np.float64)
    out_d = STS.process_arrays(V, spiky, iv1, None, None, normalize=False, calibrate=False, hampel_didv=hp)
    out_i = STS.process_arrays(V, spiky, iv1, hp, None, normalize=False, calibrate=False)
    assert np.array_equal(out_i, spiky) and np.abs(out_d - spiky).max() > 0.5 * np.abs(spiky).max(), "each toggle reaches its own target"
    assert np.allclose(out_d, filters.filter_lia(spiky, hp, None), rtol=1e-9, atol=0)
    # the Line window: the record keeps the dI/dV toggles, restore puts them back
    w = open_line(b)
    w._int_filters.set_params(None, None, hp, None)
    w._int_on_filter_change()
    assert w._int_hampel_didv == hp and w._int_sgolay_didv is None and w._int_hampel_params is None
    rec = w.marker_record()
    assert rec['hampel_didv_on'] is True and rec['sgolay_didv_on'] is False and rec['hampel_on'] is False
    w._int_filters.set_params(None, None)
    w._int_on_filter_change()
    assert w._int_hampel_didv is None
    w.restore_markers(rec)
    assert w._int_filters.states_didv() == (True, False) and w._int_hampel_didv == hp
    plt.close(w._int_fig)
    # the averaging window: the record carries the dI/dV pair
    goto_kind(b, 'point')
    b._on_add_new_avg()
    aw = list(b.averages.open().values())[-1]
    aw.filters.set_params(None, sp, hp, None)
    r = aw.record()
    assert r['hampel'] is None and r['sgolay'] == sp and r['hampel_didv'] == hp and r['sgolay_didv'] is None, r
    plt.close(aw.fig)
    b._goto(0)


if __name__ == '__main__':
    sys.exit(0 if run(sys.argv[1:]) else 1)
