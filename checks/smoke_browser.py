"""
Smoke test of the browser on a synthetic experiment (checks/synth.py), offscreen.

    QT_QPA_PLATFORM=offscreen python -m sm4_Browser.checks.smoke_browser

Builds the standard experiment in a temporary directory, opens the Browser (the event loop is not
entered), steps through every file, and exercises the controls added by the upgrade plan.  Prints
one line per check and exits non-zero on the first failure.
"""
import os
import sys
import tempfile
import time
import traceback

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from sm4_Browser import qtplot as plt
from sm4_Browser.checks import synth

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def make_browser(directory, **kw):
    from sm4_Browser.browser import Browser
    plt._RUNNING[0] = True              # plt.show() returns at once (no event loop in a test)
    b = Browser(directory, scalefactor=1.0, **kw)
    plt._app().processEvents()
    return b


def run(selected=None):
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
    i = next(i for i, p in enumerate(b.files) if b.kinds[p] == kind)
    b._goto(i)
    return b.current


def open_line(b):
    goto_kind(b, 'grid')
    n = len(b.children)
    b._on_line()
    assert len(b.children) == n + 1, "Line window did not open"
    return b.children[-1]


def open_map(b):
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
    assert hasattr(w, 'btn_stripe') and not hasattr(w, 'chk_stripe')
    plt.close(w.fig)
    b._on_drift_toggle('auto', True)
    b._on_drift_toggle('manual', True)
    d.set_fast_state(f, 'off', 'off')
    d.set_fast_state(f, 'auto', 'off')


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
    # the Line window opens with the forward row only; its rows carry the placement's offsets
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
    off = ((np.asarray(res['xy_corr'])[cols] - np.asarray(f.spec_xy_m)[cols]) @ fast) * 1e10 / w.scalefactor
    assert np.allclose(sh[:, 0], off, atol=1e-9), (sh[:3, 0], off[:3])          # the first edge: the row offset alone
    assert np.nanmax(np.abs(off)) > 0
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
    # the slice: one curve per E-marker, against position, the value at the marker's energy
    lines = [a for a in w._int_ax_slice._children if hasattr(a, '_y') and len(getattr(a, '_y', [])) == len(w._int_pos)]
    assert len(lines) == 2, len(lines)
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
    rec = w._int_marker_record() if hasattr(w, '_int_marker_record') else None
    import inspect
    src = inspect.getsource(type(w))
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


@check("G1 Line window: 'E range' and 'x range' boxes (limits, normalization, colour scale, record)")
def _line_ranges(b, exp):
    from sm4_Browser import plots
    w = open_line(b)
    e = w._int_e_axis()
    lo, hi = float(e[len(e) // 4]), float(e[3 * len(e) // 4])
    # normalize on: the shown energy range is the normalized range
    w._int_pipeline_box.set_state(calibrate=w._int_apply_calibration, normalize=True)
    w._int_on_pipeline_change({'calibrate': w._int_apply_calibration, 'normalize': True})
    assert w._int_apply_normalize
    w._int_e_range_box._on_submit(f"{lo}, {hi}")
    assert w._int_e_range == (lo, hi), w._int_e_range
    assert w._int_e_range_box.box.value == widgets_text(lo, hi), w._int_e_range_box.box.value
    assert np.allclose(w._int_ax_didv.get_xlim(), (lo, hi)), w._int_ax_didv.get_xlim()
    assert np.allclose(w._int_ax_curve.get_xlim(), (lo, hi)), w._int_ax_curve.get_xlim()
    m = (e >= lo) & (e <= hi)
    rows = np.asarray(w._int_curves)[w._int_kept_mask()]
    areas = [abs(np.trapezoid(r[m], e[m])) for r in rows if np.isfinite(r).all()]
    assert areas and np.allclose(areas, 1.0), areas[:5]
    # the colour scale from the cells inside the range only
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
    # the position range: the view only
    p0, p1 = w._int_pos_bounds()
    q0, q1 = p0 + 0.25 * (p1 - p0), p0 + 0.75 * (p1 - p0)
    curves_before = np.array(w._int_curves)
    w._int_x_range_box._on_submit(f"{q0} {q1}")
    assert w._int_x_range == (q0, q1)
    assert np.allclose(w._int_ax_didv.get_ylim(), (q0, q1)), w._int_ax_didv.get_ylim()
    assert np.allclose(w._int_ax_topo.get_ylim(), (q0, q1)), w._int_ax_topo.get_ylim()
    assert np.allclose(curves_before, w._int_curves, equal_nan=True)
    # the record carries both; blank boxes restore the full ranges; the record restores them
    rec = w.marker_record()
    assert rec['energy_range'] == [lo, hi] and rec['pos_range'] == [q0, q1], (rec['energy_range'], rec['pos_range'])
    w._int_e_range_box._on_submit('')
    w._int_x_range_box._on_submit('')
    assert w._int_e_range is None and w._int_x_range is None
    x0, x1, y0, y1 = w._int_im.extent()
    assert np.allclose(w._int_ax_didv.get_xlim(), (x0, x1)) and np.allclose(w._int_ax_didv.get_ylim(), (y0, y1))
    full = [abs(np.trapezoid(r, e)) for r in np.asarray(w._int_curves)[w._int_kept_mask()] if np.isfinite(r).all()]
    assert np.allclose(full, 1.0)
    w.restore_markers(rec)
    assert w._int_e_range == (lo, hi) and w._int_x_range == (q0, q1), (w._int_e_range, w._int_x_range)
    assert np.allclose(w._int_ax_didv.get_xlim(), (lo, hi)) and np.allclose(w._int_ax_didv.get_ylim(), (q0, q1))
    # refused input leaves the ranges and puts the boxes back
    for bad in ("abc", "1", "1, 1", "nan, 2"):
        w._int_e_range_box._on_submit(bad)
        assert w._int_e_range == (lo, hi), bad
        assert w._int_e_range_box.box.value == widgets_text(lo, hi), (bad, w._int_e_range_box.box.value)
    w._int_e_range_box._on_submit(f"{e.min() - 2}, {e.min() - 1}")            # no recorded energy inside
    assert w._int_e_range == (lo, hi) and w._int_e_range_box.box.value == widgets_text(lo, hi)
    plt.close(w._int_fig)


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
    off_lim = (m.ax_map.get_xlim(), m.ax_map.get_ylim())             # the frame, widened to the box's aspect
    m._on_drift_toggle('auto', True)
    assert m.auto_on
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
    # every drawn quad (map and topo) lies inside the limits, which hold the recorded frame too
    quads = np.vstack([q._q.reshape(-1, 2) for q in tmaps] + [pm._q.reshape(-1, 2)])
    ok = np.all(np.isfinite(quads), axis=1)
    x0, x1 = m.ax_map.get_xlim()
    y0, y1 = m.ax_map.get_ylim()
    assert x0 <= frame[0] + 1e-9 and x1 >= frame[1] - 1e-9 and y0 <= frame[2] + 1e-9 and y1 >= frame[3] - 1e-9
    assert quads[ok, 0].min() >= x0 - 1e-6 and quads[ok, 0].max() <= x1 + 1e-6, (quads[ok, 0].min(), quads[ok, 0].max(), x0, x1)
    assert quads[ok, 1].min() >= y0 - 1e-6 and quads[ok, 1].max() <= y1 + 1e-6, (quads[ok, 1].min(), quads[ok, 1].max(), y0, y1)
    assert np.allclose(m.ax_topo.get_xlim(), (x0, x1)) and np.allclose(m.ax_topo.get_ylim(), (y0, y1))
    # the limits do not jump while stepping the energy
    m.step(1)
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
    assert m._frame_shift(m.e_index) is None and not m._topo_maps.get(id(m.ax_topo))
    assert np.allclose(m.ax_map.get_xlim(), off_lim[0]) and np.allclose(m.ax_map.get_ylim(), off_lim[1])
    assert (x1 - x0) > (off_lim[0][1] - off_lim[0][0]) - 1e-9 and (y1 - y0) > (off_lim[1][1] - off_lim[1][0]) - 1e-9
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
    assert np.allclose(pos, [m.pixel_xy(px) for px in m.marker_px])
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


def widgets_text(lo, hi):
    from sm4_Browser.widgets import RangeBox
    return RangeBox.text_of((lo, hi))


if __name__ == '__main__':
    sys.exit(0 if run(sys.argv[1:]) else 1)
