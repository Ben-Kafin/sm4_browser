"""
The browser, the Line window, and the Map window on an experiment whose grid records a forward and a
reverse sweep at every pixel (offscreen).

    python -m sm4_Browser.checks.smoke_sweeps [per_pixel|pages]
"""
import os
import sys
import tempfile
import traceback

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from sm4_Browser import qtplot as plt
from sm4_Browser.checks import synth
from sm4_Browser.checks.smoke_browser import make_browser, goto_kind, open_line, open_map

RESULTS = []


def ok(name, cond, detail=''):
    """Record and print one result: PASS or FAIL, the name, the detail in parentheses."""
    RESULTS.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ''))


def run(layout='per_pixel'):
    """
    The standard experiment with a forward and a reverse sweep per grid pixel, stored `layout`
    ('per_pixel' or 'pages'): the browser's sweep view and line styles, the Line window's 'sweeps'
    radio (the views, the shear direction, a marker group's Selection carrying the view), the Map
    window's views, one drift result per view, and a selection record reloading on its view.  An
    exception fails the run.  Returns True when all passed.
    """
    tmp = tempfile.mkdtemp(prefix='sm4sweepb_')
    exp = synth.standard(tmp, grid_sweeps=('forward', 'reverse'), grid_layout=layout)
    synth.install(exp)
    b = make_browser(tmp)
    try:
        g = goto_kind(b, 'grid')
        ok("browser: grid with two sweeps per pixel", g.n_sweeps == 2 and g.sweep_mode == 'forward sweeps')
        styles = {getattr(a, '_style', {}).get('ls') for a in b.ax_spec._children}
        styles |= {getattr(a, '_ls', None) for a in b.ax_spec._children}
        styles.discard(None)
        ok("browser panel: forward solid and reverse dashed", {'-', '--'} <= styles, str(styles))
        w = open_line(b)
        ok("Line window: 'sweeps' radio offered", w._int_rad_sweeps is not None and w.sweep_mode == 'forward sweeps')
        sh_f = w._int_shear()
        w._int_on_sweeps('reverse sweeps')
        ok("Line window: reverse view applied", w.sweep_mode == 'reverse sweeps' and w._int_sm4().sweep_mode == 'reverse sweeps')
        sh_r = w._int_shear()
        if sh_f is not None and sh_r is not None:
            ok("Line window: the reverse view's shear runs the other way along the energy axis",
               np.nanmean(np.abs(sh_r[:, 0])) > np.nanmean(np.abs(sh_r[:, -1])) and
               np.nanmean(np.abs(sh_f[:, -1])) > np.nanmean(np.abs(sh_f[:, 0])))
        rec = None
        for name in ('_int_marker_record', '_int_record'):
            if hasattr(w, name):
                rec = getattr(w, name)()
        w._int_on_sweeps('all sweeps')
        ok("Line window: average view applied", w.sweep_mode == 'all sweeps')
        sel = w._int_group_selection(0)
        ok("Line window: a marker group's Selection carries the view",
           sel is not None and sel.record_entry().get('sweeps') == 'all sweeps')
        m = open_map(b)
        ok("Map window: 'sweeps' radio offered", m.rad_sweeps is not None)
        before = m.maps.copy()
        m._on_sweeps('reverse sweeps')
        ok("Map window: reverse view applied", m.grid.sweep_mode == 'reverse sweeps' and m.maps.shape == before.shape)
        m._on_sweeps('sweep 1')
        ok("Map window: 'sweep 1' equals the forward view (one forward sweep)", np.allclose(m.maps, before, equal_nan=True))
        # drift results per view are distinct entries
        f_r = g.with_sweeps('reverse sweeps')
        d_f = b.drift.result(g, b.loaded, False, True, True)
        d_r = b.drift.result(f_r, b.loaded, False, True, True)
        ok("drift: each view its own result (own clock)", d_f is not None and d_r is not None and d_f is not d_r)
        # record round trip of a selection
        from sm4_Browser.averaging import Selection
        s_rec = Selection(f_r, [0, 1, 2], {'line': 0, 'positions': [0, 1, 2]}).record_entry()
        fi = dict(s_rec, path=g.path, name=g.name)
        e = b._entry_from_record(fi)
        ok("records: a selection reloads on its sweep view", e.file.sweep_mode == 'reverse sweeps')
    except Exception:
        traceback.print_exc()
        ok("no exception", False)
    print("ALL PASS" if all(RESULTS) else "SOME FAILED")
    return all(RESULTS)


if __name__ == '__main__':
    sys.exit(0 if run(sys.argv[1] if len(sys.argv) > 1 else 'per_pixel') else 1)
