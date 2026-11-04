"""
The windows on a virtual screen, one pass on one synthetic experiment (checks/synth.py: a
topography, one grid in the 'paired' sweep layout with a forward and a reverse sweep at every
pixel, a single point with sweeps in pairs, a topography): the browser, a Line window, a Map
window, an averaging window with the point file and a marker-group selection, a line-average
window after one run, a Topo window, and the Feenstra companions of the Line, Map and averaging
windows.  For each window one non-default state is set through its boxes (every row of the filter
box, the pipeline box and the Feenstra settings changed from its default once); then
window.state is asserted equal to the sts.PipelineState built here independently, the drawn arrays
are asserted equal to view(window.state) on the window's columns (one assertion per window: the
Line heatmap, the Map's single-view map, the averaging window's sweeps, the line average's
average, the browser's spectra panels, each companion's four panels), a record is saved where the
window has one, reloaded, and the state asserted to round-trip; then everything is closed.  No
per-feature cells, no interaction beyond the boxes.

The Map window's two features ride on its pass: the percentile clip (one pixel set far above the
rest at one energy: 'clip %' 0 gives the limits of plots.finite_range, 'clip %' 2 the limits of
numpy.nanpercentile(values, [2, 98]) in both 'cmap norm' states, the drawn array unchanged by the
box, an edit outside 0 <= p < 50 refused) and its marker record (saved with the non-default state
and the markers, reloaded through Browser.load_map_record: the state, the view, the energy index,
the markers, the group sizes and centres, z_frac and clip % round-trip; a hand-edited record with
an out-of-range marker and a wrong e_index loaded, the marker clipped to the grid and the index
refused on the console).

The screen: Qt's offscreen platform is never used (it errs in the browser and freezes the
processing).  The windows open on the display at hand (DISPLAY set, or a platform that always has
one, Windows and macOS) or, without one, on an Xvfb virtual screen started here (4200 x 2600:
every window at twice its design size) and stopped at the end; without a display and without Xvfb
the check reports itself as not run.  Only QT_QPA_PLATFORM unset or plain 'offscreen' is handled
(a plain 'offscreen' is dropped); any other setting is left alone.

    python -m sm4_Browser.checks.window_smoke

Spyder: F5.  Exit status 1 on any failure.
"""

if __package__ in (None, ''):
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

fails = 0


def check(label, ok, detail=''):
    global fails
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f": {detail}" if detail else ''))
    fails += 0 if ok else 1


def _virtual_screen():
    """
    The screen the windows draw on (the module docstring).  Returns None when a display is at hand
    (nothing started), the Xvfb process when one was started (DISPLAY set to it; stop it at the
    end), or a string saying why there is no screen.
    """
    cur = os.environ.get('QT_QPA_PLATFORM', '')
    if cur not in ('', 'offscreen'):
        return None
    if cur == 'offscreen':
        os.environ.pop('QT_QPA_PLATFORM', None)
    if sys.platform.startswith('win') or sys.platform == 'darwin':
        return None
    if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
        return None
    xvfb = shutil.which('Xvfb')
    if xvfb is None:
        return "no display (DISPLAY unset) and no Xvfb to start a virtual screen"
    n = next((k for k in range(90, 200) if not os.path.exists(f"/tmp/.X{k}-lock")
              and not os.path.exists(f"/tmp/.X11-unix/X{k}")), None)
    if n is None:
        return "no free X display number for Xvfb"
    proc = subprocess.Popen([xvfb, f":{n}", '-screen', '0', '4200x2600x24', '-nolisten', 'tcp'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(200):
        if os.path.exists(f"/tmp/.X11-unix/X{n}") or proc.poll() is not None:
            break
        time.sleep(0.05)
    if proc.poll() is not None:
        return f"Xvfb did not start (exit status {proc.returncode})"
    os.environ['DISPLAY'] = f":{n}"
    return proc


def run():
    """The pass (the module docstring); returns the number of failures, or the reason it could not run."""
    global fails
    fails = 0
    screen = _virtual_screen()
    if isinstance(screen, str):
        print(f"  not run ({screen})")
        return screen
    try:
        _run()
    except Exception:
        traceback.print_exc()
        check("the pass ran to its end", False, "an exception ended it (above)")
    finally:
        if screen is not None:
            screen.terminate()
            screen.wait()
    print(f"\n{fails} failure(s)")
    return fails


# --- helpers on the controls (every change through the boxes themselves) ---

def _app():
    from sm4_Browser import qtplot as plt
    return plt._app()


def _click(control):
    """A Check control's box clicked (its callback fires)."""
    control.widget.click()
    _app().processEvents()


def _click_row(pipeline_box, label):
    """One row of a controls.PipelineBox clicked, by its label."""
    i = pipeline_box.box.labels.index(label)
    pipeline_box.box.boxes[i].click()
    _app().processEvents()


def _edit(text_control, text):
    """A Text control edited and submitted (as Enter does)."""
    text_control.widget.setText(str(text))
    text_control._enter()
    text_control.widget.editingFinished.emit()
    _app().processEvents()


HAMPEL = {'window': 3, 'n_sigma': 2.5}
SGOLAY = {'window_length': 7, 'polyorder': 2}
GATE_H, GATE_SG = (2, 2.0), (7, 2)
BROADENING = 0.3


def _set_boxes(filters, pipeline, settings):
    """
    Every row of a filter box (the four toggles on, the four boxes edited), a pipeline box (every
    row flipped, the 'H' and 'SG' boxes edited) and a Feenstra settings row ('0Vsub' and
    'broadened' flipped, the width edited) changed from its default once, through the controls.
    """
    for box in (filters.chk_h, filters.chk_s, filters.chk_hd, filters.chk_sd):
        _click(box)
    _edit(filters.box_win, HAMPEL['window'])
    _edit(filters.box_nsig, HAMPEL['n_sigma'])
    _edit(filters.box_wl, SGOLAY['window_length'])
    _edit(filters.box_po, SGOLAY['polyorder'])
    for label in ('denoise', 'soft edge', 'calibrate', 'gate', '0-lock', 'Vbias lock', 'normalize'):
        _click_row(pipeline, label)
    _edit(pipeline.box_h, f"{GATE_H[0]},{GATE_H[1]}")
    _edit(pipeline.box_sg, f"{GATE_SG[0]},{GATE_SG[1]}")
    if settings is not None:
        _click(settings.chk_zero)
        _click(settings.chk_broad)
        _edit(settings.box_width, BROADENING)


def _expected(channel, divide_iv, norm_range=None, gate_on=False, locks=True, denoise=True, soft_edge=False):
    """The state _set_boxes leaves a window in (every row flipped: the soft edge row, on at open, off), built independently of the window."""
    from sm4_Browser import calibration
    from sm4_Browser.sts import PipelineState
    return PipelineState(channel=channel, hampel=PipelineState.items(HAMPEL), sgolay=PipelineState.items(SGOLAY),
                         hampel_didv=PipelineState.items(HAMPEL), sgolay_didv=PipelineState.items(SGOLAY),
                         gate=calibration.GateSettings(on=gate_on, hampel=GATE_H, sgolay=GATE_SG, lock0=locks,
                                                       lockv=locks, denoise=denoise, soft_edge=soft_edge),
                         calibrate=True, normalize=True, divide_iv=divide_iv, iv_zero_sub=False, iv_broadened=True,
                         iv_broadening=BROADENING, norm_range=norm_range)


def _same(a, b):
    import numpy as np
    if a is None or b is None:
        return a is None and b is None
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))


def _lines(ax):
    """The curves drawn in an axes as rows, in drawing order."""
    import numpy as np
    return [np.asarray(l.get_ydata(), dtype=float) for l in ax.lines]


def _columns(ax, npts):
    """
    Every column drawn in an axes by plot calls with 2D arrays: qtplot draws the columns of one
    colour as one curve with a NaN between them (each column npts points), so a curve of
    k (npts + 1) points holds k columns; a curve of npts points is one column.
    """
    out = []
    for y in _lines(ax):
        if len(y) == npts:
            out.append(y)
        elif len(y) % (npts + 1) == 0:
            out.extend(list(y.reshape(-1, npts + 1)[:, :npts]))
        else:
            out.append(y)
    return out


def _holds(drawn, columns):
    """Every column (npts, n) of `columns` is one of the drawn curves."""
    import numpy as np
    cols = np.asarray(columns, dtype=float)
    return all(any(_same(d, cols[:, j]) for d in drawn) for j in range(cols.shape[1]))


def _companion_state(parent, fw):
    """The state a companion asks its parent for: the parent's with the companion's rows."""
    flags = fw.pipeline.state()
    return parent.state.with_(calibrate=flags['calibrate'], normalize=flags['normalize'], gate=flags['gate'])


def _companion_rows(fw):
    """The companion's own rows changed from their defaults: calibrate and normalize on, denoise on."""
    for label in ('calibrate', 'normalize', 'denoise'):
        _click_row(fw.pipeline, label)


def _run():
    import numpy as np
    from sm4_Browser import qtplot as plt
    from sm4_Browser import records
    from sm4_Browser.checks import synth

    tmp = tempfile.mkdtemp(prefix='sm4smoke_')
    exp = synth.Experiment(tmp, v=(3e-12, -2e-12))
    t = synth.T_REF
    exp.topo(t, center=(0.0, 0.0), size_m=24e-9, n=64)
    t += 64 * 64 * 2e-3 * 2 + 30
    n_fast, n_lines, npts = 32, 16, 15
    exp.grid(t, center=(0.0, 0.0), n_fast=n_fast, n_lines=n_lines, size_m=8e-9, npts=npts,
             sweeps=('forward', 'reverse'), layout='paired')
    t += n_fast * n_lines * 0.5 * 2 + n_fast * n_lines * 2e-3 + 60
    exp.point(t, xy=(2e-9, 1e-9), n_sweeps=6, npts=npts, pairs=True)       # the grid's energy step: the run goes through
    t += 60
    exp.topo(t, center=(0.0, 0.5e-9), size_m=24e-9, n=64)
    synth.install(exp)
    from sm4_Browser.browser import Browser
    plt._RUNNING[0] = True                       # plt.show() returns at once (no event loop in a check)
    b = Browser(tmp, scalefactor=1.0)
    app = plt._app()
    app.processEvents()
    check("the browser opened on the synthetic experiment on a screen that is not Qt's offscreen platform",
          app.platformName() != 'offscreen', f"platform {app.platformName()}, {len(b.files)} files")
    i_grid = next(i for i, p in enumerate(b.files) if b.kinds[p] == 'grid')
    i_point = next(i for i, p in enumerate(b.files) if b.kinds[p] == 'point')
    opened = []

    # --- the Map window and its companion ---
    b._goto(i_grid)
    b._on_map()
    m = b.children[-1]
    opened.append(m)
    _set_boxes(m.filters, m.pipeline, m.feenstra.settings)
    check("Map: the state of its boxes", m.state == _expected('didv', True), str(m.state))
    vm = m.grid.view(m.state)
    check("Map: the single-view map draws view(state).shown",
          _same(m.maps, m._cube_of(vm.shown)) and _same(m._single_im.get_array(), m._cube_of(vm.shown)[m.e_index]))
    _companion_rows(m.feenstra)
    cs = _companion_state(m, m.feenstra)
    check("Map companion: its rows override the parent's calibrate, normalize and gate and nothing else",
          cs.calibrate and cs.normalize and cs.gate.denoise and cs.gate.on and not cs.gate.lock0
          and cs._replace(calibrate=m.state.calibrate, normalize=m.state.normalize, gate=m.state.gate) == m.state)
    vc = m.grid.view(cs)
    fields = {'didv': vc.didv, 'iv': vc.current, 'ivv': vc.ivv, 'ratio': vc.ratio}
    check("Map companion: its four panels draw view(state) at the shown energy",
          all(_same(col['im'].get_array(), m._cube_of(fields[key])[m.e_index])
              for col, key in zip(m.feenstra.cols, m.feenstra.KEYS)))

    # --- the percentile clip: one pixel far above the rest at the shown energy ---
    from sm4_Browser import plots
    e = m.e_index
    i, j = m.n_lines // 2, m.n_fast // 2
    m.maps[e, i, j] = float(np.nanmax(m.maps)) * 100.0
    m.draw_single()
    lim0 = (float(m._single_im.norm.vmin), float(m._single_im.norm.vmax))
    drawn0 = np.array(m._single_im.get_array(), dtype=float)
    check("Map clip: 'clip %' 0 gives the limits of finite_range over the cube", lim0 == plots.finite_range(m.maps), str(lim0))
    _edit(m.box_clip, 2)
    lim2 = (float(m._single_im.norm.vmin), float(m._single_im.norm.vmax))
    want = tuple(float(v) for v in np.nanpercentile(m.maps, [2, 98]))
    check("Map clip: 'clip %' 2 gives nanpercentile(values, [2, 98]) over the cube with 'cmap norm' off",
          m.clip_percent == 2.0 and np.allclose(lim2, want) and lim2 != lim0, f"{lim2} vs {want}")
    _click(m.chk_cmap_norm)
    lim2n = (float(m._single_im.norm.vmin), float(m._single_im.norm.vmax))
    want_n = tuple(float(v) for v in np.nanpercentile(m.maps[e], [2, 98]))
    check("Map clip: with 'cmap norm' on each map its own nanpercentile(values, [2, 98])", np.allclose(lim2n, want_n),
          f"{lim2n} vs {want_n}")
    check("Map clip: the drawn array is unchanged by the box",
          _same(m._single_im.get_array(), drawn0) and _same(m.maps[e], drawn0))
    _edit(m.box_clip, 60)
    check("Map clip: an edit outside 0 <= p < 50 is refused and the box put back",
          m.clip_percent == 2.0 and m.box_clip.value == '2')
    _click(m.chk_cmap_norm)                       # off again
    m.recompute()                                 # the maps back from the data

    # --- the Map's marker record: saved with the non-default state and the markers, reloaded ---
    m._set_level(0.3)
    rec = m.marker_record()
    m._on_save()
    saved = (m.state, m.view_mode, m.e_index, list(m.marker_px), list(m.group_sizes), m._sync_centres(), m.z_frac,
             m.clip_percent)
    check("Map record: written under records.map_record_path with the record's keys",
          os.path.isfile(records.map_record_path(os.path.dirname(m.grid.path), m.grid.path))
          and rec['format'] == records.MAP_RECORD_FORMAT and rec['clip_percent'] == 2.0 and rec['z_frac'] == 0.3
          and len(rec['contour_segments_nm']) == len(m._contour_segments()))
    plt.close(m.fig)
    app.processEvents()
    b.load_saved_average({'kind': 'map', 'record': rec})
    m2 = b.children[-1]
    opened.append(m2)
    got = (m2.state, m2.view_mode, m2.e_index, list(m2.marker_px), list(m2.group_sizes), m2._sync_centres(), m2.z_frac,
           m2.clip_percent) if m2 is not m else None
    check("Map record: the state, the view, the energy index, the markers, the group sizes and centres, z_frac and "
          "clip % round-trip through the browser's loader", got == saved, str(got))
    bad = dict(rec)
    bad['e_index'] = 999
    bad['markers'] = [[999, 999]] + [list(px) for px in rec['markers'][1:]]
    said = io.StringIO()
    with contextlib.redirect_stdout(said):
        b.load_saved_average({'kind': 'map', 'record': bad})
    m3 = b.children[-1]
    opened.append(m3)
    text = said.getvalue()
    check("Map record: a hand-edited record with a wrong e_index and an out-of-range marker: the index refused on the "
          "console, the marker clipped to the grid",
          m3 is not m2 and "energy index 999 is outside" in text and m3.e_index == m2.e_index
          and m3.marker_px[0] == (m3.n_lines - 1, m3.n_fast - 1), text.strip().splitlines()[-1][:120] if text.strip() else '')

    # --- the Line window and its companion ---
    b._goto(i_grid)
    b._on_line()
    w = b.children[-1]
    opened.append(w)
    _set_boxes(w._int_filters, w._int_pipeline_box, w._int_feenstra.settings)
    check("Line: the state of its boxes", w.state == _expected('didv', True), str(w.state))
    vw, _ = w._pipeline(w.state)
    check("Line: the heatmap draws view(state).shown on the kept rows",
          _same(w._int_im.get_array(), w._int_orient(vw.shown)[w._int_kept_mask()])
          and _same(w._int_curves, w._int_orient(vw.shown)))
    _companion_rows(w._int_feenstra)
    cs = _companion_state(w, w._int_feenstra)
    vc, _ = w._pipeline(cs)
    fields = {'didv': vc.didv, 'iv': vc.current, 'ivv': vc.ivv, 'ratio': vc.ratio}
    check("Line companion: its four panels draw view(state) oriented like the heatmap",
          all(_same(col['im'].get_array(), w._int_orient(fields[key])[w._int_kept_mask()])
              for col, key in zip(w._int_feenstra.cols, w._int_feenstra.KEYS)))
    rec = w.marker_record()
    w._int_on_save()
    saved_state = w.state
    plt.close(w._int_fig)
    app.processEvents()
    b.load_line_record(rec)
    w2 = b.children[-1]
    opened.append(w2)
    check("Line: the record round-trips the state through the browser's loader",
          w2 is not w and w2.state == saved_state, str(w2.state) if w2 is not w else "no window opened")

    # --- the averaging window (the point file and a marker-group selection) and its companion ---
    b._goto(i_point)
    index = b._next_avg_index()
    b._on_add_new_avg()
    a = b._open_avgs()[index]
    opened.append(a)
    sel = w2._int_group_selection(0)
    a.add(sel)
    _set_boxes(a.filters, a.pipeline, a.feenstra.settings)
    check("averaging: the state of its boxes", a.state == _expected('didv', True), str(a.state))
    a._on_run()

    def entry_sweeps(e, state):
        fe = e['file']
        o = np.argsort(fe.energy)
        return fe.view(state).shown[o, :][:, fe.acquired]

    check("averaging: every entry's sweeps are view(state).shown of that entry",
          len(a.entries) == 2 and all(_same(e['sweeps'], entry_sweeps(e, a.state)) for e in a.entries)
          and all(_holds(_lines(a.ax), e['sweeps']) for e in a.entries))
    _companion_rows(a.feenstra)
    cs = _companion_state(a, a.feenstra)
    panels = {key: _lines(col['panel']) for col, key in zip(a.feenstra.cols, a.feenstra.KEYS)}
    good = True
    for e in a.entries:
        fe = e['file']
        o = np.argsort(fe.energy)
        vc = fe.view(cs)
        mp = a.average[3][a.entries.index(e)] if a.average is not None else None
        for key, arr in (('didv', vc.didv), ('iv', vc.current), ('ivv', vc.ivv), ('ratio', vc.ratio)):
            sw = arr[o, :][:, fe.acquired]
            if mp is not None:
                sw = a._to_grid(sw, mp)
            good &= _holds(panels[key], sw)
    check("averaging companion: its four panels draw every entry's view(state) on the run's grid", good)
    rec = a.record()
    a.save()
    saved_state = a.state
    plt.close(a.fig)
    app.processEvents()
    b.load_saved_average({'kind': 'record', 'record': rec, 'cache': None})
    a2 = b._open_avgs().get(index)
    if a2 is not None:
        opened.append(a2)
    check("averaging: the record round-trips the state through the browser's loader",
          a2 is not None and a2 is not a and a2.state == saved_state, str(getattr(a2, 'state', None)))

    # --- the line-average window after one run ---
    w2._int_on_line_new()
    la = list(b.line_averages.open().values())[-1]
    opened.append(la)
    la.run()
    check("line average: opened from the Line window and run once (its companion opens with the first run)",
          la.run_state is not None and la._int_feenstra is not None)
    _set_boxes(la._int_filters, la._int_pipeline_box, la._int_feenstra.settings)
    check("line average: the state of its boxes", la.state == _expected('didv', True), str(la.state))
    la.run()
    avg = la._average_of([la._member_run(mm, la.state)[0].shown for mm in la.members], la.run_state)
    check("line average: the average drawn is the members' view(state).shown through the run's maps",
          _same(la._int_curves, avg.T) and _same(la._int_im.get_array(), avg.T[la._int_kept_mask()]))
    rec = la.record()
    la.save()
    saved_state = la.state
    plt.close(la._int_fig)
    app.processEvents()
    b.load_lineavg_record(rec)
    la2 = b.children[-1]
    opened.append(la2)
    check("line average: the record round-trips the state through the browser's loader",
          la2 is not la and getattr(la2, 'state', None) == saved_state, str(getattr(la2, 'state', None)))

    # --- the browser's spectra panels (last: its Feenstra settings are handed to windows opened after) ---
    b._goto(i_grid)
    f = b.current
    _set_boxes(b.filters, b.pipeline, b.spec_settings)
    app.processEvents()
    check("browser: the state of its boxes", b.state == _expected('didv', False), str(b.state))
    v = f.view(b.state)
    order = np.argsort(f.energy)
    va = f.spectra_all.view_of(b.state)
    keep = np.asarray(f.spectra_all.acquired, dtype=bool)
    dirs = np.asarray(f.sweep_info['dir'])
    npts = len(f.energy)
    main = _columns(b.ax_spec, npts)
    want = np.hstack([va.didv[order, :][:, keep & (dirs == 1)], va.didv[order, :][:, keep & (dirs == -1)]])
    check("browser: the dI/dV panel draws every sweep of view(state).didv (the all-sweeps view)",
          _holds(main, want) and len(main) == int(keep.sum()), f"{len(main)} curves, {int(keep.sum())} sweeps")
    ok = True
    for ax, arr in ((b.ax_spec_iv, v.current), (b.ax_spec_ivv, v.ivv), (b.ax_spec_ratio, v.ratio)):
        drawn = _columns(ax, npts)
        cols = arr[order, :][:, f.acquired]
        ok &= _holds(drawn, cols) and len(drawn) == cols.shape[1]
    check("browser: the current, I/V and ratio panels draw view(state).current, .ivv and .ratio", ok)

    # --- the Topo window ---
    b._goto(i_grid)
    n_before = len(b.children)
    b._on_topo_window()
    check("Topo window: opened on the topo under the grid", len(b.children) == n_before + 1)
    opened.append(b.children[-1])

    # --- close everything ---
    for win in opened:
        fig = getattr(win, '_int_fig', None) or getattr(win, 'fig', None)
        if fig is not None:
            try:
                plt.close(fig)
            except Exception:
                pass
    try:
        b.win.close()
    except Exception:
        pass
    app.processEvents()
    check("every window closed", all(not getattr(win, 'is_open', lambda: False)() for win in opened))


if __name__ == '__main__':
    out = run()
    sys.exit(1 if out else 0)
