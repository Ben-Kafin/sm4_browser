"""
The saving of every window, exactly what it shows plus its full info panel, on one synthetic
experiment (checks/synth.py: a topography and one grid in the 'paired' sweep layout): the
rebuild_panels window on its synthetic set, the browser (each of its two canvases), and the Map
and Line windows opened from it.  For each window:

  1. its info panel ends with a 'settings' section, one 'label: value' line per control shown,
     no label twice (controls.settings_text);
  2. the vector files (mplexport.export): the PDF page and the SVG are the canvas at its size on
     screen with the info block below it, W x (H + H_info) at qtplot.DPI; the SVG holds the
     plotted paths, the 'settings' line and every line of the info block (_info_lines);
  3. the Qt SVG (export_vector): its view box is the saved area (_export_rect) with the info block
     below it, and it holds the 'settings' line;
  4. the PNG (window_image) is the saved area times WINDOW_IMAGE_DPI / DPI, at WINDOW_IMAGE_DPI;

and once: a change of a box shows in the browser's settings section, each browser toolbar saves its
own column (export_root), the EPS and the CSV of the Line window and the TIFF of the browser's
spectra column are written, a rebuild_panels save right after show(block=False) (its __main__)
holds the settings section, and a window opened before Spyder's module reloader dropped the package
from sys.modules saves as many paths as before and its settings section still follows its boxes
(qtplot binds mplexport, controls binds settings_text to qtplot).

The pass runs in a child process with its own screen (window_smoke._virtual_screen: never Qt's
offscreen platform), so the Qt application of another check (window_smoke stops its Xvfb when it
ends) is never reused; the files go to a temporary folder, removed at the end.

    python -m sm4_Browser.checks.save_check

Spyder: F5.  Exit status 1 on any failure.
"""

if __package__ in (None, ''):
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import traceback

fails = 0
_CHILD = 'SM4_SAVE_CHECK_CHILD'          # set in the child process that runs the pass


def check(label, ok, detail=''):
    global fails
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f": {detail}" if detail else ''))
    fails += 0 if ok else 1


def run():
    """The pass (the module docstring) in a child process; returns the number of failures, or the reason it could not run."""
    pkg_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(os.environ)
    env[_CHILD] = '1'
    env['PYTHONPATH'] = pkg_parent + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    proc = subprocess.Popen([sys.executable, '-m', 'sm4_Browser.checks.save_check'], cwd=pkg_parent, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                            errors='replace')
    last = []
    for line in proc.stdout:
        print(line, end='')
        last = (last + [line.strip()])[-5:]
    proc.wait()
    why = next((ln[len('not run: '):] for ln in last if ln.startswith('not run: ')), None)
    if why is not None:
        return why
    n = next((int(m.group(1)) for ln in reversed(last) for m in [re.match(r'(\d+) failure\(s\)$', ln)] if m), None)
    if n is None:
        return f"the child process ended without its result (exit status {proc.returncode})"
    return n


def _child():
    """The pass itself, in the child process: its own screen, then _run; prints 'N failure(s)' or 'not run: why'."""
    from sm4_Browser.checks.window_smoke import _virtual_screen
    screen = _virtual_screen()
    if isinstance(screen, str):
        print(f"not run: {screen}")
        return 1
    tmp = tempfile.mkdtemp(prefix='sm4save_')
    try:
        _run(tmp)
    except Exception:
        traceback.print_exc()
        check("the pass ran to its end", False, "an exception ended it (above)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if screen is not None:
            screen.terminate()
            screen.wait()
    print(f"\n{fails} failure(s)")
    return 1 if fails else 0


# --- reading the files ---

def _png_size(path):
    with open(path, 'rb') as fh:
        head = fh.read(24)
    return struct.unpack('>II', head[16:24])


def _png_dpi(path):
    data = open(path, 'rb').read()
    i = data.find(b'pHYs')
    if i < 0:
        return None
    x, _, unit = struct.unpack('>IIB', data[i + 4:i + 13])
    return x * 0.0254 if unit == 1 else None


def _pdf_box(path):
    m = re.search(rb'/MediaBox\s*\[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\]', open(path, 'rb').read())
    return tuple(float(v) for v in m.groups()) if m else None


def _svg(path):
    return open(path, encoding='utf-8', errors='replace').read()


def _svg_pt(s):
    """matplotlib's SVG: (width, height) in points."""
    w = re.search(r'<svg[^>]*\swidth="([\d.]+)pt"', s)
    h = re.search(r'<svg[^>]*\sheight="([\d.]+)pt"', s)
    return (float(w.group(1)), float(h.group(1))) if w and h else None


def _svg_viewbox(s):
    """Qt's SVG (QSvgGenerator): the view box (width, height) in logical pixels."""
    m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', s)
    return (float(m.group(1)), float(m.group(2))) if m else None


def _svg_text(s):
    return re.sub(r'<[^>]+>', '\n', s)


# --- the pass ---

def _settle(app, seconds=0.2):
    """Qt's events processed for a while (the info panel's push is a 50 ms single shot)."""
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        app.processEvents()
        time.sleep(0.01)


def _settings(text):
    """The settings section's labels of an info text (the lines after the line 'settings')."""
    parts = ('\n' + text + '\n').split('\nsettings\n', 1)
    if len(parts) < 2:
        return None
    return [ln.split(':', 1)[0] for ln in parts[1].splitlines() if ':' in ln]


def _check_window(plt, mpx, fig, tag, tmp, info_text):
    """Items 1 to 4 of the module docstring for one figure; returns the vector SVG's number of paths."""
    labs = _settings(info_text)
    check(f"{tag}: the info panel ends with a settings section", labs is not None and len(labs) > 0,
          f"{0 if labs is None else len(labs)} lines")
    dup = sorted({lab for lab in (labs or []) if labs.count(lab) > 1})
    check(f"{tag}: no settings label twice", not dup, str(dup))
    W, H = fig._canvas_size()
    lines = mpx._info_lines(fig)
    H_info = mpx._info_geometry(fig, lines)[4]
    want_pt = (W / plt.DPI * 72.0, (H + H_info) / plt.DPI * 72.0)
    pdf = fig.savefig(os.path.join(tmp, f"{tag}.pdf"))
    box = _pdf_box(pdf)
    check(f"{tag}: the vector PDF's page is the canvas with the info block below it",
          box is not None and abs(box[2] - want_pt[0]) < 1.0 and abs(box[3] - want_pt[1]) < 1.0, f"{box} vs {want_pt}")
    s = _svg(fig.savefig(os.path.join(tmp, f"{tag}.svg")))
    sz = _svg_pt(s)
    check(f"{tag}: the vector SVG is the canvas with the info block below it",
          sz is not None and abs(sz[0] - want_pt[0]) < 1.0 and abs(sz[1] - want_pt[1]) < 1.0, f"{sz} vs {want_pt}")
    txt = _svg_text(s)
    missing = [ln for ln in lines if ln.strip() and ln.strip() not in txt]
    check(f"{tag}: the vector SVG holds the 'settings' line and every line of the info block",
          '\nsettings\n' in txt and not missing, f"{len(lines)} lines, {len(missing)} missing {missing[:2]}")
    n_paths = s.count('<path')
    check(f"{tag}: the vector SVG holds the plotted paths", n_paths > 20, f"{n_paths} paths")
    rect = fig._export_rect()
    q = _svg(fig.savefig(os.path.join(tmp, f"{tag}_qt.svg"), how='qt-svg'))
    vb = _svg_viewbox(q)
    check(f"{tag}: the Qt SVG is the saved area with the info block below it",
          vb is not None and abs(vb[0] - rect.width()) < 1.0 and abs(vb[1] - (rect.height() + H_info)) < 1.0,
          f"{vb} vs {(rect.width(), rect.height() + H_info)}")
    check(f"{tag}: the Qt SVG holds the 'settings' line", 'settings' in _svg_text(q))
    png = fig.savefig(os.path.join(tmp, f"{tag}.png"))
    k = plt.WINDOW_IMAGE_DPI / plt.DPI
    w, h = _png_size(png)
    dpi = _png_dpi(png)
    check(f"{tag}: the PNG is the saved area x {k:g} at {plt.WINDOW_IMAGE_DPI} dpi",
          abs(w - rect.width() * k) <= 1 and abs(h - rect.height() * k) <= 1 and dpi is not None
          and abs(dpi - plt.WINDOW_IMAGE_DPI) < 1, f"{(w, h)} vs {(rect.width() * k, rect.height() * k)}, {dpi} dpi")
    for p in os.listdir(tmp):
        if p.startswith(tag) and p.endswith('.png'):
            os.remove(os.path.join(tmp, p))
    return n_paths


def _run(tmp):
    from sm4_Browser import qtplot as plt
    from sm4_Browser import mplexport as mpx
    from sm4_Browser.checks import synth
    app = plt._app()

    # --- the rebuild_panels window, saved right after show(block=False) as its __main__ does ---
    from sm4_Browser.tools import rebuild_panels as rp
    fig = rp.open_window(rp.load('synthetic'), 0, rp.gate_at_open(True))
    plt.show(block=False)
    txt = _svg_text(_svg(fig.savefig(os.path.join(tmp, 'rebuild_at_open.svg'))))
    check("rebuild_panels: a save right after show(block=False) holds the settings section and the gate row",
          '\nsettings\n' in txt and 'gate: on' in txt)
    _settle(app)
    _check_window(plt, mpx, fig, 'rebuild_panels', tmp, fig.manager.info.toPlainText())
    plt.close(fig)

    # --- the browser on a synthetic experiment, each canvas ---
    data = os.path.join(tmp, 'experiment')
    os.makedirs(data)
    exp = synth.Experiment(data, v=(3e-12, -2e-12))
    t = synth.T_REF
    exp.topo(t, center=(0.0, 0.0), size_m=24e-9, n=64)
    t += 64 * 64 * 2e-3 * 2 + 30
    exp.grid(t, center=(0.0, 0.0), n_fast=16, n_lines=8, size_m=8e-9, npts=11, sweeps=('forward', 'reverse'),
             layout='paired')
    synth.install(exp)
    from sm4_Browser.browser import Browser
    plt._RUNNING[0] = True                       # plt.show() returns at once (no event loop in a check)
    b = Browser(data, scalefactor=1.0)
    _settle(app)
    check("the browser opened on a screen that is not Qt's offscreen platform", app.platformName() != 'offscreen',
          app.platformName())
    b._goto(next(i for i, p in enumerate(b.files) if b.kinds[p] == 'grid'))
    _settle(app)
    b.filters.chk_h.widget.click()               # one box changed
    _settle(app)
    check("browser: a change of a box shows in the settings section", 'Hampel I(V): on' in b.info.toPlainText())
    for fig, tag, col in ((b.fig_scan, 'browser scan area', 'left'), (b.fig_spec, 'browser spectra', 'right')):
        check(f"{tag}: its toolbar saves its own column", fig.export_root() is fig.export_root_widget
              and fig.export_root() is not b.win, col)
        _check_window(plt, mpx, fig, tag.replace(' ', '_'), tmp, fig.export_info())
    tif = b.fig_spec.savefig(os.path.join(tmp, 'browser_spectra.tif'))
    check("browser spectra: the TIFF is written", os.path.getsize(tif) > 0)
    os.remove(tif)

    # --- the Map and Line windows ---
    b._on_map()
    _settle(app)
    m = b.children[-1]
    _check_window(plt, mpx, m.fig, 'Map', tmp, m.fig.manager.info.toPlainText())
    b._on_line()
    _settle(app)
    w = b.children[-1]
    n_paths = _check_window(plt, mpx, w._int_fig, 'Line', tmp, w._int_fig.manager.info.toPlainText())
    eps = w._int_fig.savefig(os.path.join(tmp, 'Line.eps'))
    csv = w._int_fig.savefig(os.path.join(tmp, 'Line.csv'))
    check("Line: the EPS and the CSV are written", os.path.getsize(eps) > 0 and os.path.getsize(csv) > 0)

    # --- Spyder's module reloader: the package dropped from sys.modules, the Line window saved again ---
    import importlib
    old = sys.modules['sm4_Browser.qtplot']
    for name in [n for n in sys.modules if n == 'sm4_Browser' or n.startswith('sm4_Browser.')]:
        del sys.modules[name]
    importlib.import_module('sm4_Browser.controls')
    check("reload: a fresh qtplot imported", sys.modules['sm4_Browser.qtplot'] is not old)
    s = _svg(w._int_fig.savefig(os.path.join(tmp, 'Line_after_reload.svg')))
    check("reload: the old Line window's vector SVG holds as many paths as before", s.count('<path') == n_paths,
          f"{s.count('<path')} vs {n_paths}")
    w._int_filters.chk_s.widget.click()
    _settle(app)
    check("reload: the old Line window's settings section still follows its boxes",
          'SG I(V): on' in w._int_fig.manager.info.toPlainText())

    # --- close everything ---
    for win in (m, w):
        try:
            plt.close(getattr(win, '_int_fig', None) or win.fig)
        except Exception:
            pass
    try:
        b.win.close()
    except Exception:
        pass
    app.processEvents()


if __name__ == '__main__':
    if os.environ.get(_CHILD):
        sys.exit(_child())
    out = run()
    sys.exit(1 if out else 0)
