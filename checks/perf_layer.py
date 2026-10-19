"""
Performance of the scan layer on a large synthetic directory (offscreen): startup, background
placing of every file, stepping with every file shown, a drift change.

    python -m sm4_Browser.checks.perf_layer [n_topos]
"""
import os
import sys
import tempfile
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from sm4_Browser import qtplot as plt
from sm4_Browser.checks import synth


def build(directory, n_topos, n_px=96):
    exp = synth.Experiment(directory, v=(2e-12, -1e-12))
    t = synth.T_REF
    rng = np.random.default_rng(3)
    centre = np.zeros(2)
    for k in range(n_topos):
        centre = centre + rng.normal(0, 1.5e-9, 2)
        exp.topo(t, center=tuple(centre), size_m=30e-9, n=n_px, period=1e-4)
        t += n_px * n_px * 1e-4 * 2 + 5
        if k % 4 == 3:
            exp.point(t, xy=tuple(centre + rng.normal(0, 3e-9, 2)), n_sweeps=2)
            t += 10
    return exp


def run(n_topos=300):
    tmp = tempfile.mkdtemp(prefix='sm4perf_')
    t0 = time.perf_counter()
    exp = build(tmp, n_topos)
    synth.install(exp)
    print(f"{len(exp.files)} synthetic files built in {time.perf_counter() - t0:.1f} s")
    from sm4_Browser.browser import Browser
    plt._RUNNING[0] = True
    t0 = time.perf_counter()
    b = Browser(tmp, scalefactor=1.0)
    app = plt._app()
    app.processEvents()
    print(f"window open (header scan, drift build, sites, first file): {time.perf_counter() - t0:.1f} s")
    b._on_show_mode('all')
    t0 = time.perf_counter()
    turns, worst = 0, 0.0
    while b._bg_running or b._bg_queue:
        s = time.perf_counter()
        app.processEvents()
        worst = max(worst, time.perf_counter() - s)
        turns += 1
    print(f"every file placed in the background: {time.perf_counter() - t0:.1f} s over {turns} event-loop turns, "
          f"longest turn {worst * 1000:.0f} ms (the window stays responsive between turns)")
    print(f"images on the scan area: {len(b.layer.items)}")
    n = min(60, len(b.files) - 1)
    tg = tp = 0.0
    for i in range(n):
        s = time.perf_counter()
        b._goto(len(b.files) - 1 - i)
        tg += time.perf_counter() - s
        s = time.perf_counter()
        app.processEvents()
        tp += time.perf_counter() - s
    print(f"step with every file shown: {(tg + tp) / n * 1000:.0f} ms per file ({tg / n * 1000:.0f} ms update incl. "
          f"loading, {tp / n * 1000:.0f} ms painting)")
    b._on_show_mode('up to current')
    t0 = time.perf_counter()
    for i in range(n):
        b._goto(len(b.files) - 1 - i)
        app.processEvents()
    print(f"step showing up to the current file: {(time.perf_counter() - t0) / n * 1000:.0f} ms per file")
    t0 = time.perf_counter()
    b._on_drift_toggle('auto', False)
    app.processEvents()
    b._on_drift_toggle('auto', True)
    app.processEvents()
    print(f"drift toggle off and on (every image moved): {(time.perf_counter() - t0) / 2 * 1000:.0f} ms each")
    t0 = time.perf_counter()
    b.win.grab()
    print(f"one full paint of the window: {(time.perf_counter() - t0) * 1000:.0f} ms")


if __name__ == '__main__':
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
