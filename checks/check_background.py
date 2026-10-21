"""
The scan layer's background placing with real worker processes (spawned, each with the synthetic
files installed), the disk cache of processed images (a second opening reads no file for the scan
area), and the recently used full files (stepping back re-reads nothing).

    python -m sm4_Browser.checks.check_background
"""
import io
import os
import sys
import time
import tempfile
import contextlib

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sm4_Browser import qtplot as plt
from sm4_Browser.checks import synth, perf_layer

RESULTS = []


def ok(name, cond, detail=''):
    """Record and print one result: PASS or FAIL, the name, the detail in parentheses."""
    RESULTS.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ''))


def settle(b, timeout=120):
    """Run the event loop until the browser's background placing (its queue, workers and futures) is idle, at most `timeout` s."""
    t0 = time.perf_counter()
    while (b._bg_running or b._bg_queue or b._bg_futures) and time.perf_counter() - t0 < timeout:
        plt._app().processEvents()
        time.sleep(0.005)


def run():
    """
    A 40-topo synthetic directory (perf_layer.build): 1. a browser with two worker processes places
    every topo and 2. writes the records to the scan cache, 3. the pool ends with the window; 4. a
    second browser without workers places every topo from the cache without reading a file; 5.
    stepping back and forth over files just visited reads nothing.  Prints each result; returns
    True when all passed.
    """
    tmp = tempfile.mkdtemp(prefix='sm4bg_')
    exp = perf_layer.build(tmp, 40)
    synth.install(exp)
    from sm4_Browser import browser as br, sm4file
    plt._RUNNING[0] = True
    # 1. worker processes
    br._BrowserCore.BG_WORKERS = 2
    br._BrowserCore.BG_POOL_INIT = (synth.install_files, (exp.directory, exp.files))
    with contextlib.redirect_stdout(io.StringIO()):
        b = br.Browser(tmp, scalefactor=1.0)
    b._on_show_mode('all')
    t0 = time.perf_counter()
    settle(b)
    pool_used = b._bg_pool not in (None, False)
    n_img = sum(1 for r in b.layer.records.values() if 'img' in r)
    n_topo = sum(1 for p in b.files if b.kinds[p] == 'topo')
    ok("1. worker processes place every topo", pool_used and n_img == n_topo,
       f"pool {'used' if pool_used else 'NOT used'}, {n_img} / {n_topo} topos placed in {time.perf_counter() - t0:.1f} s")
    cached = [f for f in os.listdir(b._scan_cache)] if os.path.isdir(b._scan_cache) else []
    ok("2. the records are written to the disk cache", len(cached) >= n_topo, f"{len(cached)} cache files")
    b.win.close()
    plt._app().processEvents()
    ok("3. the workers end with the window", b._bg_pool is False)
    # 2. a second opening: every record from the cache, no file read for the scan area
    br._BrowserCore.BG_WORKERS = 0
    br._BrowserCore.BG_POOL_INIT = None
    reads = []
    orig = sm4file.load

    def counting(path):
        reads.append(path)
        return orig(path)
    sm4file.load = counting
    with contextlib.redirect_stdout(io.StringIO()):
        b2 = br.Browser(tmp, scalefactor=1.0)
    reads.clear()
    b2._on_show_mode('all')
    settle(b2)
    n_img2 = sum(1 for r in b2.layer.records.values() if 'img' in r)
    ok("4. second opening: the scan area from the cache, no file read", n_img2 == n_topo and len(reads) == 0,
       f"{n_img2} topos placed, {len(reads)} file(s) read")
    # 3. stepping back over the loaded set re-reads nothing (recently used full files)
    for i in range(0, 36):                       # forward over a stretch of the experiment
        b2._goto(i)
    reads.clear()
    for i in range(35, 4, -1):                   # back over it (past the 25-file window) and forward again
        b2._goto(i)
    for i in range(5, 36):
        b2._goto(i)
    ok("5. back and forth over files just visited re-reads nothing (recently used files)", len(reads) == 0,
       f"{len(reads)} read")
    sm4file.load = orig
    print("ALL PASS" if all(RESULTS) else "SOME FAILED")
    return all(RESULTS)


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
