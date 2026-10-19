"""
One .sm4 file, step by step as the browser treats it, with the time of every step and what the
file records: the header scan, the loader, the pages (type, label, sizes, scales), the file's
kind, spectra (kept / stopped / never written), positions, image geometry, and the browser's
spectra pipeline.  A step that takes longer than 15 s has its Python stack printed every 15 s
(the place where it hangs); a crash of Qt or of a C extension is written to qtplot.CRASH_LOG.

Run it on the file the browser stops at (Spyder: set `path` below and press F5, or
python -m sm4_Browser.checks.diagnose_file <path>), and send the console output.
"""

if __package__ in (None, ''):
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import os
import sys
import time
import numpy as np
from sm4_Browser import qtplot


def step(name, fn):
    t0 = time.perf_counter()
    print(f"-> {name} ...", flush=True)
    with qtplot.watchdog(f"diagnose: {name}"):
        try:
            out = fn()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"   FAILED after {time.perf_counter() - t0:.2f} s: {exc!r}", flush=True)
            return None
    print(f"   ok, {time.perf_counter() - t0:.2f} s", flush=True)
    return out


def stats(name, a):
    a = np.asarray(a, dtype=np.float64)
    ok = np.isfinite(a)
    if a.size == 0:
        return f"{name}: empty"
    return (f"{name}: shape {a.shape}, finite {int(ok.sum())}/{a.size}"
            + (f", min {np.min(a[ok]):.4g}, max {np.max(a[ok]):.4g}" if ok.any() else ""))


def diagnose(path):
    from sm4_Browser import sm4file
    print(f"file: {path}  ({os.path.getsize(path)} bytes)")
    h = step("header scan (sm4_header_info)", lambda: sm4file.sm4_header_info(path))
    if h is not None:
        types, geom, line_types = h
        print(f"   page types: {types}\n   line types: {line_types}\n   kind: {sm4file.kind_from_types(types, line_types)}")
        print(f"   header geometry: {geom}")
    d = step("loader (sm4file.load: every page)", lambda: sm4file.load(path))
    if d is not None:
        for k, p in enumerate(d.pages):
            a = p.attrs
            print(f"   page {k}: {p.page_type} '{p.label}' data {None if p.data is None else np.shape(p.data)} "
                  f"X {a.get('RHK_Xsize')} x {a.get('RHK_Xscale')} {p.xunits}, Y {a.get('RHK_Ysize')} x {a.get('RHK_Yscale')}, "
                  f"grid {a.get('RHK_GridXsize')} x {a.get('RHK_GridYsize')}, "
                  f"positions {len(a.get('RHK_SpecDrift_Xcoord', []) or [])}, completed lines {a.get('RHK_CompletedLineCount')}")
    f = step("Sm4File (kind, spectra, geometry)", lambda: sm4file.Sm4File(path))
    if f is None:
        return
    print(f"   kind {f.kind}, image {f.nx} x {f.ny}, scales {f.xscale_m} / {f.yscale_m}, angle {f.angle_deg}, "
          f"centre {f.center_m}, grid {getattr(f, 'n_fast', None)} x {getattr(f, 'n_lines', None)}")
    if f.spectra is not None:
        s = f.spectra
        print(f"   spectra {s.n_spectra}, kept {s.n_acquired}, stopped {getattr(s, 'n_stopped', '?')}, "
              f"energy {len(s.energy)} points")
        print("   " + stats("positions (m)", f.spec_xy_m))
    if f.topo_pages:
        step("topo_raw('forward')", lambda: print("   " + stats("topo", f.topo_raw('forward'))))
        step("corners_m", lambda: print("   " + stats("corners", f.corners_m())))
        step("pixel_xy_m('forward')", lambda: print("   " + stats("pixel positions", f.pixel_xy_m('forward'))))
        step("spectrum_pixels('forward')", lambda: print(f"   {int(np.sum(f.spectrum_pixels('forward')[0][:, 0] >= 0))} spectra on pixels"))
    if f.spectra is not None:
        step("curves (the spectra panel's pipeline)",
             lambda: print("   " + stats("curves", f.curves(None, None, False, 0.0, channel=f.channels()[0],
                                                           normalize=False, calibrate=False))))
    step("measurement_info", lambda: print(f"   {len(f.measurement_info())} rows"))
    print(f"done; a crash would have been written to {qtplot.CRASH_LOG}")


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else r'experiment_dir/file.sm4'
    diagnose(path)
