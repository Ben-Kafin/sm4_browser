"""
Zero crossing of the raw current I(V) for every sweep of every file in the
saved averages (avg*_average.json) of the package's sm4s folder: per sweep,
per file, per saved average, and over all of them (feenstra.iv_zero_crossings
does the per-sweep work).  Run as a script from anywhere; the paths below
are relative to this file (checks/ inside the package).
"""
import os, sys, json, glob, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # the folder holding the package
import numpy as np
from sm4_Browser import sm4file
from sm4_Browser.feenstra import iv_zero_crossings

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'sm4s')   # the package's sm4s folder
A = os.path.join(D, 'averages')


def crossings(f):
    """Per acquired sweep: (V0 in volts, I at nominal 0 V in pA, slope near 0 in pA/V), nearest sign change to 0 V."""
    return iv_zero_crossings(f.energy, f.iv, f.acquired)


def stats(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    n = len(v)
    m = v.mean() if n else np.nan
    sd = v.std(ddof=1) if n > 1 else np.nan
    return n, m, sd, (sd / np.sqrt(n) if n > 1 else np.nan)


files = {}
per_avg = {}
for rec in sorted(glob.glob(os.path.join(A, 'avg*_average.json'))):
    d = json.load(open(rec))
    name = os.path.basename(rec).replace('_average.json', '')
    removed = {(int(i), int(j)) for i, j in d.get('removed', [])}
    print(f"\n=== {name}  (saved {d.get('saved')}, format {d.get('format')}, {len(d['files'])} files, "
          f"{len(removed)} removed sweep(s))")
    kept_all = []
    file_means = []
    for i, ent in enumerate(d['files']):
        fn = ent['name']
        if fn not in files:
            f = sm4file.Sm4File(os.path.join(D, fn))
            files[fn] = (f, *crossings(f))
        f, cr, e0 = files[fn]
        n_acq = int(np.sum(f.acquired)); n_all = len(f.acquired)
        gone = sorted(j for ii, j in removed if ii == i)
        keep = [j for j in range(cr.shape[0]) if j not in gone]
        v_all = cr[:, 0] * 1e3
        n, m, sd, se = stats(v_all[keep]) if keep else (0, np.nan, np.nan, np.nan)
        tag = fn.split('_CT_')[1].replace('.sm4', '')
        print(f"  {tag}: {n_all} sweeps ({n_acq} acquired), nominal-0 sample at {e0 * 1e3:+.3f} mV; "
              f"crossings (mV): {np.array2string(v_all, precision=1, separator=', ')}"
              + (f"; removed sweeps {[j + 1 for j in gone]}" if gone else ''))
        print(f"      I at nominal 0 V (pA): {np.array2string(cr[:, 1], precision=3, separator=', ')}; "
              f"slope near 0 (pA/V): {np.array2string(cr[:, 2], precision=1, separator=', ')}")
        if keep:
            print(f"      kept {n}: mean {m:+.2f} mV, std {sd:.2f} mV, SEM {se:.2f} mV")
            kept_all.extend(v_all[keep]); file_means.append(m)
        else:
            print("      every sweep removed: no contribution to the saved average")
    n, m, sd, se = stats(kept_all)
    nf, mf, sdf, sef = stats(file_means)
    print(f"  {name} pooled over its {n} kept sweeps: mean {m:+.2f} mV, std {sd:.2f} mV, SEM {se:.2f} mV")
    print(f"  {name} as the mean of its {nf} file means: {mf:+.2f} mV, std {sdf:.2f} mV, SEM {sef:.2f} mV")
    per_avg[name] = (n, m, sd, se, mf, sef)

print("\n=== all saved averages")
vals = [per_avg[k][1] for k in per_avg]
n, m, sd, se = stats(vals)
print(f"  mean of the {n} saved-average values (pooled means): {m:+.2f} mV, std {sd:.2f} mV, SEM {se:.2f} mV")
w = np.array([per_avg[k][0] for k in per_avg], float)
mw = np.sum(w * np.array(vals)) / w.sum()
print(f"  sweep-weighted mean of the same: {mw:+.2f} mV")
# every unique sweep once (a file in two averages counted once)
allv = np.concatenate([files[fn][1][:, 0] * 1e3 for fn in files])
n, m, sd, se = stats(allv)
print(f"  every unique sweep of the {len(files)} files once, including removed sweeps: n {n}, mean {m:+.2f} mV, std {sd:.2f} mV, SEM {se:.2f} mV")
allI = np.concatenate([files[fn][1][:, 1] for fn in files])
print(f"  I at nominal 0 V over the same sweeps: mean {allI.mean():.3f} pA, std {allI.std(ddof=1):.3f} pA, "
      f"min {allI.min():.3f}, max {allI.max():.3f}")
