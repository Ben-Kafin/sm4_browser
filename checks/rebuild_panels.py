"""
Five panels comparing the two ways of rebuilding the current and reading g, on one sweep.

    1. the recorded current (0 V value subtracted), the integrated calibrated LIA (page-sum rebuild),
       and the Feenstra ratio integrated and exponentiated (ratio rebuild), on one panel
    2. the current residual, recorded minus rebuilt, for each route (left axis), with g - 1 for
       each route on a right axis (dashed); both axes centred on 0 and filling the full height
    3. the recorded LIA page, and LIA / g for each route, same colours
    4. the LIA residual, recorded minus LIA / g, for each route (left axis), with g - 1 on a right
       axis (dashed), both centred on 0 and filling the full height
    5. both g's drawn over the recorded LIA (left axis, full height) and the recorded current
       (right axis, full height), with the Feenstra ratio of the calibrated page (LIAcal·V / Isub0,
       the ratio the step integrates) drawn on the LIA's axis at the same scale; g has its own
       axis, centred on g = 1 and filling the full height; a vertical grid line at every point

Both rebuilds are made per side of 0 V from the window edge inward on every point whose ratio is
finite (no band around 0 V: only the 0 V point itself, which belongs to neither side, keeps g = 1;
that point is identified as |V| below a thousandth of the point spacing, since an SM4 axis built
from float32 fields puts it at a few 1e-8 V rather than at 0),
and each is aligned once per side so that the mean of ln|Isub0| - ln|rebuilt| is 0 on that side, exactly
as gap.py aligns its rebuild.

Colours: recorded = black, integrated LIA route = blue, Feenstra ratio route = red.

Data: either the synthetic sweep of denoise_walkthrough.py (SOURCE = 'synthetic', truth known) or
one sweep of an RHK SM4 file read with spym (SOURCE = path, SWEEP = column index).  The SM4 reader
expects the LIA page and the current page at the indices given below (6 and 7 in the test file,
both 141 x 1024, scaled by RHK_Zscale, V = RHK_Xoffset + RHK_Xscale x index).
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq

GAP_MIN_POINTS = 3       # gap.py: a side needs this many usable points for its integral

COL_REC, COL_PAGE, COL_RATIO = "black", "tab:blue", "tab:red"


# ----------------------------------------------------------------------------------------------
#  data sources
# ----------------------------------------------------------------------------------------------
def synthetic_sweep(seed=7, v_lo=-0.5, v_hi=0.5, npts=101, v_mod=0.025, gap_jitter=0.07,
                    lockin_offset=0.02, lockin_floor=0.03, current_floor=2e-4, current_offset=0.003,
                    v_set=0.5):
    """The denoise_walkthrough.py model: a sloping metal with a bump at -0.3 V, the same gap factor on
    both pages, each page with its own electronic noise.  Currents in nA, slopes in nA/V."""
    rng = np.random.default_rng(seed)
    V = np.linspace(v_lo, v_hi, npts)
    D = 1.0 + 0.3 * V + 0.4 * np.exp(-((V + 0.3) / 0.08) ** 2)
    J = cumulative_trapezoid(D, V, initial=0.0)
    J = J - np.interp(0.0, V, J)
    a_true = 1.0 / v_mod
    g = 1.0 + gap_jitter * rng.standard_normal(V.size)
    e = lockin_offset * np.mean(D)
    n_L = lockin_floor * np.mean(D) / a_true * rng.standard_normal(V.size)
    n_I = current_floor * np.max(np.abs(J)) * rng.standard_normal(V.size)
    LIA = (g * D + e) / a_true + n_L
    I = g * J + current_offset * np.max(np.abs(J)) + n_I
    k = int(np.argmin(np.abs(V - v_set)))
    return dict(V=V, LIA=LIA, I=I, V_set=v_set, I_set=float(I[k]), unit="nA", g_true=g,
                title=f"synthetic sweep (seed {seed}, {100*gap_jitter:.0f} % gap jitter)")


def sm4_sweep(path, sweep, lia_page=6, current_page=7, unit_scale=1e12, unit="pA"):
    """One sweep (one column) of an RHK SM4 spectroscopy file.  Inside the repo the package's own
    reader is used (sm4_Browser.sm4file), so the energy axis, the pages and the column order are the
    browser's; elsewhere spym reads the pages at lia_page / current_page.  The Vbias lock's pin is the
    file's setpoint: V_set = RHK_Bias, I_set = RHK_Current.  Currents are returned in `unit`."""
    try:
        from sm4_Browser import sm4file
        f = sm4file.Sm4File(path)
        sp = f.spectra
        V = np.asarray(sp.energy, dtype=np.float64)
        LIA = np.asarray(sp.lia, dtype=np.float64)[:, sweep] * unit_scale
        I = np.asarray(sp.iv, dtype=np.float64)[:, sweep] * unit_scale
        V_set, I_set, name = float(sp.V_bias), float(sp.I_setpoint) * unit_scale, f.name
    except ImportError:
        from spym.io import rhksm4
        f = rhksm4.load(path)
        pL, pI = f[lia_page], f[current_page]
        a = pI.attrs
        V = float(a["RHK_Xoffset"]) + float(a["RHK_Xscale"]) * np.arange(int(a["RHK_Xsize"]))
        LIA = pL.data[:, sweep].astype(np.float64) * float(pL.attrs["RHK_Zscale"]) * unit_scale
        I = pI.data[:, sweep].astype(np.float64) * float(a["RHK_Zscale"]) * unit_scale
        V_set, I_set, name = float(a["RHK_Bias"]), float(a["RHK_Current"]) * unit_scale, os.path.basename(path)
    if not np.any(I != 0.0):
        raise ValueError(f"sweep {sweep} of {path} was never acquired (all zeros)")
    return dict(V=V, LIA=LIA, I=I, V_set=V_set, I_set=I_set, unit=unit, g_true=None,
                title=f"{name}  sweep {sweep}")


def npz_sweep(path, sweep, unit_scale=1e12, unit="pA"):
    """One column of a synthetic set saved as .npz with the keys the checks use: energy, lia, iv (in A),
    vmod, and V_bias / I_setpoint (optional); g (the gap factor put in, (npts, N)) is used when present."""
    d = np.load(path, allow_pickle=True)
    V = np.asarray(d["energy"], dtype=np.float64)
    LIA = np.asarray(d["lia"], dtype=np.float64)[:, sweep] * unit_scale
    I = np.asarray(d["iv"], dtype=np.float64)[:, sweep] * unit_scale
    V_set = float(d["V_bias"]) if "V_bias" in d else float(V[0])
    I_set = float(d["I_setpoint"]) * unit_scale if "I_setpoint" in d else float(I[np.argmin(np.abs(V - V_set))])
    g_true = np.asarray(d["g"], dtype=np.float64)[:, sweep] if "g" in d else None
    return dict(V=V, LIA=LIA, I=I, V_set=V_set, I_set=I_set, unit=unit, g_true=g_true,
                title=f"{os.path.basename(path)}  column {sweep}")


def load(source, sweep=0):
    """'synthetic' -> the built-in synthetic sweep; a .npz -> one column of a saved synthetic set;
    anything else -> one sweep of an SM4 file."""
    if source == "synthetic":
        return synthetic_sweep()
    if source.lower().endswith(".npz"):
        return npz_sweep(source, sweep)
    return sm4_sweep(source, sweep)


# ----------------------------------------------------------------------------------------------
#  the step's pieces (as gap.py / calibration.py do them on one sweep)
# ----------------------------------------------------------------------------------------------
def calibrate(V, LIA, I, lock, V_set, I_set):
    """I = a*X + b*V + c with X the integral of LIA from the lowest bias.  lock 'none': plain least
    squares.  lock 'vbias': the unshifted Vbias lock, c = I_set - a*X_set at the setpoint point and
    a, b the least squares of I - I_set against (X - X_set, V)."""
    o = np.argsort(V)
    X = np.empty_like(LIA); X[o] = cumulative_trapezoid(LIA[o], V[o], initial=0.0)
    if lock == "vbias":
        k = int(np.argmin(np.abs(V - V_set)))
        a, b = np.linalg.lstsq(np.column_stack([X - X[k], V]), I - I_set, rcond=None)[0]
        c = I_set - a * X[k]
    else:
        a, b, c = np.linalg.lstsq(np.column_stack([X, V, np.ones_like(V)]), I, rcond=None)[0]
    return a, b, c, X


def zero_sub(V, I):
    o = np.argsort(V)
    return I - np.interp(0.0, V[o], I[o])


def feenstra_ratio(V, num, Isub0):
    with np.errstate(divide="ignore", invalid="ignore"):
        F = num * V / Isub0
    o = np.argsort(V)
    Fo, Vo = F[o], V[o]
    for i in np.flatnonzero(np.isclose(Vo, 0.0)):
        if 0 < i < Vo.size - 1:
            Fo[i] = np.interp(Vo[i], [Vo[i - 1], Vo[i + 1]], [Fo[i - 1], Fo[i + 1]])
    F[o] = Fo
    return F


def usable_points(V, Isub0, F):
    """gap.py (band removed 2026-11-03): every point with a finite ratio and a non-zero current."""
    return np.isfinite(F) & (Isub0 != 0.0)


def _sides(V, usable):
    """The usable points of each side of 0 V, ordered from the window edge toward 0 V.  The point at
    0 V as the file records it (|V| below a thousandth of the point spacing) belongs to neither."""
    step = np.median(np.abs(np.diff(V))) if V.size > 1 else 1.0
    zero = np.abs(V) < 1e-3 * step
    for side in ((V < 0.0) & ~zero, (V > 0.0) & ~zero):
        pts = np.flatnonzero(side & usable)
        if pts.size < GAP_MIN_POINTS:
            continue
        yield pts if V[pts[0]] < 0.0 else pts[::-1]


def ratio_rebuild(V, F, Isub0, usable):
    """gap.py: per side, ln I_rebuilt = cumulative trapezoid of F against ln|V| from the edge inward, aligned so
    mean(ln|Isub0| - ln I_rebuilt) = 0; g = |Isub0| / I_rebuilt.  Returns g, |I_rebuilt| (NaN where not made)."""
    o = np.argsort(V)
    e, f, i0, u = V[o], F[o], Isub0[o], usable[o]
    g = np.ones_like(i0); reb = np.full_like(i0, np.nan)
    for run in _sides(e, u):
        v = e[run]
        ln_rec = cumulative_trapezoid(f[run], np.log(np.abs(v)), initial=0.0)    # F d ln|V|: exact for a power-law current between points
        ln_rec = ln_rec - np.mean(ln_rec - np.log(np.abs(i0[run])))
        reb[run] = np.exp(ln_rec)
        g[run] = np.abs(i0[run]) / reb[run]
    out_g, out_r = np.empty_like(g), np.empty_like(reb)
    out_g[o], out_r[o] = g, reb
    return out_g, out_r


def page_rebuild(V, LIAcal, Isub0, usable):
    """The no-ratio route: per side, I_rebuilt = cumulative trapezoid of LIAcal from the edge inward plus
    one constant, the constant chosen so mean(ln|Isub0| - ln|I_rebuilt|) = 0 on that side (the same
    alignment as gap.py; the constant is additive here because nothing was exponentiated).
    g = |Isub0| / |I_rebuilt|.  Returns g, |I_rebuilt| (NaN where not made)."""
    o = np.argsort(V)
    e, l, i0, u = V[o], LIAcal[o], Isub0[o], usable[o]
    g = np.ones_like(i0); reb = np.full_like(i0, np.nan)
    for run in _sides(e, u):
        v = e[run]
        s = cumulative_trapezoid(l[run], v, initial=0.0)
        target = np.mean(np.log(np.abs(i0[run])))
        fun = lambda C: np.mean(np.log(np.abs(s + C))) - target
        # s + C must keep the sign of Isub0 on this side: C > -min(s) if the current is positive there,
        # C < -max(s) if it is negative; fun is monotonic on each of those intervals
        eps, span = 1e-9 * np.max(np.abs(i0)), 50.0 * np.max(np.abs(i0))
        if np.mean(i0[run]) > 0:
            lo, hi = -np.min(s) + eps, -np.min(s) + span
        else:
            lo, hi = -np.max(s) - span, -np.max(s) - eps
        try:
            C = brentq(fun, lo, hi)
        except ValueError:
            C = np.mean(i0[run] - s)                      # fallback: plain mean alignment
        reb[run] = np.abs(s + C)
        g[run] = np.abs(i0[run]) / reb[run]
    out_g, out_r = np.empty_like(g), np.empty_like(reb)
    out_g[o], out_r[o] = g, reb
    return out_g, out_r



def residual_panel(ax, res_page, res_ratio, g_page, g_ratio, ylabel, what, kw_page, kw_ratio, Vs, o):
    """A residual for each route on the left axis and g - 1 for each route (dashed) on a right axis,
    both axes centred on 0 and spanning the largest excursion, so the two can be read against each other."""
    fin = lambda x: x[np.isfinite(x)]
    half_r = 1.03 * max(np.max(np.abs(fin(res_page))), np.max(np.abs(fin(res_ratio))), 1e-12)
    ax.plot(Vs, res_page[o], marker="o", ms=3, label=f"{what} (integrated LIA route)", **kw_page)
    ax.plot(Vs, res_ratio[o], marker="o", ms=3, label=f"{what} (Feenstra ratio route)", **kw_ratio)
    ax.set_ylim(-half_r, half_r)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="0.8", lw=0.8); ax.axvline(0, color="0.8", lw=0.8)
    axg = ax.twinx()
    axg.plot(Vs, (g_page - 1)[o], ls="--", lw=1.0, color=COL_PAGE, label="g − 1 (integrated LIA route)")
    axg.plot(Vs, (g_ratio - 1)[o], ls="--", lw=1.0, color=COL_RATIO, label="g − 1 (Feenstra ratio route)")
    half_g = 1.03 * max(np.max(np.abs(fin(g_page) - 1)), np.max(np.abs(fin(g_ratio) - 1)), 1e-6)
    axg.set_ylim(-half_g, half_g)
    axg.set_ylabel("g − 1 (dashed)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axg.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=9)


# ----------------------------------------------------------------------------------------------
#  the figure
# ----------------------------------------------------------------------------------------------
def make_figure(d, lock="vbias", save_to=None):
    V, LIA, I, unit = d["V"], d["LIA"], d["I"], d["unit"]
    a, b, c, X = calibrate(V, LIA, I, lock, d["V_set"], d["I_set"])
    LIAcal = a * LIA + b
    Isub0 = zero_sub(V, I)
    F = feenstra_ratio(V, LIAcal, Isub0)
    usable = usable_points(V, Isub0, F)
    g_ratio, I_ratio = ratio_rebuild(V, F, Isub0, usable)
    g_page, I_page = page_rebuild(V, LIAcal, Isub0, usable)
    sgn = np.sign(Isub0)                                   # the rebuilds are magnitudes; show them signed

    o = np.argsort(V)
    Vs = V[o]
    fig, axes = plt.subplots(5, 1, figsize=(11, 21), sharex=True)
    kw_rec = dict(color=COL_REC, lw=1.2)
    kw_page = dict(color=COL_PAGE, lw=1.4)
    kw_ratio = dict(color=COL_RATIO, lw=1.4)

    # 1. the three currents
    ax = axes[0]
    ax.plot(Vs, Isub0[o], marker="o", ms=3, label="recorded I (0 V value subtracted)", **kw_rec)
    ax.plot(Vs, (sgn * I_page)[o], label="integrated calibrated LIA (a·LIA + b added up, aligned)", **kw_page)
    ax.plot(Vs, (sgn * I_ratio)[o], label="Feenstra ratio integrated and exponentiated (aligned)", **kw_ratio)
    ax.axhline(0, color="0.8", lw=0.8); ax.axvline(0, color="0.8", lw=0.8)
    ax.set_ylabel(f"current ({unit})")
    ax.set_title(f"{d['title']}   calibration: lock = {lock}, a = {a:.3g}, b = {b:+.3g}, c = {c:+.3g}")
    ax.legend(loc="best", fontsize=9)

    # 2. the current residual, recorded minus rebuilt, with g - 1 on a right axis
    residual_panel(axes[1], Isub0 - sgn * I_page, Isub0 - sgn * I_ratio, g_page, g_ratio,
                   f"recorded I − rebuilt I ({unit})", "recorded I − rebuilt", kw_page, kw_ratio, Vs, o)

    # 3. the LIA page and LIA / g
    ax = axes[2]
    ax.plot(Vs, LIA[o], marker="o", ms=3, label="recorded LIA", **kw_rec)
    ax.plot(Vs, (LIA / g_page)[o], label="LIA / g (integrated LIA route)", **kw_page)
    ax.plot(Vs, (LIA / g_ratio)[o], label="LIA / g (Feenstra ratio route)", **kw_ratio)
    ax.axvline(0, color="0.8", lw=0.8)
    ax.set_ylabel("LIA (as recorded)")
    ax.legend(loc="best", fontsize=9)

    # 4. the LIA residual, recorded minus LIA / g, with g - 1 on a right axis
    residual_panel(axes[3], LIA - LIA / g_page, LIA - LIA / g_ratio, g_page, g_ratio,
                   "recorded LIA − LIA / g (LIA units)", "recorded LIA − LIA / g", kw_page, kw_ratio, Vs, o)

    # 5. both g's over the recorded LIA (left, full height) and the recorded current (right, full height);
    #    g on its own axis, centred on 1 and filling the full height
    ax = axes[4]
    pad = lambda lo, hi: (lo - 0.03 * (hi - lo), hi + 0.03 * (hi - lo))
    ax.plot(Vs, LIA[o], label="recorded LIA (left axis)", **kw_rec)
    Ff = F[np.isfinite(F)]
    ax.plot(Vs, F[o], color="tab:green", lw=1.2, label="Feenstra ratio of the calibrated page, LIAcal·V / Isub0 (left axis, same scale)")
    ax.set_ylim(*pad(min(np.min(LIA), np.min(Ff)), max(np.max(LIA), np.max(Ff))))
    ax.set_ylabel("LIA (as recorded)  /  Feenstra ratio")
    ax.set_xlabel("V (V)")
    ax_i = ax.twinx()
    ax_i.plot(Vs, Isub0[o], color="0.55", lw=1.2, label="recorded I, 0 V value subtracted (right axis)")
    ax_i.set_ylim(*pad(np.min(Isub0), np.max(Isub0)))
    ax_i.set_ylabel(f"current ({unit})", color="0.4")
    ax_g = ax.twinx()
    ax_g.spines["right"].set_position(("axes", 1.09))
    ax_g.plot(Vs, g_page[o], marker="o", ms=3, label="g (integrated LIA route)", **kw_page)
    ax_g.plot(Vs, g_ratio[o], marker="o", ms=3, label="g (Feenstra ratio route)", **kw_ratio)
    half = 1.03 * max(np.max(np.abs(g_page[np.isfinite(g_page)] - 1)), np.max(np.abs(g_ratio[np.isfinite(g_ratio)] - 1)), 1e-6)
    ax_g.set_ylim(1 - half, 1 + half)                     # g = 1 at the vertical centre, full height
    ax_g.axhline(1, color=COL_RATIO, lw=0.6, alpha=0.4)
    ax_g.set_ylabel("g (centred on 1)")
    ax.axvline(0, color="0.8", lw=0.8)
    lines = ax.get_legend_handles_labels()[0] + ax_i.get_legend_handles_labels()[0] + ax_g.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax_i.get_legend_handles_labels()[1] + ax_g.get_legend_handles_labels()[1]
    ax.legend(lines, labels, loc="best", fontsize=9)
    step = float(np.median(np.diff(Vs)))
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10 * step))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(step))
    ax.grid(axis="x", which="major", color="0.55", lw=0.6)
    ax.grid(axis="x", which="minor", color="0.85", lw=0.4)

    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=140, bbox_inches="tight")
        print(f"wrote {save_to}")
    u = usable
    print(f"a = {a:.4g}, b = {b:+.4g}, c = {c:+.4g};  usable points {u.sum()} of {u.size}")
    print(f"rms(g - 1): integrated-LIA route {100*np.sqrt(np.mean((g_page[u]-1)**2)):.2f} %, "
          f"ratio route {100*np.sqrt(np.mean((g_ratio[u]-1)**2)):.2f} %")
    if d.get("g_true") is not None:
        gt = d["g_true"]
        print(f"rms error vs the g put in: integrated-LIA route {100*np.sqrt(np.mean((g_page[u]/gt[u]-1)**2)):.2f} %, "
              f"ratio route {100*np.sqrt(np.mean((g_ratio[u]/gt[u]-1)**2)):.2f} %")
    return fig


if __name__ == "__main__":
    # Spyder: set these and press F5.  Command line: python rebuild_panels.py [SOURCE] [SWEEP] [LOCK] [OUT]
    #   SOURCE  'synthetic', a .npz synthetic set, or an .sm4 file
    #   SWEEP   column index of the sweep (ignored for 'synthetic')
    #   LOCK    'vbias' (the unshifted Vbias lock) or 'none' (plain least squares)
    #   OUT     the .png to write (default: beside this file, named after the source)
    SOURCE = "synthetic"
    SWEEP = 0
    LOCK = "vbias"
    OUT = None

    args = sys.argv[1:]
    if len(args) >= 1: SOURCE = args[0]
    if len(args) >= 2: SWEEP = int(args[1])
    if len(args) >= 3: LOCK = args[2]
    if len(args) >= 4: OUT = args[3]
    if OUT is None:
        here = os.path.dirname(os.path.abspath(__file__))
        stem = "synthetic" if SOURCE == "synthetic" else f"{os.path.splitext(os.path.basename(SOURCE))[0]}_sweep{SWEEP}"
        OUT = os.path.join(here, f"rebuild_panels_{stem}.png")
    make_figure(load(SOURCE, SWEEP), lock=LOCK, save_to=OUT)
