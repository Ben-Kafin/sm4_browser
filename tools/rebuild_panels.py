"""
Five panels of the denoise step's rebuild of the current and its g on one sweep, every piece read
from the engine (sts.STS.stages with the denoise row on; stages.step, a gap.DenoiseStep): nothing
of the step is computed here.

    1. the recorded current, 0 V value subtracted (stages.step.Isub0), and the current the step
       rebuilt from the Feenstra ratio (stages.step.I_rebuilt, aligned per side; drawn signed)
    2. the current residual, recorded minus rebuilt (left axis), with g - 1 on a right axis
       (dashed; dotted, g before the soft edge); both axes centred on 0 and filling the full height
    3. the LIA page as the step sees it (stages.lia_f; no filters here, so as recorded) and LIA / g
       (stages.lia_d)
    4. the LIA residual, LIA minus LIA / g (left axis), with g - 1 on a right axis (dashed; dotted,
       g before the soft edge), both centred on 0 and filling the full height
    5. g drawn over the LIA page (left axis, full height) and the recorded current (right axis,
       full height), with the Feenstra ratio of the step's calibrated page (stages.step.Fcal =
       cal·V / Isub0, the ratio the step integrates) on the LIA's axis at the same scale; g has
       its own axis, centred on g = 1 and filling the full height, g before the soft edge dotted
       beside it; a vertical grid line at every point

The step (gap.py): per side of 0 V from the window edge inward on every point whose ratio is
finite (no band around 0 V: the 0 V point itself, which belongs to neither side, keeps g = 1), the
rebuilt current aligned once per side so that the mean of ln|Isub0| - ln I_rebuilt is 0 there, and
g_raw = |Isub0| / I_rebuilt.  The soft edge (the 'soft edge' row under 'denoise'; SOFT_EDGE below):
each point's correction scaled by w = jitter^2 / (jitter^2 + excess^2), measured on every acquired
column of the file (gap.weight_curve, stages.step.curve), g = 1 + (g_raw - 1) w; the title gives
the jitter and the current at which w crosses 0.5, the console the curve bin by bin.  The engine
runs on every acquired column (the gate is cross-sweep) with the lock chosen below and the gate at
its defaults, and the sweep is drawn from the result; the ratio and g are shown over the points
the step solved (where I_rebuilt is finite).

Colours: recorded = black, the step = red, the ratio = green.

Data: the synthetic sweep below (SOURCE = 'synthetic'; the g put in is known; the soft edge's
curve is then measured on N_SYNTHETIC sweeps of the model, seeds 0 and up), one column of a
synthetic set saved as .npz (SOURCE = path, SWEEP = column; the keys the checks use: energy, lia,
iv, vmod, V_bias, I_setpoint, acquired, g), or one sweep of an RHK SM4 file through the package's
reader (SOURCE = path, SWEEP = the column's index among the file's columns).  A tool
(sm4_Browser/tools/): it modifies nothing in the package and only imports.

Spyder: set the run block below and press F5; or
python -m sm4_Browser.tools.rebuild_panels [SOURCE] [SWEEP] [LOCK] [OUT].
"""

if __package__ in (None, ''):
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from sm4_Browser import calibration
from sm4_Browser.sts import STS, PipelineState

COL_REC, COL_STEP, COL_RATIO = "black", "tab:red", "tab:green"
N_SYNTHETIC = 32         # the synthetic set's sweeps (seeds 0 .. N_SYNTHETIC - 1), the soft edge's curve measured on them
LOCKS = {'0lock': dict(lock0=True), 'vbias': dict(lockv=True), 'none': {}}


# ----------------------------------------------------------------------------------------------
#  data sources: (energy, lia, iv, acquired, vmod, V_bias, I_setpoint, sweeps, name, g_true, the
#  display scale and unit of the current), the arrays (npts, N) as the engine takes them
# ----------------------------------------------------------------------------------------------
def synthetic_sweep(seed=7, v_lo=-0.5, v_hi=0.5, npts=101, v_mod=0.025, gap_jitter=0.07,
                    lockin_offset=0.02, lockin_floor=0.03, current_floor=2e-4, current_offset=0.003,
                    v_set=0.5):
    """The denoise_walkthrough.py model: a sloping metal with a bump at -0.3 V, the same gap factor on
    both pages, each page with its own electronic noise.  Currents in nA, slopes in nA/V; one column,
    a forward sweep with its setpoint at v_set.  Returns (V, LIA, I, g, V_set, I_set)."""
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
    return V, LIA, I, g, v_set, float(I[k])


def synthetic_set(seed=7, n=N_SYNTHETIC, v_mod=0.025, gap_jitter=0.07):
    """The synthetic sweep of `seed` as column 0, with n - 1 more sweeps of the model (seeds 0 and up,
    `seed` itself not repeated) beside it, so the soft edge's curve has a set of sweeps to measure on;
    the setpoint current per column (the engine takes an array)."""
    seeds = [seed] + [s for s in range(n + 1) if s != seed][:n - 1]
    cols = [synthetic_sweep(seed=s, v_mod=v_mod, gap_jitter=gap_jitter) for s in seeds]
    V = cols[0][0]
    return dict(energy=V, lia=np.column_stack([c[1] for c in cols]), iv=np.column_stack([c[2] for c in cols]),
                acquired=np.ones(len(cols), dtype=bool), vmod=v_mod, V_bias=cols[0][4],
                I_setpoint=np.array([c[5] for c in cols]), sweeps=None,
                g_true=np.column_stack([c[3] for c in cols]), unit_scale=1.0, unit="nA",
                name=f"synthetic sweep (seed {seed}, {100 * gap_jitter:.0f} % gap jitter; the curve on {len(cols)} sweeps)")


def npz_set(path, unit_scale=1e12, unit="pA"):
    """A synthetic set saved as .npz with the keys the checks use (energy, lia, iv in A, vmod, V_bias,
    I_setpoint and acquired optional; g, the gap factor put in, (npts, N), used when present)."""
    d = np.load(path, allow_pickle=True)
    lia = np.asarray(d["lia"], dtype=np.float64)
    acq = np.asarray(d["acquired"], dtype=bool) if "acquired" in d else np.ones(lia.shape[1], dtype=bool)
    return dict(energy=np.asarray(d["energy"], dtype=np.float64), lia=lia, iv=np.asarray(d["iv"], dtype=np.float64),
                acquired=acq, vmod=float(d["vmod"]) if "vmod" in d else None,
                V_bias=float(d["V_bias"]) if "V_bias" in d else None,
                I_setpoint=float(d["I_setpoint"]) if "I_setpoint" in d else None, sweeps=None,
                g_true=np.asarray(d["g"], dtype=np.float64) if "g" in d else None,
                unit_scale=unit_scale, unit=unit, name=os.path.basename(path))


def sm4_file(path, unit_scale=1e12, unit="pA"):
    """An RHK SM4 spectroscopy file through the package's reader (sm4_Browser.sm4file): the energy axis,
    the pages, the acquired columns, the setpoint and the columns' sweep directions are the browser's."""
    from sm4_Browser import sm4file
    f = sm4file.Sm4File(path)
    sp = f.spectra
    if sp is None:
        raise ValueError(f"{f.name} has no spectra.")
    if sp.iv is None:
        raise ValueError(f"{f.name}: no I(V) current channel; the denoise step needs it.")
    return dict(energy=np.asarray(sp.energy, dtype=np.float64), lia=np.asarray(sp.lia, dtype=np.float64),
                iv=np.asarray(sp.iv, dtype=np.float64), acquired=np.asarray(sp.acquired, dtype=bool),
                vmod=getattr(sp, "vmod", None), V_bias=getattr(sp, "V_bias", None),
                I_setpoint=getattr(sp, "I_setpoint", None), sweeps=sp.sweep_pins(), g_true=None,
                unit_scale=unit_scale, unit=unit, name=f.name)


def load(source):
    """'synthetic' -> the built-in synthetic set (the sweep of seed 7 as column 0); a .npz -> a saved
    synthetic set; anything else -> an SM4 file."""
    if source == "synthetic":
        return synthetic_set()
    if source.lower().endswith(".npz"):
        return npz_set(source)
    return sm4_file(source)


# ----------------------------------------------------------------------------------------------
#  the step, from the engine
# ----------------------------------------------------------------------------------------------
def run_step(d, lock="0lock", soft=True):
    """
    The engine on the file's acquired columns with the denoise row on, the lock chosen ('0lock': the
    0-lock; 'vbias': the Vbias lock; 'none': no lock), the soft edge row as `soft` and the gate at
    its defaults (sts.STS.stages); the engine's console lines printed.  Returns the Stages;
    ValueError with the engine's reason when the step was not applied.
    """
    if lock not in LOCKS:
        raise ValueError(f"lock must be one of {', '.join(LOCKS)}, not {lock!r}")
    gate = calibration.GateSettings(denoise=True, soft_edge=bool(soft), **LOCKS[lock])
    state = PipelineState(channel="didv", gate=gate, calibrate=True)
    s = STS.stages(d["energy"], d["lia"], d["iv"], state, acquired=d["acquired"], vmod=d["vmod"],
                   V_bias=d["V_bias"], I_setpoint=d["I_setpoint"], name=d["name"], sweeps=d["sweeps"])
    for text in s.notes:
        print(text)
    if s.step is None:
        why = next((n for n in s.notes if n.startswith("denoise not applied")), "the step was not applied")
        raise ValueError(f"{d['name']}: {why}")
    return s


# ----------------------------------------------------------------------------------------------
#  the figure
# ----------------------------------------------------------------------------------------------
def residual_panel(ax, res, g, g_raw, ylabel, label, Vs, o):
    """A residual on the left axis and g - 1 (dashed) on a right axis, both axes centred on 0 and
    spanning the largest excursion, so the two can be read against each other; g_raw - 1 (g before
    the soft edge) dotted beside it when given."""
    fin = lambda x: x[np.isfinite(x)]
    half_r = 1.03 * max(float(np.max(np.abs(fin(res)))) if fin(res).size else 0.0, 1e-12)
    ax.plot(Vs, res[o], marker="o", ms=3, color=COL_STEP, lw=1.4, label=label)
    ax.set_ylim(-half_r, half_r)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="0.8", lw=0.8); ax.axvline(0, color="0.8", lw=0.8)
    axg = ax.twinx()
    axg.plot(Vs, (g - 1)[o], ls="--", lw=1.0, color=COL_STEP, label="g − 1 (stages.step.g, dashed)")
    if g_raw is not None:
        axg.plot(Vs, (g_raw - 1)[o], ls=":", lw=0.9, color=COL_STEP, alpha=0.6, label="g − 1 before the soft edge (stages.step.g_raw)")
    half_g = 1.03 * max(float(np.max(np.abs(fin(g) - 1))), 1e-6)         # the axis follows the step's g
    axg.set_ylim(-half_g, half_g)
    axg.set_ylabel("g − 1 (dashed)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axg.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=9)


def make_figure(d, s, column, lock, soft_on=True, save_to=None):
    """The five panels for one column of the engine's result (run_step; `soft_on` whether its soft edge row was on), saved to `save_to` when given."""
    V = np.asarray(d["energy"], dtype=np.float64)
    cols = np.flatnonzero(d["acquired"])
    if column not in cols:
        raise ValueError(f"column {column} of {d['name']} was never acquired (the acquired columns: {cols.tolist()})")
    k = int(np.flatnonzero(cols == column)[0])          # its place among the acquired columns (stages.step.fits)
    step = s.step
    u, unit = float(d["unit_scale"]), d["unit"]
    LIA = s.lia_f[:, column]
    LIA_d = s.lia_d[:, column]
    Isub0 = step.Isub0[:, column] * u
    reb = step.I_rebuilt[:, column] * u                 # NaN where the step rebuilt nothing
    F = np.array(step.Fcal[:, column], dtype=np.float64)
    g = step.g[:, column]
    g_raw = step.g_raw[:, column]
    curve = step.curve
    soft = curve is not None                            # the soft edge measured and applied
    fit = step.fits[k]
    a, b, c = fit.a, fit.b * u, fit.c * u
    solved = np.isfinite(reb)                           # the points the step solved: g read there
    F[~solved] = np.nan
    sgn = np.sign(Isub0)                                # the rebuilt current is a magnitude; shown signed

    o = np.argsort(V)
    Vs = V[o]
    fig, axes = plt.subplots(5, 1, figsize=(11, 21), sharex=True)
    kw_rec = dict(color=COL_REC, lw=1.2)
    kw_step = dict(color=COL_STEP, lw=1.4)

    # 1. the recorded current and the rebuilt current
    ax = axes[0]
    ax.plot(Vs, Isub0[o], marker="o", ms=3, label="recorded I, 0 V value subtracted (stages.step.Isub0)", **kw_rec)
    ax.plot(Vs, (sgn * reb)[o], label="the step's rebuilt current (stages.step.I_rebuilt: Fcal integrated in ln|V| "
                                      "from each edge inward, aligned per side)", **kw_step)
    ax.axhline(0, color="0.8", lw=0.8); ax.axvline(0, color="0.8", lw=0.8)
    ax.set_ylabel(f"current ({unit})")
    rep = "" if not fit.replaced else f"; the gate replaced {'a and b' if fit.replaced == 'ab' else 'a'}"
    if soft:
        wv, cc = curve["w"], curve["centres"] * u
        half = (f"w = 0.5 near |I| = {float(np.interp(0.5, wv, cc)):.2g} {unit}" if wv[0] < 0.5 <= wv[-1]
                else "w " + ("at least" if wv[0] >= 0.5 else "below") + " 0.5 in every bin")
        soft_text = f"   soft edge from the file: jitter {100 * curve['jitter']:.1f} %, {half}"
    else:
        soft_text = "   soft edge " + ("not measured (w = 1)" if soft_on else "off")
    ax.set_title(f"{d['name']}  column {column}   the step's regression: lock = {lock}, a = {a:.3g}, "
                 f"b = {b:+.3g} {unit}/V, c = {c:+.3g} {unit}{rep}{soft_text}", fontsize=9)
    ax.legend(loc="best", fontsize=9)

    # 2. the current residual, recorded minus rebuilt, with g - 1 on a right axis
    residual_panel(axes[1], Isub0 - sgn * reb, g, g_raw if soft else None,
                   f"recorded I − rebuilt I ({unit})", "recorded I − rebuilt I", Vs, o)

    # 3. the LIA page and LIA / g
    ax = axes[2]
    ax.plot(Vs, LIA[o], marker="o", ms=3, label="LIA as the step sees it (stages.lia_f; no filters: as recorded)", **kw_rec)
    ax.plot(Vs, LIA_d[o], label="LIA / g (stages.lia_d)", **kw_step)
    ax.axvline(0, color="0.8", lw=0.8)
    ax.set_ylabel("LIA (as recorded)")
    ax.legend(loc="best", fontsize=9)

    # 4. the LIA residual, LIA minus LIA / g, with g - 1 on a right axis
    residual_panel(axes[3], LIA - LIA_d, g, g_raw if soft else None, "LIA − LIA / g (LIA units)", "LIA − LIA / g", Vs, o)

    # 5. g over the LIA page (left, full height) and the recorded current (right, full height), the ratio
    #    on the LIA's axis; g on its own axis, centred on 1 and filling the full height
    ax = axes[4]
    pad = lambda lo, hi: (lo - 0.03 * (hi - lo), hi + 0.03 * (hi - lo))
    ax.plot(Vs, LIA[o], label="LIA (left axis)", **kw_rec)
    ax.plot(Vs, F[o], color=COL_RATIO, lw=1.2,
            label="the ratio the step integrates, stages.step.Fcal = cal·V / Isub0 (left axis, same scale; the points the step solved)")
    Ff = F[np.isfinite(F)]
    lo = min(float(np.min(LIA)), float(np.min(Ff))) if Ff.size else float(np.min(LIA))
    hi = max(float(np.max(LIA)), float(np.max(Ff))) if Ff.size else float(np.max(LIA))
    ax.set_ylim(*pad(lo, hi))
    ax.set_ylabel("LIA (as recorded)  /  Feenstra ratio")
    ax.set_xlabel("V (V)")
    ax_i = ax.twinx()
    ax_i.plot(Vs, Isub0[o], color="0.55", lw=1.2, label="recorded I, 0 V value subtracted (right axis)")
    ax_i.set_ylim(*pad(float(np.min(Isub0)), float(np.max(Isub0))))
    ax_i.set_ylabel(f"current ({unit})", color="0.4")
    ax_g = ax.twinx()
    ax_g.spines["right"].set_position(("axes", 1.09))
    ax_g.plot(Vs, g[o], marker="o", ms=3, label="g (stages.step.g)" + (", soft edge" if soft else ""), **kw_step)
    if soft:
        ax_g.plot(Vs, g_raw[o], ls=":", lw=0.9, color=COL_STEP, alpha=0.6, label="g before the soft edge (stages.step.g_raw)")
    half_g = 1.03 * max(float(np.max(np.abs(g[np.isfinite(g)] - 1))), 1e-6)   # the axis follows the step's g
    ax_g.set_ylim(1 - half_g, 1 + half_g)               # g = 1 at the vertical centre, full height
    ax_g.axhline(1, color=COL_STEP, lw=0.6, alpha=0.4)
    ax_g.set_ylabel("g (centred on 1)")
    ax.axvline(0, color="0.8", lw=0.8)
    lines = ax.get_legend_handles_labels()[0] + ax_i.get_legend_handles_labels()[0] + ax_g.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax_i.get_legend_handles_labels()[1] + ax_g.get_legend_handles_labels()[1]
    ax.legend(lines, labels, loc="best", fontsize=9)
    step_v = float(np.median(np.diff(Vs)))
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10 * step_v))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(step_v))
    ax.grid(axis="x", which="major", color="0.55", lw=0.6)
    ax.grid(axis="x", which="minor", color="0.85", lw=0.4)

    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=140, bbox_inches="tight")
        print(f"wrote {save_to}")
    print(f"a = {a:.4g}, b = {b:+.4g} {unit}/V, c = {c:+.4g} {unit};  points the step solved {int(solved.sum())} of {V.size} "
          f"({int(step.kept[column])} left as recorded, stages.step.kept)")
    if soft:
        print(f"soft edge, measured on the file (stages.step.curve): |Isub0| bin centre, rms(g_raw - 1), w; jitter {100 * curve['jitter']:.1f} %")
        for cc, rr, ww in zip(curve["centres"] * u, curve["rms"], curve["w"]):
            print(f"   {cc:9.3g} {unit}   {100 * rr:5.1f} %   w = {ww:.2f}")
    print(f"rms(g - 1) over the points the step solved: {100 * np.sqrt(np.mean((g[solved] - 1) ** 2)):.2f} %"
          + (f" (before the soft edge {100 * np.sqrt(np.mean((g_raw[solved] - 1) ** 2)):.2f} %)" if soft else ""))
    if d.get("g_true") is not None:
        gt = np.asarray(d["g_true"], dtype=np.float64)[:, column]
        print(f"rms error vs the g put in: {100 * np.sqrt(np.mean((g[solved] / gt[solved] - 1) ** 2)):.2f} %"
              + (f" (before the soft edge {100 * np.sqrt(np.mean((g_raw[solved] / gt[solved] - 1) ** 2)):.2f} %)" if soft else ""))
    return fig


if __name__ == "__main__":
    # Spyder: set these and press F5.  Command line:
    #   python -m sm4_Browser.tools.rebuild_panels [SOURCE] [SWEEP] [LOCK] [OUT]
    #   SOURCE     'synthetic', a .npz synthetic set, or an .sm4 file
    #   SWEEP      the column index of the sweep (ignored for 'synthetic')
    #   LOCK       '0lock' (the 0-lock), 'vbias' (the Vbias lock) or 'none' (no lock)
    #   OUT        the .png to write (default: beside this file, named after the source)
    #   SOFT_EDGE  the 'soft edge' row (on at open in the windows)
    SOURCE = "synthetic"
    SWEEP = 0
    LOCK = "0lock"
    OUT = None
    SOFT_EDGE = True

    args = sys.argv[1:]
    if len(args) >= 1: SOURCE = args[0]
    if len(args) >= 2: SWEEP = int(args[1])
    if len(args) >= 3: LOCK = args[2]
    if len(args) >= 4: OUT = args[3]
    if OUT is None:
        here = os.path.dirname(os.path.abspath(__file__))
        stem = "synthetic" if SOURCE == "synthetic" else f"{os.path.splitext(os.path.basename(SOURCE))[0]}_sweep{SWEEP}"
        OUT = os.path.join(here, f"rebuild_panels_{stem}.png")
    data = load(SOURCE)
    stages = run_step(data, LOCK, SOFT_EDGE)
    make_figure(data, stages, 0 if SOURCE == "synthetic" else SWEEP, LOCK, soft_on=SOFT_EDGE, save_to=OUT)
