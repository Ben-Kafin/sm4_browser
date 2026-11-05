"""
The denoise step's rebuild of the current and its g, one sweep at a time, in a window of the
package's own drawing layer (qtplot, the one the browser's windows use).  Every piece of the step
is read from the engine (sts.STS.stages with the denoise step on; stages.step, a gap.DenoiseStep):
nothing of the step is computed here.  Panel 5's Feenstra ratio is the engine's as well: what the
windows show with their 'Feenstra' toggle on (sts.STS.view, feenstra_shown).

The five panels, top to bottom:
    1. the recorded current, 0 V value subtracted (stages.step.Isub0), and the current the step
       rebuilt from the Feenstra ratio (stages.step.I_rebuilt, aligned per side; drawn signed),
       with the step's regression drawn in the panel: the column, the gate and the locks, the
       sweep's a, b, c (and which of them the gate replaced, when it did), and the soft edge's
       jitter with the current at which w crosses 0.5
    2. the current residual, recorded minus rebuilt (left axis), with g - 1 on a right-hand axis
       (dashed; dotted, g before the soft edge); both axes centred on 0 and filling the full height
    3. the LIA page as the step sees it (stages.lia_f; no filters in this window, so as recorded)
       and LIA / g (stages.lia_d)
    4. the LIA residual, LIA minus LIA / g (left axis), with g - 1 on a right-hand axis (dashed;
       dotted, g before the soft edge), both centred on 0 and filling the full height
    5. the LIA page on the left axis, and on right-hand axes side by side the Feenstra ratio as
       the Line window and every other window show it with their 'Feenstra' toggle on and
       calibrate, normalize, denoise and every filter off, '0Vsub' on and 'broadened' off as they
       open (sts.STS.view(...).shown for that state, feenstra_shown; the rows of this window do
       not reach it), the recorded current, and g, centred on g = 1, g before the soft edge dotted beside it;
       every axis filling the full height; a vertical line at every turning point of g
       (stages.step.g, its strict local maxima and minima among the points the step solved:
       turning_points)
The five panels share one box width (qtplot.Figure.align_column: whatever each panel carries on its
right) and one x range, the sweep's bias range with 3 % on each side, the margin the y ranges have
(span).  The two residuals are drawn over the points the step solved (where I_rebuilt is finite); g is
drawn at every point (1 where nothing was rebuilt: the point at 0 V, a point whose ratio is not
finite, a side of 0 V with fewer than gap.GAP_MIN_POINTS such points).  The first
panel's title is the source's name, shown by qtplot's title rule (qtplot.short_title; the whole
name in the window's info panel).

A right-hand axis: qtplot has no twin axes, so the second quantity of a panel is drawn on the
panel's own axes mapped linearly onto the left axis's range, and a right-hand axis
(Axes.secondary_yaxis with the map and its inverse) labels the quantity's own values (right_axis).
Two linear axes are related by exactly such a map, so the panel reads as a twin axis does, and
panning or zooming keeps the two in step.

The step (gap.py): the step's regression (calibration.calibrate with the gate and the locks as the
window's rows set them; its a and b form the page a·LIA + b whose plain 0-sub Feenstra ratio the
step integrates); then per side of 0 V, from the window's edge inward, on every point whose ratio
is finite (no band around 0 V: the point at 0 V itself, which belongs to neither side, keeps
g = 1), the ratio integrated against ln|V|, the rebuilt current aligned once per side so that the
mean of ln|Isub0| - ln I_rebuilt is 0 there, and g_raw = |Isub0| / I_rebuilt.  The soft edge
(SOFT_EDGE below): each point's correction scaled by w = jitter^2 / (jitter^2 + excess^2), the
curve measured on every acquired column of the file (gap.weight_curve, stages.step.curve),
g = 1 + (g_raw - 1) w; the console gives the curve bin by bin.

The controls, under the panels: '< sweep' / 'sweep >' step through the acquired columns in file
order (the ends are not passed: a console line), the sweep text beside them, and the calibrate
rows of the pipeline box (controls.PipelineBox without its denoise, soft edge, calibrate and
normalize rows, as calibration.py's own window has it): 'gate' with its 'H' and 'SG' boxes,
'0-lock' and 'Vbias lock'.  They set the step's regression, the one regression of the denoise
step, for every sweep.  A change runs the engine again on every acquired column (the gate and the
soft edge's curve are measured across the file) and redraws the sweep shown; a step to another
sweep only redraws.  The denoise step is always on here, the soft edge as SOFT_EDGE sets it, and
no filters run (the step sees the pages as recorded).  The engine also runs the pipeline's own
calibration of the denoised pair (calibrate on, the same gate and locks), which is not drawn; its
gain line is printed with the engine's other notes.  When the step cannot be applied (a
regression it cannot determine, a lock it cannot pin), the panels are left empty and the reason
is drawn in the first one and printed.

Saving: the window opens at its design size DESIGN; the toolbar's Save writes the figure in any of
qtplot's formats, EPS among them, at the window's size on screen, exactly as it shows it (the
vector files through mplexport, the controls left out and the full info panel, the settings of the
rows among it, below the panels); OUT below writes one file at open as well.  EPS has no
transparency: the faint lines (alpha) come out opaque.

Colours: recorded = black, the step = red, the ratio = green.

Data: the synthetic set below (SOURCE = 'synthetic'; the g put in is known; N_SYNTHETIC sweeps of
the model, seed 7 as column 0 and seeds 0 and up beside it, so the soft edge's curve has a set to
be measured on), a synthetic set saved as .npz (SOURCE = path, SWEEP = column; the keys the checks
use: energy, lia, iv, vmod, V_bias, I_setpoint, acquired, g), or an RHK SM4 file through the
package's reader (SOURCE = path, SWEEP = the column's index among the file's columns).  A tool
(sm4_Browser/tools/): it modifies nothing in the package and only imports.

Spyder: set the run block below and press F5; or
python -m sm4_Browser.tools.rebuild_panels [SOURCE] [SWEEP] [OUT].
"""

if __package__ in (None, ''):
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import os
import sys
import numpy as np
from scipy.integrate import cumulative_trapezoid
from sm4_Browser import calibration, controls
from sm4_Browser import qtplot as plt
from sm4_Browser.sts import STS, PipelineState

COL_REC, COL_STEP, COL_RATIO = "black", "tab:red", "tab:green"
N_SYNTHETIC = 32         # the synthetic set's sweeps (seeds 0 .. N_SYNTHETIC - 1), the soft edge's curve measured on them
DESIGN = (11, 21)        # the window's design size (inches): the size it opens with; a saved file has the window's size on screen


# ----------------------------------------------------------------------------------------------
#  data sources: (energy, lia, iv, acquired, vmod, V_bias, I_setpoint, sweeps, name, g_true, the
#  display scale and unit of the current), the arrays (npts, N) as the engine takes them
# ----------------------------------------------------------------------------------------------
def synthetic_sweep(seed=7, v_lo=-0.5, v_hi=0.5, npts=101, v_mod=0.025, gap_jitter=0.07,
                    lockin_offset=0.02, lockin_floor=0.03, current_floor=2e-4, current_offset=0.003,
                    v_set=0.5):
    """
    One synthetic sweep: a sloping metal-like conductance with a bump at -0.3 V,
    D(V) = 1 + 0.3 V + 0.4 exp(-((V + 0.3) / 0.08)^2), and its current J, the integral of D, 0 at
    0 V.  One gap factor per point, g = 1 + gap_jitter x a standard normal, multiplies both pages;
    each page has its own electronic noise: the LIA page is (g D + e) / a_true plus its floor
    (a_true = 1 / v_mod; e = lockin_offset x the mean of D, a lock-in offset g does not multiply;
    the floor lockin_floor x the mean of D / a_true), the current g J plus an offset of
    current_offset x max|J| plus its floor of current_floor x max|J|.  Currents in nA, slopes in
    nA/V; one column, a forward sweep with its setpoint at v_set.  Returns (V, LIA, I, g, V_set,
    I_set), I_set the current at the point nearest v_set.
    """
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
def gate_at_open(soft=True):
    """
    The step's settings the window opens with (a calibration.GateSettings): the denoise step on,
    its soft edge as `soft`, and the gate and the locks as every window's rows open (the gate on at
    its defaults, the 0-lock and the Vbias lock off).  From then on the window's rows set the gate
    and the locks.
    """
    return calibration.GateSettings(denoise=True, soft_edge=bool(soft))


def run_step(d, gate):
    """
    The engine (sts.STS.stages) on the source's acquired columns, the dI/dV channel, no filters,
    the calibration on, with `gate` (a calibration.GateSettings: the gate, its H and SG, the locks,
    denoise on and the soft edge) as the step's regression and the pipeline's calibration both take
    it; the engine's console lines printed.  Returns (the Stages, None) or, when the step was not
    applied, (the Stages, the engine's reason).
    """
    state = PipelineState(channel="didv", gate=gate, calibrate=True)
    s = STS.stages(d["energy"], d["lia"], d["iv"], state, acquired=d["acquired"], vmod=d["vmod"],
                   V_bias=d["V_bias"], I_setpoint=d["I_setpoint"], name=d["name"], sweeps=d["sweeps"])
    for text in s.notes:
        print(text)
    if s.step is None:
        return s, next((n for n in s.notes if n.startswith("denoise not applied")), "denoise not applied")
    return s, None


def feenstra_shown(d):
    """
    The Feenstra ratio of every column as the windows show it with their 'Feenstra' toggle on and
    calibrate, normalize, denoise and every filter off ('0Vsub' on, 'broadened' off, as they open):
    sts.STS.view(...).shown for that state, the engine the Line window draws from.  The rows of
    this window do not reach it.  (npts, N), the original energy order.
    """
    state = PipelineState(channel="didv", divide_iv=True)
    return STS.view(d["energy"], d["lia"], d["iv"], state, acquired=d["acquired"], vmod=d["vmod"],
                    V_bias=d["V_bias"], I_setpoint=d["I_setpoint"], sweeps=d["sweeps"]).shown


def print_curve(s, d):
    """The soft edge's curve, measured on the file (stages.step.curve), on the console bin by bin; nothing when it was not measured."""
    curve = s.step.curve
    if curve is None:
        return
    u, unit = float(d["unit_scale"]), d["unit"]
    print(f"soft edge, measured on the file (stages.step.curve): |Isub0| bin centre, rms(g_raw - 1), w; "
          f"jitter {100 * curve['jitter']:.1f} %")
    for cc, rr, ww in zip(curve["centres"] * u, curve["rms"], curve["w"]):
        print(f"   {cc:9.3g} {unit}   {100 * rr:5.1f} %   w = {ww:.2f}")


# ----------------------------------------------------------------------------------------------
#  the panels
# ----------------------------------------------------------------------------------------------
def right_axis(ax, left, right, label):
    """
    A right-hand axis on `ax` for a second quantity (qtplot has no twin axes): the quantity's range
    `right` = (lo, hi) is mapped linearly onto the left axis's range `left` = (lo, hi), and a
    secondary axis (Axes.secondary_yaxis with the map and its inverse) labels the quantity's own
    values.  Returns the map, which the quantity's values are drawn through.
    """
    (l0, l1), (r0, r1) = left, right
    k = (l1 - l0) / (r1 - r0)
    to_left = lambda r: l0 + (np.asarray(r, dtype=np.float64) - r0) * k
    to_right = lambda y: r0 + (np.asarray(y, dtype=np.float64) - l0) / k
    ax.secondary_yaxis("right", functions=(to_right, to_left)).set_ylabel(label)
    return to_left


def residual_panel(ax, res, g, g_raw, ylabel, label, Vs, o):
    """A residual on the left axis and g - 1 (dashed) on a right-hand axis, both axes centred on 0 and
    spanning the largest excursion, so the two can be read against each other; g_raw - 1 (g before
    the soft edge) dotted beside it when given."""
    fin = lambda x: x[np.isfinite(x)]
    half_r = 1.03 * max(float(np.max(np.abs(fin(res)))) if fin(res).size else 0.0, 1e-12)
    half_g = 1.03 * max(float(np.max(np.abs(fin(g) - 1))), 1e-6)         # the axis follows the step's g
    ax.plot(Vs, res[o], marker="o", ms=3, color=COL_STEP, lw=1.4, label=label)
    ax.set_ylim(-half_r, half_r)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="0.8", lw=0.8); ax.axvline(0, color="0.8", lw=0.8)
    to_g = right_axis(ax, (-half_r, half_r), (-half_g, half_g), "g − 1 (dashed)")
    ax.plot(Vs, to_g((g - 1)[o]), ls="--", lw=1.0, color=COL_STEP, label="g − 1 (stages.step.g, dashed)")
    if g_raw is not None:
        ax.plot(Vs, to_g((g_raw - 1)[o]), ls=":", lw=0.9, color=COL_STEP, alpha=0.6,
                label="g − 1 before the soft edge (stages.step.g_raw)")
    ax.legend(loc="best", fontsize=9)


def turning_points(y, mask):
    """
    The turning points of `y`: the indices of its strict local maxima and minima, each point
    compared with its two neighbours among the points where `mask` is True (in the order given;
    the first and the last of those points have one neighbour only and are not turning points).
    """
    idx = np.flatnonzero(np.asarray(mask, dtype=bool) & np.isfinite(y))
    if idx.size < 3:
        return idx[:0]
    v = np.asarray(y, dtype=np.float64)[idx]
    mid = v[1:-1]
    turn = ((mid > v[:-2]) & (mid > v[2:])) | ((mid < v[:-2]) & (mid < v[2:]))
    return idx[1:-1][turn]


def span(x):
    """An axis range for the finite values of `x`: their min to max widened by 3 % on each side (a flat or empty `x` gets a range of its own, so a right-hand axis's map stays finite)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 1.0
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi == lo:
        lo, hi = lo - 0.5 * (abs(lo) or 1.0), hi + 0.5 * (abs(hi) or 1.0)
    return lo - 0.03 * (hi - lo), hi + 0.03 * (hi - lo)


def regression_text(column, k, n, gate, fit, curve, u, unit):
    """The step's regression on the sweep shown, as panel 1 draws it: the column, the gate and the locks,
    a, b, c (and what the gate replaced), and the soft edge."""
    a, b, c = fit.a, fit.b * u, fit.c * u
    rep = "" if not fit.replaced else f"; the gate replaced {'a and b' if fit.replaced == 'ab' else 'a'}"
    if curve is not None:
        wv, cc = curve["w"], curve["centres"] * u
        half = (f"w = 0.5 near |I| = {float(np.interp(0.5, wv, cc)):.2g} {unit}" if wv[0] < 0.5 <= wv[-1]
                else "w " + ("at least" if wv[0] >= 0.5 else "below") + " 0.5 in every bin")
        soft = f"soft edge from the file: jitter {100 * curve['jitter']:.1f} %, {half}"
    else:
        soft = "soft edge " + ("not measured (w = 1)" if gate.soft_edge else "off")
    return "\n".join([f"column {column} (sweep {k + 1} of {n})",
                      "the step's regression: " + (f"gate on ({gate.text})" if gate.on else "gate off")
                      + f", {gate.locks_text}",
                      f"a = {a:.3g}, b = {b:+.3g} {unit}/V, c = {c:+.3g} {unit}{rep}",
                      soft])


def draw_sweep(fig, d, s, column, k, n, gate, feen, why=None):
    """
    The five panels of one acquired column (`column` the file's column, `k` its place among the n
    acquired columns, `s` the engine's result for `gate`, `feen` feenstra_shown of the source) in
    the figure's automatic grid; returns the axes (the window removes them before the next draw).
    With `why` (the step not applied) the panels are left empty and the reason is drawn in the first.
    """
    V = np.asarray(d["energy"], dtype=np.float64)
    axes = fig.subplots(5, 1, sharex=True)
    fig.align_column(*axes)                             # one box width for the five, whatever each carries on its right
    for ax in axes:
        ax.set_xlim(*span(V))                           # the sweep's bias range, 3 % on each side
    axes[0].set_title(d["name"], fontsize=9)
    if why is not None:
        axes[0].text(0.5, 0.5, f"column {column} (sweep {k + 1} of {n}): {why}", transform=axes[0].transAxes,
                     fontsize=9, ha="center", va="center")
        return list(axes)
    step = s.step
    u, unit = float(d["unit_scale"]), d["unit"]
    LIA = s.lia_f[:, column]
    LIA_d = s.lia_d[:, column]
    Isub0 = step.Isub0[:, column] * u
    reb = step.I_rebuilt[:, column] * u                 # NaN where the step rebuilt nothing
    g = step.g[:, column]
    g_raw = step.g_raw[:, column]
    curve = step.curve
    soft = curve is not None                            # the soft edge measured and applied
    fit = step.fits[k]                                  # stages.step.fits: one per acquired column
    solved = np.isfinite(reb)                           # the points the step solved: the residuals and the turning points of g there
    F_win = np.asarray(feen[:, column], dtype=np.float64)   # the Feenstra ratio as the windows show it (feenstra_shown)
    sgn = np.sign(Isub0)                                # the rebuilt current is a magnitude; shown signed

    o = np.argsort(V)
    Vs = V[o]
    kw_rec = dict(color=COL_REC, lw=1.2)
    kw_step = dict(color=COL_STEP, lw=1.4)

    # 1. the recorded current and the rebuilt current, the step's regression drawn in the panel
    ax = axes[0]
    ax.plot(Vs, Isub0[o], marker="o", ms=3, label="recorded I, 0 V value subtracted (stages.step.Isub0)", **kw_rec)
    ax.plot(Vs, (sgn * reb)[o], label="the step's rebuilt current (stages.step.I_rebuilt: Fcal integrated in ln|V| "
                                      "from each edge inward, aligned per side)", **kw_step)
    ax.axhline(0, color="0.8", lw=0.8); ax.axvline(0, color="0.8", lw=0.8)
    ax.set_ylabel(f"current ({unit})")
    ax.legend(loc="best", fontsize=9)
    ax.text(0.99, 0.03, regression_text(column, k, n, gate, fit, curve, u, unit), transform=ax.transAxes,
            fontsize=8, ha="right", va="bottom", bbox=dict(boxstyle="round", fc="w", ec="0.7"))

    # 2. the current residual, recorded minus rebuilt, with g - 1 on a right-hand axis
    residual_panel(axes[1], Isub0 - sgn * reb, g, g_raw if soft else None,
                   f"recorded I − rebuilt I ({unit})", "recorded I − rebuilt I", Vs, o)

    # 3. the LIA page and LIA / g
    ax = axes[2]
    ax.plot(Vs, LIA[o], marker="o", ms=3, label="LIA as the step sees it (stages.lia_f; no filters: as recorded)", **kw_rec)
    ax.plot(Vs, LIA_d[o], label="LIA / g (stages.lia_d)", **kw_step)
    ax.axvline(0, color="0.8", lw=0.8)
    ax.set_ylabel("LIA (as recorded)")
    ax.legend(loc="best", fontsize=9)

    # 4. the LIA residual, LIA minus LIA / g, with g - 1 on a right-hand axis
    residual_panel(axes[3], LIA - LIA_d, g, g_raw if soft else None, "LIA − LIA / g (LIA units)", "LIA − LIA / g", Vs, o)

    # 5. the LIA page on the left axis; the windows' Feenstra ratio, the recorded current and g on
    #    right-hand axes (g centred on 1), every axis filling the full height; a line at every turning point of g
    ax = axes[4]
    left = span(LIA)
    ax.set_ylim(*left)
    for v in Vs[turning_points(g[o], solved[o])]:        # a vertical line at every local maximum and minimum of g
        ax.axvline(v, color="0.75", lw=0.5, zorder=0)
    ax.plot(Vs, LIA[o], label="LIA (left axis)", **kw_rec)
    ax.set_ylabel("LIA (as recorded)")
    ax.set_xlabel("V (V)")
    to_f = right_axis(ax, left, span(F_win), "Feenstra (no calibration, no normalization)")
    ax.plot(Vs, to_f(F_win[o]), color=COL_RATIO, lw=1.2,
            label="Feenstra as the windows show it: 'Feenstra' on, calibrate and normalize off (sts.STS.view(...).shown; right axis)")
    to_i = right_axis(ax, left, span(Isub0), f"current ({unit})")
    ax.plot(Vs, to_i(Isub0[o]), color="0.55", lw=1.2, label="recorded I, 0 V value subtracted (right axis)")
    half_g = 1.03 * max(float(np.max(np.abs(g[np.isfinite(g)] - 1))), 1e-6)   # the axis follows the step's g
    to_g = right_axis(ax, left, (1 - half_g, 1 + half_g), "g (centred on 1)")
    ax.axhline(float(to_g(1.0)), color=COL_STEP, lw=0.6, alpha=0.4)
    ax.plot(Vs, to_g(g[o]), marker="o", ms=3, label="g (stages.step.g)" + (", soft edge" if soft else ""), **kw_step)
    if soft:
        ax.plot(Vs, to_g(g_raw[o]), ls=":", lw=0.9, color=COL_STEP, alpha=0.6, label="g before the soft edge (stages.step.g_raw)")
    ax.axvline(0, color="0.8", lw=0.8)
    ax.legend(loc="best", fontsize=9)
    return list(axes)


def print_sweep(d, s, column, k):
    """The sweep's numbers on the console: the step's a, b, c, the points it solved, rms(g - 1), and the error against the g put in when the source has it."""
    step = s.step
    u, unit = float(d["unit_scale"]), d["unit"]
    fit = step.fits[k]
    g = step.g[:, column]
    g_raw = step.g_raw[:, column]
    soft = step.curve is not None
    solved = np.isfinite(step.I_rebuilt[:, column])
    print(f"column {column}: a = {fit.a:.4g}, b = {fit.b * u:+.4g} {unit}/V, c = {fit.c * u:+.4g} {unit};  points the step "
          f"solved {int(solved.sum())} of {g.size} ({int(step.kept[column])} left as recorded, stages.step.kept)")
    print(f"rms(g - 1) over the points the step solved: {100 * np.sqrt(np.mean((g[solved] - 1) ** 2)):.2f} %"
          + (f" (before the soft edge {100 * np.sqrt(np.mean((g_raw[solved] - 1) ** 2)):.2f} %)" if soft else ""))
    if d.get("g_true") is not None:
        gt = np.asarray(d["g_true"], dtype=np.float64)[:, column]
        print(f"rms error vs the g put in: {100 * np.sqrt(np.mean((g[solved] / gt[solved] - 1) ** 2)):.2f} %"
              + (f" (before the soft edge {100 * np.sqrt(np.mean((g_raw[solved] / gt[solved] - 1) ** 2)):.2f} %)" if soft else ""))


# ----------------------------------------------------------------------------------------------
#  the window
# ----------------------------------------------------------------------------------------------
def open_window(d, column, gate):
    """
    The window on source `d` (load), opened on `column` (an acquired column of the source) with the
    step's settings `gate` (gate_at_open): the five panels in the figure's automatic grid, and under
    them one row (automatic layout, as calibration.py's window): '< sweep' / 'sweep >', the sweep
    text, and the calibrate rows of the pipeline box.  Returns the figure.
    """
    cols = np.flatnonzero(d["acquired"])
    if column not in cols:
        raise ValueError(f"column {column} of {d['name']} was never acquired (the acquired columns: {cols.tolist()})")
    fig = plt.figure(figsize=DESIGN)
    run = {'gate': gate, 'k': int(np.flatnonzero(cols == column)[0]), 'axes': []}
    feen = feenstra_shown(d)                            # the windows' Feenstra ratio: the rows do not reach it, so once

    def engine():
        """The engine on every acquired column with the window's settings; the soft edge's curve printed."""
        run['s'], run['why'] = run_step(d, run['gate'])
        if run['why'] is None:
            print_curve(run['s'], d)

    def draw():
        """The panels of the sweep shown, rebuilt; the sweep text and the console's sweep lines updated."""
        for ax in run['axes']:
            ax.remove()
        col = int(cols[run['k']])
        run['axes'] = draw_sweep(fig, d, run['s'], col, run['k'], cols.size, run['gate'], feen, run['why'])
        sweep_text.set_text(f"sweep {run['k'] + 1} of {cols.size} (file column {col})")
        if run['why'] is None:
            print_sweep(d, run['s'], col, run['k'])
        fig.canvas.draw_idle()

    def on_step(delta):
        """'< sweep' / 'sweep >': the previous / next acquired column; the ends are not passed (a console line)."""
        new = run['k'] + int(delta)
        if not 0 <= new < cols.size:
            print(f"sweep {new + 1}: {d['name']} has sweeps 1..{cols.size}.")
            return
        run['k'] = new
        draw()

    def on_box(state):
        """The 'gate' row, its 'H' / 'SG' boxes, '0-lock' or 'Vbias lock': the engine run again, the sweep shown redrawn."""
        run['gate'] = state['gate']
        engine()
        draw()

    engine()
    steps = controls.StepButtons(fig, '< sweep', 'sweep >')
    steps.on_change(on_step)
    sweep_text = controls.Label(fig, 'sweep text', fs=9)
    box = controls.PipelineBox(fig, gate=gate, steps=False)       # the calibrate rows only; denoise and the soft edge ride in `gate`
    box.on_change(on_box)
    fig.layout_root().addLayout(controls.row('stretch', steps, sweep_text, 'stretch', box))
    draw()
    return fig


if __name__ == "__main__":
    # Spyder: set these and press F5.  Command line:
    #   python -m sm4_Browser.tools.rebuild_panels [SOURCE] [SWEEP] [OUT]
    #   SOURCE     'synthetic', a .npz synthetic set, or an .sm4 file
    #   SWEEP      the file column the window opens on, an acquired one (ignored for 'synthetic', which opens on column 0)
    #   OUT        a file written at open as well, of the window as it opens on its sweep (.eps, .pdf,
    #              .svg vector, .png the window at 720 dpi, by its extension); None = the window only
    #   SOFT_EDGE  the step's soft edge for the whole session (on at open in the other windows; no row for it here)
    SOURCE = 'C:/path/to/spectra_file_1.sm4'
    #SOURCE = 'C:/path/to/spectra_file_2.sm4'
    SWEEP = 0
    OUT = None
    SOFT_EDGE = True

    args = sys.argv[1:]
    if len(args) >= 1: SOURCE = args[0]
    if len(args) >= 2: SWEEP = int(args[1])
    if len(args) >= 3: OUT = args[2]
    data = load(SOURCE)
    fig = open_window(data, 0 if SOURCE == "synthetic" else SWEEP, gate_at_open(SOFT_EDGE))
    if OUT:
        plt.show(block=False)
        fig.savefig(OUT)
        print(f"wrote {OUT}")
    plt.show()
