"""
The denoise step of one file, one sweep at a time: every intermediate of the step drawn from
sts.STS.stages for the window's state, so that each panel is a field of the engine, not a
re-derivation; nothing of the step is run again here (the step's own pieces come back from
gap.denoise as Stages.step: cal, fits, Fcal, Isub0, I_rebuilt, g, kept).  The step (gap.py): the
per-point gap factor g of every sweep, read from the recorded current against the Feenstra ratio
of the same sweep, divided out of the dI/dV page and the current before the calibration; its
regression on the filtered pair forms its page and ratio only, the calibration the windows show
is the pipeline's regression on the denoised pair (stages.fits).  A tool (sm4_Browser/tools/): it
needs a .sm4 file with a current, modifies nothing in the package and only imports.

Controls (the bottom row): '< sweep' / 'sweep >' with the sweep text; the filter box
(controls.FilterControls); the pipeline box with all six rows (controls.PipelineBox).  Every change
rebuilds the state (sts.PipelineState.of), reruns the engine on the file's acquired columns (the
gate is cross-sweep) and redraws the shown sweep; the console gets the pipeline's own lines
(stages.notes) once per run, and the Feenstra invariance's largest deviation over every acquired
column.  The channel is dI/dV only (the step runs on dI/dV only); no IETS radio.

Panels, 4 x 3, each title naming the Stages field or the package call it shows:
  row 1, inputs and premise: (a) the LIA page of the sweep as recorded (grey) and stages.lia_f
    (bold), every acquired column faint behind; (b) the current as recorded and stages.iv_f, the
    same; (c) the premise: the second differences of ln|lia_f| against those of ln|Isub0| over the
    samples outside the band (the shared gap factor appears identically on both axes), r and the
    slope in the title;
  row 2, the ratio and the reconstruction: (d) stages.step.cal, the page the step forms, a·LIA + b
    of its regression on the filtered pair with the window's gate and locks, the pins drawn when
    a lock is on and the gate's replacement noted (stages.step.fits); a, b, 1/V_mod and
    b/(a·median LIA) in the legend; stages.step.Isub0 on the right axis with the band
    |Isub0| < gap.GAP_BAND x max |Isub0| shaded; (e) stages.step.Fcal = cal·V/Isub0, the ratio the
    step integrates, the band masked; (f) ln|Isub0| (points) and ln I_rebuilt (line),
    stages.step.I_rebuilt, the current gap.gap_factor rebuilt from Fcal, anchored per side; their
    difference is ln g;
  row 3, g and the output: (g) stages.g for the sweep, 1 in the band, rms(g − 1) and the kept
    samples (stages.step.kept) in the title; (h) lia_f and lia_d overlaid, lia_f − lia_d (the
    removed part) in the inset below; (i) iv_f and iv_d, the same;
  row 4, the checks: (j) the after-premise: the second differences of ln(lia_d) against those of
    ln(stages.iv0, the denoised pair's 0-subtracted current), r (the shared component gone); (k)
    the Feenstra invariance: the
    ratio of view(state.with_(divide_iv=True, calibrate=False, normalize=False, the denoise row
    off)) and the same with the row on, overlaid, the difference inset, max |ΔF|/|F| outside the
    band in the title (calibrate and normalize are forced off in this panel whatever the box says,
    the R9 state: sts.py does not enforce it, so the tool does, and the title says so); (l) the
    pipeline output as the box has it, view(state).shown for the sweep, with the denoise row as
    the box has it (bold) and the same state with the row flipped (faint), and a readout box: a,
    b, 1/V_mod, b/(a·median LIA), rms(g − 1), the kept samples, and two consistency numbers:
    gap.gap_factor run again on the drawn Fcal and Isub0 of (e) and (d) against stages.g (max |Δ|,
    expected 0; the integration, not the regression), and STS.process_arrays with the same
    arguments against view.shown (max |Δ|, expected 0).
The 'denoise' row: the step panels (d) to (j) always show the step for the sweep (the step is the
window's subject); the row decides which state panel (l) draws bold; (k) always compares off
against on.

Spyder: set the run block below and press F5; or python -m sm4_Browser.tools.denoise_steps.
"""

if __package__ in (None, ''):
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import contextlib
import io
import numpy as np
from sm4_Browser import qtplot as plt
from sm4_Browser import calibration, controls, gap, sm4file
from sm4_Browser.sts import STS, PipelineState


def second_differences(y, order):
    """The second differences of ln|y| along the ascending energy axis (`order` sorts it), (npts - 2,); -inf / NaN where y is 0."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.diff(np.log(np.abs(np.asarray(y, dtype=np.float64)[order])), n=2)


def band_of(isub0_col):
    """The samples of one column in the band where the ratio is not conditioned, |Isub0| < gap.GAP_BAND x max |Isub0| (as gap.gap_factor has it), as a mask in the column's own order."""
    i0 = np.asarray(isub0_col, dtype=np.float64)
    top = np.max(np.abs(i0)) if i0.size else 0.0
    return ~((np.abs(i0) >= gap.GAP_BAND * top) & (top > 0))


def premise(x_col, y_col, order, band):
    """The premise's points: the second differences of ln|y| against those of ln|x| over the samples outside the band (all three samples of a difference), with r and the slope (NaN with fewer than three points)."""
    bs = np.asarray(band, dtype=bool)[order]
    ok = ~bs[:-2] & ~bs[1:-1] & ~bs[2:]
    x = second_differences(x_col, order)
    y = second_differences(y_col, order)
    ok &= np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 or np.ptp(x) == 0:
        return x, y, np.nan, np.nan
    r = float(np.corrcoef(x, y)[0, 1])
    slope = float(np.polyfit(x, y, 1)[0])
    return x, y, r, slope


def run_steps(f, state):
    """
    The engine on the file's acquired columns for a state (the gate is cross-sweep, so every
    column runs): a dict with 'state'; 'on', the state with the denoise row on (the step's);
    'cols', the acquired columns; 'stages', STS.stages for 'on' (Sm4File.stages) and 'notes', its
    console lines; 'step', stages.step (gap.DenoiseStep: cal, fits, Fcal, Isub0, I_rebuilt, g,
    kept; None, with the reason in 'refused', when the step was not applied); 'consistency_g',
    gap.gap_factor run again on step.Fcal and step.Isub0 against stages.g (max |Δ|, expected 0;
    the integration only); 'v_off' and 'v_on', the Feenstra invariance's two views (the R9 state:
    the division on, calibrate and normalize off; the denoise row off and on); 'invariance', per
    column max |ΔF|/|F| outside the band, and 'invariance_max' over every column; 'v_box',
    view(state), the output as the box has it, and 'v_other', the same with the denoise row
    flipped; 'consistency_shown', max |STS.process_arrays(the same arguments) − v_box.shown|
    (expected 0).  Nothing of the step is run again here.
    """
    V = np.asarray(f.energy, dtype=np.float64)
    sp = f.spectra
    cols = np.flatnonzero(f.acquired)
    on = state.with_(gate=state.gate._replace(denoise=True))
    s = f.stages(on)
    step = s.step
    refused = None
    if step is None:
        refused = next((n for n in s.notes if n.startswith('denoise not applied')), "the step was not applied")
    consistency_g = None
    if step is not None:
        g_re, _, _ = gap.gap_factor(V, step.Fcal[:, cols], step.Isub0[:, cols])
        consistency_g = float(np.max(np.abs(g_re - s.g[:, cols])))
    base = state.with_(divide_iv=True, calibrate=False, normalize=False)
    v_off = f.view(base.with_(gate=base.gate._replace(denoise=False)))
    v_on = f.view(base.with_(gate=base.gate._replace(denoise=True)))
    invariance = np.full(cols.size, np.nan)
    if step is not None:
        for k, c in enumerate(cols):
            outside = ~band_of(step.Isub0[:, c])
            r_off, r_on = v_off.ratio[:, c], v_on.ratio[:, c]
            ok = outside & np.isfinite(r_off) & np.isfinite(r_on)
            if ok.any() and np.max(np.abs(r_off[ok])) > 0:
                invariance[k] = float(np.max(np.abs(r_on[ok] - r_off[ok])) / np.max(np.abs(r_off[ok])))
    v_box = f.view(state)
    v_other = f.view(state.with_(gate=state.gate._replace(denoise=not state.gate.denoise)))
    hp, sg, hd, sd = state.filters()
    ref = (getattr(sp, 'lia_ref', None), getattr(sp, 'iv_ref', None))
    with contextlib.redirect_stdout(io.StringIO()):          # the earlier interface says its notes; the window says the engine's
        pa = STS.process_arrays(V, f.lia, f.iv, hp, sg, state.divide_iv, state.iv_width, channel='didv',
                                acquired=f.acquired, iv_zero_sub=state.iv_zero_sub, normalize=state.normalize,
                                calibrate=state.calibrate, vmod=sp.vmod, norm_range=state.norm_range,
                                V_bias=sp.V_bias, I_setpoint=sp.I_setpoint, ref=ref, hampel_didv=hd, sgolay_didv=sd,
                                name=f.name, gate=state.gate, sweeps=sp.sweep_pins())
    consistency_shown = float(np.nanmax(np.abs(pa - v_box.shown)))
    return {'state': state, 'on': on, 'cols': cols, 'stages': s, 'notes': list(s.notes), 'step': step,
            'refused': refused, 'consistency_g': consistency_g, 'v_off': v_off, 'v_on': v_on, 'invariance': invariance,
            'invariance_max': float(np.nanmax(invariance)) if np.isfinite(invariance).any() else np.nan,
            'v_box': v_box, 'v_other': v_other, 'consistency_shown': consistency_shown}


class DenoiseSteps:
    """The window (the module docstring): DenoiseSteps(f, column, state), f the Sm4File (the dI/dV channel with a current), column the starting sweep among the acquired columns, state the opening sts.PipelineState (its channel is dI/dV whatever it says)."""

    DESIGN = (18, 11)      # the window's design size (inches)

    def __init__(self, f, column=0, state=None):
        if f.spectra is None or 'didv' not in f.channels():
            raise ValueError(f"{f.name}: no dI/dV channel; the denoise step runs on dI/dV only.")
        if f.iv is None:
            raise ValueError(f"{f.name}: no 'Current' page; the denoise step needs it.")
        self.f = f
        self.cols = np.flatnonzero(f.acquired)
        if self.cols.size == 0:
            raise ValueError(f"{f.name}: no acquired spectra.")
        if not 0 <= int(column) < self.cols.size:
            raise ValueError(f"column must be 0..{self.cols.size - 1} (the acquired columns); got {column}.")
        self.k = int(column)
        state = PipelineState(channel='didv', gate=calibration.GateSettings(denoise=True)) if state is None else state
        self.state = state._replace(channel='didv')
        self.V = np.asarray(f.energy, dtype=np.float64)
        self.order = np.argsort(self.V)
        self.uL = f.units_lia or 'no units'
        self.uI = f.units_iv or 'no units'
        self.R = None
        self.fig = plt.figure(figsize=self.DESIGN)
        self.axes = []
        # automatic layout: the one rule for every window (nothing placed by hand): the twelve panels
        # in the figure's grid (fig.subplots, rebuilt for every sweep), the controls in a row under them
        # (fig.layout_root), as calibration.py's window has them.
        hp, sg, hd, sd = self.state.filters()
        self.filt = controls.FilterControls(self.fig, hampel=hp if hp is not None else hd,
                                            sgolay=sg if sg is not None else sd,
                                            hampel_on=hp is not None, sgolay_on=sg is not None,
                                            hampel_didv_on=hd is not None, sgolay_didv_on=sd is not None)
        self.filt.on_change(self._on_change)
        self.box = controls.PipelineBox(self.fig, calibrate=self.state.calibrate, normalize=self.state.normalize,
                                        gate=self.state.gate)
        self.box.on_change(lambda st: self._on_change())
        self.steps = controls.StepButtons(self.fig, '< sweep', 'sweep >')
        self.steps.on_change(self._on_step)
        self.sweep_text = controls.Label(self.fig, 'sweep text', fs=9)
        self.fig.layout_root().addLayout(controls.row('stretch', self.steps, self.sweep_text, 'stretch', self.filt,
                                                      self.box))
        self.compute()
        self.draw()
        plt.show()

    # --- the state and the run ---

    def _state_of_boxes(self):
        """The state the boxes say (PipelineState.of); the Feenstra settings, the division and the crop are the opening state's (no control here)."""
        s = self.state
        return PipelineState.of(self.filt, self.box.state(),
                                {'zero_sub': s.iv_zero_sub, 'broadened': s.iv_broadened, 'broadening': s.iv_broadening},
                                'didv', s.divide_iv, s.norm_range)

    def _on_change(self, *args):
        """A filter or pipeline row changed: the state rebuilt, the engine rerun, the sweep redrawn; a state the engine refuses is said and the boxes put back."""
        new = self._state_of_boxes()
        if new == self.state:
            return
        old = self.state
        self.state = new
        try:
            self.compute()
        except Exception as exc:
            print(f"denoise steps: the settings were not applied ({exc}); the previous ones stay.")
            self.state = old
            self.box.set_state(old.calibrate, old.normalize, old.gate)
            hp, sg, hd, sd = old.filters()
            self.filt.set_params(hp, sg, hd, sd)
            return
        self.draw()

    def _on_step(self, d):
        """'< sweep' / 'sweep >': the previous / next acquired sweep; the ends are not passed (a console line)."""
        new = self.k + int(d)
        if new < 0 or new >= self.cols.size:
            print(f"sweep {new + 1}: the file has sweeps 1..{self.cols.size}.")
            return
        self.k = new
        self.draw()

    def compute(self):
        """The engine on every acquired column for the state (run_steps); its console lines and the invariance's largest deviation said."""
        self.R = run_steps(self.f, self.state)
        for text in self.R['notes']:
            print(text)
        print(f"{self.f.name}: Feenstra invariance (the division on, calibrate and normalize off): max |ΔF|/|F| outside "
              f"the band over every acquired column {self.R['invariance_max']:.3g}; gap_factor recomputed against stages.g "
              f"max |Δ| {self.R['consistency_g']}; process_arrays against view.shown max |Δ| {self.R['consistency_shown']:.3g}.")

    # --- the panels ---

    def _band_spans(self, band):
        """The contiguous runs of band samples on the ascending axis as (V_lo, V_hi) pairs, half a step beyond their ends."""
        Vs = self.V[self.order]
        b = np.asarray(band, dtype=bool)[self.order]
        step = float(np.median(np.abs(np.diff(Vs)))) if Vs.size > 1 else 0.0
        spans, start = [], None
        for i, flag in enumerate(list(b) + [False]):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                spans.append((Vs[start] - step / 2, Vs[i - 1] + step / 2))
                start = None
        return spans

    def draw(self):
        """The twelve panels rebuilt for the shown sweep (the module docstring), the title and the sweep text updated."""
        for ax in self.axes:
            try:
                ax.remove()
            except Exception:
                pass
        self.axes = []
        R, s, k, c = self.R, self.R['stages'], self.k, int(self.R['cols'][self.k])
        f, V, order = self.f, self.V, self.order
        cols = R['cols']
        faint = dict(color='0.6', lw=0.5, alpha=0.5)
        bold = dict(color='C0', lw=1.2)
        uL, uI = self.uL, self.uI
        axes = self.fig.subplots(4, 3)
        self.axes.extend(axes.flat)
        state = R['state']
        self.fig.suptitle(f"{f.name}: the denoise step of column {c} (sweep {k + 1} of {cols.size}); "
                          f"{controls.PipelineBox.text({'calibrate': state.calibrate, 'normalize': state.normalize, 'gate': state.gate})}",
                          fontsize=10)
        lia_f, iv_f = s.lia_f[:, cols], s.iv_f[:, cols]
        step = R['step']
        isub0 = None if step is None else step.Isub0[:, c]
        band = np.zeros(V.size, dtype=bool) if isub0 is None else band_of(isub0)
        spans = self._band_spans(band)

        # Row 1: the inputs and the premise.
        ax = axes[0, 0]
        ax.plot(V, f.lia[:, cols], **faint)
        ax.plot(V, f.lia[:, c], color='0.35', lw=0.8, label="recorded")
        ax.plot(V, lia_f[:, k], **bold, label="stages.lia_f")
        ax.set_ylabel(f"LIA ({uL})")
        ax.set_title("(a) the LIA page as recorded and stages.lia_f (the channel's filters)", fontsize=8)
        ax.legend(fontsize=7)

        ax = axes[0, 1]
        ax.plot(V, f.iv[:, cols], **faint)
        ax.plot(V, f.iv[:, c], color='0.35', lw=0.8, label="recorded")
        ax.plot(V, iv_f[:, k], **bold, label="stages.iv_f")
        ax.set_ylabel(f"I ({uI})")
        ax.set_title("(b) the current as recorded and stages.iv_f (the I(V) filters)", fontsize=8)
        ax.legend(fontsize=7)

        ax = axes[0, 2]
        r = slope = np.nan
        if isub0 is None:
            ax.text(0.5, 0.5, f"no premise: {R['refused']}", ha='center', va='center', transform=ax.transAxes, fontsize=8, wrap=True)
        else:
            x, y, r, slope = premise(isub0, lia_f[:, k], order, band)
            ax.plot(x, y, '.', ms=4, color='C0')
            if np.isfinite(slope):
                xx = np.array([x.min(), x.max()])
                ax.plot(xx, slope * (xx - x.mean()) + y.mean(), color='C3', lw=0.8)
        ax.set_xlabel("Δ² ln|stages.step.Isub0|")
        ax.set_ylabel("Δ² ln|stages.lia_f|")
        ax.set_title(f"(c) the premise, outside the band: r = {r:.3f}, slope = {slope:.3f} (the shared g on both axes)", fontsize=8)

        # Row 2: the ratio and the reconstruction.
        ax = axes[1, 0]
        cf = None if step is None else step.fits[k]
        if step is None:
            ax.text(0.5, 0.5, f"the step was not applied: {R['refused']}", ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, wrap=True)
        else:
            med = float(np.median(lia_f[:, k]))
            ratio_b = cf.b / (cf.a * med) if cf.a * med != 0 else np.nan
            ax.plot(V, step.cal[:, c], **bold,
                    label=f"a·LIA + b: a = {cf.a:.4g}, b = {sm4file.si_text(cf.b, uI + '/V')}"
                          + (f", 1/V_mod = {cf.a_mod:.4g}" if np.isfinite(cf.a_mod) else "")
                          + f", b/(a·median LIA) = {ratio_b:.3g}")
            for name, v_pin, i_pin in cf.pins:
                ax.axvline(v_pin, color='C1', lw=0.8, ls='--', label=f"{'Vbias' if name == 'Vbias' else '0'} pin at {v_pin:+.3f} V")
            # stages.step.Isub0 on the right axis: the current mapped onto the page's range, the right ticks in its units.
            c_lo, c_hi = float(np.nanmin(step.cal[:, c])), float(np.nanmax(step.cal[:, c]))
            i_lo, i_hi = float(np.nanmin(isub0)), float(np.nanmax(isub0))
            if c_hi > c_lo and i_hi > i_lo:
                k_m = (c_hi - c_lo) / (i_hi - i_lo)
                ax.plot(V, (isub0 - i_lo) * k_m + c_lo, color='C2', lw=0.8, ls=':', label="stages.step.Isub0 (right axis)")
                ax.secondary_yaxis('right', functions=(lambda yv: (yv - c_lo) / k_m + i_lo,
                                                       lambda iv_: (iv_ - i_lo) * k_m + c_lo)).set_ylabel(f"Isub0 ({uI})")
            for lo, hi in spans:
                ax.axvspan(lo, hi, color='0.9', zorder=0)
            ax.legend(fontsize=6.5, loc='best')
        ax.set_ylabel(f"a·LIA + b ({uI}/V)")
        rep = "" if cf is None or not cf.replaced else f"; the gate replaced {'a and b' if cf.replaced == 'ab' else 'a'}"
        ax.set_title("(d) stages.step.cal: the page the step forms, its regression on (lia_f, iv_f) with the gate and locks"
                     + rep + "; the band shaded", fontsize=8)

        ax = axes[1, 1]
        if step is None:
            ax.text(0.5, 0.5, "no ratio: the step was not applied", ha='center', va='center', transform=ax.transAxes, fontsize=8)
        else:
            Fm = np.array(step.Fcal[:, c], dtype=np.float64)
            Fm[band] = np.nan
            ax.plot(V, Fm, **bold)
            for lo, hi in spans:
                ax.axvspan(lo, hi, color='0.9', zorder=0)
        ax.set_ylabel("Fcal")
        ax.set_title("(e) stages.step.Fcal = cal·V/Isub0: the ratio the step integrates (the plain 0-sub I/V), the band masked", fontsize=8)

        ax = axes[1, 2]
        if step is None:
            ax.text(0.5, 0.5, "no rebuilt current: the step was not applied", ha='center', va='center', transform=ax.transAxes, fontsize=8)
        else:
            rebuilt = step.I_rebuilt[:, c]
            usable = np.isfinite(rebuilt)
            with np.errstate(divide='ignore', invalid='ignore'):
                ln_i0 = np.log(np.abs(isub0))
                ln_rebuilt = np.log(rebuilt)
            ax.plot(V[usable], ln_i0[usable], '.', ms=4, color='0.3', label="ln|Isub0|")
            ax.plot(V, ln_rebuilt, color='C3', lw=1.0, label="ln I_rebuilt: Fcal/V integrated from each edge inward, anchored per side")
            ax.legend(fontsize=6.5)
        ax.set_ylabel("ln I")
        ax.set_title("(f) ln|stages.step.Isub0| and ln stages.step.I_rebuilt: their difference is ln g", fontsize=8)

        # Row 3: g and the output.
        ax = axes[2, 0]
        g = s.g
        if g is None:
            ax.text(0.5, 0.5, "stages.g is None: the step was not applied (the console line says why)", ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, wrap=True)
            ax.set_title("(g) stages.g", fontsize=8)
        else:
            gk = g[:, c]
            ax.plot(V, gk, **bold)
            ax.axhline(1.0, color='k', lw=0.5)
            for lo, hi in spans:
                ax.axvspan(lo, hi, color='0.9', zorder=0)
            rms = float(np.sqrt(np.mean((gk - 1.0) ** 2)))
            kept = int(step.kept[c])
            ax.set_title(f"(g) stages.g for the sweep (1 in the band): rms(g − 1) = {rms * 100:.2f} %, {kept} of {gk.size} samples kept as recorded (stages.step.kept)", fontsize=8)
        ax.set_ylabel("g")

        for ax, name, a_f, a_d, unit in ((axes[2, 1], 'lia', lia_f[:, k], s.lia_d[:, c], uL),
                                         (axes[2, 2], 'iv', iv_f[:, k], s.iv_d[:, c], uI)):
            ax.plot(V, a_f, color='0.4', lw=0.8, label=f"stages.{name}_f")
            ax.plot(V, a_d, **bold, label=f"stages.{name}_d = {name}_f / g")
            ax.set_ylabel(f"{'LIA' if name == 'lia' else 'I'} ({unit})")
            ax.legend(fontsize=7, loc='upper left')
            ins = ax.inset_axes([0.55, 0.06, 0.43, 0.3])
            ins.plot(V, a_f - a_d, color='C3', lw=0.6)
            ins.axhline(0, color='k', lw=0.4)
            ins.set_title(f"{name}_f − {name}_d (the removed part)", fontsize=6)
            ins.tick_params(labelsize=5)
            ax.set_title(f"({'h' if name == 'lia' else 'i'}) stages.{name}_f and stages.{name}_d overlaid; the removed part below", fontsize=8)

        # Row 4: the checks.
        ax = axes[3, 0]
        x, y, r2, slope2 = premise(s.iv0[:, c], s.lia_d[:, c], order, band)
        ax.plot(x, y, '.', ms=4, color='C0')
        ax.set_xlabel("Δ² ln|stages.iv0| (the denoised pair's current, 0 V subtracted)")
        ax.set_ylabel("Δ² ln|stages.lia_d|")
        ax.set_title(f"(j) the after-premise: r = {r2:.3f} (was {r:.3f}): the shared component gone", fontsize=8)

        ax = axes[3, 1]
        r_off, r_on = R['v_off'].ratio[:, c], R['v_on'].ratio[:, c]
        ax.plot(V, r_off, color='0.4', lw=0.8, label="view(...).ratio, denoise off")
        ax.plot(V, r_on, color='C0', lw=1.0, ls='--', label="the same, denoise on")
        ax.legend(fontsize=7, loc='upper left')
        ins = ax.inset_axes([0.55, 0.06, 0.43, 0.3])
        ins.plot(V, r_on - r_off, color='C3', lw=0.6)
        ins.axhline(0, color='k', lw=0.4)
        ins.set_title("on − off", fontsize=6)
        ins.tick_params(labelsize=5)
        ax.set_ylabel("(dI/dV)/(I/V)")
        ax.set_title(f"(k) the Feenstra invariance (the division on, calibrate and normalize forced off): "
                     f"max |ΔF|/|F| outside the band {R['invariance'][k]:.3g}", fontsize=8)

        ax = axes[3, 2]
        on_now = state.gate.denoise
        ax.plot(V, R['v_other'].shown[:, c], color='0.6', lw=0.7, alpha=0.7, label=f"denoise {'off' if on_now else 'on'}")
        ax.plot(V, R['v_box'].shown[:, c], **bold, label=f"view(state).shown, denoise {'on' if on_now else 'off'} (the box)")
        ax.legend(fontsize=7, loc='upper left')
        readout = []
        if cf is not None:
            med = float(np.median(lia_f[:, k]))
            readout += [f"step: a = {cf.a:.4g}, b = {sm4file.si_text(cf.b, uI + '/V')}",
                        (f"1/V_mod = {cf.a_mod:.4g}" if np.isfinite(cf.a_mod) else "no V_mod recorded"),
                        f"b/(a·median LIA) = {cf.b / (cf.a * med) if cf.a * med != 0 else np.nan:.3g}"]
        if g is not None:
            gk = g[:, c]
            readout += [f"rms(g − 1) = {float(np.sqrt(np.mean((gk - 1.0) ** 2))) * 100:.2f} %, {int(step.kept[c])} samples kept"]
        readout += [f"gap_factor on (e), (d) vs stages.g: max |Δ| = {R['consistency_g'] if R['consistency_g'] is not None else 'n/a'}",
                    f"process_arrays vs view.shown: max |Δ| = {R['consistency_shown']:.3g}"]
        ax.text(0.02, 0.03, "\n".join(readout), transform=ax.transAxes, fontsize=7, va='bottom',
                bbox=dict(boxstyle='round', fc='w', alpha=0.8))
        ax.set_ylabel("the window's quantity")
        ax.set_title("(l) the pipeline output as the box has it: view(state).shown, and the same with the denoise row flipped", fontsize=8)
        for ax in axes.flat:
            if ax not in (axes[0, 2], axes[3, 0]):
                ax.set_xlabel("Energy (eV)")
        self.sweep_text.set_text(f"sweep {k + 1} of {cols.size} (file column {c})")
        self.fig.canvas.draw_idle()


if __name__ == '__main__':
    path = r'experiment_dir/file.sm4'
    column = 0               # the starting sweep: an index among the acquired columns (0 = the first)
    hampel = None            # the I(V) filters as the windows set them (None = off), e.g. {'window': 4, 'n_sigma': 3}
    sgolay = None            # e.g. {'window_length': 6, 'polyorder': 3}
    hampel_didv = None       # the channel's own filters (None = off)
    sgolay_didv = None
    calibrate = False        # the 'calibrate' row at open
    normalize = False        # the 'normalize' row at open
    gate = None              # the gate on at the defaults, the locks off; False the gate off; or
    #                          {'on': True, 'hampel': (3, 3), 'sgolay': (9, 2), 'lock0': False, 'lockv': False}
    denoise = True           # the 'denoise' row at open

    f = sm4file.Sm4File(path)
    settings = calibration.gate_settings(gate)._replace(denoise=bool(denoise))
    state = PipelineState(channel='didv', hampel=PipelineState.items(hampel), sgolay=PipelineState.items(sgolay),
                          hampel_didv=PipelineState.items(hampel_didv), sgolay_didv=PipelineState.items(sgolay_didv),
                          gate=settings, calibrate=bool(calibrate), normalize=bool(normalize))
    window = DenoiseSteps(f, column, state)
