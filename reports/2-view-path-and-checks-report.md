# Stage 2 report: one pipeline path for every window, and checks split from tools

Date: 2026-11-04.  Implements `2-view-path-and-checks/view-path-and-checks.md` of the hand-off
folder (approved by Benjamin Kafin on 2026-11-03).  Baseline: the repository after stage 1
(5b9475e).  Gate: `python -m sm4_Browser.checks.run` exits 0 with every check reported as run
(none skipped), and the grep gate of the document's section 4 holds (below).

## Approved changes, as implemented

### `sts.py`
- ADDED `PipelineState` (NamedTuple: `channel`, `hampel`, `sgolay`, `hampel_didv`, `sgolay_didv`,
  `gate`, `calibrate`, `normalize`, `divide_iv`, `iv_zero_sub`, `iv_broadened`, `iv_broadening`,
  `norm_range`; `of(filters, pipeline, feenstra, channel, divide_iv, norm_range)`, `with_`,
  `filters()`, `iv_width`), `Stages` (the engine's intermediates: `lia_f`, `iv_f`, `g`, `lia_d`,
  `iv_d`, `I_setpoint_used`, `cal`, `fits`, `iv0`, `ivv`, `ratio`, `didv_n`, `ratio_n`, `areas`,
  `notes`), `View` (`didv`, `current`, `ivv`, `ratio`, `shown`, `g`, `fits`, `notes`),
  `state_record`, `state_from_record`, `say`, the helpers `_items` / `_dict`.
- ADDED `STS.stages(energy, lia, iv, state, acquired, vmod, V_bias, I_setpoint, ref, name, sweeps)`:
  the body of the earlier `process_arrays`, run once, every intermediate kept, the console lines
  returned as notes; `STS.view(...)` (the same data arguments): the View from one stages call,
  pure; `STS.stages_of(state)` / `STS.view_of(state)`: the per-file forms.
- MODIFIED `STS.process_arrays`: a wrapper, the same signature and every keyword kept: stages
  with a state built from the arguments, the notes said once (`_note_once`), the quantity
  `divide_iv` selects and its areas returned, the division without a current refused as before.
  `STS.process`, `Sm4File.curves` and `Selection.curves` untouched.
- The numerical change of this stage: none.  Verified below (bitwise).

### `sm4file.py`, `averaging.py` (`Selection`)
- ADDED `Sm4File.stages(state)`, `Sm4File.view(state)` (the pass-through `curves` makes, the "no
  Current channel" line said as `curves` says it), `Selection.stages(state)`, `Selection.view(state)`.

### `line.py` (Line)
- MODIFIED `_pipeline(state, source=None, ref=None, sweeps=None)`: keeps its kept-column, sweeps
  and ref logic, calls `STS.stages` / `STS.view_from`, returns (view, stages) with the
  never-acquired columns NaN; `_int_pipeline(state=None)`: `view(state).shown` oriented.
- ADDED the `state` property (the window's `PipelineState` from its attributes, which the box
  callbacks set), `view(state=None)` and `_int_pair` (the memo `{state: (view, stages)}`, dropped
  in `_int_load_source` when the source changes), `_int_versions_of(state)` (the four versions
  from `view(state)`).
- MODIFIED `feenstra_view(state=None)` (the state instead of calibrate / normalize / gate),
  `_int_apply_feenstra`, `_int_on_filter_change` (the new filters as `state.with_`),
  `_int_on_channel`, `_int_on_sweeps`, `_int_set_line`, the opening curves,
  `marker_record` (the state keys through `state_record`), `restore_markers` (through
  `state_from_record`; the filter box part unchanged), `_int_send_to_fit` (the sends from
  `_int_pair(state.with_(calibrate=True, normalize=True))` and the same with the filters off).
- REMOVED `_int_feenstra_versions`, `_int_feenstra_versions_compute`, `_int_versions_memo`; the
  imports of `iv_corrected`, `iv_broadened`, `normalize_abs`.

### `lineavg.py` (LineAverage)
- ADDED `_member_run(m, state)` (the member's columns through `STS.stages` / `view_from`, never-
  acquired NaN) and `_member_view(m, state)` (memoized in the member dict under `'_views'`);
  `view(state=None)` (the main member's view: the display's and the companion's source before a
  run); `_int_versions_of(state)` (after a run every member's view averaged through the run's
  maps; before a run the Line window's).
- MODIFIED `_int_pipeline(state=None)`, `_align()` (the members' views for the window's state
  with the division off; the current with the normalization off), `run()`, `_int_send_to_fit`,
  `record()` (through `state_record`), `AlignmentEditor.draw_member` (the member's view with the
  division off).
- REMOVED `_member_columns`, `_member_current`, `_member_ivv`, `_int_feenstra_versions`; the
  imports of `iv_corrected`, `iv_broadened`, `normalize_abs`, `calibration`.

### `map.py` (Map)
- ADDED the `state` property (`PipelineState.of` on the filter box, the window's rows, the
  Feenstra settings, the channel, the toggle), `view(state=None)` (the grid's view, memoized by
  the state; dropped in `_on_sweeps`).
- MODIFIED `recompute()` (`_cube_of(view().shown)`), `feenstra_view(state=None)` (the four
  versions from `view(state)` through `_cube_of`).
- REMOVED `_cube`, `_feenstra_cubes`, `_cube_memo`; the imports of `iv_corrected`,
  `iv_broadened`, `normalize_abs`.

### `averaging.py` (MultiAverageWindow)
- ADDED the `state` property.
- MODIFIED `_sweeps_of(point, state=None)` (the entry's `view(state).shown`), `_avg_sets(state=None)`,
  `_unfiltered_average`, `_on_send_to_fit` (the sends through states), `feenstra_view(state=None)`
  (every entry's view), `record()` (the state keys through `state_record`, its own filter keys
  kept in their form); the imports of `iv_corrected`, `iv_broadened`, the local `normalize_abs`.

### `browser.py`
- ADDED the `state` property (the browser's boxes, the current file's channel, the division off),
  `_view_of(f, state)` (a small memo, 8 entries).
- MODIFIED `_draw_spectra` (the four panels from `f.view(state)`; the all-sweeps panel from
  `f.spectra_all.view_of(state)`), `_unfinished_curves(f, state, four)` (`STS.view` on a stopped
  sweep's columns), `load_saved_average` (the record's pipeline state through `state_from_record`
  over the older formats' defaults, as before); the imports of `iv_corrected`, `iv_broadened`,
  `normalize_abs`.

### `feenstra.py` (FeenstraWindow)
- MODIFIED `refresh`: `parent.feenstra_view(parent.state.with_(calibrate=, normalize=, gate=))`;
  the class and method docstrings.

### `checks/` and `tools/`
- ADDED `checks/run.py` (the runner), `checks/pipeline_check.py` (the invariants of section 3.3),
  `checks/window_smoke.py` (section 5.4, on a virtual screen: below).
- RENAMED `check_contour` → `contour_check`, `check_pass_average` → `pass_average_check`,
  `check_affine` → `affine_check`, `check_stripe` → `stripe_check`, `check_sweeps` →
  `sweeps_check`; each `run() -> int` (the failures), one `[ok]` / `[FAIL]` line per sub-check,
  `sys.exit(1 if run() else 0)`; what they check unchanged.
- DELETED `checks/smoke_browser.py`, `checks/smoke_sweeps.py`.
- MOVED to `tools/` (new, `__init__.py` empty), each with its `__main__` block verbatim:
  `denoise_check` → `denoise_stats`, `diagnose_file`, `zero_crossing`, `alignment_windows`,
  `perf_layer`, `check_background` → `background_processes`.
- MOVED `checks/averages-topo-report.md` → `reports/averages-topo-report.md`.
- `pin_rule_check` (stage 1) and `synth.py` kept.

### `README.md`, `__init__.py`
- The pipeline section (the one path, the state, the view), the `sts.py` section, the `checks/`
  and `tools/` sections and the module map say the above.

## Additional changes made (for approval; none bundled silently)

1. `Map.view` (the 'single' / 'grid' mode attribute) RENAMED `Map.view_mode` (15 uses in
   `map.py`).  The stage's `view(state)` method must carry the name `view` on every window, and
   an instance attribute of that name shadowed it (every recompute failed: "'str' object is not
   callable"; `window_smoke` caught it).  The stage-3 record will write it under the key `'view'`
   as the document says.
2. `STS.view_from(stages, energy, state)` ADDED: the pure mapping from a `Stages` to its `View`, so
   that a window gets both from one engine run (the Line window's send-to-fit needs the areas).
   `STS.view` is `view_from(stages(...))`.
3. `state_record(state, values=None)`: the optional `values` = the filter boxes' values
   (`FilterControls.values()`), so the Line and line-average records keep writing the box values
   when a filter is off, as they did; without it the parameters of the toggles that are on.
4. `FeenstraWindow._draw_top` stores the drawn image in `col['im']` (the smoke asserts the
   companion's drawn arrays against `view(state)`).
5. `Line._int_versions_of(state)` / `LineAverage._int_versions_of(state)`: the four versions as a
   hook `feenstra_view(state)` calls (the line average overrides it after a run).
6. The averaging window's record gains the keys `hampel_on`, `sgolay_on`, `hampel_didv_on`,
   `sgolay_didv_on`, `energy_range` (from `state_record`) beside its own filter keys, which keep
   their form (the parameters when on, None when off); older records load as before
   (`state_from_record` reads a filter dict without its flag as on).
7. `window_smoke` draws on a real screen, never on Qt's offscreen platform, per Benjamin's
   instruction of 2026-11-03 (offscreen errs in the browser and freezes the processing): the
   display at hand, else an Xvfb virtual screen the check starts and stops (4200 × 2600); without
   either it reports `not run`.  The old `_offscreen_platform` is therefore not kept.  The two
   moved tools that open a browser (`perf_layer`, `background_processes`) still default to
   `QT_QPA_PLATFORM=offscreen` at their top (moved verbatim); Benjamin decides whether that
   should change too.
8. The moved tools' usage lines name their new module path (`python -m sm4_Browser.tools.<name>`),
   and `background_processes` imports `perf_layer` from `sm4_Browser.tools` (it moved with it).
9. `LineAverage.view(state=None)` override: the main member's view (`_member_view`), so that the
   display and the companion before a run use the member's own acquired mask (the Line window's
   `_pipeline` would use the never-written test on the raw columns).
10. `Browser._view_of` keeps a memo of 8 views (a window memoizes its views; the browser's are per
    file, sweep view and state).
11. `pipeline_check`'s invariant 1 runs 640 setting combinations per window (every combination of
    the rows, the division, the gate, the locks, both channels, with and without a current; the
    filters, the Feenstra settings and the crop in full around one setting of the rows) instead of
    the full factorial of every setting (9,910 per window, 6 minutes); the full factorial was run
    once here with the same result.

## Verification (run here, 2026-11-04)

- `process_arrays` before and after the stage: the pre-stage module (a worktree at 5b9475e)
  imported beside the new one, on three synthetic windows (one without 0 V, one descending with a
  sample at 0 V), 10 columns with a never-acquired column, forward and reverse sweeps, a
  reference: every combination of channel, denoise, calibrate, normalize, divide_iv, the gate,
  each lock (256), the four filters, 0Vsub, broadened, the crop, the reference (128), and the
  no-current cases (32): 1,608 outputs and areas `numpy.array_equal` (NaN-aware), 24 refusals with
  the same message, 163 console lines identical in order.  Zero differences.
- Every window's drawn arrays before and after the stage on the synthetic standard experiment
  ('paired' layout, point pairs), the same box clicks on both (the pre-stage package run once to
  record them, on the virtual screen): the browser's four panels at the grid and the point in four
  states, the Map's cube in four states and its companion's four panels for two flag sets, the
  Line heatmap and curves in four states and its companion's panels, the averaging window's
  entries' sweeps in two states, the line average before and after a run and after a state
  change: 53 arrays, all identical.  The old line-average companion could not be recorded (its
  TypeError, defect 2 of the document); the new one was.
- `python -m sm4_Browser.checks.run`: exit 0; `pipeline_check` ok (18), `pin_rule_check` ok (28),
  `contour_check` ok (22), `pass_average_check` ok (11), `affine_check` ok (2), `stripe_check` ok
  (10), `sweeps_check` ok (58), `window_smoke` ok (22, on an Xvfb screen it started itself, Qt
  platform xcb).  Every check reported as run.
- The grep gate: `_int_feenstra_versions_compute`, `_int_versions_memo`, `_feenstra_cubes`,
  `_cube_memo`, `_member_current`, `_member_ivv` occur nowhere in the package; `iv_corrected(` /
  `iv_broadened(` are called in `sts.py`, `feenstra.py`, `gap.py` (the denoise step, as before)
  and `tools/denoise_stats.py` (a tool, moved verbatim), never in a window module.
- `pyflakes` on every module: clean, except lines that pre-date this stage (unused imports in
  `topo.py`, `browser.py`, `checks/synth.py`, `tools/alignment_windows.py`, an unused local in
  `checks/stripe_check.py`, an f-string in `sm4file.py`), left as they were.
- No `.sm4` file used; `fit.py` untouched and never run.

## Not verified here

- Benjamin's windows on real files (the companions' `calibrate` and `denoise` rows on the 2023
  grid, the line average's companion): the stage removes the three defects by construction; the
  smoke covers the synthetic experiment only.
- The tools (`tools/`): import and `pyflakes` only; they need real files.
- With `denoise` on, the companions' current and I/V panels now show the denoised current I/g (the
  document's defect 3 removed); the before / after comparison above was made with `denoise` off,
  where every array is unchanged.
