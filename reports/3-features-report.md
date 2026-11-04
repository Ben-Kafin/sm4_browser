# Stage 3 report: four features on the stage-2 pipeline path

Date: 2026-11-04.  Implements `3-features/features.md` of the hand-off folder (approved by Benjamin
Kafin on 2026-11-03), in the order A, C, B, D.  Baseline: the repository after stage 2 (f7eced3).
Gate: `python -m sm4_Browser.checks.run` exits 0 with every check reported as run.

## Benjamin's answers to the OPEN points (2026-11-04)

Asked what the two points meant, he answered that he does not want the contours' and markers'
functionality to change, and that a map opened from the saved-records window must show the
contours and markers exactly as they were when `save` was pressed.  Read as:

- OPEN 1: the record stores the level (`z_frac`) and everything the slice is drawn from (the topo
  pass, the processor flags, the drift and stripe toggles), the slice is rebuilt from them at load
  as the document says, and the traces themselves are written too (`contour_segments_nm`, nm, for
  information); the load compares the rebuilt slice with the saved traces and says on the console
  when they differ (the topography's pass, processing or drift state is no longer what it was at
  the save).  Nothing about the contour's behaviour changes.
- OPEN 2: one record per file, `<file stem>_map_markers.json`, the sweep view stored inside and
  restored (the Line window's rule: one record per file and line).

## Benjamin's decision on the step's regression (2026-11-04, option 1)

Asked whether the tool may run the step's regression again to draw it (two runs of
`calibration.calibrate` solving for the same thing), he chose: keep both runs of
`calibration.calibrate` as they are.  The step regresses the filtered pair with the window's gate
and locks (the 2026-11-03 decision), uses that fit only to form Fcal, and now returns its fit, its
ratio and its zero-subtracted current as `Stages` fields so the tool draws them instead of running
the step again; the pipeline's regression on the denoised pair stays and remains the one the
windows show (`stages.fits`); no pipeline number changes.  His basis, from 200 synthetic sweeps:
g does not depend on which fit forms the ratio (1.11 % error with the pre-g a, b, 1.08 % with the
post-g a, b), but the pre-g fit's gain scatters 0.92 to 1.05 of the true value sweep to sweep
(the jitter is in both pages it fits) while the post-g fit gives 0.972 to 0.989, so a single
pre-g regression would put that scatter into every calibrated heatmap (option 2 rejected).  The
step's level stays as it is: the rebuilt current rescaled once per side from the average of
ln|Isub0| − ln(rebuilt) over the side's usable points (g geometric mean 1 per side); not the
calibration's c and not the edge point, both tested and worse.  His names: `cal` (a, b), `Fcal`,
`Isub0`, `I_rebuilt`, `g`; not I0.

## Approved changes, as implemented

### A. The companion's `calibrate` and `denoise` rows act on every panel
Nothing beyond stage 2: `feenstra.FeenstraWindow.refresh` asks every parent for
`parent.state.with_(calibrate=, normalize=, gate=)` (the gate carries the locks and `denoise`), the
four panels are `view(state)`'s `didv`, `current`, `ivv`, `ratio` in every parent (Line, Map,
averaging, line average, the browser's spectra area with its own box), the memo is keyed by the
state.  The three defects of the stage-2 document are gone by construction (the grep gate of that
stage, `pipeline_check` invariant 2, `window_smoke`'s companion assertions).  Nothing further was
seen to do nothing in a window on the synthetic experiment.

### C. The percentile colour clip of the Map
- `plots.py`: ADDED `percentile_range(img, p)` (the p-th and (100 − p)-th percentiles of the finite
  values, `numpy.nanpercentile`; `p = 0` gives `finite_range`; `(0, 1)` without a finite value).
- `map.py`: ADDED the `clip %` box (`controls.Text`, the γ box's width through its spec) beside
  `cmap norm`, `self.clip_percent` (0 at open), `_on_clip` (0 ≤ p < 50; an edit outside it, or not a
  number, reported and the box put back); MODIFIED `_limits(img=None)` (`percentile_range` of the
  whole cube with `cmap norm` off, of the given map with it on), `draw_single` and `draw_grid`
  (every map's limits through `_limits`, the `update_heatmap` path included); the γ on top as before;
  the colour bar spans the clipped limits; the data untouched.
- `widgets.py`: ADDED the `'clip %'` spec, label and short label (the box takes its size and label
  by name, as every control does).

### B. The Map window's `save`
- `records.py`: ADDED `MAP_RECORD_FORMAT = 1`, `MAP_RECORD_RE`, `map_record_path(directory, file_path)`
  (`averages/<file stem>_map_markers.json`), `SavedAveragesWindow._map_entry` (kind `'map'`, the text
  `<file name>, map, <n> markers, <saved time>`), `scan` listing the map records with the line records
  by time; the docstrings.
- `map.py`: ADDED the `save` button in the column with `Send to fit`, `marker_record()`, `_on_save`,
  `restore_markers(rec)`.  The record's keys, every value JSON-plain: `format`, `path`, `name`,
  `scalefactor`, `saved`; `sweeps`; the pipeline state (`sts.state_record` with the filter boxes'
  values); `view`; `e_index`, `energy_V`; `gammas`; `cmap_norm`; `clip_percent`; `direction`;
  `active_procs`; `auto_drift`, `manual_drift`, `stripe`; `z_frac`, `contour_level_m`,
  `contour_segments_nm`; `group_sizes`, `group_centers` ((line, fast) floats); `markers` (pixels),
  `markers_nm`; `average`, `stack`, `substack`.  `restore_markers`: the state through
  `state_from_record` into the boxes (without firing) and the attributes; the view; the energy index
  (refused with a console line outside the axis); the gammas (count checked); `cmap norm`; `clip %`;
  the pass; the processor flags; the drift toggles and the stripe toggle through their own handlers;
  `z_frac`; the groups, centres and markers (clipped to the grid; a mismatch keeps the window's); the
  stack controls; then the topo, the maps (`recompute`), the view, the markers, the contour and the
  companion drawn anew, and the rebuilt slice compared with the saved traces.
- `browser.py`: ADDED `load_map_record(rec)` (the file resolved, refused unless a grid; the browser
  goes to it; a Map opened as `_on_map` opens one, with the record's sweep view, `_record_scale`, the
  record's Feenstra state and channel; `restore_markers`; the window appended to `children`);
  MODIFIED `load_saved_average` (kind `'map'` dispatched).

### D. The denoise steps window (a tool), and the step's pieces in the engine
- `gap.py`: ADDED `DenoiseStep` (NamedTuple: `cal`, the page a · LIA + b of the step's regression
  on the filtered pair with the gate and locks; `fits`, that regression's `ColumnFit` list; `Fcal`,
  the ratio the step integrates; `Isub0`, the filtered current with its value at 0 V subtracted;
  `I_rebuilt`, the current rebuilt from Fcal, NaN where none was rebuilt; `g`; `kept`); MODIFIED
  `gap_factor` (returns `(g, kept, I_rebuilt)`: the rebuilt current kept beside g, the level rule
  unchanged) and `denoise` (returns `(lia / g, iv / g, step, note)`, the step's pieces instead of g
  alone; its regression, ratio and g unchanged).  `GAP_MIN_POINTS`, `setpoint_divided` and the
  band constant untouched (the band around 0 V was removed afterwards, gap-no-band, 2026-11-03).
- `sts.py`: ADDED the `Stages` field `step` (the `DenoiseStep` on the full width, NaN in
  never-acquired columns, g 1 there; None when the step is off or not applied); `stages` keeps it.
  `process_arrays`, `view`, every other field: unchanged (the pipeline check's invariant 1 still
  finds 0 differences over 640 combinations per window; the console lines the same).
- ADDED `tools/denoise_steps.py`: `run_steps(f, state)` (the engine on the acquired columns for
  the state with the denoise row on; the step's pieces read from `stages.step`, nothing of the step
  run again; the two consistency numbers, the Feenstra invariance), `DenoiseSteps(f, column,
  state)` (the window: `< sweep` / `sweep >` with the sweep text, the filter box, the pipeline box
  with all six rows, every change rebuilding the state with `PipelineState.of`, rerunning the
  engine and redrawing; the twelve panels of the document, each title naming the `Stages` field
  or the package call it shows: (d) `stages.step.cal` with `stages.step.fits`, (e)
  `stages.step.Fcal`, (f) `stages.step.Isub0` and `stages.step.I_rebuilt`, (g) `stages.g` with
  `stages.step.kept`, (j) `stages.iv0`; the `denoise` row deciding which state panel (l) draws
  bold, (k) always off against on), the helpers `second_differences`, `band_of`, `premise`; the
  run block with `path`, `column`, `hampel`, `sgolay`, `hampel_didv`, `sgolay_didv`, `calibrate`,
  `normalize`, `gate`, `denoise`.
- `checks/pipeline_check.py`: ADDED invariant 4c (with `denoise` on, `stages.step` carries the
  step's pieces; `cal` is `calibration.calibrate` on the filtered pair, run once in the check as
  the check; `gap.gap_factor` on `Fcal` and `Isub0` gives `stages.g` and `I_rebuilt` again;
  |Isub0| / I_rebuilt is g where rebuilt; never-acquired columns NaN; `denoise` off: `step` None).

### `checks/window_smoke.py`, `README.md`, `__init__.py`
- `window_smoke`: the clip's verification (one pixel far above the rest at one energy: `clip %` 0
  the `finite_range` limits, `clip %` 2 the `nanpercentile([2, 98])` limits in both `cmap norm`
  states, the drawn array unchanged, an edit outside the range refused) and the Map record's
  (saved with the non-default state and markers, reloaded through `Browser.load_map_record`: the
  state, the view, the energy index, the markers, the group sizes and centres, `z_frac` and
  `clip %` round-trip; a hand-edited record with a wrong `e_index` and an out-of-range marker: the
  index refused on the console, the marker clipped to the grid).
- The README (the Map window, the saved records, the files table, `map.py`, `records.py`, the
  tools) and the module map say the above.

## Additional changes made (for approval)

1. The reading of OPEN 1 above: the traces are written and compared at load (a console line when
   they differ), beyond the document's reading (b) ("never read back").
2. D, panel (d): the document names `stages.fits[k]` for the page the step forms, which are the
   fits of the pipeline's regression on the denoised pair; the page the step forms is its own
   regression on the filtered pair.  Resolved by Benjamin's decision above: the engine carries
   the step's `cal` and `fits` (`Stages.step`), and the tool draws those; `stages.fits` stay the
   windows' calibration.  The `DenoiseStep` carries `fits` and `kept` beside his five names (the
   pins and the gate's note of panel (d), the kept-sample count of (g)).
3. D: beside the pipeline's lines, the tool prints one line per run with the Feenstra invariance's
   largest deviation over every acquired column (the document asks for it) and the two consistency
   numbers (also in the readout box).  The first consistency number runs `gap.gap_factor` (the
   integration, not the regression) on the drawn Fcal and Isub0 against `stages.g`, as the
   document asks.
4. `widgets.py`: the `'clip %'` entries (item C above).
5. `Map.restore_markers` updates the filter box's values from the record's dicts even when a
   filter is off, as the Line window's restore does.

## Verification (run here, 2026-11-04)

- `python -m sm4_Browser.checks.run`: exit 0; `pipeline_check` ok (21), `pin_rule_check` ok (28),
  `contour_check` ok (22), `pass_average_check` ok (11), `affine_check` ok (2), `stripe_check` ok
  (10), `sweeps_check` ok (58), `window_smoke` ok (30, on an Xvfb screen it started itself, Qt
  platform xcb).  Every check reported as run.  After the step's pieces moved into the engine,
  invariant 1 (`view.shown` against `process_arrays`, areas and console lines) still finds 0
  differences over 640 combinations per window: no pipeline number changed.
- D, a scratch script (not in the repository), on a synthetic grid (16 × 8 pixels, 41 energies):
  in three states (denoise on with calibrate and normalize off; all on; denoise off with both locks
  and the gate) every panel's arrays come from `stages.step` and the views; `gap.gap_factor` on
  the drawn Fcal and Isub0 against `stages.g`: max |Δ| 0.0; `STS.process_arrays` with the same
  arguments against `view.shown`: max |Δ| 0.0; |Isub0| / I_rebuilt equals `stages.g` where rebuilt
  to 1e-12 relative; the window opened on the virtual screen (12 axes), stepped one sweep, took a
  pipeline row and a filter toggle.  The Feenstra invariance on the synthetic grid: max |ΔF|/|F| outside the
  band 1.8e-5 (the synthetic current is the integral of the dI/dV, so g cancels to the trapezoid's
  error).
- `pyflakes` on every module touched: clean (the lines that pre-date the stages in `drift.py`,
  `fft.py`, `fit.py`, `mplexport.py`, `qtplot.py`, `scanlayer.py`, `section.py`, `sites.py`,
  `stripe.py` and the ones named in the stage-2 report are untouched).
- No `.sm4` file used; `fit.py` untouched and never run.

## Not verified here

- The tool's window in Spyder on a real file (Benjamin's to open); its computation path and the
  window's construction were run here only on the synthetic grid.
- A map record saved and reloaded on a real file, with the drift model of a real directory.
- The two moved tools that open a browser (`perf_layer`, `background_processes`) still default to
  Qt's offscreen platform at their top (the stage-2 report, item 7); unanswered, left as moved.
