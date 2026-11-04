# Report: averages from lines and maps, Δz sub, topo range matching, topo rows

Date: 2026-10-07.  Implements the project doc `claude/averages-topo-plan.md` (approved in full,
including its section 14 defaults: least-squares lines through all maxima and all minima for `Δz sub`,
both processors may be off, the Map keeps 'Send to decomp', `Averages` lives in `sts.py`).
Files written: `topo.py`, `sm4file.py`, `sts.py`, `line.py`, `map.py`, `browser.py`.
Untouched: `fit.py`, `drift.py`, `phase.py`, `fft.py`, `__init__.py`.  No fit was run at any point.

## Approved changes (the plan's sections 3 to 12), as implemented

### topo.py
- ADDED `local_extrema(y)`, `delta_z_subtract_profile(topo_x, profile)`, `delta_z_subtract_image(img)`,
  `toggle_processor(active, index)`, `match_ranges(arrays)`, `processed_topo(sm4, direction, active)`.
- MODIFIED `TOPO_PROCESSORS`: `[("Slope Sub", slope_subtract_image, False), ("Δz sub", delta_z_subtract_image, True)]`.
- MODIFIED `TopoWindow.__init__`: the image comes from `processed_topo` (the 'average' direction is the
  range-matched mean of the two processed passes).
- MODIFIED `__main__`: `active_procs` `[True]` -> `[False, True]` (one flag per processor).

### sm4file.py
- ADDED `Sm4File.key` (= path) and `Sm4File.entry_label` (the time of day).
- MODIFIED `Sm4File.topo_raw`: 'forward' or 'reverse' only; the 'average' branch is gone (ValueError).

### sts.py
- ADDED class `Averages` (the averaging math `average`, `mean_rows`; the registry `open`,
  `saved_indices_of`, `saved_indices`, `next_index`, `by_name`, `names`, `new`, `add`, `holding`).
- MODIFIED `STS.average`, `STS.mean_rows`: one-line delegations to `Averages` (bodies moved verbatim).
- ADDED `STS.subset(cols)`.
- ADDED class `Selection` (pass-throughs, `curves`, `where_text`, `entry_label`, `record_entry`, `key`).
- MODIFIED `MultiAverageWindow`: `__init__(..., name=None)` and the name box (`NAME_BOX`); `PANEL_H` 0.70 -> 0.66;
  `RECORD_FORMAT` 2 -> 3; ADDED `stem_of`, `_set_name`, `_on_rename`; `record_path` and `cache_path_of` keyed
  by stem; `cache_path` by the window's stem; `record` (name, per-entry kind, cols, where); `save` (by stem,
  remembers the saved stem); `has` (by key); `add` (label from `entry_label`); `draw` (the empty-window title).
- MODIFIED `SavedAveragesWindow`: `RECORD_RE` `^(.+)_average\.json$`; `_record_entry(name, stem)` reads the
  index and the name from the record and names the fit history by stem; `scan` passes the stem; the header text.

### line.py
- ADDED class constants `TOPO_PROCESSORS`, `_TOPO_MODES = ('forward', 'reverse')`, the `TOPO_ROW_*` layout
  constants, `TOPO_ROWS_PER_COLUMN = 4`, `TOPO_ROWS_IN_FIGURE = 8`, `TOPO_ROW_CLEARANCE`, `TOPO_PASS_LETTER`,
  `ADD_X`, `ADD_W`, `AVG_BOX`.
- ADDED `_topo_profiles`, `_topo_stripe_profile` (per-row processing, `match_ranges`, mean, min-shift),
  `_int_max_topo_rows`, `_int_apply_topo_rows`, `_int_on_topo_row_edit`, `_int_on_topo_row_toggle`,
  `_int_on_topo_add`, `_int_on_topo_remove`, `_int_set_row_widgets`, `_int_refresh_topo_rows`,
  `_int_topo_rows_window`, `_int_sync_rows_window`, `_int_close_rows_window`, `_int_place_panels`,
  `_int_averages`, `_int_sm4`, `_int_group_selection`, `_int_on_add_group`.
- REMOVED `_TOPO_LETTER`, `_int_topo_lines_text`, `_int_parse_topo_lines`, `_int_on_topo_rows`,
  `_int_on_topo_mode`, the 'topo rows' box and the mode radio.
- MODIFIED `extract_topography` ('average' gone), `plot_line_with_topo` (stripe through `_topo_stripe_profile`),
  `_int_set_topo_lines` (keeps the list of raw profiles), `_int_on_line_index` (row controls follow),
  `_redraw_topo_stripe` (the new pipeline), `_int_on_topo_toggle` (one processor at most, both off allowed),
  `_int_refresh_group_sliders` ('add' buttons shown with the sliders; `_int_place_panels`),
  `plot_line_with_topo_interactive` (`averages=None`, `sm4=None`; default rows `(chunk_index, f), (chunk_index, r)`;
  the two-entry processor box; the 'avg' box; the per-group 'add' buttons; the row controls, '+' / '-', the note;
  the wiring).
- `__main__` unchanged.

### map.py
- ADDED the marker method (constants `MAX_GROUPS`, `STACK_OFFSET_FRAC`, the panel and control layout constants,
  `M_COLORS`; methods `pixel_xy`, `xy_pixel`, `pixel_col`, `pixel_acquired`, `layout_centers`, `place_group`,
  `groups`, `group_of`, `group_center`, `group_center_xy`, `group_order`, `marker_color`, `marker_curve`,
  `group_average`, `group_label`, `_redraw_markers`, `_draw_curves`, `_on_pick`, `_on_motion`, `_on_release`,
  `_slider_group`, `_on_group_slider`, `_on_avgs_change`, `_refresh_group_sliders`, `_on_avg_toggle`,
  `_on_stack_toggle`, `_on_substack_toggle`, `_averages`, `group_selection`, `_on_add_group`,
  `_send_to_decomp`, `_send`, `_quantity`).
- MODIFIED `Map.__init__(..., averages=None, decomp_settings=None)`: the single-view layout (topo 0.04/0.28,
  map 0.36/0.28, colour bar 0.645, curve panel 0.72/0.26), the marker controls, the embedded topo through
  `processed_topo`; `draw_single` (redraws the markers and the curve panel); `_on_direction` (`processed_topo`);
  `_apply_view` (the curve panel with the single view).
- MODIFIED `__main__`: `active_procs` `[True]` -> `[False, True]`.

### browser.py
- MODIFIED imports (`Averages`, `Selection`, `processed_topo`, `toggle_processor`; `MultiAverageWindow` no longer
  imported because no longer used directly).
- MODIFIED `__init__`: `self.averages = Averages(...)` replaces `self.avg_windows`.
- MODIFIED `_open_avgs`, `_saved_avg_indices`, `_next_avg_index`, `_avgs_holding`, `_new_avg_window(..., name=None)`,
  `_add_to_avg`: calls into `self.averages`; `_update_avg_text` (names; grid files too); `_popup_open` (names);
  `_set_controls_for_kind` (the 'in' text visible for grid files too); `_on_topo_toggle` (one processor at most);
  `_draw_scan_area` (`processed_topo`); `_on_line` (rows `(i, forward), (i, reverse)`, `averages`, `sm4`);
  `_on_map` (`averages`, `decomp_settings`); `load_saved_average` (names, selection entries, refusal of an open
  name); ADDED `_entry_from_record`.
- `__main__` unchanged.

## Additional changes (not in the plan; each a consequence the plan's changes required)

1. `sts.py` `__main__`: the next free index is `1 + max([0] + Averages.saved_indices_of(directory))` instead of
   parsing `avg<N>` out of the record file names.  Required because record file names now carry the average's
   name (the old `int(m.group(1))` would fail on a stem such as `avg1`).  His file list and settings in the block
   are untouched.
2. `line.py` layout: the 'add' buttons sit at x 0.345 (the plan said 0.325, which the sliders' value text
   occupies) and the two-entry processor box at (0.37, 0.01, 0.10, 0.08) instead of (0.35, 0.01, 0.12, 0.08) so
   the two do not overlap.
3. `map.py` layout: the 'Grid view' button moved from x 0.83 to 0.84 (width 0.06) so the 'substack' box fits
   beside it.
4. `browser.py` `_set_controls_for_kind`: the 'in <averages>' text is shown for grid files (the plan's
   section 12.5 says the text appears for grid files; the visibility line had to change for that to be seen).

## Checks run (all pass; scripts in the session's scratch folder, data copies outside the repo)

- Import of every module; pyflakes clean on all of them.
- Old-versus-new (the package as it was on disk, loaded beside the new one) on the ZZ grid and the two
  ZZ single points: `STS.process` arrays identical for four filter / Feenstra settings; `STS.average`,
  `STS.mean_rows` identical to the old; Slope Sub processing of forward and reverse identical to the old.
- `Selection(grid, cols).curves(...)` equals the columns of `grid.curves(...)`: identical without Feenstra;
  with Feenstra within 1.7e-15 (relative 1e-12), from the BLAS matrix product of the broadening running on a
  different column count, the same last-bit effect `STS.process_arrays` documents for never-acquired columns.
- An avgN window holding a point file and a selection: the run's average equals the plain mean of all pooled
  columns and the mean of the pipeline columns on the common range; record format 3 written and listed;
  rename moves the record and a fit history to the new stem with the stored labels rewritten; an empty stem and
  a stem already saved are refused; the record reloads through `restore` to the identical average.
- `match_ranges` and `delta_z_subtract_profile` on synthetic profiles: a cosine plus a slope loses the slope
  (7e-17 left), a pure cosine is unchanged to 1e-17; the matched arrays take the reference's min and max, a flat
  one the reference minimum.
- Windows under Agg: the Line window as the browser opens it (rows (1, f), (1, r), Δz sub on, two row controls),
  '+' to nine rows opens the 'topo rows' window and hides the stack, '-' back to eight closes it, the hard
  maximum 2 x 32 lines refuses the 65th row, row edit and toggle, rows following the 'line' box, the processor
  box (Slope Sub on turns Δz off, both off allowed, Δz back on), three groups with three 'add' buttons, 'add' into
  a new average and into the same one by name, the same group refused, an unknown name refused, the average
  of two line groups runs; the Map window (two pixels at the centre, drag to (0, 0), two groups, four pixels,
  rigid drag with Average on, stack and substack, 'add' on acquired pixels, 'average' direction, step and view
  toggles); the Browser in a scratch folder (registry owned, processor box, Line and Map opened with the
  registry, the 'in avg2' text for the grid, the saved average with a selection reloaded and run, a second load
  of an open average refused).
- The Map's send path with `Fit.from_curve` stubbed: two sends with Average off whose data and unfiltered
  arrays equal the pipeline on the marker columns (quantity 'ratio', stage 'column', labels '... | marker at
  (x, y) nm over I/V', delta_I from the raw current); one send with Average on equal to the plain mean of the
  processed columns.  `select_model` was never executed.

## Addendum (2026-10-07, later): the I(V) page aligned by recorded location

Requested after the delivery above: the loader must read the Current page's own location list and
align its columns with the dI/dV page's locations.

- `sts.py`: ADDED `AlignmentError(ValueError)`, `STS._page_locations(page)`, `STS.page_perm(ref_xy, page_xy,
  name, label, ref_label)`, the constants `LOCATION_TOL_FRAC = 1e-3` and `LOCATION_TOL_MIN_M = 1e-14`, and the
  attribute `STS.iv_realigned`.  MODIFIED `STS.from_pages`: the Current page's columns (and the IETS page's,
  when a dI/dV page exists) are put into the order of the reference page's recorded locations
  (`RHK_SpecDrift_Xcoord/Ycoord`): identical lists leave the columns as stored; another order is realigned,
  nearest location within the tolerance (a thousandth of the smallest step between distinct reference
  locations, at least 1e-14 m), repeated sweeps at one location keeping their relative order, with a console
  line naming how many columns moved; a location with no match, or lists of different length, raise
  `AlignmentError`.
- `sm4file.py`: MODIFIED `Sm4File.__init__`: an `AlignmentError` is printed and the file is shown without
  spectra (the pairing of its pages being unknown, as for a file without a matching channel label).
  `Line.__init__`, which loads its own pages, lets the error propagate (the browser's Line button prints it).
- Checks: on the ZZ grid, the ZZ point, and a new-format point the two lists are identical in order, so every
  array is unchanged from the previous version and nothing is realigned; synthetic Current pages stored in a
  random order and in a per-line mirrored order come back in the dI/dV order exactly; a point file (one
  location, repeated sweeps) stays as stored; locations shifted by 1 nm and a shorter list are refused.
  Checks 1 to 4 above re-run and pass.
