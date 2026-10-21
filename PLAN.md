# Work plan of record: item 4

Items 1 to 3 are done and on `main` (commits `bf4a5a2` to `5c2d7e0`). This file holds item 4, the fixes of the
problems the item 3 documentation review found, as the owner decided them on 2026-10-21. It is removed when
item 4 is done. The working agreement of items 1 to 3 holds: implement exactly this, no additions; every change
updates the docstrings and comments of the files it touches and the README where the behaviour is described;
every behaviour change gets a check in `checks/smoke_browser.py` (or a check module beside it); only verified
updates are pushed, to the branch and to `main`.

## Decisions taken by the owner

1. **`anchor to topo`: build the toggle** (option A). The drift code for it exists (`Drift.result`'s `anchor`:
   the MATLAB "drift for first point", a spectra file moved rigidly by −v (t_0 − t_topo start) to where the
   surface sat when the topo under it began). The toggle is built in the browser beside `auto drift` and
   `manual drift` (`QCheck('anchor to topo')`, handler like `_on_drift_toggle`, feeding `self.anchor_on`, the
   scan area redrawn) and in the Topo window beside its own drift toggles (`widgets.make(..., 'anchor to
   topo')`, set from the browser's state at opening, independent after), off by default.
2. **The Line window opens with `calibrate` and `normalize` both off**, like every other window:
   `browser.main` and the Spyder block pass `apply_calibration: False, apply_normalize: False`. A record still
   restores the flags it saved.
3. **The calibration method stays as it is** (the regression on V, the method in use before 2026-10-16). The
   references to the other calibration sources are removed: the three-files story and the date in the
   docstrings of `calibration.py`, `sts.py`, `fit.py`, and the two comparison methods (`gainoffset`,
   `stretch`) and their dashed curves and prints in `calibration.py`'s `__main__` demo.
4. **The two stamps stay as they are** (the scan cache folder's with the milliseconds, the stores' without).

## The direct fixes

5. Console strings: the site flag line names the settings file; `set from markers` prints each pair with the
   window's scale (`marker_pair_text(k, rec, self.scale)`); the Line window's drift toggle says "(rows placed by
   the stripe matching alone)" when a stripe matching is set and "(rows plain)" only when none is; the averaging
   window's filter line says "on the recorded current"; the browser's filter box label is `filters (on the
   current)` (`widgets.SPECS` / `SHORT_LABELS` key and `browser.QFilterControls._title`).
6. Occlusion at per-file scales: `ScanLayer._occlusion` takes the drawing factor and computes every record's
   corners at its own file's factor; the browser's `_drift_state` token carries the scale timeline's record, so
   a scale change recomputes the covers.
7. Tracked pairs: `TrackedPairs` stores each pair with the two files' signatures (`sm4file._file_sig`: size and
   modification time); `get_pair` returns None when a stored signature differs from the file's now, so a replaced
   file is tracked again; `Drift.pair` puts the 'pairs' section and writes the computed store whenever it tracks
   a pair outside `build`, so a pair tracked during a session is kept. `PAIRS_FORMAT` becomes 6 (the old
   section is recomputed, as every format change does).
8. Fit history: `norm_range` joins the cached fields of `_cache_state` (restored by `from_cache`'s field loop;
   an older history without it reads None as today). The barrier height's lower bound is one helper
   (`_phi_min`: half the largest |V| plus 0.05 eV) used by `_bg_bounds` and the edit panel's refusal.
9. Map window: with the stripe matching set and both toggles off, the frame widens and the drift lines are
   drawn as for any placement (`_frame_limits`, `_frame_lines` key on a result being at hand), and the titles say
   "stripe matched" for that case instead of "drift corrected (none (stripe match only))". The grid view keeps
   one energy label per cell (`grid_labels`), created once.
10. `plots.RowHeatmap`: one helper (`_shear_rows`) pads or clips the shear to the image's rows, used by
    `_geometry` and `extent`.
11. qtplot: `set_title` without a size keeps the size last set (`font_pt(fontsize, self._title_size)`);
    `_FILE_NAME_RE` requires the extension to start with a letter; `_apply_targets` applies the numeric aspect
    ratio in the datalim widening; the dead `if False` expressions, the empty `if origin == 'upper'` branch and
    `levels[:2]` (one level) are removed.
12. Records and helpers: the saved-records window lists an old label-named fit family once, by its stem, numbered
    members included (`OLD_CACHE_RE` accepts `_fits_<n>.pkl`); `lineavg.member_from_file` defaults `topo_lines`
    to the line's forward row alone, as the Line window does; `filters.hampel` returns float64; `zero_crossing`
    takes the whole stem when a name has no `_CT_`.
13. Dead code removed: `POINT_LOADED_COLOR`, `_BrowserCore.TOPO_CBAR`, `_BrowserCore._topo_image`; the
    `stripe_on` argument of `TopoWindow`; `Drift.auto_pair_for`; `DriftWindow`'s `auto_on` argument and the
    openers' keyword; `scales.SCALES_FORMAT`; `Sm4File._z_page`; the unused `pix` in `sweeps.chosen_columns`;
    `Fit._migrate_format_2` and format 2 in `_CACHE_FORMATS_READ`; `mplexport._text`'s `target_owner`; the
    leftovers in the E-marker check (`inspect`, `src`, `rec`).

## Checks

M1 anchor to topo (browser toggle and Topo window toggle: the file's positions move by the reported shift, off
restores them); M2 the Line window opens with both flags off; M3 the console strings; M4 occlusion at two scales;
M5 tracked pairs (a changed signature is tracked again, a pair tracked in a session is written); M6 the fit's
`norm_range` and `_phi_min`; M7 the Map window's stripe-only placement and one label per cell; M8 RowHeatmap
with a short shear; M9 qtplot (title size kept, a decimal title is not a file name, aspect 2 datalim);
M10 records and helpers (old family listed once, member rows, hampel dtype, zero_crossing stem).
