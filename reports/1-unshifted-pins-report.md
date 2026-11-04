# Stage 1 report: the calibration locks hold without shifting the bias axis

Date: 2026-11-03.  Implements `1-unshifted-pins/unshifted-pins.md` of the hand-off folder (approved
by Benjamin Kafin on 2026-11-03).  Baseline: the repository at a126458 plus the topo-default commit
(2fd6c34, Benjamin's separate request; it touches `topo.py` and two comments in `line.py` only).
`calibration.py` was 1036 lines in the repository (1037 in the hand-off's count, a trailing-newline
difference), the diff applied cleanly.

The repository's root is the package itself (there is no `sm4_Browser/` folder inside it), so the
diff was applied with `patch -p2` from the root instead of `patch -p1` from a parent folder.  The
three resulting files are identical to the copies in the hand-off folder (line endings aside).

## Approved changes, as applied (the diff, verbatim)

### `calibration.py`
- MODIFIED `_pinned_solve(R, I, pins, what, integral=True)`: the pin rule.  The pin rows carry
  the integrated signal (first column) and the constant (last column) only, the bias columns
  zeroed; one pin determines the last parameter (c, or d), two pins (with `integral`, k ≥ 3) the
  first and the last (a and c / d); the substitution into the least squares on index sets.
  `ValueError` when two pins land on one X, or when the pins exceed what the rule can hold.
- MODIFIED `_pinned_rest(R, I, pins, held, what)`: keeps `pins[:1]` (the Vbias pin before the
  0-lock pin), shifts its current by the held gain's integral part only (`R[p, 0] * held[0]`),
  calls `_pinned_solve(..., integral=False)` on the columns after the held ones.
- MODIFIED the module docstring, the `calibrate` docstring, the `calibrate_iets` docstring.
- Untouched: `calibrate`'s body, `calibrate_iets`'s body, `_pins`, `_solve`, `anchor_index`,
  `gate_sweeps`, `gate_settings`, `GateSettings`, `ColumnFit`, `IetsColumnFit`, `gate_text`,
  `gate_record`, `gate_from_record`, the `__main__` block.

### `README.md`
- MODIFIED the pipeline section's `0-lock` / `Vbias lock` bullets and the paragraph after them.

### `checks/pin_rule_check.py`
- ADDED, as the hand-off's diff carries it (the hand-off lists it as its own additional change,
  "include or drop"; the READ-FIRST names it as this stage's gate, so it is included).

### Behaviour changes (the hand-off's list)
1. `0-lock` on a window without 0 V: the line tool's fit (V unshifted); unchanged on a window with
   0 V.
2. `Vbias lock`: the unshifted one-pin form on every window.
3. Both locks: a from the two pinned currents, c from the 0-lock pin, b fitted.
4. The gate's refits of a locked sweep: the first pin only, the same rule.
5. `calibrate_iets`: the same four items with Z and d.
6. Locks off: nothing changes.

## Additional changes made

None to the package.  Two placements of mine, outside the package's behaviour:
- This report lives in `reports/` at the repository root.  The hand-off asks for each stage's
  report "beside the earlier stage reports" and stage 2 moves `checks/averages-topo-report.md`
  "to the folder holding the stage reports" (out of `checks/`, which becomes checks only), so a
  folder of its own is the reading that satisfies both.
- `patch -p2` instead of `-p1`, see above.

## Verification (run here, 2026-11-03)

Gate: `python -m sm4_Browser.checks.pin_rule_check` exits 0; 28 checks, 0 failures (the package
importable as `sm4_Browser` through a folder of that name; the clone is `sm4_browser`).
- 0-lock, gate off, against the transcribed `calibrate_didv_curves`: max relative difference
  4.2e-15 (−2.0..−0.8 V), 2.5e-13 (±0.5 V), 1.2e-13 (−0.6..+0.6 V).
- Vbias lock, both locks, the gate's refits, IETS one and two pins: the stated equations, all ok.
- A reverse sweep starting at the 0-lock sample with both locks: refused with "the pins do not
  determine the parameters" (columns 1, 5, 9 on the −2.0..−0.8 V window).
- Gain scatter, −2.0..−0.8 V, 300 sweeps, a / a_true 16th to 84th percentile: locks off
  −0.14..1.93, 0-lock 0.88..1.12, Vbias lock 0.97..1.03, both 0.92..1.08 (the hand-off's numbers).

Beyond the gate (a scratch script, not in the repository): the stage-1 `calibration.py` against the
pre-stage one (`git show a126458:calibration.py`) with the locks off, on the check's three windows,
with and without a disturbed sweep, gate on and off, dI/dV and IETS: the calibrated arrays
`numpy.array_equal` and every a, b, c (d) equal, 24 of 24 cases.  This is the hand-off's "session
version only" check, now run against the repository's own previous module.

`py_compile` and `pyflakes` clean on `calibration.py` and `checks/pin_rule_check.py`.

## Not verified here

- No `.sm4` file was used.  The 2023 grid at −2.0..−0.8 V (in Downloads on Benjamin's computer)
  is still to be opened with the `0-lock` on in the Line window.
- `sites.ohmic_test` is unchanged (the hand-off leaves it to Benjamin).
- No decomposition (`fit.py`) was run; `fit.py` is untouched.
