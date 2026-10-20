# Work plan of record

This file is the plan for the work in progress on this package. A new conversation reads it first and
continues from the **Status** section without asking for clarification: every decision below was made by
the owner and is final. When all three items are done, item 3 removes this file.

## Working agreement

* Implement exactly what is written here. Anything not written here is out of scope: stop and ask before
  changing it. Never weaken, remove or bypass a gate, filter, threshold or validation to make something pass.
* Before delivering a complex change, show the diff. With every delivery list every function, constant,
  gate, filter and logic branch that was added, modified or removed, split into "approved changes" and
  "additional changes I made"; the second list must be empty, otherwise stop and ask.
* Every change updates the docstrings and comments of the files it touches so that they describe the code
  as it is. Items 1 and 2 do not edit `README.md`; the README is item 3's work.
* Every feature gets a check in `checks/smoke_browser.py` (one `@check` per feature, offscreen, on the
  synthetic experiment of `checks/synth.py`). An update is finished when its checks and the whole suite pass.
* Publish only a finished, verified update: one commit per update on the branch
  `claude/quirky-mendel-1gq658`, pushed with `git push -u origin <branch>`, then `main` fast-forwarded to it
  with `git push origin <branch>:main`, so the files on GitHub show the update. Commit messages: a short
  title line "Area: what changed", a body paragraph saying what and why, then the attribution trailers used
  by the earlier commits. No `__pycache__` folders are committed.
* Running the checks (the package is imported as `sm4_Browser`, the directory is `sm4_browser`):
  ```
  ln -sfn /path/to/sm4_browser /path/to/sm4_Browser
  cd /path/to && PYTHONDONTWRITEBYTECODE=1 QT_QPA_PLATFORM=offscreen python -m sm4_Browser.checks.smoke_browser [names]
  ```
  Dependencies that were needed in a bare container: `PySide6<6.12`, `pyqtgraph`,
  `opencv-python-headless<5` (OpenCV 4: the tracker needs KAZE, which OpenCV 5 lacks), `scipy`,
  `matplotlib`, `spym` plus `xarray` and `hvplot` (spym imports them), and the system libraries
  `libegl1 libgl1 libxkbcommon0 libfontconfig1 libdbus-1-3 libglib2.0-0 libxcb-cursor0`.

## Status

* Done and published before this plan: the drift window's frames (`a811fcc`, `2690f41`) and the
  file-name title rule (`fa3f282`): each drift-window panel is titled with the exact name of its file, carries
  the other topo's recorded frame as a thin black outline drawn last, and both panels show one view centred
  on the two frames together.
* Item 1: not started.
* Item 2: not started.
* Item 3: not started.

## Item 1: text size and short labels

1. **Text scales with the window** in every window, the browser and every figure window alike: the factor is
   the window's size relative to its design size, the smaller of the width ratio and the height ratio,
   limited to **0.5x to 4x**. Today `qtplot.Figure._relayout` limits figure windows to 1x to 4x and
   `browser._WindowZoom` the browser to 0.8x to 4x; both become the one rule with the one pair of limits.
2. **Every window declares its design size** as one constant (`DESIGN = (width, height)` in pixels at
   `qtplot.DPI`, or the figure size it opens with) and opens from it. Today each window's size is a literal
   `figsize` in its constructor and the browser's 1700 x 950 is a literal in `browser.py`.
3. **Short labels when a window is taller than it is wide.** When a window's height exceeds its width every
   control with a short form shows it; when the window is wider than tall again the full labels return.
   One table in `widgets.py` holds the short forms, used by the browser's native controls (the `Q...`
   wrappers in `browser.py`) and by the figure windows' controls (`widgets.make`). The pattern, confirmed by
   the owner: calibrate → cal, normalize → norm, auto drift → auto, manual drift → manual,
   stripe match → stripe, Topo Window → Topo, scale factor → scale, load saved averages → load avgs,
   broadened → broad, substack → sub, cross section → x-section, add existing → existing,
   forward → fwd, reverse → rev, average → avg, experiment → exp, pulsing → pulse,
   anchor to topo → anchor, Send to fit → fit, apply velocity → apply v, set from markers → from markers,
   drift window → drift, forward sweeps → fwd, reverse sweeps → rev, all sweeps → all,
   step through → step; every other multi-word label gets a short form on the same pattern. The browser's
   editing button is labelled `section` (its full label; it was `section line`).
4. **File-name titles in the taller-than-wide mode** keep only the file's date-and-time stamp
   (`2024_01_01_10_05_01_440` for `Synth_CT_2024_01_01_10_05_01_440.sm4`, the stamp `sm4file.STAMP_RE`
   finds), exact and usable as a search term; in the normal mode the exact file name (done: `fa3f282`,
   `qtplot.short_title`). One rule, applied wherever a file name is shown as a title in any window.
5. **Check:** every window kind opened at its design size, at 2x, at 0.5x and in a portrait shape; the text
   follows the rule, no label is cut, labels and file-name titles switch to the short forms in portrait and
   back.

## Item 2: the path, the scale timeline, the caches

### Buttons
* `section` with `clear` under it (`clear` is not a feature button). Where `cross section` is now, a
  vertical stack of the three feature buttons: `cross section`, `measure`, `set scale`.

### Path editing on the scan area
* `section` turns path editing on. While editing, pan (drag), scroll zoom, double-click centring, Home
  (toolbar and key) and the R and space view keys are frozen; the scan area responds only to the path
  points: a left click away from a point appends a point, a drag moves a point, a right click on a point
  removes it. The points are drawn numbered, joined by the path.
* Pressing `section` while editing turns editing off without using any feature, re-enables pan, zoom, Home
  and the view keys, and leaves the points where they are. Using any feature ends editing the same way.
  `clear` removes the path; the editing state stays as it was, so with editing on the next click starts a
  new path.

### The three features
* `cross section`: samples the raw topography along the whole path (every segment in order; today
  `topo.cross_section` samples one segment), the distance along the path on the x axis, into the profile
  window, at the current file's scale.
* `measure`: a small window listing every point's coordinates, every segment's length and the total length,
  each **both nominal** (the recorded nanometres, no scale factor) **and calibrated** (divided by the scale
  in force at the current file's time).
* `set scale`: a small window with the nominal path length, the scale factor and the calibrated length.
  Editing the scale factor recomputes the calibrated length; editing the calibrated length recomputes the
  scale factor. Applying it sets the scale from the current file's time onward (the timeline below).

### The scale timeline
* The experiment keeps a list of scale factors, each anchored at the time of the file it was set on. A file
  takes the latest scale set at or before its own time; before the first entry the starting scale factor
  (the constructor's `scalefactor`, default 2.75) applies.
* The browser's `scale factor` box shows the current file's scale; editing it sets the scale from the
  current file's time onward, the same as `set scale`.
* **Every drawing uses the file's own scale.** The scan area draws every file at its own scale (the layer's
  `k = 1e9 / scalefactor` becomes per file), so files from different temperature regimes sit at their true
  sizes side by side. Every window opened for a file, Topo, Line, Map, drift, stripe, profile and the cross
  section, takes that file's scale. The drift window draws each topo image and each other-topo frame at the
  scale in force at that topo's time, and never applies the drift correction.
* **Drift magnitudes follow the timeline.** Every dx, dy, dL, nm/h and pm/s readout in the drift window,
  on the console, in the info lines and in the scan-area title is shown calibrated by the file's scale with
  the nominal value below or beside it. The typed `v (nm/h)` box is in calibrated units and is converted to
  the recorded frame for storage. The stored velocities and the tracker stay in recorded metres.
* Records saved in the `averages` folder keep the scale they were made with; on reload the timeline's scale
  applies, and a difference is reported on the console.

### The caches: two files by origin, plus the image folder
Today there are four stores, two of which mix computed and hand-set content: `headers_<stamp>.json`
(the header scan), `sites_<stamp>.json` (metrics plus the hand-set flags), `drift_cache_<stamp>.json`
(tracked pairs and topo times plus the manual spans and the stripe states and points), and
`scan_cache_<stamp>/` (one array file per processed image). They become:
* **`settings_<stamp>.json`**: everything set by hand and never recomputed: the manual drift spans with their
  markers, kinds and velocities; the stripe fast-axis states and points per slow-axis setting; the scale
  timeline; the site flags. Written at once on every change. Never invalidated by a format change, only
  migrated.
* **`computed_<stamp>.json`**: everything computed from the .sm4 files alone and reused: the header scan,
  the once-over metrics, positions and ohmic tests, the tracked drift pairs and the topo times. Each section
  carries its own format number, so a changed tracker or loader rebuilds that section only; each entry is
  keyed by the file's name and signature (size and modification time), so a changed or new file is
  recomputed alone. Written once at the end of each computation stage (header scan, once-over, tracking),
  never while stepping. Deleting it loses nothing: it is rebuilt at the next startup.
* **`scan_cache_<stamp>/`** stays a folder of per-image array files, keyed as today (file name, size, time,
  pass, processors, loader format).
* **Carry-over:** at startup, when the new files are absent and the old `drift_cache_`, `sites_` and
  `headers_` files exist, their contents are read once into the two new files and the old files are renamed
  with a `.old` suffix. The loader-format check the old caches do moves to the computed file.
* All writes stay "temporary file, then `os.replace`".

### The rules on settings versus computed values (enforced, documented, checked)
1. **Storage:** the computed file holds only what is computed from the files alone; a setting never enters a
   stored computed value. A cached result that depends on a setting carries the setting in its key (as the
   image cache carries the processor toggles), so a changed setting selects another entry, never overwrites one.
2. **Use:** a setting is applied at the moment a value is used, on top of the computed value, through one
   function per kind: a file's velocity (the narrowest manual span covering it, else auto), a file's flag
   (the hand-set value, else the automatic one), a file's scale (the timeline). Everything derived from such
   a value lives in memory and is rebuilt whenever the setting changes (`Drift._invalidate` on every manual
   change; `Sites` rebuilds its flags).
3. **Startup:** the computed file is loaded or rebuilt first, then the settings are applied on top. A setting
   that names a file no longer in the directory is kept and reported, never dropped.
4. **Records** keep the scale they were made with; the timeline applies on reload, with a console note.
5. **Check:** change every setting and assert the computed file's bytes do not change while the derived
   values in memory do.

### The site flags feed the whole chain
Today the clean rule and the region navigation read the automatic flags only and a hand-set flag changes one
file's label after the chain has run. Now the chain after the automatic flags runs on the final flags: a
hand-set flag on a topo sets its region's class (the region is the model; a zoomed image takes its
region's class), the clean rule runs on the final flags, and `next region` / `prev region` use the region's
final class. A hand-set flag on a grid or point file changes that file alone.

### Checks for item 2
The clicks, the drag, the removal, the frozen navigation while editing, the toggle off that keeps the points,
the measure numbers nominal and calibrated, the two-way edit of set scale, the per-file scales on the scan
area and in every window, the drift readouts, the two-file cache layout, the carry-over of old files, the
rules above, and the flag chain.

## Item 3: the documentation pass
* Every file in the package: every module docstring, class and function docstring and comment checked
  against the code as it stands after items 1 and 2, and corrected, completed and made precise. Not every
  file has a README; the README has a section for every file, and the details live in each file.
* `README.md` rewritten as a guide a new graduate student can work from without anyone to ask: what each
  window does and how to use it, button by button; a section per file of the package; a glossary (nominal
  and calibrated, the scale timeline, drift sources, spans, segments, regions, passes, settings versus
  computed); a table of every file the package writes and reads and what may be deleted; citations for the
  methods the code implements (the drift tracker's feature matching after Diao et al., Appl. Phys. Lett.
  122, 121601 (2023); Feenstra's normalized conductance; Otsu's threshold; Savitzky-Golay and Hampel
  filters; dynamic time warping; the others the code names).
* `README_QT.md` is the README as first uploaded, a stale copy: removed.
* This `PLAN.md` removed as the last step.
