# sm4_Browser

A browser and analysis package for an experiment recorded as a directory of RHK `.sm4` files
(scanning tunnelling microscopy topographies and spectroscopy). One window, the browser, shows the
experiment on its scan area in absolute coordinates, every file at its true place and size, with the
drift between scans corrected; from it every analysis window opens for the file shown: the
topography, a grid line's spectra against position, the per-energy maps of a grid, averages of
single-point spectra, line averages, the single-point fit, the manual drift and stripe-matching
windows, profiles and cross sections, and the measurement and scale windows of a path drawn on the
scan area.

This README is the guide: how to install and run, what every window and button does, the concepts
the package uses (nominal and calibrated units, the scale timeline, the drift sources, the sites,
the pipeline), the files it writes, one section per module, the checks, and the references for the
methods it implements. The details of every rule live in the modules themselves: each file begins
with a docstring that describes it exactly, and every function says what it does, with units.

Contents: [Install and run](#install-and-run) · [The data](#the-data) · [The browser window](#the-browser-window) ·
[Concepts](#concepts) · [The windows](#the-windows) · [Files the package writes and reads](#files-the-package-writes-and-reads) ·
[Saving figures](#saving-figures) · [Text size and labels](#text-size-and-labels) · [The modules](#the-modules) ·
[The checks](#the-checks) · [References](#references) · [Limits](#limits)

## Install and run

Requirements: Python 3.12 or later (or PySide6 below 6.12, see below), `PySide6`, `pyqtgraph`
(tested with 0.14), `numpy`, `scipy`, `spym` (the RHK reader; it needs `xarray` and `hvplot`),
`opencv-python` 4.x (the drift tracker uses its KAZE feature detector, which OpenCV 5 has dropped).
Optional: `matplotlib` (publication-quality figure files, and colour maps by name), `loky`
(parallel fits).

```
pip install "PySide6<6.12" pyqtgraph numpy scipy spym xarray hvplot "opencv-python-headless<5" matplotlib
```

PySide6 6.12.0 has a bug on Python below 3.12: every Qt call that returns nothing loses one
reference to `None`, and after a while any painting window aborts with `Fatal Python error:
none_dealloc`. Use Python 3.12 or later, or PySide6 below 6.12; `qtplot` prints a warning at
import when a setup is affected.

Run the browser on an experiment directory:

```
python -m sm4_Browser.browser <directory> [--scalefactor 2.75]
```

or open `browser.py` in Spyder, set the directory and the settings in its `if __name__ ==
'__main__':` block and press F5. The package must be importable as `sm4_Browser` (the folder named
so, or a link of that name). Inside Spyder the package draws with the Qt binding Spyder loaded
(PyQt5 or PyQt6); outside it uses PySide6, or PyQt6, PyQt5 or PySide2, whichever is installed;
`PYQTGRAPH_QT_LIB` chooses one when nothing is loaded yet.

The first start of an experiment takes longer: every file's header is read, every topography and
spectra file is loaded once for its height metrics and position (the sites), and every consecutive
topography pair of a segment is tracked for the drift. All of it is kept in the computed store (see
[Files](#files-the-package-writes-and-reads)) and reused at the next start.

## The data

An experiment is one directory of `.sm4` files. Every file name ends in a date-and-time stamp,
`_YYYY_MM_DD_HH_MM_SS_mmm`, which is when the recording stopped; the files are ordered by it and
the stamp is the file's time wherever the package needs one. A file's kind is read from its pages:

| kind | what it is | pages |
|---|---|---|
| `topo` | a topography image, forward and reverse passes | `RHK_PAGE_TOPOGRAPHIC` |
| `grid` | a spectroscopy grid: one spectrum per grid point, lines of `n_fast` spectra, with the grid's own topography stripes | `RHK_PAGE_INTERACTIVE_SPECTRA` |
| `point` | single-point spectroscopy: repeated sweeps at one position | `RHK_PAGE_RAMP_SPECTROSCOPY_RP` |
| `phase` | a lock-in phase rotation | line type `RHK_LINE_PHASE_ROTATE` |
| `fft` | a noise power spectrum | `RHK_PAGE_NOISE_POWER_SPECTRUM` |

Every page's stored values are converted on load to physical units (`data = stored × RHK_Zscale +
RHK_Zoffset`); "raw" in the package means the loader's data in physical units, never the stored
integers. Absolute positions come from each page's recorded offset, angle and scales; the recorded
spectrum positions of a grid are reproduced exactly by this geometry (`sm4file.py`, "Coordinates"
in `browser.py`). A grid can record several sweeps per pixel (forward and reverse); `sweeps.py`
reads how they are grouped and directed from the file itself.

## The browser window

Left, the scan area: every topography of the shown set at its absolute position and rotation, the
oldest underneath, the most recent on top, the current file's spectrum positions over it. Right,
the spectra panels of the current file and its information. Below each, the controls.

**The scan area and its toolbar.** The toolbar has Home, Pan, Zoom and Save. A scroll notch zooms
by 1.2 about the cursor (both axes together, the equal aspect kept), a left drag pans, a double
click centres the view on the clicked point at the same zoom, Home (button or key) shows the full
extent, and with the cursor over the scan area `R` saves the view and `space` returns to it. The
view persists while stepping; when the file just loaded lies outside it, the view relocates to it at
the same zoom. The panel keeps one size and shape; the equal aspect is kept by adjusting the ranges.
The topo colour bar at its right is the colour map of every topography. A grid's own topography
pixels are drawn under its markers as bands; the current grid's selected line is red with a white
square at its start; every other point or phase file is a diamond, grey when it was taken before the
most recent topography or grid at or before the current file and blue after it (the MATLAB
browser's colours); a single point's sweeps are drawn where the drift puts
them.

**Under the scan area.**

| control | what it does |
|---|---|
| file slider, `< Prev`, `Next >` | step through every file of the directory in time order; the file loads when the slider is released. `Ctrl+Left` / `Ctrl+Right` step too |
| `Topo Window` | the Topo window of the topography shown with the current file (the most recent loaded one; for a single point, the most recent loaded one whose footprint contains its position) |
| `scale factor` | the current file's scale factor; editing it sets the scale timeline from this file's time on (see [Concepts](#concepts)) |
| `topo processing` | the topography processors, one active at most (`Slope Sub`, `Δz sub`) |
| `channel` | `dI/dV` or `IETS` in the spectra panels, and the channel new Line, Map and averaging windows open on |
| `show` | which files the scan area shows: `up to current`, `all`, or `last 25` (the loaded set) |
| `step through`, `min (min)` | the kinds the stepping stops at, and the shortest duration a file must have |
| `auto drift`, `manual drift` | the two velocity sources applied to every file (see [Concepts](#concepts)); off only redraws |
| `stripe match` | the stripe window of the current grid |
| `drift` | the manual drift window of the current file |
| `anchor to topo` | grid and point files: the file moved rigidly to where the surface sat when the topography under it began (see [Concepts](#concepts)); off at start |
| `load saved averages` | the saved-records window |
| the status line | the progress of the background placing of files not yet read |

**The spectra panels** show every curve of the current grid or point through the pipeline with
every filter off: for dI/dV spectra four panels, the dI/dV, the current, its I/V and the Feenstra
ratio; for a phase rotation the lock-in output against the phase with the fitted maximum and minimum
and the set rotation; for a noise spectrum every channel against frequency. Unfinished sweeps are
dashed. Under them: the Feenstra settings (`0Vsub`, `broadened`, `width (V)`), the filters box
(`SG` with `window` and `polyorder`, `Hampel` with `window` and `nσ`, each toggle independent; the
filters act on the recorded current only), and the `calibrate` / `normalize` box (both off at
open). Then `< spectra` / `spectra >` and the `spectra` slider, which run over the spectra files
only (`Ctrl+PgUp` / `Ctrl+PgDown` too), and `prev region` / `next region`, which go to the previous
or next stop at another experiment region. The information panel lists the file's place, name,
stamp, kind, site flag and region, the loaded count, and its measurement information (setpoint,
lock-in and modulation settings, filters, sample time, range and step, points acquired, position,
notes), then one line per panel with the details cut from the panel titles. Its text can be
selected and copied: that is where a file name is copied from.

**The per-file controls.**

| control | files | what it does |
|---|---|---|
| `Line`, `Map` | grid | the Line window (the line is chosen inside it; the scan area highlights it) and the Map window |
| `add new`, `add existing` | point | the current point into a new averaging window, or into an open one chosen from the pop-up list; a line names the open averages that already hold the file |
| `section`, `clear` | any | path editing on and off, and the path removed (see [the section path](#the-section-path-and-its-three-features)) |
| `cross section`, `measure`, `set scale` | any | the three section features |
| `topo pass` | topo, grid, point | `forward`, `reverse` or `average`: the pass drawn for the topography under the current file, remembered per file and used wherever that topography is drawn |
| `site` | any | the file's flag, `pulsing`, `clean` or `experiment`, set by hand (see [sites](#sites-regions-and-flags)) |

## Concepts

### Nominal and calibrated, and the scale timeline

A file records positions and sizes in metres as the piezo calibration of the recording software had
them: the **nominal** values. The true distances are the nominal ones divided by the x/y scale
factor of the piezo at that temperature: the **calibrated** values, shown in nm. The scan area and
every window draw calibrated nm, and every measurement is given both ways.

The scale factor changes with the temperature, so an experiment keeps a **scale timeline**
(`scales.py`): a list of scale factors, each anchored at the time of the file it was set on. A file
takes the latest entry at or before its own time; the first entry set covers the whole experiment,
so a file earlier than every entry takes the earliest one; without any entry the base scale factor
applies (the `--scalefactor` argument, 2.75 by default). The `scale factor` box shows the current
file's scale and sets the timeline from the current file's time on; the `set scale` window does the
same from a measured length. Every file on the scan area is drawn at its own scale, so files from
different temperature regimes sit at their true sizes side by side; every window opened for a file
takes that file's scale; the drift window draws each topography of its pair at the scale in force at
that topography's time; the drift readouts show the calibrated value with the nominal beside it.
Nothing computed depends on the scale: it is applied at drawing and readout time. A record that
holds positions (a Line window's markers, each member of a line average) keeps the scale it was made
with, and the timeline's scale applies when it is reloaded, with a console note when the two differ;
an average holds spectra alone and needs none. The timeline is kept in the settings store.

### The pipeline

Every spectrum goes through the same steps, in `sts.STS.process_arrays`, in this order:

1. the filters on the recorded current I(V) alone (`filters.py`): Hampel (spikes), then
   Savitzky-Golay (smoothing), each on or off on its own, every filter off by default; the lock-in
   channel is never filtered;
2. the calibration of the lock-in channel against that filtered current (`calibration.py`: a
   gain-plus-offset regression per curve referenced at the sample nearest 0 V; `calibrate_iets`
   for the second-harmonic channel);
3. with a window's `Feenstra` toggle on, the division by the broadened I/V of the same filtered
   current (`feenstra.py`): Feenstra's normalized conductance (dI/dV)/(I/V), with `0Vsub` the
   current at 0 V subtracted first (a vertical shift only) and `broadened` the I/V broadened by an
   exponential weighting of the given width (V), else the plain I/V; dI/dV only, never IETS;
4. the unit-area normalization (`normalization.py`), over the whole energy axis or over the Line
   window's `E range`.

The browser's spectra panels always show the Feenstra set with no toggle; every other window has
its own `Feenstra` toggle and its Feenstra companion window. `calibrate` and `normalize` are the two
flags of each window's pipeline box, both off at open in every window (a saved record restores the
flags it was saved with). A spectrum whose column is all zeros was never acquired and takes part in
nothing.

### Drift

Two sources, one model: every file is corrected by one constant velocity `v`, the displacement of
the surface's features per second in the absolute frame (`drift.py`).

* **auto drift**: the tracker's velocity. At startup every consecutive topography pair of a
  segment is tracked by feature-point matching (KAZE features, the method of Diao et al., see
  [References](#references)); a spectra file takes the pair around it (the last topography of its
  segment before it, the first after it), a topography its pair with the next one; without a usable
  pair, the last tracked velocity when it is close to the one before it, else no correction. The
  console names the pair and the rule for every file.
* **manual drift**: a velocity set by hand in the drift window as a **span** between two
  topographies, which covers every file from the one to the other; of overlapping spans the narrowest
  applies; a span is `confirmed` (the auto velocity confirmed from seeded markers), `adjusted` (fitted
  from marker pairs) or `typed`. It overrides the auto velocity wherever it covers a file and works
  with auto drift off.

**Anchor to topo** (the MATLAB "drift for first point"): with the toggle on, a grid or point file is
also moved rigidly by −v (t_0 − t_topo start), the start of the topography under it, to where the
surface sat when that topography began; off, the recorded anchor stands. The browser's toggle sits
under the drift toggles; the Topo window has its own.

A file's anchor is its starting location at its recorded position (the first pixel of a
topography, the first spectrum of a grid, the first sweep of a point); everything else in the file
is placed relative to it by its own time: spectrum or sweep `k` at `xy_k − v (t_k − t_0)`, every
energy point of a sweep at `−v (t − t_sweep start)` from the sweep's start. On the scan area a
topography is skewed by an exact affine transform of its frame (its pixel clock is linear in row and
column), the grid's own stripes follow pixel by pixel, the Line window's heatmap rows are sheared
along position, the Map window's pixels shift with the energy, and the drift lines (a marker at every
sweep's start and a line through its sweep) are white for an auto velocity, red for a manual one and
pale yellow for a confirmed span. **Segments**: a jump of the recorded scan centre above 1 µm between
consecutive topographies is a long-range move; pairs are tracked and before/after topographies
looked up within a segment only. **Stripe matching** (the stripe window) places a grid's lines along
the fast axis against the earlier topographies on top of the velocity, per slow-axis setting; it never
changes `v`. The velocities and the tracker work in nominal metres; the readouts add the calibrated
value.

### Sites, regions and flags

At the first start every topography and spectra file is loaded once for its height metrics (the 2nd
and 98th percentiles and the median of the forward topography) and its position (`sites.py`).
Within every segment the topographies are grouped into **regions** by footprint overlap at their
recorded positions (a nearby topography joins a region under a radius rule unless its height range
is much larger). The regions are split into **pulsing** (a silver mound, high relief) and
**experiment** by Otsu's threshold on the logarithm of their height ranges; every topography takes
its region's class; a grid or point takes the flag of the most recent preceding topography whose
footprint contains it, else an ohmic test on its spectra decides pulsing; the last spectra file of a
pulsing run followed by experiment files is **clean**. The `site` radio sets a file's flag by hand,
and a hand-set flag feeds the whole chain: a topography set to pulsing or experiment sets its
region's class, every topography of the region and the spectra on them follow, a grid or point set
by hand changes that file alone, the clean rule runs on the final flags, and `prev region` / `next
region` use the regions' final classes. The hand-set flags are kept in the settings store; the
metrics in the computed store.

### Passes and sweeps

A topography has a forward and a reverse pass; the `topo pass` radios choose `forward`, `reverse`
or `average` (the two range-matched and averaged pixelwise) for the images. A grid's own stripes
are never averaged: `average` counts as `forward` for them, and with a velocity on both passes are
drawn, the chosen one on top. A grid with several sweeps per pixel gets a `sweeps` radio in the Line and Map
windows (`forward sweeps`, `reverse sweeps`, `all sweeps`, `sweep n`); the browser draws every
sweep, reverse ones dashed.

### Settings and computed values

Everything the package keeps between sessions is split by origin into two files beside the `.sm4`
files (`caches.py`): what was **set by hand** (`settings_<stamp>.json`) and what was **computed from
the files** (`computed_<stamp>.json`). The rules: a setting never enters a stored computed value; a
setting is applied at the moment a value is used, through one function per kind (a file's velocity,
a file's flag, a file's scale), and everything derived from it lives in memory and is rebuilt when
the setting changes; the computed store is loaded first and the settings applied on top; a setting
that names a file no longer in the directory is kept and reported. The computed file can be deleted
and loses nothing; the settings file is the one thing not to lose.

## The windows

Every window has a toolbar (Home, Pan, Zoom, Save, the cursor's coordinates) and an information
panel docked at its bottom with the details cut from the panel titles (closable, can float; its text
can be selected and copied). A panel title that is a file name is shown exactly as the file is
named.

### Topo window

The topography shown with the current file (the most recent loaded one; for a single point, the
most recent loaded one whose footprint contains its position, else the most recent, said on the
console) in its own scan frame, aspect locked, at the current file's scale, with the colour bar in
nm; its own `auto drift`, `manual drift` and `anchor to topo` toggles (set from the browser's at
opening, independent after); the spectra file shown with it at its corrected positions with its
drift lines; a grid's own topography pixels over the image with the `topo pass` radio; `stripe
match` for a grid.

### Line window

A grid line's dI/dV or IETS against position (`line.py`): the topography stripe of the line
(forward row at open; the row controls add rows, `+` / `−`, the line-index box and the `f` / `r`
pass button, more rows in the `topo rows` pop-up), the heatmap with its `γ` box, the marker curve
panel, the energy slice, the marker groups (`avgs` slider, one `Markers` slider per group with `add
new` / `add existing` beside it), the E-markers (`E-markers` slider, `clear E-markers`, the readout
with ΔE; drag them on the heatmap or the curve panel, Left/Right step the selected one by one
energy, Tab selects the next), `E range` and `x range` ("lo, hi"; the energy range is also the
normalization range and the colour scale's), the `line` box with `< line` / `line >`, the window's
own `Feenstra` toggle with its companion, the `calibrate` / `normalize` box, the filters,
`sweeps` for a file with several sweeps per pixel, `profile` per topography row (the raw heights
of the forward and reverse passes), `save` (the marker record), `line → new avg` / `line → existing
avg`, `Send to fit`, `drift window`, and its own drift toggles. The heatmap rows are sheared by the
drift during each sweep and carry the stripe matching's along-line placement; the stripe never
moves.

### Map window and the figure composer

The per-energy maps of a grid (`map.py`): `Grid view` toggles between the tiled grid of every
energy (the embedded topography in the first cell) and one energy beside the embedded topography
with `< E` / `E >` and the curve panel. `cmap norm` chooses one colour scale for the whole cube or one
per map; one `γ` box per energy slice. The markers are the Line window's method in 2D (pixels; the
`avgs` and `Markers` sliders, `Average`, `stack`, `substack`, `add new` / `add existing` per group,
`Send to fit`); `map → new avg` / `map → existing avg` send the whole map, every acquired pixel's
spectrum, to an averaging window. With a drift source on every pixel sits at its spectrum's chained
position plus the drift within its sweep up to the shown energy, the embedded topography is drawn as
quads at the pixels' chained positions, and the frame widens to hold every shifted pixel; with both
sources off but the grid's stripe matching set, the lines' placement alone does the same and the
titles say "stripe matched". `add to
plot` under the topography, the map and the curve panel copies that panel as shown into the figure
composer (`composer.py`): one figure four square panels wide, the topography first, the maps in
order of energy titled by bias, then the spectra, every spectrum with its own symbol drawn on every
map at its position, every spectrum panel with a down arrow at each collected map's bias, maps with
a 1 nm scale bar; Delete over a panel removes it, `new plot` empties the figure.

### Averaging windows (avg1, avg2, ...)

One indexed average of single-point files and marker-group selections (`averaging.py`). The left
panel shows every acquired sweep of every added file, overlaid or offset by `stack` (`substack`: one
curve per file or every sweep); a right-click on a sweep removes it from the average, `restore`
puts the last removed one back, `clear` empties the window. `run` averages the kept sweeps of the
files on their common energy range when their energy steps and values coincide, or with `bin` or
`interp` on, files of another step binned onto the coarsest grid or interpolated onto the finest.
The window's own `Feenstra` toggle (on at open) and companion, the `calibrate` / `normalize` box,
the filters and the `channel` radio act on every sweep. The name box names the average; `save` writes
the record; `Send to fit` writes it and sends the average to the single-point fit. The right panel
shows the average after a run.

### Line averages (lineavg1, ...) and the alignment editor

Whole grid lines sent from Line windows, aligned along position and averaged on a common grid
(`lineavg.py`, `align.py`): every member through the pipeline on its own raw columns, the alignment
by dynamic time warping against the main line with the stretch equalized over every line, or by the
marker sections of the `edit alignment` window (`apply`; `auto` returns to the DTW), the common
position and energy grids, the mean over the members covering each pixel (`crop` on: over the pixels
every member covers). `run`, `remove last`, `restore`, `save`; the Feenstra companion opens after
the first run.

### Saved records

`load saved averages` opens the saved-records window (`records.py`): every record of the `averages`
subfolder, by index, descending (name, entries and sweeps, removed sweeps, the files' times, the
filter settings, the number of fits in its fit history), then the fit histories of the earlier
label-named form, the Line windows' marker records and the line-average records. `load` rebuilds the
chosen average in a window of its index and opens the fit from its history when one exists, opens a
Line window on a marker record's file and line with everything restored, or rebuilds a line-average
window; `rescan` re-reads the directory.

### The fit

The single-point dI/dV fit (`fit.py`): the averaged curve of the sent sweeps is fitted with a sum of
split Voigt peaks (an independent Gaussian sigma and Lorentzian gamma on each side of the centre)
and known features, optionally over a tunnelling-barrier transmission background (the `WKB
background` toggle: the one-dimensional WKB transmission through a trapezoidal barrier, the model
in the module docstring; no reference is named in the code), constrained so that the model's
integral reproduces the setpoint current (no anchor for Feenstra's ratio), the candidate peaks
found above a noise-based prominence threshold, every subset of features and every peak count
fitted and the model chosen by minimum BIC. The fit window shows the data with the filter and
Feenstra controls, the stack of the raw sets, the fit and the background-divided components;
`Revert fit` / `Forward fit` walk the fit history, `Edit peaks` opens the edit window where peaks are
selected, moved, reshaped by dragging, removed, added (`Add peak`) and held per parameter, and
`Refit` runs the joint fit again; `Save fit` writes the displayed fit as JSON and CSV. Every accepted
fit is kept in the fit history file.

### The drift window

The manual drift of the current file (`drift.DriftWindow`): the pair its auto velocity comes from,
the earlier topography on the left and the later on the right, each titled with its file's name,
each drawn at the scale in force at its own time with no drift correction (the drift is measured
here), and on each panel the other topography's recorded frame as a thin black outline drawn on top,
both panels on one view centred on the two frames. A click on either topography places a numbered
marker and, with a velocity governing the pair, its partner on the other topography where that
velocity puts the same feature (hollow until moved); the k-th marker on the left pairs with the k-th
on the right, each pair in its own colour with its displacement arrow on both panels; `Markers` is
the pair count (up to 8), `clear markers` empties the draft. `< before` / `before >` and `< after` /
`after >` step either topography through the segment. One box, `v (nm/h, cal)`, in calibrated nm/h,
is what gets applied: `set from markers` fills it from the draft (every pair seeded by an auto
velocity and unmoved: that velocity, confirmed; otherwise the least-squares fit over the pairs with
their own times),
typing fills it, and `apply velocity` stores it as the span between the two topographies shown
(confirmed, adjusted or typed; an empty box removes the span). The readout lists every pair's dx,
dy, dL, dt, v_x, v_y, v_L and angle, calibrated with the nominal beside, and the applied velocity;
the box and the readouts are calibrated by the current file's scale, fixed when the window opens,
while the stepped topographies are drawn at their own. Every window redraws at once.

### The stripe window

The stripe matching of the current grid (`stripe.py`): the earlier topographies under the grid as
the scan area places them, everything drawn at the grid file's scale, the grid's two stripe passes at their placed positions and a handle at
every movable point, each line's forward end (circle) and return point (diamond); the grid's start
is anchored. `slow axis` (`off`, `auto`, `manual`) is the velocity source the window works under and
chooses which of the grid's three stored sets is edited; `fast axis` (`off`, `auto`, `manual`) is that
set's state: off places the lines by the velocity alone, auto by the extrema matching, manual by the
matching with the dragged points; `stripe pass` picks the pass drawn on top; `clear points` drops the
set's points. A drag moves a point along its line only; the first drag pins every point and sets the
fast axis to manual. The placement applies at once in the scan area, the Topo window, the Line
window and the Map window.

### The section path and its three features

`section` turns path editing on: the scan area then responds to the path's points alone (the view
is frozen, the toolbar's Home, Pan and Zoom disabled). A left click away from every point appends a
point, a drag moves a point, a right click on a point removes it; the points are numbered along the
path. `section` again turns editing off with the points kept; `clear` removes the path. Using a
feature ends editing the same way, and every open window of the path follows it while it is edited.

* `cross section`: the raw topography under the whole path, sampled at the finest pixel pitch of
  the topographies it crosses, each topography entered at its own scale, forward and reverse pages
  as two curves, into the profile window (height in nm against the calibrated distance along the
  path). The profile window's `add new` / `add existing` collect pairs into profile plots
  (`profile1`, ...), one colour per pair, forward solid, reverse dashed.
* `measure`: a window with every point's coordinates, every segment's length and the total, each
  nominal and calibrated; the lines also go to the console.
* `set scale`: a window with the path's nominal length, the `scale factor` box and the `calibrated
  length (nm)` box; editing either recomputes the other, and `apply` sets the scale timeline from
  the current file's time on. Measuring a feature of known size is how a scale factor is found.

### Phase rotation and noise spectrum

A phase rotation file shows the lock-in output against the reference phase (the fundamental and
the second harmonic when both were recorded), the phase the procedure set, and a single-phase
sinusoid fitted to every curve whose maximum and minimum are marked (`phase.py`). A noise spectrum
file shows every recorded channel against frequency with the recording's resolution and bounds
(`fft.py`).

### The Feenstra companion

Every window with its own `Feenstra` toggle (Line, Map, averaging, line average) opens a companion
(`feenstra.py`) with four versions of its spectra side by side: the dI/dV as the main window shows
it with the toggle off, the current, its I/V, and the ratio, following the main window live; its
`0Vsub`, `broadened` and `width (V)` boxes are the main window's Feenstra settings, and its own
`calibrate` / `normalize` box governs the four panels.

## Files the package writes and reads

| file | where | holds | may be deleted? |
|---|---|---|---|
| `settings_<stamp>.json` | the experiment directory | everything set by hand: the manual drift spans with their markers, kinds and velocities, the stripe states and points, the scale timeline, the site flags; written on every change | no: it is the experiment's hand-set state |
| `computed_<stamp>.json` | the experiment directory | everything computed from the files: the header scan, the sites' metrics, the tracked drift pairs (each with its two files' size and modification time, so a file replaced under the same name is tracked again) and topography times, each section with its own format number; written at the end of each computation stage and whenever a pair is tracked later | yes: rebuilt at the next start |
| `scan_cache_<stamp>/` | the experiment directory | one array file per processed topography image, keyed by the file's name, size and time, the pass, the processors and the loader format; this folder's stamp also carries the first file's millisecond field | yes: rebuilt as files are placed |
| `drift_cache_*.json.old`, `sites_*.json.old`, `headers_*.json.old` | the experiment directory | the stores of earlier versions, carried over once into the two files above and renamed | yes, once the new files exist |
| `averages/<stem>_average.json` | the `averages` subfolder | an averaging window's record: the files, the removed sweeps, the filters, the channel, the resampling, the Feenstra state, the flags, the run | they are your results |
| `averages/<stem>_fits.pkl` | the `averages` subfolder | the fit history of an average sent to the fit, with everything a refit needs | your results |
| `averages/<stem>_fit<K>.json`, `.csv` | the `averages` subfolder (next to the data file for a file fitted directly) | a saved fit: its parameters, and V, data, background, components, model | your results |
| `averages/<file stem>_line<k>_markers.json` | the `averages` subfolder | a Line window's marker record: groups, markers, E-markers, ranges, the scale it was made with | your results |
| `averages/lineavg<N>_average.json` | the `averages` subfolder | a line-average record: members (each with the scale it was made with), alignment, markers | your results |
| `<date_time>_fits.pkl` | next to a data file fitted directly | that file's fit history | your results |
| `~/sm4_browser_crash.log` | the home directory | where Qt or a C extension crashed the kernel, with the last file the browser went to | yes |

`<stamp>` is the date and time of the directory's first file, `YYYY_MM_DD_HH_MM_SS`. Nothing is
ever written among the `.sm4` files except these; the `averages` subfolder is created on the first
save or send. Every write goes to a temporary file first and then replaces the old one, so a program
that dies mid-write leaves the previous file intact. The environment variable `SM4_BROWSER_WORKERS`
is the number of worker processes that read files in the background (default 4; 0 reads in the
window's thread).

**When a file stops the browser.** Each panel is drawn on its own: if a file breaks one panel, that
panel stays empty, the console says why, and the browser carries on. Values that are infinite, NaN or
absurdly large are never drawn. If loading or drawing a file takes more than 15 s, the Python stack of
every thread is written to the console (to the crash log when there is no console), again every
15 s, so the step that hangs names itself (`qtplot.watchdog`). `python -m sm4_Browser.checks.diagnose_file <file>`
runs one file through the loader and the browser's steps, timing each step and printing what the
file records.

## Saving figures

The Save button on every window's toolbar opens a dialog; the file type picked there is always the
one written.

| file type in the dialog | made by | what you get |
|---|---|---|
| PDF / EPS / SVG, publication | matplotlib (`mplexport.py`) | vector files at the window's design size whatever its size on screen: every line, marker, heatmap cell and text an editable object, cropped to the plots, the controls left out |
| PNG / TIFF, publication, 300 dpi | matplotlib | the same figure as a high-resolution image |
| PNG / SVG, as on screen | pyqtgraph's exporters | what the canvas draws (the PNG at twice the canvas's pixels by default) |
| PNG, screenshot | Qt | the canvas's pixels, controls included, without the toolbar and the information panel |
| CSV | qtplot | every curve of every panel as x and y columns, headed by panel and label |

matplotlib is imported only at the moment of saving a publication file; without it those types are
not offered. From code: `fig.savefig('a.pdf')`, `fig.savefig('a.png', how='screen')`,
`fig.savefig('a.csv')`, or `mplexport.to_matplotlib(fig)` for a matplotlib figure to adjust before
saving.

## Text size and labels

Every window has a design size (its `DESIGN` constant; the browser 1700 × 950 pixels). Every text of
a window, the plots' ticks, labels and legends, the controls, the information panel, the toolbar
icons, is scaled by the window's size relative to its design size, the smaller of the width and
height ratios, from 0.5x to 4x (`qtplot.window_zoom`), so it stays readable on a 4K monitor or in a
maximized window and shrinks with a small one. No text is ever cut: a control whose box is too small
for its words first takes a smaller font (to 75 %) and then grows to fit; the browser's own controls
take a smaller font when they would not fit the window, and the window stops shrinking when even
that does not fit. When a window is taller than it is wide, every control with a short form shows it
(`calibrate` → `cal`, `normalize` → `norm`, `manual drift` → `manual`, ... the table
`widgets.SHORT_LABELS`) and a panel title that is a file name shows the file's date-and-time stamp
alone; the full labels and names return when the window is wider than tall again. On a high-DPI
screen that Qt does not already scale (PyQt5 inside Spyder) the windows open proportionally larger.
`qtplot.FONT_SCALE` (1.25) sets the base text size of the plots and `widgets.WIDGET_FONT_SCALE`
(1.1) the base size of the controls.

## The modules

One section per file of the package. Every module begins with a docstring that describes it
exactly, with the rules and units; this is where a question about a rule is answered. Most modules
also run on their own (`python -m sm4_Browser.<module>`, or F5 in Spyder) to show one file or a
demonstration from their `if __name__ == '__main__':` block.

### `browser.py`

The experiment and the browser window. `_BrowserCore` holds the logic: the file list, the loaded
set and its least-recently-used eviction, the scan layer, the drift, the sites, the scale timeline,
the two stores, the averaging registry, the section path and every analysis window opened for a
file. `Browser` is the Qt window: the layout, the native controls (the `Q...` wrappers of
`widgets.py`'s specifications), the keyboard shortcuts, the one text-size rule (`_WindowZoom`) and
the handlers of every button. `main` is the command line. The module docstring is the reference for
every rule of the browser: the view, the stepping, the point colours, the drift sources, the
section path, the coordinates and the caches.

### `qtplot.py`

The drawing and window layer: the subset of matplotlib's interface the package uses, rebuilt on
pyqtgraph, so every window is written as if for matplotlib (`Figure`, `Axes` placed at figure
fractions, `Line2D`, `AxesImage`, `QuadMesh`, `Text`, `Annotation`, `Colorbar`, `GridSpec`, the
mouse, key, pick and close events, `NavigationToolbar`, `FigureManager`). It also holds the
text-size rule (`window_zoom`, `ZOOM_MIN`, `ZOOM_MAX`, `FONT_SCALE`), the title rule
(`short_title`, `is_file_name`, `file_stamp`), the saving dialog and the CSV export, the crash log
and the `watchdog`, and the check of the Qt binding at import.

### `widgets.py`

Every control, built by name from one specification table (`SPECS`: kind, size in inches, label,
font; `make(fig, name, x, y, ...)` places one in a figure) with the full labels (`LABELS`) and the
short forms (`SHORT_LABELS`, `short_form`); the composite controllers the windows share
(`FilterControls`, `PipelineBox`, `FeenstraSettings`, `StackControls`, `MarkerGroupControls`,
`TopoProcessorBox`, `ResampleControls`, `RunControls`, `GammaBox`, `RangeBox`, `NameBox`,
`StepButtons`, `AveragePicker`); the scan area's `ViewNavigator` (zoom, pan, the saved view, the
freeze while editing), the section path's `DragPath` and the drift window's `PairMarkers`.

### `plots.py`

Every drawing of data into axes, so a window only decides what to show: heatmaps (`draw_heatmap`,
`RowHeatmap`, `PixelMap`), colour bars, a grid's topography stripes, curves and sweeps, marker
curves, topographies at absolute positions (`draw_topo_absolute`) or in their own frame
(`draw_topo_frame`), drift lines and markers, point sweeps, profiles, the section path
(`draw_section_path`) and a topography's recorded outline (`draw_topo_outline`); the colour
constants (`MARKER_COLORS`, the drift line colours, `PARTNER_FRAME_COLOR`).

### `sm4file.py`

One `.sm4` file. `load` reads every page in physical units into `Sm4Data` (pages, kind, stamp,
times); `Sm4File` is the view of one file the windows use: the image frame and raw topography of
each pass, the spectroscopy, phase and noise data, the sweep clock (the time of every energy point
and pixel from the recorded sample time and ramp), the recorded spectrum positions and the
measurement information. The file stamp (`filename_stamp`, `STAMP_RE`, `stamp_datetime`), the file
listing in time order (`list_sm4_files`), the kind (`sm4_kind`), the header scan kept in the
computed store (`header_cache`) and the family of cache files of a data file (`cache_family`).

### `sts.py`

The dI/dV and IETS channels of one file with its I(V) page (`STS`): the curves as recorded, their
units (`curve_units`), and the pipeline driver `process_arrays`, which runs the filters, the
calibration, the Feenstra division and the normalization in that order on any set of columns.
`AlignmentError` is raised when a file's channels do not share one energy axis.

### `filters.py`, `calibration.py`, `feenstra.py`, `normalization.py`

The four pipeline steps on arrays. `filters.hampel` and `filters.sgolay` filter the recorded
current (`filter_lia` runs the two in order). `calibration.calibrate` fits each lock-in column to
the derivative of its filtered current (`ColumnFit`: gain and offset, referenced at the sample
nearest 0 V), `calibrate_iets` the second-harmonic channel. `feenstra.divide_by_iv` divides by the
I/V (`iv_corrected` for the 0 V subtraction, `iv_broadened` for the exponential broadening),
`feenstra_titles` names the four panels, `iv_zero_crossings` is the diagnostic, and
`FeenstraWindow` is the companion window. `normalization.normalize` scales every curve to unit area
over the whole axis or the `range_mask` of an energy range.

### `sweeps.py`

A grid with several sweeps per pixel: how the sweeps are grouped and directed, read from the file
(`group_locations`, `sweep_modes`, `default_mode`), and which columns a chosen mode shows
(`chosen_columns`). The Line and Map windows' `sweeps` radio is built from it.

### `topo.py`

The topography processors (`slope_subtract_image`, `delta_z_subtract_image`, their profile forms,
`process_topo`, `match_ranges` for the averaged pass, `processed_topo` with the disk cache), the
Topo window (`TopoWindow`), the cross section along the section path (`cross_section`: the raw
heights under the path, each topography at its own scale), the profile window (`ProfileWindow`)
and the profile plots (`ProfilePlots`, `ProfilePlot`).

### `scanlayer.py`

The persistent layer of the scan area: every file placed once and kept (`ScanLayer`), the drift
applied as an affine transform of each image (`affine_matrix`, `record_corners`), the disk cache of
processed images (`cache_dir`, `cache_load`, `cache_store`) keyed by the file, the pass, the
processors and the loader format, and the update of every placed file when a velocity, a scale or
a pass changes.

### `drift.py`

The drift. The segments and the pair tracker (`track_pair` with `preprocess_image`, `kaze`,
`cluster_displacements`, the method of Diao et al.), the stored pairs (`TrackedPairs` in the
computed store), the per-file velocity (`Drift.model_for_spectra`, `Drift.model_for_topo`: the auto source, the
manual spans from the settings store; `Drift.result` is everything a window needs for one file), the stripe matching (`StripeReference`, `match_extrema`, `stripe_lag`,
`alignment_offsets`), the corrected topographies (`TopoRef`, `drift_correct_grid`), the shears and
shifts the windows apply, the marker velocity fit (`fit_marker_velocity`, `seed_partner`) and the
drift window (`DriftWindow`). Its module docstring states the drift model, the span rules and every
readout.

### `stripe.py`

The stripe window (`StripeWindow`): the grid's stripe passes over the earlier topographies, the
handles, the slow-axis and fast-axis states and the dragged points, written to the settings store.

### `sites.py`

The once-over metrics of every file (`topo_metrics`), the regions by footprint overlap
(`head_footprint`, `footprints_overlap`), the pulsing/experiment split (`otsu_threshold`), the
ohmic test of spectra (`ohmic_test`), and `Sites`: the chain from metrics and hand-set flags to
every file's final flag (`classify`, `set_flag`) and the navigation by region.

### `scales.py`

The scale timeline (`Scales`): the entries in time order, the scale in force at a file's time
(`at`, `k`), setting and removing an entry from a file's time on (`set_from`, `remove`), the
record of a scale (`record`) and the description printed to the console (`describe`). `file_time`
is a file's time from its stamp.

### `section.py`

The section path's two windows: `MeasureWindow` (every point's coordinates and every segment's
length, nominal and calibrated, from `path_lengths` and `measure_lines`) and `SetScaleWindow` (the
nominal length, the scale factor and the calibrated length, recomputed from one another; `apply`
sets the timeline).

### `caches.py`

The two stores: `Settings` (`settings_<stamp>.json`) and `Computed` (`computed_<stamp>.json`),
both `_Store`s with `get`, `put` and a temp-file-and-replace write; `open_stores` opens both for a
directory and `carry_over` imports the earlier `drift_cache_`, `sites_` and `headers_` files once
and renames them `.old`. The module docstring states the rules of settings against computed values.

### `line.py`

The Line tool: a grid line's spectra against position with its topography stripe (`line_positions`,
`line_axis`, `stripe_rows`, `common_stripe_axis`) and the Line window (`Line`): the heatmap, the
marker groups, the E-markers, the ranges, the topo rows, the profile, the records and the sends to
the averaging windows and the fit.

### `lineavg.py` and `align.py`

The line averages: `lineavg.LineAverage` is one line-average window, `LineAverages` the registry,
`member_from_file` builds a member from a file and line, and `AlignmentEditor` is the `edit
alignment` window. `align.py` is the alignment on arrays alone: the common energy grid
(`common_energy`, `on_energy_grid`), the dynamic time warping (`dtw_path`, `dtw_map`,
`align_to_main`), the stretch equalizing (`log_stretch`, `equalize_stretch`), the marker sections
(`marker_map`), the common positions, the resampling of rows, columns and profiles and the average
(`average_stack`).

### `map.py` and `composer.py`

The Map window (`Map`): the per-energy maps, the grid view and the single view, the colour scales,
the markers in two dimensions, the drift within a sweep, the sends and `add to plot`.
`composer.FigureComposer` is the figure composer: the collected panels, the symbols of the spectra
on every map, the bias arrows, the scale bars, the removal of a panel and `new plot`.

### `averaging.py`

The one averaging method (`Averages.average` on columns; a `Selection` names a file's kept
sweeps), the registry of averaging windows (`Averages`: `new`, `add`, `by_name`, `holding`), the resampling helpers (`grid_steps`, `same_step`,
`common_range`, `to_grid`), the comparison of a record with the cached fit (`compare_with_cache`),
the send to the fit (`send_sets_to_fit`) and the averaging window (`MultiAverageWindow`).

### `records.py`

The record formats and paths in the `averages` subfolder (`avg_dir_of`, `record_path`,
`line_record_path`, `cache_path_of`, `write_json`, `read_json`, `read_cache`, `saved_indices_of`,
`is_lineavg_record`, `rename_record`) and the saved-records window (`SavedAveragesWindow`). The
module docstring lists every format number.

### `fit.py`

The single-point fit (`Fit`): the model (split Voigt peaks and known features), the background
("Background model" in the docstring), the current constraint, the candidate peaks, the
enumeration of models and the selection by minimum BIC, the fit history and its cache file, the
fit window and the edit window.

### `mplexport.py`

A qtplot figure rebuilt as a matplotlib figure (`to_matplotlib`) for the publication file types;
`export` writes one and `available` says whether matplotlib can be imported.

### `phase.py` and `fft.py`

`phase.PhaseRotate` shows a phase rotation file and `fit_sinusoid` fits the single-phase sinusoid;
`fft.NoiseSpectrum` shows a noise spectrum file.

### `__init__.py`

The package docstring: what the package is, the entry point and the module map in one place.

### `checks/`

The checks, described below. `checks/averages-topo-report.md` is the report of an earlier change
(averages from lines and maps, Δz subtraction, range matching, topo rows), kept as its record.

## The checks

The checks run offscreen on synthetic experiments (`checks/synth.py`: empty `.sm4` placeholder
files with their stamps and a registry of synthetic pages, installed in place of the loader, so
every other line of the package runs as on real files; the surface and the drift are known). From
the parent directory of the package folder (named `sm4_Browser`):

```
PYTHONDONTWRITEBYTECODE=1 python -m sm4_Browser.checks.smoke_browser [names]
```

`smoke_browser` opens the browser on the standard experiment and runs every check, one line per
check, exiting non-zero on a failure; a name narrows it to the checks containing it. It configures
Qt's offscreen platform with a screen large enough for every window at twice its design size (the
built-in screen is 800 × 800). The families: the view (D), markers down to zero (E), the scan
layer and the drift transform (A), the step filter (B), the point colours (C), the drift window and
the stripe matching (G, H), the E-markers and ranges (F), the title rule (T), text size and short
labels (I), the stores and the flags (J), the scale timeline (K) and the section path (L). The other
checks: `check_affine` (the affine drift transform against per-pixel resampling), `check_stripe`
(the stripe matching against known truth), `check_sweeps` and `smoke_sweeps` (several sweeps per
pixel), `check_background` (the worker processes and the image cache), `perf_layer` (the scan layer on
a large directory), `alignment_windows` (the alignment of a line's stripe with its heatmap on real
files), `diagnose_file` (one real file, step by step), `zero_crossing` (the raw current's zero
crossing over saved averages).

## References

The methods the code implements, with the references the code names:

* The drift tracker: Z. Diao et al., "Automatic drift compensation for nanoscale imaging using
  feature point matching", Appl. Phys. Lett. 122, 121601 (2023), doi:10.1063/5.0139330; the
  features are KAZE features, P. F. Alcantarilla, A. Bartoli and A. J. Davison, "KAZE Features",
  ECCV 2012, doi:10.1007/978-3-642-33783-3_16 (`drift.py` names both and the steps it follows).
* Feenstra's normalized conductance (dI/dV)/(I/V): R. M. Feenstra, J. A. Stroscio and A. P. Fein,
  "Tunneling spectroscopy of the Si(111)2×1 surface", Surf. Sci. 181, 295 (1987),
  doi:10.1016/0039-6028(87)90170-1 (`feenstra.py`).
* The Savitzky-Golay filter: A. Savitzky and M. J. E. Golay, "Smoothing and Differentiation of
  Data by Simplified Least Squares Procedures", Anal. Chem. 36, 1627 (1964),
  doi:10.1021/ac60214a047 (`filters.py`, through `scipy.signal.savgol_filter`).
* The Hampel filter: the Hampel identifier, a moving window whose centre value is replaced by the
  window's median when it lies further than `nσ` times the scaled median absolute deviation from
  it; the estimator goes back to F. R. Hampel, "The Influence Curve and Its Role in Robust
  Estimation", J. Am. Stat. Assoc. 69, 383 (1974); the moving-window filter is reviewed in
  R. K. Pearson et al., "Generalized Hampel Filters", EURASIP J. Adv. Signal Process. 2016, 87
  (`filters.py`; the 1.4826 factor is the Gaussian consistency constant of the median absolute
  deviation).
* Otsu's threshold for the pulsing and experiment regions: N. Otsu, "A Threshold Selection Method
  from Gray-Level Histograms", IEEE Trans. Syst. Man Cybern. 9, 62 (1979),
  doi:10.1109/TSMC.1979.4310076 (`sites.py`).
* Dynamic time warping for the alignment of grid lines: H. Sakoe and S. Chiba, "Dynamic Programming
  Algorithm Optimization for Spoken Word Recognition", IEEE Trans. Acoust. Speech Signal Process.
  26, 43 (1978) (`align.py`).
* The fit (`fit.py`): the width of a Voigt peak from its Gaussian and Lorentzian widths by the
  fit of J. J. Olivero and R. L. Longbothum, "Empirical fits to the Voigt line width: a brief
  review", J. Quant. Spectrosc. Radiat. Transfer 17, 233 (1977); the noise level from the
  successive differences of the data, after J. von Neumann, "Distribution of the Ratio of the Mean
  Square Successive Difference to the Variance", Ann. Math. Stat. 12, 367 (1941),
  doi:10.1214/aoms/1177731677; the model chosen by the Bayesian information criterion of
  G. Schwarz, "Estimating the Dimension of a Model", Ann. Stat. 6, 461 (1978),
  doi:10.1214/aos/1176344136; the constrained least squares by SciPy's SLSQP. The fit's background
  is the one-dimensional WKB transmission through a trapezoidal barrier, as written in `fit.py`'s
  docstring ("Background model"); the code names no reference for it, nor for the exponential
  broadening of the I/V in the Feenstra division, nor for the calibration of the lock-in channel
  against the current (`calibration.py`, a least-squares regression of the code's own).
* The manual drift's marker method and the map composer follow the lab's MATLAB scripts named in
  `drift.py` and `composer.py`.

## Limits

* Known differences from matplotlib in `qtplot`: with an equal aspect and `'datalim'` the view
  range is widened to fit the box; `'X'` and `'x'` markers both draw pyqtgraph's filled X; text
  `wrap=True` is ignored; `clip_on` is ignored for axes-coordinate text.
* A window far narrower than half its design width crowds its controls, which keep their sizes in
  inches; the text itself is never cut.
* The fit's parallel evaluation needs `loky`; without it the candidate models are fitted one after
  the other.
