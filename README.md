# sm4_Browser on PySide6 + pyqtgraph

Every window now draws and handles events with **PySide6 + pyqtgraph**. matplotlib does no drawing
anywhere. It is only used, if it is installed, to read the colours of a colour map you give by
matplotlib name or as a matplotlib object.

## How to run
* Run **`browser.py`**: there is one browser. It has your split layout: the scan area on the left,
  the spectra panels on the right, a toolbar on top of each, and the controls in group boxes
  underneath. Edit the folder and settings in its `if __name__ == '__main__':` block and press F5 in
  Spyder, or run `python -m sm4_Browser.browser <directory>`.
* `browser_qt.py` is gone; its window now lives in `browser.py`. The old single-figure browser
  layout is removed.
* Every other module's own if-main still works.

## Colour maps
The if-main inputs `cmap_line` / `cmap_map` (and `line_window_kwargs['cmap']`, the Map `cmap`, ...)
accept any of these:
* a matplotlib name: `'jet'`, `'viridis'`, `'RdBu_r'`, ...
* a matplotlib `Colormap` object, e.g. `matplotlib.colormaps['magma']` or your own `LinearSegmentedColormap`
* a pyqtgraph `ColorMap`

`qtplot.to_cmap` converts each one once (the result is cached) into the lookup tables pyqtgraph draws
with. Without matplotlib installed, `jet`, `gray` and pyqtgraph's built-in maps still work.

## Qt binding (Spyder)
qtplot uses whichever Qt binding is already loaded in the Python session. Spyder's IPython console
loads PyQt5 or PyQt6 for its Qt event loop, and IPython refuses a second binding (the
"Importing PySide6 disabled by IPython" error), so inside Spyder the package draws with that
binding. Outside Spyder it uses PySide6, or PyQt6, PyQt5 or PySide2, whichever is installed.
`PYQTGRAPH_QT_LIB` picks one when nothing has been imported yet. The package is tested on PySide6
and PyQt5.

## Requirements
PySide6, pyqtgraph (tested 0.14), numpy, scipy, plus your existing loader dependencies (spym,
opencv for the drift tracker). matplotlib is optional.

**Important:** PySide6 **6.12.0** has a bug on **Python < 3.12**. Every Qt call that returns nothing
loses one reference to `None`, and after a while any painting window aborts with
`Fatal Python error: none_dealloc`. Either use Python ≥ 3.12, or `pip install "PySide6<6.12"`.
`qtplot` checks for this when it is imported and prints a warning if your setup is affected.

## Saving figures
The Save button on every window's toolbar opens a dialog. The file type you pick there is
always the one written: a missing or different extension is replaced.

| File type in the dialog | Made by | What you get |
|---|---|---|
| PDF / EPS / SVG, publication | matplotlib | Vector files. Every line, marker, heatmap cell and text is an editable object (text as real TrueType text), ready for Adobe Illustrator. Cropped to the plots; the controls are left out. |
| PNG / TIFF, publication, 300 dpi | matplotlib | The same figure as a high-resolution image. |
| PNG / SVG, as on screen | pyqtgraph's own exporters | Exactly what the window draws. |
| PNG, screenshot of the window | Qt | The pixels of the window, controls included. |
| CSV | qtplot | Every curve of every panel as x and y columns, headed by panel and label. |

**How matplotlib is used.** The windows draw and interact with pyqtgraph only (marker dragging,
pan, zoom, and the many xy curves). matplotlib is imported only when you save a publication file.
`mplexport.py` then rebuilds the window from what qtplot records in matplotlib's terms:
* each panel at its on-screen position and limits, with its labels, title and ticks;
* curves with their styles; images with their colour map and norm, including rotated
  topographies; the sheared heatmap rows and shifted map pixels;
* colour bars, legends, texts and arrows.

This uses matplotlib's renderers without matplotlib touching interaction. pyqtgraph's own
exporters write PNG and SVG only. Its matplotlib exporter works only on PlotItem panels, which
these windows don't use, and its SVGs don't always reproduce what is on screen. Without matplotlib
installed, the publication types are not offered and Qt writes the PDF/SVG itself.

From code: `fig.savefig('a.pdf')` (publication), `fig.savefig('a.png', how='screen')`,
`fig.savefig('a.csv')`, or `mplexport.to_matplotlib(fig)` for a matplotlib Figure you can adjust
before saving.

## Titles and the info panel
Panel titles name only what the panel shows ("Scan area", "dI/dV", "I(V)", "Topography", "Fit",
"Stack", ...). Axis labels keep only the quantity and its units. Everything else that used to be in
the titles goes to an **info** panel docked at the bottom of each window (closable, can float),
one line per panel: counts, calibrate/normalize states, file names, the energy index, drift
sources, and so on. In the main browser these lines are appended to its existing info box.
The rule is in `qtplot.short_title`: each title is written as "Simple name: details", and the
panel shows the part before the first ': ' or ', '.

## Text that is never cut
* Every control measures its own text. If its box is too small (e.g. in a window narrower than
  designed), the font first shrinks to at most 75 % and the box then grows to fit.
* The filter box's row shrinks to fit its box.
* Legends use real fonts and are refitted to their panel whenever the panel's width changes.
* The Line and Map windows' group labels are short ("avg 3: 204.4–208.0 Å"); the marker count is
  shown on each group's slider.
* Text never gets smaller than its designed size.

## Text size (any monitor)
Text scales with the window, like everything else in it, so it stays readable on a 4K monitor or
in a maximized window:
* Every window has a design size (its figure size; 1700 x 950 for the browser).
* The text (plots, ticks, legends, controls, info panels, toolbar icons) is scaled by the window's
  size relative to that design size, from 0.8x to 4x.
* On a high-DPI screen that Qt does not already scale (PyQt5 inside Spyder), the windows also open
  proportionally larger.
* `qtplot.FONT_SCALE` (1.25) sets the base text size of the plots, and
  `widgets.WIDGET_FONT_SCALE` (1.1) the base size of the controls; the window zoom multiplies both.

## Unfinished sweeps
The browser's spectra panels also draw sweeps that stopped before their end, dashed, over the
points they recorded. They go through the same pipeline, and the titles count them. Averages and
fits still use only finished sweeps, as before.

## A file that stops the browser
* **Each panel is drawn on its own.** If a file breaks one panel, that panel stays empty, the
  console says why, and the browser carries on.
* **Unusable numbers are never drawn.** Values that are infinite, NaN or absurdly large, and
  images with no size, are left out. Such values come from a recording stopped before it began,
  or from positions that were never written.
* **Watchdog.** If loading or drawing a file takes more than 15 s, the console prints the Python
  line it is stuck on, again every 15 s.
* **Crash log.** If Qt or a C extension crashes the kernel, the location goes to
  `~/sm4_browser_crash.log`, together with the name of the last file the browser went to.
* **`checks/diagnose_file.py <file>`** runs one file through the loader and the browser's steps,
  timing each step and printing what the file records.

## What changed
* `qtplot.py` (new): the pyplot subset the package uses, rebuilt on pyqtgraph.
  * Each axes is a pyqtgraph ViewBox placed at its figure-fraction rectangle, so every window keeps
    its layout.
  * Ticks follow matplotlib's rules (MaxNLocator-style, with the `1eN` offset), as do legends, the
    aspect setting (`'box'` / `'datalim'`), GridSpec and `subplots_adjust`, colour bars (gamma-spaced
    ticks), `secondary_yaxis`, `inset_axes` and `contourf`.
  * The mpl events still work (press, release, motion, scroll, pick, close), plus `transData`.
  * A toolbar on top of each window: Home, Pan, Zoom, Save, and the cursor coordinates.
* `widgets.py`: same specifications and interface, but the controls are native Qt widgets in
  group-box style. The 'add existing' list is a pop-up menu.
* `plots.py`: images and colour maps use lookup tables, so changing gamma or the colour scale only
  rebuilds the table. `RowHeatmap` and `PixelMap` are each one cached quad item instead of one mesh
  per row or pixel.
* `browser.py`: the experiment logic (`_BrowserCore`, unchanged) and your Qt window layout (`Browser`).
* Speed (the math itself is unchanged):
  * The Feenstra companion only redraws its heatmaps when their data, gamma or drift change.
  * The Line and Map companion pipelines are cached per setting.
  * Marker-curve panels update in place while you drag.
  * Mouse motion is coalesced, so a slow redraw never builds up a backlog.
  * A 2D `plot` is drawn as ≤10 items instead of one per spectrum.
* `checks/alignment_windows.py`: uses `widgets` instead of `matplotlib.widgets`.

## Known differences from matplotlib
* With an equal aspect and `'datalim'` (the scan area), the view range is widened to fit the box,
  as in matplotlib.
* Markers: 'X' and 'x' both draw pyqtgraph's filled X.
* Text `wrap=True` is ignored. `clip_on` is ignored for axes-coordinate text.
* Save writes PNG/JPG (a screenshot of the window); there is no PDF/SVG output.

## Upgrade (MATLAB features and the scan layer)

### Scan area
* **Every file placed once and kept** (`scanlayer.py`): stepping only shows and hides images, it never
  redraws them. The **show** radio picks *up to current* (MATLAB's hide^), *all*, or *last 25* (the
  previous behaviour).
* **Drift as a transform:** a topo's pixel clock is linear in row and column, so a constant velocity is
  an exact affine shear of the untouched image (`scanlayer.affine_matrix`). Changing a velocity or a drift
  toggle moves the images; nothing is reprocessed or saved. `checks/check_affine.py` compares it with the
  per-pixel resampling (0.08 % of the height range).
* **Background placing:** files not yet read are read by worker processes (`SM4_BROWSER_WORKERS`,
  default 4; 0 reads in the window's thread) and placed as they arrive. The status line under the scan
  area shows the progress. Images wholly under a later complete image are not painted.
* **Caches beside the drift and sites caches:** `headers_<stamp>.json` (the header scan: only new or
  changed files are opened at startup) and `scan_cache_<stamp>/` (the processed images, drift not
  included). The 40 most recently used full files are kept, so stepping back re-reads nothing.
* **Point colours:** other spectra files are grey before the current topo and blue after it (MATLAB).
* **Step through** box: the kinds Prev/Next, Ctrl+arrows and the sliders stop at, and **min (min)** to skip
  short scans.
* **Navigation:** double-click centres the view on the clicked point (zoom kept); with the cursor over
  the scan area **R** saves the view and **space** returns to it. **Home** (toolbar or key) shows the
  full area. Scroll-to-zoom and drag-to-pan are unchanged.

### Drift window
* A click on either topo places the marker **and its partner on the other topo at once**, with the
  displacement arrow on both panels. The partner is placed with the velocity that currently governs the
  pair: an applied manual velocity, else the auto velocity of the pair, else the extrapolated auto
  velocity. Markers placed after **apply velocity** use the new one. Each click adds a pair (up to 8);
  the Markers slider follows and can go to 0.

### Stripe matching
* Done **separately for each drift setting** (none, auto, manual). The slow axis only ever gets the
  velocity of that setting. The reference is **whatever topo lies under each part of each line** (the
  most recent there), placed with **the same drift setting** as the grid. Maxima are matched to maxima
  and minima to minima over every part of a line that lies on a topo, so a line partly on a topo is
  matched on that part. The forward factor is fitted from the matched extrema (3 or more pairs), with the
  lattice period of the nearest topo as the fallback.
* **Drift none + stripe match** still offsets every line's start along the scan direction and measures
  each line's compression and stretch. `checks/check_stripe.py` tests all of this against known truth.

### Line and Map windows
* **E-markers** (Line and line-average windows): vertical lines at an energy, separate from the position
  markers, with their own row of controls (count 0 to 8, clear, readout with ΔE). Drag them on the
  heatmap or curve panel; Left/Right step the selected one by one recorded energy; Tab selects the next.
  The **Energy slice** panel shows the quantity against position at each E-marker. They are saved in the
  marker and line-average records.
* **Map window drift**: with a drift source on, every map pixel sits at its spectrum's chained position (where
  it started, by the time from the grid's first spectrum) plus the drift within its sweep up to the shown
  energy, so the maps skew with the topo; the embedded topo is drawn as quads at the pixels' chained positions
  (nothing clipped; the chosen pass, or both with `average`, the reverse under the forward); the frame widens
  to hold every shifted pixel and stays put while stepping the energy; the grid view carries the topo in its
  first cell; the companion uses the same bounds.
* **Map → new avg / map → existing avg** (Map window): the whole map, every acquired pixel's spectrum as one
  sweep of one entry, starts an averaging window or joins an open one, beside the per-group "add new" /
  "add existing" the Map already had.
* **Figure composer** (`composer.py`; Map window): "add to plot" under the topo, the map and the curve panel
  copies that panel as shown (a map at its bias with its colour limits, gamma, zoom and drift shift; the topo
  as drawn; every marker's spectrum, or every group's average, with its position) into one figure, four
  square panels wide: the topo first, then the maps in order of energy titled by bias, then the spectra. Every
  spectrum gets its own symbol (+ x * . ^ v > < o s d p h) drawn on every map at its position; every spectrum
  panel gets a down arrow at each collected map's bias and is titled "dI/dV at '+'"; maps carry a 1 nm scale
  bar and no ticks. Delete over a panel removes it; "new plot" empties the figure. Save on its toolbar gives
  the usual exports.
* **E range / x range** boxes (Line window, top row): "lo, hi" sets the shown energy range (V) and the
  shown position range (Å); a blank box is the full range. The energy range is also the normalization
  range (with normalize on, every spectrum has unit area over the shown range, nothing else) and the
  colour scale's (only the cells inside it set the colour map). The Feenstra companion shows and
  normalizes over the same range, and **Send to fit** sends the full curves normalized over it, the
  fit's refits using the same range. The position range changes nothing but the view. Both are saved
  in the marker record.
* **Every marker count can go to 0** (Markers, avgs, the drift window, the E-markers, the alignment
  editor).
* **Several sweeps per pixel** (`sweeps.py`): the pixels and their sweeps are found from the recorded
  locations. The direction of each sweep comes from its own page's energy axis (one page per direction),
  or, on one page, from a stopped sweep or from its current matching the first sweep reversed. Reverse
  sweeps are put on the same ascending axis; each keeps its own clock for the drift. A **sweeps** radio
  (forward sweeps / reverse sweeps / all sweeps / sweep n) appears in the Line and Map windows for such
  files; the choice is stored in records. The browser shows every sweep, reverse ones dashed.

### Fixed
* **line → new avg** failed in the previous version ("too many values to unpack"): the line-average
  window unpacked two of the three widgets of each topo row.
* The qtplot `Figure` methods `show`, `clf`, `_on_close`, `tight_layout`, `legend` had ended up inside
  `export_vector` (unreachable); `plt.close(fig)` raised.
* The drift window's marker placement compared marker records with `==` (a crash with numpy arrays).

### Checks (offscreen, synthetic data: `checks/synth.py`)
`smoke_browser`, `smoke_sweeps [per_pixel|pages]`, `check_affine`, `check_stripe`, `check_sweeps`,
`check_background`, `perf_layer [n_topos]`: `python -m sm4_Browser.checks.<name>`.
Requirements unchanged; OpenCV must be 4.x (KAZE), as before.
