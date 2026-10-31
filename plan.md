# plan.md: the automatic-layout update, the control changes, and the calibration locks

This file is the single source of truth for this update. A new session (or a session after its
context was compacted) reads this file and `git log` on the branch, and continues at the first
unchecked step. Every commit of this update says which steps it completes and which step is next,
and ticks them here in the same commit.

Branch: `claude/zen-brahmagupta-6yt0x4` (started from `main` at 12dd1b6). Push with
`git push -u origin claude/zen-brahmagupta-6yt0x4`. No pull request unless the owner asks.

Status marks: `[ ]` to do, `[~]` in progress, `[x]` done.

---------------------------------------------------------------------------------------------------

## 1. Working rules from the owner (hard constraints, every step)

- Implement exactly this plan. Any deviation, substitution or "improvement": stop and ask before
  writing code.
- Every delivery lists every function, gate, filter, constant or branch modified, removed or added,
  split into "approved changes" and "additional changes". If the second list is not empty, stop and
  ask before delivering.
- Never remove, weaken or bypass a gate, filter, threshold or validation to make inputs pass.
  Diagnose, present the cause, wait.
- Verify factual or technical claims (libraries, Qt, physics, tools) by searching before stating
  them; otherwise say "I don't know".
- The README and every comment touched are kept up to date with every change.
- Code comments at the layout code say which rule is followed, e.g.
  `# automatic layout: the one rule for every window (nothing placed by hand)`.
  There is no CLAUDE.md rule file (the owner removed it).
- Commit messages end with the session's attribution lines; no model names in commits or code.
- The owner reviews only the Map window. Every other window I check myself with off-screen renders.

## 2. Decisions (the owner's answers)

### 2.1 Automatic layout, every window, every standalone drawing
- Nothing is placed by hand at fixed positions or figure fractions. Overlap prevention is the code's
  job.
- Mechanism (decided): ONE Qt grid layout per window holds the controls (real Qt widgets) and one
  empty cell per plot panel; each panel draws itself in its cell's rectangle on the plot canvas.
  Not pyqtgraph's GraphicsLayout (it holds plot items only; widgets would need
  QGraphicsProxyWidget, which Qt says is not for performance-sensitive use).
  The canvas, its scene, marker dragging, panning and drawing are unchanged.
  First confirm with a small prototype that a layout on the canvas viewport works (no OpenGL
  viewport is used, verified: no useOpenGL / setViewport in the repo).
- Look: every window except Map keeps its current look: same relative order and similar
  positions of its features.
- Shared features (calibrate box, filter box, step buttons, number boxes, ...) are one shared set
  of controls with the same operation in every window (the "consistent frame").
- Box frames around control groups everywhere.
- Windows never scroll. Panels holding only text (readouts, help texts) scroll. Exception: the
  Map window's marker box (2.2).
- Spatial panels (x and y in space: topos, maps, drift panels, ...) never change their x:y ratio.
  Every row of panels and items keeps one height (the existing row rule, Figure.align_row).
- Publication export (mplexport): laid out at the window's design size, same x:y and row rules.
- The smallest window size is Qt's default: a window cannot shrink below what its layout needs.

### 2.2 Map window
- First pass: convert to the grid keeping the current order and positions (this removes the
  overlaps). Then render single view and grid view and send both to the owner, who decides what
  moves. Keep converting the other windows meanwhile.
- The 'sweeps' choice moves to the bottom right, under the '< E' / 'E >' step buttons.
- Marker box: grows and pushes the panels up as now, until it is as tall as the panel row; then it
  keeps that height and scrolls.

### 2.3 Line window
- The 'sweeps' choice moves to the bottom right, under the '< line' / 'line >' step buttons.
- Diagnose and fix: a one-line grid file with a forward and a reverse sweep at every point cannot
  show one and then the other. No owner file: reproduce with checks/synth.py (n_lines=1,
  sweeps=('forward', 'reverse')).

### 2.4 Drift window
- '< before' / 'before >' / '< after' / 'after >' become chevron arrows ('<' and '>' shapes; glyph,
  angle and thickness my choice).
- Each pair centred directly under its own topo (before topo left, after topo right); the marker
  and velocity controls in a row below.

### 2.5 Controls: sliders and number boxes
- All sliders are removed except the browser's file slider (the one across the whole experiment).
- Replaced by number boxes: the per-group 'Markers' sliders (Map, Line, line average), 'avgs',
  'E-markers' (Line window), the drift window's 'Markers' pair count.
- A group's marker box keeps its group's colour as its FRAME colour, and its place in the stack
  (ordered by the groups' centres, highest at the top), so a box still identifies its average.
- Removed from the browser: '< spectra' / 'spectra >', the 'spectra' slider, and the
  Ctrl+PgUp / Ctrl+PgDown spectra keys (the browser's step filter covers them).
- Digit stepping in every box that holds numbers, including two-number boxes (H "3,3",
  SG "9,2", velocity "Vx, Vy", ...), each number on its own:
  - with digits selected, Up / Down raises / lowers them by one in the place of the last selected
    digit, carrying into the digits to the left as arithmetic does; the same digits stay selected;
  - with no selection, the digit just left of the cursor steps;
  - a whole-number box with limits (marker counts) stops at its limits;
  - my reading (state it at delivery): each press applies the new value at once, as Enter does.

### 2.6 Calibration locks (dI/dV and IETS)
- Two new toggles, '0-lock' and 'Vbias lock', independent of each other and of the gate; any
  combination works. Stored in GateSettings with the gate (`calibration.GateSettings`), so they
  travel the gate's existing path through every window and record. Off at open; older records:
  off. Ignored while 'calibrate' is off.
- Box rows: figure windows: calibrate / gate (H, SG beside it) / 0-lock / Vbias lock / normalize.
  Browser's box: calibrate / gate / H row / SG row / 0-lock / Vbias lock / normalize.
  The CHECKBOXES of gate, 0-lock and Vbias lock are indented by the amount the gate label is
  indented today ('    gate'); the spaces are removed from the gate label.
- 0-lock: the fit is pinned at the sample nearest 0 V (whatever is closest, not exactly 0) to the
  recorded (filtered) current there, in the exact-pin form of the old far-anchor regression
  (`far_offsets` in 7511371:calibration.py): I - I_p = a (X - X_p) + b (V - V_p). Same for
  forward and reverse sweeps.
- Vbias lock, forward sweeps: pinned at the setpoint-bias sample to the setpoint current, as in
  6761a76:calibration.py (`anchor_index`, `_anchor`, ANCHOR_TOL = 0.5 of the energy step; a sweep
  without a sample at V_bias, or a file without RHK_Bias / RHK_Current, is refused with the old
  messages).
- Vbias lock, reverse sweeps: pinned at the sweep's OWN first acquired sample, to the voltage and
  recorded current there (a reverse sweep has no setpoint current). The 'all sweeps' view pins
  each column by its own direction. The forward-sweep reference (`ref`) stays unused.
- Both locks: two pins; b and c follow from a; one-parameter least squares in a.
  No special case for both pins on one sample (the owner: it does not happen).
- Gate with locks: stage 1 (a flagged a takes the trend's a) refits the remaining parameters
  keeping the pins; stage 2 (the trend's a and b) takes c from the Vbias pin when on, else the
  0-lock pin, else the current mean rule.
- IETS: I = a Z + b V^2/2 + c V + d (a gain, b offset of the calibrated d2I/dV2; c, d constants
  of integration). The locks pin it the same way (one pin removes d; two remove c and d). The gate
  now also runs on IETS, on a and b, refitting the rest keeping the pins.
- Console line: "nothing locked" becomes the list of the locks that are on.
- The calibration file run on its own (its window) gets the gate toggle, its H and SG boxes and
  both lock toggles, and redraws on a change.

## 3. Steps

### Phase 0: setup  [x]
- [x] Dependencies (README line), system libraries for Qt off-screen:
      `pip install "PySide6<6.12" pyqtgraph numpy scipy spym xarray hvplot "opencv-python-headless<5" matplotlib`
      `apt-get install -y libegl1 libgl1 libxkbcommon0 libfontconfig1 libglib2.0-0 libdbus-1-3 fonts-dejavu-core`
- [x] Package link for imports: `mkdir -p /tmp/pkg && ln -sfn /home/user/sm4_browser /tmp/pkg/sm4_Browser`
- [x] Baseline: `cd /tmp/pkg && python -m sm4_Browser.checks.smoke_browser`: 47 PASS, 0 FAIL (3 min).
- [x] Render harness (scratch, NOT in the repo): /tmp/render/render.py opens the synthetic experiment
      through checks/smoke_browser.py's helpers (`make_browser`, `_item1_windows`, `open_map`) and
      saves every window's grab() as PNG; recreate it from that description if the container is new.

### Phase 1: shared infrastructure  [ ]
- [ ] 1.1 Prototype: a QGridLayout on a FigureCanvas viewport with real controls and mouse-transparent
      panel cells; an Axes taking its rectangle from its cell; markers still draggable; renders OK.
- [ ] 1.2 qtplot: Figure grid API (rows / columns / cells, nested boxes); Axes bound to a cell
      (the aspect 'box' shrink and the row rule work inside the cell); colour bars and titles in
      cells; figure texts that are readouts become scrolling text panels; mplexport at design size.
- [ ] 1.3 widgets: one shared control set placed by the grid (no x, y); box frames everywhere;
      portrait short labels and the text-zoom rule kept.
- [ ] 1.4 Number box with digit stepping (2.5); colour-framed marker count boxes; marker box that
      scrolls past a maximum height.
- [ ] 1.5 Smoke checks updated for the new API (no check weakened) and passing.

### Phase 2: Map window  [ ]
- [ ] 2.1 Grid conversion, same order and positions; sliders -> number boxes; sweeps choice under
      '< E' / 'E >'; marker box rule (2.2).
- [ ] 2.2 Render single and grid views; send to the owner; continue with Phase 3 meanwhile.
- [ ] 2.3 Apply the owner's placement decisions (after the renders).

### Phase 3: every other window and standalone drawing  [ ]
- [ ] browser.py: Browser main window and its embedded figures; QPipelineBox etc.; remove the
      spectra step buttons, the spectra slider, Ctrl+PgUp / Ctrl+PgDown; keep the file slider.
- [ ] line.py: Line window (sweeps choice under '< line' / 'line >'; 2.3 fix; sliders -> boxes);
      topo rows window.
- [ ] lineavg.py: line average window.
- [ ] averaging.py: MultiAverageWindow.
- [ ] records.py: SavedAveragesWindow.
- [ ] feenstra.py: FeenstraWindow.
- [ ] fit.py: the fit window; the fit edit window.
- [ ] drift.py: DriftWindow (2.4).
- [ ] stripe.py: StripeWindow.
- [ ] section.py: MeasureWindow, SetScaleWindow.
- [ ] topo.py: TopoWindow, ProfileWindow, ProfilePlot.
- [ ] composer.py: FigureComposer.
- [ ] fft.py: NoiseSpectrum. phase.py: PhaseRotate.
- [ ] Standalone drawings (`__main__` blocks that draw): calibration.py (its controls come in
      Phase 4), plots.py, widgets.py; audit every other `__main__` (align, averaging, browser,
      composer, drift, feenstra, fft, filters, fit, line, lineavg, map, normalization, phase,
      plots, records, sites, sm4file, stripe, sts, topo, widgets, checks/*) and convert those
      that draw.
- [ ] checks/alignment_windows.py window.
- [ ] Render every window; confirm no overlaps; README updated.

### Phase 4: calibration locks  [ ]
- [ ] 4.1 calibration.py: GateSettings fields, parsing, text, record keys; pins; one- and two-pin
      fits; refits with pins; IETS locks and gate; console text.
- [ ] 4.2 Per-column sweep direction reaches calibrate (forward / reverse / 'all sweeps').
- [ ] 4.3 Boxes: the rows and indent (2.6) in the shared pipeline box (figure windows and browser).
- [ ] 4.4 Records: saved and restored in every window; older records off.
- [ ] 4.5 The calibration window's controls (2.6).
- [ ] 4.6 Checks: each lock passes the fit through its point, both together, gate keeps the pins,
      IETS, boxes, records. README and comments.

### Phase 5: final  [ ]
- [ ] Full smoke checks pass; every window rendered and checked; README and comments current;
      plan.md ticked complete.

## 4. Reference

- Current calibration (main): bdf8e16, per-sweep a, b, c, nothing locked, the gate (gate_sweeps).
- 0 V reference history: 869dced:calibration.py (oldest form, b*V), 7511371:calibration.py
  `far_offsets` (exact pin, b*(V - V_far)) <- use this form.
- Setpoint anchor history: 6761a76:calibration.py through 7511371 (`anchor_index`, `_anchor`,
  `_ref_column`).
- Controls today: widgets.py `make(fig, name, x, y)` places a control at a figure fraction on the
  canvas viewport (Control._add / _layout setGeometry); qtplot Axes place a ViewBox at a
  figure-fraction rect (Axes._px_rect / _place); the browser already uses Qt layouts for its native
  controls (browser.py Q* classes).
