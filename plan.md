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

### Phase 1: shared infrastructure  [x]
- [x] 1.1 Prototype: a QGridLayout on a FigureCanvas viewport with real controls and mouse-transparent
      panel cells; an Axes taking its rectangle from its cell; renders OK.
- [x] 1.2 qtplot: PanelCell, Axes(cell=), Text(cell=), Figure.set_layout / cell / set_title_cell /
      _fit_controls / at_design_size; narrow cells (inner_w); mplexport at design size.
- [x] 1.3 controls.py: the one shared control set (see section 5), box frames, FlowLayout.
- [x] 1.4 Number boxes with digit stepping (controls.step_text, tested), Count boxes framed in a
      group's colour, the marker box that scrolls past its cap.
- [x] 1.5 Smoke check helpers cover both control sets (_is_control); full run: all pass (I1, I2 too).

### Phase 2: Map window  [x]
- [x] 2.1 Grid conversion, same order and positions; sliders -> number boxes; sweeps choice under
      '< E' / 'E >'; marker box rule (2.2).
- [x] 2.2 Renders sent; the owner approved the work and every additional change ("Push the shit").
- [x] 2.3 Placements: the owner keeps the Map as it is ("as long as everything is not overlapping and
      visible its fine").

### Phase 3: every other window and standalone drawing  [x]
- [x] browser.py: main window and embedded figures on controls.*; spectra steps, spectra slider and
      Ctrl+PgUp / Ctrl+PgDown removed; the file slider kept.
- [x] line.py (sweeps choice under '< line' / 'line >'; sliders -> boxes; topo rows pop-up),
      lineavg.py, averaging.py, records.py, feenstra.py, fit.py (fit and edit windows), drift.py
      (2.4), stripe.py, section.py, topo.py, composer.py; fft.py and phase.py draw through
      plt.subplots, which now fills the automatic grid (no edit needed).
- [x] Standalone drawings: calibration.py, plots.py, widgets.py demos converted; every other
      `__main__` audited (none of them draws its own layout); checks/alignment_windows.py.
- [x] Old widgets.py control classes deleted (Control and kinds, composites, make, size_of); kept:
      SPECS, LABELS, SHORT_LABELS, short_form, fonts, filters_text, ViewNavigator, DragPath,
      PairMarkers, MARKER_PAIRS_MAX.  Specs of the removed spectra controls deleted.
- [x] Every window rendered (/tmp/render/render.py, review/renders/): no overlaps except panel
      titles wider than their panel (pre-existing; asked in review/approval.md).
- [x] README pass (new section "Layout and controls"; drift window; text size; modules).
- [x] Full smoke suite: 47 PASS, 0 FAIL; smoke_sweeps per_pixel and pages: all PASS.
- [x] Fixed on the way (listed for approval in review/approval.md): G1 (a cleared panel sharing an
      axis kept auto-range: qtplot Axes.clear keeps the shared range); I3 (browser fit slack never
      applied; re-fit until the size holds; frames' style sheets blocked the window font: fonts set
      on every widget); Feenstra H / SG boxes stretched (fixed-width text boxes get a Fixed policy);
      an empty Feenstra-settings frame in the browser.
- [x] Approved and pushed (Phases 1-3 in one commit).  Next: Phase 4, the calibration locks.

### Phase 4: calibration locks  [x]
- [x] 4.1 calibration.py: GateSettings.lock0 / lockv (off at open), parsing, locks_text, record keys;
      anchor_index (6761a76's, ANCHOR_TOL 0.5); _pins (0-lock: nearest 0 V, recorded current;
      Vbias: forward at the setpoint sample to I_setpoint, reverse at its own first acquired sample
      to the recorded current, a column averaging both directions no Vbias pin); _pinned_solve (one
      pin: the far_offsets exact-pin form; two: one-parameter least squares in a); the gate's refits
      keep the pins (_pinned_rest; stage 2's c from the Vbias pin, else the 0-lock pin, else the
      mean); IETS pinned the same way and gated on a and b; console text lists the locks.
- [x] 4.2 STS.sweep_pins: per column (direction, first sample) for the file's columns, every sweep
      view (forward +1, reverse -1, an all-sweeps average 0) and subsets; process / process_arrays
      pass it (`sweeps`); the Line window slices it like its reference; the line average and the
      browser's stopped sweeps pass theirs.
- [x] 4.3 PipelineBox: calibrate / gate / 0-lock / Vbias lock / normalize (figure windows, H and SG
      beside the gate row); calibrate / gate / H / SG / 0-lock / Vbias lock / normalize (browser);
      the gate and lock checkboxes indented by four spaces of the box's font (INDENT); the gate
      label lost its spaces.
- [x] 4.4 Records: gate_record / gate_from_record carry lock0 / lockv (the averaging window now writes
      through gate_record); older records open with the locks off.
- [x] 4.5 The calibration window: PipelineBox(steps=False): gate, H, SG, 0-lock, Vbias lock; a change
      recalibrates and redraws; a lock the file cannot hold is refused and the box put back.
- [x] 4.6 Check N5 (each lock, both, with the gate, IETS, the sweep views, the box rows, the records);
      README pipeline section rewritten (it still described the old one-gain method); docstrings.
- Open: a column averaging forward and reverse sweeps (the 'all sweeps' view) takes no Vbias pin,
  said on the console; reported to the owner.

### Phase 5: final  [ ]
- [ ] Full smoke checks pass; every window rendered and checked; README and comments current;
      plan.md ticked complete.

## 5. The layout API built in Phase 1 (use it for every window)

- qtplot (automatic layout: the one rule for every window):
  - `cell = fig.cell(min_px=(40, 40), stretch=(True, True), inner_w=None)`: a PanelCell (QWidget,
    mouse-transparent) to put in a Qt layout. `inner_w` (px at zoom 1) for a narrow panel (colour bar,
    ramp): the cell is as wide as the bar plus its tick axes and title, the bar centred.
  - `ax = fig.add_axes(cell=cell)`: the axes takes the cell's rectangle less its tick axes / title
    (Axes._decor_px). Aspect 'box' and Figure.align_row work inside the cells as before.
  - `fig.text(x, y, s, cell=cell)`: x, y are fractions of the cell. `fig.set_title_cell(cell)`:
    suptitle centred in that cell.
  - `fig.set_layout(root_layout)`: installs the layout on canvas.viewport(). After that the window's
    controls follow `fig.control_zoom` (font zoom reduced to fit, CONTROL_FIT_MIN = 0.75); the canvas
    minimum is raised only when even that does not fit (Qt's rule; windows never scroll).
  - `with fig.at_design_size():` lays the window out at its design size (mplexport uses it).
  - `plots.draw_topo_cbar(fig, cell=narrow_cell)`.
- controls.py (the one control set; the old widgets.make / widgets.* control classes are deleted):
  - `controls.make(fig, name, **kw)` (no x, y); kinds Check, Checks(indents=...), Radio(title=),
    Button, Text (digit stepping), Count (whole numbers with limits; replaces the 'Markers', 'avgs',
    'E-markers' sliders; set_color = frame colour), Slider (only the browser's 'file'), TextPanel
    (scrolling read-only text; set_text / get_text / set_visible like a figure text).
  - Every control / composite has `.qt` (its Qt widget) for the layout; `fig=None` for the browser
    (its window font rules them, browser._WindowZoom).
  - Layout helpers: `controls.box(*items, title=None, orient='h'|'v')` (the box frame),
    `controls.row(...)`, `controls.column(...)` (Qt box layouts; 'stretch' adds a stretch),
    `controls.widget_of(layout)`, `controls.FlowLayout(push_last=False)` (a control row that wraps in
    a narrow window; use it for every window's control strip, the portrait check I2 needs it).
  - Composites: StepButtons(fig, l, r, chevrons=False), FilterControls(fig, ..., title=None),
    PipelineBox(fig, ..., hs='beside'|'rows'), FeenstraSettings(fig, settings),
    StackControls(fig, stack, substack, average=None|bool, needs_average, tag) (window places
    chk_stack / chk_substack / chk_average), MarkerGroupControls(fig, max_groups, n_groups,
    markers_max) (window places `.box` and `.s_avgs`; `set_height_cap(fn)`), TopoProcessorBox,
    ResampleControls(fig, method), RunControls(fig, names), GammaBox(fig, gamma) (.qt),
    NameBox(fig, name), AveragePicker(fig, averages).open(anchor_control_or_widget, entry, on_pick).
- Conversion recipe per window (Map is the worked example, map.py __init__): build the controls with
  controls.*; make cells; arrange cells and controls in Qt layouts mirroring the old relative order;
  `fig.set_layout(root)`; add axes with cell=; replace every place / set_position / attach /
  below_corner / size_of / stack_top use by the layout; readout figure texts -> TextPanel; update
  docstrings and README; render the window (scratch script) and check no overlap; run the smoke checks.
- Scratch tools (outside the repo, recreate if missing): /tmp/render/rep.py (exact unique string
  replacement: blocks '@@@ OLD' ... '@@@ NEW' ... '
- Current calibration (main): bdf8e16, per-sweep a, b, c, nothing locked, the gate (gate_sweeps).
- 0 V reference history: 869dced:calibration.py (oldest form, b*V), 7511371:calibration.py
  `far_offsets` (exact pin, b*(V - V_far)) <- use this form.
- Setpoint anchor history: 6761a76:calibration.py through 7511371 (`anchor_index`, `_anchor`,
  `_ref_column`).
- Controls before this update (HEAD): widgets.py `make(fig, name, x, y)` placed a control at a figure
  fraction on the canvas viewport (Control._add / _layout setGeometry); qtplot Axes placed a ViewBox
  at a figure-fraction rect (Axes._px_rect / _place); the browser used Qt layouts for its own native
  controls (browser.py Q* classes).  All of these are replaced by section 5's API.
