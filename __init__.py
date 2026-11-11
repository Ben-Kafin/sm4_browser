"""
sm4_Browser: RHK .sm4 experiment browser and analysis tools.

Modules
  browser        Browser: the experiment browser (the file to run): the experiment logic and its Qt window
  qtplot         the drawing and window layer (matplotlib's interface on PySide6 / PyQt + pyqtgraph)
  sm4file        Sm4File: one .sm4 file (kind, times, image frame, topography, spectroscopy)
  sts            STS: dI/dV and IETS spectroscopy data and the one pipeline path every window draws from:
                 PipelineState (a window's boxes), STS.stages (the engine, every intermediate kept),
                 STS.view (what a window draws), process_arrays / process (the earlier interface)
  sweeps         several sweeps per pixel in a grid file: the grouping rules, the sweep views
                 ('forward sweeps', 'reverse sweeps', 'all sweeps', 'sweep n') and their column choice
  filters        the Hampel and Savitzky-Golay filters, applied by the pipeline to the recorded current
                 once (the calibration and the Feenstra division both use that filtered current) and to
                 the channel's columns under their own toggles
  calibration    the calibration of dI/dV (and IETS) against the co-acquired I(V): per sweep the least
                 squares of the gain, the offset and the constant of integration, through the pins of
                 the '0-lock' and 'Vbias lock' rows when on, and the gate on the gains and offsets (the
                 windows' 'gate' row)
  normalization  unit-area normalization (normalize, normalize_abs, over the whole axis or an energy range)
  feenstra       the Feenstra division (I/V broadening, 0 V subtraction) and the Feenstra companion window
  averaging      Averages (the averaging method and the registry), Selection, the resampling helpers,
                 the avgN window (MultiAverageWindow), the send of an average to the fit
  records        the record formats and paths (averages, line markers, line averages) and the saved
                 averages window
  caches         the two on-disk stores of an experiment directory: settings_<stamp>.json (hand-set) and
                 computed_<stamp>.json (computed from the files), and the carry-over of the earlier caches
  scales         Scales: the scale timeline (the piezo scale factor in force at every file's time;
                 nominal recorded metres divided by it are the calibrated lengths)
  widgets        every control, built by name from one specification (sizes, labels, behaviour), and
                 the composite controllers shared by the windows; SHORT_LABELS, the short forms shown
                 when a window is taller than it is wide
  plots          every drawing of data into axes (heatmaps, colour bars, stripes, curves, topographies)
  scanlayer      the persistent layer of the browser's scan area: every file placed once, moved by an
                 affine transform when the drift changes, cached as processed images
  section        the browser's path features: cross section, measure (nominal and calibrated), set scale
  topo           topography processing (slope and Δz subtraction, range matching), the Topo window, the raw
                 cross section of the loaded topos, and the profile window (raw height against position)
  sites          Sites: the regions of a directory's segments, the pulsing / clean / experiment flags, the
                 navigation by experiment region
  align          the alignment of grid lines (DTW, stretch equalizing, marker sections, grids, resampling)
  lineavg        LineAverages, LineAverage: the line-average windows and the alignment editor
  line           Line: a grid line's dI/dV against position, with its topography stripe
  map            Map: per-energy maps of a grid
  composer       the Map window's figure composer ('add to plot' / 'new plot')
  contour        the Map window's contour slice: the iso-line of a topography at one level (marching squares)
  topo_ldos_blend  TopoLdosBlend: the colours of a map's value and the topography's height shown together
                 (the Map window's 'Height' shading; the STM_from_DFT simulator's file, as it is)
                 and its placement at the pixels as drawn
  drift          Drift: the auto drift (consecutive topo pairs tracked by feature-point matching, Diao et al.
                 2023 with KAZE), the manual drift (the drift window: marker pairs seeded by the auto velocity,
                 one velocity box applied as a confirmed, adjusted, or typed span, the MATLAB method), the optional stripe matching
                 of grid lines, and their application to spectra positions (grid lines chained: the forward
                 pass by its clock, the reverse pass rigid at the forward end), topo pixels, heatmap rows, map
                 pixels, and drift lines
  stripe         the stripe window: the stripe matching of one grid file, its handles and sets per setting
  fit            Fit: the single-point fit (model, sweep, fit history, windows)
  phase          PhaseRotate: lock-in phase rotation files
  fft            NoiseSpectrum: noise power spectrum files
  mplexport      a qtplot figure rebuilt as a matplotlib figure and saved as a vector file (PDF, EPS, SVG);
                 PNG / TIFF are the window rendered by Qt (qtplot.Figure.window_image)
  checks/        the checks, one command runs them all (run): synth (synthetic experiments served through
                 the loader), pipeline_check (the one pipeline path), pin_rule_check (the calibration
                 locks), contour_check, pass_average_check, affine_check, stripe_check, sweeps_check,
                 window_smoke (every window on a virtual screen, its drawn arrays against view(state)), and
                 save_check (the saving of every window: exactly what it shows plus its full info panel);
                 checks/README.md: use and extend these checks, never test scripts of one's own
  tools/         scripts that need a real file or an open window, run by hand: denoise_stats, diagnose_file
                 (one file step by step), zero_crossing (the current's zero crossing over the saved
                 averages), alignment_windows (the old and the current line placement side by side),
                 perf_layer, background_processes, rebuild_panels (the denoise step's rebuild of
                 the current and its g, one sweep at a time in a window, read from stages.step, five panels)
  reports/       the reports of the changes made (each with its approved and additional changes)

Every module can be run on its own (its __main__ block); nothing is imported here.
"""
