"""
sm4_Browser: RHK .sm4 experiment browser and analysis tools.

Modules
  browser        Browser: the experiment browser (the file to run): the experiment logic and its Qt window
  qtplot         the drawing and window layer (matplotlib's interface on PySide6 / PyQt + pyqtgraph)
  sm4file        Sm4File: one .sm4 file (kind, times, image frame, topography, spectroscopy)
  sts            STS: dI/dV and IETS spectroscopy data and the pipeline driver (process_arrays)
  filters        the Hampel and Savitzky-Golay filters (on the current of the Feenstra division only)
  calibration    the calibration of dI/dV (and IETS) against the raw I(V)
  normalization  unit-area normalization
  feenstra       the Feenstra division (I/V broadening, 0 V subtraction) and the Feenstra companion window
  averaging      Averages (the averaging method and the registry), Selection, the resampling helpers,
                 the avgN window (MultiAverageWindow), the send of an average to the fit
  records        the record formats and paths (averages, line markers, line averages) and the saved
                 averages window
  widgets        every control, built by name from one specification (sizes, labels, behaviour), and
                 the composite controllers shared by the windows
  plots          every drawing of data into axes (heatmaps, colour bars, stripes, curves, topographies)
  topo           topography processing (slope and Δz subtraction, range matching), the Topo window, the raw
                 cross section of the loaded topos, and the profile window (raw height against position)
  sites          Sites: the regions of a directory's segments, the pulsing / clean / experiment flags, the
                 navigation by experiment region
  align          the alignment of grid lines (DTW, stretch equalizing, marker sections, grids, resampling)
  lineavg        LineAverages, LineAverage: the line-average windows and the alignment editor
  line           Line: a grid line's dI/dV against position, with its topography stripe
  map            Map: per-energy maps of a grid
  drift          Drift: the auto drift (consecutive topo pairs tracked by feature-point matching, Diao et al.
                 2023 with KAZE), the manual drift (the drift window: marker pairs seeded by the auto velocity,
                 one velocity box applied as a confirmed, adjusted, or typed span, the MATLAB method), the optional stripe matching
                 of grid lines, and their application to spectra positions (grid lines chained: the forward
                 pass by its clock, the reverse pass rigid at the forward end), topo pixels, heatmap rows, map
                 pixels, and drift lines
  fit            Fit: the single-point fit (model, sweep, fit history, windows)
  phase          PhaseRotate: lock-in phase rotation files
  fft            NoiseSpectrum: noise power spectrum files
  checks/        stand-alone diagnostics over the package's sm4s folder (zero_crossing)

Every module can be run on its own (its __main__ block); nothing is imported here.
"""
