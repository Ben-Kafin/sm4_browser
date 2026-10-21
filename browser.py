# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 2026

@author: Benjamin Kafin

Experiment browser for a directory of RHK .sm4 files.

Files
-----
Every .sm4 in the directory, ordered by the date-and-time stamp at the end
of the file name (_YYYY_MM_DD_HH_MM_SS_mmm).  The arrows step through every
file in that order regardless of its type.  A file's type is learned the
first time it is loaded:
  'topo'  : image pages only (RHK_PAGE_TOPOGRAPHIC and the like)
  'grid'  : spectroscopy grid (RHK_PAGE_INTERACTIVE_SPECTRA pages; one spectrum
            per grid point, positions in RHK_SpecDrift_Xcoord / Ycoord; a line
            is one run of n_fast consecutive spectra, as chunk_index in the
            Line tool)
  'point' : single-point spectroscopy (RHK_PAGE_RAMP_SPECTROSCOPY_RP pages;
            repeated sweeps at one position)
  'phase' : a lock-in phase rotation (RHK_LINE_PHASE_ROTATE pages; the lock-in
            output against the phase at one position, see phase.py)
  'fft'   : a noise spectrum (RHK_PAGE_NOISE_POWER_SPECTRUM pages; the recorded
            channels against frequency, see fft.py)

Scan layer
----------
Every file of the directory is placed on the scan area once and kept
(scanlayer.ScanLayer; the MATLAB browser's model): the 'show' radio shows
the files up to the current one, all of them, or the loaded set ('last 25');
stepping only shows and hides images.  Files not read yet are read in the
background (worker processes) and placed as they arrive; their processed
images are cached on disk (scan_cache_<stamp>/), the header scan too
(headers_<stamp>.json).  The drift is applied to each image as an exact
affine transform of its frame (scanlayer.affine_matrix), never baked into
its pixels.  Other spectra files are drawn grey before the current topo and
blue after it.  The 'step through' box limits Prev / Next and the sliders to
chosen kinds and a minimum duration.

Loaded set
----------
The current file and the WINDOW - 1 files before it (WINDOW = 25) are held
in memory (the 40 most recently used full files beyond them as well).  When none of them is a topography, the nearest preceding
topography is loaded as one more file.  When the current file is a single
point whose position lies under none of the loaded topographies, the most
recent preceding topography whose scanned area contains that position is
loaded as one more file and drawn at its own time depth in the stack.
Stepping forward or back slides the window; files loaded only to learn
their type or footprint are released at once (type and footprint are
remembered).

Coordinates
-----------
Absolute position of pixel (i, j) of an image page (i = scan line, j =
pixel along the fast axis, raw file order):
    (x, y) = (RHK_Xoffset, RHK_Yoffset)
             + R(RHK_Angle) . ((j - (nx - 1) / 2) RHK_Xscale, (i - (ny - 1) / 2) RHK_Yscale)
This reproduces the recorded spectrum positions of a grid file exactly
(checked on file.sm4, all 1024 points).
spym flips image columns when RHK_Xscale < 0 and rows when RHK_Yscale > 0;
those flips are undone here so that the raw order is used.  Images are read
from the page data at the resolution they were saved at (spym discards the
SM4 thumbnail objects) and drawn without interpolation or anti-aliasing.  Every x and y
(positions, extents, markers) is divided by the scale factor in the browser
window (default 2.75) and shown in nm.  For every grid file the embedded
Topography spectra channel is compared with the image page once (self
check, printed on the console).

Browser window
--------------
Left: the scan area.  Every topography in the loaded set at its absolute
position and rotation, oldest underneath, most recent on top, with the
current file's spectrum positions drawn over it (the selected line of a grid
highlighted, its first spectrum as a white square: the start of the line;
the grid's own forward and reverse topo stripes drawn under the markers, see
"Grid stripes" below).  Every loaded single point is marked as a cyan
diamond; the current single point as a lime cross on top (its sweeps, see
"Drift correction" below); a loaded phase rotation is marked like a single
point (its recorded position, no drift correction).  A scroll notch zooms
the scan area by 1.2 about the cursor (both axes together, the equal
aspect kept), a left-button drag pans it, a double-click centres the view on
the clicked point at the same zoom, the toolbar's Home button (or the Home
key) resets it to the full extent, and with the cursor over the scan area R
saves the current view and space returns to it (widgets.ViewNavigator).  The scan-area panel keeps one size
and shape whatever is loaded and however the view is moved: the equal
aspect is kept by adjusting the axis ranges, never by reshaping the box.
The view persists while stepping files; whenever the file just loaded (a
topo by its footprint, a spectra file by its positions) lies outside the
view, even in part, the view relocates to it at the same zoom, centred on
it, widening only when the file does not fit at that zoom.  The topo
colour bar right of the scan area (no scale: the topo colour map alone) is
the one of every page that uses it.  Right: every curve of the
current grid or point through the pipeline below with every filter off
(the panel has no toggles), overlaid, on the channel the 'channel' radio
says (dI/dV or IETS; the file's own channel when it lacks that one); for a
phase rotation the panel shows its curves against the phase with the fitted
maximum and minimum and the set rotation marked, for a noise spectrum the
recorded channels against frequency.  Under the panel '< spectra' /
'spectra >' and the 'spectra' slider, which run over the spectra files of
the directory only (points, grids, phase rotations, and noise spectra, in
time order; every file's type is read from its headers at startup,
sm4_kind) and go to the chosen spectra file with the same loading rules as
the file slider; the two sliders are tied, the spectra slider staying on
the spectra file last shown while a topo is current; 'prev region' /
'next region' beside the arrows go to the previous / next stop at an
experiment region (see "Sites and regions").  Under that the info panel:
the file's place, name, stamp, type, its site flag and region, and the
loaded count, then the measurement information of the file (setpoint,
lock-in and modulation settings, filters, sample time, range and step,
points acquired, position, notes) as a grid, SI prefixes on the units.
Below: a file slider (drag anywhere in the directory; the file loads when
the mouse is released), '< Prev' / 'Next >', the 'Topo Window' button, the
'scale factor' box, the topography processing toggles (topo.TOPO_PROCESSORS,
one active at most), 'load saved averages' (the saved averages window),
and along the bottom left the drift controls: the 'auto drift' and
'manual drift' toggles (every file: the two velocity sources for the
topos' skew and the spectra's correction and drift lines on the scan area,
see "Drift correction"), the 'stripe match' button (grid files: the
stripe window, see "Stripe matching") and the 'drift' button, which opens
the manual drift window of the current file (see "Manual drift").  Right, for the current file: 'Line' / 'Map'
(grid; the line is chosen inside the Line window and the scan area
highlights it), the 'calibrate' / 'normalize' box of the spectra panels
(widgets.PipelineBox, both off at open, as in every window that uses
Feenstra; see the pipeline order below), the cross-section controls
('section', 'cross section'; see "Cross section"),
'add new' and 'add existing' (point; the second drops a list of the open
averages above the button, and a text line names the open averages that
already hold the current file), the forward / reverse / average list
('topo pass') for the topo under the current single point, for the file
itself when a topo is current, and for a grid's own topo when a grid is
current (its stripes under the markers; see "Grid stripes"; remembered per
file and used wherever that topo is drawn),
and, always, the 'site' radio (pulsing / clean / experiment) of the current
file.  Every control is built through widgets.py (one specification per
widget name, so a control shared with another window is the same there);
every drawing goes through plots.py.
Text size: the window's design size is Browser.DESIGN (1700 x 950 pixels);
every text of the window, the controls, the info panel, the panels' ticks,
labels and legends, is scaled by the window's size relative to it, from
0.5x to 4x (qtplot.window_zoom, the one rule of every window; the
figure windows apply it to their own DESIGN sizes).  When the controls
would not fit the window at that size their font is reduced, to at most
75 % of it, and when even that does not fit the window stops shrinking
(_WindowZoom), so no text is ever cut.  When the window is taller than it
is wide every control with a short form in widgets.SHORT_LABELS shows it
('calibrate' -> 'cal', 'manual drift' -> 'manual', ...) and every panel
title that is a file name shows the file's date-and-time stamp alone; the
full labels and names return when the window is wider than tall again.

Analysis windows
----------------
Topo window : the most recent topography in its own scan frame, equal aspect
              locked (for a grid: the topo it lies on, with the grid's own
              topo pixels drawn over it and a 'topo pass' radio for them);
              for a single point (or a phase rotation), the most recent
              loaded topography containing its position, with the sweep
              positions marked as on the scan area.
Averages    : indexed windows avg1, avg2, ... (the index is unique per
              experiment directory: 1 + the highest index among the open
              windows and the saved records).  'add new' in the browser
              starts the next index with the current single point; 'add
              existing' adds it to the chosen open average.  Every added
              file contributes every one of its sweeps (overlaid, or offset by
              the 'stack' toggle; stacked, 'substack' off shows one curve per
              file, the mean of its kept sweeps, 'substack' on every sweep);
              'run' averages the kept sweeps of the files that still have one
              on their common energy range when their energy steps and values
              coincide, or, with the 'bin' / 'interp' toggle on, with files of
              another step binned onto the coarsest grid or interpolated onto
              the finest (a removed sweep counts as never loaded, so a file
              emptied by right-clicks takes no part in the checks or the
              average), and shows the average in the panel beside the
              individual curves (the plain mean of every kept sweep).  'save'
              writes the record avg<N>_average.json
              in the subfolder 'averages' of the experiment directory (the
              files in added order, the removed sweeps in removal order, the
              filter settings, the run range); 'Send to fit' writes it too,
              and its fit history is avg<N>_fits.pkl in the same subfolder
              (reused on a later send only while its stored curve equals the
              current average; otherwise a fresh fit replaces it), as are the
              fit's saved fits avg<N>_fit<K>.json / .csv.  The
              subfolder is created on the first save or send; nothing is
              written among the .sm4 files.
Line averages : lineavg1, lineavg2, ... (lineavg.LineAverage): whole grid
              lines sent from Line windows ('line → new avg' / 'line →
              existing avg'), aligned along position (the DTW against the
              main line with the stretch equalized over every line, or the
              marker sections of the alignment editor) and averaged on a
              common grid; their records lineavg<N>_average.json sit in the
              same subfolder.
Saved averages : every record of the 'averages' subfolder by index,
              descending, then the fit histories of the earlier label-named
              form (<stamp>_average_of_..._fits.pkl) in the directory itself
              that hold no record, then the Line windows' marker records
              (<file stem>_line<k>_markers.json) and the line-average
              records.  'load' rebuilds the chosen average in a window of
              its index (files resolved by base name in the directory, then
              by the stored path; the removed sweeps re-applied in order; the
              filter settings set; the run executed; the rebuilt average
              compared with the fit history's stored curve on the console)
              and opens the fit from the fit history when one exists; a
              marker record opens a Line window on its file and line with
              everything restored; a line-average record rebuilds the
              line-average window (members re-added, alignment and markers
              restored, run executed).  An old label-named fit history gets
              the next free index; its files are the first by path and the
              others by time of day among the directory's files, and when a
              time matches no file or several, only the fit is opened.
Line        : Line.plot_line_with_topo_interactive on the current file and
              line (its 'add' buttons use the browser's averages, its
              'line → ...' buttons the browser's line averages).
Map         : toggle between a tiled grid of every energy and one energy
              beside the file's own embedded topography with energy arrows.
The browser, the average, Map, Line, and line-average windows, and the
fit's edit window carry the Savitzky-Golay / Hampel controls
(widgets.FilterControls; every filter off by default, each switchable on
its own) and the 'calibrate' / 'normalize' box (both off at open in every
window that uses Feenstra, each switchable on).  The filters act on the
recorded current I(V) alone, once, wherever it is used (Hampel, then
Savitzky-Golay, filters.filter_lia: the pipeline filters the current first
and hands it to the calibration and to the Feenstra division alike, before
the 0 V subtraction and the broadening); the channel's columns (dI/dV, or
IETS by the 'channel' choice of the browser and of the windows) are never
filtered.
Order per curve (STS.process_arrays): calibration against the filtered
I(V) (calibration.calibrate for dI/dV, calibration.calibrate_iets for
IETS, of whichever calibration file is in place: calibration_zeroref,
calibration_gainoffset, or calibration_stretch renamed to
calibration.py); the division by the
filtered, broadened I/V when the window's 'Feenstra' toggle is on (on by
default; the browser has no toggle: its spectra panels always show the
Feenstra set; Feenstra's normalized conductance: each calibrated dI/dV
column divided point by point by the same column's I/V, broadened by an
exponential weighting of width iv_broadening_V, see feenstra.divide_by_iv;
a dimensionless ratio; every index and energy value is kept; dI/dV only);
unit-area normalization (normalization.normalize).  The Feenstra settings
('0Vsub': each column's current at 0 V subtracted before its I/V is formed,
a vertical shift of the current only, feenstra.iv_corrected; 'broadened'
and its width box: the I/V broadened, or plain with it off) sit under the
browser's spectra panels (four panels for dI/dV spectra: dI/dV, the
current, its I/V, the ratio; the filter box under them) and in every
window's Feenstra companion; '0Vsub' starts on, 'broadened' off.  A
spectrum whose column is all zeros was never acquired: it is left out of
every calculation and never drawn.  A send to the fit carries,
per file, the raw LIA and I(V) columns of the sweeps that take part, with
the filter settings and the Feenstra settings, so that the fit
reruns this pipeline under any filter setting.  Every data array is
float64 from the file on.  The Line, Map, averaging, and line-average
windows receive the browser's Feenstra settings at opening and hold their
own from then on (their companions' boxes).
With a window's toggle on, 'Send to fit' sends the ratio as quantity
'ratio' (the fit then runs without its current anchor, see fit);
a record stores the state (keys 'divide_iv', 'iv_broadening_V', 'iv_broadened',
'iv_zero_sub') and a load sets the toggle and the settings to it (a record
without the settings loads as it was made: no subtraction).

Sites and regions
-----------------
At the first startup every topography and spectra file of the directory is
loaded once for its height metrics and its position (console progress
line), cached in sites_<stamp>.json beside the drift cache (per file with
its modification time; later startups read the cache and compute only
files not in it).  Within every segment (the topos between long-range
moves) the topos are grouped into regions by footprint overlap at their
recorded absolute positions, before any drift correction (a correction
never changes a region or a flag): any topo overlapping a member joins
the region; a topo whose centre lies within a region's radius (half the
farthest distance between its footprints' corners) joins it too unless
its own height range lies above the split threshold while the region's
does not; a topo overlapping nothing and near no region is a region of
one.  Per region the absolute Z range is max(z98) - min(z2) over its
topos; the regions are split into pulsing and experiment by Otsu's
threshold on the logarithm of their ranges (no height number in the code;
the ranges and the threshold are reported on the console); every topo of a
pulsing region is pulsing, the rest experiment (no per-image rule: a
zoomed-in image takes its region's class).  A grid or point takes the flag
and region of the most recent preceding topo (anywhere in the directory,
header geometry) whose footprint contains its position; inside no
footprint the ohmic test decides pulsing (sites.TIP_PARAMS: R² at or above
ohmic_r2 for the current's linear fit through its 0 V value and the
calibrated dI/dV's standard deviation at most ohmic_flat times its mean).
A run of consecutive pulsing files whose last spectra file is followed by
experiment files gets that last file flagged clean.  Phase and noise files
take the preceding file's flag and region.  Every flag is the auto value
until it is changed in the 'site' radio; a change is written to the sites
file.  'prev region' / 'next region' go to the previous / next stop at an
experiment region: the first spectra file ahead (or back) whose region is
an experiment region other than the current file's, a stop entered from
its first spectra file (sites.Sites.neighbour_region).

Drift correction
----------------
Two sources of drift and one model: every file is corrected by one
constant velocity v, the surface's features' displacement per second in
the absolute frame.  The browser carries the 'auto drift' and 'manual
drift' toggles (both on at startup), the 'stripe match' button (grid
files) and the 'drift' button; the Topo, Line, and Map windows have their
own toggles.
  'auto drift'   : the tracker's velocity (drift.py: every consecutive topo
      pair of a segment tracked at startup by feature-point matching, the
      method of Diao et al., Appl. Phys. Lett. 122, 121601 (2023), with the
      KAZE detector; results cached).  A spectra file takes the pair (the
      last topo of its segment before it, the first topo after it); a topo
      its pair with the next topo (the last topo of a segment the pair with
      the previous one); without a usable after pair, the last tracked
      pair's velocity when it is close to the one before it, else no
      correction.  The console names the pair and the rule for every file.
  'manual drift' : the velocity set by hand in the drift window (see
      "Manual drift"), which overrides the auto velocity wherever it covers
      a file and works with the auto drift off.
Turning a toggle off only redraws; the results stay in memory.  Anchors: a
file's starting location is its anchor at its recorded position (the first
pixel of a topo, the first spectrum of a grid, the first sweep of a point);
everything else in the file is placed relative to it by its own time:
spectrum or sweep k at xy_k - v (t_k - t_0) (Sm4File.spectrum_times,
sweep_times), every energy point of a sweep at -v (t - t_sweep start)
relative to the sweep's start, so the drift lines (a marker at every
sweep's start and a straight line through its sweep, Drift.sweep_lines; a
point's sweep starts connected dot to dot) are drawn wherever topo and
spectral locations are shown: the scan area, the Topo window, the Map
window (both panels), and the Line window's stripe, white for an auto
velocity, red for a manual one, and pale yellow for a span confirmed from
seeded markers (plots.drift_color).  Topographies: with
a source on every topo of the loaded set is skewed pixel by pixel by the
drift during its scan (Drift.corrected_topo: pixel (i, j) at its
acquisition time, Sm4File.pixel_times, moved by -v (t - t_first pixel) and
the image resampled onto its own grid), on the scan area, in the Topo
window, and as the Map window's embedded topo.  The grid's own topo stripes
under the markers follow: every pixel at its recorded location minus v
times its own pixel time.  Line window: the topo stripe is never moved;
every row keeps its anchor at its own topo pixel and is sheared along the
position axis by the drift during its own sweep (Drift.row_shear), nothing
along the energy axis.  Map window: every pixel of every energy slice sits
at its spectrum's anchor plus the drift during the sweep up to that energy
(Drift.map_shift).  The Line, Map, and Topo windows carry their own 'auto
drift' and 'manual drift' toggles, set from the browser's states at
opening and independent after; the Line and Map windows also carry 'drift
window' (see "Manual drift"), the Topo window 'stripe match'.
  Stripe matching (grid files; the 'stripe match' button of the browser and
      the Topo window opens the stripe window, stripe.StripeWindow): on top
      of the velocity, the lines' own topo stripes placed against the
      earlier topos (drift_correct_grid: the lines placed by their extrema
      against the topo's profile, the stretch of each pass from its matched
      extrema or from the extrema period ratio, the reverse pass chained to
      the next line), with hand-moved line ends and return points where
      the window's points say so; the slow axis stays v.  Each grid keeps
      one fast-axis state (off, auto, manual) and one set of points per
      slow-axis setting (off, auto, manual: the velocity source in force,
      Drift.slow_setting), so all three can be prepared from the window
      once, and the placement reaches the scan area, the Topo window, the
      Line window's heatmap and the Map window alike (Drift.result with
      stripe None).  It never changes v.
  'anchor to topo' (grid and point files; off by default; the Topo window
      only): the MATLAB "drift for first point": the file is also shifted
      rigidly by -v (t_0 - t_topo start), t_topo the start of the topo under
      it, to where the surface sat when that topo began.  Off, the recorded
      anchor stands.
Segments: a jump of the recorded scan centre above move_min_m (1 um)
between consecutive topos is a long-range move; the topos between two
moves are a segment; pairs are tracked and before/after topos looked up
within a segment only.  Cache: drift_cache_<date_time of the first
file>.json in the experiment directory (format 5: the tracked pairs, the
topo times, the applied spans with their kind and velocity).

Manual drift
------------
The MATLAB script's "Drift controls" (plot_sm4_topo_and_spec_21_...) with
several marker pairs, in the drift window (drift.DriftWindow), opened by the
browser's 'drift' button for the current file and by the Line and Map
windows' 'drift window' button for theirs.  The window shows the pair the
file's auto velocity comes from, side by side at absolute positions: for a
spectra file the last topo of its segment before it and the first after it
(never the file's own topo); for a topo file itself and the next topo of
its segment, else the previous topo and itself; each panel is titled with
the name of the file it draws and also shows the other topo's recorded
frame as a thin black outline on top of everything, both panels on one
view centred on the two frames together, so the pair's overlap is seen at
once.  A click in a panel places
a numbered marker on that topo (draggable, the pixel's acquisition time
re-read at the new place); the k-th marker on the earlier topo pairs with
the k-th on the later one; 'Markers' (1 to 8) is the number of pairs, pair
k in its own colour (plots.MARKER_COLORS: red, green, blue, yellow,
magenta, cyan, orange, purple, the eight chromatic basic colour terms that
read apart on the topo), the earlier marker a circle, the later a square.
Seeding: with the opener's 'auto drift' on and the pair tracked, a marker
clicked on either topo seeds its partner on the other where the auto
velocity puts the same feature (xy + v (t_B - t_A), the partner's own pixel
time, drift.seed_partner; hollow until moved); dragging the clicked marker
re-seeds an unmoved partner, dragging the partner adjusts it (more than
half a pixel off the seeded spot).  The markers are a draft: placing,
dragging, 'Markers', and 'clear markers' change only the draft.  One
velocity box, 'v (nm/h)' ("Vx, Vy", the MATLAB boxes), is what gets
applied.  'set from markers' fills it from the draft and stores nothing:
every pair seeded and unmoved, the auto pair velocity itself (confirmed);
otherwise the least-squares fit over the pairs with their own times, each
pair a two-point measurement d_k = v (t_B,k - t_A,k), v = sum dt_k d_k /
sum dt_k^2 (drift.fit_marker_velocity; with one pair exactly the MATLAB
two-point slope), which models the squash and stretch of the marker
pattern between two scans of different size, speed, or direction.  Typing
in the box only fills it.  'apply velocity' stores the box as the span of
the window's pair, replacing what that pair had: kind 'confirmed' when the
box still holds the unmoved seeded value (source 'auto (confirmed)'),
'adjusted' when it still holds the pairs' fit (source 'manual (markers)'),
'typed' when it was typed or edited (source 'manual (typed)'); an empty
box applied removes the pair's span.  Every pair's dx, dy, dL, dt, v_x,
v_y, v_L, and angle (the MATLAB readout) and its residual go to the
console and the window; arrows from each earlier marker to its partner
are drawn in the pair's colour.  The span covers every file from the
earlier topo to the later one, both included; of overlapping spans the
narrowest containing the file applies; a confirmed span is governed like
any other; a file no span covers takes the auto velocity (its extrapolation
rule included), so only the files between a pair applied by hand keep a
hand velocity.  Applying tells the Drift's subscribers: the scan area and
every open Line and Map window redraw, whichever drift window applied.
Spans (with their kind and velocity) are kept in the drift cache's manual
section, restored at startup, and never touched by the tracking.  The
console names every file's source ('auto (pair)', 'auto (extrapolated)',
'auto (confirmed)', 'manual (markers)', 'manual (typed)', ...), as does
the scan-area title.

Cross section
-------------
'section' places a line across the middle of the current view of the
scan area (pressed again, removes it); its endpoints drag (a press near an
endpoint moves it, elsewhere the pan works as before).  'cross section'
samples the raw topography under the line (topo.cross_section: along the
line at the finest pixel pitch of the topos it crosses, at every sample the
topmost loaded topo containing it, bilinear in its raw page, in metres as
the loader gives it and nothing else, forward and reverse pages as two curves,
NaN where no topo or no acquired pixel lies under the sample; the raw page
at the recorded pixel locations, so the drift toggles do not move the cut)
and shows it in the profile window (topo.ProfileWindow: height in nm
against the distance along the line from the filled endpoint in nm after
the scale factor), the same window type the Line window's 'profile'
buttons open for a line's forward and reverse passes.  Every profile
window carries 'add new' and 'add existing': 'add new' opens a profile plot
(topo.ProfilePlot, profile1, profile2, ...) holding the window's pair, 'add
existing' drops the list of the open profile plots and adds the pair to the
chosen one; a profile plot overlays its pairs in one axes as they are, one
colour per pair, the forward pass solid and the reverse dashed, the legend
naming each pair's file, line and row, or cross section.

Grid stripes
------------
For the current grid its own topo pixels are drawn as bands under the
markers, each pixel from the midpoint to its predecessor to the midpoint
to its successor, one slow step wide (plots.draw_grid_stripes,
topo.grid_stripe_layers).  With no velocity covering the grid the 'topo
pass' radio chooses the pass at the recorded pixels: 'forward' the forward
row of every line at the kept spectra's pixels (STS.acquired: a spectrum
whose column is all zeros or ends in zeros is not kept, so an unfinished
line stops at its last kept spectrum), 'reverse' the reverse row along the
tip's way back, 'average' the two passes range-matched and averaged
pixelwise at the forward pixels.  With a velocity covering the grid both
passes are always drawn at their corrected positions, the reverse band
under the forward one (drift.Drift.grid_chain: the forward pixels by their
points' times within the line, the reverse row rigid at the forward end so
it starts there and ends at the return point, the next line at the
recorded displacement from the return point).  Each pass is processed by
the active topo processor and coloured on its own scale (its own min to
max), as the topo images are.
"""


if __package__ in (None, ''):
    # Run as a file (Spyder F5): make the package importable from wherever its folder sits.
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import io
import time
import contextlib
import textwrap
import weakref
from collections import deque
import numpy as np
from sm4_Browser import qtplot as plt
from sm4_Browser import fit, widgets, plots
from sm4_Browser.sm4file import (Sm4File, list_sm4_files, header_cache, file_duration_s, kind_from_types, filename_stamp,
                                 stamp_text, geom_contains)
from sm4_Browser.sts import STS, curve_units
from sm4_Browser.averaging import Averages, Selection
from sm4_Browser.records import SavedAveragesWindow
from sm4_Browser.feenstra import feenstra_width, feenstra_titles, energy_has_zero, iv_corrected, iv_broadened
from sm4_Browser.normalization import normalize_abs
from sm4_Browser.topo import (TOPO_CMAP, process_topo, processed_topo, processor_labels, processor_defaults,
                              TopoWindow, ProfileWindow, ProfilePlots, cross_section, grid_stripe_layers)
from sm4_Browser.drift import DRIFT_PARAMS, Drift, DriftWindow
from sm4_Browser.sites import Sites, FLAGS
from sm4_Browser.line import Line
from sm4_Browser.lineavg import LineAverages, member_from_file
from sm4_Browser.map import Map
from sm4_Browser.scanlayer import (ScanLayer, display_record, record_corners, SHOW_MODES, DEFAULT_SHOW_MODE, cache_dir,
                                   cache_load, cache_store, read_record)
from collections import OrderedDict
from types import SimpleNamespace


# Every loaded single point on the scan area: cyan diamond (the current point and its sweeps use
# the colours of plots.draw_point_sweeps).
POINT_LOADED_COLOR = '#00e5ff'      # every loaded single point: cyan diamond (before the time colours)
POINT_BEFORE_COLOR = '#9e9e9e'      # a spectra file taken before the current topo: grey diamond (MATLAB)
POINT_AFTER_COLOR = '#1f4fff'       # a spectra file taken after it: blue diamond (MATLAB)


class _BrowserCore:
    """The experiment logic (files, drift, sites, averages, the analysis windows) and the drawing of the panels; Browser below is its window."""
    WINDOW = 25
    TOPO_CBAR = [0.565, 0.40, 0.012, 0.35]        # the page's topo colour bar (plots.draw_topo_cbar)

    def __init__(self, directory, scalefactor=2.75, hampel_defaults=None, sgolay_defaults=None,
                 line_window_kwargs=None, cmap_map='jet', drift_params=None,
                 iv_broadening_V=0.2, tip_params=None):
        """
        directory         : folder of .sm4 files of one experiment
        scalefactor       : initial x/y scale factor (editable in the window)
        hampel_defaults, sgolay_defaults : pre-fill of the filter boxes in the
                            browser and the analysis windows (None: the FilterControls defaults);
                            every filter starts off
        line_window_kwargs: keyword arguments for plot_line_with_topo_interactive (plus 'cmap');
                            'topo_lines' is set by the browser to the selected line (forward and reverse)
        cmap_map          : colormap of the dI/dV maps
        drift_params      : overrides of DRIFT_PARAMS (the tracker, the baseline, the stripe matching)
        iv_broadening_V   : broadening width (V) of I/V for the Feenstra division (divide_by_iv);
                            0 gives the plain ratio
        tip_params        : overrides of sites.TIP_PARAMS (the ohmic test)
        """
        self.directory = directory
        self.files = list_sm4_files(directory)
        if not self.files:
            raise ValueError(f"No .sm4 files with a date-time stamp in {directory}.")
        self.scalefactor = float(scalefactor)
        self.hampel_defaults = hampel_defaults
        self.sgolay_defaults = sgolay_defaults
        self.line_window_kwargs = dict(line_window_kwargs or {})
        self.cmap_map = cmap_map
        self.drift_params = dict(DRIFT_PARAMS)
        self.drift_params.update(drift_params or {})
        self.auto_on = True        # the 'auto drift' toggle starts on; off only redraws nominal positions
        self.manual_on = True      # the 'manual drift' toggle starts on (the manual velocities cover what they cover)
        self.stripe_window = None  # the stripe window opened by 'stripe match' (stripe.StripeWindow)
        self.anchor_on = False     # 'anchor to topo' is the Topo window's toggle; the browser never applies it (off there)
        self.profile_window = None # the cross-section profile window (topo.ProfileWindow)
        self.profile_plots = ProfilePlots()   # the profile plots (topo.ProfilePlots): the profile windows' 'add' buttons
        self.drift_window = None   # the manual drift window opened by 'drift' (drift.DriftWindow)
        self.topo_dir = {}         # topo path -> 'forward' | 'reverse' | 'average' (what it was left on)
        self.channel = 'didv'      # the 'channel' radio: dI/dV or IETS in the spectra panel, the Line, the Map, a new average
        # Every file's type and every topo's scan centre from a header-only read at startup
        # (sm4_header_info), so that the spectra slider knows the spectra files and the
        # long-range moves are known from the recorded locations before any file is loaded.
        # The header scan goes through the on-disk header cache (sm4file.header_cache): only new or
        # changed files are opened; the rest come from headers_<stamp>.json in the directory.
        self.kinds = {}
        self.head_geoms = {}       # topo path -> header geometry (center_m, scales, sizes), see sm4_header_info
        self.headers = header_cache(directory, self.files)   # path -> types, geom, start/end time
        for p in self.files:
            h = self.headers[p]
            self.kinds[p] = kind_from_types(h['types'], h['line_types'])
            if self.kinds[p] == 'topo' and h['geom'] is not None:
                self.head_geoms[p] = h['geom']
        self.geoms = {}            # path -> topo footprint record, learned on first load
        self.loaded = {}           # path -> Sm4File (the window, plus the extra topos described in _ensure_window)
        # The drift correction: the segments of the directory, the pair cache, the per-file
        # results and the on-disk drift cache (Drift; its results are read with self.drift.result).
        self.drift = Drift(directory, self.files, self.kinds, self.head_geoms, self.drift_params, self._load)
        # Every consecutive topo pair of every segment tracked (Drift.build; cached).
        self.drift.build()
        # Every manual change of the drift (a span applied or cleared in a drift window), made in
        # any drift window, redraws the scan area (Drift.subscribe; the Line and Map windows
        # subscribe their own redraws when they open).
        self.drift.subscribe(self._on_manual_changed)
        # The sites and regions (sites.Sites: the once-over metrics cache, the regions, the flags).
        self.sites = Sites(directory, self.files, self.kinds, self.head_geoms, self.drift.seg_first, self._load,
                           tip_params)
        self.sites.describe()
        self.spectra_files = [i for i, p in enumerate(self.files) if self.kinds[p] in ('grid', 'point', 'phase', 'fft')]
        self.spectra_pos = 0       # index into spectra_files of the spectra file last shown
        self.index = 0
        self.line_index = 0
        self.topo_active = processor_defaults()
        self.iv_broadening = float(iv_broadening_V)
        # The Feenstra settings (the '0Vsub', 'broadened', and width boxes under the spectra panels,
        # shown for dI/dV spectra): the current's value at 0 V subtracted before I/V is formed, and I/V
        # broadened by iv_broadening (off at start: the plain I/V).  They act on the browser's own
        # spectra panels (the four Feenstra panels, the division always on here) and are handed to
        # the Line, Map, averaging, and line-average windows at opening.
        self.iv_zero_sub = True
        self.iv_broadened = False
        self.children = []         # open analysis windows (widgets stay referenced)
        # The averages of the experiment: the registry of the open averaging windows (by index) and
        # the one averaging method; handed to the Line, Map, and line-average windows for their 'add' buttons.
        self.averages = Averages(directory, self.hampel_defaults, self.sgolay_defaults,
                                 fit_settings=self.line_window_kwargs.get('fit_settings'),
                                 divide_iv=True, iv_broadening=self.iv_broadening, channel=self.channel,
                                 iv_zero_sub=self.iv_zero_sub, iv_broadened=self.iv_broadened)
        # The line averages: the registry of the open line-average windows, handed to the Line windows.
        self.line_averages = LineAverages(directory, self.hampel_defaults, self.sgolay_defaults,
                                          fit_settings=self.line_window_kwargs.get('fit_settings'),
                                          divide_iv=True, iv_broadening=self.iv_broadening, channel=self.channel,
                                          iv_zero_sub=self.iv_zero_sub, iv_broadened=self.iv_broadened,
                                          cmap=self.line_window_kwargs.get('cmap', 'jet'), averages=self.averages)
        self.saved_window = None   # the saved averages window (SavedAveragesWindow)
        self._picker = None        # the 'add existing' list (widgets.AveragePicker), made with the figure
        # The persistent scan layer (scanlayer.ScanLayer, made with the figure), its 'show' mode, and the
        # background placing of the files not yet read (_bg_schedule / _bg_tick).
        self.show_mode = DEFAULT_SHOW_MODE
        self._bg_queue = deque()
        self._bg_running = False
        self._bg_pool = None               # the worker processes (made at the first need; False: not available)
        self._bg_futures = {}              # path -> future of a record being read by a worker
        self._scan_cache = cache_dir(directory, self.files)
        self._lru = OrderedDict()          # path -> Sm4File: the most recently used full files beyond the window
        self._layer_reported = set()
        self._point_artists = []
        # The step filter (the 'step through' box): the kinds Prev / Next and the sliders stop at, and the
        # shortest duration (min) a file must have; a file of unknown duration always passes.
        self.step_kinds = {'topo', 'point', 'grid', 'phase', 'fft', 'other'}
        self.min_duration_min = 0.0
        self._build_figure()
        self._goto(0)
        plt.show()

    # --- loading and the window ---

    LRU_MAX = 40                   # full files kept beyond the loaded set (stepping back re-reads nothing)

    def _load(self, path):
        f = self._lru.get(path)
        if f is not None:
            self._lru.move_to_end(path)
            return f
        with plt.watchdog(f"loading {path}"):
            f = Sm4File(path)
        self._lru[path] = f
        while len(self._lru) > self.LRU_MAX:
            self._lru.popitem(last=False)
        self.kinds[path] = f.kind
        if f.kind == 'topo':
            self.geoms[path] = f.geometry()
        if f.self_check_text:
            print(f.self_check_text)
        return f

    @staticmethod
    def _point_xy(f):
        """Absolute position (metres) of a single-point or phase rotation file, or None."""
        if f.kind in ('point', 'phase') and len(f.spec_xy_m):
            return f.spec_xy_m[0]
        return None

    def _ensure_window(self):
        i = self.index
        wanted = self.files[max(0, i - (self.WINDOW - 1)): i + 1]
        new = {}
        for p in wanted:
            new[p] = self.loaded.get(p) or self._load(p)
        if not any(f.kind == 'topo' for f in new.values()):
            j = i - self.WINDOW
            while j >= 0:
                p = self.files[j]
                if self.kinds[p] == 'topo':
                    new[p] = self.loaded.get(p) or self._load(p)
                    break
                j -= 1
        # A single point whose position lies under none of the loaded topos: the most
        # recent preceding topo whose footprint contains it is loaded as one more file
        # (drawn at its own time depth in the stack).  Every file's type is known from
        # the startup scan; a topo's footprint is learned when it is first loaded.
        xy = self._point_xy(new[self.files[i]])
        self.covering_note = None
        if xy is not None and not any(f.kind == 'topo' and f.contains_m(xy) for f in new.values()):
            j = i - 1
            found = None
            while j >= 0:
                p = self.files[j]
                if p in new or self.kinds[p] != 'topo':
                    j -= 1
                    continue
                if p not in self.geoms:
                    f = self._load(p)                              # footprint discovery
                    if f.contains_m(xy):
                        found = (p, f)
                        break
                elif geom_contains(self.geoms[p], xy):
                    found = (p, self.loaded.get(p) or self._load(p))
                    break
                j -= 1
            if found is not None:
                new[found[0]] = found[1]
                self.covering_note = f"covering topo loaded: {found[1].name}"
            else:
                self.covering_note = "no preceding topo contains this position"
                print(f"{new[self.files[i]].name}: {self.covering_note}.")
        self.loaded = new

    @property
    def current(self):
        return self.loaded[self.files[self.index]]

    def _topos(self):
        """Topography files of the loaded set, oldest first."""
        return sorted((f for f in self.loaded.values() if f.kind == 'topo'), key=lambda f: f.stamp)

    def _most_recent_topo(self):
        t = self._topos()
        return t[-1] if t else None

    def _topo_for(self, f):
        """
        The topo to show with file f: for a single point, the most recent loaded
        topo whose footprint contains its position (the most recent loaded topo
        when none does, said on the console); otherwise the most recent loaded topo.
        """
        xy = self._point_xy(f)
        if xy is not None:
            covering = [t for t in self._topos() if t.contains_m(xy)]
            if covering:
                return covering[-1]
            print(f"{f.name}: no loaded topo contains this position; using the most recent loaded topo.")
        return self._most_recent_topo()

    # --- figure ---

    def _set_controls_for_kind(self, kind):
        grid = kind == 'grid'
        point = kind == 'point'
        self._popup_close()
        for w in (self.btn_line, self.btn_map):
            w.show(grid)
        self.chk_auto.show(True)
        self.chk_manual.show(True)
        self.btn_stripe.show(grid and bool(self.current.topo_pages) and self.drift is not None)   # the Topo window's rule
        self.rad_channel.show(grid or point)
        self.btn_add_new.show(point)
        self.btn_add_existing.show(point)
        self.txt_in_avgs.set_visible(point or grid)     # a grid file: the averages holding a marker group of it
        if point or grid:
            self._update_avg_text()
        # The direction list shows the stored choice of the topo under the current spectra.
        target = self._topo_under(self.current)
        has = target is not None
        self.rad_dir.show(has)
        self.txt_dir.set_visible(has)
        if has:
            self.txt_dir.set_text("line/map topo" if target.kind == 'grid' else f"topo under: {stamp_text(target.stamp)[11:19]}")
            self.rad_dir.set(self._direction_of(target))
        # The site radio shows the current file's flag.
        self.rad_site.set(self.sites.flag(self.files[self.index]))

    def _direction_of(self, topo):
        return self.topo_dir.get(topo.path, 'forward')

    def _topo_under(self, f):
        """
        The topo the direction list acts on: the file itself when it is a
        topo or a grid (a grid's own topo, the stripes under its markers);
        otherwise the most recent loaded topo containing the file's first
        spectrum position, or the most recent loaded topo when none does.
        """
        if f.kind in ('topo', 'grid'):
            return f
        if len(f.spec_xy_m):
            covering = [t for t in self._topos() if t.contains_m(f.spec_xy_m[0])]
            if covering:
                return covering[-1]
        return self._most_recent_topo()

    # --- navigation and redraw ---

    def _goto(self, i):
        i = int(np.clip(i, 0, len(self.files) - 1))
        # The file about to be shown, written to the crash log first: should Qt itself fail on a
        # file, qtplot.CRASH_LOG names it.
        with plt.watchdog(f"going to file {i + 1}: {self.files[i]}"):
            self.index = i
            if i in self.spectra_files:
                self.spectra_pos = self.spectra_files.index(i)
            self._ensure_window()
            self.line_index = 0
            self._draw_all()

    def _goto_spectra(self, k):
        """Go to the k-th spectra file (point, grid, phase, or fft) of the directory."""
        if not self.spectra_files:
            return
        k = int(np.clip(k, 0, len(self.spectra_files) - 1))
        self._goto(self.spectra_files[k])

    def _on_region_step(self, d):
        """'prev region' / 'next region': the previous / next stop at an experiment region (Sites.neighbour_region)."""
        i = self.sites.neighbour_region(self.files[self.index], d)
        if i is None:
            print(f"regions: no {'next' if d > 0 else 'previous'} experiment region with spectra files.")
            return
        self._goto(i)
        r = self.sites.region(self.sites.region_of(self.files[i]))
        if r is not None:
            print(f"region {r['index']}: {len(r['topos'])} topos from {os.path.basename(r['first'])}, {r['flag']}; "
                  f"at {os.path.basename(self.files[i])}")

    def _draw_all(self):
        f = self.current
        self._set_controls_for_kind(f.kind)
        # Both sliders follow the file shown: the file slider the current file, the
        # spectra slider the spectra file last shown (unchanged while a topo is current).
        self.sld_file.set(self.index)
        if self.spectra_files:
            self.sld_spec.set(self.spectra_pos)
        self._nav_text(self.index)
        self._spec_nav_text(self.spectra_pos)
        # Each panel on its own: a file whose recording is incomplete in an unforeseen way (a scan
        # stopped before it began, positions not recorded) leaves that panel empty with the reason
        # on the console, and the browser goes on.
        for step in (self._draw_scan_area, self._draw_spectra, self._draw_info):
            try:
                step()
            except Exception as exc:
                import traceback
                print(f"{f.name}: {step.__name__.lstrip('_').replace('_', ' ')} not drawn ({exc!r}).")
                traceback.print_exc()
        self.fig.canvas.draw_idle()

    def _drift_active(self):
        return self.auto_on or self.manual_on

    def _topo_image(self, t):
        """The processed image of a loaded topo (nm): skewed pixel by pixel by the drift with a source on."""
        if self._drift_active():
            try:
                return self.drift.corrected_topo(t, self._direction_of(t), self.topo_active, self.loaded,
                                                 self.auto_on, self.manual_on) * 1e9
            except Exception as exc:
                print(f"{t.name}: drift correction of the image not applied ({exc}).")
        return processed_topo(t, self._direction_of(t), self.topo_active) * 1e9

    @staticmethod
    def _box_of(points):
        """(x0, x1, y0, y1) of a set of points (n, 2), or None without a finite one."""
        p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        ok = np.all(np.isfinite(p), axis=1)
        if not ok.any():
            return None
        p = p[ok]
        return float(p[:, 0].min()), float(p[:, 0].max()), float(p[:, 1].min()), float(p[:, 1].max())

    # --- the persistent scan layer (scanlayer.ScanLayer): every file placed once, shown by the mode ---

    def _direction_of_path(self, path):
        return self.topo_dir.get(path, 'forward')

    def _layer_key(self, path):
        return (self._direction_of_path(path), tuple(bool(a) for a in self.topo_active))

    def _needs_record(self, path):
        """True when the layer lacks the file's record for the current pass and processors (a spectra file: any record)."""
        kind = self.kinds.get(path)
        rec = self.layer.records.get(path)
        if kind in ('topo', 'grid'):
            if rec is None:
                return True
            return 'img' in rec and rec.get('key') != self._layer_key(path) or ('img' not in rec and kind == 'topo'
                                                                                  and not rec.get('_empty'))
        if kind in ('point', 'phase'):
            return rec is None
        return False

    def _record_of_loaded(self, f):
        """The layer's record of a loaded file, made now when missing or stale (no file read)."""
        if not self._needs_record(f.path):
            return
        rec = display_record(f, self._direction_of_path(f.path), self.topo_active)
        if rec is None:
            rec = {'path': f.path, 'kind': f.kind, 'name': f.name, '_empty': True}
        self.layer.add(rec)

    def _visible_paths(self):
        """The files the 'show' mode shows, in time order: up to the current one, all, or the loaded set."""
        i = self.index
        if self.show_mode == 'all':
            return list(self.files)
        if self.show_mode == 'up to current':
            return self.files[:i + 1]
        return [p for p in self.files if p in self.loaded]

    def _layer_model(self, path, rec):
        """The drift model a placed file is drawn with (None with both drift sources off or without a velocity)."""
        if not self._drift_active():
            return None
        try:
            if rec.get('kind') == 'grid':
                model, notes = self.drift.model_for_spectra(SimpleNamespace(path=path, start_time=rec.get('start_time', 0.0)),
                                                            self.auto_on, self.manual_on)
            else:
                model, notes = self.drift.model_for_topo(path, self.auto_on, self.manual_on)
        except Exception as exc:
            print(f"{rec.get('name', path)}: no drift skew ({exc}).")
            return None
        rk = (path, self.auto_on, self.manual_on, id(model))
        if path in self.loaded and rk not in self._layer_reported:
            self._layer_reported.add(rk)
            name = rec.get('name', path)
            if model is None:
                print(f"{name}: no drift skew (" + "; ".join(notes) + ")")
            else:
                print(f"{name}: skewed with the {model.rule} velocity, " + model.text())
        return model

    def _drift_state(self):
        """A token of everything the placed images' positions depend on: the drift sources, the manual spans, the passes, the processors."""
        return (self.auto_on, self.manual_on, getattr(self, '_drift_version', 0),
                tuple(sorted(self.topo_dir.items())), tuple(bool(a) for a in self.topo_active))

    def _clear_overlays(self):
        """Every artist of the scan area except the layer's images (the per-file overlays are redrawn)."""
        ax = self.ax_scan
        keep = set(id(a) for a in self.layer.artists())
        if getattr(ax, '_legend', None) is not None:
            ax._remove_legend()
        for a in list(ax._children):
            if id(a) not in keep:
                try:
                    a.remove()
                except Exception:
                    pass
        for t in list(getattr(ax, '_axes_texts', [])):
            try:
                t.remove()
            except Exception:
                pass

    def _point_colours(self, visible):
        """
        The other spectra files' positions in two colours (MATLAB's rule): grey for those taken before
        the most recent topo or grid at or before the current file, blue after it; drawn as diamonds.
        """
        i = self.index
        last_img = max((j for j in range(i + 1) if self.kinds.get(self.files[j]) in ('topo', 'grid')), default=-1)
        grey, blue = [], []
        for p, xy in self.layer.points(visible, self.files[i]):
            (blue if self.layer.rank.get(p, 0) > last_img else grey).append(xy)
        return np.array(grey).reshape(-1, 2), np.array(blue).reshape(-1, 2)

    def _draw_points(self, visible):
        """The other spectra files' diamonds (grey before the current topo, blue after); returns their drawing positions."""
        for a in self._point_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._point_artists = []
        k = 1e9 / self.scalefactor
        out = []
        grey, blue = self._point_colours(visible)
        for xy_m, colour in ((grey, POINT_BEFORE_COLOR), (blue, POINT_AFTER_COLOR)):
            if len(xy_m):
                oxy = xy_m * k
                out.append(oxy)
                h, = self.ax_scan.plot(oxy[:, 0], oxy[:, 1], ls='none', marker='D', ms=7, color=colour,
                                       mec="black", mew=0.6, zorder=1.9)
                self._point_artists.append(h)
        return out

    # --- the background placing of the files not yet read ---

    BG_BUDGET_S = 0.04             # reading per event-loop turn: the window stays responsive while files are placed

    def _bg_schedule(self):
        """Queue the shown files the layer has no record of, nearest the current file first; start the placing."""
        i = self.index
        need = [p for p in self._visible_paths() if p not in self.loaded and self._needs_record(p)
                and p not in self._bg_futures]
        need.sort(key=lambda p: abs(self.layer.rank.get(p, 0) - i))
        self._bg_queue = deque(need)
        if (need or self._bg_futures) and not self._bg_running:
            self._bg_running = True
            QtCore.QTimer.singleShot(0, self._bg_tick)
        self._bg_status()

    def _bg_status(self):
        n = len(self._bg_queue) + len(self._bg_futures)
        bar = getattr(self, '_status_label', None)
        if bar is not None:
            vis = self._visible_paths()
            placed = sum(1 for p in vis if not self._needs_record(p))
            bar.setText(f"scan area: {placed} / {len(vis)} files placed" + (f", placing {n} more" if n else ""))

    BG_WORKERS = int(os.environ.get('SM4_BROWSER_WORKERS', '4'))   # worker processes reading files (0: in this thread)
    BG_POOL_INIT = None            # (initializer, args) run in every worker first (the tests' synthetic files)

    def _bg_pool_get(self):
        """The worker processes (spawned once; False when they cannot start: the reading stays in this thread)."""
        if self._bg_pool is None:
            if self.BG_WORKERS <= 0:
                self._bg_pool = False
            else:
                try:
                    import multiprocessing
                    from concurrent.futures import ProcessPoolExecutor
                    init, args = self.BG_POOL_INIT if self.BG_POOL_INIT else (None, ())
                    self._bg_pool = ProcessPoolExecutor(max_workers=self.BG_WORKERS,
                                                        mp_context=multiprocessing.get_context('spawn'),
                                                        initializer=init, initargs=args)
                except Exception as exc:
                    print(f"scan area: worker processes not available ({exc}); files are read in this thread.")
                    self._bg_pool = False
        return self._bg_pool

    def _bg_place(self, p, kind, geom, rec, err, from_cache=False):
        """A record arrived (a worker, this thread, or the cache): kept, written to the cache, placed."""
        if kind is not None:
            self.kinds[p] = kind
        if geom is not None:
            self.geoms[p] = geom
        if err is not None:
            print(f"{os.path.basename(p)}: not placed on the scan area ({err}).")
        if rec is None:
            rec = {'path': p, 'kind': kind or self.kinds.get(p), '_empty': True, 'name': os.path.basename(p)}
        elif not from_cache:
            cache_store(self._scan_cache, p, self._direction_of_path(p), self.topo_active, rec)
        self.layer.add(rec)

    def _bg_tick(self):
        """
        One turn of the background placing (at most BG_BUDGET_S of work): records from the disk cache first
        (scanlayer.cache_load, no file read), the rest read by the worker processes (or, without them, in this
        thread), every finished record placed; continues on the next event-loop turn until the queue is empty.
        """
        t_end = time.perf_counter() + self.BG_BUDGET_S
        added = 0
        pool = self._bg_pool_get()
        # Finished workers.
        for p, fut in list(self._bg_futures.items()):
            if fut.done():
                del self._bg_futures[p]
                try:
                    res = fut.result()
                except Exception as exc:
                    print(f"scan area: a worker stopped ({exc}); the rest is read in this thread.")
                    self._bg_pool = pool = False
                    self._bg_queue.appendleft(p)
                    continue
                if self._needs_record(p):
                    if res[4] is not None and res[1] is None:
                        # The worker could not read the file (another environment, a file still being
                        # written): one more try here before it is given up.
                        res = read_record(p, self._direction_of_path(p), self.topo_active)
                    self._bg_place(*res)
                    added += 1
        # New work: the cache, else a worker (a few in flight per worker), else this thread.
        while self._bg_queue and time.perf_counter() < t_end:
            p = self._bg_queue[0]
            if not self._needs_record(p) or p in self._bg_futures:
                self._bg_queue.popleft()
                continue
            direction = self._direction_of_path(p)
            rec = cache_load(self._scan_cache, p, direction, self.topo_active)
            if rec is not None:
                self._bg_queue.popleft()
                self._bg_place(p, rec.get('kind'), None, rec, None, from_cache=True)
                added += 1
                continue
            if pool:
                if len(self._bg_futures) >= 3 * self.BG_WORKERS:
                    break
                self._bg_queue.popleft()
                try:
                    self._bg_futures[p] = pool.submit(read_record, p, direction, list(self.topo_active))
                except Exception as exc:
                    print(f"scan area: worker processes stopped ({exc}); files are read in this thread.")
                    self._bg_pool = pool = False
                    self._bg_queue.appendleft(p)
                continue
            self._bg_queue.popleft()
            if p in self.loaded:
                f = self.loaded[p]
                rec = display_record(f, direction, self.topo_active)
                self._bg_place(p, f.kind, f.geometry() if f.kind == 'topo' else None, rec, None)
            else:
                self._bg_place(*read_record(p, direction, self.topo_active))
            added += 1
        if added:
            self._layer_refresh()
        if self._bg_queue or self._bg_futures:
            QtCore.QTimer.singleShot(0 if not self._bg_futures else 15, self._bg_tick)
        else:
            self._bg_running = False
        self._bg_status()

    def _bg_shutdown(self):
        """The worker processes stopped (the window closed)."""
        pool, self._bg_pool = self._bg_pool, False
        self._bg_queue.clear()
        if pool:
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

    def _layer_refresh(self):
        """The layer and the diamonds updated for the current state (no overlay redraw); the full extent grows with them."""
        k = 1e9 / self.scalefactor
        visible = self._visible_paths()
        # While files are still arriving the occlusion is not recomputed (every arrival would redo it);
        # the last refresh, with the queue empty, computes it once.
        corners = self.layer.update(visible, self.current.path, k, self._direction_of_path,
                                    lambda p, rec: self._layer_model(p, rec), wanted=self._layer_key,
                                    state=None if self._bg_queue else self._drift_state())
        pts = list(corners) + self._draw_points(visible)
        box = self._box_of(np.vstack(pts)) if pts else None
        if box is not None and self.view.full is not None:
            x0, x1, y0, y1 = box
            fx0, fx1, fy0, fy1 = self.view.full
            self.view.full = self.view.fit_to_box((min(x0, fx0), max(x1, fx1), min(y0, fy0), max(y1, fy1)))
        self.fig.canvas.draw_idle()

    def _on_show_mode(self, label):
        """The 'show' radio: which files the scan area shows (scanlayer.SHOW_MODES)."""
        if label not in SHOW_MODES or label == self.show_mode:
            return
        self.show_mode = label
        print(f"scan area: show {label}.")
        self._draw_scan_area()
        self.fig.canvas.draw_idle()

    # --- the step filter ---

    def _allowed(self, i):
        """Whether file i passes the step filter: its kind is ticked and it lasted at least the minimum (unknown passes)."""
        p = self.files[i]
        if self.kinds.get(p, 'other') not in self.step_kinds:
            return False
        if self.min_duration_min > 0:
            d = file_duration_s(self.headers.get(p))
            if d is not None and d < 60.0 * self.min_duration_min:
                return False
        return True

    def _step(self, d):
        """Prev / Next (and Ctrl+Left / Ctrl+Right): the next file in that direction that passes the step filter."""
        j = self.index + d
        while 0 <= j < len(self.files):
            if self._allowed(j):
                self._goto(j)
                return
            j += d
        print(f"step filter: no {'later' if d > 0 else 'earlier'} file passes the filter.")

    def _nearest_allowed(self, i):
        """The allowed file nearest index i (i itself when allowed; ties to the earlier); None when none passes."""
        for r in range(len(self.files)):
            for j in (i - r, i + r):
                if 0 <= j < len(self.files) and self._allowed(j):
                    return j
        return None

    def _step_spectra(self, d):
        """'< spectra' / 'spectra >' (and Ctrl+PgUp / PgDown): the next spectra file that passes the step filter."""
        k = self.spectra_pos + d
        while 0 <= k < len(self.spectra_files):
            if self._allowed(self.spectra_files[k]):
                self._goto_spectra(k)
                return
            k += d
        print(f"step filter: no {'later' if d > 0 else 'earlier'} spectra file passes the filter.")

    def _on_step_kinds(self, label, states):
        names = ['topo', 'point', 'grid', 'phase', 'fft']
        self.step_kinds = {n for n, on in zip(names, states) if on} | {'other'}
        print("step filter: " + (", ".join(n for n in names if n in self.step_kinds) or "no kind") + ".")

    def _on_min_duration(self, text):
        try:
            v = float(text)
            if v < 0:
                raise ValueError("must be >= 0")
        except ValueError as exc:
            print(f"min duration '{text}' not accepted ({exc}); keeping {self.min_duration_min:g} min.")
            self.box_min_dur.set(f"{self.min_duration_min:g}")
            return
        self.min_duration_min = v
        print(f"step filter: files shorter than {v:g} min are skipped" if v > 0 else "step filter: no minimum duration.")

    def _draw_scan_area(self):
        ax = self.ax_scan
        self._clear_overlays()
        f = self.current
        k = 1e9 / self.scalefactor
        pts = []
        focus = None                            # the current file's own extent: the view relocates to it
        for g in self.loaded.values():
            self._record_of_loaded(g)
        visible = self._visible_paths()
        corners = self.layer.update(visible, f.path, k, self._direction_of_path,
                                    lambda p, rec: self._layer_model(p, rec), wanted=self._layer_key,
                                    state=self._drift_state())
        pts.extend(corners)
        rec_f = self.layer.records.get(f.path)
        if f.kind == 'topo' and rec_f is not None and 'img' in rec_f:
            focus = self._box_of(record_corners(rec_f, k, self._direction_of_path(f.path),
                                                self._layer_model(f.path, rec_f)))
        topos = [p for p in visible if self.kinds.get(p) == 'topo' and self.layer.has(p, self._layer_key(p))
                 and 'img' in self.layer.records[p]]
        # Every other spectra file shown: grey before the current topo, blue after it (MATLAB's colours).
        pts.extend(self._draw_points(visible))
        self._bg_schedule()
        corrected = False
        d = None
        source = None
        if f.kind in ('grid', 'point', 'phase') and len(f.spec_xy_m):
            xy = f.spec_xy_m * k
            lines = None
            stripe = (f.kind == 'grid' and self.drift is not None
                      and self.drift.fast_state(f, None, self.auto_on, self.manual_on) != 'off')
            if (self._drift_active() or stripe) and f.kind != 'phase':
                # The grid's stored fast-axis state (the stripe window) places the lines along the fast
                # axis, with both drift sources off as well (drift.Drift.result with a zero velocity).
                d = self.drift.result(f, self.loaded, None, self.auto_on, self.manual_on, self.anchor_on)
                if d is not None:
                    xy = np.where(np.isfinite(d['xy_corr']), d['xy_corr'], f.spec_xy_m) * k
                    # Every sweep's drift line: the line through its sweep (Drift.sweep_lines: the line's
                    # forward compression along a grid line with stripe matching, the velocity elsewhere).
                    L = self.drift.sweep_lines(f, self.loaded, None, self.auto_on, self.manual_on, self.anchor_on)
                    lines = L * k if L is not None else np.stack([d['xy_corr'], d['xy_end']], axis=1) * k
                    corrected = True
                    source = d.get('source')
        line_color = plots.drift_color(source)
        if f.kind in ('grid', 'point', 'phase') and len(f.spec_xy_m):
            pts.append(xy)
            focus = self._box_of(xy)
            if f.kind == 'grid':
                acq = f.acquired[:len(xy)]
                # The grid's own topo stripes under the markers, every pixel at its own position:
                # its recorded location from the page geometry (Sm4File.pixel_xy_m) with the
                # toggle off, the placement's corrected position (Drift.result 'pix') with it on;
                # a forward pixel is drawn only when its spectrum (found by location) is kept, the
                # reverse page's acquired pixels all; the band is one recorded slow step wide.
                slow_step = ((f.spec_xy_m[f.n_fast] - f.spec_xy_m[0]) * k
                             if f.n_lines > 1 and len(f.spec_xy_m) > f.n_fast else None)
                if f.topo_pages:
                    # The grid's own topo (topo.grid_stripe_layers): the pass the 'topo pass' radio chose at
                    # the recorded positions without a velocity, both passes at the chained positions with one.
                    pix = d['pix'] if (corrected and d is not None and d.get('pix') and d['pix'].get('forward') is not None) else None
                    try:
                        main, second, pos_m, pos_s = grid_stripe_layers(f, self._direction_of(f), self.topo_active, pix)
                    except Exception as exc:
                        print(f"{f.name}: the grid's topo pixels not drawn ({exc}).")
                        main = None
                    if main is not None:
                        plots.draw_grid_stripes(ax, f, main, second, pos_m * k, None if pos_s is None else pos_s * k,
                                                float(np.linalg.norm(slow_step)) if slow_step is not None else None,
                                                top='reverse' if self._direction_of(f) == 'reverse' else 'forward')
                ax.plot(xy[acq, 0], xy[acq, 1], ls='none', marker='o', ms=2.5, color='cyan',
                        mec='black', mew=0.3, label=f'spectra ({f.n_acquired} kept)')
                sel = np.zeros(len(xy), dtype=bool)
                sel[f.line_columns(self.line_index)] = True
                sel &= acq
                ax.plot(xy[sel, 0], xy[sel, 1], ls='none', marker='o', ms=4, color='red',
                        mec='black', mew=0.3, label=f'line {self.line_index}')
                # The first spectrum of the highlighted line: the start, unlike any other marker.
                first = np.where(sel)[0]
                if len(first):
                    k0 = first[0]
                    ax.plot(xy[k0, 0], xy[k0, 1], ls='none', marker='s', ms=9, color='white',
                            mec='black', mew=1.2, label=f'start of line {self.line_index}')
                if lines is not None:
                    # Every kept spectrum's drift line through its own sweep.
                    plots.draw_drift_lines(ax, lines[acq], markers=False, color=line_color)
                ax.legend(loc='upper right', frameon=False, fontsize=8)
            else:
                plots.draw_point_sweeps(ax, xy, corrected, lines=lines, color=line_color)
        if pts:
            allp = np.vstack(pts)
            x0, x1 = float(np.min(allp[:, 0])), float(np.max(allp[:, 0]))
            y0, y1 = float(np.min(allp[:, 1])), float(np.max(allp[:, 1]))
            span = max(x1 - x0, y1 - y0)
            if span < 1.0:                      # a lone point: a 10 nm window around it
                cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
                x0, x1, y0, y1 = cx - 5.0, cx + 5.0, cy - 5.0, cy + 5.0
            mx = 0.05 * max(x1 - x0, 1e-9)
            my = 0.05 * max(y1 - y0, 1e-9)
            if focus is not None:
                fx0, fx1, fy0, fy1 = focus
                if max(fx1 - fx0, fy1 - fy0) < 1.0:   # a lone position: a 10 nm window around it
                    cx, cy = 0.5 * (fx0 + fx1), 0.5 * (fy0 + fy1)
                    focus = (cx - 5.0, cx + 5.0, cy - 5.0, cy + 5.0)
            # The full extent; the zoomed or panned view is kept, relocating to the current
            # file when that lies outside it (widgets.ViewNavigator.keep).
            self.view.keep(x0 - mx, x1 + mx, y0 - my, y1 + my, focus=focus)
        ax.ticklabel_format(useOffset=False, style='plain')
        ax.set_aspect('equal', adjustable='datalim')      # the box never changes: the axis ranges do
        ax.set_xlabel("x (nm, absolute)")
        ax.set_ylabel("y (nm, absolute)")
        top = (os.path.basename(topos[-1]) + f" [{self._direction_of_path(topos[-1])}]") if topos else "none"
        n_wait = sum(1 for p in visible if self._needs_record(p))
        title = (f"Scan area: {len(topos)} topo shown ({self.show_mode}), most recent on top: {top}"
                 + (f"; {n_wait} file(s) still being placed" if n_wait else ""))
        if getattr(self, 'covering_note', None):
            title += f"\n{self.covering_note}"
        srcs = f"auto {'on' if self.auto_on else 'off'}, manual {'on' if self.manual_on else 'off'}"
        if not (self.auto_on or self.manual_on):
            title += ("\ndrift off: recorded positions; the stripe matching places the lines along the fast axis"
                      if (f.kind == 'grid' and corrected and d is not None and d.get('stripe')) else "\ndrift off: recorded positions")
        elif f.kind in ('grid', 'point'):
            what = "spectra" if f.kind == 'grid' else "sweeps"
            title += "\n" + (f"drift ({srcs}): topos skewed, {what} corrected by the {source} velocity (see console)"
                              if corrected else
                              f"drift ({srcs}): no velocity covers this {'grid' if f.kind == 'grid' else 'point'} (see console)")
        else:
            title += f"\ndrift ({srcs}): topos skewed by the drift during their scans where a velocity covers them"
        ax.set_title(title, fontsize=9)
        # The cross-section line lives on this axes: redrawn after every clear.
        self.section.draw()

    # --- the Feenstra settings (the boxes under the spectra panels) ---

    def iv_width(self):
        """The broadening width the division uses (feenstra.feenstra_width)."""
        return feenstra_width(self.iv_broadening, self.iv_broadened)

    def feenstra_settings(self):
        """The browser's Feenstra settings: {'zero_sub', 'broadened', 'broadening'}."""
        return {'zero_sub': self.iv_zero_sub, 'broadened': self.iv_broadened, 'broadening': self.iv_broadening}

    def set_feenstra_settings(self, zero_sub=None, broadened=None, broadening=None):
        """
        A Feenstra setting changed (the boxes; None = unchanged): the spectra
        panels are redrawn with it, and the registries' defaults follow, so
        new averaging and line-average windows (and the Line and Map windows
        opened from here) start with it.  Open windows keep their own
        settings (their companions' boxes).  Returns True.
        """
        if zero_sub is not None:
            self.iv_zero_sub = bool(zero_sub)
        if broadened is not None:
            self.iv_broadened = bool(broadened)
        if broadening is not None:
            self.iv_broadening = float(broadening)
        for reg in (self.averages, self.line_averages):
            reg.iv_zero_sub = self.iv_zero_sub
            reg.iv_broadened = self.iv_broadened
            reg.iv_broadening = self.iv_broadening
        t = feenstra_titles(self.iv_zero_sub, self.iv_broadened, self.iv_broadening)
        print(f"Feenstra settings: {t['ratio']}; new windows open with them.")
        self.spec_settings.sync(self.feenstra_settings())
        self._draw_spectra()
        self.fig.canvas.draw_idle()
        return True

    def _draw_spectra(self):
        """
        The spectra panels of the current file.  dI/dV spectra (a grid or a
        point on the dI/dV channel): the four Feenstra panels, always: the
        dI/dV through the pipeline with the division off, the current in
        amps as the division works on it (the filters and '0Vsub' applied),
        its I/V in A/V as the division divides by it (the same curves in the
        same colours), and the ratio (the division on), the filters
        (self.filters: on the current only) and the 'calibrate' /
        'normalize' box as set (with normalize on the current and the I/V
        are unit area per curve, normalize_abs, as in every Feenstra
        companion).  One panel otherwise: the IETS channel through the
        pipeline, a phase rotation, a noise spectrum, or a note for a topo.
        """
        ax = self.ax_spec
        ax.clear()
        f = self.current
        four = bool(f.kind in ('grid', 'point') and f.spectra is not None
                    and (self.channel if self.channel in f.channels() else f.channels()[0]) == 'didv')
        ax.set_position(self.SPEC_RECTS[0] if four else self.SPEC_RECT)
        for extra in (self.ax_spec_iv, self.ax_spec_ivv, self.ax_spec_ratio):
            extra.clear()
            extra.set_visible(four)
        self.spec_settings.show(four)
        self.filters.show(four)
        self.pipeline.show(f.kind in ('grid', 'point') and f.spectra is not None)
        if f.kind in ('grid', 'point') and f.spectra is not None:
            # The channel the radio says, or the file's only channel when it lacks that one.
            channel = self.channel if self.channel in f.channels() else f.channels()[0]
            zs, width = self.iv_zero_sub, self.iv_width()
            hp, sp = self.filters.params()
            cal, norm = self.pipeline.calibrate, self.pipeline.normalize
            try:
                c = f.curves(hp, sp, False, width, channel=channel, iv_zero_sub=zs, normalize=norm, calibrate=cal)
                c_ratio = (f.curves(hp, sp, True, width, channel=channel, iv_zero_sub=zs, normalize=norm, calibrate=cal)
                           if four else None)
            except Exception as exc:
                ax.text(0.5, 0.5, f"spectra not processed: {exc}", ha='center', va='center',
                        transform=ax.transAxes, fontsize=8, wrap=True)
                return
            order = np.argsort(f.energy)
            V = f.energy[order]
            if f.n_sweeps > 1 and f.spectra_all is not None:
                # Several sweeps per pixel: every sweep drawn, the forward ones solid and the reverse ones
                # dashed, through the same pipeline (sweeps.py); the other panels show the default view.
                try:
                    ca = f.spectra_all.process(channel, hp, sp, False, width, iv_zero_sub=zs, normalize=norm,
                                               calibrate=cal and f.iv is not None)
                    keep = np.asarray(f.spectra_all.acquired, dtype=bool)
                    dirs = f.sweep_info['dir']
                    for sgn, ls in ((1, '-'), (-1, '--')):
                        m = keep & (dirs == sgn)
                        if m.any():
                            ax.plot(V, ca[order, :][:, m], lw=0.5, alpha=0.6, ls=ls)
                except Exception as exc:
                    print(f"{f.name}: every sweep not drawn ({exc}); the {f.sweep_mode} view instead.")
                    ax.plot(V, c[order, :][:, f.acquired], lw=0.5, alpha=0.6)
            else:
                ax.plot(V, c[order, :][:, f.acquired], lw=0.5, alpha=0.6)
            # Sweeps stopped before their end (left out of every average and fit) are shown here too,
            # dashed, over the points they recorded, through the same pipeline.
            unf = self._unfinished_curves(f, channel, hp, sp, width, zs, norm, cal, four)
            for key, a in (('main', ax), ('ratio', self.ax_spec_ratio), ('iv', self.ax_spec_iv),
                           ('ivv', self.ax_spec_ivv)):
                if key != 'main' and not four:
                    continue
                for Vu, cu in unf.get(key, []):
                    a.plot(Vu, cu, lw=0.8, ls='--', alpha=0.8)
            flt = ("filtered" if (hp or sp) else "unfiltered")
            state = f"{flt}, {'calibrated' if cal else 'uncalibrated'}, {'normalized' if norm else 'not normalized'}"
            units = curve_units(channel, cal, norm, f.units_lia, f.units_iets, f.units_iv)
            if four:
                # Four narrow panels: short two-line titles (the words kept to what fits a panel's
                # width: the states as on/off, the filters once on the current panel), tick labels
                # small, the y label on the first panel only.
                short = f"cal {'on' if cal else 'off'}, norm {'on' if norm else 'off'}"
                n_unf = int(np.sum(getattr(f.spectra, 'ever', f.acquired) & ~np.asarray(f.acquired, dtype=bool)))
                sweeps_txt = (f", {f.n_sweeps} sweeps per pixel (reverse dashed)" if f.n_sweeps > 1 else "")
                ax.set_title(f"dI/dV, division off, {f.n_acquired}/{f.n_spectra} curves{sweeps_txt}"
                             + (f" + {n_unf} unfinished (dashed)" if n_unf else "") + f"; {short}", fontsize=6.5)
                has_zero = energy_has_zero(f.energy)
                sub = ("I(0 V) subtracted" if has_zero else "no 0 V in range") if zs else "as recorded"
                how = (f"broadened {self.iv_broadening:g} V" if self.iv_broadened else "plain")
                ax2, ax3 = self.ax_spec_iv, self.ax_spec_ivv
                if f.iv is not None:
                    iv = iv_corrected(f.energy, f.iv, zs, hp, sp)
                    ivv = iv_broadened(f.energy, f.iv, width, zs, hp, sp)
                    if norm:
                        iv = normalize_abs(f.energy, iv)[0]
                        ivv = normalize_abs(f.energy, ivv)[0]
                    ax2.plot(V, iv[order, :][:, f.acquired], lw=0.5, alpha=0.6)
                    ax3.plot(V, ivv[order, :][:, f.acquired], lw=0.5, alpha=0.6)
                else:
                    for a in (ax2, ax3):
                        a.text(0.5, 0.5, "no Current channel", ha='center', va='center', transform=a.transAxes, fontsize=8)
                ax2.set_title(f"I(V), {'unit area' if norm else 'A'}, {flt}, {sub}", fontsize=6.5)
                ax3.set_title(f"I/V, {'unit area' if norm else 'A/V'}, {how}", fontsize=6.5)
                ax4 = self.ax_spec_ratio
                ax4.plot(V, c_ratio[order, :][:, f.acquired], lw=0.5, alpha=0.6)
                ax4.set_title("(dI/dV)/(I/V), division on", fontsize=6.5)
                for a in (ax, ax2, ax3, ax4):
                    a.tick_params(labelsize=7)
                    a.set_xlabel("Energy (eV)", fontsize=8)
                ax.set_ylabel(f"dI/dV ({units})", fontsize=8)
                return
            n_unf = int(np.sum(getattr(f.spectra, 'ever', f.acquired) & ~np.asarray(f.acquired, dtype=bool)))
            ax.set_title(f"{STS.CHANNEL_NAMES[channel]}: {f.n_acquired} of {f.n_spectra} curves (acquired)"
                         + (f" + {n_unf} unfinished (dashed)" if n_unf else "") + f", {state}",
                         fontsize=9)
            ax.set_xlabel("Energy (eV)")
            ax.set_ylabel(("dI/dV" if channel == 'didv' else "d2I/dV2 (IETS)") + f" ({units})")
        elif f.kind == 'phase':
            f.phase.draw(ax)
            ax.set_title(f"Lock-in phase rotation: {f.name}, (fitted maximum and minimum, set rotation dotted)",
                         fontsize=9)
        elif f.kind == 'fft':
            f.fft.draw(ax)
            ax.set_title(f"Noise spectrum: {f.name}, every recorded channel", fontsize=9)
        else:
            ax.text(0.5, 0.5, "topography file (no spectra)" if f.kind == 'topo' else "no spectra pages",
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

    def _unfinished_curves(self, f, channel, hp, sp, width, zs, norm, cal, four):
        """
        The sweeps of the file stopped before their end (written, but ending in never-written
        samples, so not kept): each through the same pipeline as the kept ones over the points
        it recorded (the first points of its column in acquisition order).  Returns
        {'main': [(V, curve)], 'ratio': [...], 'iv': [...], 'ivv': [...]}, energies ascending;
        a sweep with fewer than MIN_UNFINISHED_POINTS points is skipped.
        """
        from sm4_Browser.feenstra import iv_corrected, iv_broadened
        sp_ = f.spectra
        ever = np.asarray(getattr(sp_, 'ever', f.acquired), dtype=bool)
        stopped = np.flatnonzero(ever & ~np.asarray(f.acquired, dtype=bool))
        out = {'main': [], 'ratio': [], 'iv': [], 'ivv': []}
        if not len(stopped):
            return out
        pts = getattr(sp_, 'points_acquired', None)
        cols = sp_.channel(channel)
        E = np.asarray(f.energy, dtype=np.float64)
        for j in stopped:
            p = int(pts[j]) if pts is not None and j < len(pts) else 0
            if p < self.MIN_UNFINISHED_POINTS:
                continue
            e = E[:p]
            o = np.argsort(e)
            lia = cols[:p, j:j + 1]
            iv = f.iv[:p, j:j + 1] if f.iv is not None else None
            try:
                c = STS.process_arrays(e, lia, iv, hp, sp, False, width, channel=channel, iv_zero_sub=zs,
                                       normalize=norm, calibrate=cal and iv is not None, vmod=getattr(sp_, 'vmod', None))
                out['main'].append((e[o], c[o, 0]))
                if four and iv is not None:
                    r = STS.process_arrays(e, lia, iv, hp, sp, True, width, channel=channel, iv_zero_sub=zs,
                                           normalize=norm, calibrate=cal, vmod=getattr(sp_, 'vmod', None))
                    out['ratio'].append((e[o], r[o, 0]))
                    a = iv_corrected(e, iv, zs, hp, sp)
                    b = iv_broadened(e, iv, width, zs, hp, sp)
                    if norm:
                        a = normalize_abs(e, a)[0]
                        b = normalize_abs(e, b)[0]
                    out['iv'].append((e[o], a[o, 0]))
                    out['ivv'].append((e[o], b[o, 0]))
            except Exception as exc:
                print(f"{f.name}: unfinished sweep {j} not shown ({exc}).")
        return out

    MIN_UNFINISHED_POINTS = 5

    # --- callbacks ---

    def _nav_text(self, i):
        self.txt_nav.set_text(f"{i + 1} / {len(self.files)}   {os.path.basename(self.files[i])}")

    def _spec_nav_text(self, k):
        if not self.spectra_files:
            self.txt_spec_nav.set_text("no spectra files in the directory")
            return
        i = self.spectra_files[k]
        self.txt_spec_nav.set_text(f"spectra {k + 1} / {len(self.spectra_files)}   "
                                   f"{os.path.basename(self.files[i])} ({self.kinds[self.files[i]]})")

    def _on_slider(self, val):
        """While the file slider moves: show the file it points at; the load waits for the release."""
        i = int(np.clip(int(round(val)), 0, len(self.files) - 1))
        self._slider_pending = i
        self._nav_text(i)
        self.fig.canvas.draw_idle()

    def _on_spec_slider(self, val):
        """While the spectra slider moves: show the spectra file it points at; the load waits for the release."""
        if not self.spectra_files:
            return
        k = int(np.clip(int(round(val)), 0, len(self.spectra_files) - 1))
        self._spec_slider_pending = k
        self._spec_nav_text(k)
        self.fig.canvas.draw_idle()

    def _on_slider_release(self, event):
        """Mouse released after moving a slider: go to the chosen file."""
        i = self._slider_pending
        k = self._spec_slider_pending
        self._slider_pending = None
        self._spec_slider_pending = None
        if i is not None:
            j = self._nearest_allowed(i)
            if j is None:
                print("step filter: no file passes the filter.")
            if j is not None and j != self.index:
                self._goto(j)
            elif j is not None and j != i:
                self.sld_file.set(self.index)
                self._nav_text(self.index)
        elif k is not None:
            allowed = [kk for kk in range(len(self.spectra_files)) if self._allowed(self.spectra_files[kk])]
            if allowed:
                kk = min(allowed, key=lambda a: (abs(a - k), a))
                if self.spectra_files[kk] != self.index:
                    self._goto_spectra(kk)

    def _on_scale(self, text):
        try:
            v = float(text)
            if v <= 0:
                raise ValueError("must be > 0")
        except ValueError as exc:
            print(f"Scale factor '{text}' not accepted ({exc}); keeping {self.scalefactor:g}.")
            self.box_scale.set(f"{self.scalefactor:g}")
            self.fig.canvas.draw_idle()
            return
        if v != self.scalefactor:
            self.scalefactor = v
            self._draw_all()

    def _on_topo_toggle(self, active):
        """The topo processor box (widgets.TopoProcessorBox; one active at most): everything redrawn."""
        self.topo_active = list(active)
        self._draw_all()

    def _on_filter_change(self):
        """A Hampel / Savitzky-Golay toggle or edit: the spectra panels redrawn (the filters act on the division's current)."""
        hp, sp = self.filters.params()
        print("filters: " + ("Hampel " + str(hp) if hp else "Hampel off") + ", " + ("SG " + str(sp) if sp else "SG off")
              + " (on the current of the Feenstra division; the dI/dV columns are never filtered).")
        self._draw_spectra()
        self.fig.canvas.draw_idle()

    def _on_pipeline_change(self, state):
        """The 'calibrate' / 'normalize' box: the spectra panels redrawn with the two pipeline steps as set."""
        print(f"spectra panels: calibrate {'on' if state['calibrate'] else 'off'}, normalize {'on' if state['normalize'] else 'off'}.")
        self._draw_spectra()
        self.fig.canvas.draw_idle()

    def _on_line_changed(self, idx):
        """The Line window changed its line: highlight that line on the scan area."""
        f = self.current
        if f.kind != 'grid':
            return
        self.line_index = int(np.clip(idx, 0, max(f.n_lines - 1, 0)))
        self._draw_scan_area()
        self.fig.canvas.draw_idle()

    def _on_drift_toggle(self, which, flag):
        """'auto drift' / 'manual drift': the scan area redrawn with the sources as set."""
        if which == 'auto':
            self.auto_on = bool(flag)
        else:
            self.manual_on = bool(flag)
        print(f"{which} drift {'on' if bool(flag) else 'off'}.")
        self._draw_scan_area()
        self.fig.canvas.draw_idle()

    def _on_stripe_window(self):
        """
        'stripe match': the stripe window of the current grid (stripe.StripeWindow),
        opened at the slow-axis setting in force here and with the grid's
        topo pass on top; one window at a time (a second press raises it).
        A change made there reaches the scan area and every open window
        through the Drift's subscribers.
        """
        f = self.current
        if self.drift is None or f.kind != 'grid' or not f.topo_pages:
            print("stripe match: the current file is not a grid with topo pages, or there is no drift model.")
            return
        w = self.stripe_window
        if w is not None and w.is_open():
            w.raise_()
            return
        from sm4_Browser.stripe import StripeWindow
        self.drift.loaded = self.loaded
        try:
            self.stripe_window = StripeWindow(self.drift, f, self.scalefactor, list(self.topo_active),
                                              setting=self.drift.slow_setting(f, self.auto_on, self.manual_on),
                                              pass_top='reverse' if self._direction_of(f) == 'reverse' else 'forward')
        except ValueError as exc:
            print(f"stripe match: window not opened ({exc}).")
            self.stripe_window = None

    def _on_drift_window(self):
        """
        'drift': the manual drift window of the current file (drift.DriftWindow:
        a spectra file's before and after topos, a topo's pair with the
        next topo of its segment, else the previous one), seeding from the
        'auto drift' state; a change applied there reaches the scan area
        and every open Line and Map window through the Drift's subscribers.
        One window at a time: a second press while it is open raises it.
        """
        w = self.drift_window
        if w is not None and w.is_open():
            w.raise_()
            return
        try:
            self.drift_window = DriftWindow(self.drift, self.current, self.scalefactor, self.topo_active,
                                            auto_on=self.auto_on)
        except ValueError as exc:
            print(f"drift window: {exc}.")
            self.drift_window = None

    def _on_manual_changed(self):
        """The Drift's manual state changed (a span applied or cleared, in any drift window): the scan area redrawn."""
        self._drift_version = getattr(self, '_drift_version', 0) + 1
        self._draw_scan_area()
        self.fig.canvas.draw_idle()

    # --- the cross section ---

    def _on_section_line(self):
        """'section': a line across the middle of the current view (pressed again, it goes)."""
        if self.section.active():
            self.section.clear()
            print("cross section: line removed.")
            return
        x0, x1 = self.ax_scan.get_xlim()
        y0, y1 = self.ax_scan.get_ylim()
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        w = 0.25 * (x1 - x0)
        self.section.set((cx - w, cy), (cx + w, cy))
        print("cross section: line placed; drag its endpoints, then 'cross section'.")

    def _on_section_moved(self):
        """An endpoint dragged: the open profile window follows."""
        if self.profile_window is not None and self.profile_window.is_open():
            self._on_cross_section()

    def _on_cross_section(self):
        """'cross section': the raw topo under the line into the profile window (topo.cross_section)."""
        pts = self.section.points()
        if pts is None:
            print("cross section: no line; 'section' places one.")
            return
        k = 1e9 / self.scalefactor
        p0_m, p1_m = pts[0] / k, pts[1] / k
        topos = self._topos()
        if not topos:
            print("cross section: no topography in the loaded set.")
            return
        try:
            x, zf, zr, names = cross_section(topos, p0_m, p1_m, self.scalefactor)
        except Exception as exc:
            print(f"cross section: not available ({exc}).")
            return
        curves = [("forward", x, zf, plots.PROFILE_FORWARD_COLOR)]
        pair = {'label': "cross section, " + (", ".join(names) if names else "no topo"), 'forward': (x, zf),
                'reverse': (x, zr) if np.isfinite(zr).any() else None}
        if np.isfinite(zr).any():
            curves.append(("reverse", x, zr, plots.PROFILE_REVERSE_COLOR))
        title = "Cross section: " + (", ".join(names) if names else "no topo under the line")
        if self.profile_window is not None and self.profile_window.is_open():
            self.profile_window.update(curves, title, pair)
        else:
            self.profile_window = ProfileWindow(curves, title, xlabel="distance along the line (nm)",
                                                ylabel="height (nm, raw)", pair=pair, registry=self.profile_plots)
            self.children.append(self.profile_window)
        print(f"cross section: {len(x)} samples over {x[-1]:.2f} nm from {len(names)} topo(s): " + ", ".join(names))

    def _on_channel(self, label):
        """
        The 'channel' radio: the spectra panel shows the chosen channel (dI/dV
        or IETS) of the current file when it has it, and the Line, Map, and
        new average windows opened from now on start on it (on the file's own
        channel when it lacks the chosen one).
        """
        channel = {v: k for k, v in STS.CHANNEL_NAMES.items()}.get(label, 'didv')
        if channel == self.channel:
            return
        self.channel = channel
        name = STS.CHANNEL_NAMES[channel]
        f = self.current
        if f.kind in ('grid', 'point') and f.spectra is not None and channel not in f.channels():
            print(f"{f.name} has no {name} channel; the panel keeps {STS.CHANNEL_NAMES[f.channels()[0]]}.")
        elif f.kind in ('grid', 'point') and f.spectra is not None:
            print(f"channel {name}: the spectra panel shows it.")
        print(f"channel {name}: new Line, Map, and average windows start on it (on the file's own "
              f"channel when it lacks {name}).")
        self._draw_spectra()
        self.fig.canvas.draw_idle()

    def _on_direction(self, label):
        target = self._topo_under(self.current)
        if target is None:
            return
        self.topo_dir[target.path] = label
        self._draw_scan_area()
        self.fig.canvas.draw_idle()

    def _on_site(self, label):
        """The 'site' radio: the current file's flag set by hand (kept in the sites file)."""
        if label not in FLAGS:
            return
        path = self.files[self.index]
        if label == self.sites.flag(path):
            return
        self.sites.set_flag(path, label)
        print(f"site: {os.path.basename(path)} flagged {label} by hand (kept in the sites file).")
        self._draw_info()
        self.fig.canvas.draw_idle()

    # --- averages (avg1, avg2, ...) ---

    def _open_avgs(self):
        """index -> window of the open average windows (closed ones are dropped): Averages.open."""
        return self.averages.open()

    def _next_avg_index(self):
        """1 + the highest index among the open windows and the saved records."""
        return self.averages.next_index()

    def _avgs_holding(self, f):
        """Indices of the open windows holding the file (a point file, or a selection of a grid file)."""
        return self.averages.holding(f.path)

    def _update_avg_text(self):
        f = self.current
        held = self._avgs_holding(f) if f.kind in ('point', 'grid') else []
        open_avgs = self._open_avgs()
        self.txt_in_avgs.set_text(("in " + ", ".join(open_avgs[i].name for i in held)) if held
                                  else "in no open average")

    def _new_avg_window(self, index, divide_iv=True, iv_broadening=None, channel=None, name=None,
                        iv_zero_sub=None, iv_broadened=None):
        """
        Open the averaging window of that index (Averages.new).  Its own
        'Feenstra' toggle starts on (or at a loaded record's state), with the
        browser's Feenstra settings (broadening, '0Vsub', 'broadened') and
        channel unless given; the name is avg<index> unless given (a loaded
        record's).
        """
        return self.averages.new(index, name=name, divide_iv=divide_iv, iv_broadening=iv_broadening,
                                 channel=channel, iv_zero_sub=iv_zero_sub, iv_broadened=iv_broadened)

    def _add_to_avg(self, index):
        """Add the current single point to the open average of that index."""
        f = self.current
        w = self._open_avgs().get(index)
        if f.kind != 'point' or w is None:
            return
        if w.has(f):
            print(f"{w.name} already holds {f.name}.")
            return
        try:
            w.add(f)
        except Exception as exc:
            print(f"add to {w.name}: {exc}")
            return
        print(f"{w.name}: added {f.name} ({len(w.entries)} entries).")
        self._update_avg_text()
        self.fig.canvas.draw_idle()

    def _on_add_new_avg(self):
        """'add new': start the next average with the current single point."""
        if self.current.kind != 'point':
            return
        self._popup_close()
        index = self._next_avg_index()
        f = self.current
        try:
            # On the browser's channel when the point holds it, else on the point's own.
            self._new_avg_window(index, channel=self.channel if self.channel in f.channels() else f.channels()[0])
        except Exception as exc:
            print(f"add new: {exc}")
            return
        self._add_to_avg(index)

    def _on_add_existing_avg(self):
        """'add existing': toggle the list of the open averages above the button."""
        if self.current.kind != 'point':
            return
        if self._picker.is_open():
            self._popup_close()
        else:
            self._popup_open()

    def _popup_close(self):
        self._picker.close()

    def _on_load_saved(self):
        """Open the saved averages window (rescanned when it is already open)."""
        if self.saved_window is not None and self.saved_window.is_open():
            self.saved_window.scan()
            return
        try:
            self.saved_window = SavedAveragesWindow(self.directory, self.load_saved_average)
        except Exception as exc:
            print(f"saved averages: {exc}")

    def _resolve_file(self, path, name):
        """
        The Sm4File for a recorded file: by base name in the browser's
        directory, then by the recorded path.  FileNotFoundError otherwise.
        """
        by_name = {os.path.basename(p): p for p in self.files}
        p = by_name.get(name)
        if p is None and path and os.path.isfile(path):
            p = path
        if p is None:
            raise FileNotFoundError(f"{name} is not in {self.directory} and {path or '(no path)'} does not exist")
        return self.loaded.get(p) or self._load(p)

    def _entry_from_record(self, fi):
        """
        The entry of a record's file item: the Sm4File of a point item, or the
        Selection of a 'selection' item (its grid file resolved, the recorded
        columns checked against the file's spectra: ValueError when a column
        is outside them or on a never-acquired spectrum).
        """
        f = self._resolve_file(fi.get('path'), fi.get('name'))
        if fi.get('kind') != 'selection':
            return f
        if fi.get('sweeps'):
            f = f.with_sweeps(fi['sweeps'])          # the columns index that sweep view
        cols = [int(c) for c in fi.get('cols', [])]
        if f.kind != 'grid':
            raise ValueError(f"{f.name} is a {f.kind} file, not the grid file of a marker group")
        if not cols:
            raise ValueError(f"the marker group of {f.name} names no columns")
        bad = [c for c in cols if c < 0 or c >= f.n_spectra]
        if bad:
            raise ValueError(f"column {bad[0]} of the marker group is outside {f.name}'s "
                             f"{f.n_spectra} spectra")
        never = [c for c in cols if not f.acquired[c]]
        if never:
            raise ValueError(f"column {never[0]} of the marker group is a never-acquired spectrum of {f.name}")
        where = fi.get('where') or {'line': None, 'positions': []}
        return Selection(f, cols, where)

    def _files_from_label(self, parsed):
        """
        Files of an old label-named fit history: the first by its path (base
        name in the directory, then the path itself, then its time), the
        others by time of day among the directory's files.  None, with the
        reason on the console, when a time matches no file or several.
        """
        by_name = {os.path.basename(p): p for p in self.files}
        by_time = {}
        for p in self.files:
            by_time.setdefault(filename_stamp(p)[3:], []).append(p)
        paths = []
        for k, t in enumerate(parsed['times']):
            p = None
            if k == 0:
                p = by_name.get(os.path.basename(parsed['path']))
                if p is None and os.path.isfile(parsed['path']):
                    p = parsed['path']
            if p is None:
                cands = by_time.get(t, [])
                tt = f"{t[0]:02d}:{t[1]:02d}:{t[2]:02d}.{t[3]:03d}"
                if len(cands) != 1:
                    print(f"saved averages: file {k + 1} of the old fit history ({tt}) matches "
                          f"{len(cands)} files in {self.directory}; only the fit is opened.")
                    return None
                p = cands[0]
            paths.append(p)
        return [self.loaded.get(p) or self._load(p) for p in paths]

    def load_saved_average(self, entry):
        """
        Load an entry of the saved averages window: an averaging record (or
        an old fit history) rebuilt in a window of its index, run, compared
        with its fit history, and the fit opened from the fit history when
        one exists; a Line window's marker record opened on its file and
        line with everything restored; a line-average record rebuilt in its
        window.
        """
        if entry['kind'] == 'line':
            self.load_line_record(entry['record'])
            return
        if entry['kind'] == 'lineavg':
            self.load_lineavg_record(entry['record'])
            return
        cache = entry.get('cache')
        if entry['kind'] == 'record':
            rec = entry['record']
            index = int(rec['index'])
            name = str(rec.get('name') or f"avg{index}")
            if index in self._open_avgs():
                print(f"{name} (index {index}) is open; load refused.")
                return
            if self.averages.by_name(name) is not None:
                print(f"an open average is already named {name}; load refused.")
                return
            try:
                files = [self._entry_from_record(fi) for fi in rec.get('files', [])]
            except (FileNotFoundError, ValueError) as exc:
                print(f"{name} load refused: {exc}.")
                return
            removed = [(int(i), int(j)) for i, j in rec.get('removed', [])]
            all_columns = int(rec.get('format', 1)) < 2      # format 1: indexed among all the file's sweeps
            hp, sp = rec.get('hampel'), rec.get('sgolay')   # a 'notch' entry of an older record is ignored
            rs = rec.get('resample')                       # absent before the resampling: bin / interp off
            ch = rec.get('channel', 'didv')                # absent in records written before the IETS: dI/dV
            div = bool(rec.get('divide_iv', False))        # ('weighted' / 'weight_c' of older records are ignored)
            bro = float(rec.get('iv_broadening_V', 0.0))
            # Format 4 records hold the Feenstra settings; older ones are loaded as they were made:
            # no 0 V subtraction, the I/V broadened when a width was recorded.
            zs = bool(rec.get('iv_zero_sub', False))
            brd = bool(rec.get('iv_broadened', bro > 0))
            cal = bool(rec.get('calibrate', True))         # format 5: the pipeline flags; older records: both on
            nrm = bool(rec.get('normalize', True))
        else:
            index = self._next_avg_index()
            name = None                                    # avg<index>
            files = self._files_from_label(entry['parsed'])
            if files is None:
                try:
                    an = fit.Fit.from_cache(cache)
                    an.plot()
                    self.children.append(an)
                except Exception as exc:
                    print(f"saved averages: fit history not opened: {exc}")
                return
            removed = list(entry['parsed']['removed'])
            all_columns = True                             # an old fit history's label: the same
            hp, sp = entry.get('hampel'), entry.get('sgolay')
            rs, ch = None, 'didv'                          # old fit histories: no resampling, dI/dV
            div = False                                    # old fit histories hold dI/dV
            bro = self.iv_broadening
            zs, brd = False, bro > 0
            cal, nrm = True, True
        if not files:
            print(f"avg{index} load refused: the record names no files.")
            return
        for f in files:
            if f.kind not in ('point', 'selection'):
                print(f"avg{index} load refused: {f.name} is a {f.kind} file, not a single point "
                      f"or a marker-group selection.")
                return
        if all_columns:
            # Written by the old browser, the removed indices count all the sweeps of a file; the
            # window holds the acquired ones (decision 2), so each index moves to its place among
            # those, and the removal of a never-acquired sweep (not in the window) is dropped.
            mapped = []
            for i, j in removed:
                acq = files[i].acquired if 0 <= i < len(files) else None
                known = acq is not None and 0 <= j < len(acq)
                if known and not acq[j]:
                    print(f"avg{index}: removed sweep {i + 1}:{j + 1} of the old record was never acquired "
                          "(not in the window); the removal is dropped.")
                    continue
                mapped.append((i, int(np.count_nonzero(acq[:j])) if known else j))
            removed = mapped
        # A record's removed index counts the file's acquired sweeps (the window holds those).
        for i, j in removed:
            if not (0 <= i < len(files)) or not (0 <= j < files[i].n_acquired):
                print(f"avg{index} load refused: removed sweep {i + 1}:{j + 1} does not exist "
                      f"({len(files)} files; file {min(max(i, 0), len(files) - 1) + 1} has "
                      f"{files[min(max(i, 0), len(files) - 1)].n_acquired} acquired sweeps).")
                return
        # The record's Feenstra state goes to the new window's own toggle (the browser's is left alone).
        t = feenstra_titles(zs, brd, bro if div else self.iv_broadening)
        print(f"avg{index}: the record was saved with Feenstra {'on' if div else 'off'}"
              + (f" ({t['ivv']})" if div else "") + "; the window's toggle and settings are set to it.")
        try:
            w = self._new_avg_window(index, divide_iv=div, iv_broadening=bro if div else self.iv_broadening,
                                     channel=ch, name=name, iv_zero_sub=zs, iv_broadened=brd)
            w.restore(files, removed, hp, sp, rs, ch, calibrate=cal, normalize=nrm)
        except Exception as exc:
            print(f"avg{index} load: {exc}")
            return
        if cache and os.path.isfile(cache):
            w.open_fit(cache)
        else:
            print(f"avg{index}: no fit history for this average.")
        self._update_avg_text()
        self.fig.canvas.draw_idle()

    def _line_window_kwargs(self):
        """The Line window's keyword arguments from the browser's settings (the Feenstra settings and registries)."""
        kw = dict(self.line_window_kwargs)
        kw.pop('cmap', None)
        kw['iv_broadening_V'] = self.iv_broadening
        kw['iv_zero_sub'] = self.iv_zero_sub
        kw['iv_broadened'] = self.iv_broadened
        kw['averages'] = self.averages             # its 'add' buttons use the browser's averaging windows
        kw['line_averages'] = self.line_averages   # its 'line → ...' buttons the browser's line averages
        kw['profile_plots'] = self.profile_plots   # its 'profile' windows add to the browser's profile plots
        kw['drift'] = self.drift                   # its own drift toggles start as the browser's
        kw['auto_on'] = self.auto_on
        kw['manual_on'] = self.manual_on
        self.drift.loaded = self.loaded
        return kw

    def load_line_record(self, rec):
        """
        A Line window's marker record: the browser goes to the record's
        file, a Line window opens on its line with the record's filter
        boxes, Feenstra settings, topo rows, and gamma, and the rest of the
        record (restore_markers) is put into it.
        """
        try:
            f = self._resolve_file(rec.get('path'), rec.get('name'))
        except FileNotFoundError as exc:
            print(f"line markers load refused: {exc}.")
            return
        if f.kind != 'grid':
            print(f"line markers load refused: {f.name} is a {f.kind} file, not a grid.")
            return
        if f.path in self.files:
            self._goto(self.files.index(f.path))
        line = int(rec.get('line', 0))
        kw = self._line_window_kwargs()
        kw['hampel_filter_params'] = rec.get('hampel', kw.get('hampel_filter_params'))
        kw['sgolay_filter_params'] = rec.get('sgolay', kw.get('sgolay_filter_params'))
        kw['apply_calibration'] = bool(rec.get('calibrate', True))
        kw['apply_normalize'] = bool(rec.get('normalize', True))
        kw['divide_by_iv'] = bool(rec.get('divide_iv', True))
        kw['iv_broadening_V'] = float(rec.get('iv_broadening_V', self.iv_broadening))
        kw['iv_zero_sub'] = bool(rec.get('iv_zero_sub', self.iv_zero_sub))
        kw['iv_broadened'] = bool(rec.get('iv_broadened', self.iv_broadened))
        kw['gamma'] = float(rec.get('gamma', 1.0))
        kw['topo_lines'] = [(int(r), m) for r, m in rec.get('topo_lines', [])] or None
        kw['marker_positions'] = None
        kw['sm4'] = f
        try:
            analyzer = Line(f.path, chunk_index=line, cmap=self.line_window_kwargs.get('cmap', 'jet'),
                            scalefactor=self.scalefactor, scan_mode='single', combine_scans=False,
                            sweeps=rec.get('sweeps') or f.sweep_mode)
            ch = rec.get('channel', 'didv')
            if ch != analyzer.channel and ch in analyzer.spectra.channels():
                analyzer.channel = ch
                analyzer.extract_data()
            if self.current.path == f.path:
                analyzer._int_line_callback = self._on_line_changed
            analyzer.plot_line_with_topo_interactive(**kw)
            analyzer.restore_markers(rec)
            self._on_line_changed(line)
        except Exception as exc:
            print(f"line markers load: {exc}")
            return
        self.children.append(analyzer)

    def load_lineavg_record(self, rec):
        """
        A line-average record: the members rebuilt from their files
        (member_from_file: the line's columns, positions, stripe rows, and
        the Line window's markers), the window opened of its index and name
        with the record's Feenstra state, the members added, and the
        record's state restored and run (LineAverage.restore).
        """
        index = int(rec.get('index', self.line_averages.next_index()))
        name = str(rec.get('name') or f"lineavg{index}")
        if index in self.line_averages.open():
            print(f"{name} (index {index}) is open; load refused.")
            return
        if self.line_averages.by_name(name) is not None:
            print(f"an open line average is already named {name}; load refused.")
            return
        members = []
        for rm in rec.get('members', []):
            try:
                f = self._resolve_file(rm.get('path'), rm.get('name'))
                if rm.get('sweeps'):
                    f = f.with_sweeps(rm['sweeps'])
                members.append(member_from_file(f, int(rm.get('line', 0)), float(rm.get('scalefactor', self.scalefactor)),
                                                topo_lines=[(int(r), m) for r, m in rm.get('topo_lines', [])] or None,
                                                markers=rm.get('markers'), centers=rm.get('centers')))
            except (FileNotFoundError, ValueError) as exc:
                print(f"{name} load refused: {exc}.")
                return
        if not members:
            print(f"{name} load refused: the record names no lines.")
            return
        ch = rec.get('channel', 'didv')
        try:
            w = self.line_averages.new(members[0], index=index, name=name, divide_iv=bool(rec.get('divide_iv', True)),
                                       iv_broadening=float(rec.get('iv_broadening_V', self.iv_broadening)),
                                       channel=ch, iv_zero_sub=bool(rec.get('iv_zero_sub', self.iv_zero_sub)),
                                       iv_broadened=bool(rec.get('iv_broadened', self.iv_broadened)))
            for m in members[1:]:
                w.add(m)
            w.restore(rec)
        except Exception as exc:
            print(f"{name} load: {exc}")
            return
        self.children.append(w)

    def _on_topo_window(self):
        topo = self._topo_for(self.current)
        if topo is None:
            print("Topo window: no topography in the loaded set.")
            return
        f = self.current
        spec = f if f.kind in ('point', 'phase', 'grid') and len(f.spec_xy_m) else None
        title = topo.name + (f"   (x: {f.name})" if spec is not None else "")
        self.drift.loaded = self.loaded
        self.children.append(TopoWindow(topo, list(self.topo_active), self.scalefactor, None, title,
                                        direction=self._direction_of(topo), spec=spec, drift=self.drift,
                                        auto_on=self.auto_on, manual_on=self.manual_on,
                                        anchor_on=self.anchor_on,
                                        spec_dir=self._direction_of(f) if f.kind == 'grid' else 'forward'))

    def _on_line(self):
        f = self.current
        if f.kind != 'grid':
            return
        kw = self._line_window_kwargs()
        kw['topo_lines'] = [(self.line_index, 'forward')]          # the line's forward row; the row controls add more
        kw['sm4'] = f
        try:
            analyzer = Line(f.path, chunk_index=self.line_index, cmap=self.line_window_kwargs.get('cmap', 'jet'),
                            scalefactor=self.scalefactor, scan_mode='single',
                            combine_scans=False, sweeps=f.sweep_mode)
            if self.channel != analyzer.channel and self.channel in analyzer.spectra.channels():
                analyzer.channel = self.channel        # open on the browser's channel
                analyzer.extract_data()
            analyzer._int_line_callback = self._on_line_changed
            analyzer.plot_line_with_topo_interactive(**kw)
        except Exception as exc:
            print(f"Line window: {exc}")
            return
        self.children.append(analyzer)

    def _on_map(self):
        f = self.current
        if f.kind != 'grid':
            return
        try:
            # The Map window has its own 'Feenstra' toggle (on at open); the Feenstra settings are handed over.
            self.children.append(Map(f, list(self.topo_active), self.scalefactor,
                                     self.hampel_defaults, self.sgolay_defaults, self.cmap_map,
                                     divide_iv=True, iv_broadening=self.iv_broadening,
                                     iv_zero_sub=self.iv_zero_sub, iv_broadened=self.iv_broadened,
                                     channel=self.channel if self.channel in f.channels() else f.channels()[0],
                                     averages=self.averages,
                                     fit_settings=self.line_window_kwargs.get('fit_settings'),
                                     drift=self.drift, auto_on=self.auto_on, manual_on=self.manual_on))
        except Exception as exc:
            print(f"Map window: {exc}")


# ====================================================================== #
#  The window: native Qt (PySide6 / PyQt through qtplot + pyqtgraph)       #
# ====================================================================== #
#
# The scan area and the spectra panels are two qtplot figures embedded in one Qt main window
# (a splitter), each with its toolbar on top (Home, Pan, Zoom, Save as PDF / SVG / EPS / PNG);
# the controls are native Qt widgets in group boxes under them; the info box takes the file's
# information and the details cut from the panel titles.  A control repaints only itself, and a
# pan of the scan area never redraws the spectra.

import sys
from sm4_Browser.qtplot import QtWidgets, QtCore, QtGui, NavigationToolbar
from sm4_Browser.topo import TOPO_DIRECTIONS


def Figure(figsize=(6.4, 4.8)):
    """A qtplot figure embedded in the browser's layout (its canvas is the widget)."""
    return plt.Figure(figsize=figsize, embed=True)


def FigureCanvasQTAgg(fig):
    return fig.canvas


def NavigationToolbar2QT(canvas, parent):
    return NavigationToolbar(canvas.figure, parent)


Qt = QtCore.Qt


# ---------------------------------------------------------------------- #
#  Native controls with the widgets.Control interface                     #
#  (value, set (never fires), on_change, show, enable)                    #
# ---------------------------------------------------------------------- #

class QControl:
    """
    The interface of widgets.Control on a native Qt widget: value, set (never fires), on_change,
    show, enable, and set_short (the browser's window taller than wide: every label with a short
    form in widgets.SHORT_LABELS shows it; _WindowZoom switches every registered control).
    """

    _all = weakref.WeakSet()       # every native control made, for _WindowZoom's label switch

    def __init__(self, name, qwidget):
        self.name = name
        self.qt = qwidget
        self._cb = None
        self._quiet = False
        self._short = False
        self.qt.setObjectName(name)
        QControl._all.add(self)

    def _shown(self, text):
        """The text a label shows: its short form in the short mode, else as it is."""
        return widgets.short_form(text) if self._short else str(text)

    def _apply_labels(self):
        """Every label of the control set to its shown form (the kinds with text override this)."""
        return None

    def set_short(self, flag):
        flag = bool(flag)
        if flag != self._short:
            self._short = flag
            self._apply_labels()

    def show(self, flag):
        self.qt.setVisible(bool(flag))

    def enable(self, flag):
        self.qt.setEnabled(bool(flag))

    def on_change(self, callback):
        self._cb = callback

    def _fire(self, *args):
        if self._cb is not None and not self._quiet:
            self._cb(*args)

    def _silently(self, fn, *args):
        self._quiet = True
        blocked = self.qt.blockSignals(True)
        try:
            fn(*args)
        finally:
            self.qt.blockSignals(blocked)
            self._quiet = False


def _label_of(name):
    lab = widgets.LABELS.get(name, name)
    return name if lab is None else lab.strip()


class QCheck(QControl):
    """A one-row toggle: callback(flag)."""

    def __init__(self, name, value=False, label=None):
        self._label = _label_of(name) if label is None else label
        super().__init__(name, QtWidgets.QCheckBox(self._label))
        self.qt.setChecked(bool(value))
        self.qt.toggled.connect(lambda flag: self._fire(bool(flag)))

    def _apply_labels(self):
        self.qt.setText(self._shown(self._label))

    @property
    def value(self):
        return self.qt.isChecked()

    def set(self, flag):
        if self.value != bool(flag):
            self._silently(self.qt.setChecked, bool(flag))

    def set_label_color(self, color):
        self.qt.setStyleSheet(f"color: {color};")


class QChecks(QControl):
    """A box of toggle rows: callback(label, states) after a click (the clicked row already flipped)."""

    def __init__(self, name, labels, values=None, title=None):
        box = QtWidgets.QGroupBox(title or '')
        super().__init__(name, box)
        self._title = title or ''
        lay = QtWidgets.QVBoxLayout(box)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(2)
        self._labels = list(labels)
        vals = [False] * len(self._labels) if values is None else [bool(v) for v in values]
        self._boxes = []
        for lab, v in zip(self._labels, vals):
            cb = QtWidgets.QCheckBox(lab)
            cb.setChecked(v)
            cb.toggled.connect(lambda _f, lab=lab: self._fire(lab, self.value))
            lay.addWidget(cb)
            self._boxes.append(cb)

    def _apply_labels(self):
        self.qt.setTitle(self._shown(self._title))
        for cb, lab in zip(self._boxes, self._labels):
            cb.setText(self._shown(lab))

    @property
    def labels(self):
        return list(self._labels)

    @property
    def value(self):
        return [b.isChecked() for b in self._boxes]

    def set(self, i, flag):
        b = self._boxes[i]
        if b.isChecked() != bool(flag):
            self._quiet = True
            blocked = b.blockSignals(True)
            try:
                b.setChecked(bool(flag))
            finally:
                b.blockSignals(blocked)
                self._quiet = False

    def set_all(self, flags):
        for i, f in enumerate(flags):
            self.set(i, f)


class QRadio(QControl):
    """A radio list: callback(label)."""

    def __init__(self, name, labels, value=None, title=None):
        box = QtWidgets.QGroupBox(title or '')
        super().__init__(name, box)
        self._title = title or ''
        lay = QtWidgets.QVBoxLayout(box)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(2)
        self._labels = list(labels)
        self._group = QtWidgets.QButtonGroup(box)
        self._buttons = []
        active = self._labels.index(value) if value in self._labels else 0
        for i, lab in enumerate(self._labels):
            rb = QtWidgets.QRadioButton(lab)
            rb.setChecked(i == active)
            self._group.addButton(rb, i)
            lay.addWidget(rb)
            self._buttons.append(rb)
        self._group.idClicked.connect(lambda i: self._fire(self._labels[i]))

    def _apply_labels(self):
        self.qt.setTitle(self._shown(self._title))
        for rb, lab in zip(self._buttons, self._labels):
            rb.setText(self._shown(lab))

    @property
    def labels(self):
        return list(self._labels)

    @property
    def value(self):
        i = self._group.checkedId()
        return self._labels[i] if i >= 0 else None

    def set(self, label):
        if label in self._labels and self.value != label:
            # QButtonGroup.idClicked fires on user clicks only; setChecked never fires it.
            self._buttons[self._labels.index(label)].setChecked(True)


class QPush(QControl):
    """A button: callback()."""

    def __init__(self, name, label=None):
        self._text = _label_of(name) if label is None else label
        super().__init__(name, QtWidgets.QPushButton(self._text))
        self.qt.clicked.connect(lambda _=False: self._fire())

    @property
    def value(self):
        return self.qt.text()

    def set_text(self, text):
        self._text = str(text)
        self._apply_labels()

    def _apply_labels(self):
        self.qt.setText(self._shown(self._text))


class QRange(QControl):
    """
    An integer slider with its value shown: callback(value) while it moves.
    release(callback) registers callback() for the end of a move: the mouse
    released after a drag, or at once for a keyboard / page-click step (the
    browser loads the file then).
    """

    def __init__(self, name, value=0, vmin=0, vmax=1, label=None):
        w = QtWidgets.QWidget()
        super().__init__(name, w)
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        self._label = _label_of(name) if label is None else label
        self.label = QtWidgets.QLabel(self._label)
        lay.addWidget(self.label)
        self.slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(vmin), int(vmax))
        self.slider.setValue(int(value))
        self.slider.setPageStep(max(1, (int(vmax) - int(vmin)) // 20))
        self.readout = QtWidgets.QLabel(str(int(value)))
        self.readout.setMinimumWidth(36)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.readout)
        self._release = None
        self.slider.valueChanged.connect(self._changed)
        self.slider.sliderReleased.connect(self._released)

    def _apply_labels(self):
        self.label.setText(self._shown(self._label))

    def _changed(self, v):
        self.readout.setText(str(v))
        if self._quiet:
            return
        self._fire(v)
        if not self.slider.isSliderDown():          # keyboard, wheel, or page click: no drag to wait for
            self._released()

    def _released(self):
        if self._release is not None and not self._quiet:
            self._release()

    def release(self, callback):
        self._release = callback

    @property
    def value(self):
        return self.slider.value()

    def set(self, value):
        v = int(round(value))
        if self.slider.value() != v:
            self._quiet = True
            blocked = self.slider.blockSignals(True)
            try:
                self.slider.setValue(v)
                self.readout.setText(str(v))
            finally:
                self.slider.blockSignals(blocked)
                self._quiet = False


class QText(QControl):
    """A text box: callback(text) on Enter or focus-out when the text changed."""

    def __init__(self, name, value='', label=None, width=70):
        w = QtWidgets.QWidget()
        super().__init__(name, w)
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lab = _label_of(name) if label is None else label
        self._label = lab
        self.label = QtWidgets.QLabel(lab)
        if lab:
            lay.addWidget(self.label)
        self.edit = QtWidgets.QLineEdit(str(value))
        self.edit.setMaximumWidth(width)
        self.edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.edit)
        self._last = str(value)
        self.edit.editingFinished.connect(self._submit)

    def _submit(self):
        text = self.edit.text()
        if text == self._last:
            return
        self._last = text
        self._fire(text)

    @property
    def value(self):
        return self.edit.text()

    def set(self, text):
        text = str(text)
        self._last = text
        if self.edit.text() != text:
            blocked = self.edit.blockSignals(True)
            try:
                self.edit.setText(text)
            finally:
                self.edit.blockSignals(blocked)

    def set_label_color(self, color):
        self.label.setStyleSheet(f"color: {color};")

    def _apply_labels(self):
        self.label.setText(self._shown(self._label))


class QTextLabel:
    """
    Stands in for a matplotlib figure text: set_text, set_visible.  The text wraps: a long line (a
    file name in the navigation text) never widens the window beyond its design size, and nothing
    of it is cut; the text can be selected and copied.
    """

    def __init__(self, text=''):
        self.qt = QtWidgets.QLabel(text)
        self.qt.setWordWrap(True)
        self.qt.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

    def set_text(self, text):
        self.qt.setText(str(text))

    def set_visible(self, flag):
        self.qt.setVisible(bool(flag))

    def get_text(self):
        return self.qt.text()


# ---------------------------------------------------------------------- #
#  Composite controllers: the widgets.py logic, Qt construction            #
# ---------------------------------------------------------------------- #

def _row(*items, stretch=True):
    w = QtWidgets.QWidget()
    lay = QtWidgets.QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    for it in items:
        lay.addWidget(it.qt if hasattr(it, 'qt') else it)
    if stretch:
        lay.addStretch(1)
    return w


class QStepButtons(widgets.StepButtons):
    def __init__(self, left_name, right_name):
        self.left = QPush(left_name)
        self.right = QPush(right_name)
        self._cb = None
        self.left.on_change(lambda: self._fire(-1))
        self.right.on_change(lambda: self._fire(+1))
        self.qt = _row(self.left, self.right, stretch=False)


class QPipelineBox(widgets.PipelineBox):
    def __init__(self, calibrate=True, normalize=True):
        self.box = QChecks('pipeline', widgets.SPECS['pipeline']['labels'], [calibrate, normalize])
        self._cb = None
        self.box.on_change(lambda label, states: self._fire())
        self.qt = self.box.qt


class QTopoProcessorBox(widgets.TopoProcessorBox):
    def __init__(self, labels, active):
        self.box = QChecks('topo processors', labels, active, title='topo processing')
        self._cb = None
        self.box.on_change(self._on_click)
        self.qt = self.box.qt


class QFeenstraSettings(widgets.FeenstraSettings):
    def __init__(self, settings):
        self.chk_zero = QCheck('0Vsub', settings['zero_sub'])
        self.chk_broad = QCheck('broadened', settings['broadened'])
        self.box_width = QText('width', f"{settings['broadening']:g}", label='width (V)')
        self._cb = None
        self._settings = dict(settings)
        self.chk_zero.on_change(lambda flag: self._fire(zero_sub=flag))
        self.chk_broad.on_change(lambda flag: self._fire(broadened=flag))
        self.box_width.on_change(self._on_width)
        self.qt = _row(self.chk_zero, self.chk_broad, self.box_width)

    def show(self, flag):
        self.qt.setVisible(bool(flag))

    def place(self, x, y):
        pass

    def color_labels(self, zero_color, broad_color):
        self.chk_zero.set_label_color(zero_color)
        self.chk_broad.set_label_color(broad_color)
        self.box_width.set_label_color(broad_color)


class QFilterControls(widgets.FilterControls):
    """The Hampel / Savitzky-Golay box: widgets.FilterControls' parsing and state, Qt widgets."""

    def __init__(self, hampel=None, sgolay=None, hampel_on=False, sgolay_on=False):
        self._hp = dict(self.HAMPEL_DEFAULT if hampel is None else hampel)
        self._sp = dict(self.SGOLAY_DEFAULT if sgolay is None else sgolay)
        self._on = [bool(hampel_on), bool(sgolay_on)]
        self._callback = None
        self._quiet = False
        self._title = 'filters (on the current of the division)'
        self.qt = QtWidgets.QGroupBox(self._title)
        grid = QtWidgets.QGridLayout(self.qt)
        grid.setContentsMargins(6, 4, 6, 4)
        self.chk_s = QCheck('sg', self._on[1], label='SG')
        self.chk_h = QCheck('hampel', self._on[0], label='Hampel')
        self.box_wl = QText('window_length', self._fmt(self._sp['window_length']), label='window', width=45)
        self.box_po = QText('polyorder', self._fmt(self._sp['polyorder']), label='polyorder', width=45)
        self.box_win = QText('window', self._fmt(self._hp['window']), label='window', width=45)
        self.box_nsig = QText('n_sigma', self._fmt(self._hp['n_sigma']), label='nσ', width=45)
        for r, items in enumerate([(self.chk_s, self.box_wl, self.box_po), (self.chk_h, self.box_win, self.box_nsig)]):
            for c, it in enumerate(items):
                grid.addWidget(it.qt, r, c)
        self.chk_h.on_change(lambda flag: self._toggle(0))
        self.chk_s.on_change(lambda flag: self._toggle(1))
        for name in ('window', 'n_sigma', 'window_length', 'polyorder'):
            self._box(name).on_change(lambda text, name=name: self._submit(name, text))

    def set_rect(self, rect):
        pass

    def set_short(self, flag):
        """Short labels on or off (the browser's window taller than wide): the box's title shows 'filters'; its controls switch on their own."""
        self.qt.setTitle(widgets.short_form(self._title) if flag else self._title)

    def show(self, flag):
        self.qt.setVisible(bool(flag))

    def _set_box(self, name, value):
        self._box(name).set(self._fmt(value))

    def _set_toggle(self, i, state):
        (self.chk_h, self.chk_s)[i].set(bool(state))
        self._on[i] = bool(state)

    def _fire(self):
        if self._callback is not None:
            self._callback()

    def color_labels(self, hampel_color, sgolay_color=None):
        self.chk_h.set_label_color(hampel_color)
        self.chk_s.set_label_color(hampel_color if sgolay_color is None else sgolay_color)

    def set_params(self, hampel, sgolay):
        if hampel is not None:
            for k in ('window', 'n_sigma'):
                self._hp[k] = hampel[k]
                self._set_box(k, hampel[k])
        if sgolay is not None:
            for k in ('window_length', 'polyorder'):
                self._sp[k] = sgolay[k]
                self._set_box(k, sgolay[k])
        self._set_toggle(0, hampel is not None)
        self._set_toggle(1, sgolay is not None)


class QAveragePicker:
    """The 'add existing' list as a native pop-up menu under the button (widgets.AveragePicker's rows)."""

    def __init__(self, averages):
        self.averages = averages
        self.menu = None

    def is_open(self):
        return self.menu is not None and self.menu.isVisible()

    def open(self, anchor, entry, on_pick):
        self.close()
        open_avgs = self.averages.open()
        if not open_avgs:
            print("add existing: no average is open ('add new' starts one).")
            return
        menu = QtWidgets.QMenu(anchor)
        for i in sorted(open_avgs):
            win = open_avgs[i]
            text = f"{win.name}  ({len(win.entries)} entries)" + ("  holds this entry" if win.has(entry) else "")
            act = menu.addAction(text)
            act.triggered.connect(lambda _=False, i=i: on_pick(i))
        self.menu = menu
        menu.popup(anchor.mapToGlobal(QtCore.QPoint(0, anchor.height())))

    def close(self):
        if self.menu is not None:
            self.menu.close()
            self.menu = None


# ---------------------------------------------------------------------- #
#  The figures proxy: draw_idle redraws only the canvases that changed     #
# ---------------------------------------------------------------------- #

class _WindowZoom(QtCore.QObject):
    """
    The main window's own controls (native Qt) follow the one text-size rule of every window
    (qtplot.window_zoom): their font is the window's size relative to its design size, ZOOM_MIN
    to ZOOM_MAX, as the figures' text does (qtplot's font_zoom); when the controls would not fit
    the window at that zoom the font is reduced, to at most FIT_MIN of it, until they do (_fit),
    so no text is cut, as a figure window's control shrinks its font to fit its box.  The same resize decides the
    window's shape: taller than wide, every native control with a short form shows it
    (QControl.set_short, widgets.SHORT_LABELS), the objects given to track_short switch their own
    titles, and the embedded figures are told (Figure.set_portrait), so their controls and their
    file-name titles switch with the window and not with their own canvas.
    """

    def __init__(self, win, design):
        super().__init__(win)
        self.win = win
        self.design = design
        self.base = QtGui.QFont(win.font())
        if self.base.pointSizeF() <= 0:
            self.base.setPointSizeF(9.0)
        self.base.setPointSizeF(self.base.pointSizeF() * plt.FONT_SCALE)
        self.extra = []              # (widget, its own base font)
        self.figs = []               # the embedded figures: their portrait flag follows the window
        self.shorts = []             # objects with set_short(flag) that are not QControls (the filter box)
        self.zoom = None
        self.portrait = None
        self._busy = False           # against re-entry: the fit delivers layout events itself
        win.installEventFilter(self)

    def track(self, widget, font):
        self.extra.append((widget, QtGui.QFont(font)))
        self._apply(force=True)

    def track_figures(self, *figs):
        self.figs.extend(figs)

    def track_short(self, obj):
        self.shorts.append(obj)

    def eventFilter(self, obj, ev):
        # A resize, and a layout request (a label's text changed, so the controls may need more or
        # less room at the current font): the zoom and the fit are applied again.
        if obj is self.win and ev.type() in (QtCore.QEvent.Type.Resize, QtCore.QEvent.Type.LayoutRequest) \
                and not self._busy:
            self._busy = True
            try:
                self._apply()
            finally:
                self._busy = False
        return False

    def _apply_shape(self):
        """The window's shape: taller than wide switches every label with a short form, and back."""
        p = self.win.height() > self.win.width()
        if p == self.portrait:
            return
        self.portrait = p
        for c in list(QControl._all):
            try:
                if self.win.isAncestorOf(c.qt):
                    c.set_short(p)
            except RuntimeError:
                pass
        for o in self.shorts:
            o.set_short(p)
        for f in self.figs:
            f.set_portrait(p)

    FIT_MIN = 0.75               # the controls' font shrinks to at most this fraction of the rule's zoom to fit the window

    def _apply(self, force=False):
        self._apply_shape()
        k = plt.screen_scale()
        z = plt.window_zoom(self.win.width(), self.win.height(), self.design[0] * k, self.design[1] * k) * k
        z = self._fit(z)
        if not force and self.zoom is not None and abs(z - self.zoom) < 0.03 * self.zoom:
            return
        self._set_fonts(z)

    def _fit(self, z):
        """
        The text is never cut: when the window's controls, at the rule's zoom, need more room than
        the window has (Qt's layout minimum), the font is reduced step by step, to at most FIT_MIN
        of that zoom, until they fit, as a figure window's control shrinks its font to fit its box.
        Returns the zoom to use.
        """
        central = self.win.centralWidget()
        lay = central.layout() if central is not None else None
        if lay is None:
            return z
        z0, zc = z, z
        for _ in range(12):
            self._set_fonts(zc)
            # The widgets' size hints follow the new font only once the layout events the font
            # change posted are delivered; then the layout's minimum is current.
            QtWidgets.QApplication.sendPostedEvents()
            lay.invalidate()
            lay.activate()
            m = lay.minimumSize()
            if m.width() <= self.win.width() and m.height() <= self.win.height():
                return zc
            if zc <= z0 * self.FIT_MIN + 1e-9:
                break
            zc = max(zc * 0.96, z0 * self.FIT_MIN)
        # Even at the smallest font the controls need more room than the window has: the window's
        # minimum is raised to what they need, so that nothing is cut (the window grows to it).
        central.setMinimumSize(max(central.minimumWidth(), m.width()), max(central.minimumHeight(), m.height()))
        return zc

    def _set_fonts(self, z):
        """Every font of the window at the zoom z: the window font, the indicators, the boxes' widths, the icons, the tracked widgets."""
        k = plt.screen_scale()
        self.zoom = z
        f = QtGui.QFont(self.base)
        f.setPointSizeF(self.base.pointSizeF() * z / k)
        self.win.setFont(f)
        from sm4_Browser.widgets import indicator_css
        self.win.setStyleSheet(indicator_css(z))
        for e in self.win.findChildren(QtWidgets.QLineEdit):
            bw = e.property('base_width')
            if bw is None and e.maximumWidth() < 10000:
                bw = e.maximumWidth()
                e.setProperty('base_width', bw)
            if bw:
                e.setMaximumWidth(int(bw * z))
        for tb in self.win.findChildren(QtWidgets.QToolBar):
            n = int(round(22 * z))
            tb.setIconSize(QtCore.QSize(n, n))
        for w, bf in self.extra:
            g = QtGui.QFont(bf)
            g.setPointSizeF(bf.pointSizeF() * plt.FONT_SCALE * z / k)
            w.setFont(g)


class _OnClose(QtCore.QObject):
    """Runs a callback when the watched window closes."""

    def __init__(self, win, callback):
        super().__init__(win)
        self.callback = callback
        win.installEventFilter(self)

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.Type.Close:
            try:
                self.callback()
            except Exception:
                pass
        return False


class _StaleCanvases:
    def __init__(self, canvases):
        self.canvases = list(canvases)

    def draw_idle(self):
        for c in self.canvases:          # pyqtgraph repaints what changed by itself
            c.draw_idle()

    def draw(self):
        for c in self.canvases:
            c.draw()

    def mpl_connect(self, name, fn):
        return [c.mpl_connect(name, fn) for c in self.canvases]


class _Figures:
    """What browser.Browser reaches through self.fig: canvas.draw_idle() (and get_size_inches)."""

    def __init__(self, *figs):
        self.figs = figs
        self.canvas = _StaleCanvases([f.canvas for f in figs])

    def get_size_inches(self):
        return self.figs[0].get_size_inches()


# ---------------------------------------------------------------------- #
#  The browser                                                            #
# ---------------------------------------------------------------------- #

class Browser(_BrowserCore):
    """
    browser.Browser in a native Qt main window.  Same constructor arguments;
    the window is shown at the end of construction; Browser.run(...) also
    starts the event loop.
    """

    # The design sizes: the window in pixels at qtplot.DPI (the text zoom is the window's size
    # relative to it, qtplot.window_zoom), and the two embedded canvases' figure sizes in inches.
    DESIGN = (1700, 950)
    SCAN_DESIGN = (9, 7)          # the scan-area canvas
    SPEC_DESIGN = (8, 3.6)        # the spectra canvas
    # The spectra figure is its own canvas: the panel rects are fractions of IT.
    SPEC_RECT = [0.07, 0.17, 0.91, 0.72]
    SPEC_RECTS = [[0.06, 0.17, 0.205, 0.72], [0.305, 0.17, 0.205, 0.72], [0.55, 0.17, 0.205, 0.72],
                  [0.795, 0.17, 0.195, 0.72]]

    def __init__(self, *args, **kwargs):
        self.app = plt._app()
        super().__init__(*args, **kwargs)      # builds the window (_build_figure), loads file 0
        self.win.show()
        plt._figures_main_windows.append(self.win)

    @classmethod
    def run(cls, *args, **kwargs):
        b = cls(*args, **kwargs)
        b.app.exec() if hasattr(b.app, 'exec') else b.app.exec_()
        return b

    # --- the window ---

    def _build_figure(self):
        self.win = QtWidgets.QMainWindow()
        self._closer = _OnClose(self.win, self._bg_shutdown)     # the worker processes end with the window
        self.win.setWindowTitle(f"sm4 browser: {self.directory}")
        k = plt.screen_scale()
        self.win.resize(int(self.DESIGN[0] * k), int(self.DESIGN[1] * k))
        self._zoomer = _WindowZoom(self.win, self.DESIGN)
        central = QtWidgets.QWidget()
        # An explicit minimum size (the design size at the smallest text zoom): with it Qt's layout does
        # not pin the window to the layout's own minimum at the current font (QLayout's default size
        # constraint), which would stop the window from shrinking back after it was enlarged, since
        # the text only shrinks once the window does.
        central.setMinimumSize(int(self.DESIGN[0] * k * plt.ZOOM_MIN), int(self.DESIGN[1] * k * plt.ZOOM_MIN))
        self.win.setCentralWidget(central)
        outer = QtWidgets.QHBoxLayout(central)
        split = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(split)

        # Left: the scan area (own canvas, own toolbar) and the file navigation / drift controls.
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.fig_scan = Figure(figsize=self.SCAN_DESIGN)
        self.canvas_scan = FigureCanvasQTAgg(self.fig_scan)
        self.ax_scan = self.fig_scan.add_axes([0.11, 0.09, 0.78, 0.81])
        self.ax_topo_cbar = plots.draw_topo_cbar(self.fig_scan, [0.925, 0.25, 0.02, 0.5])
        self.toolbar_scan = NavigationToolbar2QT(self.canvas_scan, left)
        lv.addWidget(self.toolbar_scan)
        lv.addWidget(self.canvas_scan, 1)
        self.view = widgets.ViewNavigator(self.fig_scan, self.ax_scan)
        self.section = widgets.DragLine(self.fig_scan, self.ax_scan, self.view)
        self.layer = ScanLayer(self.ax_scan, self.files, TOPO_CMAP)

        n = len(self.files)
        self.sld_file = QRange('file', 0, 0, max(n - 1, 1))
        self.txt_nav = QTextLabel()
        self._slider_pending = None
        self.btn_prev, self.btn_next = QPush('< Prev'), QPush('Next >')
        self.btn_topo = QPush('Topo Window')
        self.box_scale = QText('scale factor', f"{self.scalefactor:g}")
        self.topo_box = QTopoProcessorBox(processor_labels(), self.topo_active)
        self.chk_auto = QCheck('auto drift', self.auto_on)
        self.chk_manual = QCheck('manual drift', self.manual_on)
        self.btn_stripe = QPush('stripe match')
        self.btn_drift = QPush('drift')
        self.btn_saved = QPush('load saved averages')
        self.rad_channel = QRadio('channel', widgets.SPECS['channel']['labels'],
                                  STS.CHANNEL_NAMES[self.channel], title='channel')
        lv.addWidget(self.txt_nav.qt)
        lv.addWidget(self.sld_file.qt)
        bottom = QtWidgets.QHBoxLayout()
        nav = QtWidgets.QVBoxLayout()
        nav.addWidget(_row(self.btn_prev, self.btn_next, self.btn_topo, self.box_scale))
        nav.addWidget(_row(self.chk_auto, self.chk_manual, self.btn_stripe, self.btn_drift))
        nav.addWidget(_row(self.btn_saved))
        self.rad_show = QRadio('show', list(SHOW_MODES), self.show_mode, title='show')
        self.chk_kinds = QChecks('step through', ['topo', 'point', 'grid', 'phase', 'fft'], [True] * 5,
                                 title='step through')
        self.box_min_dur = QText('min duration', '0', label='min (min)', width=45)
        filt = QtWidgets.QWidget()
        fl = QtWidgets.QVBoxLayout(filt)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.addWidget(self.chk_kinds.qt)
        fl.addWidget(self.box_min_dur.qt)
        bottom.addLayout(nav, 1)
        bottom.addWidget(self.topo_box.qt)
        bottom.addWidget(self.rad_channel.qt)
        bottom.addWidget(self.rad_show.qt)
        bottom.addWidget(filt)
        lv.addLayout(bottom)
        self._status_label = QtWidgets.QLabel('')
        self._status_label.setWordWrap(True)           # a long status line wraps rather than widening the window
        lv.addWidget(self._status_label)
        split.addWidget(left)

        # Right: the spectra canvas, its settings, navigation, info, and the per-file controls.
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.fig_spec = Figure(figsize=self.SPEC_DESIGN)
        self.canvas_spec = FigureCanvasQTAgg(self.fig_spec)
        self.ax_spec = self.fig_spec.add_axes(self.SPEC_RECT)
        self.ax_spec_iv = self.fig_spec.add_axes(self.SPEC_RECTS[1])
        self.ax_spec_ivv = self.fig_spec.add_axes(self.SPEC_RECTS[2])
        self.ax_spec_ratio = self.fig_spec.add_axes(self.SPEC_RECTS[3])
        rv.addWidget(NavigationToolbar2QT(self.canvas_spec, right))
        rv.addWidget(self.canvas_spec, 4)
        self.spec_settings = QFeenstraSettings(self.feenstra_settings())
        self.spec_settings.on_change(self.set_feenstra_settings)
        self.filters = QFilterControls(hampel=self.hampel_defaults, sgolay=self.sgolay_defaults)
        self.filters.on_change(self._on_filter_change)
        self._zoomer.track_short(self.filters)
        self.pipeline = QPipelineBox(calibrate=False, normalize=False)
        self.pipeline.on_change(self._on_pipeline_change)
        settings = QtWidgets.QHBoxLayout()
        sv = QtWidgets.QVBoxLayout()
        sv.addWidget(self.spec_settings.qt)
        sv.addStretch(1)
        settings.addLayout(sv)
        settings.addWidget(self.filters.qt)
        settings.addWidget(self.pipeline.qt)
        settings.addStretch(1)
        rv.addLayout(settings)

        n_spec = len(self.spectra_files)
        self.spec_steps = QStepButtons('< spectra', 'spectra >')
        self.spec_steps.on_change(self._step_spectra)
        self.set_steps = QStepButtons('prev region', 'next region')
        self.set_steps.on_change(self._on_region_step)
        self.sld_spec = QRange('spectra', 0, 0, max(n_spec - 1, 1))
        self.txt_spec_nav = QTextLabel()
        self._spec_slider_pending = None
        rv.addWidget(self.txt_spec_nav.qt)
        sn = QtWidgets.QHBoxLayout()
        sn.addWidget(self.spec_steps.qt)
        sn.addWidget(self.set_steps.qt)
        sn.addWidget(self.sld_spec.qt, 1)
        rv.addLayout(sn)

        self.info = QtWidgets.QPlainTextEdit()
        self.info.setReadOnly(True)
        self.info.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        font.setPointSize(8)
        self.info.setFont(font)
        self._zoomer.track(self.info, font)
        rv.addWidget(self.info, 1)

        self.btn_line, self.btn_map = QPush('Line'), QPush('Map')
        self.btn_add_new, self.btn_add_existing = QPush('add new'), QPush('add existing')
        self.txt_in_avgs = QTextLabel()
        self.btn_section, self.btn_cross = QPush('section'), QPush('cross section')
        self.txt_dir = QTextLabel('topo under')
        self.rad_dir = QRadio('topo pass', list(TOPO_DIRECTIONS), 'forward')
        self.txt_site = QTextLabel('site')
        self.rad_site = QRadio('site', list(widgets.SPECS['site']['labels']), 'experiment')
        actions = QtWidgets.QHBoxLayout()
        col = QtWidgets.QVBoxLayout()
        col.addWidget(_row(self.btn_line, self.btn_map))
        col.addWidget(self.txt_in_avgs.qt)
        col.addWidget(_row(self.btn_add_new, self.btn_add_existing))
        col.addWidget(_row(self.btn_section, self.btn_cross))
        col.addStretch(1)
        actions.addLayout(col, 1)
        for lab, rad in ((self.txt_dir, self.rad_dir), (self.txt_site, self.rad_site)):
            c = QtWidgets.QVBoxLayout()
            c.addWidget(lab.qt)
            c.addWidget(rad.qt)
            c.addStretch(1)
            actions.addLayout(c)
        rv.addLayout(actions)
        split.addWidget(right)
        split.setSizes([900, 800])

        # Everything the experiment logic redraws goes through self.fig.canvas.draw_idle():
        # only the canvases whose figure changed are redrawn.
        self.fig = _Figures(self.fig_scan, self.fig_spec)
        self._zoomer.track_figures(self.fig_scan, self.fig_spec)
        # The details cut from the panel titles go into this window's info panel.
        self.fig_scan.on_info = self.fig_spec.on_info = lambda text: self._draw_info()
        self._picker = QAveragePicker(self.averages)

        self.btn_prev.on_change(lambda: self._step(-1))
        self.btn_next.on_change(lambda: self._step(+1))
        self.rad_show.on_change(self._on_show_mode)
        self.chk_kinds.on_change(self._on_step_kinds)
        self.box_min_dur.on_change(self._on_min_duration)
        self.sld_file.on_change(self._on_slider)
        self.sld_spec.on_change(self._on_spec_slider)
        self.sld_file.release(lambda: self._on_slider_release(None))
        self.sld_spec.release(lambda: self._on_slider_release(None))
        self.btn_topo.on_change(self._on_topo_window)
        self.box_scale.on_change(self._on_scale)
        self.topo_box.on_change(self._on_topo_toggle)
        self.btn_line.on_change(self._on_line)
        self.btn_map.on_change(self._on_map)
        self.btn_add_new.on_change(self._on_add_new_avg)
        self.btn_add_existing.on_change(self._on_add_existing_avg)
        self.btn_saved.on_change(self._on_load_saved)
        self.chk_auto.on_change(lambda flag: self._on_drift_toggle('auto', flag))
        self.chk_manual.on_change(lambda flag: self._on_drift_toggle('manual', flag))
        self.btn_stripe.on_change(self._on_stripe_window)
        self.btn_drift.on_change(self._on_drift_window)
        self.btn_section.on_change(self._on_section_line)
        self.btn_cross.on_change(self._on_cross_section)
        self.section.on_change(self._on_section_moved)
        self.rad_dir.on_change(self._on_direction)
        self.rad_site.on_change(self._on_site)
        self.rad_channel.on_change(self._on_channel)

        # Keyboard: Ctrl+Left / Ctrl+Right step files, Ctrl+PgUp / Ctrl+PgDown step spectra files
        # (with Ctrl so that the arrows still move the cursor in the text boxes).
        QShortcut = getattr(QtGui, 'QShortcut', None) or QtWidgets.QShortcut   # Qt6 / Qt5
        for key, fn in (('Ctrl+Left', lambda: self._step(-1)),
                        ('Ctrl+Right', lambda: self._step(+1)),
                        ('Ctrl+PgUp', lambda: self._step_spectra(-1)),
                        ('Ctrl+PgDown', lambda: self._step_spectra(+1))):
            sc = QShortcut(QtGui.QKeySequence(key), self.win)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(fn)

    def _popup_open(self):
        self._picker.open(self.btn_add_existing.qt, self.current, self._add_to_avg)

    def _draw_spectra(self):
        super()._draw_spectra()
        # The settings rows under the panels are Qt widgets: no figure space to keep for them.

    def _draw_info(self):
        """The info panel as selectable text: the head lines, then one 'name: value' per row."""
        f = self.current
        path = self.files[self.index]
        lines = [f"[{self.index + 1} / {len(self.files)}]  {f.name}",
                 f"{stamp_text(f.stamp)}   type: {f.kind}   site: {self.sites.flag(path)}   "
                 f"{self.sites.region_text(path)}   loaded: {len(self.loaded)} files", ""]
        try:
            rows = f.measurement_info()
        except Exception as exc:
            rows = [('measurement information', f"not available ({exc})")]
        width = max(len(str(name)) for name, _ in rows) if rows else 0
        for name, value in rows:
            lines.append(f"{str(name):<{width}} : {' | '.join(str(value).splitlines())}")
        panels = [t for t in (self.fig_scan.info_text(), self.fig_spec.info_text()) if t]
        if panels:
            lines += [""] + panels
        self.info.setPlainText("\n".join(lines))


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="sm4 experiment browser (Qt)")
    p.add_argument('directory', help="folder of .sm4 files of one experiment")
    p.add_argument('--scalefactor', type=float, default=2.75)
    a = p.parse_args(argv)
    hampel_defaults = {'window': 4, 'n_sigma': 3}
    sgolay_defaults = {'window_length': 6, 'polyorder': 3}
    fit_settings = {'known_features': [], 'n_jobs': -1, 'max_width_ratio': 3, 'noise_z': 2.5, 'use_cache': True}
    line_window_kwargs = {'cmap': 'jet', 'hampel_filter_params': hampel_defaults,
                          'sgolay_filter_params': sgolay_defaults, 'apply_calibration': True,
                          'apply_normalize': True, 'marker_positions': None, 'fit_settings': fit_settings}
    Browser.run(a.directory, scalefactor=a.scalefactor, hampel_defaults=hampel_defaults,
                  sgolay_defaults=sgolay_defaults, line_window_kwargs=line_window_kwargs, cmap_map='jet',
                  drift_params={'sg_window': 5, 'sg_poly': 2, 'prominence_frac': 0.20})


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main()                      # python -m sm4_Browser.browser <directory> [--scalefactor 2.75]
        sys.exit()
    # Run from Spyder (F5) or as a script: the experiment folder and the settings here.
    directory = r'experiment_dir'

    # Colour maps: any matplotlib colour-map name ('jet', 'viridis', 'RdBu_r', ...), a matplotlib
    # Colormap object (e.g. matplotlib.colormaps['magma'] or a LinearSegmentedColormap of your own),
    # or a pyqtgraph ColorMap.  They are converted once into pyqtgraph lookup tables (qtplot.to_cmap).
    cmap_line = 'nipy_spectral'               # the Line and line-average heatmaps
    cmap_map = 'nipy_spectral'                # the Map window's per-energy maps

    hampel_defaults = {'window': 4, 'n_sigma': 3}
    sgolay_defaults = {'window_length': 6, 'polyorder': 3}
    fit_settings = {'known_features': [], 'n_jobs': -1, 'max_width_ratio': 3, 'noise_z': 2.5, 'use_cache': True}
    line_window_kwargs = {'cmap': cmap_line, 'hampel_filter_params': hampel_defaults,
                          'sgolay_filter_params': sgolay_defaults, 'apply_calibration': True,
                          'apply_normalize': True, 'marker_positions': None, 'fit_settings': fit_settings}
    drift_params = {'sg_window': 5, 'sg_poly': 2, 'prominence_frac': 0.20}

    browser = Browser(directory, scalefactor=2.75, hampel_defaults=hampel_defaults,
                        sgolay_defaults=sgolay_defaults, line_window_kwargs=line_window_kwargs,
                        cmap_map=cmap_map, drift_params=drift_params)
    plt.show()                      # runs the event loop outside IPython; returns at once inside Spyder
