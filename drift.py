"""
Drift correction: the displacement of the surface's features between
consecutive topographies of an experiment, measured one topo pair at a time
by feature-point matching, and its application to the spectrum positions of
grids and single points, to every pixel of the topographies, to the rows of
the line heatmaps and the pixels of the maps, and to the drift lines of the
sweeps.  The module-level functions are the pair tracker (track_pair) and
the optional stripe matching of a grid's lines (drift_correct_grid); Drift is
the orchestration the Browser drives (the segments of the directory, the
pairs tracked at startup and cached, the per-file results).

References
----------
The pair tracker is the drift-compensation method of
  Z. Diao, K. Ueda, L. Hou, H. Yamashita, O. Custance, and M. Abe,
  "Automatic drift compensation for nanoscale imaging using feature point
  matching", Appl. Phys. Lett. 122, 121601 (2023), doi:10.1063/5.0139330,
applied offline to one pair of consecutive topographies at a time (their
three-image feedforward loop collapses to the two-point slope of one pair).
The paper's feature detector and descriptor is AKAZE,
  P. F. Alcantarilla, J. Nuevo, and A. Bartoli, "Fast Explicit Diffusion for
  Accelerated Features in Nonlinear Scale Spaces", British Machine Vision
  Conference (BMVC) 2013, pp. 13.1-13.11,
which the paper describes as matching the precision of KAZE, the same
authors' original detector,
  P. F. Alcantarilla, A. Bartoli, and A. J. Davison, "KAZE Features",
  European Conference on Computer Vision (ECCV) 2012, Part VI, pp. 214-227,
  doi:10.1007/978-3-642-33783-3_16.
KAZE is the detector used here: OpenCV's AKAZE searches each scale level
only inside a border of round(10 sqrt(2) x sigma_size) + 1 pixels (29 px at
the finest level), which leaves the central 6 x 6 pixels of a 64 x 64
topography and nothing at the coarser levels, so on the small images of
this experiment it finds no features; KAZE has no such border and runs on
the images at their native resolution (nothing is upsampled).  Both are
implemented in OpenCV (the cv2 module of the opencv-python package, OpenCV
4.x: cv2.KAZE, cv2.BFMatcher, cv2.GaussianBlur, cv2.erode; in OpenCV 5.x
KAZE and AKAZE live in the contrib module xfeatures2d of
opencv-contrib-python), and the displacement clustering is k-means as
implemented in SciPy (scipy.cluster.vq.kmeans2).

The tracker (track_pair), per consecutive topo pair A (earlier) and B (later)
of a segment, following Diao et al. step by step:
  1. the forward pass of each; B resampled onto A's pixel grid
     (resample_to_frame) so that the two share one pixel size and
     orientation (the paper's frames never change; the method is neither
     rotation nor scale invariant, so this step is needed here);
  2. preprocessing as the paper: a plane fit subtracted from each image, then
     2D smoothing with a Gaussian kernel of smooth_kernel_px x
     smooth_kernel_px pixels and smooth_sigma_px standard deviation (the
     paper: 11 x 11, sigma 1); the image is then scaled to 0..1 over its
     acquired pixels and handed to KAZE as float32 (OpenCV divides 8-bit
     input by 255 before the same computation, so 0..1 float is the
     equivalent input and the default detector threshold applies on that
     scale);
  3. KAZE at OpenCV's defaults (the paper gives no detector parameters),
     keypoints restricted to the pixels both images acquired, that mask
     eroded by the kernel's half-width so no keypoint sits on the resampling
     edge;
  4. matching with the brute-force matcher with crossCheck on (a pair
     survives only when it is the nearest match in both directions), the L2
     norm, KAZE's descriptor being a float vector (the paper's AKAZE
     descriptor is binary and takes the Hamming norm); every surviving pair
     gives a displacement (x_B - x_A, y_B - y_A) in pixels of A's grid;
  5. k-means on the displacement list from k = 2 upward: every cluster must
     satisfy (1 / m) sum_j |g_j - g_c| <= d (m points g_j, centroid g_c);
     when any cluster fails, k increases by one and k-means runs again
     (capped at the number of pairs).  d is absolute: cluster_d_m metres
     (the paper: 1 nm for 256 px over 20 nm, "empirically determined
     according to the image acquisition parameters"), converted to pixels
     with the pair's pixel size, so image size and resolution do not enter;
  6. majority rule: the centroid of the largest cluster is the pair's
     displacement (a tie between equally large clusters, not covered by the
     paper, goes to the tighter one); with fewer than min_pairs matched
     pairs (the paper: five) the mean of all of them; with no matched pair
     the pair is untracked (the paper re-acquires; here the interval has no
     drift and the console says so);
  7. the displacement into metres, R_A (dx xscale_A, dy yscale_A), the
     features' displacement from A to B in the absolute frame, and the
     velocity v = displacement / (t_mid,B - t_mid,A), the two-point form of
     the paper's regression of the centroids against the image timestamps.

The baseline correction (Drift.result, corrected_topo, row_shear, map_shift,
sweep_lines): within a file the drift is linear in time, d(t) - d(t_0) =
v (t - t_0).  A spectra file takes the pair (the last topo of its segment
before it, the first topo of its segment after it): every sweep of a point
sits at its recorded position minus v (t_k - t_0), t_k its own time
(Sm4File.sweep_times) and t_0 the first sweep's, so the first stays at the
recorded position (the file's absolute anchor); every energy point of a
sweep sits at -v (t - t_sweep start) relative to the sweep's start (the
drift lines are straight; for the heatmap rows the along-line component,
for the map pixels the full vector); every topo pixel at -v (t_ij - t_first
pixel) and the image resampled onto its own grid.  A topo takes its pair
with the next topo of the segment, the last topo of a segment the pair with
the previous one.

Grid files (Drift.grid_chain; the forward pass of a line is topo point,
spectrum at every point, minutes to hours long; the reverse pass a plain
topo scan back at the setpoint, seconds long; the tip is at the next line's
start a fraction of a second after the return point).  The clock is the
file's derived timing (Sm4File.grid_timing, "The sweep clock" in sm4file:
the sample time per energy point from the file span less the return
passes over the points of the started spectra, sweep after sweep).
Within a forward line every point (its topo pixel and its spectrum) sits at
its recorded position plus the line's anchor shift minus the drift since
the line's first point, by the points' own times.  The reverse pass of a
line carries no time: the whole row sits rigidly at the forward end's
shift, so it starts at the forward end and runs back at the recorded
spacing to the return point.  The link to the next line carries no drift
term: its first point sits at the recorded displacement from the return
point (or from the forward end without a reverse page), so the next line's
anchor shift is the previous forward end's.  The first kept spectrum of
the file sits at its recorded position.  The grid's own topo (scan area,
Topo window, Map window) is placed by the same shifts.  When a file's after
topo is missing (the last spectra of
a segment) or its pair is untracked, the velocity of the last tracked pair
ending at or before the before topo is used, but only when it is close to
the velocity of the tracked pair before that: |v_last - v_prev| <=
extrapolate_tol x max(|v_last|, |v_prev|); otherwise, and with fewer than
two tracked pairs behind the file, the file gets no correction.  The
console says which rule applied to every file.

Manual drift (the 'manual drift' toggle, separate from the auto drift of the
tracker, usable with the auto drift off; it overrides the auto velocity
wherever it covers a file).  The MATLAB script's method
(plot_sm4_topo_and_spec_21_with_scrollsubplot_170_2014a.m, "Drift controls")
with several marker pairs: the operator places markers on the same surface
features in two topographies (the drift window; the 'Markers' slider sets
the number of pairs), each marker carrying its topo,
its absolute position, and the acquisition time of the pixel it sits on
(Sm4File.pixel_times); the k-th marker on the earlier topo pairs with the
k-th on the later one.  Each pair is a two-point measurement, d_k = v
(t_B,k - t_A,k), and the span's velocity is the least-squares fit over the
pairs with their own times, v = sum_k dt_k d_k / sum_k dt_k^2
(fit_marker_velocity), which models the squash and stretch of the marker
pattern between two scans of different size, speed, or direction: a feature
scanned late in B and early in A has drifted longer than one scanned the
other way round.  With one pair this is exactly the MATLAB two-point slope.
Every pair's dx, dy, dL, dt, v_x, v_y, v_L, and angle are reported (the
MATLAB readout) with its residual d_k - v dt_k; a residual pattern that
still stretches one way after the times are in is not drift and is left
alone.  The span's velocity covers every file from the earlier topo to the
later one, both included; of overlapping spans the narrowest one containing
the file applies.  A file no span covers keeps the auto velocity: only the
files between a pair whose velocity was applied by hand keep that hand
velocity, and the files after the last topo of a segment follow the auto
extrapolation rule as before (an applied span never extrapolates).
The MATLAB "drift for first point" is the 'anchor to topo' toggle (off by
default): on, a spectra file is also shifted rigidly by -v (t_0 - t_topo
start), t_topo the start of the topo under it, to where the surface sat when
that topo began; off, the file's first spectrum stays at its recorded
position.  The MATLAB "deform topo" is the topo skew above, applied with the
manual velocity when manual drift is on (MATLAB anchors the skewed image at
the scan's end, this package at its first pixel: a rigid v T apart).
The applied spans are the 'drift' section of the settings store
(settings_<stamp>.json, caches.Settings), never touched by the tracking;
the console names the source of every file's velocity
('auto (pair)', 'auto (next)', 'auto (previous)', 'auto (extrapolated)',
'auto (confirmed)', 'manual (markers)', 'manual (typed)'; PairDrift's rule).
The drift window (DriftWindow: the Browser's 'drift' button for the current
file, the 'drift window' button of the Line and Map windows) shows the pair
a file's auto velocity comes from side by side (Drift.window_pair: a spectra
file's before and after topos; a topo and the next of its segment, else the
previous and it), the panels in calibrated nm with each topo image and each
other topo's recorded frame (a thin black outline on top of its panel) at
the scale in force at its own file's time, and takes the marker pairs for
exactly that span; its one velocity box, 'v (nm/h, cal)', is what gets
applied: it is typed and shown in calibrated nm/h (the window's file's
scale), stored nominal (recorded metres), and every readout shows the
calibrated value with the nominal beside it.  Its '< before' / 'before >' and
'< after' / 'after >' buttons step either topo through the segment in time
order, and the box then applies to the span between the topos shown, which
covers every file between them.  The Browser applies the manual
velocities wherever they cover a file; the Browser, Topo, Line, and Map
windows each have their own 'auto drift' and 'manual drift' toggles.

Seeded markers, the velocity box, 'set from markers', 'apply velocity'.
A marker clicked on one topo of the pair seeds its partner on the other
topo where the velocity governing the pair puts the same feature
(DriftWindow._seed_velocity: the pair's applied span, else the file's
velocity with both sources on, else the pair's tracked velocity; nothing
without one): xy_B = xy_A + v (t_B -
t_A), t_B the pixel time at the seeded spot (seed_partner; three fixed-point
passes, since v x row time is tiny; no seed when the spot falls outside the
other topo).  Dragging the clicked marker re-seeds an unmoved partner;
dragging the partner adjusts it ("moved" = more than half a pixel of its topo
from the seeded spot).  The markers are a draft: placing, dragging,
'Markers', and 'clear markers' change only the draft and the readout.  'set
from markers' writes the draft's velocity into the 'v (nm/h, cal)' box (every
pair seeded by an auto velocity and unmoved: the pair's tracked velocity
itself, marked confirmed; otherwise the least-squares fit of the pairs) and
stores nothing; typing
into the box only fills it.  'apply velocity' stores the box as the span of
the window's pair, replacing what that pair had: kind 'confirmed' when the
box still holds the unmoved seeded value (source 'auto (confirmed)'),
'adjusted' when it still holds the pairs' fit (source 'manual (markers)',
the markers stored with it), 'typed' when it was typed or edited (source
'manual (typed)', the velocity itself stored); an empty box applied removes
the pair's span, and the files between the two topos go back to auto.  A
confirmed span is a stored span like the others, governed by the 'manual
drift' toggle (also with auto drift off); its drift lines take
plots.CONFIRMED_DRIFT_COLOR (auto white, manual red).  Each span carries its
kind in the settings store; a span without one (carried over from an older
drift cache) is adjusted, since it was hand-placed.  Pair k of a span has its own
marker colour in both marker windows (plots.MARKER_COLORS, red, green, blue,
then yellow, magenta, cyan, orange, purple: the eight chromatic basic colour
terms of Boynton and Olson that read apart on a topo, so 'Markers' runs 1 to
8); the earlier marker is a circle, the later one a square; a seeded,
unmoved marker is hollow.

Segments (Drift.index_segments): the directory's topos cut at the long-range
moves, a jump of the recorded scan centre above move_min_m (1 um) between
consecutive topos; only pairs within a segment are tracked, and a file's
before and after topos are looked up within its segment (sites.py groups
its regions by the same segments through seg_first).

The stripe matching (drift_correct_grid; the stripe window, stripe.py, sets
a grid's fast-axis state per slow-axis setting, Drift.stripe_sets: off, the
velocity alone; auto, the matching; manual, the matching with hand-moved
line ends and return points, Drift.stripe_points): with it on, a grid's
fast-axis placement comes from matching the lines' own topo stripes against the
earlier topo (the most recent one before the file whose footprint holds the
first spectrum): the lines placed by their extrema (maxima on maxima,
minima on minima) against the topo's profile along each line; the stretch
of each pass from its matched extrema where it has three or more pairs,
else from the ratio of the reference period to the period the stripe's own
extrema show, the reference period being the extrema spacing of the topo
profile along the part of the line that lies on a topo, and for a line off
every topo the spacing measured on the other lines, else on the reference
topos along the fast axis (no Fourier transform anywhere); the slow axis
stays the baseline v, and the within-sweep shift along the line of a line
with a measured forward factor is that line's own forward compression v_i =
(1 - b_f) L_f / T_f (Drift.line_velocities), the baseline across it.  The
stripe matching informs nothing: it never changes v.  Off, both axes come
from the baseline.  A manual point replaces a line's forward end (its
forward factor follows) or its return point (its reverse pass stretches,
the next line's start moves with it), and no other point moves
(drift_correct_grid's `points`).

Locations.  Spectra sit at their recorded locations and topo pixels at the
locations their page's geometry records (Sm4File.pixel_xy_m); a spectrum's
topo pixel is the one recorded at its location (Sm4File.spectrum_pixels),
never pixel j for spectrum j; the reverse page has no spectroscopy.
Alignment of topo to spectra and of topo to topo is done in those recorded,
drift-uncorrected positions; the drift correction is applied afterwards.

Stores (caches.py): the tracked pairs (their displacement, time separation,
keypoint, match, and cluster counts, note) and every loaded topo's times are
the 'pairs' section of the computed store (computed_<stamp>.json, format
PAIRS_FORMAT), written once at the end of a build, so a later startup
tracks only pairs not in it; a section of another format or loader is
rebuilt.  The manual drift (the applied spans with their markers, kind,
and velocity, and the stripe states and points) is the 'drift' section of
the settings store (settings_<stamp>.json), written on every change and
never discarded.  The tracking never touches the settings, and no setting
enters the computed section.
"""

if __package__ in (None, ''):
    # Run as a file (Spyder F5): make the package importable from wherever its folder sits.
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json
import os
import warnings
from datetime import datetime
import numpy as np
import cv2
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import map_coordinates
from scipy.cluster.vq import kmeans2
from sm4_Browser import qtplot as plt
from sm4_Browser import sm4file


DRIFT_PARAMS = {
    # --- the pair tracker (Diao et al. 2023; see the module docstring) ---
    'smooth_kernel_px': 11,    # the Gaussian smoothing kernel before feature detection (pixels, odd)
    'smooth_sigma_px': 1.0,    # and its standard deviation (pixels)
    'cluster_d_m': 1.0e-9,     # a displacement cluster's mean distance to its centroid must be within this (metres,
    #                            absolute; the paper's 1 nm), converted to pixels per pair
    'min_pairs': 5,            # with fewer matched pairs than this their plain mean is the displacement (no clustering)
    'kmeans_seed': 0,          # the k-means++ start is seeded, so a pair tracks the same way every run
    # --- the baseline model ---
    'move_min_m': 1.0e-6,      # a jump of the recorded scan centre (absolute, metres) between consecutive topos
    #                            above this is a long-range move (a segment boundary)
    'extrapolate_tol': 0.5,    # a file without a usable after topo takes the last tracked pair's velocity when that is
    #                            within this fraction of the larger magnitude of the two last velocities
    # --- the stripe matching of grid lines (the stripe window; drift_correct_grid) ---
    'sg_window': 5,            # Savitzky-Golay window (points) before peak finding
    'sg_poly': 2,              # Savitzky-Golay polynomial order
    'prominence_frac': 0.20,   # peak prominence threshold as a fraction of the profile range
    'peak_accept': 0.60,       # correlation peaks >= this fraction of the best are considered (stripe_lag)
    'profile_oversample': 4,   # image profile samples per spectrum spacing
    'profile_reach': 2.0,      # image profile sampled from -reach_back L to +reach L (L = nominal line length)
    'profile_reach_back': 1.0,
    'stretch_limits': (0.33, 3.0),   # a fitted stretch outside this range is treated as no match
    'min_extrema_pairs': 1,    # matched extrema a line needs for the image to place it (a stripe partly on a topo counts)
    'reference_max_topos': 8,  # the most recent earlier topos reaching the grid that form its stripe reference
    'period_agree': 0.30,      # the spacings of a stripe's extrema count as consistent within this fraction
    'area_profiles': 5,        # profiles across each reference topo for the scan-area extrema spacing (period_along)
}


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def factor_text(b, digits=3):
    """The console wording of a stripe's correction factor b: 'compressed to b' below 1, 'stretched to b' above, 'factor 1' at 1."""
    if b < 1.0:
        return f"compressed to {b:.{digits}f}"
    if b > 1.0:
        return f"stretched to {b:.{digits}f}"
    return f"factor {b:.{digits}f}"


def sample_topo(topo, img, xy_m):
    """Heights of a topo image (raw pixel order, NaN where not acquired) at absolute positions (M, 2); NaN outside."""
    uv = topo.to_frame_m(xy_m)
    j = uv[:, 0] / topo.xscale_m + (topo.nx - 1) / 2.0
    i = uv[:, 1] / topo.yscale_m + (topo.ny - 1) / 2.0
    return map_coordinates(img, [i, j], order=1, mode='constant', cval=np.nan)


def resample_to_frame(src, src_img, dst):
    """The source image resampled onto the destination image's pixel grid (NaN where it does not reach)."""
    jj, ii = np.meshgrid(np.arange(dst.nx), np.arange(dst.ny))
    u = (jj - (dst.nx - 1) / 2.0) * dst.xscale_m
    v = (ii - (dst.ny - 1) / 2.0) * dst.yscale_m
    uv = np.column_stack([u.ravel(), v.ravel()])
    xy = uv @ dst.rotation().T + np.asarray(dst.center_m)
    return sample_topo(src, src_img, xy).reshape(dst.ny, dst.nx)


def _deplane(a):
    """A plane fitted to the finite pixels removed; NaN kept."""
    ny, nx = a.shape
    ok = np.isfinite(a)
    J, I = np.meshgrid(np.arange(nx), np.arange(ny))
    A = np.c_[J.ravel(), I.ravel(), np.ones(nx * ny)]
    if ok.sum() < 3:
        return a
    coef, *_ = np.linalg.lstsq(A[ok.ravel()], a.ravel()[ok.ravel()], rcond=None)
    return a - (A @ coef).reshape(ny, nx)


# ---------------------------------------------------------------------- #
#  The pair tracker (Diao et al. 2023 with KAZE features, cross-checked   #
#  matching, k-means majority)                                            #
# ---------------------------------------------------------------------- #

def kaze():
    """
    The KAZE detector at OpenCV's defaults: cv2.KAZE_create in OpenCV 4.x
    (the opencv-python package), cv2.xfeatures2d.KAZE_create in OpenCV 5.x,
    where KAZE moved to the contrib module (the opencv-contrib-python
    package).  Raises ImportError naming the package when neither is there.
    """
    if hasattr(cv2, 'KAZE_create'):
        return cv2.KAZE_create()
    xf = getattr(cv2, 'xfeatures2d', None)
    if xf is not None and hasattr(xf, 'KAZE_create'):
        return xf.KAZE_create()
    raise ImportError(f"OpenCV {cv2.__version__} has no KAZE: install opencv-python 4.x (pip install \"opencv-python<5\") "
                      "or, for OpenCV 5, opencv-contrib-python (KAZE lives in its xfeatures2d module).")


def preprocess_image(img, params=DRIFT_PARAMS):
    """
    An image (NaN = not acquired) prepared for the feature detector, as
    Diao et al. prepare theirs: a plane fitted to the acquired pixels
    removed (_deplane), the missing pixels set to zero, 2D Gaussian
    smoothing with a smooth_kernel_px square kernel of smooth_sigma_px
    standard deviation (cv2.GaussianBlur), then the acquired pixels scaled
    to 0..1.  Returns (the float32 image, the uint8 mask of the acquired
    pixels eroded by the kernel's half-width, so that no keypoint sits on
    the edge of the acquired area where the smoothing mixes in zeros), or
    (None, None) when fewer than three pixels are acquired or the image is
    flat.
    """
    a = np.asarray(img, dtype=np.float64)
    ok = np.isfinite(a)
    if ok.sum() < 3:
        return None, None
    k = int(params['smooth_kernel_px'])
    if k % 2 == 0:
        k += 1
    k = max(k, 1)
    d = np.where(ok, _deplane(a), 0.0).astype(np.float32)
    sm = cv2.GaussianBlur(d, (k, k), float(params['smooth_sigma_px'])) if k > 1 else d
    lo, hi = float(np.min(sm[ok])), float(np.max(sm[ok]))
    if not hi > lo:
        return None, None
    out = ((sm - lo) / (hi - lo)).astype(np.float32)
    out[~ok] = 0.0
    mask = ok.astype(np.uint8) * 255
    half = k // 2
    if half > 0:
        mask = cv2.erode(mask, np.ones((2 * half + 1, 2 * half + 1), dtype=np.uint8))
    return out, mask


def cluster_displacements(disp, d_px, params=DRIFT_PARAMS):
    """
    The displacement of a pair from its matched feature points, Diao et
    al.'s clustering and majority rule: k-means (scipy.cluster.vq.kmeans2,
    k-means++ start seeded with kmeans_seed) on the (x, y) displacements from
    k = 2 upward until every cluster's mean distance of its points to its
    centroid is within d_px (k capped at the number of points); the
    centroid of the largest cluster is the displacement, a tie going to the
    cluster with the smaller mean distance.  With fewer than min_pairs
    points the plain mean of all of them (k = 0).  Returns ((dx, dy) in
    pixels, k, the size of the chosen cluster, its mean distance in pixels).
    """
    disp = np.asarray(disp, dtype=np.float64).reshape(-1, 2)
    n = len(disp)
    if n == 0:
        return None, 0, 0, None
    if n < int(params['min_pairs']):
        c = disp.mean(axis=0)
        spread = float(np.mean(np.linalg.norm(disp - c, axis=1)))
        return c, 0, n, spread
    if np.allclose(disp, disp[0]):
        return disp[0].copy(), 1, n, 0.0           # every pair agrees exactly: one cluster
    best = None
    for k in range(2, n + 1):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                cent, lab = kmeans2(disp, k, minit='++', seed=int(params['kmeans_seed']))
            except TypeError:                      # an older SciPy without the seed keyword
                np.random.seed(int(params['kmeans_seed']))
                cent, lab = kmeans2(disp, k, minit='++')
        clusters = []
        for c in range(k):
            members = disp[lab == c]
            if len(members) == 0:
                continue
            g = members.mean(axis=0)
            spread = float(np.mean(np.linalg.norm(members - g, axis=1)))
            clusters.append((len(members), spread, g))
        if not clusters:
            continue
        if all(spread <= d_px for _, spread, _ in clusters):
            m, spread, g = max(clusters, key=lambda c: (c[0], -c[1]))
            best = (g, k, m, spread)
            break
    if best is None:                               # every point its own cluster passes at k = n; this is unreachable
        c = disp.mean(axis=0)
        return c, n, n, float(np.mean(np.linalg.norm(disp - c, axis=1)))
    return best


def track_pair(earlier, later, params=DRIFT_PARAMS):
    """
    The drift between two topo files by feature-point matching (the module
    docstring's steps 1 to 7): the later image resampled onto the earlier
    one's grid, both preprocessed (preprocess_image), KAZE keypoints and
    descriptors on the pixels both acquired, brute-force L2 matching with
    crossCheck, the matched pairs' displacements clustered
    (cluster_displacements), the majority centroid taken into the absolute
    frame.  Returns a dict: 'delta_m' (the features' displacement from the
    earlier to the later image, metres, absolute frame; None when
    untracked), 'dt_s' (the mid-time separation), 'v' (m/s; None when
    untracked), 'n_a', 'n_b' (keypoints), 'n_match' (cross-checked pairs),
    'k' (clusters; 0 when the plain mean was used), 'n_major' (points in
    the chosen cluster), 'spread_px' (its mean distance to the centroid),
    'd_px' (the cluster distance in pixels), 'earlier', 'later' (names),
    'note' (why untracked, else '').
    """
    out = {'delta_m': None, 'dt_s': float(later.mid_time - earlier.mid_time), 'v': None,
           'n_a': 0, 'n_b': 0, 'n_match': 0, 'k': 0, 'n_major': 0, 'spread_px': None, 'd_px': None,
           'earlier': earlier.name, 'later': later.name, 'note': ''}
    try:
        A = earlier.topo_raw('forward')
        B = resample_to_frame(later, later.topo_raw('forward'), earlier)
    except Exception as exc:
        out['note'] = f"images not comparable ({exc})"
        return out
    both = np.isfinite(A) & np.isfinite(B)
    if not both.any():
        out['note'] = "the two images do not overlap"
        return out
    a, mask_a = preprocess_image(np.where(both, A, np.nan), params)
    b, mask_b = preprocess_image(np.where(both, B, np.nan), params)
    if a is None or b is None:
        out['note'] = "too few overlapping pixels, or a flat image"
        return out
    mask = ((mask_a > 0) & (mask_b > 0)).astype(np.uint8) * 255
    if not mask.any():
        out['note'] = "the overlap is thinner than the smoothing kernel"
        return out
    detector = kaze()
    kp_a, des_a = detector.detectAndCompute(a, mask)
    kp_b, des_b = detector.detectAndCompute(b, mask)
    out['n_a'], out['n_b'] = len(kp_a), len(kp_b)
    if des_a is None or des_b is None or len(kp_a) == 0 or len(kp_b) == 0:
        out['note'] = f"no feature point in {'the earlier' if len(kp_a) == 0 else 'the later'} image"
        return out
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(des_a, des_b)
    out['n_match'] = len(matches)
    if not matches:
        out['note'] = "no matched feature point"
        return out
    disp = np.array([[kp_b[m.trainIdx].pt[0] - kp_a[m.queryIdx].pt[0],
                      kp_b[m.trainIdx].pt[1] - kp_a[m.queryIdx].pt[1]] for m in matches], dtype=np.float64)
    px = 0.5 * (abs(float(earlier.xscale_m)) + abs(float(earlier.yscale_m)))
    d_px = float(params['cluster_d_m']) / px if px > 0 else float('inf')
    out['d_px'] = d_px
    c, k, n_major, spread = cluster_displacements(disp, d_px, params)
    out['k'], out['n_major'], out['spread_px'] = int(k), int(n_major), spread
    R = earlier.rotation()
    delta = R @ np.array([c[0] * earlier.xscale_m, c[1] * earlier.yscale_m])
    out['delta_m'] = delta
    if out['dt_s'] <= 0:
        out['note'] = "the two images have no time separation"
        return out
    out['v'] = delta / out['dt_s']
    return out


def pair_text(est):
    """One console line for a tracked pair."""
    head = f"{est['earlier']} -> {est['later']}"
    if est.get('v') is None:
        return head + f": untracked ({est.get('note', '')}; {est['n_a']}/{est['n_b']} keypoints, {est['n_match']} matched)"
    d = np.asarray(est['delta_m'], dtype=np.float64) * 1e9
    v = np.asarray(est['v'], dtype=np.float64) * 1e12
    how = ("plain mean" if est['k'] == 0 else f"{est['k']} cluster(s), majority {est['n_major']} of {est['n_match']}")
    spread = est.get('spread_px')
    return (head + f": {est['n_a']}/{est['n_b']} keypoints, {est['n_match']} matched, {how}"
            + (f", spread {spread:.2f} px (d {est['d_px']:.1f} px)" if spread is not None and est.get('d_px') else "")
            + f"; displacement ({d[0]:+.3f}, {d[1]:+.3f}) nm over {est['dt_s'] / 60:.1f} min"
            f" = ({v[0]:+.2f}, {v[1]:+.2f}) pm/s")


def fit_marker_velocity(markers):
    """
    The manual velocity of a marker span (the module docstring, "Manual
    drift"): markers (n, 6) rows of (x_A, y_A, t_A, x_B, y_B, t_B), absolute
    metres and seconds, one row per marker pair; d_k = (x_B - x_A, y_B -
    y_A), dt_k = t_B - t_A; v = sum_k dt_k d_k / sum_k dt_k^2, the
    least-squares velocity over the pairs with their own times (the MATLAB
    two-point slope with one pair).  Pairs with dt_k <= 0 are left out.
    Returns (v (2,) m/s or None, [per-pair dicts: 'dx', 'dy', 'dL' (m),
    'dt' (s), 'vx', 'vy', 'vL' (m/s), 'alpha' (deg), 'rx', 'ry' (the
    residual d_k - v dt_k, m), 'used'], note).
    """
    m = np.asarray(markers, dtype=np.float64).reshape(-1, 6)
    pairs = []
    if len(m) == 0:
        return None, pairs, "no marker pair"
    d = m[:, 3:5] - m[:, 0:2]
    dt = m[:, 5] - m[:, 2]
    used = np.isfinite(dt) & (dt > 0) & np.all(np.isfinite(d), axis=1)
    v = None
    if used.any():
        v = (dt[used][:, None] * d[used]).sum(axis=0) / float(np.sum(dt[used] ** 2))
    for k in range(len(m)):
        rec = {'dx': float(d[k, 0]), 'dy': float(d[k, 1]), 'dL': float(np.hypot(d[k, 0], d[k, 1])), 'dt': float(dt[k]),
               'used': bool(used[k]), 'vx': np.nan, 'vy': np.nan, 'vL': np.nan, 'alpha': np.nan, 'rx': np.nan, 'ry': np.nan}
        if used[k]:
            vk = d[k] / dt[k]
            rec.update({'vx': float(vk[0]), 'vy': float(vk[1]), 'vL': float(np.hypot(vk[0], vk[1])),
                        'alpha': float(np.degrees(np.arctan2(vk[1], vk[0])))})
            if v is not None:
                r = d[k] - v * dt[k]
                rec.update({'rx': float(r[0]), 'ry': float(r[1])})
        pairs.append(rec)
    note = "" if v is not None else "no pair with a positive time separation"
    return v, pairs, note


def marker_pair_text(k, rec, scale=None):
    """
    One readout line per marker pair: the MATLAB readout (nm, h:m:s, nm/h, degrees) and the
    residual, nominal (the recorded metres); with `scale` (the file's scale factor) every length
    and velocity is followed by its calibrated value in brackets, the nominal divided by the scale.
    """
    dt = rec['dt']
    hms = f"{int(dt // 3600):d}:{int((dt % 3600) // 60):02d}:{dt % 60:04.1f}" if np.isfinite(dt) else "n/a"

    def nm(x, fmt='+.3f'):
        s = f"{x * 1e9:{fmt}} nm"
        return s if scale is None else s + f" [{x * 1e9 / scale:{fmt}} cal]"

    def nmh(x, fmt='+.3f'):
        s = f"{x * 3.6e12:{fmt}} nm/h"
        return s if scale is None else s + f" [{x * 3.6e12 / scale:{fmt}} cal]"

    line = f"pair {k + 1}: dx {nm(rec['dx'])}, dy {nm(rec['dy'])}, dL {nm(rec['dL'], '.3f')}, dt {hms}"
    if rec['used']:
        line += (f"; v_x {nmh(rec['vx'])}, v_y {nmh(rec['vy'])}, v_L {nmh(rec['vL'], '.3f')}, "
                 f"alpha {rec['alpha']:.1f} deg; residual ({rec['rx'] * 1e9:+.3f}, {rec['ry'] * 1e9:+.3f}) nm")
    else:
        line += "; left out (no positive time separation)"
    return line


def pixel_time_at(topo, xy_m, direction='forward'):
    """
    The acquisition time of the pixel of a topo (Sm4File) under the
    absolute position xy_m (metres), from Sm4File.pixel_times of the given
    pass (the nearest pixel, clipped to the page); the page's earliest time
    when that pixel has none, the file's start time when none has.
    """
    uv = topo.to_frame_m(np.asarray(xy_m, dtype=np.float64))[0]
    j = int(np.clip(round(uv[0] / topo.xscale_m + (topo.nx - 1) / 2.0), 0, topo.nx - 1))
    i = int(np.clip(round(uv[1] / topo.yscale_m + (topo.ny - 1) / 2.0), 0, topo.ny - 1))
    tt = topo.pixel_times(direction)
    val = float(tt[i, j])
    if not np.isfinite(val):
        val = float(np.nanmin(tt)) if np.isfinite(tt).any() else float(topo.start_time)
    return val


def seed_partner(topo_from, topo_to, xy_m, t_from, v, passes=3):
    """
    Where the auto velocity puts a feature clicked at xy_m (absolute metres,
    pixel time t_from) of one topo on the other topo of its pair (the module
    docstring, "Seeded markers"): xy_to = xy_m + v (t_to - t_from), t_to the
    forward-pass pixel time of topo_to at xy_to (pixel_time_at), iterated
    `passes` times from t_to at xy_m.  Returns (xy_to (2,), t_to), or None
    when the spot lies outside topo_to.
    """
    xy_m = np.asarray(xy_m, dtype=np.float64).reshape(2)
    v = np.asarray(v, dtype=np.float64).reshape(2)
    xy_to = xy_m.copy()
    t_to = float(t_from)
    for _ in range(max(int(passes), 1)):
        if not topo_to.contains_m(xy_to):
            return None
        t_to = pixel_time_at(topo_to, xy_to, 'forward')
        xy_to = xy_m + v * (t_to - float(t_from))
    if not topo_to.contains_m(xy_to):
        return None
    return xy_to, t_to


class PairDrift:
    """
    The drift model of one file: a constant velocity v (m/s, absolute frame,
    the surface's features' displacement per second) from one tracked topo
    pair, so d(t) = v (t - t_ref) and only differences of d enter.
      v      : (2,) metres per second
      t_ref  : the reference time (s since the epoch; the displacement is 0 there)
      rule   : how the velocity was chosen, the 'source' of a result and the
               console's wording: 'auto (pair)', 'auto (next)', 'auto (previous)',
               'auto (extrapolated)' (the tracker: model_for_spectra, model_for_topo),
               'auto (confirmed)', 'manual (markers)', 'manual (typed)' (a stored
               span: _manual_model), 'none (stripe match only)' (a zero velocity,
               result)
      source : the pair(s) the velocity came from, for the console
    """

    def __init__(self, v, t_ref, rule, source):
        self.v = np.asarray(v, dtype=np.float64).reshape(2)
        self.t_ref = float(t_ref)
        self.rule = rule
        self.source = source

    def displacement(self, t):
        """d(t) = v (t - t_ref): t a number or an array of any shape; the result has one more axis of length 2 for an array; NaN where t is."""
        t = np.asarray(t, dtype=np.float64)
        scalar = t.ndim == 0
        tt = t.reshape(-1)
        out = (tt - self.t_ref)[:, None] * self.v[None, :]
        out = np.where(np.isfinite(tt)[:, None], out, np.nan)
        if scalar:
            return out[0]
        return out.reshape(t.shape + (2,))

    def relative(self, t_ref):
        """The function t -> d(t) - d(t_ref) (t a number or an array)."""
        d_ref = self.displacement(float(t_ref))
        return lambda t: self.displacement(t) - d_ref

    def text(self, scale=None):
        """
        The velocity on the console: nominal (the recorded metres, as stored and applied), and with
        `scale` (the file's scale factor, scales.Scales) the calibrated value beside it, the nominal
        divided by the scale (the scan area's own units).
        """
        v = self.v * 1e12
        out = (f"v = ({v[0]:+.2f}, {v[1]:+.2f}) pm/s = ({self.v[0] * 3.6e12:+.3f}, {self.v[1] * 3.6e12:+.3f}) nm/h nominal")
        if scale is not None:
            out += f", calibrated (/ {float(scale):g}) ({self.v[0] * 3.6e12 / scale:+.3f}, {self.v[1] * 3.6e12 / scale:+.3f}) nm/h"
        return out + f", from {self.source}"


PAIRS_FORMAT = 6          # the 'pairs' section of the computed store: the tracked pairs (with the files' signatures) and the topo times


class TrackedPairs:
    """
    The tracked topo pairs and the topo times of one experiment, the 'pairs'
    section of the computed store (caches.Computed, format PAIRS_FORMAT):
      pairs : every tracked topo pair (track_pair's result) by
              '<earlier name>|<later name>', each with the two files'
              signatures ('sig': [size, modification time] of each,
              sm4file._file_sig), so a later startup tracks only pairs not
              in it or whose files changed since
      topos : every topo loaded during a build: its start, end, and mid
              times (seconds since the epoch), so TopoRef has the times
              without loading the file again
    Computed from the files alone: no setting enters here (the manual spans
    and the stripe states live in the settings store, Drift.manual_record).
    record() is the section's content as stored; Drift.build writes it once
    at the end of the tracking, and Drift.pair writes it again whenever a
    pair is tracked later in the session.
    """

    def __init__(self, content=None):
        content = content or {}
        self.pairs = dict(content.get('pairs') or {})
        self.topos = dict(content.get('topos') or {})

    def record(self):
        """The section's content as stored (the dicts themselves)."""
        return {'pairs': self.pairs, 'topos': self.topos}

    @staticmethod
    def pair_key(a_name, b_name):
        """The key of a pair in `pairs`: '<earlier name>|<later name>'."""
        return f"{a_name}|{b_name}"

    def get_pair(self, a_name, b_name, a_sig=None, b_sig=None):
        """
        The cached track_pair result of a pair as the dict track_pair returns, or None: none stored,
        or (with the files' signatures given, sm4file._file_sig) a stored signature that differs
        from the file's now, so a file replaced under the same name is tracked again.
        """
        rec = self.pairs.get(self.pair_key(a_name, b_name))
        if rec is None:
            return None
        if (a_sig is not None or b_sig is not None) and rec.get('sig') != [a_sig, b_sig]:
            return None
        out = {'delta_m': None, 'dt_s': float(rec.get('dt_s', 0.0)), 'v': None,
               'n_a': int(rec.get('n_a', 0)), 'n_b': int(rec.get('n_b', 0)), 'n_match': int(rec.get('n_match', 0)),
               'k': int(rec.get('k', 0)), 'n_major': int(rec.get('n_major', 0)), 'spread_px': rec.get('spread_px'),
               'd_px': rec.get('d_px'), 'earlier': a_name, 'later': b_name, 'note': rec.get('note', '')}
        if rec.get('delta_m') is not None:
            out['delta_m'] = np.asarray(rec['delta_m'], dtype=np.float64)
            if out['dt_s'] > 0:
                out['v'] = out['delta_m'] / out['dt_s']
        return out

    def put_pair(self, a_name, b_name, est, a_sig=None, b_sig=None):
        """A track_pair result stored under the pair's key as plain numbers and lists (JSON): the displacement, the time separation, the counts, the spread and cluster distance, the note, and the two files' signatures ('sig'); not the velocity (get_pair recomputes it)."""
        self.pairs[self.pair_key(a_name, b_name)] = {
            'sig': [a_sig, b_sig],
            'delta_m': None if est.get('delta_m') is None else [float(x) for x in est['delta_m']],
            'dt_s': float(est.get('dt_s', 0.0)), 'n_a': int(est.get('n_a', 0)), 'n_b': int(est.get('n_b', 0)),
            'n_match': int(est.get('n_match', 0)), 'k': int(est.get('k', 0)), 'n_major': int(est.get('n_major', 0)),
            'spread_px': None if est.get('spread_px') is None else float(est['spread_px']),
            'd_px': None if est.get('d_px') is None else float(est['d_px']), 'note': est.get('note', '')}


# ---------------------------------------------------------------------- #
#  The stripe matching of grid lines (the stripe window, stripe.py)       #
# ---------------------------------------------------------------------- #
#
# One optional correction of a grid file's spectrum positions along the fast
# axis (along a line), on top of the baseline velocity, which owns the slow
# axis (across the lines) and, with the fast-axis state 'off', both axes:
#
#   slow axis (across the lines): the baseline d(t) = v (t - t_0) of the
#       file's topo pair applied per spectrum as -(d(t_k) - d(t_0)) projected
#       on the slow axis, t_k the spectrum's own time and t_0 the first
#       spectrum's, so that the first spectrum stays at its recorded position
#       (the file's absolute anchor) and every later one is placed relative
#       to it by its own time.  The fast-axis component of the baseline is
#       discarded here (printed as a consistency check only).
#   fast axis (along a line): the stretch of the along-line coordinate is
#       the ratio of the reference period along the line to the period the
#       line's own topo stripe shows, both from extrema (maxima and minima:
#       same kinds a period apart, opposite kinds half a period apart,
#       extrema_period; no Fourier transform).  The reference period is the
#       extrema spacing of the image's height profile sampled along the part
#       of the line that lies on a topo; a line off every topo takes the
#       median spacing found on the other lines, else the spacing of the
#       reference topos along the fast axis (StripeReference.period_along).
#       A partial line is matched over its kept range only (first to last
#       kept spectrum), never over the width it would have covered.  With
#       the stretch fixed, the extrema of the forward stripe are matched, in
#       order from the line's start, to the extrema of the image profile
#       along the line's true path, which gives the offset; where that gives
#       three or more pairs the stretch is refitted from the pairs
#       themselves (the period ratio stands otherwise; a pass without a
#       period takes the nearest measured line's factor, else 1).  A line
#       with no image match (off the page) takes the chained start,
#       phase-locked to the previous line's reverse stripe by cross-correlation
#       after the slow-step shift measured on the links the image placed.  The
#       reverse stripe gets its own stretch and is matched the same way from
#       the end of the forward pass backwards, and where it puts the return
#       point is the start of the next line (chained).
#
# Everything is guarded: a step that cannot be done is skipped with a console
# note and the other step still applies.

def profile_extrema(profile, s, params=DRIFT_PARAMS):
    """
    Maxima and minima of a height profile: the finite points are detrended
    (linear), Savitzky-Golay smoothed, and peaks of the profile and of its
    negative found with a prominence threshold of prominence_frac x range,
    each placed to sub-sample accuracy by the parabola through the peak and
    its two neighbours.  Returns a list of (s, 'max' | 'min') sorted by s.
    """
    y = np.asarray(profile, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    ok = np.isfinite(y)
    if np.sum(ok) < 3:
        return []
    y = y[ok]
    s = s[ok]
    y = y - np.polyval(np.polyfit(s, y, 1), s)
    w = int(params['sg_window'])
    if w % 2 == 0:
        w += 1
    if len(y) >= w + 2 and w > params['sg_poly']:
        y = savgol_filter(y, w, int(params['sg_poly']))
    rng = float(np.max(y) - np.min(y))
    if rng <= 0:
        return []
    prom = params['prominence_frac'] * rng
    out = []
    for sign, kind in ((1.0, 'max'), (-1.0, 'min')):
        z = sign * y
        pk, _ = find_peaks(z, prominence=prom)
        for k in pk:
            # Sub-sample position from the parabola through the peak and its neighbours.
            dk = _parabolic(z[k - 1], z[k], z[k + 1]) if 0 < k < len(z) - 1 else 0.0
            ds = (s[k + 1] - s[k]) if (dk >= 0 and k < len(s) - 1) else (s[k] - s[k - 1] if k > 0 else 0.0)
            out.append((float(s[k] + dk * ds), kind))
    out.sort(key=lambda t: t[0])
    return out


def match_extrema(stripe_ext, image_ext, reverse=False, image_limit=None, stretch_limits=(0.33, 3.0),
                  fixed_b=None, offset_tol=None, a_prior=0.0, max_offset=None):
    """
    Pair the stripe's extrema with the image profile's extrema and fit
    s_image = a + b * s_stripe.

    Forward: both sequences run upward from the line's start.  Reverse: both
    run downward from the end of the forward pass (image extrema at or below
    image_limit only).  Candidate alignments skip 0, 1 or 2 leading stripe
    extrema and any number of leading image extrema; from the first pair
    on, extrema are paired in order while their kinds agree (max to max, min
    to min), stopping at the first disagreement.  With max_offset given, an
    alignment is a candidate only when its offset a lies within max_offset
    of a_prior, the position the pass is predicted to have (the chained
    start, a = 0, for a forward pass; the return point from the stretches
    for a reverse pass): the image fixes the phase within half a period, it
    never relocates a pass by whole periods onto the page.  Among the
    candidates the one with the most pairs wins, ties broken by the fit
    residual (residuals within offset_tol count as equal, so that on a
    periodic surface, where every alignment one period apart fits equally
    well, the tie goes on), the offset nearest a_prior, and a stretch
    closest to 1; stretches outside stretch_limits are not accepted.
    With fixed_b given (the stretch from the lattice period, see
    period_stretch) only the offset a is fitted, by least squares with that
    slope, from one pair on.  Otherwise with two or more pairs both a and b
    are free, with one pair b = 1.  With no candidate the mapping is the
    identity (b = fixed_b when given) and the pass counts as unmatched.
    Returns (a, b, n_pairs, pairs) with pairs as [(s_stripe, s_image), ...].
    """
    if reverse:
        st = sorted(stripe_ext, key=lambda t: -t[0])
        im = sorted([e for e in image_ext if image_limit is None or e[0] <= image_limit], key=lambda t: -t[0])
    else:
        st = sorted(stripe_ext, key=lambda t: t[0])
        im = sorted(image_ext, key=lambda t: t[0])
    lo, hi = stretch_limits
    best = None
    for skip_s in (0, 1, 2):
        for skip_i in range(len(im)):
            pairs = []
            i, j = skip_s, skip_i
            while i < len(st) and j < len(im) and st[i][1] == im[j][1]:
                pairs.append((st[i][0], im[j][0]))
                i += 1
                j += 1
            if not pairs:
                continue
            x = np.array([q[0] for q in pairs])
            y = np.array([q[1] for q in pairs])
            if fixed_b is not None:
                b = float(fixed_b)
                a = float(np.mean(y - b * x))
                res = float(np.sqrt(np.mean((a + b * x - y) ** 2)))
            elif len(pairs) == 1:
                a, b, res = pairs[0][1] - pairs[0][0], 1.0, 0.0
            else:
                b, a = np.polyfit(x, y, 1)
                if not (np.isfinite(a) and np.isfinite(b)) or not (lo <= b <= hi):
                    continue
                res = float(np.sqrt(np.mean((a + b * x - y) ** 2)))
            res_key = res if offset_tol is None else round(res / offset_tol)
            dev = abs(a - a_prior)
            if max_offset is not None and dev > max_offset:
                continue                                   # a whole-period relocation, not a match
            key = (-len(pairs), res_key, dev, abs(np.log(b)))
            if best is None or key < best[0]:
                best = (key, float(a), float(b), pairs)
    if best is None:
        return 0.0, 1.0 if fixed_b is None else float(fixed_b), 0, []
    _, a, b, pairs = best
    return a, b, len(pairs), pairs


def _ext_text(ext):
    return "[" + ", ".join(f"{s * 1e10:.1f}{'+' if k == 'max' else '-'}" for s, k in ext) + "]"


def _parabolic(cm, c0, cp):
    """Sub-sample offset of a peak from its three samples (0 when they do not form a cap)."""
    d = cm - 2 * c0 + cp
    return 0.0 if not np.isfinite(d) or d >= 0 else float(0.5 * (cm - cp) / d)


def extrema_period(ext, with_count=False):
    """
    The period of an extrema list from every spacing it holds: the spacings
    of successive maxima and of successive minima are full periods, and the
    spacing of every successive pair of opposite kind (a maximum to the next
    minimum, a minimum to the next maximum) is a half period, entered
    doubled.  The median over the pool (None without any spacing, i.e. with
    fewer than two extrema); with_count also returns the number of spacings
    in the pool and their largest / smallest ratio (their consistency).  A
    stripe that holds one void and one molecule, a minimum and a maximum,
    thus has a period.
    """
    spacings = []
    for kind in ('max', 'min'):
        s = [e[0] for e in ext if e[1] == kind]
        spacings.extend(np.diff(s))
    pos = [e[0] for e in ext]
    kinds = [e[1] for e in ext]
    for k in range(len(ext) - 1):
        if kinds[k] != kinds[k + 1]:
            spacings.append(2.0 * (pos[k + 1] - pos[k]))
    if not spacings:
        return (None, 0, np.inf) if with_count else None
    period = float(np.median(spacings))
    if not with_count:
        return period
    spread = float(max(spacings) / min(spacings)) if min(spacings) > 0 else np.inf
    return period, len(spacings), spread


def stripe_period(profile, s, params=DRIFT_PARAMS):
    """
    The period a stripe shows along the line, from its local maxima and
    minima alone (extrema_period of profile_extrema: every spacing, same
    kind as a period and opposite kinds as a half period, the median over
    the pool).  Returns (period or None, text); the text reports the
    number of spacings and whether they agree within period_agree (their
    largest / smallest ratio), so a period resting on disagreeing
    spacings is seen for what it is.
    """
    p_ext, n_sp, spread = extrema_period(profile_extrema(profile, s, params), with_count=True)
    if p_ext is None:
        return None, "no period (fewer than two extrema)"
    agree = n_sp < 2 or np.log(spread) <= np.log(1.0 + params['period_agree'])
    return p_ext, (f"extrema {p_ext * 1e10:.2f} A from {n_sp} spacing(s)"
                   + ("" if agree else f", disagreeing by x{spread:.2f}"))


def period_stretch(measured, references, stretch_limits):
    """
    Stretch b = reference / measured with the strongest reference period
    among `references` ([(period, weight), ...]) whose ratio to the measured
    period lies within stretch_limits: the references are taken in order of
    weight, strongest first (the dominant lattice component sets the period
    the stripe crosses), equal weights in order of the ratio nearest 1, and
    the first within the limits is used.  None when there is no measured
    period, no reference, or every ratio lies outside the limits (then the
    strongest reference is returned with it).  Returns (b or None, reference
    used or None).
    """
    if measured is None or not references:
        return None, None
    lo, hi = stretch_limits
    order = sorted(references, key=lambda t: (-t[1], abs(np.log(t[0] / measured))))
    for ref, _ in order:
        b = ref / measured
        if lo <= b <= hi:
            return float(b), float(ref)
    return None, float(order[0][0])


def stripe_lag(x_a, y_a, x_b, y_b, dx, max_lag, params=DRIFT_PARAMS):
    """
    The lag by which the pattern of profile b (at positions x_b) is shifted
    against profile a (at x_a): the two are interpolated onto one grid of
    spacing dx, detrended, and the normalized cross-correlation over their
    overlap evaluated for lags up to max_lag; among the peaks at least
    peak_accept times the best, the one nearest zero is taken (a periodic
    pattern repeats every period) and refined to sub-sample accuracy.
    Positive: b's pattern sits further along than a's.  Returns (lag, ncc)
    or (None, 0.0) when the overlap is too short.
    """
    x_a, y_a, x_b, y_b = (np.array(v, dtype=np.float64) for v in (x_a, y_a, x_b, y_b))
    for x, y in ((x_a, y_a), (x_b, y_b)):
        o = np.argsort(x)
        x[:], y[:] = x[o], y[o]
    lo = max(x_a[0], x_b[0]) - max_lag
    hi = min(x_a[-1], x_b[-1]) + max_lag
    if hi - lo < 4 * dx:
        return None, 0.0
    grid = np.arange(lo, hi + dx / 2, dx)
    fa = np.interp(grid, x_a, y_a, left=np.nan, right=np.nan)
    fb = np.interp(grid, x_b, y_b, left=np.nan, right=np.nan)
    for f in (fa, fb):
        m = np.isfinite(f)
        if m.sum() >= 2:
            f[m] = f[m] - np.polyval(np.polyfit(grid[m], f[m], 1), grid[m])
    n_lag = int(round(max_lag / dx))
    n_min = max(4, int(0.5 * min(np.isfinite(fa).sum(), np.isfinite(fb).sum())))
    lags = np.arange(-n_lag, n_lag + 1)
    ncc = np.full(len(lags), -np.inf)
    for q, k in enumerate(lags):
        # b shifted by +k samples: compare a[t] with b[t - k]
        if k >= 0:
            a_part, b_part = fa[k:], fb[:len(fb) - k]
        else:
            a_part, b_part = fa[:len(fa) + k], fb[-k:]
        m = np.isfinite(a_part) & np.isfinite(b_part)
        if m.sum() < n_min:
            continue
        u, v = a_part[m] - np.mean(a_part[m]), b_part[m] - np.mean(b_part[m])
        den = np.sqrt(np.sum(u * u) * np.sum(v * v))
        if den > 0:
            ncc[q] = float(np.sum(u * v) / den)
    if not np.any(np.isfinite(ncc)):
        return None, 0.0
    loc = [q for q in range(len(lags)) if np.isfinite(ncc[q])
           and (q == 0 or ncc[q] >= ncc[q - 1]) and (q == len(lags) - 1 or ncc[q] >= ncc[q + 1])]
    best = max(ncc[q] for q in loc)
    accept = [q for q in loc if ncc[q] >= params['peak_accept'] * best]
    q = min(accept, key=lambda q: (abs(lags[q]), -ncc[q]))
    dq = _parabolic(ncc[q - 1], ncc[q], ncc[q + 1]) if 0 < q < len(lags) - 1 else 0.0
    # The loop pairs a[t] with b[t - k], so a positive k means b's pattern sits earlier.
    return float(-(lags[q] + dq) * dx), float(ncc[q])


def _along(xy, origin, fast):
    """The along-line coordinate (metres) of absolute locations (..., 2) from `origin` along the unit vector `fast`."""
    return (np.asarray(xy, dtype=np.float64) - origin) @ fast


def _across(xy, origin, fast):
    """The recorded across-line vector (..., 2) of absolute locations: their offset from `origin` minus its along-line part."""
    d = np.asarray(xy, dtype=np.float64) - origin
    return d - (d @ fast)[..., None] * fast


def alignment_score(stripe_ext, image_ext, a, b, tol):
    """
    How well the stripe's extrema fit the image's under the mapping s_image
    = a + b s_stripe: the number of stripe extrema with an image extremum
    of the same kind within `tol` of their mapped position, and the sum of
    those distances.  Returns (pairs, residual, [(s_stripe, s_image), ...]).
    """
    pairs, res, matched = 0, 0.0, []
    for s, kind in stripe_ext:
        m = a + b * s
        best = None
        for si, ki in image_ext:
            if ki != kind:
                continue
            d = abs(si - m)
            if d <= tol and (best is None or d < best[0]):
                best = (d, si)
        if best is not None:
            pairs += 1
            res += best[0]
            matched.append((s, best[1]))
    return pairs, res, matched


def alignment_offsets(stripe_ext, image_ext, b, prior, window):
    """
    Every offset a within `window` of `prior` at which one of the stripe's
    extrema lands on an image extremum of the same kind under s_image = a +
    b s_stripe (the candidate alignments of the stripe).
    """
    out = []
    for s, kind in stripe_ext:
        for si, ki in image_ext:
            if ki == kind:
                a = si - b * s
                if abs(a - prior) <= window:
                    out.append(float(a))
    return out


def profile_extrema_runs(profile, s, src=None, params=DRIFT_PARAMS):
    """
    profile_extrema over every run of the profile that is finite and comes from one source
    (src: the reference topo of every sample, StripeReference.sample; None: one source), the runs
    joined in order of s: a gap (a never-acquired pixel, the edge of a topo) or a change of topo
    never forms an extremum, and two topos' height offsets never meet in one detrend.
    """
    y = np.asarray(profile, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    src = np.zeros(len(y), dtype=int) if src is None else np.asarray(src, dtype=int)
    ok = np.isfinite(y) & (src >= 0)
    out = []
    i, n = 0, len(y)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and ok[j + 1] and src[j + 1] == src[i]:
            j += 1
        if j - i + 1 >= 3:
            out.extend(profile_extrema(y[i:j + 1], s[i:j + 1], params))
        i = j + 1
    out.sort(key=lambda t: t[0])
    return out


class StripeReference:
    """
    The surface the stripe matching measures the grid lines against: the earlier topos of the
    grid's segment that reach the grid's area, each placed as the scan area places it under the
    drift setting in force (the same setting as the grid: scanlayer.affine_matrix with the topo's
    own velocity of that setting, or no correction with drift none), sampled at any absolute
    position from the topmost one there (the most recent topo whose acquired pixels cover it).
      topos  : loaded Sm4Files, most recent first
      models : their PairDrift under the setting (None: placed at the recorded positions)
    sample(xy) -> (heights (M,), source index (M,), -1 where no topo covers the point); the raw
    forward page in metres, bilinear, as sample_topo samples one topo.  period_along(direction,
    ds) is the extrema spacing of the reference topos along a direction (profiles across every
    topo's placed footprint, the median of their periods): the scan-area spacing a line off every
    topo takes; nearest(xy) the topo whose placed centre is nearest a point.
    """

    def __init__(self, topos, models):
        from sm4_Browser.scanlayer import geometry_record, affine_matrix
        self.entries = []
        for f, model in zip(topos, models):
            rec = geometry_record(f)
            if rec is None:
                continue
            M = affine_matrix(rec, 1.0, 'forward', model)
            try:
                img = f.topo_raw('forward')
            except Exception:
                continue
            self.entries.append({'f': f, 'img': img, 'rec': rec, 'M': M, 'Minv': np.linalg.inv(M),
                                 'model': model})

    @property
    def names(self):
        return [e['f'].name for e in self.entries]

    @property
    def name(self):
        return ", ".join(self.names) if self.entries else "no topo"

    def __bool__(self):
        return bool(self.entries)

    def sample(self, xy_m):
        """(heights (M,), source index (M,)) at absolute positions xy_m (M, 2): the raw forward page of the topmost entry covering each point, bilinear; NaN and -1 where none does."""
        xy = np.atleast_2d(np.asarray(xy_m, dtype=np.float64))
        h = np.full(len(xy), np.nan)
        src = np.full(len(xy), -1, dtype=int)
        for idx, e in enumerate(self.entries):
            need = src < 0
            if not need.any():
                break
            q = xy[need]
            uv = q @ e['Minv'][:2, :2].T + e['Minv'][:2, 2]
            r = e['rec']
            j = uv[:, 0] / r['xscale_m'] + (r['nx'] - 1) / 2.0
            i = uv[:, 1] / r['yscale_m'] + (r['ny'] - 1) / 2.0
            val = map_coordinates(e['img'], [i, j], order=1, mode='constant', cval=np.nan)
            ok = np.isfinite(val)
            sel = np.flatnonzero(need)[ok]
            h[sel] = val[ok]
            src[sel] = idx
        return h, src

    def contains(self, xy_m):
        """True when some reference topo covers the point (absolute metres)."""
        return bool(self.sample(np.reshape(xy_m, (1, 2)))[1][0] >= 0)

    def period_along(self, direction, ds, params=DRIFT_PARAMS):
        """
        The extrema spacing of the reference topos along a unit direction
        (absolute frame): area_profiles profiles per topo, parallel to the
        direction, spread evenly across the topo's placed footprint and
        running its full diagonal, sampled every ds metres (sample), their
        periods from extrema_period of profile_extrema_runs, the median over
        every profile with a period; None without one.
        """
        u = _unit(np.asarray(direction, dtype=np.float64))
        w = np.array([-u[1], u[0]])
        periods = []
        n_prof = max(int(params.get('area_profiles', 5)), 1)
        for idx, e in enumerate(self.entries):
            r = e['rec']
            c = self.centre(idx)
            corners = np.array([[x, y] for x in (-(r['nx'] - 1) / 2.0 * r['xscale_m'], (r['nx'] - 1) / 2.0 * r['xscale_m'])
                                for y in (-(r['ny'] - 1) / 2.0 * r['yscale_m'], (r['ny'] - 1) / 2.0 * r['yscale_m'])])
            corners = corners @ e['M'][:2, :2].T + e['M'][:2, 2]
            half = 0.5 * float(np.max(np.linalg.norm(corners - c, axis=1))) * 2.0
            across = (corners - c) @ w
            offsets = np.linspace(0.8 * across.min(), 0.8 * across.max(), n_prof) if n_prof > 1 else np.zeros(1)
            s = np.arange(-half, half + ds, ds)
            for off in offsets:
                path = c + off * w + s[:, None] * u[None, :]
                prof, src = self.sample(path)
                if np.isfinite(prof).sum() < 4:
                    continue
                ext = profile_extrema_runs(prof, s, src, params)
                per = extrema_period(ext)
                if per is not None:
                    periods.append(float(per))
        return float(np.median(periods)) if periods else None

    def centre(self, idx):
        """The placed centre (absolute metres) of entry idx."""
        e = self.entries[idx]
        return e['M'][:2, 2]

    def nearest(self, xy_m):
        """The index of the entry whose placed centre is nearest xy_m; None without entries."""
        if not self.entries:
            return None
        xy = np.asarray(xy_m, dtype=np.float64).reshape(2)
        return int(np.argmin([np.linalg.norm(self.centre(i) - xy) for i in range(len(self.entries))]))


def drift_correct_grid(grid, prev, drift_fn, t_ref, params=DRIFT_PARAMS, points=None):
    """
    Corrected absolute positions (N, 2; NaN for a spectrum not kept) of a
    grid file against the earlier topo 'prev' (raw heights), with the
    slow-axis drift drift_fn: a function t -> d(t) - d(t_ref), the
    displacement (metres, absolute frame) of the surface's features since
    the reference time t_ref, the time of the grid's first spectrum, so that
    the first spectrum carries no shift (it sits where the file says it
    started, the absolute anchor) and spectrum k carries its own
    d(t_k) - d(t_0) (None = no slow correction).

    Locations.  Every spectrum sits at its recorded location; every pixel of
    the forward and reverse topo pages at the location its page's geometry
    records for it (Sm4File.pixel_xy_m); a spectrum's forward pixel is the
    one recorded at its location (Sm4File.spectrum_pixels, within
    PIXEL_PAIR_TOL_PX), never pixel j for spectrum j.  A forward pixel whose
    spectrum is not kept is not used or drawn; the reverse page has no
    spectroscopy and its acquired pixels are all used.  The along-line
    coordinate of anything is its recorded location projected on the grid's
    fast axis from the grid's first spectrum; its recorded across-line
    offset is kept as recorded, and the slow-axis drift -(d(t) - d(t_0))
    projected on the slow axis is added at its own time.

    Fast axis, per line: the factor b of the along-line coordinate is the
    ratio of the reference period along the line to the period the line's
    own stripe shows (period_stretch; both from extrema, stripe_period), the
    stripe's pixels at their recorded along-line coordinates; where the
    forward stripe then matches three or more image extrema the forward
    factor is refitted from those pairs (below).  The reference period is
    the extrema spacing of the topo profile sampled along the part of the
    line's path that lies on a topo (extrema_period of the image extrema);
    a line off every topo takes the median of the periods so found on the
    other lines, else the spacing of the reference topos along the fast axis
    (StripeReference.period_along).  A pass whose stripe shows no period
    takes the nearest measured line's factor, else 1 (reported as such).  A
    partial line (the scan stopped in it) is measured over its kept range
    only.  Within a line the forward spectra
    sit at the line's corrected start plus b_f times their recorded distance
    from the line's first spectrum; the reverse pixels from the line's
    forward end, where the reverse pass began (the same tip position seconds
    later: the two coincide up to the slow drift of those seconds, carried
    by the per-pixel slow-axis shift), back to the return point, each at the
    forward end minus b' times its recorded distance from the far end, b' =
    (forward end - return point) / recorded reverse length, the factor the
    placement implies (the measured b_r is the grid observation; the two are
    printed side by side per link).

    Manual points (`points`, {'E:i' | 'R:i': metres along the fast axis
    from the grid's first spectrum}, the stripe window's dragged points):
    'E:i' replaces line i's forward end, which sets that line's forward
    factor (the end divided by the kept range) and where its reverse pass
    begins; 'R:i' replaces line i's return point, which stretches its
    reverse pass between the fixed forward end and the point and moves
    line i + 1's start with it (the recorded separation kept, one point),
    line i + 1's own end staying where its point or its match puts it.  No
    other point moves.  Without points the placement is the matching's.

    Placement, line 0: its first spectrum stays at the recorded position
    (the file's absolute anchor); what its forward stripe's match against
    the image would do is printed, not applied.
    Placement, every link (line i's reverse pass, line i + 1's forward
    pass): the last reverse pixel of line i (the return point) and the
    first spectrum of line i + 1 are two points separated by exactly the
    displacement the tip recorded between them, moved as one rigid piece by
    one offset along the line, with no drift term inside that separation;
    moving the return point stretches line i's reverse pass between its
    fixed forward end and the return point (b' above) and moves line i +
    1's forward pass whole.  The prior is the model's: the reverse returns
    the recorded length plus the model's along-line drift during the return
    pass.  Candidates are every return point at which an extremum of the
    reverse stripe (through the b' that return point implies) or of the
    forward stripe (with its measured b_f) lands on an image extremum of the
    same kind, within one reference period of the prior and with b' inside
    stretch_limits;
    each is scored by the pairs it gives on both stripes together
    (alignment_score, within the spectrum spacing), ties by the mean
    residual per pair (residuals within a spectrum spacing count as equal,
    so on a periodic surface the alignments one period apart tie), then by
    nearness to the prior.  A link where only one stripe
    reaches the page is placed by that stripe's pairs, the other following
    at the recorded separation.  Where neither reaches the page the prior
    is refined by the phase lock between the two stripes after the
    slow-step shift measured on the links the image placed (stripe_lag; zero
    before any link was placed by the image).  A line without a reverse pass (one topo page) links
    through its forward end: the next start is the forward end plus the
    recorded displacement from the forward's last pixel to the next first
    spectrum, refined by the forward stripe's pairs alone.  The last line's
    reverse pass, with no link after it, sits at the model's prior.
    Every link is printed: the offset against the prior, the pairs each
    stripe contributed, the reverse factor the placement implies beside the
    measured one, the phase-lock lag when used.

    Returns (xy_corr, report lines, lines, pix): 'lines' holds one dict per
    line with its measured factors (Drift.line_velocities reads the forward
    one, the stripe window the ends): 'i', 'n_kept', 'b_f', 'b_f_measured'
    (False when no factor could be measured and 1 was used, or the factor
    was borrowed), 'b_f_pairs' (the factor refitted from the matched
    extrema, None without three pairs), 'n_fit' (those pairs),
    'b_f_borrowed', 'b_r_borrowed' (the nearest measured line's factor
    used), 'topo' (the name of the reference topo under most of the line,
    None off every topo), 'ref_f', 'a_f' (always 0.0), 'k_f' (the forward
    stripe's pairs in its link), 'b_r', 'b_r_measured', 'ref_r', 'k_r', 'c'
    (the line's corrected start coordinate, metres along the fast axis from
    the grid's first spectrum), 'E' (its corrected forward end, the same
    units), 'R' (its return point, where the next line's start was measured
    from; None for the last line without a reverse pass), 'row' (the page
    row of the line); 'pix' holds {'forward': (ny, nx, 2), 'reverse':
    (ny, nx, 2) or None}, the corrected absolute position of every pixel
    (NaN where not used).
    """
    report = []
    lines = []
    points = {str(k): float(v) for k, v in (points or {}).items()}
    N = grid.n_spectra
    xy = np.asarray(grid.spec_xy_m, dtype=np.float64)
    acq = np.asarray(grid.acquired, dtype=bool)[:N]
    xy_corr = np.full((N, 2), np.nan)
    pix = {'forward': None, 'reverse': None}
    fast, slow = grid.grid_axes()
    if fast is None or N < 2:
        report.append("no recorded positions: nothing corrected")
        return xy_corr, report, lines, pix
    origin = xy[0].copy()
    step = float(np.linalg.norm(xy[1] - xy[0]))
    n_fast = grid.n_fast
    S_spec = _along(xy, origin, fast)                    # every spectrum's recorded along-line coordinate
    P_spec = _across(xy, origin, fast)                   # and its recorded across-line offset
    t_spec = grid.spectrum_times()
    tm = grid.grid_timing()

    # The reference surface (StripeReference): the earlier topos under the grid placed in the grid's
    # drift setting; a single Sm4File is taken as a reference of one topo at its recorded place.
    ref = prev if isinstance(prev, StripeReference) else StripeReference([prev], [None])
    min_pairs = int(params.get('min_extrema_pairs', 1))
    # The topo pages: pixel locations by the page geometry, the pairing of the spectra with the
    # forward pixels by location, the raw heights.
    have_pages = bool(grid.topo_pages)
    two = have_pages and len(grid.topo_pages) > 1
    pix_f = pix_r = img_f = img_r = None
    idx_f = np.full((N, 2), -1)
    try:
        if have_pages:
            pix_f = grid.pixel_xy_m('forward')
            img_f = grid.topo_raw('forward')
            idx_f, _ = grid.spectrum_pixels('forward')
            if two:
                pix_r = grid.pixel_xy_m('reverse')
                img_r = grid.topo_raw('reverse')
    except Exception as exc:
        pix_f = pix_r = img_f = img_r = None
        idx_f = np.full((N, 2), -1)
        have_pages = two = False
        report.append(f"no stripes in the grid file ({exc}): fast axis uncorrected")
    pix_kept = None
    if have_pages:
        report.append(grid.pairing_text('forward'))
        pix['forward'] = np.full(pix_f.shape, np.nan)
        if two:
            pix['reverse'] = np.full(pix_r.shape, np.nan)
        pix_kept = np.zeros(img_f.shape, dtype=bool)          # the forward pixels paired with a kept spectrum
        for k in np.flatnonzero(acq & (idx_f[:, 0] >= 0)):
            pix_kept[idx_f[k, 0], idx_f[k, 1]] = True
        n_unpaired = int(np.sum(acq & (idx_f[:, 0] < 0)))
        if n_unpaired:
            report.append(f"{n_unpaired} kept spectra have no forward pixel at their location (deviation above "
                          f"{sm4file.PIXEL_PAIR_TOL_PX:g} px): no stripe cell and no factor from them")

    m = int(params['profile_oversample'])
    limits = params['stretch_limits']
    tol = step                                           # a stripe extremum pairs with an image extremum within this
    L_nom = (n_fast - 1) * step

    report.append(f"reference: {len(ref.entries)} topo(s) under the grid, most recent on top: {ref.name}")
    area_cache = {}

    def area_period():
        """The reference topos' extrema spacing along the fast axis (StripeReference.period_along), once."""
        if 'p' not in area_cache:
            area_cache['p'] = ref.period_along(fast, step / m, params) if ref else None
            if area_cache['p'] is not None:
                report.append(f"scan-area extrema spacing along the fast axis {area_cache['p'] * 1e10:.2f} A "
                              f"(the reference topos' profiles)")
            else:
                report.append("no scan-area extrema spacing along the fast axis (no reference profile with a period)")
        return area_cache['p']

    local_refs = []                                      # the lines' own reference periods (their on-topo profiles)
    last_b = {'f': None, 'r': None}                      # the nearest measured factors, for a pass without a period
    geo_measured = []                                    # slow-step shifts measured on the links the image placed

    state = {'model': drift_fn}

    def shift_at(t):
        """The slow-axis shift -(d(t) - d(t_ref)) projected on the slow axis, (..., 2); zero without a model."""
        t = np.asarray(t, dtype=np.float64)
        if state['model'] is None:
            return np.zeros(t.shape + (2,))
        tt = np.where(np.isfinite(t), t, t_ref)
        d = state['model'](tt)
        return (-(d @ slow))[..., None] * slow

    def model_along(t_a, t_b):
        """The model's along-line displacement of the features between two times (0 without a model or times)."""
        if state['model'] is None or not (np.isfinite(t_a) and np.isfinite(t_b)):
            return 0.0
        return float(np.dot(fast, state['model'](float(t_b)) - state['model'](float(t_a))))

    def page_row(cols_kept):
        """The page row the line's kept spectra pair with (the most common one), or None."""
        rows = idx_f[cols_kept, 0]
        rows = rows[rows >= 0]
        if len(rows) == 0:
            return None
        vals, counts = np.unique(rows, return_counts=True)
        return int(vals[np.argmax(counts)])

    def row_stripe(row, pix_xy, heights, use, S0):
        """A page row as a stripe: (s, h) sorted by s, the pixels' recorded along-line coordinates from S0 and heights (NaN where not used)."""
        s = _along(pix_xy[row], origin, fast) - S0
        h = np.array(heights[row], dtype=np.float64)
        h[~use] = np.nan
        order = np.argsort(s)
        return s[order], h[order]

    def best_single(ext, ext_img, b, window):
        """The best alignment of one stripe alone: (offset, pairs) or None."""
        best = None
        for a in alignment_offsets(ext, ext_img, b, 0.0, window):
            pf, rf, _ = alignment_score(ext, ext_img, a, b, tol)
            key = (-pf, round(rf / max(pf, 1) / tol), abs(a))
            if best is None or key < best[0]:
                best = (key, a, pf)
        return None if best is None or best[2] == 0 else (best[1], best[2])

    def place_reverse(row_p, cols_p, E, R, s_min, s_max, i_p):
        """
        The reverse pixels of line i_p at their corrected positions: from the
        line's forward end E (where the pass began) back to the return point
        R, each at E minus b' times its recorded distance from the far end,
        b' = (E - R) / (s_max - s_min).
        """
        if pix['reverse'] is None or row_p is None or s_max <= s_min:
            return
        s_all = _along(pix_r[row_p], origin, fast) - float(S_spec[cols_p[0]])
        use = np.isfinite(img_r[row_p])
        t_rev = float(tm['rev_start'][i_p]) if i_p < len(tm['rev_start']) and np.isfinite(tm['rev_start'][i_p]) else t_ref
        sh = shift_at(np.full(int(np.sum(use)), t_rev))
        b_imp = (E - R) / (s_max - s_min)
        S_corr = E - b_imp * (s_max - s_all[use])
        pix['reverse'][row_p, use] = origin + S_corr[:, None] * fast[None, :] + _across(pix_r[row_p][use], origin, fast) + sh

    # --- the lines, in order ---
    prev_line = None                                     # the previous line: its placement and stripes
    for i in range(grid.n_lines):
        cols = np.arange(i * n_fast, min((i + 1) * n_fast, N))
        a_cols = cols[acq[cols]]
        if len(a_cols) == 0:
            break
        S0 = float(S_spec[cols[0]])                      # the line's first spectrum, recorded, along the line
        s_spec = S_spec[cols] - S0                       # its spectra's recorded distances from it
        shift_cols = shift_at(t_spec[cols])
        line_shift = np.mean(shift_cols[acq[cols]], axis=0)
        L_i = float(s_spec[a_cols[-1] - cols[0]] - s_spec[a_cols[0] - cols[0]])
        partial = len(a_cols) < n_fast
        if partial:
            report.append(f"line {i}: partial line, {len(a_cols)} of {n_fast} spectra kept; the image profile, "
                          f"the factor, and the forward end over its kept range {L_i * 1e10:.1f} A only")
        row = page_row(a_cols) if have_pages else None
        fwd = row_stripe(row, pix_f, img_f, pix_kept[row], S0) if row is not None else None
        if fwd is not None and np.isfinite(fwd[1]).sum() < 3:
            fwd = None
        rev = None
        if row is not None and two and np.isfinite(img_r[row]).sum() >= 3:
            rev = row_stripe(row, pix_r, img_r, np.isfinite(img_r[row]), S0)

        # The prior start of this line: line 0 pinned; a later line from its link's prior.
        sep = 0.0
        R_prior = None
        if i == 0:
            C_prior = 0.0
        elif prev_line['rev'] is not None:
            pr = prev_line
            s_min, s_max = pr['s_min'], pr['s_max']
            # The recorded separation from the previous line's last reverse pixel (the one on the
            # line-start side) to this line's first spectrum, and the model's prior for the return point.
            sep = S0 - (pr['S0'] + s_min)
            L_r = s_max - s_min
            d_rev = model_along(tm['rev_start'][i - 1], tm['rev_end'][i - 1]) if i - 1 < len(tm['rev_start']) else 0.0
            R_prior = pr['E'] - (L_r + d_rev)
            C_prior = R_prior + sep
        else:
            pr = prev_line
            last_pix = None
            if pix_f is not None:
                k_last = pr['a_cols'][-1]
                if idx_f[k_last, 0] >= 0:
                    last_pix = pix_f[idx_f[k_last, 0], idx_f[k_last, 1]]
            sep = S0 - (float(_along(last_pix, origin, fast)) if last_pix is not None else float(S_spec[pr['a_cols'][-1]]))
            C_prior = pr['E'] + sep

        # The image profile along the line's path from its prior start (image coordinates from there).
        span = L_i if partial else L_nom
        s_img = np.arange(-params['profile_reach_back'] * span, params['profile_reach'] * span + step / m, step / m)
        start_actual = origin + C_prior * fast + P_spec[cols[0]] + line_shift
        path_xy = start_actual[None, :] + s_img[:, None] * fast[None, :]
        prof, src = ref.sample(path_xy)
        ext_img = profile_extrema_runs(prof, s_img, src, params)
        inside = (s_img >= 0) & (s_img <= span)
        on_page = bool(np.isfinite(prof[inside]).any())
        counts = np.bincount(src[inside & (src >= 0)], minlength=max(len(ref.entries), 1))
        topo_idx = int(np.argmax(counts)) if np.any(counts > 0) else None      # the topo under most of the line
        cover = (f"; on {', '.join(ref.entries[t]['f'].name for t in np.flatnonzero(counts))}"
                 f" over {100.0 * np.mean(np.isfinite(prof[inside])):.0f}% of its path" if counts.any() else "")
        # The reference period: the extrema spacing of the topo profile along the part of this line's
        # path that lies on a topo; off every topo, the median of the other lines', else the scan area's.
        p_img = extrema_period(ext_img)
        if p_img is not None:
            local_refs.append(float(p_img))
            refs, ref_how = [(float(p_img), 1.0)], "the topo profile along this line"
        elif local_refs:
            refs, ref_how = [(float(np.median(local_refs)), 1.0)], \
                f"the median spacing on the other lines' topo profiles ({len(local_refs)} line(s))"
        else:
            p_area = area_period()
            refs, ref_how = ([(p_area, 1.0)] if p_area else []), "the reference topos' extrema spacing along the fast axis"
        report.append(f"line {i}: image extrema (A from the prior start, + max, - min) {_ext_text(ext_img)}"
                      + (cover if on_page else " (line off every topo)")
                      + (f"; reference period {refs[0][0] * 1e10:.2f} A from {ref_how}" if refs else "; no reference period"))

        # --- the factors from the stripes ---
        b_f, b_r = 1.0, 1.0
        ref_f = ref_r = None
        b_f_measured = b_r_measured = False
        p_f = p_r = None
        ext_f = ext_r = []
        b_f_borrowed = b_r_borrowed = False
        if fwd is not None:
            p_f, txt_f = stripe_period(fwd[1], fwd[0], params)
            b_f, ref_f = period_stretch(p_f, refs, limits)
            b_f_measured = b_f is not None
            if b_f is None and last_b['f'] is not None:
                b_f, b_f_borrowed = last_b['f'][0], True
                how_f = f"no period ratio; the nearest measured line's factor (line {last_b['f'][1]}) {factor_text(b_f)}"
            elif b_f is None:
                b_f, how_f = 1.0, "no factor measured (1)"
            else:
                how_f = f"-> {factor_text(b_f)} (reference {ref_f * 1e10:.2f} A)"
            ext_f = profile_extrema_runs(fwd[1], fwd[0], None, params)
            report.append(f"line {i}: forward stripe period {txt_f}; {how_f}; extrema {_ext_text(ext_f)}")
        if rev is not None:
            p_r, txt_r = stripe_period(rev[1], rev[0], params)
            b_r, ref_r = period_stretch(p_r, refs, limits)
            b_r_measured = b_r is not None
            if b_r is None and last_b['r'] is not None:
                b_r, b_r_borrowed = last_b['r'][0], True
                how_r = f"no period ratio; the nearest measured line's factor (line {last_b['r'][1]}) {factor_text(b_r)}"
            elif b_r is None:
                b_r, how_r = 1.0, "no factor measured (1)"
            else:
                how_r = f"-> {factor_text(b_r)}"
            ext_r = profile_extrema_runs(rev[1], rev[0], None, params)
            report.append(f"line {i}: reverse stripe period {txt_r}; {how_r}; extrema {_ext_text(ext_r)}")
            if b_r_measured:
                last_b['r'] = (float(b_r), i)
        period_ref = ref_f or ref_r or (prev_line['ref_r'] if prev_line else None) or p_f or p_r \
            or (refs[0][0] if refs else None)
        window = float(period_ref) if period_ref else L_nom

        # --- the placement: line 0 pinned, every other line through its link with the previous one ---
        k_f = k_r = 0
        one = None
        if i == 0:
            C_i = 0.0
            one = best_single(ext_f, ext_img, b_f, window) if fwd is not None else None
            report.append("line 0: start pinned at the recorded position"
                          + (f"; the image would put it {one[0] * 1e10:+.2f} A away ({one[1]} pairs)" if one else ""))
        elif prev_line['rev'] is not None:
            pr = prev_line
            s_rev, h_rev = pr['rev']
            ext_rp = profile_extrema_runs(h_rev, s_rev, None, params)
            b_rp, s_min, s_max, E_p = pr['b_r'], pr['s_min'], pr['s_max'], pr['E']
            L_r = s_max - s_min
            start_p = pr['img_start']

            def b_implied(R):
                return (E_p - R) / L_r

            def rev_map(R):
                """(a, b') of the previous reverse stripe in the previous line's image coordinates at the return point R."""
                b_ = b_implied(R)
                return (E_p - start_p) - b_ * s_max, b_          # s_image = (E_p - start_p) - b' (s_max - s_rev)

            # Candidates: a reverse extremum at recorded s_rev landing on an image extremum at X (corrected)
            # needs E_p - b' (s_max - s_rev) = X with b' = (E_p - R) / L_r, so R = E_p - (E_p - X) L_r / (s_max - s_rev)
            # (an extremum at the far end is fixed by E_p and gives no candidate); a forward extremum gives
            # C = C_prior + a, R = C - sep.
            cands = []
            for s_e, kind in ext_rp:
                dist = s_max - s_e
                if dist <= 0.5 * step:
                    continue
                for x_img, k_img in pr['ext_img']:
                    if k_img != kind:
                        continue
                    X = start_p + x_img
                    R = E_p - (E_p - X) * L_r / dist
                    if abs(R - R_prior) <= window and limits[0] <= b_implied(R) <= limits[1]:
                        cands.append(float(R))
            for a in alignment_offsets(ext_f, ext_img, b_f, 0.0, window):
                R = C_prior + a - sep
                if limits[0] <= b_implied(R) <= limits[1]:
                    cands.append(float(R))
            best = None
            for R in cands:
                a_r, b_ = rev_map(R)
                pr_, rr, _ = alignment_score(ext_rp, pr['ext_img'], a_r, b_, tol)
                pf, rf, _ = alignment_score(ext_f, ext_img, R + sep - C_prior, b_f, tol)
                key = (-(pr_ + pf), round((rr + rf) / max(pr_ + pf, 1) / tol), abs(R - R_prior))
                if best is None or key < best[0]:
                    best = (key, R, pr_, pf)
            if best is not None and best[2] + best[3] >= min_pairs:
                R_i, k_r, k_f = best[1], best[2], best[3]
                how = f"image pairs (reverse {k_r}, forward {k_f})"
            else:
                R_i = R_prior
                how = "no image pairs; the model's prior"
                if fwd is not None:
                    # Off the page: the phase lock between the previous reverse stripe (at the prior) and this
                    # forward stripe after the shift one slow step makes along the line.
                    shift_geo = float(np.median(geo_measured)) if geo_measured else 0.0
                    x_r = E_p - b_implied(R_prior) * (s_max - s_rev)
                    x_f = C_prior + b_f * fwd[0] + shift_geo
                    lag, ncc = stripe_lag(x_r, h_rev, x_f, fwd[1], step / m, window, params)
                    if lag is not None:
                        R_i = R_prior - lag
                        how = f"phase lock to the previous reverse stripe (lag {lag * 1e10:+.2f} A, ncc {ncc:.3f})"
            if f'R:{i - 1}' in points:
                R_i, how = points[f'R:{i - 1}'], "the manual point"
            if lines:
                lines[-1]['R'] = float(R_i)
            C_i = R_i + sep
            report.append(f"link {i - 1}->{i}: return point {R_i * 1e10:+.2f} A ({(R_i - R_prior) * 1e10:+.2f} A from the "
                          f"model's prior) by {how}; recorded separation to line {i}'s first spectrum {sep * 1e10:+.2f} A "
                          f"-> start {C_i * 1e10:+.2f} A; the placement implies the reverse {factor_text(b_implied(R_i))} "
                          f"(measured {factor_text(b_rp)})")
            if k_r + k_f > 0 and fwd is not None:
                # The image placed this link: the lag between the two stripes measures the slow-step shift.
                x_r = E_p - b_implied(R_i) * (s_max - s_rev)
                x_f = C_i + b_f * fwd[0]
                lag, _ = stripe_lag(x_r, h_rev, x_f, fwd[1], step / m, 0.5 * L_nom, params)
                if lag is not None:
                    geo_measured.append(-lag)
            place_reverse(pr['row'], pr['cols'], E_p, R_i, s_min, s_max, i - 1)
        else:
            one = best_single(ext_f, ext_img, b_f, window) if fwd is not None else None
            if one is not None and one[1] >= min_pairs:
                C_i, k_f = C_prior + one[0], one[1]
                how = f"forward pairs {k_f}"
            else:
                C_i, how = C_prior, "no image pairs; the recorded separation from the forward end"
            if f'R:{i - 1}' in points:
                C_i, how = points[f'R:{i - 1}'] + sep, "the manual point"
            if lines:
                lines[-1]['R'] = float(C_i - sep)
            report.append(f"link {i - 1}->{i} (no reverse pass): start {C_i * 1e10:+.2f} A by {how}")

        # --- the forward factor from the matched extrema themselves (maxima on maxima, minima on minima) ---
        # With the line's start placed, the slope that lays the stripe's extrema on the reference's is
        # fitted from their pairs (the offset held, so the chain of links stands); with fewer than
        # three pairs the lattice-period ratio stands.  Only the part of the stripe that lies on a
        # topo has pairs, so a stripe partly on a topo is fitted from that part.
        b_f_pairs, n_fit = None, 0
        if fwd is not None and ext_f and ext_img:
            # The pairs are found at the line's placement (line 0, pinned, at the offset the image
            # itself gives), the slope fitted with the offset free (s_image = a + b s_stripe), so a
            # pinned start or a rigid offset never tilts the slope; the placement is not moved.
            a_line = (one[0] if (i == 0 and one is not None) else C_i - C_prior)
            b_try = b_f
            for _ in range(3):
                _, _, pr_list = alignment_score(ext_f, ext_img, a_line, b_try, tol)
                if len(pr_list) < 3:
                    break
                xs_ = np.array([q[0] for q in pr_list]); ys_ = np.array([q[1] for q in pr_list])
                if np.ptp(xs_) <= 0:
                    break
                b_new, a_new = np.polyfit(xs_, ys_, 1)
                if not (np.isfinite(b_new) and limits[0] <= b_new <= limits[1]):
                    break
                b_try, a_line, n_fit = float(b_new), float(a_new), len(pr_list)
            if n_fit >= 3:
                b_f_pairs = b_try
                report.append(f"line {i}: forward factor from {n_fit} matched extrema {factor_text(b_f_pairs)}"
                              + (f" (period ratio {factor_text(b_f)})" if b_f_measured else ""))
                b_f, b_f_measured, b_f_borrowed = b_f_pairs, True, False
        if b_f_measured:
            last_b['f'] = (float(b_f), i)
        if f'E:{i}' in points:
            # The manual forward end: the forward factor that puts the line's last kept spectrum there.
            L_f = float(s_spec[a_cols[-1] - cols[0]])
            if L_f > 0:
                b_f = (points[f'E:{i}'] - C_i) / L_f
                b_f_measured, b_f_borrowed = True, False
                report.append(f"line {i}: forward end at the manual point {points[f'E:{i}'] * 1e10:+.2f} A -> {factor_text(b_f)}")

        # --- this line's forward spectra and pixels at their corrected positions ---
        S_corr_spec = C_i + b_f * s_spec
        for k in a_cols:
            j = k - cols[0]
            xy_corr[k] = origin + S_corr_spec[j] * fast + P_spec[k] + shift_cols[j]
            if pix['forward'] is not None and idx_f[k, 0] >= 0:
                pi, pj = idx_f[k]
                s_pix = float(_along(pix_f[pi, pj], origin, fast)) - S0
                pix['forward'][pi, pj] = origin + (C_i + b_f * s_pix) * fast + _across(pix_f[pi, pj], origin, fast) + shift_cols[j]
        E_i = C_i + b_f * float(s_spec[a_cols[-1] - cols[0]])
        report.append(f"line {i}: {len(a_cols)} spectra; start {C_i * 1e10:+.2f} A; {factor_text(b_f)}; forward end "
                      f"{E_i * 1e10:+.2f} A" + ("" if rev is not None else "; reverse pass missing"))
        lines.append({'i': i, 'n_kept': int(len(a_cols)), 'b_f': float(b_f), 'b_f_measured': bool(b_f_measured),
                      'b_f_pairs': None if b_f_pairs is None else float(b_f_pairs), 'n_fit': int(n_fit),
                      'b_f_borrowed': bool(b_f_borrowed), 'b_r_borrowed': bool(b_r_borrowed),
                      'topo': None if topo_idx is None else ref.entries[topo_idx]['f'].name,
                      'ref_f': ref_f, 'a_f': 0.0, 'k_f': int(k_f), 'b_r': float(b_r),
                      'b_r_measured': bool(b_r_measured and rev is not None), 'ref_r': ref_r, 'k_r': int(k_r),
                      'c': float(C_i), 'E': float(E_i), 'R': None, 'row': row})
        entry = {'i': i, 'C': C_i, 'E': E_i, 'row': row, 'cols': cols, 'a_cols': a_cols, 'S0': S0,
                 'img_start': C_prior, 'ext_img': ext_img, 'rev': rev, 'b_r': b_r, 'ref_r': ref_r}
        if rev is not None:
            fin = np.isfinite(rev[1])
            entry['s_min'], entry['s_max'] = float(np.min(rev[0][fin])), float(np.max(rev[0][fin]))
        prev_line = entry

    # The last line's reverse pass, with no link after it: at the model's prior.
    if prev_line is not None and prev_line['rev'] is not None:
        pr = prev_line
        i_p = pr['i']
        L_r = pr['s_max'] - pr['s_min']
        d_rev = model_along(tm['rev_start'][i_p], tm['rev_end'][i_p]) if i_p < len(tm['rev_start']) else 0.0
        R_last = pr['E'] - (L_r + d_rev)
        how = "the model's prior"
        if f'R:{i_p}' in points:
            R_last, how = points[f'R:{i_p}'], "the manual point"
        if lines:
            lines[-1]['R'] = float(R_last)
        place_reverse(pr['row'], pr['cols'], pr['E'], R_last, pr['s_min'], pr['s_max'], i_p)
        report.append(f"line {i_p}: last line; its reverse pass placed at {how} (return point {R_last * 1e10:+.2f} A)")
    if state['model'] is not None:
        dt = t_spec - t_ref
        ok = acq & np.isfinite(dt)
        if ok.any():
            d = state['model'](np.where(np.isfinite(t_spec), t_spec, t_ref))
            d_slow = d @ slow
            d_fast = d @ fast
            report.append(f"slow-axis drift applied over {np.nanmin(dt[ok]) / 60:.1f} to {np.nanmax(dt[ok]) / 60:.1f} min after "
                          f"the first spectrum: {(d_slow[ok].max() - d_slow[ok].min()) * 1e10:.2f} A across the grid "
                          f"(fast-axis component {(d_fast[ok].max() - d_fast[ok].min()) * 1e10:.2f} A not applied: the stripes "
                          f"own that axis)")
    else:
        report.append("no slow-axis drift: slow axis uncorrected")
    return xy_corr, report, lines, pix


class TopoRef:
    """
    A topo of the directory without its data: what the baseline model reads
    of a topo (its name, path, times, pixel scales, and footprint), built from
    the header scan (sm4_header_info's geometry, `geom`) and the topo times
    of the computed store ('pairs' section, TrackedPairs.topos, `times`), so
    a file's pair is looked up without loading its topos; the pair tracking
    loads the files (Drift.pair) only for pairs not in the store.  The
    times: the end is the file-name stamp; the start is the stored start
    (the file's start_time, recorded with its end when the topo was loaded
    in a build), else the header scan's RHK_DateTime (geom 'start_time'),
    else the end minus the scan's duration (rows x pixels x passes x
    RHK_Period) when the page has a period, else the end; mid_time is the
    middle of the two.  The geometry methods (rotation, frame_extent_m,
    geometry, contains_m, to_frame_m) are Sm4File's, from the header values.
    """

    kind = 'topo'

    def __init__(self, path, geom, times=None):
        self.path = path
        self.name = os.path.basename(path)
        self.stamp = sm4file.filename_stamp(path)
        self.center_m = tuple(float(x) for x in geom['center_m'])
        self.angle_deg = float(geom.get('angle_deg', 0.0))
        self.xscale_m = float(geom['xscale_m'])
        self.yscale_m = float(geom['yscale_m'])
        self.nx = int(geom['nx'])
        self.ny = int(geom['ny'])
        self.n_topo_pages = int(geom.get('n_topo_pages', 1) or 1)
        self.scan_period = geom.get('period_s')
        end = sm4file.stamp_datetime(self.stamp).timestamp()
        if times is not None and times.get('start') is not None:
            start = float(times['start'])
            end = float(times.get('end', end))
        elif geom.get('start_time') is not None:
            start = float(geom['start_time'])
        elif self.scan_period:
            start = end - self.ny * self.nx * (2 if self.n_topo_pages > 1 else 1) * float(self.scan_period)
        else:
            start = end
        self.start_time = min(start, end)
        self.end_time = end
        self.mid_time = 0.5 * (self.start_time + self.end_time)

    def set_times(self, start, end):
        """The times from the loaded file (a build): start, end and mid replaced."""
        self.start_time = float(min(start, end))
        self.end_time = float(end)
        self.mid_time = 0.5 * (self.start_time + self.end_time)

    def times_record(self):
        """{'start', 'end', 'mid'} (seconds since the epoch): the store's entry for this topo."""
        return {'start': float(self.start_time), 'end': float(self.end_time), 'mid': float(self.mid_time)}

    def rotation(self):
        th = np.deg2rad(self.angle_deg)
        return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    def frame_extent_m(self):
        u0 = -(self.nx - 1) / 2.0 * self.xscale_m
        u1 = (self.nx - 1) / 2.0 * self.xscale_m
        v0 = -(self.ny - 1) / 2.0 * self.yscale_m
        v1 = (self.ny - 1) / 2.0 * self.yscale_m
        return [u0 - self.xscale_m / 2.0, u1 + self.xscale_m / 2.0, v0 - self.yscale_m / 2.0, v1 + self.yscale_m / 2.0]

    def geometry(self):
        return {'center_m': tuple(self.center_m), 'angle_deg': self.angle_deg, 'extent_m': list(self.frame_extent_m())}

    def contains_m(self, xy_m):
        return sm4file.geom_contains(self.geometry(), xy_m)

    def to_frame_m(self, xy_m):
        xy = np.atleast_2d(np.asarray(xy_m, dtype=np.float64)) - np.asarray(self.center_m)
        return xy @ self.rotation()


class Drift:
    """
    The drift correction of an experiment directory, driven by the Browser:
    the directory's topos cut into segments at the long-range moves
    (index_segments, from the header scan the Browser made), every
    consecutive topo pair of every segment tracked at startup (build;
    track_pair, the results kept in the 'pairs' section of the computed
    store, TrackedPairs, so a later startup tracks only new pairs), one
    result per spectra file (result:
    the baseline velocity of the file's before/after pair, the stripe
    matching on top of it when asked), the corrected topographies
    (corrected_topo), the drift within every sweep (sweep_shift), the
    heatmap row shears (row_shear), the map pixel shifts (map_shift), and
    the sweep lines (sweep_lines).
      directory  : the experiment directory (the two stores live there)
      files      : the directory's .sm4 files in stamp order (the Browser's list)
      kinds      : {path: kind} of every file from the header scan (the Browser's dict,
                   shared: a file loaded later is added to it)
      head_geoms : {topo path: header geometry} (the Browser's dict)
      params     : the drift parameters (DRIFT_PARAMS with the Browser's overrides)
      loader     : loader(path) -> Sm4File, the Browser's _load
      settings, computed : the experiment's stores (caches.Settings,
                   caches.Computed), the Browser's; opened here
                   (caches.open_stores) when either is None
    result(f, loaded, stripe) keeps the Browser's loaded set (the dict
    itself, from that call on), so files the Browser holds are not loaded
    again;
    `stripe` is the fast-axis placement asked for a grid (None: the grid's
    stored state for the setting in force; True / 'auto', 'manual', False /
    'off'; ignored for a point).
    """

    LRU = 3               # topos kept loaded during a build

    def __init__(self, directory, files, kinds, head_geoms, params, loader, settings=None, computed=None):
        self.directory = directory
        self.files = list(files)
        self.kinds = kinds
        self.head_geoms = head_geoms
        self.drift_params = dict(params)
        self._load = loader
        self.loaded = {}
        self.drift_index = {}      # (path, fast-axis state or False, points, auto, manual, anchor, sweep mode) -> result entry (see result)
        self.pair_cache = {}       # (earlier path, later path) -> track_pair result (this session)
        self._models = {}          # (path, 'spec' | 'topo') -> (PairDrift or None, [notes])
        self._lru = {}             # path -> Sm4File, the topos kept loaded during a build (LRU)
        self._img_cache = {}       # (path, direction, active flags, id(model)) -> corrected image
        self._listeners = []       # (fn, alive) pairs told of every manual change (subscribe / notify)
        self.scale_of = None       # path -> scale factor (the browser's scale timeline), for the readouts only
        self.built = False
        self._building = False     # True inside build(): pairs tracked there are written once at its end
        self.index_segments()
        # The two stores (caches.py): the tracked pairs and topo times in the computed one, the
        # manual spans and stripe states in the settings; opened here when the caller has none.
        if settings is None or computed is None:
            from sm4_Browser.caches import open_stores
            s, c = open_stores(directory, self.files)
            settings, computed = settings or s, computed or c
        self.settings, self.computed = settings, computed
        self.tracked = TrackedPairs(self.computed.section('pairs', PAIRS_FORMAT))
        self.refs = {}
        for p in self.topo_seq:
            g = self.head_geoms.get(p)
            if g is not None:
                self.refs[p] = TopoRef(p, g, self.tracked.topos.get(os.path.basename(p)))
        self.manual_spans = []     # [{'a', 'b': paths, 'markers': (n, 6), 'v': (2,) or None, 'pairs': [...], 'kind'}]
        # The stripe matching's state per grid and per slow-axis setting ('off', 'auto', 'manual'): the
        # fast-axis state ('off': the velocity alone; 'auto': the extrema matching; 'manual': the matching
        # with the dragged points) and the points, {'E:i' | 'R:i': metres along the fast axis}.
        self.stripe_sets = {}      # grid path -> {setting -> {'fast': str, 'points': {key: s}}}
        self._load_manual()
        if self.tracked.pairs or self.manual_spans:
            print(f"drift: {len(self.tracked.pairs)} tracked pair(s) from {self.computed.name}, "
                  f"{len(self.manual_spans)} manual span(s) from {self.settings.name}")

    # --- the manual drift (the spans applied in the drift window) ---

    def _topo_by_name(self, name):
        """The path of the topo of the directory named `name`, or None."""
        return next((p for p in self.topo_seq if os.path.basename(p) == name), None)

    def _load_manual(self):
        """
        The settings' 'drift' section into manual_spans: a confirmed or
        adjusted span's velocity refitted from its markers, a typed span's
        taken from its stored 'v'.  An older drift cache's global 'typed'
        entry, carried over with the section, is no longer a source and is
        not read.  A span or a stripe set naming a
        file that is not in the directory is reported and left out of the
        session, but stays in the file (a setting is never dropped).
        """
        manual = self.settings.get('drift') or {}
        self.manual_spans = []
        self._orphan_spans = []        # spans of files not in the directory: kept in the file, unused
        self._orphan_stripe = {}       # stripe sets of such files
        for rec in manual.get('spans', []):
            a, b = self._topo_by_name(rec.get('a', '')), self._topo_by_name(rec.get('b', ''))
            if a is None or b is None:
                print(f"{self.settings.name}: a manual span's topos ({rec.get('a')}, {rec.get('b')}) are not in the "
                      "directory; left out of this session (kept in the file).")
                self._orphan_spans.append(rec)
                continue
            v = rec.get('v')
            self._put_span(a, b, np.asarray(rec.get('markers', []), dtype=np.float64).reshape(-1, 6),
                           kind=rec.get('kind', 'adjusted'), quiet=True,
                           v=None if v is None else np.asarray(v, dtype=np.float64).reshape(2))
        # The stripe matching's sets (format: {grid name: {setting: {'fast', 'points'}}}).
        self.stripe_sets = {}
        by_name = {os.path.basename(p): p for p in self.files}
        for name, sets in (manual.get('stripe') or {}).items():
            path = by_name.get(name)
            if path is None:
                print(f"{self.settings.name}: the stripe sets of {name} belong to no file in the directory; "
                      "left out of this session (kept in the file).")
                self._orphan_stripe[name] = sets
                continue
            self.stripe_sets[path] = {}
            for setting, st in (sets or {}).items():
                if setting not in self.SLOW_SETTINGS:
                    continue
                fast = st.get('fast', 'off')
                self.stripe_sets[path][setting] = {
                    'fast': fast if fast in self.FAST_STATES else 'off',
                    'points': {str(k): float(v) for k, v in (st.get('points') or {}).items()}}

    def manual_record(self):
        """The settings' 'drift' section from manual_spans (each span's topo names, kind, markers, and velocity) and stripe_sets, the entries of files not in the directory kept as they were."""
        stripe = {os.path.basename(path): {setting: {'fast': st['fast'],
                                                      'points': {k: float(v) for k, v in st['points'].items()}}
                                            for setting, st in sets.items()}
                  for path, sets in self.stripe_sets.items() if sets}
        stripe.update(getattr(self, '_orphan_stripe', {}))
        return {'spans': [{'a': os.path.basename(sp['a']), 'b': os.path.basename(sp['b']), 'kind': sp['kind'],
                           'markers': [[float(x) for x in row] for row in np.asarray(sp['markers']).reshape(-1, 6)],
                           'v': None if sp.get('v') is None else [float(x) for x in np.asarray(sp['v']).reshape(2)]}
                          for sp in self.manual_spans] + list(getattr(self, '_orphan_spans', [])),
                'stripe': stripe}

    # --- the stripe matching's state (the stripe window) ---

    SLOW_SETTINGS = ('off', 'auto', 'manual')
    FAST_STATES = ('off', 'auto', 'manual')

    def slow_setting(self, f, auto=True, manual=True):
        """
        The slow-axis setting in force for a file under the toggles: 'manual'
        when a manual span's velocity governs it (manual drift on and a span
        covering it), 'auto' when the tracker's velocity does (the pair's or
        the extrapolation, a confirmed span included), 'off' without a
        velocity.  The stripe sets are kept per setting.
        """
        if f.kind == 'topo':
            model, _ = self.model_for_topo(f.path, auto, manual)
        else:
            model, _ = self.model_for_spectra(f, auto, manual)
        if model is None:
            return 'off'
        return 'manual' if str(model.rule).startswith('manual') else 'auto'

    def _stripe_set(self, path, setting, create=False):
        """The set of a grid for a slow-axis setting ({'fast', 'points'}), None when absent; with `create` made when absent (fast 'off', no points)."""
        if create:
            return self.stripe_sets.setdefault(path, {}).setdefault(setting, {'fast': 'off', 'points': {}})
        return self.stripe_sets.get(path, {}).get(setting)

    def fast_state(self, f, setting=None, auto=True, manual=True):
        """The fast-axis state of a grid for a slow-axis setting (the one in force under the toggles when None): 'off', 'auto', or 'manual'; 'off' for any other file."""
        if f.kind != 'grid':
            return 'off'
        setting = self.slow_setting(f, auto, manual) if setting is None else setting
        st = self._stripe_set(f.path, setting)
        return st['fast'] if st else 'off'

    def stripe_points(self, f, setting):
        """The dragged points of a grid for a slow-axis setting: {'E:i' | 'R:i': metres along the fast axis} (a copy)."""
        st = self._stripe_set(f.path, setting)
        return dict(st['points']) if st else {}

    def set_fast_state(self, f, setting, state):
        """The fast-axis state of a grid for a slow-axis setting set; every result dropped, the settings store written, the subscribers told."""
        if state not in self.FAST_STATES or setting not in self.SLOW_SETTINGS:
            raise ValueError(f"fast-axis state {state!r} / setting {setting!r} unknown.")
        st = self._stripe_set(f.path, setting, create=True)
        if st['fast'] == state:
            return
        st['fast'] = state
        self._invalidate()
        self._save_manual()
        print(f"stripe matching: {f.name}, drift {setting}: stripe {state}.")
        self.notify()

    def set_stripe_point(self, f, setting, key, s, commit=True):
        """
        A point of a grid's set moved to s (metres along the fast axis from
        the grid's first spectrum): key 'E:i' (line i's forward end) or
        'R:i' (its return point, one point with line i + 1's start).  Every
        result is dropped; with `commit` the settings store is written and
        the subscribers told (a drag commits on release only).
        """
        st = self._stripe_set(f.path, setting, create=True)
        st['points'][str(key)] = float(s)
        self._invalidate()
        if commit:
            self._save_manual()
            self.notify()

    def pin_stripe_points(self, f, setting, lines, commit=True):
        """
        Every line's forward end and return point of a placement (the
        'lines' of a result: their 'E' and 'R') stored as the points of a
        grid's set, so that the placement reproduces itself exactly and a
        drag of one point then moves that point alone (the stripe window
        pins the placement when a drag begins).  Every result is dropped;
        with `commit` the settings store is written and the subscribers
        told (the stripe window pins without and commits on release).
        Returns the points.
        """
        st = self._stripe_set(f.path, setting, create=True)
        for ln in lines or []:
            if ln.get('E') is not None:
                st['points'][f"E:{int(ln['i'])}"] = float(ln['E'])
            if ln.get('R') is not None:
                st['points'][f"R:{int(ln['i'])}"] = float(ln['R'])
        self._invalidate()
        if commit:
            self._save_manual()
            self.notify()
        return dict(st['points'])

    def clear_stripe_points(self, f, setting):
        """Every dragged point of a grid's set dropped (the fast-axis state kept): back to the matching's placement."""
        st = self._stripe_set(f.path, setting)
        if not st or not st['points']:
            return
        st['points'] = {}
        self._invalidate()
        self._save_manual()
        print(f"stripe matching: {f.name}, drift {setting}: the points cleared.")
        self.notify()

    def _save_manual(self):
        """Every manual change written to the settings store at once (the 'drift' section)."""
        self.settings.put('drift', self.manual_record())

    def _invalidate(self):
        """Every per-file result, model, and corrected image dropped (a manual change or a cleared span)."""
        self.drift_index = {}
        self._models = {}
        self._img_cache = {}

    # --- the windows showing this Drift ---

    def subscribe(self, fn, alive=None):
        """
        fn() is run after every manual change (a span set or cleared, a
        stripe set's state or points changed): the browser, every open Line
        and Map window and the stripe window subscribe their own redraw, so
        a change made in any drift or stripe window reaches all of them.
        `alive`, when given, is a callable that
        returns False once the subscriber's window is gone; such a
        subscriber is dropped at the next notify without being run.  A
        subscriber that raises is reported and dropped.  Subscribing the
        same fn twice keeps one entry.
        """
        self.unsubscribe(fn)
        self._listeners.append((fn, alive))

    def unsubscribe(self, fn):
        """fn removed from the subscribers (nothing happens when it is not one)."""
        self._listeners = [(f, a) for f, a in self._listeners if f != fn]

    def notify(self):
        """Every live subscriber run (the manual state changed); dead or raising ones dropped."""
        dead = []
        for fn, alive in list(self._listeners):
            try:
                if alive is not None and not alive():
                    dead.append(fn)
                    continue
                fn()
            except Exception as exc:
                print(f"drift: a window could not be refreshed and is no longer notified ({exc}).")
                dead.append(fn)
        for fn in dead:
            self.unsubscribe(fn)

    def auto_velocity_of(self, a_path, b_path):
        """The tracked auto velocity (2,) of the pair of two topos (either order; the earlier first), or None."""
        if a_path not in self.refs or b_path not in self.refs:
            return None
        if self.files.index(a_path) > self.files.index(b_path):
            a_path, b_path = b_path, a_path
        est = self.pair(self.refs[a_path], self.refs[b_path])
        return None if est.get('v') is None else np.asarray(est['v'], dtype=np.float64).reshape(2)

    def _put_span(self, a_path, b_path, markers, kind='adjusted', quiet=False, v=None):
        """
        A span (a before b in the directory) stored or replaced with its
        markers, its kind, and its velocity: the markers' least-squares fit
        ('adjusted'), the pair's auto velocity itself ('confirmed'; back to
        'adjusted' with a note when the pair is not tracked), or the given
        `v` ('typed': the velocity applied from the drift window's box as it
        stood, (2,) m/s; the markers, if any, are kept for the readout only).
        """
        if self.files.index(a_path) > self.files.index(b_path):
            a_path, b_path = b_path, a_path
            markers = np.asarray(markers, dtype=np.float64).reshape(-1, 6)[:, [3, 4, 5, 0, 1, 2]]
        markers = np.asarray(markers, dtype=np.float64).reshape(-1, 6)
        v_fit, pairs, note = fit_marker_velocity(markers)
        kind = kind if kind in ('confirmed', 'adjusted', 'typed') else 'adjusted'
        if kind == 'typed':
            if v is None:
                print(f"manual drift: the span {os.path.basename(a_path)} -> {os.path.basename(b_path)} has no typed velocity; "
                      "stored as adjusted (the markers' fit).")
                kind, v = 'adjusted', v_fit
            else:
                v = np.asarray(v, dtype=np.float64).reshape(2)
        elif kind == 'confirmed':
            v_auto = self.auto_velocity_of(a_path, b_path)
            if v_auto is None:
                print(f"manual drift: the pair {os.path.basename(a_path)} -> {os.path.basename(b_path)} is not tracked; "
                      "the span is stored as adjusted (the markers' fit).")
                kind, v = 'adjusted', v_fit
            else:
                v = v_auto
        else:
            v = v_fit
        span = {'a': a_path, 'b': b_path, 'markers': markers, 'v': v, 'pairs': pairs, 'note': note, 'kind': kind}
        self.manual_spans = [sp for sp in self.manual_spans if not (sp['a'] == a_path and sp['b'] == b_path)]
        self.manual_spans.append(span)
        if not quiet:
            print(f"manual drift: span {os.path.basename(a_path)} -> {os.path.basename(b_path)}, {len(markers)} marker pair(s), "
                  f"{kind}:")
            sc = self.scale_of(a_path) if self.scale_of is not None else None
            for k, rec in enumerate(pairs):
                print("  " + marker_pair_text(k, rec, sc))
            if v is not None:
                what = {'confirmed': "auto v confirmed", 'adjusted': "fitted v", 'typed': "typed v"}[kind]
                print(f"  {what} = ({v[0] * 3.6e12:+.3f}, {v[1] * 3.6e12:+.3f}) nm/h nominal = ({v[0] * 1e12:+.2f}, {v[1] * 1e12:+.2f}) pm/s"
                      + (f"; calibrated (/ {sc:g}) ({v[0] * 3.6e12 / sc:+.3f}, {v[1] * 3.6e12 / sc:+.3f}) nm/h" if sc else ""))
            else:
                print(f"  no velocity ({note})")
        return span

    def set_manual_span(self, a_path, b_path, markers, kind='adjusted', v=None):
        """
        The manual velocity of the span between two topos, stored, reported,
        written to the settings store; every per-file result is recomputed from here
        on and the subscribers are told.  markers (n, 6): x_A, y_A, t_A,
        x_B, y_B, t_B, absolute metres and seconds, one row per marker pair
        (may be empty for a typed velocity).  kind 'adjusted': the
        least-squares velocity of the markers (fit_marker_velocity), source
        'manual (markers)'; 'confirmed' (every pair seeded by the auto
        velocity and left unmoved): the pair's auto velocity itself, source
        'auto (confirmed)'; 'typed': the given `v` ((2,) m/s, the drift
        window's box as applied), source 'manual (typed)'.  Returns the span.
        """
        span = self._put_span(a_path, b_path, markers, kind=kind, v=v)
        self._invalidate()
        self._save_manual()
        self.notify()
        return span

    def clear_manual_span(self, a_path, b_path):
        """The span between two topos removed (either order); nothing happens without one."""
        n = len(self.manual_spans)
        self.manual_spans = [sp for sp in self.manual_spans if {sp['a'], sp['b']} != {a_path, b_path}]
        if len(self.manual_spans) != n:
            self._invalidate()
            self._save_manual()
            print(f"manual drift: span {os.path.basename(a_path)} -> {os.path.basename(b_path)} cleared.")
            self.notify()

    def clear_manual_spans(self):
        """Every marker span removed."""
        if self.manual_spans:
            self.manual_spans = []
            self._invalidate()
            self._save_manual()
            print("manual drift: every marker span cleared.")
            self.notify()

    def _manual_model(self, path, t_ref):
        """
        The manual PairDrift covering a file: the narrowest applied span
        whose two topos enclose the file in the directory's order (both
        topos included; ties to the latest set); None when no span covers
        it (the file then takes the auto velocity, when auto drift is on).
        Source by the span's kind: 'auto (confirmed)', 'manual (markers)'
        (adjusted), 'manual (typed)'.
        """
        i = self.files.index(path) if path in self.files else None
        best = None
        if i is not None:
            for sp in self.manual_spans:
                if sp['v'] is None:
                    continue
                ia, ib = self.files.index(sp['a']), self.files.index(sp['b'])
                if ia <= i <= ib and (best is None or ib - ia <= best[0]):
                    best = (ib - ia, sp)
        if best is not None:
            sp = best[1]
            source = {'confirmed': 'auto (confirmed)', 'typed': 'manual (typed)'}.get(sp.get('kind'), 'manual (markers)')
            what = (f"{os.path.basename(sp['a'])} -> {os.path.basename(sp['b'])}, "
                    + ("the box as applied" if sp.get('kind') == 'typed' else f"{len(sp['markers'])} pair(s)"))
            return PairDrift(sp['v'], t_ref, source, what)
        return None

    # --- the directory's segments ---

    def index_segments(self):
        """
        The directory's topos in time order (topo_seq) cut into segments by
        the long-range moves: a jump of the recorded scan centre (the header
        geometry, absolute metres) between consecutive topos above move_min_m.
        seg_first maps every topo to the first topo of its segment; moves maps
        the first topo after a move to {'vector': the jump, 'from': the last
        topo before it}; segment_keys lists the segments' first topos in
        time order.  Locations only; no file is loaded.
        """
        self.topo_seq = [p for p in self.files if self.kinds[p] == 'topo']
        self.seg_first = {}
        self.moves = {}
        self.segment_keys = []
        first = None
        prev = None
        for p in self.topo_seq:
            g = self.head_geoms.get(p)
            if first is None:
                first = p
                self.segment_keys.append(p)
            elif prev is not None and g is not None and self.head_geoms.get(prev) is not None:
                jump = np.asarray(g['center_m'], dtype=np.float64) - np.asarray(self.head_geoms[prev]['center_m'], dtype=np.float64)
                if float(np.linalg.norm(jump)) > self.drift_params['move_min_m']:
                    first = p
                    self.segment_keys.append(p)
                    self.moves[p] = {'vector': jump, 'from': prev}
            self.seg_first[p] = first
            prev = p
        if self.moves:
            print(f"long-range moves (scan-centre jumps above {self.drift_params['move_min_m'] * 1e6:g} um): "
                  f"{len(self.moves)} in {len(self.topo_seq)} topos, {len(self.moves) + 1} segments")

    def segment_key_of(self, path):
        """
        The segment a file belongs to (its first topo's path): a topo's own
        segment; for any other file the segment of the latest topo at or
        before it in the directory, the first segment for a file before
        every topo; None when the directory has no topo.
        """
        if not self.topo_seq:
            return None
        if self.kinds.get(path) == 'topo' and path in self.seg_first:
            return self.seg_first[path]
        i = self.files.index(path) if path in self.files else len(self.files)
        for j in range(i, -1, -1):
            q = self.files[j]
            if self.kinds.get(q) == 'topo' and q in self.seg_first:
                return self.seg_first[q]
        return self.segment_keys[0]

    def segment_members(self, key):
        """The topo paths of a segment, in time order."""
        return [p for p in self.topo_seq if self.seg_first.get(p) == key and p in self.refs]

    # --- files during a build ---

    def _topo_file(self, path):
        """A topo as a loaded Sm4File: the Browser's loaded set, the build's LRU, else loaded now (times recorded)."""
        f = self.loaded.get(path) or self._lru.get(path)
        if f is None:
            f = sm4file.Sm4File(path)
            self._lru[path] = f
            while len(self._lru) > self.LRU:
                self._lru.pop(next(iter(self._lru)))
        ref = self.refs.get(path)
        if ref is not None:
            ref.set_times(f.start_time, f.end_time)
            self.tracked.topos[f.name] = ref.times_record()
        return f

    def pair(self, A, B):
        """
        track_pair of a topo pair (A, B TopoRef or Sm4File, A the earlier):
        the session's result (pair_cache), else the computed store's
        (TrackedPairs.get_pair, for the files as they are now: a stored pair
        whose file changed is tracked again), else the two files loaded
        (_topo_file) and tracked now, the result put into TrackedPairs with
        the files' signatures and reported.  Inside build the store is
        written once at its end; a pair first tracked later in the session
        (the drift window stepped to a non-consecutive pair) is written at
        once, so it is not tracked again at the next startup.
        """
        key = (A.path, B.path)
        if key in self.pair_cache:
            return self.pair_cache[key]
        sa, sb = sm4file._file_sig(A.path), sm4file._file_sig(B.path)
        est = self.tracked.get_pair(A.name, B.name, sa, sb)
        if est is None:
            fa, fb = self._topo_file(A.path), self._topo_file(B.path)
            est = track_pair(fa, fb, self.drift_params)
            self.tracked.put_pair(A.name, B.name, est, sa, sb)
            print("drift: " + pair_text(est))
            if not self._building:
                self.computed.put('pairs', PAIRS_FORMAT, self.tracked.record())
                self.computed.write()
        self.pair_cache[key] = est
        return est

    def build(self):
        """
        Every consecutive topo pair of every segment tracked (pair: from the
        store or now), in time order, then the tracked pairs and the topo
        times put into the computed store's 'pairs' section and the store
        written, once; the build's LRU of loaded topos emptied.  The console
        reports every pair tracked now and the totals.
        """
        if not self.segment_keys:
            print("drift: the directory has no topography; nothing to build.")
            self.built = True
            return
        n_pairs0 = len(self.tracked.pairs)
        n_ok = 0
        n_all = 0
        self._building = True
        try:
            for n, key in enumerate(self.segment_keys, 1):
                members = self.segment_members(key)
                print(f"\ndrift build: segment {n} of {len(self.segment_keys)} ({len(members)} topos, "
                      f"{max(len(members) - 1, 0)} pairs)")
                for a, b in zip(members, members[1:]):
                    est = self.pair(self.refs[a], self.refs[b])
                    n_all += 1
                    if est.get('v') is not None:
                        n_ok += 1
        finally:
            self._building = False
        # The tracked pairs and the topo times into the computed store, written once here.
        self.computed.put('pairs', PAIRS_FORMAT, self.tracked.record())
        self.computed.write()
        self._lru.clear()
        self.built = True
        print(f"\ndrift build done: {len(self.segment_keys)} segment(s), {n_all} consecutive pair(s), {n_ok} tracked, "
              f"{len(self.tracked.pairs) - n_pairs0} new; written: {self.computed.name}")

    # --- the velocity of a file ---

    @staticmethod
    def _close(v1, v2, tol):
        """True when two velocities agree: |v1 - v2| <= tol x max(|v1|, |v2|) (two zero velocities agree)."""
        v1 = np.asarray(v1, dtype=np.float64)
        v2 = np.asarray(v2, dtype=np.float64)
        return float(np.linalg.norm(v1 - v2)) <= tol * max(float(np.linalg.norm(v1)), float(np.linalg.norm(v2)))

    def _tracked_before(self, members, last):
        """The tracked consecutive pairs (a, b, est) of a segment whose later topo is at or before `last` (a member), in time order."""
        out = []
        stop = members.index(last) if last in members else len(members) - 1
        for j in range(stop):
            est = self.pair(self.refs[members[j]], self.refs[members[j + 1]])
            if est.get('v') is not None:
                out.append((members[j], members[j + 1], est))
        return out

    def _extrapolated(self, members, last, t_ref, notes):
        """
        The fallback velocity of a file without a usable after pair: the last
        tracked pair ending at or before `last`, when it is close to the pair
        before that (extrapolate_tol); None otherwise, the reason in notes.
        """
        tracked = self._tracked_before(members, last)
        if len(tracked) < 2:
            notes.append(f"fewer than two tracked pairs before the file in its segment ({len(tracked)}): no correction")
            return None
        a1, b1, e1 = tracked[-1]
        a2, b2, e2 = tracked[-2]
        n1 = f"{os.path.basename(a1)} -> {os.path.basename(b1)}"
        n2 = f"{os.path.basename(a2)} -> {os.path.basename(b2)}"
        if self._close(e1['v'], e2['v'], float(self.drift_params['extrapolate_tol'])):
            return PairDrift(e1['v'], t_ref, 'auto (extrapolated)', f"{n1}, close to {n2}")
        v1, v2 = np.asarray(e1['v']) * 1e12, np.asarray(e2['v']) * 1e12
        notes.append(f"the last two tracked pairs before the file disagree ({n1}: ({v1[0]:+.2f}, {v1[1]:+.2f}) pm/s; "
                     f"{n2}: ({v2[0]:+.2f}, {v2[1]:+.2f}) pm/s; tolerance {self.drift_params['extrapolate_tol']:g}): "
                     "no correction")
        return None

    def _before_topo(self, f):
        """The path of the last topo of the file's segment before the file (None without one)."""
        seg = self.segment_key_of(f.path)
        members = self.segment_members(seg) if seg is not None else []
        i = self.files.index(f.path) if f.path in self.files else len(self.files)
        before = [p for p in members if self.files.index(p) < i]
        return before[-1] if before else None

    def _after_topo(self, f):
        """The path of the first topo of the file's segment after the file (None without one)."""
        seg = self.segment_key_of(f.path)
        members = self.segment_members(seg) if seg is not None else []
        i = self.files.index(f.path) if f.path in self.files else len(self.files)
        after = [p for p in members if self.files.index(p) > i]
        return after[0] if after else None

    def pair_topos(self, f):
        """The before and after topos of a spectra file as loaded Sm4Files (None where the segment has none)."""
        a, b = self._before_topo(f), self._after_topo(f)
        return (self._topo_file(a) if a is not None else None), (self._topo_file(b) if b is not None else None)

    def span_between(self, a_path, b_path):
        """The stored manual span between two topos (either order), or None."""
        return next((sp for sp in self.manual_spans if {sp['a'], sp['b']} == {a_path, b_path}), None)

    def window_pair(self, f):
        """
        The two topos the drift window shows for a file, as loaded Sm4Files
        (earlier, later): a spectra file's before and after topos
        (pair_topos); a topo file and the next topo of its segment, else the
        previous topo and it.  Raises ValueError with the reason when the
        file has no such pair.
        """
        if f.kind == 'topo':
            seg = self.segment_key_of(f.path)
            members = self.segment_members(seg) if seg is not None else []
            if f.path not in members:
                raise ValueError(f"{f.name} has no header geometry; no pair to show")
            j = members.index(f.path)
            if j + 1 < len(members):
                other, order = members[j + 1], (f.path, members[j + 1])
            elif j > 0:
                other, order = members[j - 1], (members[j - 1], f.path)
            else:
                raise ValueError(f"{f.name} is the only topo of its segment; no pair to show")
            return self._topo_file(order[0]), self._topo_file(order[1])
        a, b = self.pair_topos(f)
        if a is None or b is None:
            raise ValueError(f"{f.name} has no {'before' if a is None else 'after'} topo in its segment; no span to set")
        return a, b

    def model_for_spectra(self, f, auto=True, manual=True):
        """
        The PairDrift of a spectra file (None when the file gets no
        correction) and the console notes.  With `manual`, a manual velocity
        covering the file (_manual_model) comes first; with `auto`, the
        tracker's: the pair (last topo of its segment before it, first topo
        of its segment after it), and without an after topo or with that
        pair untracked, the extrapolation rule (_extrapolated).  Computed
        once per file and flag pair.
        """
        auto, manual = bool(auto), bool(manual)
        key = (f.path, 'spec', auto, manual)
        if key in self._models:
            return self._models[key]
        notes = []
        model = None
        seg = self.segment_key_of(f.path)
        members = self.segment_members(seg) if seg is not None else []
        i = self.files.index(f.path) if f.path in self.files else len(self.files)
        before = [p for p in members if self.files.index(p) < i]
        after = [p for p in members if self.files.index(p) > i]
        t_ref = float(self.refs[before[-1]].mid_time) if before else float(f.start_time)
        if manual:
            model = self._manual_model(f.path, t_ref)
            if model is None:
                notes.append("no manual velocity covers the file")
        if model is None and auto:
            if not before:
                notes.append("no topography before the file in its segment: no auto correction")
            else:
                A = before[-1]
                if after:
                    B = after[0]
                    est = self.pair(self.refs[A], self.refs[B])
                    if est.get('v') is not None:
                        model = PairDrift(est['v'], t_ref, 'auto (pair)', f"{os.path.basename(A)} -> {os.path.basename(B)}")
                    else:
                        notes.append(f"the pair {os.path.basename(A)} -> {os.path.basename(B)} is untracked ({est.get('note', '')})")
                else:
                    notes.append("no topography after the file in its segment")
                if model is None:
                    model = self._extrapolated(members, A, t_ref, notes)
        elif model is None and not auto:
            notes.append("auto drift off")
        self._models[key] = (model, notes)
        return model, notes

    def model_for_topo(self, path, auto=True, manual=True):
        """
        The PairDrift of a topo (None when it gets no skew) and the console
        notes: with `manual`, a manual velocity covering it first; with
        `auto`, its pair with the next topo of its segment, else (the last
        topo of a segment, or that pair untracked) its pair with the previous
        one, else the extrapolation rule over the pairs before it.
        """
        auto, manual = bool(auto), bool(manual)
        key = (path, 'topo', auto, manual)
        if key in self._models:
            return self._models[key]
        notes = []
        model = None
        seg = self.segment_key_of(path)
        members = self.segment_members(seg) if seg is not None else []
        ref = self.refs.get(path)
        t_ref = float(ref.mid_time) if ref is not None else 0.0
        if manual:
            model = self._manual_model(path, t_ref)
            if model is None:
                notes.append("no manual velocity covers the topo")
        if model is None and auto:
            if path not in members:
                notes.append("no header geometry for this topo: no skew")
            else:
                j = members.index(path)
                if j + 1 < len(members):
                    est = self.pair(self.refs[path], self.refs[members[j + 1]])
                    if est.get('v') is not None:
                        model = PairDrift(est['v'], t_ref, 'auto (next)',
                                          f"{os.path.basename(path)} -> {os.path.basename(members[j + 1])}")
                    else:
                        notes.append(f"the pair with the next topo is untracked ({est.get('note', '')})")
                else:
                    notes.append("last topo of its segment")
                if model is None and j > 0:
                    est = self.pair(self.refs[members[j - 1]], self.refs[path])
                    if est.get('v') is not None:
                        model = PairDrift(est['v'], t_ref, 'auto (previous)',
                                          f"{os.path.basename(members[j - 1])} -> {os.path.basename(path)}")
                    else:
                        notes.append(f"the pair with the previous topo is untracked ({est.get('note', '')})")
                if model is None and j > 0:
                    model = self._extrapolated(members, members[j - 1], t_ref, notes)
                elif model is None:
                    notes.append("no pair at hand: no skew")
        elif model is None and not auto:
            notes.append("auto drift off")
        self._models[key] = (model, notes)
        return model, notes

    def earlier_topo(self, f, key=None):
        """
        The earlier topo the stripe matching works against: the most recent
        topo of the file's segment before the file whose footprint contains
        the first spectrum's position, else the most recent topo of the
        segment before the file, else the segment's first topo; as a loaded
        Sm4File (the Browser's loaded set when it holds it).  None without
        topos.
        """
        key = self.segment_key_of(f.path) if key is None else key
        if key is None:
            return None
        members = self.segment_members(key)
        if not members:
            return None
        i = self.files.index(f.path) if f.path in self.files else len(self.files)
        before = [p for p in members if self.files.index(p) < i]
        xy0 = f.spec_xy_m[0] if len(f.spec_xy_m) else None
        chosen = None
        if xy0 is not None:
            for p in reversed(before):
                if self.refs[p].contains_m(xy0):
                    chosen = p
                    break
        if chosen is None:
            chosen = before[-1] if before else members[0]
        return self._topo_file(chosen)

    def stripe_reference(self, f, auto=True, manual=True):
        """
        The stripe matching's reference for a grid (StripeReference): the earlier topos of its segment
        whose footprint reaches the grid's area (the grid's recorded positions widened by half the
        grid's size on every side, the reach of the lines' profiles), at most reference_max_topos of
        them, the most recent first, each with its own velocity under the same drift setting as the
        grid (model_for_topo with these auto / manual states; drift none: none).  Falls back to the
        earlier topo (earlier_topo) when no footprint reaches the grid; an empty reference without
        any topo.
        """
        key = self.segment_key_of(f.path)
        members = self.segment_members(key) if key is not None else []
        i = self.files.index(f.path) if f.path in self.files else len(self.files)
        before = [p for p in members if self.files.index(p) < i]
        xy = np.asarray(f.spec_xy_m, dtype=np.float64)
        chosen = []
        if len(xy) and before:
            lo, hi = xy.min(axis=0), xy.max(axis=0)
            m = 0.5 * float(max(hi - lo)) + 1e-12
            lo, hi = lo - m, hi + m
            for p in reversed(before):
                r = self.refs.get(p)
                if r is None:
                    continue
                l_, r_, b_, t_ = r.frame_extent_m()
                c = np.array([[l_, b_], [r_, b_], [r_, t_], [l_, t_]]) @ r.rotation().T + np.asarray(r.center_m)
                if np.all(c.max(axis=0) >= lo) and np.all(c.min(axis=0) <= hi):
                    chosen.append(p)
                if len(chosen) >= int(self.drift_params.get('reference_max_topos', 8)):
                    break
        topos = [self._topo_file(p) for p in chosen]
        topos = [t for t in topos if t is not None]
        if not topos:
            t = self.earlier_topo(f, key)
            topos = [t] if t is not None else []
        models = []
        for t in topos:
            mdl = None
            if auto or manual:
                mdl, _ = self.model_for_topo(t.path, auto, manual)
            models.append(mdl)
        return StripeReference(topos, models)

    # --- the per-file results ---

    def report(self, f, notes, report):
        """The console report of a file's result: its name, the velocity notes, the report lines."""
        print(f"\n{f.name}: drift correction")
        for n in notes:
            print("  " + n)
        for line in report:
            print("  " + line)

    def grid_chain(self, f, model):
        """
        The chained placement of a grid file under a velocity (the module
        docstring, "Grid files"): the content displacement (metres, absolute
        frame; NaN where nothing is placed) of every spectrum ('spec', (N,
        2)), every forward topo pixel ('forward', (ny, nx, 2): the shift of
        the kept spectrum recorded at it, Sm4File.spectrum_pixels), and
        every reverse topo pixel ('reverse', (ny, nx, 2), None without a
        reverse page: each return pass carries its line's forward end
        shift).  Line by line: the first kept point of line i at the line's
        anchor shift a_i; point k of the line at a_i - (d(t_k) - d(t_first))
        with the points' own times on the file's clock (ramp starts,
        spectrum_times); the reverse pass of a line with a return pass
        (grid_timing's lines_done) rigidly at the forward end's shift e_i;
        the link to the next line carries no drift term: a_(i+1) = e_i (a
        line without a kept point passes its anchor on).  a_0 = 0: the file's
        first kept spectrum at its recorded position.  'lines' counts the
        lines placed, 'span_min' the clock time inside the lines in minutes.
        """
        N = f.n_spectra
        xy = np.asarray(f.spec_xy_m, dtype=np.float64)
        acq = np.asarray(f.acquired, dtype=bool)[:N]
        t_spec = np.asarray(f.spectrum_times(), dtype=np.float64)
        n_fast = max(int(f.n_fast), 1)
        n_lines = max(int(f.n_lines), 1)
        tm = f.grid_timing()
        lines_done = int(tm.get('lines_done', 0))
        shift = np.full((N, 2), np.nan)
        end_shift = [None] * n_lines
        a = np.zeros(2)
        placed = 0
        inside = 0.0
        for i in range(n_lines):
            cols = np.arange(i * n_fast, min((i + 1) * n_fast, N))
            kept = cols[acq[cols] & np.isfinite(t_spec[cols])] if len(cols) else cols
            if len(kept) == 0:
                continue
            t_first = float(t_spec[kept[0]])
            d_first = model.displacement(t_first)
            for k in kept:
                shift[k] = a - (model.displacement(float(t_spec[k])) - d_first)
            end_shift[i] = shift[kept[-1]].copy()
            inside += float(t_spec[kept[-1]]) - t_first
            placed += 1
            a = end_shift[i].copy()                         # the link: the recorded displacement, no drift term
        out = {'spec': shift, 'forward': None, 'reverse': None, 'lines': placed, 'span_min': inside / 60.0}
        if not f.topo_pages:
            return out
        try:
            P = f.pixel_xy_m('forward')
            idx, _ = f.spectrum_pixels('forward')
            F = np.full(P.shape, np.nan)
            for k in np.flatnonzero(acq[:len(idx)] & (idx[:, 0] >= 0)):
                if np.all(np.isfinite(shift[k])):
                    F[idx[k, 0], idx[k, 1]] = shift[k]
            out['forward'] = F
            if len(f.topo_pages) > 1:
                Pr = f.pixel_xy_m('reverse')
                R = np.full(Pr.shape, np.nan)
                for i in range(min(n_lines, Pr.shape[0])):
                    if i < lines_done and end_shift[i] is not None:
                        R[i, :, :] = end_shift[i]
                out['reverse'] = R
        except Exception as exc:
            print(f"{f.name}: the grid's topo pixels could not be placed ({exc}); spectra only.")
        return out

    @staticmethod
    def _grid_pixels(f, chain):
        """
        The grid's stripe pixels at their corrected positions from a chain
        (grid_chain): {'forward': (ny, nx, 2), 'reverse': (ny, nx, 2) or
        None}, every pixel at its recorded location (Sm4File.pixel_xy_m) plus
        its shift; NaN where nothing is placed.  None without topo pages.
        """
        pix = {'forward': None, 'reverse': None}
        if not f.topo_pages or chain.get('forward') is None:
            return pix
        pix['forward'] = f.pixel_xy_m('forward') + chain['forward']
        if chain.get('reverse') is not None:
            pix['reverse'] = f.pixel_xy_m('reverse') + chain['reverse']
        return pix

    def result(self, f, loaded=None, stripe=False, auto=True, manual=True, anchor=False):
        """
        Drift correction of a spectra file (computed once per file and
        toggle state, kept in the index; None when it cannot be made).
        The velocity v is the file's PairDrift (model_for_spectra with the
        'auto drift' and 'manual drift' states).
        Baseline, single point: the first sweep stays at the recorded
        position (the file's absolute anchor); sweep k sits at xy_k - v (t_k
        - t_0), t_k the sweep's time (sweep_times) and t_0 the first one's.
        Baseline, grid (stripe False): the chained placement of grid_chain:
        the first kept spectrum at its recorded position; within a line
        every point at its own time on the file's clock; the reverse pass
        rigid at the forward end; the next line at the recorded displacement
        from the return point, no drift term; the grid's stripe pixels from
        the same chain (_grid_pixels).
        Stripe matching (a grid; `stripe` None: the grid's stored fast-axis
        state for the slow-axis setting in force, fast_state; True / 'auto':
        the extrema matching; 'manual': the matching with the grid's dragged
        points of that setting, stripe_points; False / 'off': none): the
        fast axis from drift_correct_grid against the earlier topos
        (stripe_reference), the slow axis from v, line 0 keeping its
        recorded start; the entry also lists the lines whose corrected
        positions leave every reference topo ('off_page').
        Anchor to topo (anchor True, the MATLAB "drift for first point"):
        every position, pixel, and line end shifted rigidly by -v (t_0 -
        t_topo start), t_topo the start of the topo under the file
        (earlier_topo).
        Every spectrum also gets the drift within its own sweep ('shift',
        sweep_shift) and its end: 'xy_end' = its start plus the shift at its
        last energy point, the far end of its drift line (sweep_lines).
        Entry: {'xy_corr', 'xy_end', 'shift', 'line_v' (grids with stripe
        matching: line_velocities; None otherwise), 'report', 'drift' (the
        PairDrift), 'source' (its rule: 'auto (...)', 'manual (...)' or
        'none (stripe match only)'), 'prev' (the before topo's name; the
        reference's names with stripe matching), 'next' (None; single
        points only), 'off_page' (grids), 'lines' (grids
        with stripe matching: drift_correct_grid's per-line factors), 'pix',
        'stripe', 'anchor' (the states used), 'anchor_shift' (the rigid
        shift applied, metres)}.
        """
        if loaded is not None:
            self.loaded = loaded
        auto, manual, anchor = bool(auto), bool(manual), bool(anchor)
        if stripe is None:
            fast = self.fast_state(f, None, auto, manual)
        elif stripe is True:
            fast = 'auto'
        elif stripe is False:
            fast = 'off'
        else:
            fast = str(stripe)
        stripe = fast != 'off' and f.kind == 'grid'
        points = self.stripe_points(f, self.slow_setting(f, auto, manual)) if (stripe and fast == 'manual') else {}
        key = (f.path, fast if stripe else False, tuple(sorted(points.items())), auto, manual, anchor,
               getattr(f, 'sweep_mode', None))                      # a sweep view has its own clock
        if key in self.drift_index:
            return self.drift_index[key]
        if f.spectra is None or f.n_spectra == 0 or not np.any(f.acquired):
            if f.spectra is not None and f.n_spectra:
                print(f"{f.name}: no kept spectrum; nothing to correct.")
            self.drift_index[key] = None
            return None
        model, notes = self.model_for_spectra(f, auto, manual)
        if model is None and stripe:
            # Drift none (or no velocity) with stripe matching on: the slow axis gets no drift, the
            # stripe matching still places every line along the fast axis (its start offset, its
            # forward compression and reverse stretch) against the topos at their recorded places.
            t_spec0 = f.spectrum_times()
            t0_ = float(np.nanmin(t_spec0)) if np.isfinite(t_spec0).any() else float(f.start_time)
            model = PairDrift(np.zeros(2), t0_, 'none (stripe match only)', 'no velocity: slow axis as recorded')
            notes = list(notes) + ["no velocity: the slow axis as recorded; the stripe matching places the lines "
                                   "along the fast axis"]
        if model is None:
            rk = (f.path, 'spec-reported', auto, manual)
            if rk not in self._models:
                self._models[rk] = True
                self.report(f, notes, ["no correction (no usable velocity)"])
            self.drift_index[key] = None
            return None
        notes = list(notes) + [f"velocity, {model.rule}: " + model.text()]
        before = self._before_topo(f)
        prev_name = os.path.basename(before) if before is not None else None
        win = f.spectrum_windows()
        if f.kind == 'point':
            xy0 = f.spec_xy_m[0] if len(f.spec_xy_m) else None
            if xy0 is None:
                self.drift_index[key] = None
                return None
            t_sweep = f.sweep_times()
            fn = model.relative(float(t_sweep[0]))
            d = fn(t_sweep)
            xy_corr = np.tile(xy0, (f.n_spectra, 1)) - d
            report = [f"first sweep at the recorded position; {f.n_spectra - 1} later sweep(s) moved by "
                      f"-v (t_k - t_0) over {(t_sweep[-1] - t_sweep[0]) / 60:.1f} min; the last sweep lies "
                      f"{np.linalg.norm(xy_corr[-1] - xy_corr[0]) * 1e9:.3f} nm from the first"]
            entry = {'xy_corr': xy_corr, 'report': report, 'drift': model, 'prev': prev_name, 'next': None,
                     'off_page': set(), 'lines': [], 'pix': None, 'stripe': False}
            t_0 = float(t_sweep[0])
        else:
            t_spec = f.spectrum_times()
            t_0 = float(np.nanmin(t_spec)) if np.isfinite(t_spec).any() else float(f.start_time)
            fn = model.relative(t_0)
            prev = None
            if stripe:
                prev = self.stripe_reference(f, auto, manual)
            if stripe and prev:
                xy_corr, report, lines, pix = drift_correct_grid(f, prev, fn, t_0, self.drift_params, points=points)
                setting = "drift none" if model.rule.startswith('none') else f"the {model.rule} velocity"
                report.insert(0, f"stripe matching against {prev.name}, placed with {setting} as the grid; "
                                 f"the slow axis from that velocity only"
                                 + (f"; {len(points)} manual point(s)" if points else ""))
            else:
                prev = None
                if stripe:
                    notes.append("stripe matching asked but no earlier topo reaches the grid: baseline only")
                xy = np.asarray(f.spec_xy_m, dtype=np.float64)
                chain = self.grid_chain(f, model)
                xy_corr = xy + chain['spec']
                lines = []
                pix = self._grid_pixels(f, chain)
                ok = np.all(np.isfinite(xy_corr), axis=1)
                moved = np.linalg.norm(xy_corr[ok] - xy[ok], axis=1) * 1e9 if ok.any() else np.zeros(0)
                report = [f"first spectrum at the recorded position; {chain['lines']} line(s) chained: within a line "
                          f"every point at its own time ({chain['span_min']:.1f} min inside the lines), the reverse pass "
                          f"rigid at the forward end, the next line at the recorded displacement from the return point; "
                          f"the largest move {(np.max(moved) if len(moved) else 0):.3f} nm"]
            off_page = set()
            ref = prev if prev is not None else self.refs.get(before)
            if ref is not None:
                inside = (ref.contains if prev is not None else ref.contains_m)
                for i in range(f.n_lines):
                    cols = np.arange(i * f.n_fast, min((i + 1) * f.n_fast, f.n_spectra))
                    ok = np.all(np.isfinite(xy_corr[cols]), axis=1)
                    if np.any(ok) and not all(inside(xy_corr[k]) for k in cols[ok]):
                        off_page.add(i)
                if off_page:
                    report.append("lines leaving " + ("every reference topo" if prev is not None else "the page of " + ref.name)
                                  + ": " + ", ".join(str(i) for i in sorted(off_page)))
            if prev is not None:
                prev_name = prev.name
            entry = {'xy_corr': xy_corr, 'report': report, 'drift': model, 'prev': prev_name, 'off_page': off_page,
                     'lines': lines, 'pix': pix, 'stripe': fast if stripe else False}
        entry['source'] = model.rule
        entry['anchor'] = anchor
        entry['anchor_shift'] = np.zeros(2)
        if anchor:
            # The MATLAB "drift for first point": the whole file moved to where the surface sat when the
            # topo under it began, -v (t_0 - t_topo start).
            ref_topo = self.earlier_topo(f)
            if ref_topo is None:
                entry['report'].append("anchor to topo: no topo under the file; the recorded anchor kept")
            else:
                sh = -(model.displacement(t_0) - model.displacement(float(ref_topo.start_time)))
                entry['anchor_shift'] = sh
                entry['xy_corr'] = entry['xy_corr'] + sh[None, :]
                if entry['pix'] is not None:
                    for d_ in ('forward', 'reverse'):
                        if entry['pix'].get(d_) is not None:
                            entry['pix'][d_] = entry['pix'][d_] + sh
                entry['report'].append(f"anchor to topo: the file moved rigidly by ({sh[0] * 1e9:+.3f}, {sh[1] * 1e9:+.3f}) nm "
                                       f"to the surface at the start of {ref_topo.name} ({(t_0 - ref_topo.start_time) / 60:.1f} min earlier)")
        # The drift within every sweep (the stripe layer's forward compression along the fast axis
        # where measured, the baseline otherwise) and the far end of every spectrum's drift line.
        line_v = self.line_velocities(f, entry['lines']) if (f.kind != 'point' and entry['lines']) else None
        entry['line_v'] = line_v
        shift = self._sweep_shift(f, model, line_v)
        entry['shift'] = shift
        xy_end = np.full_like(entry['xy_corr'], np.nan)
        if shift is not None and shift.shape[1] > 0:
            last = shift[:, -1, :]
            n = min(len(last), len(xy_end))
            ok = np.all(np.isfinite(last[:n]), axis=1) & np.all(np.isfinite(entry['xy_corr'][:n]), axis=1)
            if ok.any():
                xy_end[:n][ok] = entry['xy_corr'][:n][ok] + last[:n][ok]
                n_v = int(np.sum(np.isfinite(line_v[:n]) & ok)) if line_v is not None else 0
                entry['report'].append(f"drift lines: the longest within-sweep drift "
                                       f"{np.nanmax(np.linalg.norm(last[:n][ok], axis=1)) * 1e10:.3f} A"
                                       + (f"; the along-line rate of {n_v} spectra from their lines' forward "
                                          f"compression, the file's velocity elsewhere" if n_v else "; from the file's velocity"))
        else:
            # No energy points: the velocity over the ramp window.
            n = min(len(win), len(xy_end))
            ok = np.isfinite(win[:n, 0]) & np.isfinite(win[:n, 1]) & np.all(np.isfinite(entry['xy_corr'][:n]), axis=1)
            if ok.any():
                d_end = model.displacement(win[:n, 1][ok]) - model.displacement(win[:n, 0][ok])
                xy_end[:n][ok] = entry['xy_corr'][:n][ok] - d_end
        entry['xy_end'] = xy_end
        self.drift_index[key] = entry
        self.report(f, notes, entry['report'])
        return entry

    def line_velocities(self, f, lines):
        """
        (N,) the along-line drift velocity (metres per second, along the
        grid's fast axis, the surface's features' displacement along it) of
        every spectrum's line from the line's own forward compression
        measured by the stripe matching, v_i = (1 - b_f) L_f / T_f, L_f the
        line's kept range (its first kept spectrum to its last) and T_f the
        window from the first kept spectrum's ramp start to the last's ramp
        end (grid_timing); NaN for the spectra of a line without a measured
        forward factor or with fewer than three kept spectra (and for every
        spectrum of a single point).
        """
        v = np.full(f.n_spectra, np.nan)
        if not lines or f.kind == 'point' or f.n_fast < 2:
            return v
        fast, _ = f.grid_axes()
        if fast is None:
            return v
        xy = f.spec_xy_m
        step = float(np.linalg.norm(xy[1] - xy[0]))
        tm = f.grid_timing()
        acq = np.asarray(f.acquired, dtype=bool)
        n_fast = int(f.n_fast)
        for ln in lines:
            i = ln['i']
            cols = np.arange(i * n_fast, min((i + 1) * n_fast, f.n_spectra))
            kept = cols[acq[cols]]
            if ln['b_f_measured'] and len(kept) >= 3:
                t_a, t_b = tm['ramp_start'][kept[0]], tm['ramp_end'][kept[-1]]
                if np.isfinite(t_a) and np.isfinite(t_b) and t_b > t_a:
                    L_f = float(kept[-1] - kept[0]) * step
                    v[cols] = (1.0 - ln['b_f']) * L_f / float(t_b - t_a)
        return v

    def _sweep_shift(self, f, model, line_v):
        """
        sweep_shift of a file from its PairDrift and its line velocities
        (line_velocities; None without stripe matching or for a single
        point): (N, npts, 2) metres, NaN for a spectrum not kept; None
        without energy points.
        """
        if f.spectra is None:
            return None
        et = f.energy_times()
        if et.shape[1] == 0:
            return None
        win = f.spectrum_windows()
        d = model.displacement(et) - model.displacement(win[:, 0])[:, None, :]
        out = -d
        if line_v is not None:
            fast, _ = f.grid_axes()
            has = np.isfinite(line_v)
            if fast is not None and has.any():
                along = out[has] @ fast                                          # (n, npts) the baseline along the line
                perp = out[has] - along[:, :, None] * fast[None, None, :]        # the baseline across it, kept
                dt = et[has] - win[has, 0][:, None]
                out[has] = perp - (line_v[has][:, None] * dt)[:, :, None] * fast[None, None, :]
        return out

    def sweep_shift(self, f, loaded=None, stripe=False, auto=True, manual=True, anchor=False):
        """
        The drift within every sweep of a spectra file: (N, npts, 2)
        absolute metres, the content's position relative to the sweep's
        start at every energy point's time (Sm4File.energy_times, the file's
        column order): -v (t - t_start) with the pair velocity v; on a grid
        line with a measured forward factor (stripe matching on) the line's
        own forward compression along the fast axis instead, v across it.
        NaN for a spectrum not kept; None without a result or energy points.
        """
        d = self.result(f, loaded, stripe, auto, manual, anchor)
        if d is None:
            return None
        return d['shift']

    def sweep_lines(self, f, loaded=None, stripe=False, auto=True, manual=True, anchor=False):
        """
        The drift lines of a spectra file: (N, npts, 2) absolute metres, for
        every spectrum the polyline from its start (the corrected position,
        result's 'xy_corr') through its position at every energy point's
        time (sweep_shift), whose last point is 'xy_end': a straight line
        along -v through the sweep.  None without a result or energy points.
        """
        d = self.result(f, loaded, stripe, auto, manual, anchor)
        if d is None or d['shift'] is None:
            return None
        return d['xy_corr'][:, None, :] + d['shift']

    # --- the corrected topographies ---

    def corrected_topo(self, sm4, direction, active, loaded=None, auto=True, manual=True):
        """
        The processed topography of a file (topo.process_topo of its forward
        or reverse page; 'average' the two corrected passes combined as
        topo.processed_topo combines them) with every pixel moved to where
        its content sits at the time of the image's first pixel: pixel
        (i, j), acquired at t(i, j) (Sm4File.pixel_times), is displaced by
        -v (t - t_first) with the topo's own velocity (model_for_topo with the
        'auto drift' and 'manual drift' states); a grid's own topo takes the
        grid's velocity (model_for_spectra) and the chained shifts of
        grid_chain (forward pixels by their spectra, each reverse row by its
        forward end), so it agrees with the scan area's bands;
        and the image is resampled onto its own pixel grid by inverting that
        displacement field (one fixed-point step; bilinear; NaN where no
        content lands).  The frame of the image is unchanged, so it is drawn
        as before.  The processed image itself without a velocity (the
        console says why).  Cached per file, direction, processor flags, and
        model.
        """
        if loaded is not None:
            self.loaded = loaded
        is_grid = sm4.kind == 'grid' and sm4.spectra is not None
        if is_grid:
            # A grid's own topo: the grid's velocity (its before / after pair) and the chained placement.
            model, notes = self.model_for_spectra(sm4, auto, manual)
        else:
            model, notes = self.model_for_topo(sm4.path, auto, manual)
        rk = (sm4.path, 'topo-reported', bool(auto), bool(manual))
        if rk not in self._models:
            self._models[rk] = True
            if model is None:
                print(f"{sm4.name}: no drift skew (" + "; ".join(notes) + ")")
            else:
                print(f"{sm4.name}: skewed with the {model.rule} velocity, "
                      + model.text(self.scale_of(sm4.path) if self.scale_of is not None else None))
        key = (sm4.path, direction, tuple(bool(a) for a in active), id(model), getattr(sm4, 'sweep_mode', None))
        img = self._img_cache.get(key)
        if img is not None:
            return img
        from sm4_Browser import topo as topo_mod
        chain = self.grid_chain(sm4, model) if (is_grid and model is not None) else None
        if direction == 'average':
            if len(sm4.topo_pages) < 2:
                out = self._corrected_pass(sm4, 'forward', active, model, topo_mod, chain)
            else:
                a = self._corrected_pass(sm4, 'forward', active, model, topo_mod, chain)
                b = self._corrected_pass(sm4, 'reverse', active, model, topo_mod, chain)
                a, b = topo_mod.match_ranges([a, b])
                both = np.isfinite(a) & np.isfinite(b)
                out = np.where(np.isfinite(a), a, b)
                out[both] = 0.5 * (a[both] + b[both])
                if np.any(np.isfinite(out)):
                    out = out - np.nanmin(out)
        else:
            out = self._corrected_pass(sm4, direction, active, model, topo_mod, chain)
        self._img_cache[key] = out
        return out

    def _corrected_pass(self, sm4, direction, active, model, topo_mod, chain=None):
        """
        One pass of a file resampled onto its own grid: the content of pixel
        (i, j) displaced by -(d(t) - d(t_first)) at its pixel time (a topo
        image), or by the chained shift of grid_chain (a grid's own topo:
        forward pixels by their spectra, each reverse row by its forward
        end; a pixel without one takes its row's nearest placed shift, a
        row without any the zero shift).
        """
        img = topo_mod.process_topo(sm4.topo_raw(direction), active)
        if model is None or sm4.nx is None:
            return img
        ny, nx = img.shape
        if chain is not None:
            sh = chain.get(direction)
            if sh is None or sh.shape[:2] != img.shape or not np.isfinite(sh).any():
                return img
            disp = np.array(sh, dtype=np.float64)
            for i in range(ny):
                ok = np.all(np.isfinite(disp[i]), axis=1)
                if not ok.any():
                    disp[i] = 0.0
                elif not ok.all():
                    jj = np.arange(nx)
                    for c in range(2):
                        disp[i, ~ok, c] = np.interp(jj[~ok], jj[ok], disp[i, ok, c])
            disp_uv = disp.reshape(-1, 2) @ sm4.rotation()
        else:
            try:
                t = sm4.pixel_times(direction)
            except ValueError:
                return img
            if t.shape != img.shape or not np.isfinite(t).any():
                return img
            t_anchor = float(t[0, 0]) if np.isfinite(t[0, 0]) else float(np.nanmin(t))
            tt = np.where(np.isfinite(t), t, t_anchor)
            d = model.displacement(tt) - model.displacement(t_anchor)        # (ny, nx, 2), absolute
            disp_uv = (-d).reshape(-1, 2) @ sm4.rotation()                   # where the content sits: -(d(t) - d(t0)), frame
        dj = (disp_uv[:, 0] / sm4.xscale_m).reshape(ny, nx)
        di = (disp_uv[:, 1] / sm4.yscale_m).reshape(ny, nx)
        jj, ii = np.meshgrid(np.arange(nx, dtype=np.float64), np.arange(ny, dtype=np.float64))
        si, sj = ii - di, jj - dj
        di2 = map_coordinates(di, [si, sj], order=1, mode='nearest')
        dj2 = map_coordinates(dj, [si, sj], order=1, mode='nearest')
        si, sj = ii - di2, jj - dj2
        out = map_coordinates(img, [si, sj], order=1, mode='constant', cval=np.nan)
        # A pixel whose bilinear neighbours include a never-acquired (NaN) pixel takes its nearest
        # source pixel when that one is acquired, so the edges of the acquired area are kept.
        gap = ~np.isfinite(out)
        if gap.any():
            near = map_coordinates(img, [si, sj], order=0, mode='constant', cval=np.nan)
            out[gap] = near[gap]
        return out

    # --- the heatmap rows and the map pixels ---

    def row_shear(self, f, cols, loaded=None, stripe=False, auto=True, manual=True):
        """
        The within-sweep shear of the Line window's rows for the spectra in
        `cols` (global column indices): for each, the along-line component
        (the grid's fast axis) of the sweep's drift (sweep_shift's rule) at
        the npts + 1 energy-cell edges of its sweep, edge e at the sweep's
        start + e x its sample time (the sweep window over the points, the
        derived Sm4File.dwell_s; the first edge at the start, shear 0), in
        metres, in the file's column order of the energy axis: -v_i x (edge
        time - ramp start) on a line with a measured forward factor (stripe
        matching on; v_i the line's own forward compression,
        line_velocities), -(v . fast) x (edge time - ramp start) with the
        pair velocity otherwise.  Shape (len(cols), npts + 1); NaN rows for
        spectra not kept; None without a result or a fast axis.
        """
        d_res = self.result(f, loaded, stripe, auto, manual)
        fast, _ = f.grid_axes()
        if d_res is None or fast is None or f.spectra is None:
            return None
        model = d_res['drift']
        line_v = d_res.get('line_v')
        cols = np.asarray(cols, dtype=int)
        win = f.spectrum_windows()
        npts = len(f.energy)
        # Every energy-cell edge at its own time (Sm4File.energy_edge_times: a sweep acquired down the
        # axis, a reverse sweep of a file with several sweeps per pixel, has its last edge first).
        E = f.energy_edge_times() if hasattr(f, 'energy_edge_times') else None
        out = np.full((len(cols), npts + 1), np.nan)
        for r, k in enumerate(cols):
            if k < 0 or k >= len(win) or not np.isfinite(win[k, 0]):
                continue
            t_e = E[k] if E is not None else win[k, 0] + (win[k, 1] - win[k, 0]) * np.arange(npts + 1) / max(npts, 1)
            if line_v is not None and k < len(line_v) and np.isfinite(line_v[k]):
                out[r] = -line_v[k] * (t_e - win[k, 0])
            else:
                d = model.displacement(t_e) - model.displacement(float(win[k, 0]))
                out[r] = -(d @ fast)
        return out

    def map_shift(self, f, loaded=None, stripe=False, auto=True, manual=True):
        """
        The shift of every map pixel: (N, npts, 2) absolute metres, where
        spectrum k's content at energy point e sits relative to the sweep's
        start (sweep_shift); NaN for a spectrum not kept; None without a
        result or energy points.
        """
        return self.sweep_shift(f, loaded, stripe, auto, manual)


class DriftWindow:
    """
    The manual drift window of one file (the browser's 'drift' button, the
    Line and Map windows' 'drift window' button): the pair its auto velocity
    comes from (Drift.window_pair: a spectra file's before and after topos,
    never its own; a topo and the next of its segment, else the previous
    and it), side by side at absolute positions in calibrated nm, every
    object at the scale in force at its own file's time (`scalefactor`: the
    browser's scale_of(path), or one number for everything; the topo image
    at its topo's scale, the spectra at the file's, the other topo's frame
    at that topo's), each panel titled with the name of the file it draws,
    through the caller's topo processors (`active_procs`), a spectra file's
    recorded positions on both, and on each panel the recorded frame of the
    other topo of the pair (its
    centre, angle and extent as recorded, no drift correction: the drift is
    measured from the uncorrected topos), a thin black outline drawn last
    so it lies on top of the topo, the markers and the arrows
    (plots.draw_topo_outline).  Both panels show one view: the outline of
    the two frames together centred, the same limits and the same box
    shape on both, so neither image is centred on its own.  Clicks place
    numbered markers
    (widgets.PairMarkers: the k-th on the left pairs with the k-th on the
    right, pair k in its own colour; a press on a marker drags it); a
    click on either topo places the partner on the other at once where the
    velocity that currently governs the pair puts it (_seed_velocity: the
    pair's applied manual velocity, else the file's velocity with both
    sources on, else the pair's tracked velocity; seed_partner, the module
    docstring, "Seeded markers"), with the
    pair's arrow on both panels.
    'Markers' is the pair count (0 to 8; a click raises it), 'clear
    markers' empties the draft.  The markers are a draft: the readout
    lists the pairs (seeded and unmoved, adjusted by how much, or placed by
    hand).  One velocity box, 'v (nm/h, cal)' ("Vx, Vy" in calibrated nm/h
    by the window's file's scale, self.scale, fixed at opening; the stored
    velocity is nominal, recorded metres, _parse_box), is what gets
    applied; every readout shows the calibrated value first and the nominal
    beside it (_v_text, marker_pair_text).  'set from markers' fills the box
    from the draft (every pair seeded by an auto velocity and unmoved: the
    pair's tracked velocity, confirmed; otherwise the pairs'
    least-squares fit; the MATLAB readout per pair printed) and stores
    nothing; typing or Enter in the box only fills it; 'apply velocity'
    stores the box as the span of this pair (Drift.set_manual_span:
    'confirmed' when the box still holds the unmoved seeded value,
    'adjusted' when it still holds the pairs' fit, 'typed' when it was typed
    or edited; an empty box removes the pair's span) and shows the applied
    velocity.  Applying changes the one Drift, which tells its subscribers
    (Drift.notify: the browser, every open Line and Map window and the
    stripe window redraw from it, whichever drift window made the change);
    `on_change`, when a
    caller outside the package gives one, runs as well (_changed).  A
    stored span between the two topos is restored at opening, its markers
    and its velocity in the box.  '< before' / 'before >' and '< after' /
    'after >' step the earlier and the later topo through the file's
    segment in time order (the earlier always before the later): the pair
    is redrawn, the draft emptied, the box cleared, the new pair's stored
    span restored (its markers and its velocity in the box), and 'apply
    velocity' then stores the span between the topos shown, which covers
    every file between them, whatever its kind (Drift._manual_model); the
    stepped topos are drawn at their own scales, the box and the readouts
    stay calibrated by the window's file's scale.  Raises ValueError when
    the file has no pair to show.
    """

    DESIGN = (14, 7.6)     # the design size (inches): the text zoom of the window is its size relative to this (qtplot.window_zoom)

    def __init__(self, drift, f, scalefactor, active_procs, on_change=None):
        from sm4_Browser import widgets, plots
        from sm4_Browser import topo as topo_mod
        self.drift = drift
        self.f = f
        # The scale: scalefactor(path) (the browser's timeline: each topo of the pair is drawn at the
        # scale in force at its own time, the panels in calibrated nm) or one number for everything.
        self.scale_of = scalefactor if callable(scalefactor) else (lambda p, v=float(scalefactor): v)
        self.scale = self.scale_of(f.path)       # the file's scale: the readouts and the box are calibrated by it
        self.scalefactor = self.scale            # the same under the name the other windows use
        self.active = list(active_procs)
        self._on_change = on_change
        a, b = drift.window_pair(f)
        self.a, self.b = a, b
        self.v_auto = drift.auto_velocity_of(a.path, b.path)
        k = 1e9 / self.scale
        self.k = k
        self.fig = plt.figure(figsize=self.DESIGN)
        try:
            self.fig.canvas.manager.set_window_title(f"manual drift: {f.name}")
        except Exception:
            pass
        self.ax_a = self.fig.add_axes([0.05, 0.30, 0.42, 0.64])
        self.ax_b = self.fig.add_axes([0.55, 0.30, 0.42, 0.64], sharex=self.ax_a, sharey=self.ax_a)
        # The topos of the file's segment in time order: what the step buttons walk through.
        seg = drift.segment_key_of(f.path)
        self.members = list(drift.segment_members(seg)) if seg is not None else []
        self._draw_pair()
        self.markers = widgets.PairMarkers(self.fig, [(self.ax_a, a.path, a.name), (self.ax_b, b.path, b.name)],
                                           self._resolve_on, n_pairs=0, seed=self._seed,
                                           on_count=lambda n: self.sld_markers.set(n),
                                           to_display=lambda xy_m, p: np.asarray(xy_m, dtype=np.float64) * self.k_of(p))
        self.markers.on_change(self._on_markers)
        self.sld_markers = widgets.make(self.fig, 'Markers', 0.08, 0.235, value=0, vmax=widgets.MARKER_PAIRS_MAX)
        self.sld_markers.on_change(lambda val: self.markers.set_pairs(int(round(float(val)))))
        self.btn_clear = widgets.make(self.fig, 'clear markers', 0.20, 0.225)
        self.btn_clear.on_change(self._on_clear)
        self.btn_set_v = widgets.make(self.fig, 'set from markers', 0.29, 0.225)
        self.btn_set_v.on_change(self._on_set_from_markers)
        self.box_v = widgets.make(self.fig, 'drift v', 0.49, 0.228, value='')
        self.box_v.on_change(self._on_typed)
        self.btn_apply = widgets.make(self.fig, 'apply velocity', 0.59, 0.225)
        self.btn_apply.on_change(self._on_apply)
        # The step buttons: the earlier topo and the later topo each one step back or forward in time.
        self.btn_before_prev = widgets.make(self.fig, '< before', 0.70, 0.225)
        self.btn_before_prev.on_change(lambda: self._step('before', -1))
        self.btn_before_next = widgets.make(self.fig, 'before >', 0.748, 0.225)
        self.btn_before_next.on_change(lambda: self._step('before', +1))
        self.btn_after_prev = widgets.make(self.fig, '< after', 0.83, 0.225)
        self.btn_after_prev.on_change(lambda: self._step('after', -1))
        self.btn_after_next = widgets.make(self.fig, 'after >', 0.878, 0.225)
        self.btn_after_next.on_change(lambda: self._step('after', +1))
        self.txt = self.fig.text(0.05, 0.02, '', fontsize=7, family='monospace', ha='left', va='bottom')
        self.help_txt = self.fig.text(0.05, 0.17, '', fontsize=8, ha='left', va='bottom')
        self._write_help()
        # The box's provenance: where its text came from ('confirmed', 'markers', or 'typed'), the
        # markers it was fitted from, and the text as written, so an edited box applies as typed.
        self._box_kind = None
        self._box_markers = None
        self._box_text = ''
        self._applied = None
        self._restore_span()
        plt.show()

    def k_of(self, path):
        """The drawing factor of a file: calibrated nm per recorded metre at the scale in force at its time."""
        return 1e9 / self.scale_of(path)

    def _draw_pair(self):
        """
        Both panels drawn anew for the pair (self.a, self.b): the processed forward pass of each at
        absolute positions, every object at the scale in force at its own file's time (the panels
        are in calibrated nm: the topo image at its scale, the file's spectra at the file's, the
        other topo's recorded frame at that topo's), titled with its file, the frame drawn last and
        on top, one view centred on both frames.  No drift correction: the drift is measured here.
        """
        from sm4_Browser import plots
        from sm4_Browser import topo as topo_mod
        f = self.f
        corners = []
        self._outlines = {}          # panel -> the artists of the other topo's recorded frame (drawn last)
        for ax, t, other, label in ((self.ax_a, self.a, self.b, "earlier"), (self.ax_b, self.b, self.a, "later")):
            ax.clear()
            img = topo_mod.process_topo(t.topo_raw('forward'), self.active) * 1e9
            plots.draw_topo_absolute(ax, t, img, self.scale_of(t.path), cmap=plots.TOPO_CMAP)
            corners.append(t.corners_m() * self.k_of(t.path))
            corners.append(other.corners_m() * self.k_of(other.path))
            if f.kind != 'topo' and len(f.spec_xy_m):
                xy = np.asarray(f.spec_xy_m, dtype=np.float64) * self.k_of(f.path)
                acq = np.asarray(f.acquired, dtype=bool)[:len(xy)] if f.kind == 'grid' else np.ones(len(xy), dtype=bool)
                ax.plot(xy[acq, 0], xy[acq, 1], ls='none', marker='o', ms=2.5, color='cyan', mec='black', mew=0.3)
            # The title above the panel is the name of the file drawn in it; the rest of the line
            # (earlier or later, the pass, the start time, whether it is the window's file) goes to
            # the window's info panel (qtplot.short_title).
            mark = ", this file" if t is f or t.path == f.path else ""
            ax.set_title(f"{t.name}: {label} topography, forward pass, "
                         f"{datetime.fromtimestamp(t.start_time).strftime('%m-%d %H:%M:%S')}{mark}", fontsize=8)
            ax.set_xlabel("x (nm, absolute)")
            ax.set_ylabel("y (nm, absolute)")
            ax.ticklabel_format(useOffset=False, style='plain')
            ax.set_aspect('equal', adjustable='box')
            # The other topo of the pair as its recorded frame, drawn last: a thin black outline on
            # top of everything on the panel (the markers and arrows drawn later sit below its zorder).
            self._outlines[ax] = plots.draw_topo_outline(ax, other, self.scale_of(other.path))
        # One view on both panels: the outline of the two frames together, centred, with a margin.
        # The limits are set on each axes (not only on the one the other shares), so that each
        # panel's box is shrunk to the equal aspect the same way and neither image sits off-centre.
        allc = np.vstack(corners)
        mx = 0.05 * max(np.ptp(allc[:, 0]), 1e-9)
        my = 0.05 * max(np.ptp(allc[:, 1]), 1e-9)
        for ax in (self.ax_a, self.ax_b):
            ax.set_xlim(allc[:, 0].min() - mx, allc[:, 0].max() + mx)
            ax.set_ylim(allc[:, 1].min() - my, allc[:, 1].max() + my)

    def _write_help(self):
        """The help line under the panels, for the pair shown (its seeding velocity)."""
        v0, src0 = self._seed_velocity()
        seeding = (f"a click on either topo places its partner on the other at once, with the current velocity "
                   f"({src0}); drag to adjust" if v0 is not None else
                   "no velocity covers this pair yet: place each partner by hand (or apply a velocity)")
        self.help_txt.set_text(f"{self.f.name}: click the same feature in both topos (the k-th marker on the left pairs "
                               f"with the k-th on the right).\n{seeding}; 'set from markers' puts the markers' velocity "
                               f"in the 'v (nm/h, cal)' box (calibrated by this file's scale factor {self.scale:g}; the "
                               "nominal value is shown beside it), or type one; 'apply velocity' stores the box for the files between "
                               f"these two topos ({self.a.name} -> {self.b.name}); '< before' / 'before >' and "
                               "'< after' / 'after >' step the topos through the segment.")

    def _restore_span(self):
        """A stored span between the pair shown put into the window: its markers in the draft, its velocity in the box, the readout."""
        a, b = self.a, self.b
        sp = self.drift.span_between(a.path, b.path)
        if sp is None:
            self._show_draft()
            return
        m = np.asarray(sp['markers']).reshape(-1, 6)
        if sp['a'] != a.path:                                  # stored the other way round
            m = m[:, [3, 4, 5, 0, 1, 2]]
        if len(m):
            self.markers.restore(m, kind=sp.get('kind', 'adjusted'))     # each panel at its own scale
            self.sld_markers.set(self.markers.n_pairs)
        if sp.get('v') is not None:
            self._fill_box(sp['v'], {'confirmed': 'confirmed', 'adjusted': 'markers'}.get(sp.get('kind'), 'typed'),
                           m if len(m) else None)
        self._applied = sp
        self._show(sp)

    def _step(self, which, d):
        """
        '< before' / 'before >' ('after'): the earlier (later) topo one step
        back (d = -1) or forward (+1) through the segment's topos in time
        order; refused at the segment's ends and where the earlier would
        not stay before the later.  The pair is redrawn, the draft emptied,
        the box cleared, the new pair's stored span restored (markers, box,
        readout), the help line rewritten.
        """
        paths = self.members
        if self.a.path not in paths or self.b.path not in paths:
            print("drift window: the pair is not in the file's segment; no stepping.")
            return
        ia, ib = paths.index(self.a.path), paths.index(self.b.path)
        if which == 'before':
            ia += int(d)
        else:
            ib += int(d)
        if not (0 <= ia < ib < len(paths)):
            print(f"drift window: no {'earlier' if d < 0 else 'later'} topo for the {which} panel "
                  f"(the segment holds {len(paths)} topos; the earlier stays before the later).")
            return
        self.a = self.drift._topo_file(paths[ia])
        self.b = self.drift._topo_file(paths[ib])
        self.v_auto = self.drift.auto_velocity_of(self.a.path, self.b.path)
        self._draw_pair()
        self.markers.retarget([(self.ax_a, self.a.path, self.a.name), (self.ax_b, self.b.path, self.b.name)])
        self.sld_markers.set(0)
        self.box_v.set('')
        self._box_kind, self._box_markers, self._box_text = None, None, ''
        self._applied = None
        self._write_help()
        self._restore_span()
        print(f"drift window: {self.a.name} -> {self.b.name}" + (f", auto v = {self._v_text(self.v_auto)}" if self.v_auto is not None else ", pair not tracked"))
        self.fig.canvas.draw_idle()

    def is_open(self):
        """True while the window is open (the figure itself is tested, not its reusable number)."""
        return plt.fignum_exists(self.fig)

    def raise_(self):
        """The window brought to the front."""
        try:
            self.fig.canvas.manager.show()
        except Exception:
            pass

    def _topo(self, path):
        """The loaded topo of a panel's path (self.a or self.b)."""
        return self.a if path == self.a.path else self.b

    def _resolve_on(self, path, x, y):
        """A marker's data at (x, y) in the panel of `path` (drawing units at that topo's scale): 'xy_m' (absolute metres), 't' (the forward-pass pixel time, pixel_time_at), 'px_m' (the larger pixel size); None outside the topo."""
        t = self._topo(path)
        xy_m = np.array([x, y], dtype=np.float64) / self.k_of(path)     # the panel's topo at its own scale
        if not t.contains_m(xy_m):
            return None
        return {'xy_m': xy_m, 't': pixel_time_at(t, xy_m, 'forward'), 'px_m': max(abs(t.xscale_m), abs(t.yscale_m))}

    def _seed_velocity(self):
        """
        The velocity that currently governs this pair, for placing partners: the velocity applied
        to the pair here (its manual span), else the file's velocity as the Drift gives it with
        both sources on (a manual span covering the file, else the auto pair, else the auto
        extrapolation); independent of the opener's drift toggles.  ((2,) m/s or None, its source).
        """
        sp = self.drift.span_between(self.a.path, self.b.path)
        if sp is not None and sp.get('v') is not None:
            return np.asarray(sp['v'], dtype=np.float64), f"manual ({sp.get('kind', 'adjusted')})"
        try:
            if self.f.kind == 'topo':
                model, _ = self.drift.model_for_topo(self.f.path, True, True)
            else:
                model, _ = self.drift.model_for_spectra(self.f, True, True)
        except Exception:
            model = None
        if model is not None:
            return np.asarray(model.v, dtype=np.float64), model.rule
        if self.v_auto is not None:
            return np.asarray(self.v_auto, dtype=np.float64), 'auto (pair)'
        return None, None

    def _seed(self, path, xy_m, t):
        """
        The partner of a marker clicked on either topo: (other path, xy_m, t, px_m, source) where the
        current velocity (_seed_velocity) puts it, or None (no velocity, or the spot lies outside the
        other topo).
        """
        v, source = self._seed_velocity()
        if v is None:
            print("drift markers: no velocity covers this pair; place the partner by hand.")
            return None
        src, dst = (self.a, self.b) if path == self.a.path else (self.b, self.a)
        res = seed_partner(src, dst, xy_m, t, v)
        if res is None:
            print(f"drift markers: the {source} velocity puts the partner outside {dst.name}; place it by hand.")
            return None
        xy_to, t_to = res
        return dst.path, xy_to, t_to, max(abs(dst.xscale_m), abs(dst.yscale_m)), source

    def _v_text(self, v):
        """A velocity ((2,) m/s, nominal) as text: calibrated by the file's scale first (the scan area's units), the nominal beside it."""
        s = self.scale
        return (f"({v[0] * 3.6e12 / s:+.3f}, {v[1] * 3.6e12 / s:+.3f}) nm/h calibrated (/ {s:g}); "
                f"nominal ({v[0] * 3.6e12:+.3f}, {v[1] * 3.6e12:+.3f}) nm/h = ({v[0] * 1e12:+.2f}, {v[1] * 1e12:+.2f}) pm/s")

    def _fill_box(self, v, kind, markers):
        """The box set to v ((2,) m/s, nominal) in calibrated nm/h (the file's scale), with its provenance (kind 'confirmed', 'markers', or 'typed') and the markers it came from."""
        v = np.asarray(v, dtype=np.float64).reshape(2) * 3.6e12 / self.scale
        text = f"{v[0]:.4g}, {v[1]:.4g}"
        self.box_v.set(text)
        self._box_kind, self._box_markers, self._box_text = kind, markers, text

    def _parse_box(self):
        """The box as a velocity: ((2,) m/s, or None when empty), or False when the text is not 'Vx, Vy'."""
        txt = str(self.box_v.value).strip().replace(';', ',')
        if not txt:
            return None
        parts = [p for p in txt.replace('Vx=', '').replace('Vy=', '').split(',') if p.strip()]
        try:
            vx, vy = (float(p) for p in parts[:2])
            if len(parts) != 2:
                raise ValueError
        except ValueError:
            return False
        # The box is in calibrated nm/h (the file's scale): the stored velocity is nominal, in recorded metres.
        return np.array([vx, vy], dtype=np.float64) / 3.6e12 * self.scale

    def _box_line(self):
        """The readout's line for the box: its velocity and where it came from, not applied."""
        v = self._parse_box()
        if v is None:
            return "box empty: 'apply velocity' removes this pair's velocity (the files between go back to auto)"
        if v is False:
            return f"box: '{self.box_v.value}' is not 'Vx, Vy' in calibrated nm/h"
        edited = str(self.box_v.value).strip() != self._box_text
        origin = ("typed" if edited or self._box_kind in (None, 'typed') else
                  ("from the markers, the auto velocity confirmed" if self._box_kind == 'confirmed' else
                   "from the markers, fitted"))
        return f"box: v = {self._v_text(v)}, {origin}; not applied: 'apply velocity' stores it for this pair"

    def _show(self, span):
        """The readout for a stored span: its marker pairs (marker_pair_text, nominal with the calibrated value in brackets) and the applied velocity (the last 12 lines)."""
        lines = [marker_pair_text(k, rec, self.scale) for k, rec in enumerate(span.get('pairs', []))]
        v = span.get('v')
        if v is not None:
            what = {'confirmed': "auto v confirmed", 'typed': "typed v"}.get(span.get('kind'), "fitted v")
            lines.append(f"applied ({span.get('kind', 'adjusted')}): {what} = {self._v_text(v)}"
                         f"; covers every file from {self.a.name} to {self.b.name}")
        else:
            lines.append(f"applied: no velocity ({span.get('note', '')})")
        self.txt.set_text("\n".join(lines[-12:]))
        self.fig.canvas.draw_idle()

    def _show_draft(self):
        """The readout for the draft: the markers' status and pair lines, the box's line, and the span applied now when there is one (the last 12 lines)."""
        lines = [self.markers.status()] + self.markers.pair_lines() + [self._box_line()]
        if self._applied is not None and self._applied.get('v') is not None:
            lines.append(f"applied now ({self._applied.get('kind', 'adjusted')}): v = {self._v_text(self._applied['v'])}")
        self.txt.set_text("\n".join(lines[-12:]))
        self.fig.canvas.draw_idle()

    def _changed(self):
        """A caller's own on_change, when given; the Drift's subscribers were already told by the setter that changed it."""
        if self._on_change is not None:
            try:
                self._on_change()
            except Exception as exc:
                print(f"drift window: the opener could not be refreshed ({exc}).")

    def _on_markers(self):
        """A marker placed, dragged or removed (PairMarkers.on_change): the status on the console, the readout refreshed."""
        print(self.markers.status())
        self._show_draft()

    def _on_apply(self):
        """
        'apply velocity': the box stored as this pair's span (Drift.set_manual_span),
        replacing what the pair had: 'confirmed' or 'adjusted' with the markers it
        was set from when the box is still as 'set from markers' wrote it, 'typed'
        (the box's value itself, no markers) when it was typed or edited; an empty
        box removes the pair's span.  Every subscriber of the Drift redraws.
        """
        v = self._parse_box()
        if v is False:
            print(f"apply velocity: '{self.box_v.value}' is not 'Vx, Vy' in calibrated nm/h; nothing applied.")
            self._show_draft()
            return
        if v is None:
            if self.drift.span_between(self.a.path, self.b.path) is not None:
                self.drift.clear_manual_span(self.a.path, self.b.path)
            else:
                print(f"apply velocity: the box is empty and {self.a.name} -> {self.b.name} had no velocity; nothing to remove.")
            self._applied = None
            self._box_kind, self._box_markers, self._box_text = None, None, ''
            self.txt.set_text(self.markers.status() + f"; applied: none, the files from {self.a.name} to {self.b.name} take the auto velocity")
            self.fig.canvas.draw_idle()
        else:
            edited = str(self.box_v.value).strip() != self._box_text
            if not edited and self._box_kind == 'confirmed' and self._box_markers is not None:
                span = self.drift.set_manual_span(self.a.path, self.b.path, self._box_markers, kind='confirmed')
            elif not edited and self._box_kind == 'markers' and self._box_markers is not None:
                span = self.drift.set_manual_span(self.a.path, self.b.path, self._box_markers, kind='adjusted')
            else:
                span = self.drift.set_manual_span(self.a.path, self.b.path, np.zeros((0, 6)), kind='typed', v=v)
            self._applied = span
            self._box_kind = {'confirmed': 'confirmed', 'adjusted': 'markers'}.get(span.get('kind'), 'typed')
            self._box_text = str(self.box_v.value).strip()
            self._show(span)
        self._changed()

    def _on_clear(self):
        """'clear markers': the draft emptied (PairMarkers.clear; its on_change refreshes the readout)."""
        self.markers.clear()

    def _on_typed(self, text):
        """The 'v (nm/h, cal)' box submitted (Enter or focus lost): it only fills the box (an edit makes its provenance 'typed'); the readout says what it holds, not applied."""
        if str(text).strip() != self._box_text:
            self._box_kind = 'typed'
            self._box_markers = None
        self._show_draft()

    def _on_set_from_markers(self):
        """
        'set from markers': the draft's velocity into the box, nothing
        stored: every pair seeded by an auto velocity and unmoved, the
        pair's tracked velocity itself (confirmed); otherwise the pairs'
        least-squares fit
        (fit_marker_velocity; the MATLAB readout per pair printed).  No
        complete pair, or no pair with a positive time difference: the box
        is left alone.
        """
        a, b, m = self.markers.span()
        if m is None:
            print("set from markers: no complete marker pair (the k-th marker on each topo makes a pair).")
            self._show_draft()
            return
        v_fit, pairs, note = fit_marker_velocity(m)
        for k, rec in enumerate(pairs):
            print("  " + marker_pair_text(k, rec, self.scale))
        sources = self.markers.seed_sources()
        auto_seeded = bool(sources) and all(str(x).startswith('auto') for x in sources)
        if self.markers.confirmable() and auto_seeded and self.v_auto is not None:
            self._fill_box(self.v_auto, 'confirmed', m)
            print(f"set from markers: every pair seeded and unmoved; the auto velocity {self._v_text(self.v_auto)} is in the box (confirmed), not applied.")
        elif v_fit is not None:
            self._fill_box(v_fit, 'markers', m)
            print(f"set from markers: fitted v = {self._v_text(v_fit)} from {len(m)} pair(s) is in the box, not applied.")
        else:
            print(f"set from markers: no velocity from the markers ({note}); the box is unchanged.")
        self._show_draft()


if __name__ == '__main__':
    # The drift build of an experiment directory (every consecutive topo pair of every segment
    # tracked, the computed store written) and the drift result of one spectra file, loaded the
    # way the Browser loads it.
    directory = r'experiment_dir'
    name = 'file.sm4'

    # Drift correction settings (see DRIFT_PARAMS for the full list).
    drift_params = {
        'sg_window': 5,            # Savitzky-Golay window (points) before peak finding
        'sg_poly': 2,
        'prominence_frac': 0.20,   # peak prominence threshold, fraction of the profile range
    }

    files = sm4file.list_sm4_files(directory)
    kinds, head_geoms = {}, {}
    for p in files:
        types, geom, line_types = sm4file.sm4_header_info(p)
        kinds[p] = sm4file.kind_from_types(types, line_types)
        if kinds[p] == 'topo' and geom is not None:
            head_geoms[p] = geom
    loaded = {}

    def load(p):
        f = sm4file.Sm4File(p)
        kinds[p] = f.kind
        if f.self_check_text:
            print(f.self_check_text)
        loaded[p] = f
        return f

    params = dict(DRIFT_PARAMS)
    params.update(drift_params)
    correction = Drift(directory, files, kinds, head_geoms, params, load)
    correction.build()
    path = next(p for p in files if os.path.basename(p) == name)
    f = load(path)
    entry = correction.result(f, loaded)
    if entry is not None:
        xy = entry['xy_corr'] * 1e9
        print(f"\ncorrected positions of {f.name} (nm, absolute frame):")
        for k in range(len(xy)):
            if np.all(np.isfinite(xy[k])):
                print(f"  {k:5d}  x {xy[k, 0]:12.3f}  y {xy[k, 1]:12.3f}")
