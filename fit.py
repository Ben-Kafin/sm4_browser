# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 16:51:00 2026

@author: Benjamin Kafin

Single-point dI/dV fit (Fit) for RHK .sm4 files.

Pipeline
--------
1. Load the dI/dV and I(V) spectroscopy pages (STS: the dI/dV channel by its
   label, 'LIA Current', and the 'Current' page; any number of repeated
   sweeps, never-acquired sweeps left out), in physical units as the loader
   gives them (sm4file.load: every page's stored values x RHK_Zscale +
   RHK_Zoffset, the current and the lock-in output in A), through
   calibration and normalization (same convention as the Line tool).
2. Per sweep: the recorded I(V) through the two filters once, each on or
   off on its own (every filter off unless given), in the order Hampel
   (spikes), Savitzky-Golay (filters.filter_lia; the LIA column is never
   filtered); the calibration of the LIA column against that current
   (calibration.calibrate, the Line tool's calibration: whichever
   calibration file is in place, calibration_zeroref, calibration_gainoffset,
   or calibration_stretch renamed to calibration.py; each returns per sweep
   a, c, and rms, which the console report prints); for quantity 'ratio' the
   division by the broadened I/V of the same filtered current
   (feenstra.divide_by_iv, before the 0 V subtraction and the broadening);
   then unit-area normalization.
3. Average all sweeps: the plain mean of every sweep (a curve sent from the
   Browser's averages: of the pooled sweeps of its raw sets), and
   every rebuild averages the same way.  The result is the "data" curve that
   is fitted; the same pipeline with every filter off is kept as the
   "unfiltered" curve.
   The filter settings are recorded per fit: every fit in the history carries
   the Hampel and Savitzky-Golay settings it was fitted against
   together with its data curve, anchor target, and background ceiling (its
   "data context"), and showing a fit restores that context.  The main figure
   and the edit window both have the Savitzky-Golay and Hampel controls
   (FilterControls, always in step) and the Feenstra settings ('0Vsub',
   'broadened', and the width box, FeenstraSettings, in step too) for the
   left panel; their labels are blue while a setting differs from the shown
   fit's (set but not yet run) and red when the shown fit was run with it,
   the rule of the 'WKB background' and 'Feenstra (refit)' toggles; a
   Refit from the edit window rebuilds the data context with the control
   settings when they differ from the shown fit's, by rerunning the pipeline
   on the raw sets (self.raw_sets: one or more energy axes, each with the raw
   LIA and I(V) columns of the sweeps that take part, in physical units, with
   their units and the modulation amplitude; a file opened directly or a Line tool send is one set, an average
   of several files from the Browser one set per file, each
   processed on its own axis and then taken on the common energy range: its
   points when they coincide with the curve's axis, otherwise, as the send's
   'resample' setting says, binned onto the axis (half an axis step either
   side of each point) or interpolated at the axis points, every point of
   such a set then counting in the average with axis step / set step, the
   set's measured points per axis step; without the setting a set off the
   axis is refused).  A
   fit history that holds no raw sets (sends made before this rule) cannot be
   rebuilt: a refit with changed settings is refused on the console.  The
   noise level and the candidate list of the initial sweep are not
   recomputed.
4. Background: the energy-dependent tunneling-barrier transmission factor
   (see "Background model" below).  Its largest member that stays at or below
   the data everywhere (the envelope E) is the ceiling: the fitted background
   must satisfy B <= E at every point (hence B <= data) and B > 0.  Between
   those limits it is solved jointly with the features and peaks by least
   squares.  The seed is the unconstrained least-squares member of the family
   with its amplitude at half the largest value that keeps it under E (a
   feasible, mid-range start).
   Background mode 'none' (select_model(background='none'), the 'background'
   entry of the Browser's and the Line tool's fit_settings; the 'WKB
   background' toggle of the fit and edit windows sets the mode of the next
   Refit): there is no background at all, the model is the plain sum of the
   features and peaks in data units, with no ceiling, no positivity
   constraint, and no offset.  The mode is part of every fit's data context
   (history, cache, saved fits); the figure hides the background panel for a
   fit without one.
5. Noise level ('differences'): successive-difference sigma of the unfiltered
   curve (1.4826 * MAD(diff y) / sqrt 2); no filter smooths the fitted
   dI/dV, so there is no filter gain (sg_gain is 1 and kept for the
   record).  The candidate threshold is noise_z times
   this level: the prominence threshold for candidate peak positions, local
   maxima of (data - envelope) with the background, of the data without.
6. Fit: the barrier transmission B is a factor of the dI/dV, so
     data = B * [1 + sum(features) + sum(peaks)]        (fit on data/B - 1)
   with B fitted jointly with the features and peaks; without a background
     data = sum(features) + sum(peaks)                  (fit on the data)
   with the amplitudes in data units.
   Constraints: B <= E (envelope ceiling, so B <= data) and B > 0 at every
   point (with a background), every amplitude >= 0, every peak half width >=
   one energy step, the ratio of the two half widths of every peak <=
   max_width_ratio, and the anchor equality on the data-domain model:
   for a window containing 0 V the integral from 0 V to the setpoint bias
   (analytic grid) equals the setpoint current; for a window without 0 V the
   integral over the measured window equals the measured I(V_bias) - I(V_ref).
   Quantity 'ratio' (divide_by_iv of __init__, quantity='ratio' of from_curve;
   the Browser's and the Line tool's 'Feenstra' sends): the fitted
   curve is Feenstra's normalized conductance (dI/dV)/(I/V), every sweep
   divided after its calibration by its own exponentially broadened I/V
   (STS.divide_by_iv, width iv_broadening_V), then normalized.  Its
   integral is not the current, so there is no anchor (anchor_mode 'none');
   the background is set by the least squares under the ceiling and
   positivity alone.  Everything else (background family, constraints, noise
   level, selection, edit window, refits, cache) is the same; the cache and
   the saved fits record the quantity and the broadening.  The quantity is
   part of every fit's data context: the 'Feenstra (refit)' toggle of the fit and
   edit windows (always in step) is the quantity of the next Refit, the
   curve rebuilt from the raw sets as dI/dV or as the ratio (blue while it
   differs from the shown fit's quantity, red once a fit was run with it;
   a history without raw sets cannot change quantity), so one history may
   hold fits of both quantities; the history file is still chosen by the
   sender's label.  The noise level and candidate list are not recomputed
   for a change of quantity, as for a change of background mode.
   Holds: in the edit window any parameter of any peak (center, height,
   per-side HWHM, and Lorentzian share) and of the background (amp, phi,
   kappa) can be held at its shown value for the next Refit; a held
   parameter is fixed and leaves the free list (BIC counts the free ones);
   a side with only its HWHM or only its share held keeps sigma and gamma
   free under an equality constraint on that quantity.  Every other
   constraint still applies to the held values.  Holds are kept per peak
   ('hold') and per fit ('bg_hold') through refits and the cache.
   Every peak is a split Voigt ('asym_voigt'): an independent (sigma, gamma)
   pair on each side of the center, Gaussian sigma convolved with Lorentzian
   gamma; gamma = 0 is exactly a Gaussian, sigma = 0 exactly a Lorentzian, so
   the fit chooses the character of each side, and equal sides make a
   symmetric peak.  Each side is reported with its Lorentzian share
   L/(G+L) = gamma / (gamma + Gaussian HWHM) and a designation (Gaussian
   <= 0.1, Lorentzian >= 0.9, Voigt between).
   The fit is SLSQP (scipy's finite-difference gradients); the model is
   evaluated once per parameter vector and shared by the objective and the
   constraints.  After every fit the peaks are renumbered by center energy
   (most negative first); the permutation from the optimizer's order is kept
   as 'peak_order'.
7. Enumeration: every subset of known features x every peak count 0..M (the k
   most prominent candidates).  All fits are submitted at once and can be
   spread over CPU cores with loky (n_jobs; pip install loky); one console
   line per finished fit.  Selection by minimum BIC on the data-domain
   residual.  The selected model is fit 1.
8. Fit history and cache: every accepted fit (fit 1, then each accepted Refit
   from the edit window) is kept, numbered in order of calculation; nothing is
   discarded.  The history and everything a refit needs are written to
   <folder of the data file>/<date_time>_fits.pkl after every accepted fit and
   after every Revert/Forward, where <date_time> is the trailing run of
   underscore-separated numbers of the .sm4 name (e.g. 2023_12_29_11_33_34_590),
   plus the marker label for curves sent from the Line tool; a label whose
   path part is a directory ('<folder> | avg3', the Browser's
   averages) gives <folder>/avg3_fits.pkl.  Every cache records the loader
   format it was written from (sm4file.LOADER_FORMAT); one written from
   another loader (the stored values before the physical units: no
   'loader_format') is out of date and is never read or written: a new
   history goes beside it with the next number, <date_time>_fits_2.pkl,
   _3, and so on (sm4file.cache_family), and the newest up-to-date one of
   the family is the one read and rewritten (cache_path).
   Fit.from_cache(path) opens the figure from that file without
   process() or select_model(); has_cache(ifile) tells whether an
   up-to-date cache exists for a data file or send label.  Cache format 3
   holds the raw columns, the filter stage, and the per-fit data contexts;
   a format-2 cache (one smoothing setting per analyzer) predates the
   loader format and is out of date.
9. Figure: data (left panel, with the Savitzky-Golay / Hampel
   controls: the curve a Refit would fit with the current settings; the shown
   fit's own curve is overlaid when it differs) and, beside it in the same
   slot, the stack of the raw sets' curves at the control settings (one
   colour per set, offset vertically, the average at the bottom; the two
   share the slot so that one curve's range in the stack has the same
   height-to-width ratio as the data panel's curve; 'substack' switches
   between one mean per set and every sweep; no stack without raw sets);
   fit ('Fit N of M'); background-divided
   data with the components, legends in order of center energy (most
   negative at the top), shown only for a fit with the WKB background (the
   window is as wide as the panels shown).  The 'WKB background' toggle (fit
   window and edit window, always in step) is the mode of the next Refit:
   blue while it differs from the shown fit's mode, red once a fit was run
   with it.
   Buttons: 'Revert fit' (back to the fit displayed before the current one;
   repeated presses walk back to fit 1), 'Forward fit' (the reverse; a new
   accepted refit clears the forward chain), 'Edit peaks' (second window) and
   'Save fit' (the displayed fit as <date_time>_fit<N>.json: background
   equation and parameters, every feature and peak with center, amplitude,
   per-side sigma, gamma, half width, Lorentzian share, and designation,
   asymmetry ratio, FWHM; and <date_time>_fit<N>.csv: V, data, background
   (when the fit has one), every component, model; both next to the data
   file).
   In the edit window every fitted peak can be selected (drawn dashed, thick,
   and orange; the others solid), moved (click elsewhere places the selected
   peak's center and height), reshaped (drag from within two energy steps of
   its center: the x displacement sets that side's half width, the y
   displacement its Gaussian/Lorentzian character, up = more Gaussian, down =
   more Lorentzian, half the peak height spans the range), removed
   (right-click) or added ('Add peak' arms the next press-drag), and 'Refit'
   runs the joint fit from the edited peaks (free unless held, same
   constraints); an accepted refit is appended to the history as the next
   fit, a rejected one prints why.  Right of the plot, as tall as it, the
   parameter panel of the selected curve: one row per parameter, a hold
   toggle on the left (the word 'hold' once above the first), the name,
   and an editable box whose submitted value sets the parameter (the
   provisional curve follows; a value outside the parameter's own range is
   refused and the box reset).  'edit background', shown while the
   displayed fit has the WKB background, selects the background for the
   panel (amp, phi, kappa); clicking a peak selects that peak again.
   Held peaks are marked in the legend.

Background model ('wkb')
------------------------
The one-dimensional WKB transmission through a trapezoidal barrier, evaluated
at the probed energy (the edge of the bias window), with the tip-sample
distance absorbed into a dimensionless exponent:

    B(V) = amp * exp( kappa * (1 - sqrt(1 - |V| / (2 phi))) )

    phi   : average barrier height (eV)
    kappa : 2 z sqrt(2 m_e phi) / hbar, the zero-bias decay exponent (the
            tip-sample distance z never appears on its own)
    amp   : the background at 0 V (the shape is exactly 1 there)

Minimum at 0 V (a cusp, |V|), rising on both sides.  phi is bounded below by
half the largest |V| of the window (plus 0.05 eV) so that the root is real.
"""

if __package__ in (None, ''):
    # Run as a file (Spyder F5): make the package importable from wherever its folder sits.
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import itertools
import json
import os
import pickle
import re
import time
from concurrent.futures import as_completed
import numpy as np
from sm4_Browser import qtplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize
# erf is referenced through the module (not imported as a bare name) so that the
# worker processes can unpickle the __main__-defined line shapes when run in Spyder.
from scipy import special as _special
from sm4_Browser import sts, calibration, normalization, feenstra, widgets, averaging, filters, sm4file


# ---------------------------------------------------------------------- #
#  Feature line shapes (all non-negative for amp >= 0)                    #
# ---------------------------------------------------------------------- #

_SQRT_2LN2 = np.sqrt(2.0 * np.log(2.0))          # Gaussian HWHM = sigma * sqrt(2 ln 2)


def _voigt_hwhm(sigma, gamma):
    """
    Half width at half maximum of a Voigt profile with Gaussian sigma and
    Lorentzian HWHM gamma (Olivero-Longbothum, accurate to 0.02 %; the formula
    is homogeneous, so it holds for half widths as it does for full widths).
    """
    fG = sigma * _SQRT_2LN2
    return 0.5346 * gamma + np.sqrt(0.2166 * gamma ** 2 + fG ** 2)


def _voigt_mix(sigma, gamma):
    """
    Lorentzian share of a Voigt side, gamma / (gamma + Gaussian HWHM):
    0 = pure Gaussian, 1 = pure Lorentzian, 0.5 = equal half widths.
    """
    tot = gamma + sigma * _SQRT_2LN2
    return float(gamma / tot) if tot > 0 else 0.5


def _voigt_widths_from(hwhm, mix=0.5):
    """
    (sigma, gamma) of a Voigt side with the given half width at half maximum
    and Lorentzian share 'mix' (see _voigt_mix).  With h_L = mix * t and
    h_G = (1 - mix) * t the Olivero-Longbothum width equals hwhm for
    t = hwhm / [0.5346 mix + sqrt(0.2166 mix^2 + (1 - mix)^2)].
    """
    mix = min(max(float(mix), 0.0), 1.0)
    t = float(hwhm) / (0.5346 * mix + np.sqrt(0.2166 * mix ** 2 + (1.0 - mix) ** 2))
    return (1.0 - mix) * t / _SQRT_2LN2, mix * t


def _voigt_seed_widths(hwhm):
    """(sigma, gamma) with equal Gaussian and Lorentzian half widths whose Voigt HWHM is hwhm."""
    return _voigt_widths_from(hwhm, 0.5)


def _voigt_designation(mix):
    """Branch designation from the Lorentzian share."""
    if mix <= 0.1:
        return 'Gaussian'
    if mix >= 0.9:
        return 'Lorentzian'
    return 'Voigt'


def _voigt_shape(x, sigma, gamma):
    """
    Peak-normalized Voigt profile (1 at x = 0): Gaussian (sigma) convolved with
    Lorentzian (HWHM gamma).  gamma = 0 is exactly a Gaussian, sigma = 0 exactly
    a Lorentzian.  Non-finite (both widths zero) is returned as NaN so that the
    fit rejects it.
    """
    peak = _special.voigt_profile(0.0, sigma, gamma)
    if not np.isfinite(peak) or peak <= 0:
        return np.full_like(np.asarray(x, dtype=np.float64), np.nan)
    return _special.voigt_profile(x, sigma, gamma) / peak


def _asym_voigt(V, amp, center, sigma_left, gamma_left, sigma_right, gamma_right):
    """
    Split Voigt: an independent (sigma, gamma) pair on each side of the
    maximum, so either side can be Gaussian, Lorentzian, or in between on its
    own; equal sides make a symmetric peak.  Both halves equal amp at
    V = center, so the shape is continuous.
    """
    x = V - center
    return amp * np.where(x < 0, _voigt_shape(x, sigma_left, gamma_left),
                          _voigt_shape(x, sigma_right, gamma_right))


def _erf_step(V, amp, center, width, direction=1.0):
    """
    Rises from 0 to amp across 'center' (direction=+1 rises with energy,
    direction=-1 rises toward lower energy).  Never negative for amp >= 0.
    """
    return amp * 0.5 * (1.0 + direction * _special.erf((V - center) / width))


# 'widths' lists the parameters after (amp, center).  'sides' groups the width
# parameters of each half of a peak; each half's Voigt half width must be >=
# one energy step, and the ratio of the two halves' half widths is capped by
# max_width_ratio.  'equation' documents the shape for the saved fit.
FEATURE_REGISTRY = {
    'asym_voigt': {'func': _asym_voigt, 'widths': ['sigma_left', 'gamma_left', 'sigma_right', 'gamma_right'],
                   'sides': [('sigma_left', 'gamma_left'), ('sigma_right', 'gamma_right')],
                   'equation': ("P(V) = amp * Voigt(V - center; sigma_s, gamma_s) / Voigt(0; sigma_s, gamma_s), "
                                "s = left for V < center and right for V >= center; "
                                "Voigt(x; sigma, gamma) = Gaussian(sigma) convolved with Lorentzian(HWHM gamma) "
                                "(scipy.special.voigt_profile)")},
    'erf_step':   {'func': _erf_step,   'widths': ['width'],
                   'sides': [],
                   'equation': "F(V) = amp * 0.5 * (1 + direction * erf((V - center) / width))"},
}

# The generic peak shape: every peak is a split Voigt (a symmetric peak has equal sides).
PEAK_SHAPE = 'asym_voigt'

# The one noise definition and the one decomposition (recorded in the saved fits).
NOISE_METHOD = 'differences'
DECOMP_MODE = 'multiplicative'


# ---------------------------------------------------------------------- #
#  Background: trapezoidal-barrier WKB transmission, amp * shape(V, phi, kappa)
# ---------------------------------------------------------------------- #

def _shape_wkb(V, phi, kappa):
    """exp(kappa * (1 - sqrt(1 - |V| / (2 phi)))); NaN where phi <= |V| / 2."""
    q = np.sqrt(1.0 - np.abs(V) / (2.0 * phi))
    return np.exp(kappa * (1.0 - q))


BACKGROUND = {
    'name': 'wkb',
    'shape': _shape_wkb,
    'names': ['phi_eV', 'kappa'],
    'bounds': [(None, 20.0), (0.1, 60.0)],      # phi lower bound set from the energy range (_bg_bounds)
    'grid': [np.linspace(1.5, 8.0, 40), np.linspace(1.0, 40.0, 60)],
    'equation': ("B(V) = amp * exp(kappa * (1 - sqrt(1 - |V| / (2 phi_eV)))); "
                 "kappa = 2 z sqrt(2 m_e phi) / hbar (dimensionless; trapezoidal-barrier WKB "
                 "transmission at the probed energy, tip-sample distance absorbed)"),
}


def _noise_sigma(y):
    """
    von Neumann successive-difference noise estimate:
    sigma = 1.4826 * MAD(diff y) / sqrt(2), consistent for Gaussian noise.
    """
    d = np.diff(np.asarray(y, dtype=np.float64))
    mad = np.median(np.abs(d - np.median(d)))
    return float(1.4826 * mad / np.sqrt(2.0))


def _run_fit(state, feature_subset, peaks, seeds):
    """
    Worker entry point: run one constrained fit on a lightweight analyzer
    state.  Returns (fit dict or None, rejection reason or None, seconds).
    """
    t0 = time.perf_counter()
    fit = state._fit_model(feature_subset, peaks, seeds)
    return fit, (None if fit is not None else state._last_reject), time.perf_counter() - t0


class Fit:
    def __init__(self, ifile, hampel_filter_params=None, sgolay_filter_params=None,
                 divide_by_iv=False, iv_broadening_V=0.0, iv_zero_sub=False):
        """
        Initialize the single-point dI/dV fit.
          ifile                : Path to an SM4 single-point spectroscopy file.
          hampel_filter_params : dict for the Hampel filter (or None to skip).
          sgolay_filter_params : dict for the Savitzky-Golay filter (or None to skip).
          divide_by_iv         : fit the normalized conductance (dI/dV)/(I/V) instead of
                                 dI/dV (quantity 'ratio': every sweep divided after its
                                 calibration by its own broadened I/V, see
                                 STS.divide_by_iv; no current anchor, see _set_anchor).
          iv_broadening_V      : broadening width (V) of I/V for divide_by_iv; 0 = plain ratio.
          iv_zero_sub          : subtract each sweep's current at 0 V before I/V is formed
                                 (STS.iv_corrected; a vertical shift of the current only).
        """
        self.ifile = ifile
        self.hampel_filter_params = hampel_filter_params
        self.sgolay_filter_params = sgolay_filter_params
        self.resample = None               # files opened directly: one raw set on the curve's own axis
        self.background = 'wkb'            # the model's background mode until select_model sets it
        self._sweep_done = False           # True once select_model (or a cache) has given a fit history
        self.filter_stage = 'average'      # files opened directly: one raw set, the pipeline of _process_raw_sets
        self.quantity = 'ratio' if divide_by_iv else 'didv'
        self.iv_broadening = float(iv_broadening_V)
        self.iv_zero_sub = bool(iv_zero_sub)

        self.extract_data()

    # ------------------------------------------------------------------ #
    #  Loading                                                            #
    # ------------------------------------------------------------------ #

    def extract_data(self):
        """
        The LIA (dI/dV) and I(V) sweeps in physical units (never-acquired
        sweeps left out), the energy axis, the setpoint, the pages' units, and
        the modulation amplitude (RHK_CH1Drive_Amplitude), through
        STS.from_file on the file's ramp-spectroscopy pages.
        """
        spectra = sts.STS.from_file(self.ifile)
        if spectra is None or spectra.lia is None:
            raise ValueError("No 'LIA Current' ramp-spectroscopy page found in this SM4 file.")
        if spectra.iv is None:
            if spectra.iv_dropped is not None:
                raise ValueError(f"'LIA Current' shape {spectra.lia.shape} does not match "
                                 f"'Current' shape {spectra.iv_dropped}.")
            raise ValueError("No 'Current' ramp-spectroscopy page found in this SM4 file.")

        # The loader's data (physical units) go through calibration and normalization (the Line
        # tool's convention); only the acquired sweeps take part.
        cols = np.flatnonzero(spectra.acquired)
        if cols.size == 0:
            raise ValueError("No acquired sweep in this SM4 file.")
        self.lia = spectra.lia[:, cols]
        self.iv = spectra.iv[:, cols]
        self.units_lia, self.units_iv = spectra.units_lia, spectra.units_iv
        self.vmod = spectra.vmod                             # RHK_CH1Drive_Amplitude (V), None when not recorded
        self.n_sweeps = self.lia.shape[1]
        self.energy = spectra.energy.copy()

        # Bias / current setpoint from metadata (amps).
        self.V_bias = spectra.V_bias
        self.I_setpoint = spectra.I_setpoint

    # ------------------------------------------------------------------ #
    #  Axes, anchor, noise                                                #
    # ------------------------------------------------------------------ #

    def _setup_axes(self, V_sorted):
        """
        Ascending energy axis, reference indices, and the analytic grid from 0 V
        to V_bias for the setpoint anchor (independent of the data window, at
        least as fine as the data step).
        """
        self.V = np.asarray(V_sorted, dtype=np.float64)
        self.i_ref = int(np.argmin(np.abs(self.V)))          # data point nearest 0 V
        self.i_bias = int(np.argmin(np.abs(self.V - self.V_bias)))
        self.window_has_zero = bool(self.V[0] <= 0.0 <= self.V[-1])
        dV = float(np.mean(np.abs(np.diff(self.V))))
        n_anchor = max(2001, int(np.ceil(abs(self.V_bias) / dV)) * 10 + 1)
        self.V_anchor = np.linspace(0.0, self.V_bias, n_anchor)

    def _set_anchor(self, areas_A, delta_I_window_A):
        """
        Anchor for the model integral, in the curve's arbitrary units.
          window contains 0 V : 'setpoint' anchor, integral of the model from 0 V to
                                V_bias (analytic grid) = I_setpoint * mean(1/areas_A)
          window without 0 V  : 'window' anchor, integral of the model over the
                                measured window from V_ref to V_bias =
                                mean(delta_I_window_A / areas_A), where
                                delta_I_window_A = I(V_bias) - I(V_ref) per sweep (A).
                                Nothing is extrapolated; the setpoint at 0 V is not
                                observable for such a window.
        """
        areas_A = np.atleast_1d(np.asarray(areas_A, dtype=np.float64))
        if getattr(self, 'quantity', 'didv') == 'ratio':
            # (dI/dV)/(I/V): its integral is not the current, so there is no anchor;
            # the background is set by the least squares under the ceiling and positivity.
            self.anchor_mode = 'none'
            self.target = None
            return
        if self.window_has_zero:
            self.anchor_mode = 'setpoint'
            self.target = self.I_setpoint * np.mean(1.0 / areas_A)
        else:
            if delta_I_window_A is None:
                raise ValueError("Window does not contain 0 V: the measured I(V_bias) - I(V_ref) "
                                 "per sweep (delta_I_window_A) is required for the window anchor.")
            dI = np.atleast_1d(np.asarray(delta_I_window_A, dtype=np.float64))
            if dI.shape != areas_A.shape:
                raise ValueError("delta_I_window_A must have one value per area.")
            self.anchor_mode = 'window'
            self.target = float(np.mean(dI / areas_A))

    def _qlabel(self):
        """Name of the fitted quantity for labels: dI/dV or (dI/dV)/(I/V)."""
        return "(dI/dV)/(I/V)" if getattr(self, 'quantity', 'didv') == 'ratio' else "dI/dV"

    def _anchor_text(self):
        if self.anchor_mode == 'none':
            return "Anchor: none ((dI/dV)/(I/V) has no current anchor)"
        return f"Anchor: {self.anchor_mode} (target in arb. units = {self.target:.6g})"

    def _set_noise(self):
        """
        Noise level (see module docstring): successive-difference sigma of the
        unfiltered curve.  The filters act on the current of the Feenstra
        division only, never on the fitted dI/dV, so there is no filter gain
        (sg_gain stays 1, kept in the record).
        """
        self.sigma_diff = _noise_sigma(self.data_unfiltered)
        self.sg_gain = 1.0
        self.noise_level = self.sigma_diff * self.sg_gain

    def _print_noise(self):
        print(f"Noise level ({NOISE_METHOD}): {self.noise_level:.4g} arb. units "
              f"(sigma of the unfiltered curve; no filter acts on the fitted dI/dV)")

    # ------------------------------------------------------------------ #
    #  Data context: the curve a fit is made against and what goes with it #
    # ------------------------------------------------------------------ #

    # Everything that depends on the smoothing settings, the background mode and
    # the quantity (dI/dV or the Feenstra ratio, with the I/V broadening), stored
    # with every fit.
    _CONTEXT_KEYS = ('hampel_filter_params', 'sgolay_filter_params', 'background',
                     'quantity', 'iv_broadening', 'iv_zero_sub',
                     'data', 'target', 'anchor_mode', 'areas_A', 'delta_I_window_A',
                     'envelope', 'bg_seed', 'env_bg', 'env_residual')
    QUANTITIES = ('didv', 'ratio')
    BACKGROUND_MODES = ('wkb', 'none')

    @staticmethod
    def _same_params(a, b):
        """True when two filter settings (dict or None) are the same."""
        if a is None or b is None:
            return a is None and b is None
        return all(float(a[k]) == float(b[k]) for k in a) and set(a) == set(b)

    ENERGY_TOL = 1e-6      # V: a raw set's energies must coincide with the common axis to this

    def _raw_set_columns(self, hampel, sgolay, quantity=None):
        """
        The one pipeline on self.raw_sets for these filter settings: per set,
        the set's I(V) through the filters once (filters.filter_lia),
        calibration of its LIA columns against that current on its own
        axis (calibration.calibrate), the division by the broadened I/V of
        the same current for quantity 'ratio' (the analyzer's quantity, or
        `quantity` when given; feenstra.divide_by_iv with the analyzer's
        iv_broadening and iv_zero_sub, told not to filter again), unit-area normalization
        (normalization.normalize), then the rows on the common
        axis self.V: the set's own rows when its points coincide with the
        axis (ENERGY_TOL), otherwise brought onto the axis as self.resample
        says ('bin': the mean of its points within half an axis step either
        side of each axis point; 'interp': linear interpolation at the axis
        points; None: refused), the raw I(V) the same way.
        Returns (cols, areas_A, delta_I_window_A): one array (n_points,
        n_sweeps) per set, and per pooled sweep the normalization area and
        I(V_bias) - I(V_ref) in amps.  The per-set point weights of the
        average (axis step / set step for a binned or interpolated set, 1 for
        a set on the axis; see STS.average) are left in self._set_weights,
        one (n_points,) array per set.
        """
        tol = self.ENERGY_TOL
        method = getattr(self, 'resample', None)
        if quantity is None:
            quantity = getattr(self, 'quantity', 'didv')
        cols, areas, dI, weights = [], [], [], []
        saved = self.energy
        try:
            for rs in self.raw_sets:
                self.energy = rs['energy']
                iv_f = filters.filter_lia(rs['iv'], hampel, sgolay)   # the set's current through the filters once
                cal, _ = calibration.calibrate(self.energy, rs['lia'], iv_f, rs.get('vmod'))
                if quantity == 'ratio':
                    cal = feenstra.divide_by_iv(self.energy, cal, iv_f, self.iv_broadening,
                                               bool(getattr(self, 'iv_zero_sub', False)), None, None)
                norm, norm_areas = normalization.normalize(self.energy, cal)
                order = np.argsort(self.energy)
                e = self.energy[order]
                norm = norm[order, :]
                iv_raw = np.asarray(rs['iv'], dtype=np.float64)[order, :]
                m = (e >= self.V[0] - tol) & (e <= self.V[-1] + tol)
                w = np.ones(len(self.V))
                if int(m.sum()) == len(self.V) and float(np.max(np.abs(e[m] - self.V))) <= tol:
                    on_axis = lambda c: c[m, :]
                elif method == 'bin':
                    step = float(np.median(np.diff(self.V)))
                    bins = [np.nonzero((e >= g - 0.5 * step - tol) & (e < g + 0.5 * step - tol))[0] for g in self.V]
                    if min(len(rows) for rows in bins) == 0:
                        raise ValueError("a raw set leaves a bin of the curve's axis empty.")
                    on_axis = lambda c, bins=bins: np.vstack([c[rows, :].mean(axis=0) for rows in bins])
                    w = np.full(len(self.V), step / float(np.median(np.diff(e))))
                elif method == 'interp':
                    on_axis = lambda c, e=e: np.column_stack([np.interp(self.V, e, c[:, j]) for j in range(c.shape[1])])
                    w = np.full(len(self.V), float(np.median(np.diff(self.V))) / float(np.median(np.diff(e))))
                else:
                    raise ValueError("a raw set's energies do not coincide with the curve's axis on its range.")
                cols.append(on_axis(norm))
                weights.append(w)
                areas.extend(np.array(norm_areas))
                ivs = on_axis(iv_raw)
                i_ref = int(np.argmin(np.abs(self.V)))
                i_bias = int(np.argmin(np.abs(self.V - self.V_bias)))
                dI.extend(ivs[i_bias, :] - ivs[i_ref, :])
        finally:
            self.energy = saved
        self._set_weights = weights
        return cols, np.array(areas, dtype=np.float64), np.array(dI, dtype=np.float64)

    def _process_raw_sets(self, hampel, sgolay, quantity=None):
        """
        The pipeline of _raw_set_columns followed by the average of
        STS.average (the per-set point weights of _raw_set_columns).
        Returns (data, areas_A, delta_I_window_A).
        """
        cols, areas, dI = self._raw_set_columns(hampel, sgolay, quantity)
        return averaging.Averages.average(cols, self._set_weights)[0], areas, dI

    def _build_data(self, hampel, sgolay, quantity=None):
        """
        The curve a fit with these settings is made against, ascending energy
        order: the pipeline on the raw sets (see _process_raw_sets) for the
        analyzer's quantity or `quantity` when given.  Without
        raw sets (a fit history written before the raw sets existed) only the
        stored settings (and quantity) can be served; other settings are refused.
        Returns (data, areas_A, delta_I_window_A).
        """
        if getattr(self, 'raw_sets', None):
            return self._process_raw_sets(hampel, sgolay, quantity)
        if (self._same_params(hampel, self.hampel_filter_params)
                and self._same_params(sgolay, self.sgolay_filter_params)
                and (quantity is None or quantity == getattr(self, 'quantity', 'didv'))):
            return self.data, getattr(self, 'areas_A', None), getattr(self, 'delta_I_window_A', None)
        raise ValueError("this fit history holds no raw columns, so the curve cannot be rebuilt with "
                         "other filter settings or quantity; only the stored settings can be fitted.")

    def _rebuild(self, hampel, sgolay, background=None, quantity=None):
        """
        Make these filter settings (and background mode and quantity, each
        unchanged when None) the current data context: data, anchor (none
        for the ratio), and (once a sweep has run) the background ceiling,
        seed, and residual of the mode.
        """
        self.hampel_filter_params = None if hampel is None else dict(hampel)
        self.sgolay_filter_params = None if sgolay is None else dict(sgolay)
        if background is not None:
            self.background = self._background_mode(background)
        if quantity is not None:
            self.quantity = self._quantity_mode(quantity)
        data, areas_A, delta_I = self._build_data(hampel, sgolay)
        self.data = data
        if areas_A is not None:
            self.areas_A = np.atleast_1d(np.asarray(areas_A, dtype=np.float64))
            self.delta_I_window_A = (None if delta_I is None
                                     else np.atleast_1d(np.asarray(delta_I, dtype=np.float64)))
            self._set_anchor(self.areas_A, self.delta_I_window_A)
        if getattr(self, '_sweep_done', False):
            self._set_background_context()

    @classmethod
    def _background_mode(cls, mode):
        """'wkb' or 'none' (anything else is an error)."""
        if mode not in cls.BACKGROUND_MODES:
            raise ValueError(f"background must be one of {cls.BACKGROUND_MODES} (got {mode!r}).")
        return mode

    @classmethod
    def _quantity_mode(cls, quantity):
        """'didv' or 'ratio' (anything else is an error)."""
        if quantity not in cls.QUANTITIES:
            raise ValueError(f"quantity must be one of {cls.QUANTITIES} (got {quantity!r}).")
        return quantity

    @staticmethod
    def _qname(quantity):
        """Name of a quantity for labels: dI/dV or (dI/dV)/(I/V)."""
        return "(dI/dV)/(I/V)" if quantity == 'ratio' else "dI/dV"

    def _feenstra_text(self):
        """The analyzer's Feenstra settings in words: the I/V the ratio divides by."""
        w = float(getattr(self, 'iv_broadening', 0.0))
        return (f"I/V broadened by {w:g} V" if w > 0 else "plain I/V") + \
            (", I(0 V) subtracted" if getattr(self, 'iv_zero_sub', False) else "")

    def _feenstra_differs(self, b, quantity):
        """
        True when the pending quantity is the ratio and the analyzer's
        Feenstra settings (iv_broadening, iv_zero_sub) differ from the shown
        fit's data context, i.e. a Refit would divide by another I/V.
        """
        if quantity != 'ratio' or b is None:
            return False
        return (float(b.get('iv_broadening', self.iv_broadening)) != float(self.iv_broadening)
                or bool(b.get('iv_zero_sub', False)) != bool(getattr(self, 'iv_zero_sub', False)))

    def set_feenstra(self, iv_broadening, iv_zero_sub):
        """
        The Feenstra settings of the window this curve was sent from changed
        (its companion's '0Vsub', 'broadened', and width boxes): the next
        Refit as the ratio divides by that I/V (iv_broadening, 0 the plain
        I/V; iv_zero_sub), and the data panel shows the curve a Refit would
        fit when the 'Feenstra (refit)' toggle is on.  The shown fit keeps its own
        settings in its context.
        """
        new = (float(iv_broadening), bool(iv_zero_sub))
        if new == (float(getattr(self, 'iv_broadening', 0.0)), bool(getattr(self, 'iv_zero_sub', False))):
            return
        self.iv_broadening, self.iv_zero_sub = new
        if self.iv_broadening > 0:
            self.iv_width_V = self.iv_broadening
        print(f"Fit ({self.ifile}): Feenstra settings now {self._feenstra_text()}; "
              "a Refit as (dI/dV)/(I/V) uses them.")
        self._sync_feenstra_widgets()
        if getattr(self, '_plot_axes', None) is not None:
            try:
                self._draw_left()
            except Exception as exc:
                print(f"Feenstra settings: {exc}")

    # --- the Feenstra settings and filter controls of the fit and edit windows (in step, set-versus-run colours) ---

    def _fs_settings(self):
        """The pending Feenstra settings as the boxes show them: {'zero_sub', 'broadened', 'broadening'}."""
        w = float(getattr(self, 'iv_broadening', 0.0))
        box = float(getattr(self, 'iv_width_V', 0.0) or 0.0)
        if box <= 0:
            box = w if w > 0 else self.FEENSTRA_WIDTH_DEFAULT_V
            self.iv_width_V = box
        return {'zero_sub': bool(getattr(self, 'iv_zero_sub', False)), 'broadened': w > 0, 'broadening': box}

    def _fs_widgets(self):
        """The FeenstraSettings rows that exist: the fit window's and the open edit window's."""
        out = [getattr(self, '_fs', None)]
        st = self._edit_open()
        if st is not None:
            out.append(st.get('fs'))
        return [w for w in out if w is not None]

    def _filter_widgets(self):
        """The FilterControls boxes that exist: the fit window's and the open edit window's."""
        out = [getattr(self, '_filter_controls', None)]
        st = self._edit_open()
        if st is not None:
            out.append(st.get('filters'))
        return [w for w in out if w is not None]

    def _sync_feenstra_widgets(self):
        """Every Feenstra settings row shows the pending settings (without firing)."""
        settings = self._fs_settings()
        for w in self._fs_widgets():
            w.sync(settings)

    def _on_fs_change(self, zero_sub=None, broadened=None, broadening=None):
        """
        A Feenstra setting changed in either window: the pending settings of
        the next Refit as the ratio ('0Vsub': iv_zero_sub; 'broadened' with
        the width box: iv_broadening, 0 for the plain I/V); both rows follow,
        the labels take the set-versus-run colours, and the data panel shows
        the curve a Refit would fit.
        """
        if zero_sub is not None:
            self.iv_zero_sub = bool(zero_sub)
        if broadening is not None:
            self.iv_width_V = float(broadening)
        if broadened is not None or broadening is not None:
            on = bool(broadened) if broadened is not None else float(getattr(self, 'iv_broadening', 0.0)) > 0
            self.iv_broadening = float(self._fs_settings()['broadening']) if on else 0.0
        print(f"Fit ({self.ifile}): Feenstra settings now {self._feenstra_text()}; a Refit as (dI/dV)/(I/V) uses them.")
        self._sync_feenstra_widgets()
        self._draw_left()
        st = self._edit_open()
        if st is not None:
            st['fig'].canvas.draw_idle()
        return True

    def _on_edit_filter_change(self):
        """The edit window's filter controls changed: the fit window's follow (without firing) and the data panel redraws."""
        st = self._edit_open()
        fc = getattr(self, '_filter_controls', None)
        if st is not None and st.get('filters') is not None and fc is not None:
            hp, sp = st['filters'].params()
            fc.set_params(hp, sp)
        self._on_filter_change()

    def _color_setting_controls(self):
        """
        The set-versus-run colours of the filter rows and the Feenstra
        settings in both windows: blue while the control differs from the
        shown fit's setting (set but not yet run), red when the shown fit was
        run with it (the rule of the 'WKB background' and 'Feenstra (refit)'
        toggles).
        """
        b = self.shown_fit()
        if b is None:
            return
        hp, sp = self._control_params()
        c_h = 'red' if self._same_params(hp, b.get('hampel_filter_params')) else 'blue'
        c_s = 'red' if self._same_params(sp, b.get('sgolay_filter_params')) else 'blue'
        for fc in self._filter_widgets():
            fc.color_labels(c_h, c_s)
        c_zero = 'red' if bool(b.get('iv_zero_sub', False)) == bool(getattr(self, 'iv_zero_sub', False)) else 'blue'
        c_broad = ('red' if float(b.get('iv_broadening', self.iv_broadening)) == float(getattr(self, 'iv_broadening', 0.0))
                   else 'blue')
        for w in self._fs_widgets():
            w.color_labels(c_zero, c_broad)

    def _set_background_context(self):
        """
        The background part of the data context for self.background:
        'wkb', the envelope ceiling, the fit seed, their curve, and the residual
        data - ceiling (candidates are its local maxima); 'none', no ceiling
        and no seed (None), the residual is the data itself.
        """
        if self.background == 'wkb':
            self.envelope = self._envelope()
            self.bg_seed = self._bg_seed()
            self.env_bg = self.envelope[2]
            self.env_residual = self.data - self.env_bg
        else:
            self.envelope = None
            self.bg_seed = None
            self.env_bg = None
            self.env_residual = self.data

    def _context(self):
        """Copy of the current data context (stored with an accepted fit)."""
        ctx = {}
        for k in self._CONTEXT_KEYS:
            v = getattr(self, k, None)
            if isinstance(v, np.ndarray):
                v = v.copy()
            elif isinstance(v, dict):
                v = dict(v)
            elif isinstance(v, tuple):
                v = tuple(x.copy() if isinstance(x, np.ndarray) else x for x in v)
            ctx[k] = v
        return ctx

    def _restore_context(self, fit):
        """
        Make a fit's data context the current one (a fit stored before the
        background mode: 'wkb'; before the per-fit quantity: the analyzer's
        quantity and broadening stay).
        """
        for k in self._CONTEXT_KEYS:
            if k in ('quantity', 'iv_broadening', 'iv_zero_sub') and fit.get(k) is None:
                continue
            setattr(self, k, fit.get(k))
        if self.background not in self.BACKGROUND_MODES:
            self.background = 'wkb'

    @classmethod
    def from_curve(cls, V, data, data_unfiltered, V_bias, I_setpoint, areas_A, label,
                   hampel_filter_params=None, sgolay_filter_params=None,
                   delta_I_window_A=None, lia_raw=None, iv_raw=None,
                   units_lia=None, units_iv=None, vmod=None,
                   energy_raw=None, filter_stage=None,
                   quantity='didv', iv_broadening=0.0, raw_sets=None,
                   resample=None, iv_zero_sub=False):
        """
        Build an analyzer from an already processed curve (calibrated, unit-area
        normalized, averaged, and smoothed elsewhere, e.g. sent from the Line
        tool), skipping file loading and process().

          V                : energy axis (V), any order; sorted ascending here
          data             : the smoothed curve to fit (arbitrary unit-area units), same order as V
          data_unfiltered  : the same curve built with every filter off
          V_bias           : setpoint bias (V)
          I_setpoint       : setpoint current (A)
          areas_A          : normalization area(s) in amps of the sweep(s) that make up
                             the curve
          label            : text used where the file name would be printed; its
                             '<path> | <text>' form names the cache and saved fits
          delta_I_window_A : measured I(V_bias) - I(V_ref) per sweep (A); required when
                             the window does not contain 0 V (see _set_anchor)
          lia_raw, iv_raw  : the raw LIA and I(V) columns (the loader's data in physical units,
                             shape (npts, n), in the order of energy_raw) the curve was
                             built from: one raw set (see raw_sets), so that a refit with
                             other filter settings reruns the pipeline
          units_lia, units_iv : the RHK_Zunits of the LIA and Current pages (with lia_raw /
                             iv_raw; None when not known)
          vmod             : the modulation amplitude RHK_CH1Drive_Amplitude (V; None when
                             not recorded)
          energy_raw       : energy axis of the raw columns (any order; V when omitted)
          filter_stage     : kept for the record ('column' or 'average' with raw columns,
                             'sets' with raw_sets, 'post' without); the pipeline is the
                             same whatever the stage (filters on the LIA columns)
          raw_sets         : list of dicts {'energy', 'lia', 'iv', 'units_lia', 'units_iv',
                             'vmod'} (STS.from_arrays), one per
                             file, each on its own full energy axis with the raw columns
                             of the sweeps that take part (the Browser's
                             averages); a refit rebuilds the curve from them on the
                             common axis V
          resample         : how a raw set whose points do not coincide with V reaches
                             V on a rebuild, as the Browser's run did: 'bin' (its points
                             averaged within half an axis step either side of each axis
                             point), 'interp' (linear interpolation at the axis points),
                             None (such a set is refused); either way the set's points
                             are weighted axis step / set step in the average
          quantity         : 'didv' (the curve is dI/dV) or 'ratio' (the curve is the normalized
                             conductance (dI/dV)/(I/V): no current anchor, and a rebuild from
                             raw columns divides after the calibration, STS.divide_by_iv)
          iv_broadening    : broadening width (V) of I/V for quantity 'ratio' (0: the plain I/V)
          iv_zero_sub      : for quantity 'ratio', each sweep's current at 0 V subtracted
                             before I/V is formed (STS.iv_corrected), as the sender did
        """
        self = cls.__new__(cls)
        self.ifile = label
        self.quantity = 'ratio' if quantity == 'ratio' else 'didv'
        self.iv_broadening = float(iv_broadening)
        self.iv_zero_sub = bool(iv_zero_sub)
        self.hampel_filter_params = None if hampel_filter_params is None else dict(hampel_filter_params)
        self.sgolay_filter_params = None if sgolay_filter_params is None else dict(sgolay_filter_params)
        self.resample = resample if resample in ('bin', 'interp') else None
        self.background = 'wkb'
        self._sweep_done = False
        V = np.asarray(V, dtype=np.float64)
        data = np.asarray(data, dtype=np.float64)
        data_u = np.asarray(data_unfiltered, dtype=np.float64)
        if V.shape != data.shape or V.ndim != 1 or data_u.shape != data.shape:
            raise ValueError("V, data, and data_unfiltered must be 1-D and the same length "
                             f"(got {V.shape}, {data.shape}, {data_u.shape}).")
        order = np.argsort(V)
        self.V_bias = float(V_bias)
        self.I_setpoint = float(I_setpoint)
        areas_A = np.atleast_1d(np.asarray(areas_A, dtype=np.float64))
        if np.any(areas_A <= 0):
            raise ValueError("areas_A must be positive normalization areas in amps.")
        self.n_sweeps = int(len(areas_A))
        # Raw sets (what a refit with other filter settings reruns): raw_sets as given,
        # or one set from lia_raw / iv_raw; none for a curve sent without raw columns.
        self.lia = self.iv = None
        self.units_lia, self.units_iv = units_lia, units_iv
        self.vmod = None if vmod is None else float(vmod)
        self.energy = V.copy()
        if raw_sets:
            self.raw_sets = [sts.STS.from_arrays(rs['energy'], rs['lia'], rs['iv'], rs.get('units_lia'),
                                                 rs.get('units_iv'), rs.get('vmod')) for rs in raw_sets]
            self.filter_stage = 'sets'
        elif lia_raw is not None and iv_raw is not None:
            energy_raw = V if energy_raw is None else np.asarray(energy_raw, dtype=np.float64)
            if len(energy_raw) != len(V):
                raise ValueError(f"energy_raw ({len(energy_raw)}) must have the {len(V)} points of V.")
            self.raw_sets = [sts.STS.from_arrays(energy_raw, lia_raw, iv_raw, units_lia, units_iv, vmod)]
            self.lia, self.iv = self.raw_sets[0]['lia'], self.raw_sets[0]['iv']
            self.energy = self.raw_sets[0]['energy'].copy()
            self.filter_stage = filter_stage if filter_stage in ('column', 'average') else 'sets'
        else:
            self.raw_sets = None
            self.filter_stage = 'post'
        self._setup_axes(V[order])
        self.data = data[order]
        self.data_unfiltered = data_u[order]
        self.areas_A = areas_A
        self.delta_I_window_A = (None if delta_I_window_A is None
                                 else np.atleast_1d(np.asarray(delta_I_window_A, dtype=np.float64)))
        self._set_anchor(areas_A, delta_I_window_A)
        self._set_noise()
        pA = 1e12
        print(f"Curve: {self.ifile}")
        print(f"Sweeps represented: {self.n_sweeps}, points: {len(self.V)}, "
              f"energy {self.V[0]:.3f} to {self.V[-1]:.3f} V"
              + ("" if self.window_has_zero else "  (window does not contain 0 V)"))
        if self.quantity == 'ratio':
            print(f"Quantity: (dI/dV)/(I/V), {self._feenstra_text()}; "
                  f"setpoint V_bias = {self.V_bias:.3f} V, I_setpoint = {self.I_setpoint * pA:.2f} pA; "
                  f"normalization areas of the ratio curves (arb. units x V) = "
                  f"{', '.join(f'{a:.4g}' for a in areas_A)}")
        else:
            print(f"Setpoint: V_bias = {self.V_bias:.3f} V, I_setpoint = {self.I_setpoint * pA:.2f} pA; "
                  f"areas = {', '.join(f'{a * pA:.3f}' for a in areas_A)} pA")
        if delta_I_window_A is not None:
            dI = np.atleast_1d(np.asarray(delta_I_window_A, dtype=np.float64))
            print(f"Measured I(V_bias) - I(V_ref) per sweep: {', '.join(f'{d * pA:.2f}' for d in dI)} pA")
        print(self._anchor_text())
        self._print_noise()
        if self.raw_sets:
            rebuilt, _, _ = self._build_data(self.hampel_filter_params, self.sgolay_filter_params)
            dev = float(np.max(np.abs(rebuilt - self.data))) if len(rebuilt) == len(self.data) else np.inf
            print(f"Raw sets: {len(self.raw_sets)}, averaged as the plain mean"
                  + (f", sets off the axis {'binned' if self.resample == 'bin' else 'interpolated'} "
                     f"(their points weighted by axis step / set step)" if self.resample else "")
                  + "; rebuilding the curve from them "
                  f"reproduces the sent curve to {dev:.3g} (max abs, arb. units)")
        else:
            print("No raw columns with this curve: refits use the stored filter settings only.")
        return self

    def process(self):
        """
        The pipeline of a file opened directly: one raw set (this file's
        sweeps), every sweep filtered on its LIA with the initial settings,
        calibrated, divided by I/V for quantity 'ratio', normalized, and the
        sweeps averaged (the plain mean; _process_raw_sets, the same path a
        refit with other settings reruns).

        Produces
          self.V               : energy axis, ascending (V)
          self.data_unfiltered : averaged curve with every filter off (arb. units)
          self.data            : averaged, smoothed dI/dV (arb. units), same order as self.V
          self.target          : anchor target for the model integral, arb. units
          self.noise_level     : candidate-threshold noise level (arb. units)
        """
        # One raw set: this file's acquired sweeps on its own axis.
        self.raw_sets = [sts.STS.from_arrays(self.energy, self.lia, self.iv, self.units_lia, self.units_iv,
                                             self.vmod)]
        order = np.argsort(self.energy)
        self._setup_axes(self.energy[order])

        # The curve with every filter off (the "unfiltered" curve of the record).
        self.data_unfiltered, _, _ = self._process_raw_sets(None, None)

        # Data context for the initial settings: the filtered curve, the anchor
        # (areas in amps so that it is consistent with the currents; per sweep
        # I(V_bias) - I(V_ref) in amps).  No sweep has run yet, so no ceiling.
        self.envelope = None
        self._rebuild(self.hampel_filter_params, self.sgolay_filter_params)
        areas_A = self.areas_A
        self._set_noise()

        # Console report (the loader's amps): the calibration of the fitted curve before normalization
        # (the raw LIA columns, never filtered, against the current through the fit's filters).
        pA = 1e12
        iv_f = filters.filter_lia(self.iv, self.hampel_filter_params, self.sgolay_filter_params)
        calibrated, self.calib_fits = calibration.calibrate(self.energy, self.lia, iv_f, getattr(self, 'vmod', None))
        cal_avg = np.mean(calibrated, axis=1)[order]
        I_cal = cumulative_trapezoid(cal_avg, self.V, initial=0)
        I_cal = I_cal - I_cal[self.i_ref]
        iv_avg = np.mean(self.iv, axis=1)[order]
        print(f"File: {self.ifile}")
        print(f"Sweeps: {self.n_sweeps}, points: {len(self.V)}, "
              f"energy {self.V[0]:.3f} to {self.V[-1]:.3f} V"
              + ("" if self.window_has_zero else "  (window does not contain 0 V)"))
        print(f"Setpoint: V_bias = {self.V_bias:.3f} V, I_setpoint = {self.I_setpoint * pA:.2f} pA")
        vmod = getattr(self, 'vmod', None)
        print("Modulation amplitude: " + (f"V_mod = {vmod * 1e3:.3f} mV, 1/V_mod = {1.0 / vmod:.4f} /V"
                                          if vmod else "not recorded"))
        ul, ui = getattr(self, 'units_lia', None), getattr(self, 'units_iv', None)
        ua = " /V" if ul == ui else f" {ui}/({ul}·V)"          # the stretch's units (1/V: both pages in A)
        for j, (cf, area) in enumerate(zip(self.calib_fits, areas_A)):
            print(f"  sweep {j}: a = {cf.a:.4f}{ua}"
                  + (f" (1/V_mod = {1.0 / vmod:.4f} /V)" if vmod else "")
                  + f", c = {cf.c * pA:.3f} pA, "
                  f"rms residual = {cf.rms * pA:.3f} pA, area = {area * pA:.3f} pA")
        print(f"Calibration reference V_ref = {self.V[self.i_ref]:.4f} V")
        print(f"Un-normalized averaged calibrated dI/dV integrated V_ref -> V_bias: "
              f"{I_cal[self.i_bias] * pA:.2f} pA  (measured I(V_bias) - I(V_ref) = "
              f"{(iv_avg[self.i_bias] - iv_avg[self.i_ref]) * pA:.2f} pA, "
              f"RHK_Current = {self.I_setpoint * pA:.2f} pA)")
        if self.quantity == 'ratio':
            print(f"Quantity: (dI/dV)/(I/V), {self._feenstra_text()} "
                  f"(the areas above are of the ratio curves, in arb. units x A)")
        print(self._anchor_text())
        self._print_noise()

    # ------------------------------------------------------------------ #
    #  Background evaluation, envelope ceiling, and fit seed              #
    # ------------------------------------------------------------------ #

    def _bg_bounds(self):
        """Background parameter bounds; phi_eV bounded below by half the largest |V| (plus 0.05 eV)."""
        bounds = list(BACKGROUND['bounds'])
        phi_min = 0.5 * float(np.max(np.abs(self.V))) + 0.05
        bounds[0] = (phi_min, bounds[0][1])
        return bounds

    def _envelope(self):
        """
        Tightest background at or below self.data.  For each shape-parameter
        grid point the largest amplitude with amp*shape <= data at every point
        is amp = min(data / shape) (shape must be positive everywhere); the
        grid point is chosen by least squares.
        Returns (amp, shape_params, bg_curve, rss).
        """
        V, data = self.V, self.data
        bounds = self._bg_bounds()
        best = None
        for params in itertools.product(*BACKGROUND['grid']):
            ok = all((lo is None or p >= lo) and (hi is None or p <= hi)
                     for p, (lo, hi) in zip(params, bounds))
            if not ok:
                continue
            shape = BACKGROUND['shape'](V, *params)
            if np.any(~np.isfinite(shape)) or np.any(shape <= 0):
                continue
            amp = np.min(data / shape)
            bg = amp * shape
            rss = float(np.sum((data - bg) ** 2))
            if best is None or rss < best[3]:
                best = (amp, tuple(float(p) for p in params), bg, rss)
        if best is None:
            raise RuntimeError("No positive background envelope found.")
        return best

    def _bg_seed(self):
        """
        Starting point of the background for the joint fit: the shape-parameter
        grid point whose unconstrained least-squares member fits the data best,
        with the amplitude set to half the largest value that keeps that shape
        under the envelope ceiling E (feasible and mid-range, neither the
        ceiling nor zero).  Returns (amp, shape_params, bg_curve).
        """
        V, data = self.V, self.data
        E = self.envelope[2]
        bounds = self._bg_bounds()
        best = None
        for params in itertools.product(*BACKGROUND['grid']):
            ok = all((lo is None or p >= lo) and (hi is None or p <= hi)
                     for p, (lo, hi) in zip(params, bounds))
            if not ok:
                continue
            shape = BACKGROUND['shape'](V, *params)
            if np.any(~np.isfinite(shape)) or np.any(shape <= 0):
                continue
            amp_ls = float(np.sum(data * shape) / np.sum(shape ** 2))     # unconstrained least squares
            rss = float(np.sum((data - amp_ls * shape) ** 2))
            if best is None or rss < best[0]:
                best = (rss, tuple(float(p) for p in params), shape)
        if best is None:
            raise RuntimeError("No positive background shape found.")
        _, params, shape = best
        amp = 0.5 * float(np.min(E / shape))
        return amp, params, amp * shape

    def _find_candidates(self, noise_level):
        """
        Candidate generic-peak positions: local maxima of the residual
        env_residual (data - envelope with the WKB background; the data itself
        without a background).  Prominence threshold = noise_level (none if
        None).  Returns a list of dicts sorted by prominence, descending, with
        the seed height in the model's amplitude units ('height': LDOS units,
        residual / ceiling, with the background; data units without) and the
        half width at half maximum ('hwhm') from the residual.
        """
        res = self.env_residual
        if noise_level is not None:
            peaks, props = find_peaks(res, prominence=noise_level)
        else:
            peaks, props = find_peaks(res)
        dV = float(np.mean(np.diff(self.V)))
        cands = []
        if len(peaks) > 0:
            widths, _, _, _ = peak_widths(res, peaks, rel_height=0.5)
            for p, prom, w in zip(peaks, props['prominences'], widths):
                fwhm = max(w * dV, 2.0 * dV)
                height = float(res[p] / self.env_bg[p]) if self.background == 'wkb' else float(res[p])
                cands.append({'center': float(self.V[p]),
                              'height': height,
                              'prominence': float(prom),
                              'hwhm': 0.5 * fwhm})
            cands.sort(key=lambda c: c['prominence'], reverse=True)
        return cands

    # ------------------------------------------------------------------ #
    #  Joint constrained fit (background + features + peaks)              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _feature_width_values(feat, entry):
        """
        Starting width values of a known feature, by width-parameter name.
        Accepted keys, in order of precedence: the parameter names themselves
        ('sigma', 'gamma', 'sigma_left', ...; 'width' for erf_step); 'hwhm'
        (and 'hwhm_right' for an asymmetric peak), split into equal Gaussian and
        Lorentzian parts; 'width' as one value for every width parameter.
        """
        vals = {}
        for si, side in enumerate(entry['sides']):
            key = 'hwhm' if si == 0 else 'hwhm_right'
            hw = feat.get(key, feat.get('hwhm'))
            if hw is not None:
                s, g = _voigt_seed_widths(hw)
                vals[side[0]], vals[side[1]] = s, g
        for wname in entry['widths']:
            if wname in feat:
                vals[wname] = float(feat[wname])
            elif wname not in vals:
                if 'width' not in feat:
                    raise ValueError(f"Known feature '{feat['type']}' needs '{wname}', 'hwhm' or 'width'.")
                vals[wname] = float(feat['width'])
        return vals

    # Holdable peak parameters (the edit window's parameter panel), in panel order:
    # (peak-dict field, panel name).  'amp' is the field 'amp_seed'.
    PEAK_PARAMS = (('center', 'center (V)'), ('amp', 'height'),
                   ('hwhm', 'HWHM left (V)'), ('hwhm_right', 'HWHM right (V)'),
                   ('mix', 'L share left'), ('mix_right', 'L share right'))
    # Holdable background parameters: (parameter name, panel name).
    BG_PARAMS = (('bg_amp', 'amp (0 V)'), ('bg_phi_eV', 'phi (eV)'), ('bg_kappa', 'kappa'))

    def _build_param_spec(self, feature_subset, peaks, seeds, bg_hold=None):
        """
        Free-parameter list: the background (amplitude and shape parameters,
        seeded by _bg_seed; none in background mode 'none'), then known
        features, then generic peaks.  Each peak is a dict {'shape', 'center',
        'amp_seed', 'hwhm'[, 'hwhm_right', 'mix', 'mix_right', 'hold']} with
        amp_seed in the model's amplitude units (LDOS units with the
        background, data units without); each Voigt side starts with equal
        Gaussian and Lorentzian half widths.
        'seeds' (previous solution, by parameter name) override the seeds when
        present.
        Holds: a peak's 'hold' lists the fields held at the dict's values
        ('center', 'amp', 'hwhm', 'hwhm_right', 'mix', 'mix_right'); center and
        amp are fixed parameters; a side whose HWHM and Lorentzian share are
        both held has its sigma and gamma fixed; a side with only one of the
        two held keeps sigma and gamma free under an equality constraint on
        that quantity (see _fit_model).  bg_hold, {'bg_amp' | 'bg_phi_eV' |
        'bg_kappa': value}, fixes those background parameters (background
        mode 'wkb' only).
        Returns (spec, fixed, holds): spec is a list of (name, start, bounds),
        fixed a dict of the parameters held at a value, holds a list of
        (label, sigma name, gamma name, 'hwhm' | 'mix', value) equality
        constraints.
        """
        V = self.V
        e_lo, e_hi = float(V[0]), float(V[-1])
        dV = float(np.mean(np.diff(V)))
        w_lo, w_hi = 2.0 * dV, (e_hi - e_lo)
        seeds = seeds or {}
        bg_hold = bg_hold or {}
        spec = []
        fixed = {}
        holds = []
        if self.background == 'wkb':
            env_bg = self.envelope[2]
            seed_amp, seed_params, _ = self.bg_seed
            res = self.data / env_bg - 1.0                       # feature amplitude seeds (LDOS units)
            if 'bg_amp' in bg_hold:
                fixed['bg_amp'] = float(bg_hold['bg_amp'])
            else:
                spec.append(('bg_amp', seeds.get('bg_amp', seed_amp), (0.0, None)))
            for name, seed, bnd in zip(BACKGROUND['names'], seed_params, self._bg_bounds()):
                if f'bg_{name}' in bg_hold:
                    fixed[f'bg_{name}'] = float(bg_hold[f'bg_{name}'])
                else:
                    spec.append((f'bg_{name}', seeds.get(f'bg_{name}', seed), bnd))
        else:
            res = self.data                                      # feature amplitude seeds (data units)
        for fi, feat in enumerate(feature_subset):
            ftype = feat['type']
            if ftype not in FEATURE_REGISTRY:
                raise ValueError(f"Unknown feature type '{ftype}'. "
                                 f"Registry: {list(FEATURE_REGISTRY)}")
            entry = FEATURE_REGISTRY[ftype]
            side_names = {n for side in entry['sides'] for n in side}
            bounds = feat.get('bounds', {})
            center = float(feat['center'])
            idx_c = int(np.argmin(np.abs(V - center)))
            amp_seed = seeds.get(f'f{fi}_amp', max(float(res[idx_c]), 1e-12))
            spec.append((f'f{fi}_amp', amp_seed, bounds.get('amp', (0.0, None))))
            if feat.get('fix_center', False):
                fixed[f'f{fi}_center'] = center
            else:
                spec.append((f'f{fi}_center', seeds.get(f'f{fi}_center', center),
                             bounds.get('center', (e_lo, e_hi))))
            widths = self._feature_width_values(feat, entry)
            fix_names = set(entry['widths']) if feat.get('fix_width', False) else set(feat.get('fix_widths', []))
            for wname in entry['widths']:
                if wname in fix_names:
                    fixed[f'f{fi}_{wname}'] = widths[wname]
                else:
                    default = (0.0, w_hi) if wname in side_names else (w_lo, w_hi)
                    spec.append((f'f{fi}_{wname}', seeds.get(f'f{fi}_{wname}', widths[wname]),
                                 bounds.get(wname, default)))

        for pi, pk in enumerate(peaks):
            entry = FEATURE_REGISTRY[pk['shape']]
            hold = set(pk.get('hold', ()))
            if 'amp' in hold:
                fixed[f'p{pi}_amp'] = max(float(pk['amp_seed']), 1e-12)
            else:
                spec.append((f'p{pi}_amp', seeds.get(f'p{pi}_amp', max(float(pk['amp_seed']), 1e-12)), (0.0, None)))
            if 'center' in hold:
                fixed[f'p{pi}_center'] = float(pk['center'])
            else:
                spec.append((f'p{pi}_center', seeds.get(f'p{pi}_center', float(pk['center'])), (e_lo, e_hi)))
            for si, side in enumerate(entry['sides']):
                hw = pk['hwhm'] if si == 0 else pk.get('hwhm_right', pk['hwhm'])
                mix = pk.get('mix', 0.5) if si == 0 else pk.get('mix_right', pk.get('mix', 0.5))
                w_held = ('hwhm' if si == 0 else 'hwhm_right') in hold
                m_held = ('mix' if si == 0 else 'mix_right') in hold
                s_seed, g_seed = _voigt_widths_from(max(float(hw), dV), mix)
                s_name, g_name = f'p{pi}_{side[0]}', f'p{pi}_{side[1]}'
                if w_held and m_held:
                    fixed[s_name] = float(s_seed)
                    fixed[g_name] = float(g_seed)
                    continue
                spec.append((s_name, seeds.get(s_name, s_seed), (0.0, w_hi)))
                spec.append((g_name, seeds.get(g_name, g_seed), (0.0, w_hi)))
                label = f"peak {pi} {'left' if si == 0 else 'right'}"
                if w_held:
                    holds.append((label, s_name, g_name, 'hwhm', float(hw)))
                elif m_held:
                    holds.append((label, s_name, g_name, 'mix', float(mix)))
        return spec, fixed, holds

    def _model_components(self, x, spec, fixed, feature_subset, peaks, V=None):
        """
        Evaluate the background, each known feature, and each generic peak for
        parameter vector x, on self.V or on a given grid.  With the WKB
        background the features and peaks are in LDOS units (the bracket of
        B * [1 + ...]); without a background (bg None) they are in data units
        and the model is their sum; the caller combines them (_combine).
        Returns (bg, feats, pks, params).
        """
        if V is None:
            V = self.V
        p = {name: val for (name, _, _), val in zip(spec, x)}
        p.update(fixed)

        if self.background == 'wkb':
            shape_params = [p[f'bg_{name}'] for name in BACKGROUND['names']]
            bg = p['bg_amp'] * BACKGROUND['shape'](V, *shape_params)
        else:
            bg = None

        feats = []
        for fi, feat in enumerate(feature_subset):
            entry = FEATURE_REGISTRY[feat['type']]
            widths = [p[f'f{fi}_{w}'] for w in entry['widths']]
            if feat['type'] == 'erf_step':
                feats.append(entry['func'](V, p[f'f{fi}_amp'], p[f'f{fi}_center'], *widths,
                                           direction=float(feat.get('direction', 1.0))))
            else:
                feats.append(entry['func'](V, p[f'f{fi}_amp'], p[f'f{fi}_center'], *widths))

        pks = []
        for pi, pk in enumerate(peaks):
            entry = FEATURE_REGISTRY[pk['shape']]
            widths = [p[f'p{pi}_{w}'] for w in entry['widths']]
            pks.append(entry['func'](V, p[f'p{pi}_amp'], p[f'p{pi}_center'], *widths))
        return bg, feats, pks, p

    @staticmethod
    def _combine(bg, feats, pks, n=None):
        """
        Data-domain model: B * (1 + sum of features and peaks) with a
        background, the plain sum of the features and peaks without one (bg
        None; n = number of points, for the empty sum).
        """
        s = sum(feats) + sum(pks) if (feats or pks) else 0.0
        if bg is None:
            return s if (feats or pks) else np.zeros(n if n is not None else 0)
        return bg * (1.0 + s)

    def _model_anchor_integral(self, model_fn):
        """
        Anchor integral of the data-domain model (see _set_anchor):
          'setpoint' : from 0 V to V_bias on the analytic grid self.V_anchor
          'window'   : over the measured window, from V_ref to V_bias on self.V
        """
        if self.anchor_mode == 'setpoint':
            model = model_fn(self.V_anchor)
            return float(np.trapezoid(model, x=self.V_anchor))
        model = model_fn(self.V)
        I = cumulative_trapezoid(model, self.V, initial=0)
        return float(I[self.i_bias] - I[self.i_ref])

    @staticmethod
    def _renumber_peaks(fit):
        """
        Renumber the peaks of a fit by center energy (most negative first;
        ties keep the optimizer's order): 'peaks', 'peak_curves', the 'p{i}_*'
        parameters, and the 'p{i}_*' names of 'spec' and 'fixed' are reordered
        in place; 'x' keeps its order (it is indexed like 'spec').  The
        permutation is stored as fit['peak_order'] (optimizer-order index of
        each renumbered peak).
        """
        k = len(fit['peaks'])
        order = sorted(range(k), key=lambda i: (fit['peaks'][i]['center'], i))
        fit['peak_order'] = order
        if order == list(range(k)):
            return fit
        rename = {f'p{old}_': f'p{new}_' for new, old in enumerate(order)}

        def ren(name):
            if name.startswith('p'):
                head = name.split('_', 1)[0] + '_'
                if head in rename:
                    return rename[head] + name[len(head):]
            return name

        fit['peaks'] = [fit['peaks'][i] for i in order]
        fit['peak_curves'] = [fit['peak_curves'][i] for i in order]
        fit['params'] = {ren(name): val for name, val in fit['params'].items()}
        fit['fixed'] = {ren(name): val for name, val in fit['fixed'].items()}
        fit['spec'] = [(ren(name), start, bnd) for name, start, bnd in fit['spec']]
        return fit

    def _fit_model(self, feature_subset, peaks, seeds=None, bg_hold=None):
        """
        Joint constrained least-squares fit of the background + known features
        + generic peaks with SLSQP (finite-difference gradients).
        With the WKB background (self.background 'wkb'): minimize
        sum (data/B - [1 + sum])^2; bounds: background amplitude >= 0 and shape
        parameters within _bg_bounds; inequalities at every data point: B <= E
        (the envelope, the ceiling; hence B <= data) and B >= floor > 0.
        Without a background ('none'): the model is the plain sum of the
        features and peaks in data units, minimize sum (data - sum)^2, no
        ceiling, no positivity constraint, no offset.
        Both: all feature/peak amplitudes >= 0, Voigt widths >= 0; per peak
        side: Voigt half width >= one energy step; for an asymmetric peak the
        ratio of its two half widths <= max_width_ratio; equality: anchor
        integral of the data-domain model = target.
        Holds (see _build_param_spec): held parameters are fixed and leave
        the free list; a side with only its HWHM or only its Lorentzian share
        held gets an equality constraint on that quantity.  bg_hold fixes
        background parameters.  Every constraint still applies to the held
        values (a hold that breaks one is rejected like any other solution).
        The model is evaluated once per parameter vector and shared by the
        objective and the constraints (memo keyed on x).
        All of the constraints are re-checked on the returned solution; the
        peaks of an accepted solution are renumbered by center energy.

        Returns a dict with the fit result, or None if infeasible; the reason
        for a rejection is left in self._last_reject.
        """
        data = self.data
        n = len(data)
        with_bg = self.background == 'wkb'
        spec, fixed, holds = self._build_param_spec(feature_subset, peaks, seeds, bg_hold)
        x0 = np.array([s[1] for s in spec], dtype=np.float64)
        bounds = [s[2] for s in spec]
        env_bg = self.envelope[2] if with_bg else None
        floor = 1e-6 * float(np.max(data))
        dV = float(np.mean(np.diff(self.V)))
        scale = max(np.sum((data / env_bg - 1.0) ** 2) if with_bg else np.sum(data ** 2), 1e-30)

        # Memo: the last parameter vector's (bg, feats, pks) per grid, so that the
        # objective and the constraints share one evaluation at the same x.
        memo = {'data': [None, None], 'anchor': [None, None]}

        def parts(x, grid='data'):
            slot = memo[grid]
            key = x.tobytes()
            if slot[0] != key:
                V = self.V if grid == 'data' else self.V_anchor
                bg, feats, pks, _ = self._model_components(x, spec, fixed, feature_subset, peaks, V=V)
                slot[0], slot[1] = key, (bg, feats, pks)
            return slot[1]

        def objective(x):
            bg, feats, pks = parts(x)
            if with_bg:
                if np.any(~np.isfinite(bg)) or np.any(bg <= 0):
                    return 1e30
                s = sum(feats) + sum(pks) if (feats or pks) else 0.0
                r = data / bg - (1.0 + s)
            else:
                r = data - self._combine(None, feats, pks, n)
            if np.any(~np.isfinite(r)):
                return 1e30
            return np.sum(r ** 2) / scale

        def ineq_bg_below_envelope(x):
            bg, _, _ = parts(x)
            return np.nan_to_num(env_bg - bg, nan=-1.0)

        def ineq_bg_positive(x):
            bg, _, _ = parts(x)
            return np.nan_to_num(bg - floor, nan=-1.0)

        def eq_integral(x):
            def model_fn(V):
                grid = 'anchor' if V is self.V_anchor else 'data'
                bg, feats, pks = parts(x, grid)
                return self._combine(bg, feats, pks, len(V))
            val = self._model_anchor_integral(model_fn)
            if not np.isfinite(val):
                return 1.0
            return (val - self.target) / abs(self.target)

        # Voigt half widths per peak side: each >= dV; two-sided peaks capped in ratio.
        r_max = float(self.max_width_ratio)
        names = [s[0] for s in spec]

        def _getter(key):
            if key in names:
                idx = names.index(key)
                return lambda x: x[idx]
            val = fixed[key]
            return lambda x, v=val: v
        sided = []                                   # (label, [hwhm_fn per side])
        for fi, feat in enumerate(feature_subset):
            sides = FEATURE_REGISTRY[feat['type']]['sides']
            if sides:
                sided.append((f'feature {fi}', [(_getter(f'f{fi}_{s}'), _getter(f'f{fi}_{g}')) for s, g in sides]))
        for pi, pk in enumerate(peaks):
            sides = FEATURE_REGISTRY[pk['shape']]['sides']
            if sides:
                sided.append((f'peak {pi}', [(_getter(f'p{pi}_{s}'), _getter(f'p{pi}_{g}')) for s, g in sides]))

        def half_widths(x):
            return [(label, [_voigt_hwhm(fs(x), fg(x)) for fs, fg in fns]) for label, fns in sided]

        def ineq_widths(x):
            out = []
            for _, hws in half_widths(x):
                for h in hws:
                    out.append(h - dV)
                if len(hws) == 2:
                    out.append(r_max * hws[1] - hws[0])
                    out.append(r_max * hws[0] - hws[1])
            return np.nan_to_num(np.array(out, dtype=np.float64), nan=-1.0)

        # Holds of one side quantity (HWHM or Lorentzian share) with the other free:
        # equalities on the side's sigma and gamma, relative to the held value
        # (a share is held as an absolute difference).
        hold_fns = []                                # (label, kind, value, fn(x) -> residual)
        for label, s_name, g_name, kind, value in holds:
            fs, fg = _getter(s_name), _getter(g_name)
            if kind == 'hwhm':
                hold_fns.append((label, kind, value,
                                 lambda x, fs=fs, fg=fg, v=value: (_voigt_hwhm(fs(x), fg(x)) - v) / max(abs(v), dV)))
            else:
                hold_fns.append((label, kind, value,
                                 lambda x, fs=fs, fg=fg, v=value: _voigt_mix(fs(x), fg(x)) - v))

        def eq_holds(x):
            return np.nan_to_num(np.array([fn(x) for _, _, _, fn in hold_fns], dtype=np.float64), nan=1.0)

        cons = []
        if with_bg:
            cons.append({'type': 'ineq', 'fun': ineq_bg_below_envelope})
            cons.append({'type': 'ineq', 'fun': ineq_bg_positive})
        if self.anchor_mode != 'none':
            cons.append({'type': 'eq', 'fun': eq_integral})
        if sided:
            cons.append({'type': 'ineq', 'fun': ineq_widths})
        if hold_fns:
            cons.append({'type': 'eq', 'fun': eq_holds})

        if len(spec) == 0:
            # Everything held: nothing to optimize, the held model is checked as it is.
            x_best = np.zeros(0, dtype=np.float64)
            success, message = True, 'every parameter held'
        else:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                              constraints=cons, options={'maxiter': 1500, 'ftol': 1e-10})
            x_best = result.x
            success, message = bool(result.success), result.message

        def reject(reason):
            self._last_reject = reason
            return None

        bg, feats, pks = parts(x_best)
        if with_bg and np.any(~np.isfinite(bg)):
            return reject("background not finite")
        model = self._combine(bg, feats, pks, n)
        if np.any(~np.isfinite(model)):
            return reject("model not finite")
        if with_bg:
            max_env_violation = float(np.max(bg - env_bg))
            max_violation = float(np.max(bg - data))
            min_bg = float(np.min(bg))
        else:
            max_env_violation = max_violation = min_bg = 0.0
        integral_err = float(eq_integral(x_best)) if self.anchor_mode != 'none' else 0.0
        tol = 1e-9 * float(np.max(data))
        hw_report = half_widths(x_best)
        problems = []
        if not success:
            problems.append(f"SLSQP: {message}")
        for label, kind, value, fn in hold_fns:
            miss = float(fn(x_best))
            if abs(miss) > 1e-4:
                what = "HWHM" if kind == 'hwhm' else "Lorentzian share"
                problems.append(f"{label}: held {what} {value:.4g} missed by {miss:.3g}"
                                + (" (relative)" if kind == 'hwhm' else ""))
        if max_env_violation > tol:
            problems.append(f"background above the envelope ceiling by {max_env_violation:.3g}")
        if max_violation > tol:
            problems.append(f"background above the data by {max_violation:.3g}")
        if with_bg and min_bg <= 0.0:
            problems.append(f"background not positive (min {min_bg:.3g})")
        if abs(integral_err) > 1e-4:
            problems.append(f"anchor missed by {integral_err * 100:.3g} %")
        for label, hws in hw_report:
            if min(hws) < dV * (1.0 - 1e-6):
                problems.append(f"{label}: half width {min(hws):.4g} V below one energy step")
            if len(hws) == 2 and max(hws) / max(min(hws), 1e-300) > r_max * (1.0 + 1e-6):
                problems.append(f"{label}: half-width ratio {max(hws) / max(min(hws), 1e-300):.3g} above the cap")
        if problems:
            return reject("; ".join(problems))
        self._last_reject = None

        rss = float(np.sum((data - model) ** 2))            # data domain, always
        p_free = len(spec)                                  # background parameters (when any) are in spec
        bic = n * np.log(rss / n) + p_free * np.log(n)
        bg_c, feats_c, pks_c, params = self._model_components(x_best, spec, fixed, feature_subset, peaks)
        peaks_out = []
        for pi, pk in enumerate(peaks):
            sides = FEATURE_REGISTRY[pk['shape']]['sides']
            hws = [_voigt_hwhm(params[f'p{pi}_{s}'], params[f'p{pi}_{g}']) for s, g in sides]
            mixes = [_voigt_mix(params[f'p{pi}_{s}'], params[f'p{pi}_{g}']) for s, g in sides]
            out = {'shape': pk['shape'], 'center': float(params[f'p{pi}_center']),
                   'amp_seed': float(params[f'p{pi}_amp']), 'hwhm': hws[0], 'mix': mixes[0],
                   'manual': bool(pk.get('manual', False)),
                   'hold': sorted(set(pk.get('hold', ())))}
            if len(hws) == 2:
                out['hwhm_right'] = hws[1]
                out['mix_right'] = mixes[1]
            peaks_out.append(out)
        fit = {'feature_subset': feature_subset,
               'peaks': peaks_out, 'k_peaks': len(peaks),
               'bg_hold': dict(bg_hold) if (bg_hold and with_bg) else {},
               'x': x_best, 'spec': spec, 'fixed': fixed, 'params': params,
               'model': model, 'bg': bg_c, 'features': feats_c, 'peak_curves': pks_c,
               'rss': rss, 'rms': np.sqrt(rss / n), 'bic': bic, 'n_free': p_free,
               'max_bg_violation': max_violation, 'max_env_violation': max_env_violation,
               'min_bg': min_bg, 'integral_err': integral_err}
        return self._renumber_peaks(fit)

    # ------------------------------------------------------------------ #
    #  Enumeration and selection                                          #
    # ------------------------------------------------------------------ #

    def _worker_state(self):
        """
        Lightweight copy of this analyzer holding only what _fit_model needs
        (plain arrays and dicts, no SM4 file object), for shipping to worker processes.
        """
        state = Fit.__new__(Fit)
        for name in ('V', 'data', 'target', 'anchor_mode', 'i_ref', 'i_bias', 'V_bias', 'V_anchor',
                     'envelope', 'bg_seed', 'max_width_ratio', 'background'):
            setattr(state, name, getattr(self, name))
        return state

    def _run_sweep(self, jobs):
        """
        Run every fit of the sweep.  'jobs' is a list of dicts with
        'subset_idx', 'subset' (the known features of that subset), 'k' and
        'peaks' (the k most prominent candidates as peak seeds).  Serially the
        fits run in order; with an executor they are all submitted at once and
        collected as they complete.  One console line per finished fit.
        Returns the rows in job order: fit dicts (with 'subset_idx') or
        infeasible markers (with the rejection reason).
        """
        total = len(jobs)
        rows = [None] * total
        done = 0

        def finish(idx, fit, reason, seconds):
            nonlocal done
            done += 1
            job = jobs[idx]
            ftxt = ",".join(str(i) for i in job['subset_idx']) or "-"
            if fit is None:
                rows[idx] = {'subset_idx': job['subset_idx'], 'k_peaks': job['k'],
                             'infeasible': True, 'reason': reason}
                print(f"  fit {done}/{total} done: features {ftxt}, {job['k']} peak(s): "
                      f"infeasible ({reason}), {seconds:.1f} s")
            else:
                fit['subset_idx'] = job['subset_idx']
                rows[idx] = fit
                print(f"  fit {done}/{total} done: features {ftxt}, {job['k']} peak(s): "
                      f"feasible, BIC = {fit['bic']:.4f}, {seconds:.1f} s")

        if self._executor is None:
            for idx, job in enumerate(jobs):
                finish(idx, *_run_fit(self, job['subset'], job['peaks'], None))
            return rows
        state = self._worker_state()
        futures = {self._executor.submit(_run_fit, state, job['subset'], job['peaks'], None): idx
                   for idx, job in enumerate(jobs)}
        for f in as_completed(futures):
            finish(futures[f], *f.result())
        return rows

    def select_model(self, known_features=None, n_jobs=1, max_width_ratio=2.5, noise_z=2.0,
                     background='wkb'):
        """
        Enumerate every subset of known features x every peak count (split
        Voigt peaks) at the candidate threshold noise_z x noise level, fit
        each, and select the minimum-BIC model, which becomes fit 1 of the fit
        history (and writes the cache).  Sets self.candidates (list) and
        self.results (all rows).

        n_jobs          : 1 runs the fits serially; -1 uses every CPU core; any other
                          positive integer uses that many worker processes (loky).
        max_width_ratio : cap on the ratio of the two half widths of every peak.
        noise_z         : candidate prominence threshold = noise_z x noise level.
        background      : 'wkb' (the barrier transmission background, the default) or
                          'none' (the model is the plain sum of the features and
                          peaks: no ceiling, no offset); the mode of fit 1, which the
                          'WKB background' toggle of the figures changes for Refits.
        """
        if known_features is None:
            known_features = []
        if max_width_ratio is None or float(max_width_ratio) < 1.0:
            raise ValueError(f"max_width_ratio must be >= 1 (got {max_width_ratio}).")
        if noise_z is None or float(noise_z) <= 0:
            raise ValueError(f"noise_z must be > 0 (got {noise_z}).")
        self.max_width_ratio = float(max_width_ratio)
        self.noise_z = float(noise_z)
        self.known_features = list(known_features)
        self.background = self._background_mode(background)
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1
        if n_jobs < 1:
            raise ValueError(f"n_jobs must be -1 or a positive integer (got {n_jobs}).")
        self._executor = None
        if n_jobs > 1:
            try:
                from loky import get_reusable_executor
            except ImportError as exc:
                raise ImportError("n_jobs > 1 requires the 'loky' package: pip install loky") from exc
            self._executor = get_reusable_executor(max_workers=n_jobs)

        # Background context of the mode: ceiling, seed, and residual (candidates come
        # from the residual: data - ceiling with the background, the data without).
        self._sweep_done = True
        self._set_background_context()
        if self.background == 'wkb':
            amp, params, _, rss = self.envelope
            s_amp, s_params, _ = self.bg_seed
            names = BACKGROUND['names']
            ptxt = ", ".join(f"{n} = {v:.4g}" for n, v in zip(names, params))
            stxt = ", ".join(f"{n} = {v:.4g}" for n, v in zip(names, s_params))
            print(f"\nBackground ({BACKGROUND['name']}) ceiling (tightest member under the data) and fit seed:")
            print(f"  ceiling amp = {amp:.6g}, {ptxt}, RSS = {rss:.4g};  seed amp = {s_amp:.6g}, {stxt}")
        else:
            print("\nNo background: the model is the sum of the features and peaks (data units); "
                  "candidates are local maxima of the data.")

        F = len(known_features)
        subsets = []
        for r in range(F + 1):
            subsets.extend(itertools.combinations(range(F), r))

        # Fit history: every accepted fit in order of calculation (fit N = index N - 1).
        self.fits = []
        self._shown = None
        self._back = []
        self._fwd = []

        level = self.noise_z * self.noise_level
        cands = self._find_candidates(level)
        self.candidates = cands
        M = len(cands)
        print(f"\n##### candidate threshold {level:.4g} = {self.noise_z:g} x noise level {self.noise_level:.4g}: "
              f"{M} candidate peak positions: " + ", ".join(f"{c['center']:.3f} V" for c in cands))

        def make_peaks(k):
            return [{'shape': PEAK_SHAPE, 'center': c['center'], 'amp_seed': c['height'],
                     'hwhm': c['hwhm']} for c in cands[:k]]

        jobs = [{'subset_idx': subset_idx, 'subset': [known_features[i] for i in subset_idx],
                 'k': k, 'peaks': make_peaks(k)}
                for subset_idx in subsets for k in range(M + 1)]
        print(f"\n===== fitting {len(jobs)} candidate models "
              f"({'serially' if self._executor is None else f'{n_jobs} worker processes'}) ...")
        t0 = time.perf_counter()
        rows = self._run_sweep(jobs)
        print(f"  all {len(jobs)} fits done in {time.perf_counter() - t0:.1f} s")
        self.results = rows
        feasible = [r for r in rows if not r.get('infeasible')]
        if not feasible:
            print("  No candidate model satisfied the constraints.")
            return
        best = min(feasible, key=lambda r: r['bic'])
        print(f"{'features':>10s} {'peaks':>6s} {'free':>5s} {'RSS':>12s} {'RMS':>12s} {'BIC':>12s}")
        for r in rows:
            ftxt = ",".join(str(i) for i in r['subset_idx']) or "-"
            if r.get('infeasible'):
                print(f"{ftxt:>10s} {r['k_peaks']:6d} {'':>5s} {'infeasible':>12s}")
                continue
            mark = "  <-- selected" if r is best else ""
            print(f"{ftxt:>10s} {r['k_peaks']:6d} {r['n_free']:5d} "
                  f"{r['rss']:12.5g} {r['rms']:12.5g} {r['bic']:12.4f}{mark}")
        print(f"  Selected: known features = {[f['type'] for f in best['feature_subset']]}, "
              f"{best['k_peaks']} peak(s)")
        self._print_model(best)
        self._accept_fit(dict(best))

    @staticmethod
    def _side_text(params, prefix, sides):
        """Per-side Voigt description: sigma, gamma, half width, and Lorentzian share."""
        parts = []
        for si, (s_name, g_name) in enumerate(sides):
            s, g = params[f'{prefix}_{s_name}'], params[f'{prefix}_{g_name}']
            hw = _voigt_hwhm(s, g)
            share = _voigt_mix(s, g)
            side = "" if len(sides) == 1 else (" left" if si == 0 else " right")
            parts.append(f"{s_name} = {s:.4f} V, {g_name} = {g:.4f} V (HWHM{side} = {hw:.4f} V, "
                         f"L/(G+L) = {share:.2f} {_voigt_designation(share)})")
        return "; ".join(parts)

    def _print_model(self, b):
        """Console description of a fit: background, every known feature, every peak (energy order), fit quality."""
        if b.get('background', self.background) == 'wkb':
            ptxt = ", ".join(f"{n} = {b['params'][f'bg_{n}']:.4g}" for n in BACKGROUND['names'])
            units = " (LDOS units)"
            print(f"    background ({BACKGROUND['name']}, fitted): amp = {b['params']['bg_amp']:.6g}, {ptxt}")
        else:
            units = " (data units)"
            print("    background: none (the model is the sum of the features and peaks)")
        for fi, feat in enumerate(b['feature_subset']):
            entry = FEATURE_REGISTRY[feat['type']]
            if entry['sides']:
                wtxt = self._side_text(b['params'], f'f{fi}', entry['sides'])
            else:
                wtxt = ", ".join(f"{w} = {b['params'][f'f{fi}_{w}']:.4f} V" for w in entry['widths'])
            print(f"    feature {fi} ({feat['type']}): amp = {b['params'][f'f{fi}_amp']:.6g}{units}, "
                  f"center = {b['params'][f'f{fi}_center']:.4f} V, {wtxt}")
        for pi, pk in enumerate(b['peaks']):
            wtxt = self._side_text(b['params'], f'p{pi}', FEATURE_REGISTRY[pk['shape']]['sides'])
            tag = " [manual]" if pk.get('manual') else ""
            print(f"    peak {pi} ({pk['shape']}){tag}: amp = {b['params'][f'p{pi}_amp']:.6g}{units}, "
                  f"center = {b['params'][f'p{pi}_center']:.4f} V, {wtxt}")
        print(f"    RMS(data - model) = {b['rms']:.4g}; BIC = {b['bic']:.2f}; "
              f"max(background - ceiling) = {b['max_env_violation']:.3g}, "
              f"max(background - data) = {b['max_bg_violation']:.3g}, min(background) = {b['min_bg']:.3g}, "
              f"anchor relative error = {b['integral_err']:.3g}")

    # ------------------------------------------------------------------ #
    #  Fit history (refits from the edit window, Revert / Forward)        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _guess_text(g):
        ml, mr = g.get('mix', 0.5), g.get('mix_right', g.get('mix', 0.5))
        return (f"peak at {g['center']:.3f} V, height {g['amp_seed']:.4g}, "
                f"HWHM left {g['hwhm']:.3f} V ({_voigt_designation(ml)} {ml:.2f}), "
                f"right {g.get('hwhm_right', g['hwhm']):.3f} V ({_voigt_designation(mr)} {mr:.2f})")

    def shown_fit(self):
        """The displayed fit dict (None when the history is empty)."""
        return None if self._shown is None else self.fits[self._shown]

    def _accept_fit(self, fit):
        """
        Append an accepted fit to the history as the next numbered fit and show
        it: the previously shown fit goes onto the Revert chain, the Forward
        chain is cleared, and the cache is written.
        """
        if self._shown is not None:
            self._back.append(self._shown)
        self._fwd = []
        fit['number'] = len(self.fits) + 1
        fit.update(self._context())              # the smoothing and curve this fit was made against
        self.fits.append(fit)
        self._shown = len(self.fits) - 1
        path = self._write_cache()
        print(f"Fit {fit['number']} accepted (history: {len(self.fits)} fits; "
              f"Hampel {self._params_text(fit['hampel_filter_params'])}, "
              f"SG {self._params_text(fit['sgolay_filter_params'])}; "
              f"background {fit.get('background', 'wkb')}; quantity {self._qname(fit.get('quantity', self.quantity))}; "
              f"cache: {path})")

    @staticmethod
    def _params_text(p):
        """Short text of a filter setting: 'off' or its parameters."""
        if p is None:
            return 'off'
        return ", ".join(f"{k} {v:g}" for k, v in p.items())

    def revert_fit(self):
        """
        Show the fit that was displayed before the current one (the current
        one goes onto the Forward chain), restore its data context, and write
        the cache.  Returns False at the start of the chain.
        """
        if not self._back:
            print("Revert fit: no earlier fit.")
            return False
        self._fwd.append(self._shown)
        self._shown = self._back.pop()
        self._restore_context(self.fits[self._shown])
        self._write_cache()
        print(f"Revert fit: showing fit {self._shown + 1} of {len(self.fits)}.")
        return True

    def forward_fit(self):
        """
        Show the fit that was displayed before the last Revert (the current
        one goes onto the Revert chain), restore its data context, and write
        the cache.  Returns False at the end of the chain.
        """
        if not self._fwd:
            print("Forward fit: no later fit.")
            return False
        self._back.append(self._shown)
        self._shown = self._fwd.pop()
        self._restore_context(self.fits[self._shown])
        self._write_cache()
        print(f"Forward fit: showing fit {self._shown + 1} of {len(self.fits)}.")
        return True

    def _refit_peaks(self, peaks, bg=None):
        """
        Joint refit with the given peak list, keeping fit 1's known-feature
        subset.  Background and feature parameters are seeded from the
        displayed fit; peak parameters from the peak dicts (center, amp_seed,
        hwhm, hwhm_right, mix, mix_right; the fields in 'hold' are held, see
        _build_param_spec).  `bg` (the edit window's working background,
        {'bg_amp', 'bg_phi_eV', 'bg_kappa', 'hold'}) seeds the background
        with its values and holds the names in its 'hold' (background mode
        'wkb' only).  The result's peaks are renumbered by energy (see
        _renumber_peaks).  Returns the fit dict or None (reason in
        self._last_reject).
        """
        first = self.fits[0]
        current = self.shown_fit()
        seeds = {k: v for k, v in current['params'].items() if not k.startswith('p')}
        bg_hold = {}
        if bg is not None and self.background == 'wkb':
            for name, _ in self.BG_PARAMS:
                if bg.get(name) is not None:
                    seeds[name] = float(bg[name])
                    if name in bg.get('hold', ()):
                        bg_hold[name] = float(bg[name])
        fit = self._fit_model(first['feature_subset'], peaks, seeds, bg_hold)
        if fit is not None:
            fit['subset_idx'] = first.get('subset_idx', ())
        return fit

    # ------------------------------------------------------------------ #
    #  Cache (fit history on disk) and saving a fit                       #
    # ------------------------------------------------------------------ #

    # Bumped whenever the cached state changes shape (format 1 held the earlier
    # background families and prominence-ordered peaks; format 2 one smoothing
    # setting per analyzer).  Every write is format 3 with 'loader_format'
    # (sm4file.LOADER_FORMAT); a cache is up to date when its format is read
    # here and its loader format is the loader's (cache_up_to_date).  Format 2
    # predates the loader format, so a format-2 cache is out of date and its
    # migration (from_cache) is no longer reached.  A cache that is not up to
    # date is never read or written: a new one goes beside it (cache_path).
    _CACHE_FORMAT = 3
    _CACHE_FORMATS_READ = (2, 3)

    @staticmethod
    def _stem_of(label):
        """
        (directory, stem) of the cache and saved-fit files for a data file path
        or a send label '<path> | <text>'.  The stem is the trailing run of
        underscore-separated number groups of the file's base name (the RHK
        date and time, e.g. 2023_12_29_11_33_34_590), or the cleaned base name
        when there is none, plus the cleaned text after ' | ' when present.
        The directory is the data file's folder (the working directory if the
        path part is not a file).  A directory as the path part
        ('<folder> | <text>', the Browser's averages) names that
        folder, and the stem is the cleaned text alone (e.g. avg3).
        """
        label = str(label)
        path_part, _, rest = label.partition(' | ')
        path_part = path_part.strip()
        clean = lambda s: re.sub(r'[^A-Za-z0-9._-]+', '_', s).strip('_')
        if rest and os.path.isdir(path_part):
            return os.path.abspath(path_part), (clean(rest) or 'curve')
        if os.path.isfile(path_part):
            directory = os.path.dirname(os.path.abspath(path_part))
        else:
            directory = os.getcwd()
        base = os.path.splitext(os.path.basename(path_part))[0]
        m = re.search(r'(\d+(?:_\d+)+)$', base)
        stem = m.group(1) if m else (clean(base) or 'curve')
        if rest:
            stem += '_' + clean(rest)
        return directory, stem

    @classmethod
    def cache_up_to_date(cls, state):
        """Whether a cache state is readable here and was written from this loader (sm4file.LOADER_FORMAT)."""
        return (isinstance(state, dict) and state.get('format') in cls._CACHE_FORMATS_READ
                and state.get('loader_format') == sm4file.LOADER_FORMAT)

    @staticmethod
    def _read_state(path):
        """The unpickled state of a cache file, or None when it cannot be read."""
        try:
            with open(path, 'rb') as fh:
                return pickle.load(fh)
        except Exception:
            return None

    @classmethod
    def cache_file(cls, directory, stem):
        """
        The fit history of <directory>/<stem>_fits.pkl's family
        (sm4file.cache_family: the base and <stem>_fits_<n>.pkl): the newest
        up-to-date file (the highest number), or, when none is up to date,
        the path a new history is written to (the base when the family is
        empty, else the next number: an out-of-date file is never replaced).
        """
        base = os.path.join(directory, f"{stem}_fits.pkl")
        for _, path in reversed(sm4file.cache_family(base)):
            if cls.cache_up_to_date(cls._read_state(path)):
                return path
        return sm4file.cache_new_path(base)

    @classmethod
    def cache_path(cls, ifile):
        """Path of the fit history cache for a data file or send label (cache_file of its folder and stem)."""
        directory, stem = cls._stem_of(ifile)
        return cls.cache_file(directory, stem)

    @classmethod
    def has_cache(cls, ifile):
        """True when an up-to-date fit history cache exists for the data file or send label."""
        path = cls.cache_path(ifile)
        return os.path.isfile(path) and cls.cache_up_to_date(cls._read_state(path))

    def _cache_state(self):
        """
        Plain dict of everything the figure, the edit window, and a refit need
        (arrays, dicts, lists, and scalars only: no analyzer object and no SM4
        file object, so the file unpickles however the module was imported).
        Format 3: the raw columns ('lia', 'iv', with 'units_lia', 'units_iv',
        'vmod'), the filter stage and the per-fit data contexts (inside
        'fits') are included, and 'loader_format' (sm4file.LOADER_FORMAT).
        """
        fields = ('ifile', 'hampel_filter_params', 'sgolay_filter_params',
                  'energy', 'V', 'data',
                  'data_unfiltered', 'V_bias', 'I_setpoint', 'n_sweeps', 'i_ref', 'i_bias',
                  'window_has_zero', 'V_anchor', 'anchor_mode', 'target', 'sigma_diff', 'sg_gain',
                  'noise_level', 'noise_z', 'max_width_ratio', 'known_features',
                  'envelope', 'bg_seed', 'env_bg', 'env_residual', 'candidates', 'results', 'fits',
                  'lia', 'iv', 'units_lia', 'units_iv', 'vmod', 'filter_stage', 'areas_A', 'delta_I_window_A',
                  'quantity', 'iv_broadening', 'iv_zero_sub', 'raw_sets', 'resample', 'background')
        state = {name: getattr(self, name, None) for name in fields}
        state['shown'] = self._shown
        state['back'] = list(self._back)
        state['fwd'] = list(self._fwd)
        state['format'] = self._CACHE_FORMAT
        state['loader_format'] = sm4file.LOADER_FORMAT
        return state

    def _migrate_format_2(self, state):
        """
        Bring a format-2 state (one smoothing setting per analyzer, no raw
        columns) up to format 3 in memory: every fit is stamped with the
        analyzer-level data context; the raw sweeps are reloaded from the .sm4
        when the cache names a file (one raw set), otherwise the stage is
        'post' (no raw columns: refits only with the stored settings).
        """
        ctx = {k: state.get(k) for k in self._CONTEXT_KEYS}
        for fit in self.fits:
            for k, v in ctx.items():
                fit.setdefault(k, v)
        self.lia = self.iv = None
        self.units_lia = self.units_iv = self.vmod = None
        self.raw_sets = None
        self.filter_stage = 'post'
        self.areas_A = None
        self.delta_I_window_A = None
        path_part = str(self.ifile).partition(' | ')[0].strip()
        if ' | ' not in str(self.ifile) and os.path.isfile(path_part):
            try:
                saved_ifile = self.ifile             # the label stays as it was, whatever the load does
                self.ifile = path_part
                try:
                    self.extract_data()              # self.lia, self.iv, the units, vmod, self.energy
                finally:
                    self.ifile = saved_ifile
                self.filter_stage = 'average'
                self.raw_sets = [sts.STS.from_arrays(self.energy, self.lia, self.iv, self.units_lia,
                                                     self.units_iv, self.vmod)]
                order = np.argsort(self.energy)
                iv_f = filters.filter_lia(self.iv, self.hampel_filter_params, self.sgolay_filter_params)
                calibrated, _ = calibration.calibrate(self.energy, self.lia.copy(), iv_f, self.vmod)
                _, norm_areas = normalization.normalize(self.energy, calibrated)
                self.areas_A = np.array(norm_areas)
                iv_sorted = self.iv[order, :]
                self.delta_I_window_A = iv_sorted[self.i_bias, :] - iv_sorted[self.i_ref, :]
                for fit in self.fits:
                    fit.setdefault('areas_A', self.areas_A)
                    fit.setdefault('delta_I_window_A', self.delta_I_window_A)
                print(f"Format-2 cache: raw sweeps reloaded from {path_part} (filter stage 'average').")
            except Exception as exc:
                self.lia = self.iv = None
                self.filter_stage = 'post'
                print(f"Format-2 cache: could not reload the raw sweeps ({exc}); "
                      "refits filter the stored unsmoothed curve (stage 'post').")
        else:
            print("Format-2 cache of a sent curve holds no raw columns; "
                  "refits filter the stored unsmoothed curve (stage 'post').")
        for fit in self.fits:
            fit.setdefault('areas_A', self.areas_A)
            fit.setdefault('delta_I_window_A', self.delta_I_window_A)

    def _write_cache(self):
        """
        Write the fit history cache (to a temporary file, then replaced in
        place).  The file is this analyzer's own once it has one (the cache
        it was opened from or first wrote: rewritten after every accepted fit
        and every Revert/Forward); before that, cache_path's (the newest
        up-to-date file of the family, or a new one beside an out-of-date
        file).  Returns its path.
        """
        path = getattr(self, '_cache_file', None) or self.cache_path(self.ifile)
        tmp = path + '.tmp'
        with open(tmp, 'wb') as fh:
            pickle.dump(self._cache_state(), fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        self._cache_file = path
        return path

    @classmethod
    def from_cache(cls, path):
        """
        Reopen an analyzer from a fit history cache written by _write_cache:
        the figure, the edit window, Revert/Forward, and Save fit work without
        process() or select_model().  Refits run serially.  A cache that is
        not up to date (cache_up_to_date) is refused and left as it is.
        """
        with open(path, 'rb') as fh:
            state = pickle.load(fh)
        if not isinstance(state, dict) or state.get('format') not in cls._CACHE_FORMATS_READ:
            raise ValueError(f"{path} is not a fit history cache of a readable version.")
        if not cls.cache_up_to_date(state):
            raise ValueError(f"{path} was written from another loader (loader format "
                             f"{state.get('loader_format', 1)}, this loader {sm4file.LOADER_FORMAT}); it is out of "
                             f"date and left as it is (a fresh fit writes a new history beside it).")
        self = cls.__new__(cls)
        for name, value in state.items():
            if name not in ('shown', 'back', 'fwd', 'format', 'calibration_shift', 'notch_filter_params'):
                setattr(self, name, value)         # a stored calibration_shift or notch setting is ignored
        self._executor = None
        self._last_reject = None
        self._cache_file = path                          # rewritten in place from here on (_write_cache)
        if getattr(self, 'quantity', None) not in ('didv', 'ratio'):
            self.quantity = 'didv'                       # caches written before the ratio existed
        if getattr(self, 'iv_broadening', None) is None:
            self.iv_broadening = 0.0
        if getattr(self, 'iv_zero_sub', None) is None:
            self.iv_zero_sub = False                     # caches written before the 0 V subtraction
        if getattr(self, 'resample', None) not in ('bin', 'interp'):
            self.resample = None                         # caches written before the resampling
        self._sweep_done = True                           # a cache exists only after select_model
        if getattr(self, 'background', None) not in self.BACKGROUND_MODES:   # caches written before the mode
            self.background = 'wkb'
        for fit in self.fits:
            fit.setdefault('background', 'wkb')
            if fit.get('quantity') not in self.QUANTITIES:   # caches written before the per-fit quantity
                fit['quantity'] = self.quantity
                fit['iv_broadening'] = self.iv_broadening
            if fit.get('iv_zero_sub') is None:               # caches written before the 0 V subtraction
                fit['iv_zero_sub'] = False
        if getattr(self, 'raw_sets', None) is None:      # caches written before the raw sets existed
            self.raw_sets = ([sts.STS.from_arrays(self.energy, self.lia, self.iv, getattr(self, 'units_lia', None),
                                                  getattr(self, 'units_iv', None), getattr(self, 'vmod', None))]
                             if (getattr(self, 'lia', None) is not None and getattr(self, 'iv', None) is not None)
                             else None)
        self._shown = state['shown']
        self._back = list(state['back'])
        self._fwd = list(state['fwd'])
        print(f"Fit history loaded: {path}")
        if state.get('format') == 2:
            print("Format-2 cache: migrating to format 3 (per-fit smoothing).")
            self._migrate_format_2(state)
        if self._shown is not None:
            self._restore_context(self.fits[self._shown])
        if state.get('format') == 2:
            print(f"Cache rewritten as format 3: {self._write_cache()}")
        print(f"Curve: {self.ifile}")
        print(f"Points: {len(self.V)}, energy {self.V[0]:.3f} to {self.V[-1]:.3f} V"
              + ("" if self.window_has_zero else "  (window does not contain 0 V)")
              + "; " + self._anchor_text()[0].lower() + self._anchor_text()[1:]
              + (f"; quantity (dI/dV)/(I/V), {self._feenstra_text()}"
                 if self.quantity == 'ratio' else ""))
        self._print_noise()
        print(f"Fits in the history: {len(self.fits)}; showing fit {self._shown + 1}:")
        self._print_model(self.fits[self._shown])
        return self

    def _fit_record(self, b):
        """Dictionary describing a fit (background equation or none, every feature and peak in energy order)."""
        p = b['params']
        with_bg = b.get('background', 'wkb') == 'wkb'
        if with_bg:
            bg_params = {'amp': float(p['bg_amp'])}
            bg_params.update({n: float(p[f'bg_{n}']) for n in BACKGROUND['names']})
            background = {'family': BACKGROUND['name'], 'equation': BACKGROUND['equation'], 'parameters': bg_params}
            units = 'LDOS units: multiplies the background (data / B - 1)'
            model_eq = f"{self._qlabel()}(V) = B(V) * [1 + sum_i F_i(V) + sum_j P_j(V)]"
        else:
            background = {'family': 'none', 'equation': 'no background: the model is the sum of the components',
                          'parameters': {}}
            units = 'data units (the same as the fitted curve)'
            model_eq = f"{self._qlabel()}(V) = sum_i F_i(V) + sum_j P_j(V)"

        def side_record(prefix, s_name, g_name):
            s, g = float(p[f'{prefix}_{s_name}']), float(p[f'{prefix}_{g_name}'])
            share = _voigt_mix(s, g)
            return {'sigma_V': s, 'gamma_V': g, 'hwhm_V': float(_voigt_hwhm(s, g)),
                    'lorentzian_share': share, 'designation': _voigt_designation(share)}

        def component_record(prefix, ftype, extra):
            entry = FEATURE_REGISTRY[ftype]
            rec = {'shape': ftype, 'equation': entry['equation'],
                   'center_V': float(p[f'{prefix}_center']), 'amplitude': float(p[f'{prefix}_amp'])}
            rec.update(extra)
            if entry['sides']:
                left = side_record(prefix, *entry['sides'][0])
                right = side_record(prefix, *entry['sides'][1])
                rec['left'] = left
                rec['right'] = right
                rec['asymmetry_ratio_right_over_left'] = right['hwhm_V'] / left['hwhm_V'] if left['hwhm_V'] > 0 else None
                rec['fwhm_V'] = left['hwhm_V'] + right['hwhm_V']
            else:
                rec['widths_V'] = {w: float(p[f'{prefix}_{w}']) for w in entry['widths']}
            return rec

        features = [component_record(f'f{fi}', feat['type'], {'index': fi, 'direction': feat.get('direction', 1.0)}
                                     if feat['type'] == 'erf_step' else {'index': fi})
                    for fi, feat in enumerate(b['feature_subset'])]
        peaks = [component_record(f'p{pi}', pk['shape'], {'index': pi, 'manual': bool(pk.get('manual', False))})
                 for pi, pk in enumerate(b['peaks'])]
        return {
            'source': str(self.ifile),
            'quantity': self._qlabel(),
            'filters': {'hampel': b.get('hampel_filter_params'), 'sgolay': b.get('sgolay_filter_params')},
            'iv_broadening_V': float(getattr(self, 'iv_broadening', 0.0)),
            'iv_zero_sub': bool(getattr(self, 'iv_zero_sub', False)),
            'fit_number': int(b['number']),
            'fits_in_session': len(self.fits),
            'noise_option': NOISE_METHOD,
            'decomposition_mode': DECOMP_MODE,
            'model_equation': model_eq,
            'amplitude_units': units,
            'energy_window_V': [float(self.V[0]), float(self.V[-1])],
            'anchor': {'mode': self.anchor_mode,
                       'target_arb_units': None if self.target is None else float(self.target),
                       'V_bias_V': float(self.V_bias), 'I_setpoint_A': float(self.I_setpoint),
                       'relative_error': float(b['integral_err'])},
            'background': background,
            'features': features,
            'peaks': peaks,
            'fit': {'rms': float(b['rms']), 'bic': float(b['bic']), 'n_free': int(b['n_free']),
                    'max_background_minus_ceiling': float(b['max_env_violation']),
                    'max_background_minus_data': float(b['max_bg_violation']),
                    'min_background': float(b['min_bg']),
                    'max_width_ratio': float(self.max_width_ratio)},
        }

    def save_fit(self, directory=None):
        """
        Write the displayed fit as <stem>_fit<N>.json (parameters and
        equations) and <stem>_fit<N>.csv (V, data, background when the fit
        has one, every component in the data domain, total model), N being
        its number in the fit history, next to the data file.  Returns the
        two paths.
        """
        b = self.shown_fit()
        if b is None:
            print("Nothing to save: no fit.")
            return None
        d, stem = self._stem_of(self.ifile)
        if directory is not None:
            d = directory
        stem = os.path.join(d, f"{stem}_fit{b['number']}")
        record = self._fit_record(b)
        with open(stem + '.json', 'w') as fh:
            json.dump(record, fh, indent=2)
        with_bg = b.get('background', 'wkb') == 'wkb'
        to_data = b['bg'] if with_bg else 1.0                # components to the data domain
        cols = [('V', self.V), ('data', self.data)]
        if with_bg:
            cols.append(('background', b['bg']))
        for fi, feat in enumerate(b['feature_subset']):
            cols.append((f"feature_{fi}_{feat['type']}", to_data * b['features'][fi]))
        for pi, pk in enumerate(b['peaks']):
            cols.append((f"peak_{pi}" + ("_manual" if pk.get('manual') else ""), to_data * b['peak_curves'][pi]))
        cols.append(('model', b['model']))
        np.savetxt(stem + '.csv', np.column_stack([np.asarray(c, dtype=np.float64) for _, c in cols]),
                   delimiter=',', header=','.join(n for n, _ in cols), comments='')
        print(f"Fit {b['number']} saved: {stem}.json and {stem}.csv")
        return stem + '.json', stem + '.csv'

    # ------------------------------------------------------------------ #
    #  Plot                                                               #
    # ------------------------------------------------------------------ #

    def _dV(self):
        return float(np.mean(np.diff(self.V)))

    @staticmethod
    def _pending_curve(V, g):
        """Curve of a peak guess in LDOS units (the bracket of data / B - 1)."""
        s_l, g_l = _voigt_widths_from(g['hwhm'], g.get('mix', 0.5))
        s_r, g_r = _voigt_widths_from(g.get('hwhm_right', g['hwhm']), g.get('mix_right', g.get('mix', 0.5)))
        return _asym_voigt(V, g['amp_seed'], g['center'], s_l, g_l, s_r, g_r)

    @staticmethod
    def _new_guess(x, y, dV):
        return {'shape': PEAK_SHAPE, 'center': float(x), 'amp_seed': max(float(y), 1e-12),
                'hwhm': 2.0 * dV, 'hwhm_right': 2.0 * dV, 'mix': 0.5, 'mix_right': 0.5, 'hold': []}

    @staticmethod
    def _apply_drag(g, drag, x, y, dV):
        """
        Update guess g from a drag: x displacement from the center sets the
        half width (both sides for a creation drag, the dragged side otherwise);
        y displacement from the press point sets that side's Lorentzian share
        (up = more Gaussian, down = more Lorentzian; half the peak height covers
        the range from the starting share to a pure limit).
        """
        dx = float(x) - g['center']
        dy = float(y) - drag['y0']
        scale = 0.5 * max(g['amp_seed'], 1e-12)
        if drag['kind'] == 'new':
            g['hwhm'] = g['hwhm_right'] = max(abs(dx), 2.0 * dV)
            g['mix'] = g['mix_right'] = min(max(0.5 - dy / scale, 0.0), 1.0)
        elif dx >= 0:
            g['hwhm_right'] = max(dx, dV)
            g['mix_right'] = min(max(drag['mix0'][1] - dy / scale, 0.0), 1.0)
        else:
            g['hwhm'] = max(-dx, dV)
            g['mix'] = min(max(drag['mix0'][0] - dy / scale, 0.0), 1.0)

    @staticmethod
    def _hit(peaks, x):
        """Index of the peak whose center is nearest x if x lies within its half width on that side."""
        best = None
        for i, g in enumerate(peaks):
            d = x - g['center']
            hw = g.get('hwhm_right', g['hwhm']) if d >= 0 else g['hwhm']
            if abs(d) <= hw and (best is None or abs(d) < best[1]):
                best = (i, abs(d))
        return None if best is None else best[0]

    @staticmethod
    def _components_by_energy(b):
        """Known features and peaks of a fit as (center, label, LDOS curve), sorted by center energy ascending."""
        components = []
        for fi, feat in enumerate(b['feature_subset']):
            c = float(b['params'][f'f{fi}_center'])
            components.append((c, f"{feat['type']} {c:.2f} V", b['features'][fi]))
        for pi, pk in enumerate(b['peaks']):
            c = float(b['params'][f'p{pi}_center'])
            components.append((c, f"peak {pi} {c:.2f} V" + (" (manual)" if pk.get('manual') else ""),
                               b['peak_curves'][pi]))
        components.sort(key=lambda t: t[0])
        return components

    @staticmethod
    def _fit_has_bg(b):
        """True when a fit (or the analyzer's mode, for None) carries the WKB background."""
        return b is None or b.get('background', 'wkb') == 'wkb'

    def _draw_current(self):
        """
        Redraw the fit panel and the background panel for the displayed fit
        (components and legends in energy order); the background panel is
        shown only for a fit with the WKB background, and the figure takes
        the width of the panels shown (_layout).
        """
        b = self.shown_fit()
        self._layout()
        ax_m, ax_r = self._plot_axes[1], self._plot_axes[2]
        ax_m.clear()
        ax_r.clear()
        V = self.V
        with_bg = self._fit_has_bg(b)
        if b is None:
            ax_m.text(0.5, 0.5, "no feasible model", ha='center', va='center', transform=ax_m.transAxes)
            ax_m.set_title("Fit")
            ax_r.set_title(f"Background-divided {self._qlabel()}")
            ax_r.set_ylabel(f"{self._qlabel()} / background - 1")
        else:
            components = self._components_by_energy(b)
            comp_sum = sum(c for _, _, c in components) if components else np.zeros_like(V)
            to_data = b['bg'] if with_bg else 1.0

            ax_m.plot(V, self.data, color='black', lw=1.5, label='data')
            if with_bg:
                ax_m.plot(V, b['bg'], color='gray', ls='--', lw=1.5, label=f"background ({BACKGROUND['name']})")
            for _, label, curve in components:
                ax_m.plot(V, to_data * curve, lw=1.0, label=label)
            ax_m.plot(V, b['model'], color='red', lw=1.5, label='total model')
            ax_m.set_title(f"Fit: fit {b['number']} of {len(self.fits)}" + ("" if with_bg else ", no background"))
            ax_m.legend(loc='upper right', frameon=False, fontsize=8)

            if with_bg:
                ax_r.plot(V, self.data / b['bg'] - 1.0, color='black', lw=1.5, label='data / background - 1')
                ax_r.set_title(f"Background-divided {self._qlabel()}")
                ax_r.set_ylabel(f"{self._qlabel()} / background - 1")
                for _, label, curve in components:
                    ax_r.plot(V, curve, lw=1.0, label=label)
                ax_r.plot(V, comp_sum, color='red', lw=1.5, label='sum of features and peaks')
                ax_r.legend(loc='upper right', frameon=False, fontsize=8)
        ax_m.set_xlabel("Energy (eV)")
        ax_m.set_ylabel(f"{self._qlabel()} (arb. units)")
        ax_r.set_xlabel("Energy (eV)")
        self._color_bg_toggles()
        self._plot_fig.canvas.draw_idle()

    # --- figure layout: panel slots in inches, the width following the panels shown ---

    FIG_H = 6.2            # in
    PANEL_W = 4.567        # in, one panel slot (three slots make the 18 in figure)
    PANEL_GAP = 1.4        # in, between slots
    MARGIN_L = 1.1         # in, room for the y labels
    MARGIN_R = 0.4         # in
    PANEL_Y, PANEL_H = 0.30, 0.62     # figure fractions: the panels' bottom and height
    STACK_OFFSET_FRAC = 0.25          # stack: offset per curve, fraction of the largest curve range
    STACK_GAP = 0.15       # in, between the data panel and its stack
    STACK_MIN, STACK_MAX = 0.12, 0.5  # the stack's share of the data slot, limits

    def _fig_width(self, n_panels):
        return self.MARGIN_L + n_panels * self.PANEL_W + (n_panels - 1) * self.PANEL_GAP + self.MARGIN_R

    def _in(self, x, y, w, h):
        """A rect in inches as a figure-fraction rect of the current figure size."""
        W, H = self._plot_fig.get_size_inches()
        return [x / W, y / H, w / W, h / H]

    def _layout(self):
        """
        Size the figure for the panels the shown fit needs (three with the WKB
        background, two without, the background panel hidden), place the fit
        and background panels and the widget row; the data slot is split
        between the data panel and the stack by _place_data_stack.
        """
        with_bg = self._fit_has_bg(self.shown_fit())
        n = 3 if with_bg else 2
        fig = self._plot_fig
        W = self._fig_width(n)
        if abs(fig.get_size_inches()[0] - W) > 1e-6:
            fig.set_size_inches(W, self.FIG_H, forward=True)
        ax_d, ax_m, ax_r = self._plot_axes
        slot = self.PANEL_W
        ax_m.set_position(self._in(self.MARGIN_L + slot + self.PANEL_GAP, self.PANEL_Y * self.FIG_H,
                                   slot, self.PANEL_H * self.FIG_H))
        ax_r.set_visible(with_bg)
        if with_bg:
            ax_r.set_position(self._in(self.MARGIN_L + 2 * (slot + self.PANEL_GAP), self.PANEL_Y * self.FIG_H,
                                       slot, self.PANEL_H * self.FIG_H))
        self._place_data_stack(getattr(self, '_stack_share', 0.0))
        # Widget row (inches from the lower left corner).
        self._filter_controls.set_rect(self._in(0.6, 0.12, 3.4, 0.93))
        self._chk_bg.place_in(4.4, 0.72)
        self._chk_substack.place_in(6.2, 0.72)
        self._chk_feenstra.place_in(7.4, 0.72)
        W, H = fig.get_size_inches()
        self._fs.place(8.85 / W, 0.72 / H)
        for i, btn in enumerate((self._btn_revert, self._btn_forward, self._btn_edit, self._btn_save)):
            btn.place_in(4.4 + 1.6 * i, 0.22)

    def _place_data_stack(self, share):
        """
        Split the data slot: the stack takes `share` of the slot width (0: no
        stack, the data panel fills the slot), a gap, the data panel the rest.
        """
        ax_d = self._plot_axes[0]
        ax_s = self._ax_stack
        slot = self.PANEL_W
        y, h = self.PANEL_Y * self.FIG_H, self.PANEL_H * self.FIG_H
        if share <= 0:
            ax_s.set_visible(False)
            ax_d.set_position(self._in(self.MARGIN_L, y, slot, h))
            return
        w_s = share * slot
        w_d = slot - w_s - self.STACK_GAP
        ax_d.set_position(self._in(self.MARGIN_L, y, w_d, h))
        ax_s.set_position(self._in(self.MARGIN_L + w_d + self.STACK_GAP, y, w_s, h))
        ax_s.set_visible(True)

    # --- the background toggles (fit window and edit window) ---

    def _color_bg_toggles(self):
        """
        Colour the 'WKB background' toggles: blue (the edit window's
        provisional colour) while the pending mode differs from the shown
        fit's, i.e. set but not yet run; red (the current-fit colour) when the
        shown fit was run with the pending mode.
        """
        shown = self.shown_fit()
        shown_mode = shown.get('background', 'wkb') if shown is not None else self.background
        color = 'red' if self._bg_pending == shown_mode else 'blue'
        chks = [getattr(self, '_chk_bg', None)]
        st = self._edit_open()
        if st is not None:
            chks.append(st.get('chk_bg'))
        for chk in chks:
            if chk is not None:
                chk.labels[0].set_color(color)

    def _edit_open(self):
        """The edit window's state while its figure is open, else None."""
        st = getattr(self, '_edit', None)
        return st if (st is not None and plt.fignum_exists(st['fig'].number)) else None

    def _set_bg_toggle(self, chk, on):
        """Set a toggle's box without firing its callback (the widget wrapper's quiet set)."""
        if chk is not None:
            chk.set(bool(on))

    def _on_bg_toggle(self, label):
        """Either 'WKB background' toggle: the pending mode for the next Refit; both toggles agree."""
        self._bg_pending = 'none' if self._bg_pending == 'wkb' else 'wkb'
        self._set_bg_toggle(getattr(self, '_chk_bg', None), self._bg_pending == 'wkb')
        st = self._edit_open()
        if st is not None:
            self._set_bg_toggle(st.get('chk_bg'), self._bg_pending == 'wkb')
        self._color_bg_toggles()
        self._draw_left()
        if st is not None:
            st['fig'].canvas.draw_idle()

    # --- the Feenstra toggles (fit window and edit window), the same rule as the background toggles ---

    def _q_pending_of(self):
        """The quantity of the next Refit ('didv' or 'ratio'; the shown fit's until toggled)."""
        q = getattr(self, '_q_pending', None)
        if q not in self.QUANTITIES:
            b = self.shown_fit()
            q = b.get('quantity', self.quantity) if b is not None else self.quantity
        return q

    def _color_feenstra_toggles(self):
        """
        Colour the 'Feenstra (refit)' toggles: blue while the pending quantity differs
        from the shown fit's, i.e. set but not yet run; red when the shown fit
        was run with the pending quantity.
        """
        shown = self.shown_fit()
        shown_q = shown.get('quantity', self.quantity) if shown is not None else self.quantity
        color = 'red' if self._q_pending_of() == shown_q else 'blue'
        chks = [getattr(self, '_chk_feenstra', None)]
        st = self._edit_open()
        if st is not None:
            chks.append(st.get('chk_feenstra'))
        for chk in chks:
            if chk is not None:
                chk.labels[0].set_color(color)

    def _on_feenstra_toggle(self, label):
        """
        Either 'Feenstra (refit)' toggle: the quantity of the next Refit (on: the
        normalized conductance (dI/dV)/(I/V), the division by the broadened
        I/V after the calibration; off: dI/dV).  Both toggles agree.  The
        data panel shows the curve a Refit would fit.  Without raw sets the
        curve cannot be rebuilt: the toggle is put back and the reason printed.
        """
        new_q = 'didv' if self._q_pending_of() == 'ratio' else 'ratio'
        if not getattr(self, 'raw_sets', None):
            shown = self.shown_fit()
            shown_q = shown.get('quantity', self.quantity) if shown is not None else self.quantity
            if new_q != shown_q:
                print("Feenstra: this fit history holds no raw columns, so the curve cannot be rebuilt "
                      "as another quantity; the toggle stays at the shown fit's quantity.")
                self._set_bg_toggle(getattr(self, '_chk_feenstra', None), shown_q == 'ratio')
                st = self._edit_open()
                if st is not None:
                    self._set_bg_toggle(st.get('chk_feenstra'), shown_q == 'ratio')
                return
        self._q_pending = new_q
        self._set_bg_toggle(getattr(self, '_chk_feenstra', None), new_q == 'ratio')
        st = self._edit_open()
        if st is not None:
            self._set_bg_toggle(st.get('chk_feenstra'), new_q == 'ratio')
        self._color_feenstra_toggles()
        self._draw_left()
        if st is not None:
            st['fig'].canvas.draw_idle()

    def _on_stack_toggle(self, label):
        """'substack' toggle of the stack panel."""
        self._substack = bool(self._chk_substack.get_status()[0])
        self._draw_left()

    # --- widget callbacks (main figure) ---

    @staticmethod
    def _toolbar_idle(fig):
        tb = getattr(fig.canvas, 'toolbar', None)
        return tb is None or not getattr(tb, 'mode', '')

    def _sync_controls_to_shown(self):
        """
        Set the left-panel filter controls and the background toggles to the
        shown fit's settings and redraw the left panel.
        """
        b = self.shown_fit()
        if b is not None:
            for fc in self._filter_widgets():
                fc.set_params(b.get('hampel_filter_params'), b.get('sgolay_filter_params'))
            if b.get('iv_zero_sub') is not None:
                self.iv_zero_sub = bool(b['iv_zero_sub'])
            if b.get('iv_broadening') is not None:
                self.iv_broadening = float(b['iv_broadening'])
                if self.iv_broadening > 0:
                    self.iv_width_V = self.iv_broadening
            self._sync_feenstra_widgets()
        if b is not None:
            self._bg_pending = b.get('background', 'wkb')
            self._set_bg_toggle(getattr(self, '_chk_bg', None), self._bg_pending == 'wkb')
            self._q_pending = b.get('quantity', self.quantity)
            self._set_bg_toggle(getattr(self, '_chk_feenstra', None), self._q_pending == 'ratio')
            st = self._edit_open()
            if st is not None:
                self._set_bg_toggle(st.get('chk_bg'), self._bg_pending == 'wkb')
                self._set_bg_toggle(st.get('chk_feenstra'), self._q_pending == 'ratio')
            self._color_feenstra_toggles()
        self._draw_left()

    def _on_revert(self, event):
        if not self.revert_fit():
            return
        self._sync_controls_to_shown()
        self._draw_current()
        # An open edit window reloads its peaks and background from the fit now shown.
        st = getattr(self, '_edit', None)
        if st is not None and plt.fignum_exists(st['fig'].number):
            st['peaks'] = self._working_peaks(self.shown_fit())
            st['bg'] = self._working_bg(self.shown_fit())
            st['selected'] = None
            st['bg_selected'] = False
            st['drag'] = None
            st['add_armed'] = False
            self._edit_draw()

    def _on_forward(self, event):
        if not self.forward_fit():
            return
        self._sync_controls_to_shown()
        self._draw_current()
        # An open edit window reloads its peaks and background from the fit now shown.
        st = getattr(self, '_edit', None)
        if st is not None and plt.fignum_exists(st['fig'].number):
            st['peaks'] = self._working_peaks(self.shown_fit())
            st['bg'] = self._working_bg(self.shown_fit())
            st['selected'] = None
            st['bg_selected'] = False
            st['drag'] = None
            st['add_armed'] = False
            self._edit_draw()

    # --- left panel: the curve a refit would fit with the current control settings ---

    def _control_params(self):
        """(hampel, sgolay) from the figure's controls, or the shown fit's settings without a figure."""
        fc = getattr(self, '_filter_controls', None)
        if fc is not None:
            return fc.params()
        b = self.shown_fit()
        if b is not None:
            return b.get('hampel_filter_params'), b.get('sgolay_filter_params')
        return self.hampel_filter_params, self.sgolay_filter_params

    def _same_settings(self, hp, sp, source):
        """True when (hp, sp) equal the settings of a fit dict or of the analyzer (source None)."""
        if source is None:
            return (self._same_params(hp, self.hampel_filter_params)
                    and self._same_params(sp, self.sgolay_filter_params))
        return (self._same_params(hp, source.get('hampel_filter_params'))
                and self._same_params(sp, source.get('sgolay_filter_params')))

    def _settings_text(self, hp, sp):
        return f"Hampel {self._params_text(hp)}; SG {self._params_text(sp)}"

    def _display_curve(self):
        """
        The left-panel curve: the shown fit's own data when the controls and
        the pending quantity match its settings, otherwise the curve rebuilt
        with the control settings as the pending quantity.
        Returns (curve, hampel, sgolay, same_as_shown).
        """
        hp, sp = self._control_params()
        b = self.shown_fit()
        q = self._q_pending_of()
        shown_q = b.get('quantity', self.quantity) if b is not None else self.quantity
        fs_differs = self._feenstra_differs(b, q)
        if b is not None and self._same_settings(hp, sp, b) and q == shown_q and not fs_differs:
            return self.data, hp, sp, True
        if b is None and self._same_settings(hp, sp, None) and q == shown_q:
            return self.data, hp, sp, True
        try:
            curve, _, _ = self._build_data(hp, sp, q)
        except ValueError as exc:
            print(f"Filter controls: {exc}")
            return self.data, hp, sp, True
        return curve, hp, sp, False

    def _draw_left(self):
        """
        Redraw the data panel from the control settings and the pending
        quantity, and beside it the stack of the raw sets' curves
        (_draw_stack) when the curve has raw sets; the two share the data
        slot by the aspect rule (see _draw_stack).
        """
        if getattr(self, '_plot_axes', None) is None:
            return
        ax = self._plot_axes[0]
        ax.clear()
        curve, hp, sp, same = self._display_curve()
        b = self.shown_fit()
        q = self._q_pending_of()
        shown_q = b.get('quantity', self.quantity) if b is not None else self.quantity
        q_differs = q != shown_q
        ax.plot(self.V, curve, color='black', lw=1.5,
                label=self._settings_text(hp, sp) + (f"; {self._qname(q)}" if q_differs else ""))
        if not same and b is not None:
            ax.plot(self.V, self.data, color='gray', ls='--', lw=1.0,
                    label=f"fit {b['number']} curve ("
                          + self._settings_text(b.get('hampel_filter_params'), b.get('sgolay_filter_params'))
                          + (f"; {self._qname(shown_q)}" if q_differs else "") + ")")
        shown_mode = b.get('background', 'wkb') if b is not None else self.background
        bg_differs = getattr(self, '_bg_pending', shown_mode) != shown_mode
        settings_differ = not self._same_settings(hp, sp, b)
        fs_differs = self._feenstra_differs(b, q)
        if same and not bg_differs:
            title = self._qlabel()
        else:
            what = " and ".join(t for t, on in (("controls", settings_differ),
                                                 ("quantity", q_differs),
                                                 ("Feenstra settings", fs_differs),
                                                 ("background", bg_differs)) if on)
            title = f"{self._qname(q)}: {what} differ from the shown fit; Refit uses them"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel(f"{self._qname(q)} (arb. units)")
        ax.legend(loc='upper right', frameon=False, fontsize=8)
        self._draw_stack(hp, sp)
        self._color_setting_controls()
        self._plot_fig.canvas.draw_idle()

    # --- the stack panel beside the data panel ---

    def _draw_stack(self, hp, sp):
        """
        The stack beside the data panel: the raw sets' curves at the control
        settings (hp, sp) as the pending quantity, one colour per set (file), offset vertically
        (STACK_OFFSET_FRAC of the largest curve range per step, the first
        set's first curve at the top) with the curve they average to at the
        bottom in black.  'substack' off: one curve per set, its plain mean;
        on: every sweep.  The data slot is shared:
        the stack's width is such that one curve's range in it has the same
        height-to-width ratio as the data panel's curve, w_stack = slot x r /
        (1 + r) with r = range / stack span, kept within STACK_MIN..STACK_MAX
        of the slot.  No raw sets: no stack, the data panel fills the slot.
        """
        ax = self._ax_stack
        ax.clear()
        if not getattr(self, 'raw_sets', None):
            self._stack_share = 0.0
            self._place_data_stack(0.0)
            return
        try:
            cols, _, _ = self._raw_set_columns(hp, sp, self._q_pending_of())
        except ValueError as exc:
            print(f"Stack: {exc}")
            self._stack_share = 0.0
            self._place_data_stack(0.0)
            return
        avg, centres = averaging.Averages.average(cols, getattr(self, '_set_weights', None))
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        curves = []                                   # (set index, curve)
        for k, (a, c) in enumerate(zip(cols, centres)):
            if self._substack:
                for j in range(a.shape[1]):
                    curves.append((k, a[:, j]))
            else:
                curves.append((k, c))
        max_range = max([float(np.ptp(c)) for _, c in curves] + [float(np.ptp(avg))])
        offset = self.STACK_OFFSET_FRAC * max_range
        n_slots = len(curves) + 1
        yspan = (n_slots - 1) * offset + max_range
        r = max_range / yspan if yspan > 0 else 1.0
        self._stack_share = float(np.clip(r / (1.0 + r), self.STACK_MIN, self.STACK_MAX))
        self._place_data_stack(self._stack_share)
        labelled = set()
        lo, hi = float(np.min(avg)), float(np.max(avg))
        for i, (k, c) in enumerate(curves):
            off = (n_slots - 1 - i) * offset
            rgb = colors[k % len(colors)]
            label = None
            if k not in labelled:
                labelled.add(k)
                label = f"set {k + 1} ({cols[k].shape[1]} sweeps)"
            ax.plot(self.V, c + off, color=rgb, lw=0.9, label=label)
            lo, hi = min(lo, float(np.min(c)) + off), max(hi, float(np.max(c)) + off)
        ax.plot(self.V, avg, color='black', lw=1.4, label='average')
        ax.set_xlim(float(self.V[0]), float(self.V[-1]))
        pad = 0.03 * (hi - lo if hi > lo else 1.0)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_yticks([])
        ax.set_xlabel("Energy (eV)")
        how = "sweeps" if self._substack else "per-set means"
        ax.set_title(f"Stack: {how}", fontsize=8)
        if len(cols) > 1:
            ax.legend(loc='upper right', frameon=False, fontsize=6)

    def _on_filter_change(self):
        """The fit window's filter controls changed: the edit window's follow (without firing) and the data panel redraws."""
        st = self._edit_open()
        fc = getattr(self, '_filter_controls', None)
        if st is not None and st.get('filters') is not None and fc is not None:
            hp, sp = fc.params()
            st['filters'].set_params(hp, sp)
        try:
            self._draw_left()
        except Exception as exc:
            print(f"Filter controls: {exc}")

    def _on_save(self, event):
        self.save_fit()

    # --- edit figure ---

    @staticmethod
    def _working_peaks(b):
        """Editable copy of a fit's peaks (center, height, per-side half width, and Lorentzian share, holds)."""
        return [{'shape': p['shape'], 'center': float(p['center']), 'amp_seed': float(p['amp_seed']),
                 'hwhm': float(p['hwhm']), 'hwhm_right': float(p.get('hwhm_right', p['hwhm'])),
                 'mix': float(p.get('mix', 0.5)), 'mix_right': float(p.get('mix_right', p.get('mix', 0.5))),
                 'manual': bool(p.get('manual', False)),
                 'hold': sorted(set(p.get('hold', ())))} for p in b['peaks']]

    def _working_bg(self, b):
        """
        Editable copy of a fit's background (amp, phi, kappa, and the names
        held), or None for a fit without the WKB background.
        """
        if b is None or not self._fit_has_bg(b):
            return None
        bg = {name: float(b['params'][name]) for name, _ in self.BG_PARAMS if name in b['params']}
        bg['hold'] = sorted(set(b.get('bg_hold', {}).keys()))
        return bg

    @staticmethod
    def _hold_text(names, params):
        """Short list of the held parameters' panel names for a legend entry, '' when none."""
        held = [label for key, label in params if key in names]
        return f" [held: {', '.join(held)}]" if held else ""

    # Edit window geometry (figure fractions): the plot, and right of it the
    # parameter panel of the selected curve, as tall as the plot; panel rows from
    # the top at a compact pitch, three columns: hold toggle, name, text box.
    EDIT_FIGSIZE = (14.5, 6.5)
    EDIT_PLOT = [0.055, 0.30, 0.60, 0.61]            # left, bottom, width, height (the settings band under it)
    FEENSTRA_WIDTH_DEFAULT_V = 0.2                   # the width box when the fit holds no broadening (plain I/V)
    EDIT_ROW_H, EDIT_ROW_PITCH = 0.05, 0.072
    EDIT_COL_CHK, EDIT_COL_NAME, EDIT_COL_BOX = 0.70, 0.745, 0.865   # column x positions
    EDIT_CHK_W, EDIT_BOX_W = 0.03, 0.115
    EDIT_FONTSIZE = 8

    def _open_edit_figure(self, event=None):
        """
        Second window for editing every peak of the displayed fit.  Click a
        peak to select it (drawn dashed, thick, and orange; the others solid);
        click elsewhere to move the selected peak's center and height there;
        drag from within two energy steps of the selected peak's center to set
        that side's half width (x) and Gaussian/Lorentzian character (y);
        right-click a peak to remove it; 'Add peak' arms the next press-drag
        to draw a new peak; 'Refit' runs the joint fit from the edited peaks,
        appends an accepted result to the fit history, and updates the main
        figure.  Right of the plot, the parameter panel of the selected curve
        (see _edit_panel_refresh): one row per parameter with a hold toggle,
        the name, and an editable box; a submitted value sets that parameter
        (the provisional curve follows), a hold keeps it at the shown value in
        the next Refit.  'edit background' (shown while the displayed fit has
        the WKB background) selects the background, whose amp, phi, and kappa
        are edited and held the same way.  The 'WKB background' and
        'Feenstra (refit)' toggles are the fit window's (pending mode and quantity of
        the next Refit, always in step), as are the Feenstra settings
        ('0Vsub', 'broadened', width) and the Hampel / Savitzky-Golay
        controls in the band under the plot (the same colours: blue while a
        setting differs from the shown fit's, red once a fit was run with it).
        """
        b = self.shown_fit()
        if b is None:
            print("No fit to edit.")
            return
        old = getattr(self, '_edit', None)
        if old is not None:
            try:
                plt.close(old['fig'])
            except Exception:
                pass
        fig = plt.figure(figsize=self.EDIT_FIGSIZE)
        ax = fig.add_axes(self.EDIT_PLOT)
        st = {'fig': fig, 'ax': ax, 'peaks': self._working_peaks(b), 'bg': self._working_bg(b),
              'selected': None, 'bg_selected': False, 'drag': None, 'add_armed': False,
              'quiet': False}
        # Bottom band: the two toggles, 'Add peak', 'Refit', 'edit background'.
        # The controls through widgets.py (the same specifications as every window's).
        st['chk_bg'] = widgets.make(fig, 'WKB background', 0.055, 0.06, value=self._bg_pending == 'wkb')
        st['ax_bg'] = st['chk_bg'].ax
        st['chk_bg'].on_change(lambda flag: self._on_bg_toggle(None))
        st['chk_feenstra'] = widgets.make(fig, 'Feenstra (refit)', 0.185, 0.06, value=self._q_pending_of() == 'ratio')
        st['ax_feenstra'] = st['chk_feenstra'].ax
        st['chk_feenstra'].on_change(lambda flag: self._on_feenstra_toggle(None))
        st['btn_add'] = widgets.make(fig, 'Add peak', 0.31, 0.055)
        st['ax_add'] = st['btn_add'].ax
        st['btn_add'].on_change(lambda: self._edit_on_add(None))
        st['btn_refit'] = widgets.make(fig, 'Refit', 0.43, 0.055)
        st['ax_refit'] = st['btn_refit'].ax
        st['btn_refit'].on_change(lambda: self._edit_on_refit(None))
        st['btn_bg_edit'] = widgets.make(fig, 'edit background', 0.55, 0.055)
        st['ax_bg_edit'] = st['btn_bg_edit'].ax
        st['btn_bg_edit'].on_change(lambda: self._edit_on_bg_select(None))
        # The settings band between the plot and the buttons: the Feenstra settings and the Hampel /
        # Savitzky-Golay controls of the next Refit, in step with the fit window's (same colours).
        st['fs'] = widgets.FeenstraSettings(fig, 0.055, 0.14, self._fs_settings())
        st['fs'].on_change(self._on_fs_change)
        hp0, sp0 = self._control_params()
        st['filters'] = widgets.FilterControls(fig, [0.40, 0.11, 0.25, 0.11], hampel=hp0, sgolay=sp0,
                                               hampel_on=hp0 is not None, sgolay_on=sp0 is not None)
        st['filters'].on_change(self._on_edit_filter_change)
        # Parameter panel: as many rows as the longest parameter list, the word
        # 'hold' once above the first toggle.
        n_rows = max(len(self.PEAK_PARAMS), len(self.BG_PARAMS))
        top = self.EDIT_PLOT[1] + self.EDIT_PLOT[3]
        fig.text(self.EDIT_COL_CHK + 0.5 * self.EDIT_CHK_W, top - 0.01, 'hold',
                 fontsize=self.EDIT_FONTSIZE, ha='center', va='bottom')
        st['rows'] = []
        for k in range(n_rows):
            y = top - 0.03 - self.EDIT_ROW_H - k * self.EDIT_ROW_PITCH
            chk = widgets.make(fig, 'hold', self.EDIT_COL_CHK, y, value=False)
            ax_chk = chk.ax
            ax_chk.set_frame_on(False)
            chk.on_change(lambda flag, k=k: self._edit_on_hold(k))
            name = fig.text(self.EDIT_COL_NAME, y + 0.5 * self.EDIT_ROW_H, '',
                            fontsize=self.EDIT_FONTSIZE, ha='left', va='center')
            box = widgets.make(fig, 'param', self.EDIT_COL_BOX, y, value='')
            ax_box = box.ax
            box.on_change(lambda text, k=k: self._edit_on_param(k, text))
            st['rows'].append({'ax_chk': ax_chk, 'chk': chk, 'name': name, 'ax_box': ax_box, 'box': box})
        fig.canvas.mpl_connect('button_press_event', self._edit_on_press)
        fig.canvas.mpl_connect('motion_notify_event', self._edit_on_motion)
        fig.canvas.mpl_connect('button_release_event', self._edit_on_release)
        self._edit = st
        self._color_bg_toggles()
        self._color_feenstra_toggles()
        self._color_setting_controls()
        self._edit_draw()
        plt.show()

    # --- the parameter panel of the edit window ---

    def _edit_current(self):
        """
        The selected curve of the edit window: ('bg', working background),
        ('peak', peak dict) or (None, None) when nothing is selected.
        """
        st = self._edit
        if st.get('bg_selected') and st.get('bg') is not None:
            return 'bg', st['bg']
        sel = st.get('selected')
        if sel is not None and sel < len(st['peaks']):
            return 'peak', st['peaks'][sel]
        return None, None

    @staticmethod
    def _peak_field(g, key):
        """Value of a holdable peak parameter from a peak dict ('amp' is 'amp_seed')."""
        if key == 'amp':
            return float(g['amp_seed'])
        if key == 'hwhm_right':
            return float(g.get('hwhm_right', g['hwhm']))
        if key == 'mix_right':
            return float(g.get('mix_right', g.get('mix', 0.5)))
        if key == 'mix':
            return float(g.get('mix', 0.5))
        return float(g[key])

    @staticmethod
    def _fmt_param(v):
        return f"{v:.6g}"

    def _edit_panel_refresh(self):
        """
        Fill the parameter panel for the selected curve: the rows of its
        parameters (toggle at the hold state, name, box at the value), the
        remaining rows hidden; nothing selected: every row blank and inactive.
        The 'edit background' button is shown while the displayed fit has the
        WKB background.
        """
        st = self._edit
        kind, cur = self._edit_current()
        if kind == 'bg':
            params = [(name, label, float(cur[name])) for name, label in self.BG_PARAMS if cur.get(name) is not None]
        elif kind == 'peak':
            params = [(key, label, self._peak_field(cur, key)) for key, label in self.PEAK_PARAMS]
        else:
            params = []
        held = set(cur.get('hold', ())) if cur is not None else set()
        st['quiet'] = True
        try:
            for k, row in enumerate(st['rows']):
                shown = k < len(params)
                row['ax_chk'].set_visible(shown)
                row['ax_box'].set_visible(shown)
                row['name'].set_visible(shown)
                # Widget.active (CheckButtons.set_active is the box toggler, by index).
                row['chk'].enable(shown)
                row['box'].enable(shown)
                chk = row['chk']
                chk.widget.drawon = False                    # one draw at the end of _edit_draw instead of one per box
                try:
                    if not shown:
                        row['name'].set_text('')
                        row['box'].set('')
                        chk.set(False)
                        continue
                    key, label, value = params[k]
                    row['name'].set_text(label)
                    row['box'].set(self._fmt_param(value))
                    chk.set(key in held)
                finally:
                    chk.widget.drawon = True
        finally:
            st['quiet'] = False
        with_bg = self._fit_has_bg(self.shown_fit())
        st['ax_bg_edit'].set_visible(with_bg)
        st['btn_bg_edit'].enable(with_bg)

    def _edit_param_keys(self):
        """The parameter keys of the panel rows for the selected curve (in row order)."""
        kind, cur = self._edit_current()
        if kind == 'bg':
            return [name for name, _ in self.BG_PARAMS if cur.get(name) is not None]
        if kind == 'peak':
            return [key for key, _ in self.PEAK_PARAMS]
        return []

    def _edit_on_hold(self, k):
        """Row k's hold toggle: hold or free that parameter of the selected curve."""
        st = getattr(self, '_edit', None)
        if st is None or st.get('quiet'):
            return
        keys = self._edit_param_keys()
        kind, cur = self._edit_current()
        if cur is None or k >= len(keys):
            return
        key = keys[k]
        held = set(cur.get('hold', ()))
        if key in held:
            held.discard(key)
        else:
            held.add(key)
        cur['hold'] = sorted(held)
        what = "background" if kind == 'bg' else f"peak {st['selected']}"
        print(f"{what}: {key} {'held' if key in held else 'free'}"
              + (f" at {self._fmt_param(cur[key] if kind == 'bg' else self._peak_field(cur, key))}" if key in held else ""))
        self._edit_draw()

    def _edit_on_param(self, k, text):
        """
        Row k's box: set that parameter of the selected curve to the entered
        value (a value that does not parse, or breaks the parameter's own
        range, is reported and the box reset).  A held parameter is held at
        the new value.
        """
        st = getattr(self, '_edit', None)
        if st is None or st.get('quiet'):
            return
        keys = self._edit_param_keys()
        kind, cur = self._edit_current()
        if cur is None or k >= len(keys):
            return
        key = keys[k]
        try:
            v = float(text.strip())
            if not np.isfinite(v):
                raise ValueError("not a finite number")
            if kind == 'bg':
                if key == 'bg_amp' and v <= 0:
                    raise ValueError("the background amplitude must be > 0")
                if key == 'bg_phi_eV' and v <= 0.5 * float(np.max(np.abs(self.V))):
                    raise ValueError(f"phi must exceed half the largest |V| of the window "
                                     f"({0.5 * float(np.max(np.abs(self.V))):.3f} eV)")
                if key == 'bg_kappa' and v <= 0:
                    raise ValueError("kappa must be > 0")
                cur[key] = v
            else:
                if key == 'center' and not (float(self.V[0]) <= v <= float(self.V[-1])):
                    raise ValueError(f"the center must lie in the window {self.V[0]:.3f} to {self.V[-1]:.3f} V")
                if key == 'amp' and v <= 0:
                    raise ValueError("the height must be > 0")
                if key in ('hwhm', 'hwhm_right') and v < self._dV():
                    raise ValueError(f"a half width must be at least one energy step ({self._dV():.4f} V)")
                if key in ('mix', 'mix_right') and not (0.0 <= v <= 1.0):
                    raise ValueError("a Lorentzian share lies between 0 and 1")
                cur['amp_seed' if key == 'amp' else key] = v
        except ValueError as exc:
            print(f"Parameter panel: '{text}' not accepted for {key}: {exc}")
            self._edit_panel_refresh()
            return
        what = "background" if kind == 'bg' else f"peak {st['selected']}"
        print(f"{what}: {key} set to {self._fmt_param(v)}"
              + (" (held)" if key in set(cur.get('hold', ())) else ""))
        self._edit_draw()

    def _edit_on_bg_select(self, event):
        """'edit background': the background becomes the selected curve (peaks deselected)."""
        st = getattr(self, '_edit', None)
        if st is None or st.get('bg') is None:
            return
        st['bg_selected'] = True
        st['selected'] = None
        st['drag'] = None
        print("Background selected: " + ", ".join(f"{label} {self._fmt_param(st['bg'][name])}"
                                                   for name, label in self.BG_PARAMS if st['bg'].get(name) is not None))
        self._edit_draw()

    def _edit_draw(self):
        """
        Redraw the edit window: the background-divided data (the data itself
        for a fit without background), the known features, the working peaks
        in order of center energy (labels carry their list index, which the
        next accepted refit renumbers by energy), the current fit and the
        provisional sum of the edited peaks.
        """
        st = self._edit
        b = self.shown_fit()
        ax = st['ax']
        ax.clear()
        V = self.V
        with_bg = self._fit_has_bg(b)
        if with_bg:
            ax.plot(V, self.data / b['bg'] - 1.0, color='black', lw=1.5, label='data / background - 1')
        else:
            ax.plot(V, self.data, color='black', lw=1.5, label='data (no background)')
        for fi, feat in enumerate(b['feature_subset']):
            ax.plot(V, b['features'][fi], color='gray', lw=1.0, label=f"{feat['type']} (known feature)")
        curves = [self._pending_curve(V, g) for g in st['peaks']]
        for i in sorted(range(len(st['peaks'])), key=lambda j: (st['peaks'][j]['center'], j)):
            g, curve = st['peaks'][i], curves[i]
            lab = (f"peak {i} {g['center']:.2f} V" + (" (manual)" if g.get('manual') else "")
                   + self._hold_text(set(g.get('hold', ())), self.PEAK_PARAMS))
            if i == st['selected']:
                ax.plot(V, curve, color='orange', ls='--', lw=2.5, label=lab + " [selected]")
            else:
                ax.plot(V, curve, lw=1.0, label=lab)
        fitted = sum(list(b['features']) + list(b['peak_curves'])) if (b['features'] or b['peak_curves']) else np.zeros_like(V)
        ax.plot(V, fitted, color='red', lw=1.2, label='current fit (features + peaks)')
        prov = sum(list(b['features']) + curves) if (b['features'] or curves) else np.zeros_like(V)
        ax.plot(V, prov, color='blue', lw=1.5, label='provisional (edited peaks)')
        armed = "  [Add peak armed: press and drag on the plot]" if st['add_armed'] else ""
        bg_sel = ""
        if st.get('bg_selected') and st.get('bg') is not None:
            bg_sel = "  [background selected" + self._hold_text(set(st['bg'].get('hold', ())), self.BG_PARAMS) + "]"
        ax.set_title(f"Peak editor: fit {b['number']} of {len(self.fits)}, click a peak to select, "
                     f"click elsewhere to move it, drag from its center for widths (x) and G/L (y), "
                     f"right-click to remove{armed}{bg_sel}", fontsize=9)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel(f"{self._qlabel()} / background - 1" if with_bg else f"{self._qlabel()} (arb. units)")
        ax.legend(loc='upper right', frameon=False, fontsize=8)
        self._edit_panel_refresh()
        st['fig'].canvas.draw_idle()

    def _edit_on_press(self, event):
        st = getattr(self, '_edit', None)
        if st is None or event.inaxes is not st['ax'] or not self._toolbar_idle(st['fig']):
            return
        x, y = float(event.xdata), float(event.ydata)
        dV = self._dV()
        peaks = st['peaks']
        if event.button == 3:
            i = self._hit(peaks, x)
            if i is not None:
                g = peaks.pop(i)
                if st['selected'] == i:
                    st['selected'] = None
                elif st['selected'] is not None and st['selected'] > i:
                    st['selected'] -= 1
                print(f"Peak removed from the edit set: {self._guess_text(g)}")
                self._edit_draw()
            return
        if event.button != 1:
            return
        if st['add_armed']:
            peaks.append(self._new_guess(x, y, dV))
            st['selected'] = len(peaks) - 1
            st['bg_selected'] = False
            st['drag'] = {'kind': 'new', 'index': st['selected'], 'y0': y}
            st['add_armed'] = False
            self._edit_draw()
            return
        sel = st['selected']
        if sel is not None and abs(x - peaks[sel]['center']) <= 2.0 * dV:
            g = peaks[sel]
            st['drag'] = {'kind': 'adjust', 'index': sel, 'y0': y, 'mix0': (g['mix'], g['mix_right'])}
            return
        i = self._hit(peaks, x)
        if i is not None:
            st['selected'] = i
            st['bg_selected'] = False
            print(f"Selected peak {i}: {self._guess_text(peaks[i])}")
            self._edit_draw()
            return
        if sel is not None:
            g = peaks[sel]
            g['center'] = x
            g['amp_seed'] = max(y, 1e-12)
            print(f"Peak {sel} moved: {self._guess_text(g)}")
            self._edit_draw()

    def _edit_on_motion(self, event):
        st = getattr(self, '_edit', None)
        if st is None or st['drag'] is None or event.inaxes is not st['ax'] or event.xdata is None:
            return
        self._apply_drag(st['peaks'][st['drag']['index']], st['drag'], event.xdata, event.ydata, self._dV())
        self._edit_draw()

    def _edit_on_release(self, event):
        st = getattr(self, '_edit', None)
        if st is None or st['drag'] is None:
            return
        drag = st['drag']
        st['drag'] = None
        g = st['peaks'][drag['index']]
        action = "Peak added to the edit set" if drag['kind'] == 'new' else f"Peak {drag['index']} adjusted"
        print(f"{action}: {self._guess_text(g)}")
        self._edit_draw()

    def _edit_on_add(self, event):
        st = getattr(self, '_edit', None)
        if st is None:
            return
        st['add_armed'] = True
        print("Add peak: press at the new peak's center and height on the edit plot, drag outward for its widths.")
        self._edit_draw()

    def _edit_on_refit(self, event):
        st = getattr(self, '_edit', None)
        if st is None:
            return
        peaks = [dict(g) for g in st['peaks']]
        bg = dict(st['bg']) if st.get('bg') is not None else None
        # Smoothing from the main figure's controls, the pending background mode, and
        # the pending quantity: when any differs from the shown fit's, the data
        # context (curve, anchor, ceiling, or none) is rebuilt for this refit.
        shown = self.shown_fit()
        hp, sp = self._control_params()
        mode = self._bg_pending
        q = self._q_pending_of()
        shown_q = shown.get('quantity', self.quantity)
        changed = (not self._same_settings(hp, sp, shown) or mode != shown.get('background', 'wkb')
                   or q != shown_q or self._feenstra_differs(shown, q))
        if changed:
            try:
                self._rebuild(hp, sp, mode, q)
            except Exception as exc:
                print(f"Edit refit: could not rebuild the curve with the control settings ({exc}).")
                self._restore_context(shown)
                self._sync_feenstra_widgets()
                return
            print(f"Edit refit with {self._settings_text(hp, sp)}, background {mode}, {self._qname(q)}"
                  + (f", {self._feenstra_text()}" if q == 'ratio' else "")
                  + f" (shown fit {shown['number']}: "
                  + self._settings_text(shown.get('hampel_filter_params'), shown.get('sgolay_filter_params'))
                  + f", background {shown.get('background', 'wkb')}, {self._qname(shown_q)}); "
                  + ("ceiling recomputed." if mode == 'wkb' else "no ceiling."))
        held = [f"peak {i}: {', '.join(g['hold'])}" for i, g in enumerate(peaks) if g.get('hold')]
        if bg is not None and bg.get('hold') and mode == 'wkb':
            held.append("background: " + ", ".join(bg['hold']))
        if held:
            print("Edit refit holds: " + "; ".join(held))
        t0 = time.perf_counter()
        fit = self._refit_peaks(peaks, bg)
        seconds = time.perf_counter() - t0
        if fit is None:
            print(f"Edit refit rejected ({seconds:.1f} s): {self._last_reject}")
            if changed:
                self._restore_context(shown)
                self._sync_feenstra_widgets()
            return
        print(f"Edit refit accepted ({seconds:.1f} s), peaks renumbered by energy:")
        self._accept_fit(fit)
        self._print_model(fit)
        st['peaks'] = self._working_peaks(fit)
        st['bg'] = self._working_bg(fit)
        st['selected'] = None
        st['bg_selected'] = False
        self._color_feenstra_toggles()
        self._draw_left()
        self._draw_current()
        self._edit_draw()

    def plot(self):
        """
        The fit window.  Panels: the data (with the stack of the raw sets'
        curves beside it, see _draw_stack); the fit ('Fit N of M'); the
        background-divided data with the components, shown only for a fit
        with the WKB background (the window is as wide as the panels shown).
        Widget row: the filter controls; the 'WKB background' toggle (the mode
        of the next Refit; blue while it differs from the shown fit's mode,
        red once a fit was run with it); the stack's 'substack' toggle; the
        'Feenstra (refit)' toggle (the quantity of the next Refit, (dI/dV)/(I/V) on
        or dI/dV off, coloured by the same rule; the data panel shows the
        curve a Refit would fit); 'Revert fit' and 'Forward fit' (walk the fit
        history, see the module docstring), 'Edit peaks' (second window for
        every fitted peak, with the parameter panel and the background edit),
        'Save fit' (JSON + CSV of the displayed fit).
        """
        fig = plt.figure(figsize=(self._fig_width(3), self.FIG_H))
        self._plot_fig = fig
        self._plot_axes = (fig.add_axes([0.1, 0.3, 0.2, 0.6]), fig.add_axes([0.4, 0.3, 0.2, 0.6]),
                           fig.add_axes([0.7, 0.3, 0.2, 0.6]))        # placed by _layout
        self._ax_stack = fig.add_axes([0.3, 0.3, 0.05, 0.6])
        self._stack_share = 0.0
        self._substack = False
        self._edit = None

        # Filter controls for the data panel, starting at the shown fit's settings
        # (a dict turns the filter on, None leaves it off); the background toggle at
        # the shown fit's mode.
        b = self.shown_fit()
        hp0 = b.get('hampel_filter_params') if b is not None else self.hampel_filter_params
        sp0 = b.get('sgolay_filter_params') if b is not None else self.sgolay_filter_params
        self._bg_pending = b.get('background', 'wkb') if b is not None else self.background
        self._filter_controls = widgets.FilterControls(fig, [0.04, 0.02, 0.19, 0.15],
                                               hampel=hp0, sgolay=sp0,
                                               hampel_on=hp0 is not None, sgolay_on=sp0 is not None)
        self._filter_controls.on_change(self._on_filter_change)
        # The controls through widgets.py (the same specifications as every window's); placed by _layout.
        self._chk_bg = widgets.make(fig, 'WKB background', 0.25, 0.1, value=self._bg_pending == 'wkb')
        self._chk_bg.on_change(lambda flag: self._on_bg_toggle(None))
        self._chk_substack = widgets.make(fig, 'substack', 0.36, 0.1, value=self._substack)
        self._chk_substack.on_change(lambda flag: self._on_stack_toggle(None))
        # The 'Feenstra (refit)' toggle: the quantity of the next Refit, starting at the shown fit's
        # (blue while it differs from the shown fit's, red once a fit was run with it).
        self._q_pending = b.get('quantity', self.quantity) if b is not None else self.quantity
        self._chk_feenstra = widgets.make(fig, 'Feenstra (refit)', 0.43, 0.1, value=self._q_pending == 'ratio')
        self._chk_feenstra.on_change(lambda flag: self._on_feenstra_toggle(None))
        # The Feenstra settings of the next Refit as the ratio ('0Vsub', 'broadened', width), starting at the
        # analyzer's (the sender's at the send, the shown fit's after a revert or forward); placed by _layout.
        self._fs = widgets.FeenstraSettings(fig, 0.7, 0.1, self._fs_settings())
        self._fs.on_change(self._on_fs_change)

        # Buttons (kept as attributes so they are not garbage-collected); placed by _layout.
        self._btn_revert = widgets.make(fig, 'Revert fit', 0.30, 0.04)
        self._btn_revert.on_change(lambda: self._on_revert(None))
        self._btn_forward = widgets.make(fig, 'Forward fit', 0.42, 0.04)
        self._btn_forward.on_change(lambda: self._on_forward(None))
        self._btn_edit = widgets.make(fig, 'Edit peaks', 0.54, 0.04)
        self._btn_edit.on_change(lambda: self._open_edit_figure(None))
        self._btn_save = widgets.make(fig, 'Save fit', 0.66, 0.04)
        self._btn_save.on_change(lambda: self._on_save(None))

        self._draw_current()          # sizes and places everything (_layout), then the fit panels
        self._draw_left()             # the data panel and the stack
        plt.show()


# Sample run code:
if __name__ == '__main__':
    #ifile = 'experiment_dir/file.sm4'
    ifile = 'experiment_dir/file.sm4'
    #ifile = 'experiment_dir/file.sm4'
    #ifile = 'experiment_dir/file.sm4'


    hampel_filter_params = {'window': 3, 'n_sigma': 2}
    sgolay_filter_params = {'window_length': 6, 'polyorder': 3}

    # Known modeled dI/dV features (the background is always the 'wkb' model, see
    # the module docstring).  Each entry:
    #   {'type': 'asym_voigt' | 'erf_step',
    #    'center': <V>, 'fix_center': True/False,
    #    widths: 'hwhm' and 'hwhm_right' (asym_voigt, split into equal Gaussian and
    #            Lorentzian parts), or the parameter names themselves ('sigma_left',
    #            'gamma_left', 'sigma_right', 'gamma_right'; 'width' for erf_step),
    #            or 'width' as one value for every width parameter;
    #    'fix_width': True/False (all widths) or 'fix_widths': [names],
    #    'bounds': {'amp': (0, None), 'center': (lo, hi), <width name>: (lo, hi)},
    #    'direction': +1 or -1}                                             # erf_step only
    known_features = [
        # {'type': 'asym_voigt', 'center': -1.70, 'fix_center': True, 'hwhm': 0.05, 'fix_width': False},
    ]

    # Worker processes for the fits: 1 = serial, -1 = all CPU cores (needs: pip install loky).
    n_jobs = -1

    # Cap on width_long / width_short for every asymmetric peak (features and generic peaks).
    max_width_ratio = 3

    # Candidate threshold = noise_z x noise level (successive-difference sigma x SG gain).
    noise_z = 2.5

    # Fit history cache: True opens the figure from <date_time>_fits.pkl next to the
    # .sm4 when that file exists (and is of the current format); False runs a fresh
    # sweep and overwrites the cache.
    use_cache = True

    if use_cache and Fit.has_cache(ifile):
        analyzer = Fit.from_cache(Fit.cache_path(ifile))
    else:
        analyzer = Fit(ifile,
                                     hampel_filter_params=hampel_filter_params,
                                     sgolay_filter_params=sgolay_filter_params)
        analyzer.process()
        analyzer.select_model(known_features=known_features,
                              n_jobs=n_jobs,
                              max_width_ratio=max_width_ratio,
                              noise_z=noise_z)
    analyzer.plot()

