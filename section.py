# -*- coding: utf-8 -*-
"""
The section features of the browser's path (the 'section' button and the
three feature buttons stacked beside it; the path itself is
widgets.DragPath on the scan area):

  cross section : the raw topography sampled along the whole path into the
                  profile window (topo.cross_section, topo.ProfileWindow;
                  the browser's _on_cross_section).
  measure       : MeasureWindow, every point of the path with its
                  coordinates, every segment with its length and the total
                  length, each both nominal and calibrated.
  set scale     : SetScaleWindow, the path's nominal length, the scale
                  factor and the calibrated length; editing the scale factor
                  recomputes the calibrated length, editing the calibrated
                  length recomputes the scale factor; 'apply' sets the
                  browser's scale timeline from the current file's time on
                  (browser.Browser.set_scale_from_current).

Nominal and calibrated.  The scan area's drawing units are calibrated
nanometres: recorded metres divided by the scale factor in force at the
file's time (scales.Scales).  So for the path's points, p_cal = the drawing
coordinates, p_nom = p_cal x scale; the same for every length.  A measured
distance of a known feature therefore gives the scale factor directly:
scale = nominal length / true length, which is what the set-scale window
does when its calibrated length is edited.

Both windows follow the path while they are open (update): a point added,
moved or removed changes their numbers at once.  Using a feature ends the
path's editing (the browser's rule): the clicks no longer add or remove
points until 'section' is pressed again.
"""

import numpy as np

from sm4_Browser import qtplot as plt


def path_lengths(points):
    """(segment lengths, total) of a polyline (n, 2) in its own units; no segments without two points."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return np.zeros(0), 0.0
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return seg, float(seg.sum())


def measure_lines(points, scale, name=''):
    """
    The measure window's text, one line per point and per segment, nominal and calibrated:
    `points` (n, 2) in calibrated nm (the drawing units), `scale` the file's scale factor.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    s = float(scale)
    lines = [f"{name}: scale factor {s:g}".strip(': '),
             f"nominal = recorded (nm); calibrated = nominal / {s:g} (nm)",
             "", f"{'point':>5}  {'x nominal':>12} {'y nominal':>12}  | {'x calibrated':>12} {'y calibrated':>12}"]
    for i, p in enumerate(pts, 1):
        lines.append(f"{i:>5}  {p[0] * s:>12.3f} {p[1] * s:>12.3f}  | {p[0]:>12.3f} {p[1]:>12.3f}")
    seg, total = path_lengths(pts)
    if len(seg):
        lines += ["", f"{'segment':>7}  {'nominal (nm)':>14}  | {'calibrated (nm)':>16}"]
        for i, L in enumerate(seg, 1):
            lines.append(f"{i:>3}-{i + 1:<3}  {L * s:>14.3f}  | {L:>16.3f}")
        lines.append(f"{'total':>7}  {total * s:>14.3f}  | {total:>16.3f}")
    else:
        lines += ["", "one point: no segment yet"]
    return lines


class MeasureWindow:
    """The measure window: the path's points and lengths as text (measure_lines); update(points, scale, name) follows the path."""

    DESIGN = (7.5, 4.2)    # the design size (inches): the text zoom of the window is its size relative to this (qtplot.window_zoom)

    def __init__(self, points, scale, name=''):
        self.fig = plt.figure(figsize=self.DESIGN)
        try:
            self.fig.canvas.manager.set_window_title("measure: the section path")
        except Exception:
            pass
        self.txt = self.fig.text(0.03, 0.96, '', fontsize=8, family='monospace', ha='left', va='top')
        self.update(points, scale, name)
        plt.show()

    def is_open(self):
        return plt.fignum_exists(self.fig)

    def update(self, points, scale, name=''):
        self.lines = measure_lines(points, scale, name) if points is not None and len(points) else ["no path: 'section' and a click on the scan area place one"]
        self.txt.set_text("\n".join(self.lines))
        self.fig.canvas.draw_idle()

    def raise_(self):
        try:
            self.fig.canvas.manager.show()
        except Exception:
            pass


class SetScaleWindow:
    """
    The set-scale window: the path's nominal length (fixed: the recorded
    length, calibrated length x the scale in force), the 'scale factor' box
    and the 'calibrated length (nm)' box, each recomputing the other from
    the nominal length; 'apply' hands the scale factor to on_apply (the
    browser's set_scale_from_current).  update(cal_length, scale) follows
    the path and the timeline.
    """

    DESIGN = (6.5, 2.8)    # the design size (inches): the text zoom of the window is its size relative to this (qtplot.window_zoom)

    def __init__(self, cal_length_nm, scale, on_apply):
        from sm4_Browser import widgets
        self.on_apply = on_apply
        self.nominal = float(cal_length_nm) * float(scale)
        self.scale = float(scale)
        self.fig = plt.figure(figsize=self.DESIGN)
        try:
            self.fig.canvas.manager.set_window_title("set scale: the section path")
        except Exception:
            pass
        self.txt = self.fig.text(0.05, 0.93, '', fontsize=9, ha='left', va='top')
        self.box_scale = widgets.make(self.fig, 'scale factor', 0.42, 0.26, value=f"{self.scale:g}")
        self.box_cal = widgets.make(self.fig, 'cal length', 0.42, 0.08, value=f"{self.nominal / self.scale:.4f}")
        self.btn_apply = widgets.make(self.fig, 'apply', 0.72, 0.08)
        self.box_scale.on_change(self._on_scale)
        self.box_cal.on_change(self._on_cal)
        self.btn_apply.on_change(self._on_apply)
        self._write()
        plt.show()

    def is_open(self):
        return plt.fignum_exists(self.fig)

    def raise_(self):
        try:
            self.fig.canvas.manager.show()
        except Exception:
            pass

    def _write(self):
        self.txt.set_text(f"nominal length of the path: {self.nominal:.4f} nm (recorded)\n"
                          f"calibrated length = nominal / scale factor\n"
                          f"editing either box recomputes the other;\n"
                          f"'apply' sets the scale factor from the current file's time on")
        self.fig.canvas.draw_idle()

    def update(self, cal_length_nm, scale):
        """The path or the timeline changed: the nominal length and both boxes follow."""
        self.scale = float(scale)
        self.nominal = float(cal_length_nm) * self.scale
        self.box_scale.set(f"{self.scale:g}")
        self.box_cal.set(f"{self.nominal / self.scale:.4f}")
        self._write()

    def _on_scale(self, text):
        try:
            v = float(text)
            if v <= 0:
                raise ValueError("must be > 0")
        except ValueError as exc:
            print(f"set scale: scale factor '{text}' not accepted ({exc}); keeping {self.scale:g}.")
            self.box_scale.set(f"{self.scale:g}")
            return
        self.scale = v
        self.box_cal.set(f"{self.nominal / v:.4f}")
        print(f"set scale: scale factor {v:g} -> calibrated length {self.nominal / v:.4f} nm (nominal {self.nominal:.4f} nm); 'apply' to use it.")

    def _on_cal(self, text):
        try:
            c = float(text)
            if c <= 0:
                raise ValueError("must be > 0")
        except ValueError as exc:
            print(f"set scale: calibrated length '{text}' not accepted ({exc}); keeping {self.nominal / self.scale:.4f} nm.")
            self.box_cal.set(f"{self.nominal / self.scale:.4f}")
            return
        self.scale = self.nominal / c
        self.box_scale.set(f"{self.scale:g}")
        print(f"set scale: calibrated length {c:.4f} nm -> scale factor {self.scale:g} (nominal {self.nominal:.4f} nm); 'apply' to use it.")

    def _on_apply(self):
        if self.on_apply is not None:
            self.on_apply(self.scale)
