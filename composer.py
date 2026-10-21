"""
The figure composer of the Map window: the MATLAB map script's "add to
plot" and "new plot" (plot_dIdVmap_9_56_1a_n.m, plotmapseries_CB and
new_plotmapseries_CB) rebuilt on qtplot.

Each "add to plot" button of the Map window copies what its panel shows
into a list: a map at its energy with its colour limits, gamma and zoom
(and its drift shift), the embedded topo as drawn, or the spectra of the
curve panel, each with the position it was taken at.  After every press
one figure shows everything collected so far, COLUMNS panels wide, square
panels, sorted: the topo first, then the maps in order of energy (titled
with their bias, "0.45 V"), then the spectra.  The panels are
cross-referenced: every collected spectrum gets its own symbol (SYMBOLS,
in the order collected), drawn in black on every map and topo at that
spectrum's position; every spectrum panel gets a down arrow at the energy
of every collected map and is titled "<quantity> at '<symbol>'".  Maps and
the topo carry a 1 nm scale bar at the bottom right and no axis ticks.  In
the figure, Delete with the cursor over a panel removes it, and "new plot"
in the Map window empties the list.  The figure's toolbar Save gives the
usual exports (mplexport for the publication types).

If run as a file (Spyder F5) the if-main shows the composer on synthetic
panels.
"""

if __package__ in (None, ''):
    # Run as a file (Spyder F5): make the package importable from wherever its folder sits.
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
from sm4_Browser import qtplot as plt
from sm4_Browser import plots


class FigureComposer:
    """
    The collected panels (items) and the one figure that shows them.
      add_map(img, extent, limits, vmin, vmax, gamma, cmap, bias_V, shift=None)
      add_topo(layers, extent, limits)      layers: [(img, shift or None), ...], the first at the bottom
      add_spectrum(x, y, xlim, ylim, position, quantity)
      clear()                               the list emptied (the figure stays, empty)
      ordered()                             the items in the figure's order
    Every add rebuilds the figure (rebuild), which is made at the first add
    and made again when it was closed.  `panels` is [(axes, item, info)] of
    the last build, info = {'symbols': n, 'arrows': n} drawn on that panel.
    """

    COLUMNS = 4
    CELL_IN = 1.9             # every cell a square of this size (inches); the panel a square inside it
    DESIGN = (COLUMNS * CELL_IN, CELL_IN)   # the design size of one row of cells (inches); the window opens
                                            # at as many rows as the panels need (qtplot.window_zoom)
    PANEL = 0.74              # the panel's side, fraction of the cell (room above for the title, below for tick labels)
    BOTTOM = 0.11             # the panel's bottom, fraction of the cell
    SYMBOLS = ['+', 'x', '*', '.', '^', 'v', '>', '<', 'o', 's', 'd', 'p', 'h']   # MATLAB's specMarker list
    HEADROOM = 0.85           # a spectrum's curve fills this fraction of its panel's height; the arrows sit above
    SCALE_BAR_NM = 1.0        # the maps' scale bar (the frame's units are nm)

    def __init__(self, title):
        self.title = title
        self.items = []
        self.fig = None
        self.panels = []
        self._rows = 0

    # --- the list ---

    def add_map(self, img, extent, limits, vmin, vmax, gamma, cmap, bias_V, shift=None):
        """A map at one energy as the Map window shows it (its colour limits vmin / vmax, gamma, cmap, the axes limits, the pixel shifts)."""
        self.items.append({'kind': 'map', 'img': np.array(img, dtype=np.float64), 'extent': [float(v) for v in extent],
                           'limits': [float(v) for v in limits],
                           'vmin': None if vmin is None else float(vmin), 'vmax': None if vmax is None else float(vmax),
                           'gamma': float(gamma), 'cmap': cmap, 'bias': float(bias_V),
                           'shift': None if shift is None else np.array(shift, dtype=np.float64)})
        self.rebuild()

    def add_topo(self, layers, extent, limits):
        """The embedded topo as drawn: layers [(img, shift or None), ...] drawn in order, the first at the bottom."""
        self.items.append({'kind': 'topo',
                           'layers': [(np.array(img, dtype=np.float64), None if sh is None else np.array(sh, dtype=np.float64))
                                      for img, sh in layers],
                           'extent': [float(v) for v in extent], 'limits': [float(v) for v in limits]})
        self.rebuild()

    def add_spectrum(self, x, y, xlim, ylim, position, quantity):
        """One spectrum (x, y) with the curve panel's limits, taken at `position` (x, y in the frame), of `quantity`."""
        self.items.append({'kind': 'spectrum', 'x': np.array(x, dtype=np.float64), 'y': np.array(y, dtype=np.float64),
                           'xlim': [float(v) for v in xlim], 'ylim': [float(v) for v in ylim],
                           'position': (float(position[0]), float(position[1])), 'quantity': str(quantity)})
        self.rebuild()

    def clear(self):
        """'new plot': the list emptied; the figure, if open, shows nothing."""
        self.items = []
        self.rebuild()

    def ordered(self):
        """The items in the figure's order: the topo first, the maps by bias, then the spectra as collected."""
        rank = {'topo': 0, 'map': 1, 'spectrum': 2}
        return sorted(self.items, key=lambda it: (rank[it['kind']], it['bias'] if it['kind'] == 'map' else 0.0))

    def is_open(self):
        return self.fig is not None and plt.fignum_exists(self.fig)

    # --- the figure ---

    def rebuild(self):
        """The figure made (or made again after a close) and filled with the ordered items, COLUMNS wide."""
        items = self.ordered()
        n = len(items)
        rows = max((n - 1) // self.COLUMNS + 1, 1)
        if self.is_open() and rows != self._rows:
            plt.close(self.fig)                      # a new row count: the figure made again at its size
            self.fig = None
        if not self.is_open():
            if n == 0:
                return
            self.fig = plt.figure(figsize=(self.DESIGN[0], rows * self.DESIGN[1]))
            self.fig.canvas.mpl_connect('key_press_event', self._on_key)
            try:
                self.fig.canvas.manager.set_window_title(self.title)
            except Exception:
                pass
        self._rows = rows
        self.fig.clf()
        self.panels = []
        if n == 0:
            self.fig.canvas.draw_idle()
            return
        spectra = [it for it in items if it['kind'] == 'spectrum']
        biases = [it['bias'] for it in items if it['kind'] == 'map']
        side, bottom = self.PANEL, self.BOTTOM
        k_spec = 0
        for i, it in enumerate(items):
            r, c = divmod(i, self.COLUMNS)
            # The panel: a square of PANEL of the cell, centred across, BOTTOM up from the cell's bottom.
            ax = self.fig.add_axes([(c + 0.5 * (1.0 - side)) / self.COLUMNS, (rows - 1 - r + bottom) / rows,
                                    side / self.COLUMNS, side / rows])
            if it['kind'] == 'spectrum':
                info = self._draw_spectrum_panel(ax, it, biases, self.SYMBOLS[k_spec % len(self.SYMBOLS)])
                k_spec += 1
            else:
                info = self._draw_image_panel(ax, it, spectra)
            self.panels.append((ax, it, info))
        self.fig.canvas.draw_idle()

    def _draw_image_panel(self, ax, it, spectra):
        """A map or the topo: the image(s) as collected, square, no ticks, the scale bar, the spectra's symbols."""
        if it['kind'] == 'map':
            if it['shift'] is not None:
                plots.PixelMap(ax, it['img'], it['extent'], it['cmap'], gamma=it['gamma'], shift=it['shift'],
                               vmin=it['vmin'], vmax=it['vmax'])
            else:
                plots.draw_heatmap(ax, it['img'], it['extent'], it['cmap'], gamma=it['gamma'], vmin=it['vmin'],
                                   vmax=it['vmax'], origin='lower', aspect='equal', interpolation='none')
            ax.set_title(f"{it['bias']:.2f} V", fontsize=8)
        else:
            for img, sh in it['layers']:
                if sh is not None:
                    plots.PixelMap(ax, img, it['extent'], plots.TOPO_CMAP, shift=sh)
                else:
                    ax.imshow(img, origin='lower', extent=it['extent'], cmap=plots.TOPO_CMAP, interpolation='none')
            ax.set_title("Topo", fontsize=8)
        ax.set_aspect('equal', adjustable='box')
        x0, x1, y0, y1 = it['limits']
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xticks([])
        ax.set_yticks([])
        # The scale bar: SCALE_BAR_NM long, 5 % in from the right edge, 4 % up from the bottom (the MATLAB
        # rectangle), white on a black outline so it reads on any colour.
        bx1 = x1 - 0.05 * (x1 - x0)
        bx0 = bx1 - self.SCALE_BAR_NM
        by = y0 + 0.04 * (y1 - y0)
        ax.plot([bx0, bx1], [by, by], color='black', lw=4.5, solid_capstyle='butt')
        ax.plot([bx0, bx1], [by, by], color='white', lw=2.5, solid_capstyle='butt')
        # Every collected spectrum's position with its symbol.
        for k, sp in enumerate(spectra):
            px, py = sp['position']
            ax.plot([px], [py], marker=self.SYMBOLS[k % len(self.SYMBOLS)], color='black', ms=8, mew=1.2, ls='none')
        return {'symbols': len(spectra), 'arrows': 0}

    def _draw_spectrum_panel(self, ax, it, biases, symbol):
        """A spectrum: the curve, no y ticks, headroom above it with a down arrow at every collected map's bias."""
        ax.plot(it['x'], it['y'], color='black', lw=1.2)
        ax.set_xlim(*it['xlim'])
        y0, y1 = it['ylim']
        if not (np.isfinite(y0) and np.isfinite(y1)) or y1 <= y0:
            ok = np.isfinite(it['y'])
            y0, y1 = (float(np.min(it['y'][ok])), float(np.max(it['y'][ok]))) if ok.any() else (0.0, 1.0)
            if y1 <= y0:
                y1 = y0 + 1.0
        ax.set_ylim(y0, y0 + (y1 - y0) / self.HEADROOM)
        ax.set_yticks([])
        ax.tick_params(labelsize=6)
        for b in biases:
            ax.text(b, y1, "↓", ha='center', va='bottom', fontsize=12)
        ax.set_title(f"{it['quantity']} at '{symbol}'", fontsize=8)      # the x ticks are the bias in V, as MATLAB had it
        return {'symbols': 0, 'arrows': len(biases)}

    def _on_key(self, event):
        """Delete with the cursor over a panel: that item removed and the figure rebuilt."""
        if getattr(event, 'key', None) != 'delete' or getattr(event, 'inaxes', None) is None:
            return
        for ax, it, _ in self.panels:
            if ax is event.inaxes:
                self.items = [x for x in self.items if x is not it]
                event.handled = True
                self.rebuild()
                return


if __name__ == '__main__':
    # The composer on synthetic panels: a topo, two maps, and two spectra.
    x = np.linspace(-1, 1, 101)
    yy, xx = np.mgrid[0:16, 0:16]
    c = FigureComposer("composer demo")
    c.add_topo([(np.sin(xx / 2.0) + 0.3 * yy, None)], [0, 8, 0, 8], [0, 8, 0, 8])
    c.add_map(np.cos(xx / 2.0) * yy, [0, 8, 0, 8], [0, 8, 0, 8], None, None, 1.0, 'jet', 0.45)
    c.add_map(np.cos(xx / 3.0) * yy, [0, 8, 0, 8], [0, 8, 0, 8], None, None, 1.0, 'jet', -0.30)
    c.add_spectrum(x, np.exp(-(x - 0.45) ** 2 / 0.05), [-1, 1], [0, 1], (2.0, 3.0), "dI/dV")
    c.add_spectrum(x, np.exp(-(x + 0.30) ** 2 / 0.05), [-1, 1], [0, 1], (6.0, 5.0), "dI/dV")
    plt.show()
