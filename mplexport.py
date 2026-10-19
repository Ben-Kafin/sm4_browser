"""
mplexport: a qtplot figure (what a window shows) rebuilt as a matplotlib figure and saved by
matplotlib, for publication files: PDF, EPS, SVG (vector, every line and text an editable object,
e.g. in Adobe Illustrator), and PNG / TIFF at a chosen dpi.

matplotlib is used here and only here, at the moment of saving: the windows draw and interact
with pyqtgraph (fast marker dragging, panning, many curves), and matplotlib's renderers make
the file.  The rebuild reads what qtplot already records in matplotlib's terms: every axes at its
figure-fraction position with its current limits, labels, title, ticks, aspect, and legend; every
curve with its style; images with their colour map and norm (and the rotation of a placed
topography); quad meshes (the Line window's sheared rows, the Map window's shifted pixels, the
grid stripes) as filled polygons; reference lines, spans, line collections, texts, annotations,
colour bars, and the figure's texts.  The controls (buttons, boxes) are not part of the figure.

    export(fig, path, dpi=300)       the figure into `path` (format from the extension)
    to_matplotlib(fig)               the matplotlib Figure itself (to adjust before saving)
"""

import numpy as np

from sm4_Browser import qtplot as q


def available():
    try:
        import matplotlib            # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------- #
#  Conversions                                                             #
# ---------------------------------------------------------------------- #

def _mpl_cmap(cm):
    from matplotlib.colors import ListedColormap
    cm = q.to_cmap(cm)
    return ListedColormap(cm._table, name=cm.name)


def _mpl_norm(norm):
    import matplotlib.colors as mc
    if norm is None:
        return None
    if isinstance(norm, q.PowerNorm):
        return mc.PowerNorm(norm.gamma, vmin=norm.vmin, vmax=norm.vmax)
    if isinstance(norm, q.TwoSlopeNorm):
        return mc.TwoSlopeNorm(norm.vcenter, vmin=norm.vmin, vmax=norm.vmax)
    return mc.Normalize(vmin=norm.vmin, vmax=norm.vmax)


def _ls(ls):
    return 'none' if ls in (None, 'none', 'None', '', ' ') else ls


def _px_to_pt(px):
    return float(px) / q.pt2px(1.0)


def _axis_hidden_ticks(axis):
    lev = getattr(axis, '_tickLevels', None)
    return lev is not None and all(len(level) == 0 for level in lev)


# ---------------------------------------------------------------------- #
#  The rebuild                                                             #
# ---------------------------------------------------------------------- #

def to_matplotlib(fig):
    """The qtplot figure as a matplotlib Figure (no GUI backend: Figure + savefig)."""
    from matplotlib.figure import Figure
    W0, H0 = fig._design_px
    mfig = Figure(figsize=(W0 / q.DPI, H0 / q.DPI), dpi=q.DPI)
    mfig.patch.set_facecolor('white')
    axes = [ax for ax in fig._axes if ax._visible]
    cbar_axes = {id(ax) for ax in axes if getattr(ax, '_colorbar', None) is not None}
    made = {}
    for ax in axes:
        if id(ax) in cbar_axes:
            continue
        made[id(ax)] = _axes(mfig, ax)
    for ax in axes:
        if id(ax) in cbar_axes:
            _colorbar(mfig, ax)
    for ax in axes:                      # shared y: the second axes shows no y values (as on screen)
        if ax._shared_y is not None and id(ax) in made:
            made[id(ax)].tick_params(axis='y', labelleft=False)
    # Figure texts of the plot area only (a text below every panel is a control's caption).
    bottom = min((_pos(ax)[1] for ax in axes), default=0.0)
    for t in fig._texts:
        if t.get_visible() and t.get_text() and t._y >= bottom - 0.02:
            _text(mfig, t, transform=mfig.transFigure, target=mfig)
    return mfig


def _pos(ax):
    """The axes' rectangle as shown (after the aspect, colour-bar clearance, ...), in figure fractions."""
    x0, y0, w, h = ax._px_rect()
    W, H = ax.figure._canvas_size()
    return [x0 / W, 1.0 - (y0 + h) / H, w / W, h / H]


def _axes(mfig, ax):
    m = mfig.add_axes(_pos(ax))
    m.set_facecolor('white')
    handle_map = {}
    for a in ax._children:
        if not a.get_visible():
            continue
        made = _artist(m, a)
        if made is not None:
            handle_map[id(a)] = made
    # Quad meshes added straight to the view (RowHeatmap, PixelMap): filled polygons.
    for it in getattr(ax.vb, 'addedItems', []):
        if isinstance(it, q.QuadItem) and it.isVisible() and not any(it is getattr(a, 'item', None) for a in ax._children):
            _quads(m, it)
    # limits as shown
    (xa, xb), (ya, yb) = ax.vb.viewRange()
    if ax.vb.state['xInverted']:
        xa, xb = xb, xa
    if ax.vb.state['yInverted']:
        ya, yb = yb, ya
    if ax._aspect == 'equal':
        m.set_aspect('equal', adjustable='box' if getattr(ax, '_adjust_box', False) else 'datalim')
    m.set_xlim(xa, xb)
    m.set_ylim(ya, yb)
    # labels, ticks, title
    xi, yi = ax.xaxis_item, ax.yaxis_item
    if ax._xlabel:
        m.set_xlabel(q.short_label(ax._xlabel), fontsize=xi._label_pt)
    if ax._ylabel:
        m.set_ylabel(q.short_label(ax._ylabel), fontsize=yi._label_pt)
    m.tick_params(axis='x', labelsize=xi._tick_pt)
    m.tick_params(axis='y', labelsize=yi._tick_pt)
    if _axis_hidden_ticks(xi):
        m.set_xticks([])
    if _axis_hidden_ticks(yi):
        m.set_yticks([])
    if not xi.style['showValues']:
        m.tick_params(axis='x', labelbottom=False)
    if not yi.style['showValues']:
        m.tick_params(axis='y', labelleft=False)
    if not yi.style['tickLength']:
        m.tick_params(axis='y', left=False)
    if not xi.style['tickLength']:
        m.tick_params(axis='x', bottom=False)
    if xi.grid is not False or yi.grid is not False:
        m.grid(True, lw=0.4, alpha=0.4)
    if not ax._axison:
        m.axis('off')
    elif not ax._frame:
        m.set_frame_on(False)
    if ax._title is not None and ax._title.get_text():
        m.set_title(ax._title.get_text(), fontsize=ax._title._size)
    for side in ax._side_axes:
        mode = getattr(side, '_mode', None)
        if mode is not None and mode[0] == 'func':
            sec = m.secondary_yaxis('right', functions=(mode[1], mode[2]))
            if side.labelText:
                sec.set_ylabel(q.short_label(side.labelText), fontsize=side._label_pt)
    # legend as on screen
    if ax._legend is not None and getattr(ax, '_legend_args', None) is not None:
        args, kw = ax._legend_args
        loc = kw.get('loc', 'best')
        loc = 'upper right' if loc in ('best', 0, 1) else loc
        lk = dict(loc=loc, frameon=kw.get('frameon', True), fontsize=q.font_pt(kw.get('fontsize'), 10.0),
                  ncol=kw.get('ncol', 1))
        if len(args) >= 2:
            hs = [handle_map.get(id(h)) for h in args[0]]
            pairs = [(h, l) for h, l in zip(hs, args[1]) if h is not None and l and not str(l).startswith('_')]
            if pairs:
                m.legend([p[0] for p in pairs], [p[1] for p in pairs], **lk)
        else:
            m.legend(**lk)
    return m


def _line_kw(st, label, zorder):
    kw = dict(color=st['color'], lw=st['lw'], ls=_ls(st['ls']), alpha=st['alpha'], zorder=zorder,
              label=label if (label and not str(label).startswith('_')) else None)
    if st.get('marker'):
        kw.update(marker=st['marker'], ms=st['ms'], mew=st['mew'])
        if st.get('mec') is not None:
            kw['mec'] = st['mec']
        if st.get('mfc') is not None:
            kw['mfc'] = st['mfc']
    return kw


def _artist(m, a):
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.transforms import Affine2D
    if isinstance(a, q.Line2D):
        ln, = m.plot(a._x, a._y, **_line_kw(a._style, a.get_label(), a.zorder))
        return ln
    if isinstance(a, q._RefLine):
        st = a._style
        f = m.axhline if a._horizontal else m.axvline
        return f(a._value, color=st['color'], lw=st['lw'], ls=_ls(st['ls']), alpha=st['alpha'], zorder=a.zorder,
                 label=a.get_label() if not str(a.get_label()).startswith('_') else None)
    if isinstance(a, q._Span):
        lo, hi, vertical, color, alpha = a._span
        f = m.axvspan if vertical else m.axhspan
        return f(lo, hi, color=color, alpha=alpha, lw=0, zorder=a.zorder,
                 label=a.get_label() if not str(a.get_label()).startswith('_') else None)
    if isinstance(a, q.AxesImage):
        if not a.item.isVisible():
            return None
        im = m.imshow(a._A, cmap=_mpl_cmap(a.cmap), norm=_mpl_norm(a.norm), origin=a.origin, extent=a._extent,
                      interpolation='none', aspect='auto', zorder=a.zorder)
        if a._affine is not None:
            im.set_transform(Affine2D(a._affine.get_matrix()) + m.transData)
        return im
    if isinstance(a, q.QuadMesh):
        return m.pcolormesh(a._X, a._Y, np.ma.masked_invalid(a._C), cmap=_mpl_cmap(a.cmap), norm=_mpl_norm(a.norm),
                            shading='flat', zorder=a.zorder)
    if isinstance(a, q.ContourSet):
        X, Y, Z, levels, cmap = a._args
        return m.contourf(X, Y, Z, levels=levels, cmap=_mpl_cmap(cmap), zorder=a.zorder)
    if isinstance(a, q.LineCollection):
        lc = LineCollection(a._segments, colors=a._color, linewidths=a._lw, linestyles=_ls(a._ls) if a._ls != '-' else 'solid',
                            alpha=a._alpha, zorder=a.zorder)
        m.add_collection(lc)
        return lc
    if isinstance(a, q.PolyCollection):
        if a.item is None:
            return None
        return _quads(m, a.item, zorder=a.zorder)
    if isinstance(a, q.Annotation):
        p = a._ann
        kw = dict(color=p['color'], fontsize=q.font_pt(p['fontsize'], 10.0), fontweight=p['fontweight'],
                  ha=p['ha'], va=p['va'], zorder=a.zorder)
        if p['arrowprops'] is not None:
            kw['arrowprops'] = p['arrowprops']
        return m.annotate(p['s'] or '', xy=p['xy'], xytext=p['xytext'], textcoords=p['textcoords'], **kw)
    if isinstance(a, q.Text):
        if not a.get_text():
            return None
        tr = m.transAxes if a._coords == 'axes' else None
        return _text(m, a, transform=tr, target=m)
    return None


def _quads(m, item, zorder=None):
    from matplotlib.collections import PolyCollection
    quads = getattr(item, '_quads', None)
    cols = getattr(item, '_colors', None)
    if quads is None or cols is None or not len(quads):
        return None
    pc = PolyCollection(quads, facecolors=np.asarray(cols, dtype=float) / 255.0, edgecolors='face', linewidths=0.1,
                        antialiased=False, zorder=item.zValue() if zorder is None else zorder)
    m.add_collection(pc)
    return pc


def _text(target_owner, t, transform=None, target=None):
    kw = dict(fontsize=t._size, ha=t._ha, va=t._va, color=t._color, rotation=t._rotation)
    if t._weight:
        kw['fontweight'] = t._weight
    if t._family in ('monospace', 'mono', 'courier'):
        kw['family'] = 'monospace'
    if t._bbox:
        b = dict(t._bbox)
        kw['bbox'] = {k: v for k, v in b.items() if k in ('facecolor', 'fc', 'alpha', 'edgecolor', 'ec', 'lw',
                                                         'linewidth', 'pad', 'boxstyle')}
    if transform is not None:
        kw['transform'] = transform
    if t._offset_px != (0.0, 0.0):
        from matplotlib.transforms import offset_copy
        base = transform if transform is not None else target.transData
        kw['transform'] = offset_copy(base, fig=target.figure if hasattr(target, 'figure') else target,
                                      x=_px_to_pt(t._offset_px[0]), y=_px_to_pt(t._offset_px[1]), units='points')
    return target.text(t._x, t._y, t.get_text(), **kw)


def _colorbar(mfig, cax):
    from matplotlib.cm import ScalarMappable
    cb = cax._colorbar
    m = mfig.add_axes(_pos(cax))
    sm = ScalarMappable(norm=_mpl_norm(getattr(cb.mappable, 'norm', None)), cmap=_mpl_cmap(cb.mappable.cmap))
    mcb = mfig.colorbar(sm, cax=m)
    if cb._label:
        mcb.set_label(q.short_label(cb._label), fontsize=cb._axis._label_pt)
    m.tick_params(labelsize=cb._axis._tick_pt)
    return mcb


# ---------------------------------------------------------------------- #
#  Saving                                                                  #
# ---------------------------------------------------------------------- #

def export(fig, path, dpi=300):
    """
    Save `fig` through matplotlib: the format from the extension (pdf, eps, ps, svg, png, tif,
    jpg); text stays text in the vector formats (TrueType fonts embedded, editable in
    Illustrator); raster formats at `dpi`.
    """
    import matplotlib
    mfig = to_matplotlib(fig)
    with matplotlib.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'}):
        # Cropped to what is drawn (the window's control strip is empty in the figure).
        mfig.savefig(path, dpi=dpi, facecolor='white', bbox_inches='tight', pad_inches=0.05)
    return path
