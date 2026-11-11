"""
Display machinery, no physics: one class, TopoLdosBlend, that combines a topography colour map and an LDOS
colour map into one two-dimensional colour scale, so that a single map shows the LDOS and the height
together.  It imports NumPy only and nothing of this package, so that the file can be taken into another
program as it is (the sm4_Browser):

    from topo_ldos_blend import TopoLdosBlend
    blend = TopoLdosBlend(topo_cmap, ldos_cmap)        # two colour maps, used exactly as they are
    rgb = blend(ldos_fraction, height_fraction)        # sRGB (..., 3), 0 to 1, for arrays of fractions 0 to 1
    legend = blend.square()                            # the colour square: row = height, column = LDOS

The two colour maps are never altered; only the way they are combined is this class's.
"""

import numpy as np


class TopoLdosBlend:
    """
    The colours of an LDOS value and a height shown together, from the two colour maps given: `topo`, the
    topography's map, and `ldos`, the LDOS map.  Each is a callable taking fractions 0 to 1 and returning
    sRGB colours (..., 3 or 4), 0 to 1 (a matplotlib Colormap, the simulator's qtplot Colormap), or an
    (m, 3 or 4) array of the map's colours from its lowest to its highest value (interpolated linearly).

    With s the LDOS fraction (0 lowest, 1 highest; the value through the map's own norm) and t the height
    fraction (0 lowest, 1 highest), M = ldos(s) and T = topo(t):

    1. Opacity.  a = L*(T) / 100: the darker the topo colour, the less it alters the LDOS colour (a black
       topo colour alters nothing, so the lowest height shows the LDOS map exactly).
    2. Film.  The topo colour laid over the LDOS colour as a coloured film at that opacity, in linear light:
       F = M (1 - a + a Tf).  Tf is T up to the height where the topo map is at its yellowest (its largest
       CIELAB b*), and that yellowest colour above it: a topo map that fades from yellow to white at its
       top would stop tinting there and its colours would run back over colours already used lower down.
    3. Lift.  F lightened by the topo colour's grey scale (its L* with no colour) at the strength LIFT, by
       screen: B = 1 - (1 - F)(1 - LIFT grey(T)).  This is the base colour, and its L* is the result's.
    4. Hue.  The LDOS colour's hue turned toward the hue of the topo map's yellowest colour, through cyan
       and green, by (TURN + TURN_LDOS s) a of the way.
    5. Saturation.  C = (1 - a) C(M) + a Ctarget, with Ctarget between the mean of the two colours'
       saturations and the topo colour's own (SATURATION of the way to the latter).  As in step 2, above
       the height where the topo map is at its yellowest the topo colour's saturation is held at that
       yellowest colour's.
    6. The colour of the base's L* with that hue and saturation, brought into the display's range by
       reducing its saturation alone, is mixed with the base in CIELAB at the share SHARE.
    7. Corner.  The result is darkened toward black in CIELAB (L*, a* and b* scaled together, so no hue
       changes) by CORNER (1 - s) t: most where the LDOS is lowest and the height highest, not at all along
       the lowest height or the highest LDOS.
    8. Lightness.  The colour of steps 1 to 7, as the display shows it, keeps its a* and b* and takes the
       LDOS colour's own L*: at every height the result is exactly as light as the LDOS map's colour of that
       LDOS value, so a black-and-white print of a combined map is the print of the LDOS map alone, and the
       height shows as colour only.  Where that colour falls outside the display's range its saturation
       alone is reduced.  (Steps 3 and 7 therefore act through the a* and b* they leave, not through the
       result's lightness.)

    CIELAB is taken against the D65 white, with CSS Color 4's conversions
    (https://www.w3.org/TR/css-color-4/#color-conversion-code).

    blend(s, t) looks the colours up in a table of the rule made once (TABLE points along each axis,
    bilinear between them); blend.colours(s, t) evaluates the rule itself at every point (slower: about
    twenty conversions per colour).
    """

    SHARE = 0.25          # how far from the base colour to the colour of the hue and saturation rules (step 6)
    TURN = 0.83           # the hue's turn toward the topo map's yellowest hue at full opacity and the lowest LDOS
    TURN_LDOS = 0.06      # the extra turn at the highest LDOS
    SATURATION = 0.5      # the saturation target: 0 the mean of the two colours' saturations, 1 the topo colour's
    LIFT = 0.5            # the strength of the grey-scale lightening
    CORNER = 0.42         # the darkening at the lowest LDOS and the highest height
    TABLE = 257           # the table's points along each axis
    SAMPLES = 1001        # the points a colour map is read at (its yellowest colour is looked for among them)
    FIT_STEPS = 20        # the bisection steps that bring a colour into the display's range

    _D65 = (0.3127 / 0.3290, 1.0, (1.0 - 0.3127 - 0.3290) / 0.3290)
    _XYZ_TO_LIN = np.array(((12831 / 3959, -329 / 214, -1974 / 3959),
                            (-851781 / 878810, 1648619 / 878810, 36519 / 878810),
                            (705 / 12673, -2585 / 12673, 705 / 667)))
    _LIN_TO_XYZ = np.linalg.inv(_XYZ_TO_LIN)
    _EPS = 216 / 24389
    _KAPPA = 24389 / 27

    def __init__(self, topo, ldos):
        q = np.linspace(0.0, 1.0, self.SAMPLES)
        self._topo = self._samples(topo, q)                # (SAMPLES, 3) sRGB, lowest to highest
        self._ldos = self._samples(ldos, q)
        lab = self._rgb_to_lab(self._topo)
        i = int(np.argmax(lab[:, 2]))                      # the topo map's yellowest colour
        self.top_height = i / (self.SAMPLES - 1.0)         # the height fraction it is at
        self.top_hue = float(np.degrees(np.arctan2(lab[i, 2], lab[i, 1])) % 360.0)
        self._top_rgb = self._topo[i].copy()
        self._top_chroma = float(np.hypot(lab[i, 1], lab[i, 2]))
        self._table = None

    # --- the two colour maps ---

    @classmethod
    def _samples(cls, cmap, q):
        """A colour map as (len(q), 3) sRGB at the fractions q: a callable is called with them, an array of colours is interpolated linearly."""
        if callable(cmap):
            c = np.asarray(cmap(q), dtype=np.float64)
        else:
            c = np.asarray(cmap, dtype=np.float64)
            if c.ndim != 2 or c.shape[0] < 2 or c.shape[1] < 3:
                raise ValueError(f"a colour map given as colours must be (m, 3 or 4) with m >= 2, not {c.shape}")
            x = np.linspace(0.0, 1.0, c.shape[0])
            c = np.stack([np.interp(q, x, c[:, k]) for k in range(3)], axis=-1)
        if c.ndim != 2 or c.shape[0] != len(q) or c.shape[1] < 3:
            raise ValueError(f"a colour map must give (n, 3 or 4) colours for n fractions, not {c.shape}")
        return np.clip(c[:, :3], 0.0, 1.0)

    @staticmethod
    def _lookup(samples, x):
        """The colours (..., 3) of a sampled colour map at the fractions x, linear between its samples."""
        x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0) * (len(samples) - 1.0)
        i = np.minimum(x.astype(np.int64), len(samples) - 2)
        f = (x - i)[..., None]
        return samples[i] * (1.0 - f) + samples[i + 1] * f

    # --- colour conversions (CSS Color 4, D65 white) ---

    @staticmethod
    def _decode(c):
        """sRGB values to linear light."""
        a = np.abs(c)
        return np.where(a <= 0.04045, c / 12.92, np.sign(c) * np.power((a + 0.055) / 1.055, 2.4))

    @staticmethod
    def _encode(c):
        """Linear light to sRGB values."""
        a = np.abs(c)
        return np.where(a > 0.0031308, np.sign(c) * (1.055 * np.power(a, 1 / 2.4) - 0.055), 12.92 * c)

    @classmethod
    def _rgb_to_lab(cls, rgb):
        """sRGB (..., 3) to CIELAB (..., 3)."""
        xyz = cls._decode(np.asarray(rgb, dtype=np.float64)) @ cls._LIN_TO_XYZ.T / np.array(cls._D65)
        f = np.where(xyz > cls._EPS, np.cbrt(xyz), (cls._KAPPA * xyz + 16.0) / 116.0)
        return np.stack([116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]), 200.0 * (f[..., 1] - f[..., 2])],
                        axis=-1)

    @classmethod
    def _lab_to_lin(cls, lab):
        """CIELAB (..., 3) to linear sRGB (..., 3), not clipped."""
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        f1 = (L + 16.0) / 116.0
        f0 = f1 + a / 500.0
        f2 = f1 - b / 200.0
        x = np.where(f0 ** 3 > cls._EPS, f0 ** 3, (116.0 * f0 - 16.0) / cls._KAPPA)
        y = np.where(L > cls._KAPPA * cls._EPS, f1 ** 3, L / cls._KAPPA)
        z = np.where(f2 ** 3 > cls._EPS, f2 ** 3, (116.0 * f2 - 16.0) / cls._KAPPA)
        return (np.stack([x, y, z], axis=-1) * np.array(cls._D65)) @ cls._XYZ_TO_LIN.T

    @classmethod
    def _lab_to_rgb(cls, lab):
        """CIELAB (..., 3) to sRGB (..., 3), clipped to the display's range."""
        return cls._encode(np.clip(cls._lab_to_lin(lab), 0.0, 1.0))

    @classmethod
    def _fit(cls, lab):
        """A CIELAB colour brought into the display's range at its own L* and hue by reducing its saturation alone (bisection per colour), as sRGB."""
        lab = np.array(lab, dtype=np.float64)
        lab[..., 0] = np.clip(lab[..., 0], 0.0, 100.0)

        def at(k):
            return lab * np.stack([np.ones_like(k), k, k], axis=-1)

        def inside(c):
            lin = cls._lab_to_lin(c)
            return np.all((lin >= -1e-6) & (lin <= 1.0 + 1e-6), axis=-1)

        lo = np.zeros(lab.shape[:-1])
        hi = np.ones(lab.shape[:-1])
        full = inside(at(hi))
        for _ in range(cls.FIT_STEPS):
            mid = 0.5 * (lo + hi)
            ok = inside(at(mid))
            lo = np.where(ok, mid, lo)
            hi = np.where(ok, hi, mid)
        return cls._lab_to_rgb(at(np.where(full, 1.0, lo)))

    # --- the rule ---

    def colours(self, ldos, height):
        """
        The rule itself at every point: sRGB (..., 3), 0 to 1, for the LDOS fractions `ldos` and the height
        fractions `height` (arrays of one shape, or values that broadcast; both clipped to 0..1).
        """
        s, t = np.broadcast_arrays(np.clip(np.asarray(ldos, dtype=np.float64), 0.0, 1.0),
                                   np.clip(np.asarray(height, dtype=np.float64), 0.0, 1.0))
        M = self._lookup(self._ldos, s)
        T = self._lookup(self._topo, t)
        lab_m, lab_t = self._rgb_to_lab(M), self._rgb_to_lab(T)
        a = lab_t[..., 0] / 100.0                                              # 1. opacity
        Tf = np.where((t > self.top_height)[..., None], self._top_rgb, T)
        F = np.clip(self._encode(self._decode(M) * (1.0 - a[..., None] + a[..., None] * self._decode(Tf))),
                    0.0, 1.0)                                                  # 2. film
        grey = self._lab_to_rgb(np.stack([lab_t[..., 0], np.zeros_like(a), np.zeros_like(a)], axis=-1))
        base = self._rgb_to_lab(np.clip(1.0 - (1.0 - F) * (1.0 - self.LIFT * grey), 0.0, 1.0))   # 3. lift
        c_m, c_t = np.hypot(lab_m[..., 1], lab_m[..., 2]), np.hypot(lab_t[..., 1], lab_t[..., 2])
        c_t = np.where(t > self.top_height, self._top_chroma, c_t)             # held above the yellowest height
        h_m = np.degrees(np.arctan2(lab_m[..., 2], lab_m[..., 1])) % 360.0
        h = np.radians(h_m - (self.TURN + self.TURN_LDOS * s) * a * ((h_m - self.top_hue) % 360.0))   # 4. hue
        c_target = (1.0 - self.SATURATION) * 0.5 * (c_m + c_t) + self.SATURATION * c_t
        c = (1.0 - a) * c_m + a * c_target                                     # 5. saturation
        ruled = self._rgb_to_lab(np.clip(self._fit(np.stack([base[..., 0], c * np.cos(h), c * np.sin(h)], axis=-1)),
                                         0.0, 1.0))
        out = (1.0 - self.SHARE) * base + self.SHARE * ruled                   # 6. the share of the two
        out = out * (1.0 - self.CORNER * (1.0 - s) * t)[..., None]             # 7. corner
        ab = self._rgb_to_lab(self._lab_to_rgb(out))[..., 1:]
        return self._fit(np.concatenate([lab_m[..., :1], ab], axis=-1))        # 8. the LDOS colour's own lightness

    def table(self):
        """The rule on a grid, made once: (TABLE, TABLE, 3) sRGB, [i, j] the colour at the height fraction i / (TABLE - 1) and the LDOS fraction j / (TABLE - 1)."""
        if self._table is None:
            q = np.linspace(0.0, 1.0, self.TABLE)
            ss, tt = np.meshgrid(q, q)
            self._table = self.colours(ss, tt)
        return self._table

    def __call__(self, ldos, height):
        """
        The colours looked up in the table (bilinear between its points): sRGB (..., 3), 0 to 1, for the LDOS
        fractions `ldos` and the height fractions `height` (arrays of one shape, or values that broadcast;
        both clipped to 0..1).
        """
        tab = self.table()
        n = self.TABLE - 1
        s, t = np.broadcast_arrays(np.clip(np.asarray(ldos, dtype=np.float64), 0.0, 1.0) * n,
                                   np.clip(np.asarray(height, dtype=np.float64), 0.0, 1.0) * n)
        j = np.minimum(s.astype(np.int64), n - 1)
        i = np.minimum(t.astype(np.int64), n - 1)
        fs, ft = (s - j)[..., None], (t - i)[..., None]
        return ((tab[i, j] * (1.0 - fs) + tab[i, j + 1] * fs) * (1.0 - ft)
                + (tab[i + 1, j] * (1.0 - fs) + tab[i + 1, j + 1] * fs) * ft)

    def square(self, n_ldos=256, n_height=256):
        """The colour square, as a legend: (n_height, n_ldos, 3) sRGB, row i the height fraction (i + 0.5) / n_height (the first row the lowest height) and column j the LDOS fraction (j + 0.5) / n_ldos (the first column the lowest LDOS)."""
        s = (np.arange(int(n_ldos)) + 0.5) / int(n_ldos)
        t = (np.arange(int(n_height)) + 0.5) / int(n_height)
        ss, tt = np.meshgrid(s, t)
        return self(ss, tt)

    def topo_bar(self, n=256):
        """The topography's colour map alone, unaltered: (n, 3) sRGB from the lowest height to the highest."""
        return self._lookup(self._topo, (np.arange(int(n)) + 0.5) / int(n))

    def ldos_bar(self, n=256):
        """The LDOS colour map alone, unaltered: (n, 3) sRGB from the lowest LDOS to the highest."""
        return self._lookup(self._ldos, (np.arange(int(n)) + 0.5) / int(n))
