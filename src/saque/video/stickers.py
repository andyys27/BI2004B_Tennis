"""Detección de las estampitas (marcadores adhesivos verde-amarillo) cerca de cada landmark.

Estrategia: MediaPipe da una posición aproximada de la articulación; en una ventana de
`search_radius_px` alrededor se segmenta por color HSV (tono verde-amarillo y saturación alta,
lo que separa la estampita de la playera, que es menos saturada) y se acepta la mancha más
cercana al landmark que cumpla tamaño y circularidad. Si no hay mancha válida se devuelve NaN
y el pipeline usa el landmark de MediaPipe corregido por el desfase medio estampita-landmark.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class StickerParams:
    hsv_low: tuple[int, int, int] = (25, 170, 90)
    hsv_high: tuple[int, int, int] = (45, 255, 255)
    search_radius_px: int = 45
    min_area_px: int = 40
    max_area_px: int = 1500
    min_circularity: float = 0.55

    @classmethod
    def from_config(cls, cfg: dict) -> "StickerParams":
        keys = cls.__dataclass_fields__
        return cls(**{k: (tuple(v) if isinstance(v, list) else v)
                      for k, v in cfg.items() if k in keys})


def find_sticker(rgb: np.ndarray, x: float, y: float, p: StickerParams) -> tuple[float, float, float]:
    """Centroide (x, y) en px de la estampita más cercana a (x, y) y su área; NaN si no hay."""
    nan = (np.nan, np.nan, np.nan)
    if not (np.isfinite(x) and np.isfinite(y)):
        return nan
    h, w = rgb.shape[:2]
    r = p.search_radius_px
    x0, y0 = max(0, int(x) - r), max(0, int(y) - r)
    x1, y1 = min(w, int(x) + r + 1), min(h, int(y) + r + 1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return nan
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb[y0:y1, x0:x1]), cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array(p.hsv_low, np.uint8), np.array(p.hsv_high, np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_d = nan, np.inf
    for c in contours:
        area = cv2.contourArea(c)
        if not p.min_area_px <= area <= p.max_area_px:
            continue
        perim = cv2.arcLength(c, True)
        if perim == 0 or 4 * np.pi * area / perim ** 2 < p.min_circularity:
            continue
        m = cv2.moments(c)
        cx, cy = x0 + m["m10"] / m["m00"], y0 + m["m01"] / m["m00"]
        d = np.hypot(cx - x, cy - y)
        if d <= r and d < best_d:
            best, best_d = (cx, cy, area), d
    return best
