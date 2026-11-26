"""Fechado del impacto con la pelota (solo para el tiempo de contacto, no se analiza su velocidad).

Tras el lanzamiento la pelota cae casi en línea recta con aceleración ~g; se sigue fotograma a
fotograma prediciendo su posición con una parábola ajustada a los puntos previos. El impacto es
el último fotograma en que la pelota sigue esa trayectoria de caída: después el contacto con la
raqueta la saca de la trayectoria o la oculta.

Las manchas de color pelota que no se mueven (pintura de la cancha, cerco) se descartan.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

from .reader import VideoInfo, iter_frames


@dataclass(frozen=True)
class BallParams:
    hsv_low: tuple[int, int, int] = (22, 70, 70)
    hsv_high: tuple[int, int, int] = (50, 255, 255)
    roi_y_max_px: int = 1100        # se busca por encima de esta fila (cielo / fondo superior)
    min_area_px: int = 60
    max_area_px: int = 2500
    min_aspect: float = 0.5
    gate_px: float = 15.0           # distancia máxima a la posición predicha (con parábola)
    max_missing: int = 4            # fotogramas sin detección que terminan el seguimiento
    min_track: int = 8
    max_start_y_px: int = 450       # la caída debe empezar arriba (la pelota reentra por el cielo)

    @classmethod
    def from_config(cls, cfg: dict) -> "BallParams":
        keys = cls.__dataclass_fields__
        return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.items() if k in keys})


def ball_candidates(rgb: np.ndarray, p: BallParams) -> np.ndarray:
    """Centroides (N, 2) de manchas con color, tamaño y forma de pelota."""
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb[: p.roi_y_max_px]), cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array(p.hsv_low, np.uint8), np.array(p.hsv_high, np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(mask)
    keep = [k for k in range(1, n)
            if p.min_area_px <= st[k, 4] <= p.max_area_px
            and min(st[k, 2], st[k, 3]) / max(st[k, 2], st[k, 3]) >= p.min_aspect]
    return cen[keep] if keep else np.empty((0, 2))


def _drop_static(dets: list[tuple[float, np.ndarray]], grid_px: int = 4,
                 max_repeats: int = 8) -> list[tuple[float, np.ndarray]]:
    keys = [tuple((c // grid_px).astype(int)) for _, cs in dets for c in cs]
    counts = pd.Series(keys).value_counts() if keys else pd.Series(dtype=int)
    static = set(counts[counts >= max_repeats].index)
    return [(t, np.array([c for c in cs if tuple((c // grid_px).astype(int)) not in static])
             .reshape(-1, 2)) for t, cs in dets]


def track_falling_ball(dets: list[tuple[float, np.ndarray]], p: BallParams) -> pd.DataFrame:
    """Sigue la pelota en caída desde que reaparece por arriba hasta que deja la trayectoria.

    Se conserva el seguimiento descendente más largo (el ascenso del lanzamiento se ignora).

    `dets`: lista ordenada de (t, centroides). Devuelve el seguimiento más largo (t, x, y).
    """
    best: list[tuple[float, float, float]] = []
    used: set[tuple[int, int]] = set()
    for i0, (t0, cs0) in enumerate(dets):
        for j0, c0 in enumerate(cs0):
            if (i0, j0) in used or c0[1] > p.max_start_y_px:
                continue
            track, missing = [(t0, *c0)], 0
            used.add((i0, j0))
            for i in range(i0 + 1, len(dets)):
                t, cs = dets[i]
                tt = np.array([q[0] for q in track])
                xy = np.array([q[1:] for q in track])
                if len(track) >= 3:
                    deg = 2 if len(track) >= 5 else 1
                    pred = np.array([np.polyval(np.polyfit(tt, xy[:, k], deg), t) for k in (0, 1)])
                else:
                    pred = xy[-1] + (xy[-1] - xy[-2]) * (t - tt[-1]) / (tt[-1] - tt[-2]) \
                        if len(track) == 2 else xy[-1]
                gate = p.gate_px * (1 + missing) * (3.0 if len(track) < 5 else 1.0)
                d = np.hypot(*(cs - pred).T) if len(cs) else np.array([])
                if d.size and d.min() <= gate:
                    j = int(d.argmin())
                    track.append((t, *cs[j]))
                    used.add((i, j))
                    missing = 0
                else:
                    missing += 1
                    if missing > p.max_missing:
                        break
            falling = track[-1][2] > track[0][2] + 20      # y en px crece hacia abajo
            if falling and len(track) > len(best):
                best = track
    return pd.DataFrame(best, columns=["t", "x", "y"])


def ball_impact_time(info: VideoInfo, times: np.ndarray, t_guess: float, p: BallParams,
                     before_s: float = 0.6, after_s: float = 0.4) -> dict:
    """Impacto = último punto de la caída de la pelota en torno a `t_guess`.

    Devuelve t_impacto (NaN si no se pudo seguir la pelota) y datos de calidad.
    """
    f0 = int(np.searchsorted(times, t_guess - before_s))
    n = int((before_s + after_s) * info.fps)
    dets = [(float(times[i]), ball_candidates(img, p))
            for i, img in iter_frames(info, f0, n, times=times)]
    dets = _drop_static(dets)
    tr = track_falling_ball(dets, p)
    if len(tr) < p.min_track:
        return dict(t_impacto=np.nan, n_puntos=len(tr), metodo="pelota_no_seguida")
    return dict(t_impacto=float(tr["t"].iloc[-1]), n_puntos=len(tr), metodo="fin_caida_pelota",
                y_contacto_px=float(tr["y"].iloc[-1]), x_contacto_px=float(tr["x"].iloc[-1]))
