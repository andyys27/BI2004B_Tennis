"""Datos sintéticos con verdad conocida para las pruebas.

- `arm_markers`: brazo planar de 2 eslabones (hombro fijo) con θ1(t), θ2(t) analíticos.
- `pose_frame`: tabla de pose con el formato de MediaPipe (px, y hacia abajo) para N saques
  simulados: la muñeca sube por encima de la cabeza en cada impacto y la cadera "salta".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from saque.video.pose import LANDMARKS

FS = 120.0
L1, L2 = 0.30, 0.28
SHOULDER = (0.50, 1.40)


def th1(t): return 1.0 + 0.6 * np.sin(2 * np.pi * 0.7 * t)
def th2(t): return 0.4 + 0.9 * np.sin(2 * np.pi * 0.9 * t + 0.5)
def w1(t): return 0.6 * 2 * np.pi * 0.7 * np.cos(2 * np.pi * 0.7 * t)
def a1(t): return -0.6 * (2 * np.pi * 0.7) ** 2 * np.sin(2 * np.pi * 0.7 * t)


def arm_markers(t, noise=0.0, seed=1) -> pd.DataFrame:
    """Coordenadas en m (y hacia arriba) de hombro, codo y muñeca."""
    rng = np.random.default_rng(seed)
    sx, sy = SHOULDER
    ex, ey = sx + L1 * np.cos(th1(t)), sy + L1 * np.sin(th1(t))
    wx, wy = ex + L2 * np.cos(th2(t)), ey + L2 * np.sin(th2(t))
    d = {"time_s": t}
    for n, (x, y) in dict(hombro=(np.full_like(t, sx), np.full_like(t, sy)),
                          codo=(ex, ey), muneca=(wx, wy)).items():
        d[f"x_{n}"] = x + rng.normal(0, noise, t.size)
        d[f"y_{n}"] = y + rng.normal(0, noise, t.size)
    return pd.DataFrame(d)


def pose_frame(impacts_s=(3.0, 8.0, 13.0), duration_s=16.0, fs=FS, px_per_m=500.0,
               lost_frames=(), seed=0) -> pd.DataFrame:
    """Pose sintética en píxeles con el esquema que guarda `video.pose.extract_pose`.

    Postura de pie; en cada impacto la muñeca derecha sube en forma de campana y las caderas
    se elevan (salto). `lost_frames`: índices de ranura que se omiten (fotogramas perdidos).
    """
    rng = np.random.default_rng(seed)
    slots = np.array([s for s in range(int(duration_s * fs)) if s not in set(lost_frames)])
    t = slots / fs
    base_y = 1800.0
    body = {  # (x en m, altura en m)
        "nariz": (0.0, 1.55), "hombro_izq": (-0.15, 1.35), "hombro_der": (0.15, 1.35),
        "codo_izq": (-0.20, 1.10), "codo_der": (0.20, 1.10),
        "muneca_izq": (-0.22, 0.85), "muneca_der": (0.22, 0.85),
        "indice_izq": (-0.22, 0.78), "indice_der": (0.22, 0.78),
        "cadera_izq": (-0.10, 0.90), "cadera_der": (0.10, 0.90),
        "rodilla_izq": (-0.10, 0.48), "rodilla_der": (0.10, 0.48),
        "tobillo_izq": (-0.10, 0.08), "tobillo_der": (0.10, 0.08),
        "talon_izq": (-0.14, 0.03), "talon_der": (0.14, 0.03),
        "punta_izq": (-0.02, 0.02), "punta_der": (0.18, 0.02),
    }
    bump = sum(np.exp(-0.5 * ((t - c) / 0.15) ** 2) for c in impacts_s)
    jump = sum(np.exp(-0.5 * ((t - c) / 0.20) ** 2) for c in impacts_s)
    d = {"frame": np.arange(len(t)), "time_s": t, "detected": np.ones(len(t), bool),
         "fps": fs, "pts_s": t}
    for name, (x, h) in body.items():
        hh = h + 0.10 * jump
        xx = np.full_like(t, x)
        if name == "muneca_der":
            hh = hh + 1.0 * bump
        if name == "codo_der":
            hh = hh + 0.5 * bump
        d[f"x_{name}"] = 540 + xx * px_per_m + rng.normal(0, 0.5, len(t))
        d[f"y_{name}"] = base_y - hh * px_per_m + rng.normal(0, 0.5, len(t))
        d[f"vis_{name}"] = np.full(len(t), 0.99)
    assert set(body) == set(LANDMARKS)
    return pd.DataFrame(d)
