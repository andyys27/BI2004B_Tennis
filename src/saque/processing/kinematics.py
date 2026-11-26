"""Ángulos de segmentos/articulaciones, ω, α y cinemática lineal.

Convención: ángulo absoluto = atan2(dy, dx) del vector proximal→distal respecto a la horizontal,
con y hacia arriba (las coordenadas calibradas ya vienen así; flip_y solo para datos con y hacia abajo).
Todas las unidades angulares salen en rad, rad/s y rad/s².
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .filtering import derivative, finite_runs


def unwrap_nan(a) -> np.ndarray:
    """np.unwrap por tramos finitos (np.unwrap propaga un NaN a todo lo que sigue)."""
    a = np.asarray(a, float)
    out = np.full_like(a, np.nan)
    for i, j in finite_runs(a):
        out[i:j] = np.unwrap(a[i:j])
    return out


def _xy(df, m, flip_y=False):
    x, y = df[f"x_{m}"].to_numpy(float), df[f"y_{m}"].to_numpy(float)
    return x, (-y if flip_y else y)


def segment_vector(df, prox: str, dist: str, flip_y=False) -> np.ndarray:
    xp, yp = _xy(df, prox, flip_y)
    xd, yd = _xy(df, dist, flip_y)
    return np.stack([xd - xp, yd - yp], axis=1)


def abs_angle(v: np.ndarray) -> np.ndarray:
    """Ángulo absoluto respecto a la horizontal, desenvuelto (np.unwrap)."""
    return unwrap_nan(np.arctan2(v[:, 1], v[:, 0]))


def interior_angle(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Ángulo entre dos vectores que salen de la articulación, en [0, π]."""
    cross = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    dot = (u * v).sum(axis=1)
    return np.arctan2(np.abs(cross), dot)


def signed_joint_angle(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Ángulo con signo (v respecto a u), desenvuelto."""
    cross = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    return unwrap_nan(np.arctan2(cross, (u * v).sum(axis=1)))


EDGE_TRIM = 3   # muestras descartadas junto a cada hueco (borde de filtro y de derivada)


def trim_edges(d: np.ndarray, ref: np.ndarray, k: int = EDGE_TRIM) -> np.ndarray:
    """NaN en las k muestras de cada extremo de los tramos finitos de `ref`.

    El filtrado por tramos y las diferencias finitas son poco fiables en los bordes de un hueco:
    sin esto aparecen picos espurios de ω y α justo donde se pierde o se recupera un marcador.
    Los extremos del registro completo no se recortan.
    """
    d = np.asarray(d, float).copy()
    n = len(d)
    d[~np.isfinite(np.asarray(ref, float))] = np.nan
    for a, b in finite_runs(ref):
        if a > 0:
            d[a:min(a + k, b)] = np.nan
        if b < n:
            d[max(b - k, a):b] = np.nan
    return d


def angular_derivatives(theta: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    w, a = derivative(theta, fs, 1), derivative(theta, fs, 2)
    return trim_edges(w, theta), trim_edges(a, theta)


def compute_kinematics(df: pd.DataFrame, fs: float, segments: dict | None = None,
                       joints: dict | None = None, linear: list[str] | None = None,
                       angle_cols: list[str] | None = None, flip_y: bool = False) -> pd.DataFrame:
    """Calcula θ, ω, α para segmentos/articulaciones/columnas angulares y velocidad lineal.

    segments: {"brazo": ["hombro", "codo"]}            -> theta_seg_brazo, omega_seg_brazo, alpha_seg_brazo
    joints:   {"codo": ["hombro", "codo", "muneca"]}  -> [proximal, articulación, distal]
              genera theta_interior_<nombre> y theta_signed_<nombre> (+ omega, alpha)
    linear:   marcadores para x, y, vx, vy, speed, ax, ay
    angle_cols: columnas de ángulo en rad ya presentes (se derivan tal cual)
    """
    out = pd.DataFrame({"time_s": df["time_s"].to_numpy()})
    for name, (p, d) in (segments or {}).items():
        th = abs_angle(segment_vector(df, p, d, flip_y))
        w, a = angular_derivatives(th, fs)
        out[f"theta_seg_{name}"], out[f"omega_seg_{name}"], out[f"alpha_seg_{name}"] = th, w, a
    for name, (p, j, d) in (joints or {}).items():
        u = segment_vector(df, j, p, flip_y)    # articulación -> proximal
        v = segment_vector(df, j, d, flip_y)    # articulación -> distal
        for kind, th in (("interior", interior_angle(u, v)), ("signed", signed_joint_angle(u, v))):
            w, a = angular_derivatives(th, fs)
            out[f"theta_{kind}_{name}"], out[f"omega_{kind}_{name}"], out[f"alpha_{kind}_{name}"] = th, w, a
    for c in angle_cols or []:
        th = unwrap_nan(df[c].to_numpy(float))
        w, a = angular_derivatives(th, fs)
        base = c.replace("_rad", "")
        out[f"theta_{base}"], out[f"omega_{base}"], out[f"alpha_{base}"] = th, w, a
    for m in linear or []:
        x, y = _xy(df, m, flip_y)
        out[f"x_{m}"], out[f"y_{m}"] = x, y
        vx, vy = trim_edges(derivative(x, fs), x), trim_edges(derivative(y, fs), y)
        out[f"vx_{m}"], out[f"vy_{m}"], out[f"speed_{m}"] = vx, vy, np.hypot(vx, vy)
        out[f"ax_{m}"] = trim_edges(derivative(vx, fs), vx)
        out[f"ay_{m}"] = trim_edges(derivative(vy, fs), vy)
    out.attrs["coord_unit"] = df.attrs.get("coord_unit")
    return out
