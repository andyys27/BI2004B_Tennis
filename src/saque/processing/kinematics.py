"""[M6] Ángulos de segmentos/articulaciones, ω, α y cinemática lineal.

Convención: ángulo absoluto = atan2(dy, dx) del vector proximal→distal respecto a la horizontal,
con y hacia arriba (si el CSV viene con y hacia abajo, se aplica flip_y).
Todas las unidades angulares salen en rad, rad/s y rad/s².
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .filtering import derivative


def _xy(df, m, flip_y=True):
    x, y = df[f"x_{m}"].to_numpy(float), df[f"y_{m}"].to_numpy(float)
    return x, (-y if flip_y else y)


def segment_vector(df, prox: str, dist: str, flip_y=True) -> np.ndarray:
    xp, yp = _xy(df, prox, flip_y)
    xd, yd = _xy(df, dist, flip_y)
    return np.stack([xd - xp, yd - yp], axis=1)


def abs_angle(v: np.ndarray) -> np.ndarray:
    """Ángulo absoluto respecto a la horizontal, desenvuelto (np.unwrap)."""
    return np.unwrap(np.arctan2(v[:, 1], v[:, 0]))


def interior_angle(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Ángulo entre dos vectores que salen de la articulación, en [0, π]."""
    cross = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    dot = (u * v).sum(axis=1)
    return np.arctan2(np.abs(cross), dot)


def signed_joint_angle(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Ángulo con signo (v respecto a u), desenvuelto."""
    cross = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    return np.unwrap(np.arctan2(cross, (u * v).sum(axis=1)))


def angular_derivatives(theta: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    return derivative(theta, fs, 1), derivative(theta, fs, 2)


def compute_kinematics(df: pd.DataFrame, fs: float, segments: dict | None = None,
                       joints: dict | None = None, linear: list[str] | None = None,
                       angle_cols: list[str] | None = None, flip_y: bool = True) -> pd.DataFrame:
    """Calcula θ, ω, α para segmentos/articulaciones/columnas angulares y velocidad lineal.

    segments: {"brazo": ["hombro", "codo"]}            -> theta_seg_brazo, omega_seg_brazo, alpha_seg_brazo
    joints:   {"codo": ["hombro", "codo", "muneca"]}  -> [proximal, articulación, distal]
              genera theta_interior_<nombre> y theta_signed_<nombre> (+ omega, alpha)
    linear:   marcadores para vx, vy, speed, ax, ay
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
        th = np.unwrap(df[c].to_numpy(float))
        w, a = angular_derivatives(th, fs)
        base = c.replace("_rad", "")
        out[f"theta_{base}"], out[f"omega_{base}"], out[f"alpha_{base}"] = th, w, a
    for m in linear or []:
        x, y = _xy(df, m, flip_y)
        vx, vy = derivative(x, fs), derivative(y, fs)
        out[f"vx_{m}"], out[f"vy_{m}"], out[f"speed_{m}"] = vx, vy, np.hypot(vx, vy)
        out[f"ax_{m}"], out[f"ay_{m}"] = derivative(vx, fs), derivative(vy, fs)
    out.attrs["coord_unit"] = df.attrs.get("coord_unit")
    return out


def wrap_deg(d):
    return (np.asarray(d, float) + 180.0) % 360.0 - 180.0


def compare_with_kinovea(theta_rad: np.ndarray, kinovea_deg: np.ndarray) -> dict:
    """RMSE (°) entre un ángulo recalculado y el de Kinovea, probando ángulo directo y suplementario."""
    mine = np.degrees(theta_rad)
    k = np.asarray(kinovea_deg, float)
    direct = np.sqrt(np.mean(wrap_deg(mine - k) ** 2))
    suppl = np.sqrt(np.mean(wrap_deg((180.0 - mine) - k) ** 2))
    return dict(rmse_directo_deg=float(direct), rmse_suplementario_deg=float(suppl),
                convencion=("directo" if direct <= suppl else "suplementario"))
