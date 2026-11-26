"""[M8] Velocidad de la mano con el Jacobiano:  v = v_hombro + J(θ) ω  (brazo planar de 2 eslabones).

Ángulos absolutos θ1 (brazo) y θ2 (antebrazo) respecto a la horizontal, y hacia arriba.
OJO: la validación frente a la derivada directa de la muñeca es casi una identidad (las longitudes
salen de los mismos marcadores); lo informativo es el CV de las longitudes (¿rigidez/perspectiva?)
y la descomposición por articulación.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def segment_lengths(df: pd.DataFrame, prox: str, dist: str) -> dict:
    d = np.hypot(df[f"x_{dist}"] - df[f"x_{prox}"], df[f"y_{dist}"] - df[f"y_{prox}"]).to_numpy()
    return dict(L=float(np.median(d)), cv=float(np.std(d) / np.mean(d)), l_min=float(d.min()),
                l_max=float(d.max()))


def jacobian_2link(th1, th2, L1: float, L2: float) -> np.ndarray:
    th1, th2 = np.asarray(th1, float), np.asarray(th2, float)
    J = np.empty((len(th1), 2, 2))
    J[:, 0, 0], J[:, 0, 1] = -L1 * np.sin(th1), -L2 * np.sin(th2)
    J[:, 1, 0], J[:, 1, 1] = L1 * np.cos(th1), L2 * np.cos(th2)
    return J


def hand_velocity(J: np.ndarray, w1, w2, v_shoulder: np.ndarray | None = None) -> np.ndarray:
    w = np.stack([np.asarray(w1, float), np.asarray(w2, float)], axis=1)[..., None]
    v = (J @ w)[..., 0]
    return v if v_shoulder is None else v + v_shoulder


def contributions(J: np.ndarray, w1, w2, v_hand: np.ndarray) -> pd.DataFrame:
    """Aporte de cada articulación a la velocidad de la mano, proyectado sobre su dirección.

    frac_* = (c_i · v) / |v|².  Con hombro fijo, frac_hombro + frac_codo = 1.
    """
    c1 = J[:, :, 0] * np.asarray(w1, float)[:, None]
    c2 = J[:, :, 1] * np.asarray(w2, float)[:, None]
    den = (v_hand ** 2).sum(axis=1)
    ok = den > 1e-9
    f1, f2 = np.full(len(den), np.nan), np.full(len(den), np.nan)
    f1[ok], f2[ok] = (c1 * v_hand).sum(1)[ok] / den[ok], (c2 * v_hand).sum(1)[ok] / den[ok]
    return pd.DataFrame(dict(contrib_hombro_x=c1[:, 0], contrib_hombro_y=c1[:, 1],
                             contrib_codo_x=c2[:, 0], contrib_codo_y=c2[:, 1],
                             frac_hombro=f1, frac_codo=f2))


def jacobian_from_kinematics(kin: pd.DataFrame, filt: pd.DataFrame, upper="brazo", fore="antebrazo",
                             shoulder="hombro", elbow="codo", wrist="muneca",
                             cv_warn: float = 0.05) -> tuple[pd.DataFrame, dict]:
    """Calcula v_J, contribuciones y métricas de validación. `kin` y `filt` comparten la malla."""
    l1 = segment_lengths(filt, shoulder, elbow)
    l2 = segment_lengths(filt, elbow, wrist)
    J = jacobian_2link(kin[f"theta_seg_{upper}"], kin[f"theta_seg_{fore}"], l1["L"], l2["L"])
    vs = np.stack([kin[f"vx_{shoulder}"], kin[f"vy_{shoulder}"]], axis=1)
    vJ = hand_velocity(J, kin[f"omega_seg_{upper}"], kin[f"omega_seg_{fore}"], vs)
    vd = np.stack([kin[f"vx_{wrist}"], kin[f"vy_{wrist}"]], axis=1)
    out = pd.DataFrame({"time_s": kin["time_s"], "vx_J": vJ[:, 0], "vy_J": vJ[:, 1],
                        "speed_J": np.hypot(*vJ.T), "speed_directa": np.hypot(*vd.T)})
    out = pd.concat([out, contributions(J, kin[f"omega_seg_{upper}"], kin[f"omega_seg_{fore}"], vJ - vs)],
                    axis=1)
    pico = float(out["speed_directa"].max())
    rmse = float(np.sqrt(np.mean((out["speed_J"] - out["speed_directa"]) ** 2)))
    met = dict(L_brazo=l1["L"], cv_brazo=l1["cv"], L_antebrazo=l2["L"], cv_antebrazo=l2["cv"],
               rmse_speed=rmse, rmse_rel_pico=rmse / pico if pico else np.nan,
               corr_speed=float(np.corrcoef(out["speed_J"], out["speed_directa"])[0, 1]),
               pico_speed_directa=pico, pico_speed_J=float(out["speed_J"].max()),
               longitud_rigida=bool(max(l1["cv"], l2["cv"]) <= cv_warn),
               nota=("" if max(l1["cv"], l2["cv"]) <= cv_warn else
                     "CV de longitud alto: tracking inestable o movimiento fuera de plano (perspectiva)"))
    return out, met
