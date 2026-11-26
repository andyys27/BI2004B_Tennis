"""Análisis de cuerpo completo: centro de masa, fuerza del suelo estimada, salto, posturas,
rango de movimiento, palancas y potencia articular.

Todo en el plano de cada cámara (2D), en m, s, N; y hacia arriba con el origen en el suelo
(calibración `ground_plane`).

- Centro de masa (CoM): media ponderada de los centros de masa segmentarios (fracciones de Winter,
  `body.segments` en la configuración). Si falta un segmento en un fotograma, los pesos se
  renormalizan con los disponibles (lo registra `com_frac_masa`).
- Fuerza vertical del suelo estimada: F = m·(a_CoM,y + g), con el CoM filtrado a `body.com_cutoff_hz`.
  Es la fuerza total sobre el cuerpo (suma de ambos pies); en vuelo debe ser ≈ 0.
- Tiempo de vuelo: NO se reporta. Con marcadores 2D la altura de los pies no es fiable (el pie de
  aterrizaje sin estampita, la pierna trasera que se eleva, el desplazamiento en profundidad con una
  escala fija) y la aceleración del CoM cerca del impacto está dominada por errores del brazo.
  En su lugar: elevación máxima del CoM y velocidad vertical máxima del CoM (fin del impulso).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..processing.filtering import butter_lp_gaps, derivative
from ..processing.kinematics import trim_edges, unwrap_nan
from .levers import horizontal_offset_to_vertical_line

G = 9.81

# Segmento: (masa/M, extremo proximal, extremo distal, fracción del CoM desde el proximal).
# Extremos "medio_hombros"/"medio_caderas" son puntos medios izquierda-derecha.
DEFAULT_SEGMENTS = {
    "cabeza": (0.081, "nariz", "nariz", 0.0),
    "tronco": (0.497, "medio_hombros", "medio_caderas", 0.50),
    "brazo_{s}": (0.028, "hombro_{s}", "codo_{s}", 0.436),
    "antebrazo_{s}": (0.016, "codo_{s}", "muneca_{s}", 0.430),
    "mano_{s}": (0.006, "muneca_{s}", "indice_{s}", 0.506),
    "muslo_{s}": (0.100, "cadera_{s}", "rodilla_{s}", 0.433),
    "pierna_{s}": (0.0465, "rodilla_{s}", "tobillo_{s}", 0.433),
    "pie_{s}": (0.0145, "talon_{s}", "punta_{s}", 0.50),
}
FOOT = ("talon_izq", "talon_der", "punta_izq", "punta_der")


def _point(m: pd.DataFrame, name: str) -> np.ndarray:
    if name.startswith("medio_"):
        base = {"medio_hombros": "hombro", "medio_caderas": "cadera"}[name]
        return 0.5 * (_point(m, f"{base}_izq") + _point(m, f"{base}_der"))
    return m[[f"x_{name}", f"y_{name}"]].to_numpy(float)


def expand_segments(segments: dict | None = None) -> dict:
    out = {}
    for name, (frac, p, d, c) in (segments or DEFAULT_SEGMENTS).items():
        sides = ("izq", "der") if "{s}" in name else ("",)
        for s in sides:
            out[name.format(s=s)] = (float(frac), p.format(s=s), d.format(s=s), float(c))
    total = sum(v[0] for v in out.values())
    if not 0.95 <= total <= 1.05:
        raise ValueError(f"Las fracciones de masa suman {total:.3f}, se esperaba ~1")
    return out


def center_of_mass(markers: pd.DataFrame, segments: dict | None = None) -> pd.DataFrame:
    """x_com, y_com y fracción de la masa con segmentos disponibles en cada fotograma."""
    segs = expand_segments(segments)
    num = np.zeros((len(markers), 2))
    den = np.zeros(len(markers))
    for frac, p, d, c in segs.values():
        a, b = _point(markers, p), _point(markers, d)
        com = a + c * (b - a)
        ok = np.isfinite(com).all(axis=1)
        num[ok] += frac * com[ok]
        den[ok] += frac
    total = sum(v[0] for v in segs.values())
    with np.errstate(invalid="ignore", divide="ignore"):
        xy = num / den[:, None]
    frac_ok = den / total
    xy[frac_ok < 0.8] = np.nan                      # demasiados segmentos ausentes
    return pd.DataFrame({"x_com": xy[:, 0], "y_com": xy[:, 1], "com_frac_masa": frac_ok})


def com_kinetics(markers: pd.DataFrame, fs: float, body_mass_kg: float, cutoff_hz: float = 6.0,
                 segments: dict | None = None) -> pd.DataFrame:
    """CoM, su velocidad/aceleración y fuerza vertical del suelo estimada."""
    com = center_of_mass(markers, segments)
    out = pd.DataFrame({"time_s": markers["time_s"].to_numpy()})
    for ax in ("x", "y"):
        pos = butter_lp_gaps(com[f"{ax}_com"].to_numpy(), fs, cutoff_hz)
        v = trim_edges(derivative(pos, fs), pos)
        a = trim_edges(derivative(v, fs), v)
        out[f"{ax}_com"], out[f"v{ax}_com"], out[f"a{ax}_com"] = pos, v, a
    out["com_frac_masa"] = com["com_frac_masa"].to_numpy()
    out["grf_z_N"] = body_mass_kg * (out["ay_com"] + G)
    out["grf_z_bw"] = out["grf_z_N"] / (body_mass_kg * G)
    return out


def com_serve_metrics(ck: pd.DataFrame, t_ini: float, t_load: float, t_impact: float,
                      body_mass_kg: float, setup_s: float = 0.3, peak_window_s: float = 0.3) -> dict:
    """Elevación del CoM, velocidad vertical máxima e impulso de piernas de un saque.

    - Referencia: altura mediana del CoM en los `setup_s` que siguen a `t_ini` (el pipeline pasa
      los 0.3 s de quietud previos al inicio del gesto).
    - Elevación máxima: máximo del CoM en impacto ± `peak_window_s` menos la referencia.
    - Fin del impulso: instante de máxima velocidad vertical del CoM entre la máxima flexión de
      rodilla (`t_load`) y el pico del CoM (aproxima el despegue).
    - Pico de fuerza estimada e impulso neto ∫(F − mg)dt entre `t_load` y ese instante;
      `v_por_impulso` = impulso/m sirve de control frente a la velocidad medida.
    """
    t = ck["time_s"].to_numpy()
    y, vy = ck["y_com"].to_numpy(), ck["vy_com"].to_numpy()
    nan = dict(com_referencia_m=np.nan, elevacion_max_com_m=np.nan, t_pico_com_rel_s=np.nan,
               vy_max_com_m_s=np.nan, t_fin_impulso_rel_s=np.nan, grf_pico_bw=np.nan,
               impulso_neto_Ns=np.nan, v_por_impulso_m_s=np.nan, desplazamiento_x_com_m=np.nan)
    ref_m = (t >= t_ini) & (t < t_ini + setup_s)
    win = (t >= t_impact - peak_window_s) & (t <= t_impact + peak_window_s)
    if not (np.isfinite(y[ref_m]).any() and np.isfinite(y[win]).any()):
        return nan
    ref = float(np.nanmedian(y[ref_m]))
    i_pk = int(np.nanargmax(np.where(win, y, np.nan)))
    out = dict(nan, com_referencia_m=ref, elevacion_max_com_m=float(y[i_pk] - ref),
               t_pico_com_rel_s=float(t[i_pk] - t_impact))
    x = ck["x_com"].to_numpy()
    if np.isfinite(x[ref_m]).any() and np.isfinite(x[i_pk]):
        out["desplazamiento_x_com_m"] = float(x[i_pk] - np.nanmedian(x[ref_m]))
    push = (t >= t_load) & (t <= t[i_pk])
    if not (np.isfinite(t_load) and np.isfinite(vy[push]).sum() >= 3):
        return out
    i_v = int(np.nanargmax(np.where(push, vy, np.nan)))
    seg = (t >= t_load) & (t <= t[i_v])
    dt = float(np.median(np.diff(t)))
    net = (ck.loc[seg, "grf_z_N"] - body_mass_kg * G).fillna(0).to_numpy()
    imp = float(net.sum() * dt)
    out.update(vy_max_com_m_s=float(vy[i_v]), t_fin_impulso_rel_s=float(t[i_v] - t_impact),
               grf_pico_bw=float(ck.loc[seg, "grf_z_bw"].max()), impulso_neto_Ns=imp,
               v_por_impulso_m_s=imp / body_mass_kg)
    return out


def stance(markers: pd.DataFrame, t: float, side_a: str = "tobillo_izq",
           side_b: str = "tobillo_der") -> float:
    """Separación horizontal entre tobillos en el instante t (base de apoyo en el plano)."""
    i = int(np.argmin(np.abs(markers["time_s"].to_numpy() - t)))
    return float(abs(markers[f"x_{side_a}"].iloc[i] - markers[f"x_{side_b}"].iloc[i]))


def _continuous_line_angle(a: np.ndarray, max_gap: int = 36) -> np.ndarray:
    """Ángulo de línea (periodo π) continuo dentro de bloques, anclado cerca de la horizontal.

    Los tramos finitos separados por huecos de <= `max_gap` muestras (p. ej. el giro rápido en el
    impacto) forman un bloque y se desenvuelven juntos, así una inclinación real que pasa por la
    vertical no salta 180°. Cada bloque (separado por un hueco largo, cuando el cuerpo queda de
    perfil a la cámara) se ancla para que su mediana quede en ±90°: tras un hueco largo no se
    puede saber cuántas medias vueltas giró la línea proyectada.
    """
    a = np.asarray(a, float).copy()
    ok = np.isfinite(a)
    idx = np.flatnonzero(ok)
    if idx.size < 2:
        return a
    breaks = np.flatnonzero(np.diff(idx) > max_gap + 1) + 1
    for block in np.split(idx, breaks):
        u = 0.5 * np.unwrap(2 * a[block])
        u -= np.round(np.median(u) / np.pi) * np.pi
        a[block] = u
    return a


def derived_angles(kin: pd.DataFrame, markers: pd.DataFrame | None = None,
                   min_line_frac: float = 0.5) -> pd.DataFrame:
    """Inclinaciones legibles a partir de los ángulos absolutos de segmento (rad).

    - theta_incl_tronco: tronco respecto a la vertical (0 = erguido), envuelto a ±π.
    - theta_incl_linea_*: línea de hombros/caderas respecto a la horizontal, continua (periodo π).
    - theta_sep_hombros_caderas: diferencia entre ambas líneas (aprox. 2D de la separación
      hombros-caderas; en la vista posterior refleja inclinación lateral más que rotación).

    Si se pasan `markers`, la inclinación de una línea se anula cuando su longitud proyectada es
    menor que `min_line_frac` de su p95: con el cuerpo de perfil a la cámara la línea se ve casi
    como un punto y su ángulo es solo ruido (una línea horizontal que gira cambia de largo, no de
    inclinación).
    """
    out = kin.copy()
    if "theta_seg_tronco" in kin:
        out["theta_incl_tronco"] = (kin["theta_seg_tronco"] - np.pi / 2 + np.pi) % (2 * np.pi) - np.pi
    for line in ("linea_hombros", "linea_caderas"):
        c = f"theta_seg_{line}"
        if c in kin:
            # Una línea no tiene sentido (izq->der o der->izq da igual): periodo π. Se desenvuelve
            # con ese periodo para que no salte 180° cuando la línea pasa por la vertical al girar,
            # y se centra en la mediana del registro (≈ 0 = horizontal).
            a = 0.5 * unwrap_nan(2 * kin[c].to_numpy(float))
            med = np.nanmedian(a)
            out[f"theta_incl_{line}"] = a - np.round(med / np.pi) * np.pi
            ends = {"linea_hombros": "hombro", "linea_caderas": "cadera"}[line]
            if markers is not None and f"x_{ends}_izq" in markers:
                L = np.hypot(markers[f"x_{ends}_izq"] - markers[f"x_{ends}_der"],
                             markers[f"y_{ends}_izq"] - markers[f"y_{ends}_der"]).to_numpy(float)
                short = ~(L >= min_line_frac * np.nanpercentile(L, 95))
                out.loc[short, f"theta_incl_{line}"] = np.nan
                out[f"theta_incl_{line}"] = _continuous_line_angle(out[f"theta_incl_{line}"].to_numpy())
    if {"theta_incl_linea_hombros", "theta_incl_linea_caderas"} <= set(out):
        out["theta_sep_hombros_caderas"] = out["theta_incl_linea_hombros"] - out["theta_incl_linea_caderas"]
    return out


def values_at_events(kin: pd.DataFrame, events: dict[str, float], columns: list[str],
                     to_degrees: bool = True) -> pd.DataFrame:
    """Valor de cada columna en cada evento (muestra más cercana)."""
    t = kin["time_s"].to_numpy()
    rows = []
    for ev, te in events.items():
        if not np.isfinite(te):
            continue
        i = int(np.argmin(np.abs(t - te)))
        if abs(t[i] - te) > 0.05:
            continue
        for c in columns:
            if c in kin:
                v = float(kin[c].iloc[i])
                rows.append(dict(evento=ev, t=te, variable=c,
                                 valor=np.degrees(v) if to_degrees and c.startswith("theta") else v))
    return pd.DataFrame(rows)


def range_of_motion(kin: pd.DataFrame, t0: float, t1: float, columns: list[str]) -> pd.DataFrame:
    m = (kin["time_s"] >= t0) & (kin["time_s"] <= t1)
    rows = []
    for c in columns:
        if c not in kin:
            continue
        y = np.degrees(kin.loc[m, c].to_numpy(float))
        if np.isfinite(y).sum() < 3:
            continue
        rows.append(dict(variable=c, min_deg=float(np.nanmin(y)), max_deg=float(np.nanmax(y)),
                         rom_deg=float(np.nanmax(y) - np.nanmin(y)),
                         frac_valida=float(np.isfinite(y).mean())))
    return pd.DataFrame(rows)


def body_weight_levers(markers: pd.DataFrame, ck: pd.DataFrame, t: float, joints: list[str],
                       body_mass_kg: float) -> pd.DataFrame:
    """Brazo de palanca del peso corporal (línea vertical por el CoM) respecto a cada articulación.

    Momento estático del peso = m·g·brazo: indica qué tan "cargada" está la articulación por la
    postura (positivo = CoM delante de la articulación en +x). Ignora fuerzas inerciales y musculares.
    """
    i = int(np.argmin(np.abs(markers["time_s"].to_numpy() - t)))
    x_com = float(ck["x_com"].iloc[i])
    rows = []
    for j in joints:
        if f"x_{j}" not in markers:
            continue
        p = markers[[f"x_{j}", f"y_{j}"]].iloc[i].to_numpy(float)
        arm = float(horizontal_offset_to_vertical_line(p, x_com)[0])
        rows.append(dict(articulacion=j, t=t, brazo_palanca_m=arm,
                         momento_peso_Nm=body_mass_kg * G * arm))
    return pd.DataFrame(rows)


JOINT_POWER = {   # momento -> (ω segmento distal, ω segmento proximal)
    "T_elbow_Nm": ("antebrazo", "brazo"),
    "T_shoulder_Nm": ("brazo", "tronco"),
    "T_ankle_Nm": ("pie", "pierna"),
    "T_knee_Nm": ("pierna", "muslo"),
    "T_hip_Nm": ("muslo", "tronco"),
}


def joint_power(idd: pd.DataFrame, kin: pd.DataFrame) -> pd.DataFrame:
    """Potencia articular P = M·(ω_distal − ω_proximal) (exploratoria; signos según convención 2D)."""
    k = kin.set_index(kin["time_s"].round(6))
    t = idd["time_s"].round(6)
    out = pd.DataFrame({"time_s": idd["time_s"]})
    for col, (dist, prox) in JOINT_POWER.items():
        cd, cp = f"omega_seg_{dist}", f"omega_seg_{prox}"
        if col in idd and cd in k and cp in k:
            w = (k[cd] - k[cp]).reindex(t).to_numpy()
            out[col.replace("T_", "P_").replace("_Nm", "_W")] = idd[col].to_numpy() * w
    return out


def peak_table(df: pd.DataFrame, prefix: str, unit: str) -> pd.DataFrame:
    rows = []
    for c in [c for c in df if c.startswith(prefix)]:
        y = df[c].to_numpy(float)
        if np.isfinite(y).any():
            i = int(np.nanargmax(np.abs(y)))
            rows.append(dict(variable=c, pico=float(y[i]), pico_abs=float(abs(y[i])),
                             t=float(df["time_s"].iloc[i]), unidad=unit))
    return pd.DataFrame(rows)
