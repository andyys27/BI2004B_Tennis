"""Dinámica inversa planar (Newton-Euler 2D) exploratoria por saque.

Supuestos explícitos (no hay plataforma de fuerza ni medición de raqueta):
- tren inferior (pie-pierna-muslo): GRF estimada a partir del centro de masa (sin plataforma),
  `grf.mode: com`: F = m·(a_CoM − g) en x y z, repartida entre los dos pies con `grf.share`
  (pies juntos en la carga del saque -> 0.5) y aplicada en un centro de presión fijo entre tobillo
  y punta (`grf.cop_frac`). Contacto cuando la GRF vertical de esa pierna supera
  `grf.contact_bw` del peso. Con `grf.mode: none` se usa GRF = 0 (fase aérea asumida). El
  reparto entre pies y el CoP son supuestos: los momentos del apoyo son estimaciones exploratorias;
- tren superior (brazo-antebrazo): fuerza externa nula en muñeca; mano, raqueta y pelota excluidas.
Longitudes de segmento: mediana de la distancia entre marcadores del saque; masas, centros de
masa y radios de giro: fracciones de Winter configuradas en `inverse_dynamics.anthropometry`.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .jacobian import segment_lengths


@dataclass(frozen=True)
class Segment:
    length_m: float
    mass_kg: float
    com_from_proximal_m: float
    inertia_kgm2: float

    def __post_init__(self):
        if self.length_m <= 0 or self.mass_kg <= 0 or self.inertia_kgm2 < 0:
            raise ValueError("longitud y masa deben ser positivas; inercia no negativa")
        if not 0 <= self.com_from_proximal_m <= self.length_m:
            raise ValueError("el centro de masa debe estar dentro de la longitud del segmento")


@dataclass(frozen=True)
class LowerLimb:
    foot: Segment
    shank: Segment
    thigh: Segment
    gravity_m_s2: float = 9.81

    def __post_init__(self):
        if self.gravity_m_s2 <= 0:
            raise ValueError("gravity_m_s2 debe ser positivo")


@dataclass(frozen=True)
class UpperLimb:
    upper_arm: Segment
    forearm: Segment
    gravity_m_s2: float = 9.81

    def __post_init__(self):
        if self.gravity_m_s2 <= 0:
            raise ValueError("gravity_m_s2 debe ser positivo")


def _segment_from_anthropometry(length_m: float, body_mass_kg: float,
                                name: str, anthropometry: dict) -> Segment:
    mass = body_mass_kg * float(anthropometry[f"{name}_mass_fraction"])
    com = length_m * float(anthropometry[f"{name}_com_fraction"])
    inertia = mass * (length_m * float(anthropometry[f"{name}_gyration_fraction"])) ** 2
    return Segment(length_m, mass, com, inertia)


def _rigid_segment_length_outliers(filtered: pd.DataFrame, segment_markers: list[tuple[str, str]],
                                   threshold_mad: float, margin_samples: int) -> np.ndarray:
    """Marca saltos extremos de longitud y una vecindad para proteger derivadas angulares."""
    invalid = np.zeros(len(filtered), dtype=bool)
    if threshold_mad <= 0 or margin_samples < 0:
        raise ValueError("threshold_mad debe ser positivo y margin_samples no negativo")
    for proximal, distal in segment_markers:
        length = np.hypot(
            filtered[f"x_{distal}"].to_numpy(float) - filtered[f"x_{proximal}"].to_numpy(float),
            filtered[f"y_{distal}"].to_numpy(float) - filtered[f"y_{proximal}"].to_numpy(float),
        )
        median = float(np.nanmedian(length))
        mad = float(np.nanmedian(np.abs(length - median)))
        tolerance = threshold_mad * max(mad, abs(median) * 1e-6, 1e-9)
        invalid |= ~np.isfinite(length) | (np.abs(length - median) > tolerance)

    flagged = np.flatnonzero(invalid)
    for offset in range(1, margin_samples + 1):
        before, after = flagged - offset, flagged + offset
        invalid[before[before >= 0]] = True
        invalid[after[after < len(invalid)]] = True
    return invalid


def _vector(name, values, n):
    a = np.asarray(values, float)
    if a.ndim != 1 or len(a) != n or not np.isfinite(a).all():
        raise ValueError(f"{name} debe ser un vector finito de longitud {n}")
    return a


def _cross2(a, b):
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def inverse_dynamics_2d(time_s, angles, angular_velocity, angular_acceleration,
                        grf_x, grf_z, cop_offset_from_ankle_m, ankle_acceleration,
                        parameters: LowerLimb, contact_threshold_n: float = 20.0) -> pd.DataFrame:
    """Resuelve las reacciones articulares de distal a proximal en el plano x-z.

    Ángulos absolutos en radianes; velocidades en rad/s; aceleraciones angulares
    en rad/s²; coordenadas en metros; fuerzas en N y momentos en N·m.
    `cop_offset_from_ankle_m` es el vector (x,z) de tobillo a centro de presión.
    Las fuerzas/momentos devueltos son los ejercidos por la articulación proximal
    sobre cada segmento, con momento antihorario positivo.
    """
    t = np.asarray(time_s, float)
    if t.ndim != 1 or len(t) == 0 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("time_s debe ser finito y estrictamente creciente")
    n = len(t)
    keys = ("foot", "shank", "thigh")
    th = {k: _vector(f"angles[{k}]", angles[k], n) for k in keys}
    om = {k: _vector(f"angular_velocity[{k}]", angular_velocity[k], n) for k in keys}
    al = {k: _vector(f"angular_acceleration[{k}]", angular_acceleration[k], n) for k in keys}
    gx = _vector("grf_x", grf_x, n)
    gz = _vector("grf_z", grf_z, n)
    cop = np.asarray(cop_offset_from_ankle_m, float)
    abase = np.asarray(ankle_acceleration, float)
    if cop.shape != (n, 2) or not np.isfinite(cop).all():
        raise ValueError("cop_offset_from_ankle_m debe tener forma (N, 2) y ser finito")
    if abase.shape != (n, 2) or not np.isfinite(abase).all():
        raise ValueError("ankle_acceleration debe tener forma (N, 2) y ser finito")
    if contact_threshold_n < 0:
        raise ValueError("contact_threshold_n no puede ser negativo")

    contact = gz > contact_threshold_n
    gx = np.where(contact, gx, 0.0)
    gz = np.where(contact, gz, 0.0)
    gvec = np.array([0.0, -parameters.gravity_m_s2])
    fgrf = np.column_stack([gx, gz])

    def angular_accel(k):
        e = np.column_stack([np.cos(th[k]), np.sin(th[k])])
        e_perp = np.column_stack([-e[:, 1], e[:, 0]])
        return e, al[k][:, None] * e_perp - om[k][:, None] ** 2 * e

    e1, q1 = angular_accel("foot")
    foot = parameters.foot
    a_com_foot = abase + foot.com_from_proximal_m * q1
    f_ankle = (foot.mass_kg * (a_com_foot - gvec) - fgrf)
    r_ankle_com = -foot.com_from_proximal_m * e1
    r_cop_com = cop - foot.com_from_proximal_m * e1
    m_ankle = (foot.inertia_kgm2 * al["foot"] -
               _cross2(r_ankle_com, f_ankle) - _cross2(r_cop_com, fgrf))

    e2, q2 = angular_accel("shank")
    shank = parameters.shank
    d2 = shank.length_m - shank.com_from_proximal_m
    a_com_shank = abase - d2 * q2
    a_knee = abase - shank.length_m * q2
    f_ankle_on_shank = -f_ankle
    f_knee = shank.mass_kg * (a_com_shank - gvec) - f_ankle_on_shank
    r_knee_com = -shank.com_from_proximal_m * e2
    r_ankle_com_shank = d2 * e2
    m_knee = (shank.inertia_kgm2 * al["shank"] -
              _cross2(r_knee_com, f_knee) -
              _cross2(r_ankle_com_shank, f_ankle_on_shank) + m_ankle)

    e3, q3 = angular_accel("thigh")
    thigh = parameters.thigh
    d3 = thigh.length_m - thigh.com_from_proximal_m
    a_com_thigh = a_knee - d3 * q3
    f_knee_on_thigh = -f_knee
    f_hip = thigh.mass_kg * (a_com_thigh - gvec) - f_knee_on_thigh
    r_hip_com = -thigh.com_from_proximal_m * e3
    r_knee_com_thigh = d3 * e3
    m_hip = (thigh.inertia_kgm2 * al["thigh"] -
             _cross2(r_hip_com, f_hip) -
             _cross2(r_knee_com_thigh, f_knee_on_thigh) + m_knee)

    return pd.DataFrame({
        "time_s": t, "contact": contact, "GRF_x_N": gx, "GRF_z_N": gz,
        "Fx_ankle_N": f_ankle[:, 0], "Fz_ankle_N": f_ankle[:, 1], "T_ankle_Nm": m_ankle,
        "Fx_knee_N": f_knee[:, 0], "Fz_knee_N": f_knee[:, 1], "T_knee_Nm": m_knee,
        "Fx_hip_N": f_hip[:, 0], "Fz_hip_N": f_hip[:, 1], "T_hip_Nm": m_hip,
    })


def inverse_dynamics_two_segment_2d(
        time_s, angles, angular_velocity, angular_acceleration,
        proximal_acceleration, middle_acceleration, proximal_segment: Segment,
        distal_segment: Segment, gravity_m_s2: float = 9.81) -> pd.DataFrame:
    """Resuelve una cadena de dos segmentos con extremo distal libre."""
    t = np.asarray(time_s, float)
    if t.ndim != 1 or len(t) == 0 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("time_s debe ser finito y estrictamente creciente")
    n = len(t)
    keys = ("proximal", "distal")
    th = {k: _vector(f"angles[{k}]", angles[k], n) for k in keys}
    om = {k: _vector(f"angular_velocity[{k}]", angular_velocity[k], n) for k in keys}
    al = {k: _vector(f"angular_acceleration[{k}]", angular_acceleration[k], n) for k in keys}
    proximal_a = np.asarray(proximal_acceleration, float)
    middle_a = np.asarray(middle_acceleration, float)
    for name, value in (("proximal_acceleration", proximal_a),
                        ("middle_acceleration", middle_a)):
        if value.shape != (n, 2) or not np.isfinite(value).all():
            raise ValueError(f"{name} debe tener forma (N, 2) y ser finita")

    if gravity_m_s2 <= 0:
        raise ValueError("gravity_m_s2 debe ser positivo")
    gravity = np.array([0.0, -gravity_m_s2])

    def segment_geometry(name):
        e = np.column_stack([np.cos(th[name]), np.sin(th[name])])
        e_perp = np.column_stack([-e[:, 1], e[:, 0]])
        q = al[name][:, None] * e_perp - om[name][:, None] ** 2 * e
        return e, q

    e_distal, q_distal = segment_geometry("distal")
    a_com_distal = middle_a + distal_segment.com_from_proximal_m * q_distal
    f_middle = distal_segment.mass_kg * (a_com_distal - gravity)
    r_middle_com_distal = -distal_segment.com_from_proximal_m * e_distal
    m_middle = (distal_segment.inertia_kgm2 * al["distal"] -
                _cross2(r_middle_com_distal, f_middle))

    e_proximal, q_proximal = segment_geometry("proximal")
    a_com_proximal = proximal_a + proximal_segment.com_from_proximal_m * q_proximal
    f_middle_on_proximal = -f_middle
    f_proximal = (proximal_segment.mass_kg * (a_com_proximal - gravity)
                  - f_middle_on_proximal)
    r_proximal_com = -proximal_segment.com_from_proximal_m * e_proximal
    distal_distance = proximal_segment.length_m - proximal_segment.com_from_proximal_m
    r_middle_com_proximal = distal_distance * e_proximal
    m_proximal = (proximal_segment.inertia_kgm2 * al["proximal"] -
                  _cross2(r_proximal_com, f_proximal) -
                  _cross2(r_middle_com_proximal, f_middle_on_proximal) + m_middle)

    return pd.DataFrame({
        "time_s": t,
        "Fx_middle_N": f_middle[:, 0], "Fz_middle_N": f_middle[:, 1],
        "T_middle_Nm": m_middle,
        "Fx_proximal_N": f_proximal[:, 0], "Fz_proximal_N": f_proximal[:, 1],
        "T_proximal_Nm": m_proximal,
        "distal_external_force_assumed_N": 0.0,
        "proximal_segment_length_m": proximal_segment.length_m,
        "distal_segment_length_m": distal_segment.length_m,
        "proximal_segment_mass_kg": proximal_segment.mass_kg,
        "distal_segment_mass_kg": distal_segment.mass_kg,
    })


def inverse_dynamics_upper_limb_2d(
        time_s, angles, angular_velocity, angular_acceleration,
        shoulder_acceleration, elbow_acceleration, parameters: UpperLimb) -> pd.DataFrame:
    """Calcula reacciones y momentos de hombro/codo suponiendo muñeca sin carga externa."""
    result = inverse_dynamics_two_segment_2d(
        time_s, {"proximal": angles["upper_arm"], "distal": angles["forearm"]},
        {"proximal": angular_velocity["upper_arm"], "distal": angular_velocity["forearm"]},
        {"proximal": angular_acceleration["upper_arm"], "distal": angular_acceleration["forearm"]},
        shoulder_acceleration, elbow_acceleration, parameters.upper_arm, parameters.forearm,
        parameters.gravity_m_s2,
    )
    result = result.rename(columns={
        "Fx_middle_N": "Fx_elbow_N", "Fz_middle_N": "Fz_elbow_N",
        "T_middle_Nm": "T_elbow_Nm", "Fx_proximal_N": "Fx_shoulder_N",
        "Fz_proximal_N": "Fz_shoulder_N", "T_proximal_Nm": "T_shoulder_Nm",
        "distal_external_force_assumed_N": "wrist_external_force_assumed_N",
        "proximal_segment_length_m": "upper_arm_length_m",
        "distal_segment_length_m": "forearm_length_m",
        "proximal_segment_mass_kg": "upper_arm_mass_kg",
        "distal_segment_mass_kg": "forearm_mass_kg",
    })
    result["external_force_mode"] = "Fuerza externa en muñeca=0; mano/raqueta excluidas"
    return result


def _xy_accel(kin: pd.DataFrame, marker: str) -> np.ndarray:
    return np.column_stack([kin[f"ax_{marker}"], kin[f"ay_{marker}"]]).astype(float)


def _seg_series(kin: pd.DataFrame, seg: str, kind: str) -> np.ndarray:
    return kin[f"{kind}_seg_{seg}"].to_numpy(float)


def _valid_rows(kin: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return np.isfinite(kin[cols].to_numpy(float)).all(axis=1)


def estimated_grf(kin: pd.DataFrame, markers: pd.DataFrame, ankle: str, toe: str,
                  body_mass_kg: float, share: float = 0.5, cop_frac: float = 0.5,
                  gravity_m_s2: float = 9.81) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GRF de una pierna estimada con el CoM: F = share·m·(a_CoM + g ẑ); CoP entre tobillo y punta.

    Devuelve (grf_x, grf_z, cop_offset_from_ankle) con NaN donde falte el CoM o el pie.
    """
    if not 0 < share <= 1 or not 0 <= cop_frac <= 1:
        raise ValueError("share debe estar en (0, 1] y cop_frac en [0, 1]")
    gx = share * body_mass_kg * kin["ax_com"].to_numpy(float)
    gz = share * body_mass_kg * (kin["ay_com"].to_numpy(float) + gravity_m_s2)
    a = np.column_stack([markers[f"x_{ankle}"], markers[f"y_{ankle}"]]).astype(float)
    b = np.column_stack([markers[f"x_{toe}"], markers[f"y_{toe}"]]).astype(float)
    return gx, gz, cop_frac * (b - a)


def run_serve_inverse_dynamics(kin: pd.DataFrame, markers: pd.DataFrame, view_cfg: dict,
                               body_mass_kg: float, anthropometry: dict,
                               gravity_m_s2: float = 9.81, grf: dict | None = None
                               ) -> dict[str, pd.DataFrame]:
    """Calcula tren superior y tren inferior para un saque. Omite lo que no tenga marcadores.

    `kin` y `markers` (filtrados, en m) deben compartir la malla temporal. `grf` configura la
    fuerza del suelo del tren inferior (ver docstring del módulo); None = GRF estimada por el CoM.
    """
    grf = {"mode": "com", "share": 0.5, "cop_frac": 0.5, "contact_bw": 0.2, **(grf or {})}
    seg = view_cfg.get("segments", {})
    out: dict[str, pd.DataFrame] = {}

    def segment(name, prox, dist):
        L = segment_lengths(markers, prox, dist)["L"]
        return _segment_from_anthropometry(L, body_mass_kg, name, anthropometry)

    # Tren superior: hombro-codo-muñeca, muñeca libre.
    if {"brazo", "antebrazo"} <= set(seg):
        sh, el = seg["brazo"]
        _, wr = seg["antebrazo"]
        cols = [f"{k}_seg_{s}" for s in ("brazo", "antebrazo") for k in ("theta", "omega", "alpha")]
        cols += [f"a{a}_{m}" for m in (sh, el) for a in "xy"]
        if set(cols) <= set(kin):
            ok = _valid_rows(kin, cols)
            if ok.sum() >= 3:
                k = kin.loc[ok]
                res = inverse_dynamics_upper_limb_2d(
                    k["time_s"].to_numpy(float),
                    {"upper_arm": _seg_series(k, "brazo", "theta"), "forearm": _seg_series(k, "antebrazo", "theta")},
                    {"upper_arm": _seg_series(k, "brazo", "omega"), "forearm": _seg_series(k, "antebrazo", "omega")},
                    {"upper_arm": _seg_series(k, "brazo", "alpha"), "forearm": _seg_series(k, "antebrazo", "alpha")},
                    _xy_accel(k, sh), _xy_accel(k, el),
                    UpperLimb(segment("upper_arm", sh, el), segment("forearm", el, wr), gravity_m_s2))
                out["tren_superior"] = res

    # Tren inferior: pie-pierna-muslo con GRF estimada por el CoM (o GRF=0).
    if {"pie", "pierna", "muslo"} <= set(seg):
        ankle, toe = seg["pie"]
        knee, _ = seg["pierna"]
        hip, _ = seg["muslo"]
        cols = [f"{k}_seg_{s}" for s in ("pie", "pierna", "muslo") for k in ("theta", "omega", "alpha")]
        cols += [f"a{a}_{ankle}" for a in "xy"]
        use_grf = grf["mode"] == "com" and {"ax_com", "ay_com"} <= set(kin)
        if use_grf:
            gx_all, gz_all, cop_all = estimated_grf(kin, markers, ankle, toe, body_mass_kg,
                                                    grf["share"], grf["cop_frac"], gravity_m_s2)
        if set(cols) <= set(kin):
            ok = _valid_rows(kin, cols)
            if use_grf:
                ok &= np.isfinite(gx_all) & np.isfinite(gz_all) & np.isfinite(cop_all).all(axis=1)
            if ok.sum() >= 3:
                k = kin.loc[ok]
                n = len(k)
                if use_grf:
                    gx, gz, cop = gx_all[ok], gz_all[ok], cop_all[ok]
                    thr = grf["contact_bw"] * body_mass_kg * gravity_m_s2 * grf["share"]
                    mode = (f"GRF estimada por el CoM; {grf['share']:.0%} por pierna; CoP a "
                            f"{grf['cop_frac']:.0%} tobillo->punta")
                else:
                    gx, gz, cop, thr = np.zeros(n), np.zeros(n), np.zeros((n, 2)), 20.0
                    mode = "GRF=0; fase aérea asumida"
                names = {"foot": "pie", "shank": "pierna", "thigh": "muslo"}
                limb = LowerLimb(segment("foot", ankle, toe), segment("shank", knee, ankle),
                                 segment("thigh", hip, knee), gravity_m_s2)
                res = inverse_dynamics_2d(
                    k["time_s"].to_numpy(float),
                    {e: _seg_series(k, s, "theta") for e, s in names.items()},
                    {e: _seg_series(k, s, "omega") for e, s in names.items()},
                    {e: _seg_series(k, s, "alpha") for e, s in names.items()},
                    gx, gz, cop, _xy_accel(k, ankle), limb, contact_threshold_n=thr)
                res["external_force_mode"] = mode
                out["tren_inferior"] = res
    return out
