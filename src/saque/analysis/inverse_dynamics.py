"""M12: dinámica inversa planar de pie, pierna y muslo con Newton-Euler."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
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


def winter_parameters(height_m: float, body_mass_kg: float, foot_length_m: float) -> LowerLimb:
    """Construye parámetros iniciales basados en las proporciones documentadas de Winter.

    Son estimaciones de referencia y deben verificarse para cada participante.
    """
    if height_m <= 0 or body_mass_kg <= 0 or foot_length_m <= 0:
        raise ValueError("estatura, masa y longitud del pie deben ser positivas")

    def segment(length, mass_fraction, com_fraction, gyration_fraction):
        mass = body_mass_kg * mass_fraction
        inertia = mass * (length * gyration_fraction) ** 2
        return Segment(length, mass, length * com_fraction, inertia)

    return LowerLimb(
        foot=segment(foot_length_m, 0.0145, 0.50, 0.475),
        shank=segment(0.246 * height_m, 0.0465, 0.433, 0.302),
        thigh=segment(0.245 * height_m, 0.10, 0.433, 0.323),
    )


def upper_limb_parameters(body_mass_kg: float, upper_arm_length_m: float,
                          forearm_length_m: float, anthropometry: dict) -> UpperLimb:
    """Genera parámetros segmentarios iniciales; antropometría individual no validada."""
    if body_mass_kg <= 0 or upper_arm_length_m <= 0 or forearm_length_m <= 0:
        raise ValueError("masa corporal y longitudes de brazo deben ser positivas")

    return UpperLimb(
        upper_arm=_segment_from_anthropometry(
            upper_arm_length_m, body_mass_kg, "upper_arm", anthropometry),
        forearm=_segment_from_anthropometry(
            forearm_length_m, body_mass_kg, "forearm", anthropometry),
    )


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


def _write_status(out: Path, status: str, message: str, warnings: list[str]) -> dict:
    table_dir = out / "reports/tables/inverse_dynamics"
    table_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"status": status, "message": message}]).to_csv(
        table_dir / "status.csv", index=False)
    return {"status": status, "message": message, "warnings": warnings}


def run_upper_limb_inverse_dynamics(out_dir=".", config="config/pipeline.yaml") -> dict:
    """Corre dinámica inversa 2D de brazo con la hipótesis exploratoria de muñeca libre."""
    cfg, out = load_config(config), Path(out_dir)
    spec = cfg.get("upper_limb_inverse_dynamics") or {}
    status_path = out / "reports/tables/inverse_dynamics/upper_limb_status.csv"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status, message, warnings = "disabled", "M12 superior está desactivado.", []

    if spec.get("enabled"):
        force_mode = spec.get("force_mode")
        if force_mode != "assume_zero_wrist_force":
            raise ValueError(
                "M12 superior requiere force_mode='assume_zero_wrist_force'; "
                "no se dispone de fuerzas medidas en muñeca."
            )
        group = spec["kinematics_group"]
        kin_path = out / "data/processed/kinematics" / f"{group}_kinematics.csv"
        filtered_path = out / "data/processed/filtered" / f"{group}_filtered.csv"
        missing_paths = [str(path) for path in (kin_path, filtered_path) if not path.is_file()]
        if missing_paths:
            status, message = "not_run", f"M12 superior requiere archivos: {missing_paths}"
        else:
            kin, filtered = pd.read_csv(kin_path), pd.read_csv(filtered_path)
            markers = spec.get("markers", {})
            shoulder = markers.get("shoulder", "hombro")
            elbow = markers.get("elbow", "codo")
            wrist = markers.get("wrist", "muneca")
            lengths_upper = segment_lengths(filtered, shoulder, elbow)
            lengths_fore = segment_lengths(filtered, elbow, wrist)
            if not np.isfinite([lengths_upper["L"], lengths_fore["L"]]).all():
                raise ValueError("M12 superior: no se pudieron estimar longitudes de segmento finitas")
            kin_time = kin["time_s"].to_numpy(float)
            filtered_time = filtered["time_s"].to_numpy(float)
            if (len(kin_time) != len(filtered_time)
                    or not np.allclose(kin_time, filtered_time, rtol=0, atol=1e-9)):
                raise ValueError("M12 superior: cinemática y coordenadas filtradas no comparten la malla temporal")

            anthropometry = spec["anthropometry"]
            params = upper_limb_parameters(
                float(anthropometry["body_mass_kg"]), lengths_upper["L"], lengths_fore["L"],
                anthropometry,
            )
            angles = {
                "upper_arm": kin["theta_seg_brazo"].to_numpy(float),
                "forearm": kin["theta_seg_antebrazo"].to_numpy(float),
            }
            omega = {
                "upper_arm": kin["omega_seg_brazo"].to_numpy(float),
                "forearm": kin["omega_seg_antebrazo"].to_numpy(float),
            }
            alpha = {
                "upper_arm": kin["alpha_seg_brazo"].to_numpy(float),
                "forearm": kin["alpha_seg_antebrazo"].to_numpy(float),
            }
            result = inverse_dynamics_upper_limb_2d(
                kin_time, angles, omega, alpha,
                np.column_stack([kin[f"ax_{shoulder}"], kin[f"ay_{shoulder}"]]),
                np.column_stack([kin[f"ax_{elbow}"], kin[f"ay_{elbow}"]]),
                params,
            )
            result_path = out / "data/processed/inverse_dynamics" / f"{group}_inverse_dynamics.csv"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(result_path, index=False)
            status = "completed_assumed_zero_wrist_force"
            message = (
                f"Procesadas {len(result)} muestras; fuerza externa en muñeca=0. "
                "Se excluyen mano, raqueta y pelota; resultado exploratorio."
            )
            warnings.append(
                "M12 superior supone fuerza externa nula en muñeca y excluye mano/raqueta/"
                "pelota; no representa las cargas reales del saque."
            )
            if max(lengths_upper["cv"], lengths_fore["cv"]) > float(spec.get("cv_warn", 0.05)):
                warnings.append(
                    "M12 superior: CV alto en longitudes hombro-codo/codo-muñeca; "
                    "revisar tracking o movimiento fuera del plano."
                )

    pd.DataFrame([{"status": status, "message": message}]).to_csv(status_path, index=False)
    if status != "completed_assumed_zero_wrist_force" and status != "disabled":
        warnings.append(message)
    return {"status": status, "message": message, "warnings": warnings}


def run_left_lower_limb_inverse_dynamics(out_dir=".", config="config/pipeline.yaml") -> dict:
    """Calcula cadera/rodilla de PI1 con el tobillo libre; el pie y la GRF se excluyen."""
    cfg, out = load_config(config), Path(out_dir)
    spec = cfg.get("left_lower_limb_inverse_dynamics") or {}
    status_path = out / "reports/tables/inverse_dynamics/left_limb_status.csv"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status, message, warnings = "disabled", "M12 de pierna izquierda está desactivado.", []

    if spec.get("enabled"):
        if spec.get("force_mode") != "assume_zero_ankle_force":
            raise ValueError(
                "M12 izquierda requiere force_mode='assume_zero_ankle_force'; "
                "no se dispone de fuerza externa medida en tobillo."
            )
        group = spec["kinematics_group"]
        kin_path = out / "data/processed/kinematics" / f"{group}_kinematics.csv"
        filtered_path = out / "data/processed/filtered" / f"{group}_filtered.csv"
        missing_paths = [str(path) for path in (kin_path, filtered_path) if not path.is_file()]
        if missing_paths:
            status, message = "not_run", f"M12 izquierda requiere archivos: {missing_paths}"
        else:
            kin, filtered = pd.read_csv(kin_path), pd.read_csv(filtered_path)
            markers = spec.get("markers", {})
            hip = markers.get("hip", "cadera")
            knee = markers.get("knee", "rodilla")
            ankle = markers.get("ankle", "tobillo")
            thigh_lengths = segment_lengths(filtered, hip, knee)
            shank_lengths = segment_lengths(filtered, knee, ankle)
            if not np.isfinite([thigh_lengths["L"], shank_lengths["L"]]).all():
                raise ValueError("M12 izquierda: longitudes de segmentos no finitas")
            kin_time = kin["time_s"].to_numpy(float)
            filtered_time = filtered["time_s"].to_numpy(float)
            if (len(kin_time) != len(filtered_time)
                    or not np.allclose(kin_time, filtered_time, rtol=0, atol=1e-9)):
                raise ValueError("M12 izquierda: cinemática y coordenadas no comparten la malla temporal")

            anthropometry = spec["anthropometry"]
            body_mass = float(anthropometry["body_mass_kg"])
            thigh = _segment_from_anthropometry(
                thigh_lengths["L"], body_mass, "thigh", anthropometry)
            shank = _segment_from_anthropometry(
                shank_lengths["L"], body_mass, "shank", anthropometry)
            angles = {
                "proximal": kin["theta_seg_muslo"].to_numpy(float),
                "distal": kin["theta_seg_pierna"].to_numpy(float),
            }
            omega = {
                "proximal": kin["omega_seg_muslo"].to_numpy(float),
                "distal": kin["omega_seg_pierna"].to_numpy(float),
            }
            alpha = {
                "proximal": kin["alpha_seg_muslo"].to_numpy(float),
                "distal": kin["alpha_seg_pierna"].to_numpy(float),
            }
            result = inverse_dynamics_two_segment_2d(
                kin_time, angles, omega, alpha,
                np.column_stack([kin[f"ax_{hip}"], kin[f"ay_{hip}"]]),
                np.column_stack([kin[f"ax_{knee}"], kin[f"ay_{knee}"]]),
                thigh, shank, float(spec.get("gravity_m_s2", 9.81)),
            ).rename(columns={
                "Fx_middle_N": "Fx_knee_N", "Fz_middle_N": "Fz_knee_N",
                "T_middle_Nm": "T_knee_Nm", "Fx_proximal_N": "Fx_hip_N",
                "Fz_proximal_N": "Fz_hip_N", "T_proximal_Nm": "T_hip_Nm",
                "distal_external_force_assumed_N": "ankle_external_force_assumed_N",
                "proximal_segment_length_m": "thigh_length_m",
                "distal_segment_length_m": "shank_length_m",
                "proximal_segment_mass_kg": "thigh_mass_kg",
                "distal_segment_mass_kg": "shank_mass_kg",
            })
            invalid_tracking = _rigid_segment_length_outliers(
                filtered,
                [(hip, knee), (knee, ankle)],
                float(spec.get("length_outlier_mad_threshold", 5.0)),
                int(spec.get("derivative_margin_samples", 2)),
            )
            result["tracking_valid"] = ~invalid_tracking
            result["tracking_qc_reason"] = np.where(
                invalid_tracking, "longitud_segmentaria_atipica_o_margen_derivada", "")
            force_moment_columns = [
                "Fx_knee_N", "Fz_knee_N", "T_knee_Nm",
                "Fx_hip_N", "Fz_hip_N", "T_hip_Nm",
            ]
            result.loc[invalid_tracking, force_moment_columns] = np.nan
            result["tracked_side"] = str(
                spec.get("tracked_side", "izquierda (PI1; vista lateral)"))
            result["external_force_mode"] = (
                "Fuerza externa en tobillo=0; pie/GRF excluidos")
            output_name = spec.get("output_name", "lateral_PI1_left")
            result_path = out / "data/processed/inverse_dynamics" / f"{output_name}_inverse_dynamics.csv"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(result_path, index=False)
            status = "completed_assumed_zero_ankle_force"
            message = (
                f"Procesadas {len(result)} muestras para PI1; fuerza externa en tobillo=0. "
                f"Se excluyen pie y GRF; {int(invalid_tracking.sum())} muestras se enmascararon "
                "por QC de longitud; resultado exploratorio."
            )
            warnings.append(
                "M12 izquierda supone fuerza externa nula en tobillo y excluye pie/GRF; "
                "no representa las cargas reales de apoyo."
            )
            cv_warn = float(spec.get("cv_warn", 0.05))
            if max(thigh_lengths["cv"], shank_lengths["cv"]) > cv_warn:
                warnings.append(
                    "M12 izquierda: CV alto en longitudes cadera-rodilla/rodilla-tobillo "
                    f"({thigh_lengths['cv']:.1%}, {shank_lengths['cv']:.1%}); "
                    "revisar tracking o movimiento fuera del plano."
                )
            if invalid_tracking.any():
                warnings.append(
                    f"M12 izquierda: se enmascararon {int(invalid_tracking.sum())} muestras "
                    "con longitudes segmentarias atípicas y su margen de derivación."
                )

    pd.DataFrame([{"status": status, "message": message}]).to_csv(status_path, index=False)
    if status != "completed_assumed_zero_ankle_force" and status != "disabled":
        warnings.append(message)
    return {"status": status, "message": message, "warnings": warnings}


def run_inverse_dynamics(out_dir=".", config="config/pipeline.yaml", raw_dir="data/raw") -> dict:
    """Corre M12 con GRF medida o bajo la hipótesis explícita de fase aérea."""
    cfg, out = load_config(config), Path(out_dir)
    spec = cfg.get("inverse_dynamics") or {}
    warnings = []
    status = "disabled"
    message = "M12 está desactivado; no se calcularon fuerzas ni torques."
    if spec.get("enabled"):
        force_mode = spec.get("force_mode", "measured_grf")
        if force_mode not in ("measured_grf", "assume_flight_no_grf"):
            raise ValueError(f"M12: force_mode no reconocido: {force_mode!r}")
        kin_group = spec["kinematics_group"]
        kin_path = out / "data/processed/kinematics" / f"{kin_group}_kinematics.csv"
        if not kin_path.is_file():
            status, message = "not_run", f"M12 requiere cinemática procesada: {kin_path}"
            warnings.append(message)
            return _write_status(out, status, message, warnings)
        kin = pd.read_csv(kin_path)
        anthropometry = spec["anthropometry"]
        params = winter_parameters(float(anthropometry["height_m"]),
                                   float(anthropometry["body_mass_kg"]),
                                   float(anthropometry["foot_length_m"]))
        segment_columns = spec.get("segment_columns") or {}
        segment_columns = {
            key: segment_columns.get(key, key)
            for key in ("foot", "shank", "thigh")
        }
        missing_kinematics = [
            f"{prefix}_{segment_columns[segment]}"
            for segment in segment_columns
            for prefix in ("theta_seg", "omega_seg", "alpha_seg")
            if f"{prefix}_{segment_columns[segment]}" not in kin
        ]
        if missing_kinematics:
            raise ValueError(f"M12: faltan columnas de cinemática en {kin_path}: {missing_kinematics}")
        required_linear = ("time_s", "ax_tobillo", "ay_tobillo")
        missing_linear = [column for column in required_linear if column not in kin]
        if missing_linear:
            raise ValueError(f"M12: faltan columnas de aceleración en {kin_path}: {missing_linear}")

        tk = kin["time_s"].to_numpy(float)
        common = np.ones(len(kin), dtype=bool)
        if force_mode == "measured_grf" and not spec.get("shared_clock_verified", False):
            status, message = "not_run", "Se requiere verificar la sincronización entre GRF y video."
        else:
            if force_mode == "measured_grf":
                grf_path = Path(spec.get("file", ""))
                if not grf_path.is_absolute():
                    grf_path = Path(raw_dir) / grf_path
            else:
                grf_path = None

            if force_mode == "measured_grf" and not grf_path.is_file():
                status, message = "not_run", f"No existe el archivo GRF configurado: {grf_path}"
            else:
                if force_mode == "measured_grf":
                    columns = spec["columns"]
                    grf = pd.read_csv(grf_path)
                    needed = [columns[k] for k in ("time", "force_x", "force_z", "cop_x", "cop_z")]
                    missing = [c for c in needed if c not in grf]
                    if missing:
                        raise ValueError(f"M12: faltan columnas GRF configuradas en {grf_path}: {missing}")
                    tg = grf[columns["time"]].to_numpy(float)
                    if not np.isfinite(tg).all() or np.any(np.diff(tg) <= 0):
                        raise ValueError("M12: el tiempo de GRF debe ser finito y estrictamente creciente")
                    common = (tk >= tg[0]) & (tk <= tg[-1])
                    if common.sum() < 3:
                        raise ValueError("M12: GRF y cinemática no tienen al menos tres muestras en común")
                    tk = tk[common]

                    def interp(name):
                        values = grf[columns[name]].to_numpy(float)
                        if not np.isfinite(values).all():
                            raise ValueError(f"M12: la columna GRF {columns[name]!r} contiene valores no finitos")
                        return np.interp(tk, tg, values)

                    grf_x, grf_z = interp("force_x"), interp("force_z")
                    cop = np.column_stack([interp("cop_x"), interp("cop_z")])
                else:
                    grf_x = np.zeros(len(tk))
                    grf_z = np.zeros(len(tk))
                    cop = np.zeros((len(tk), 2))

                angles = {
                    key: kin[f"theta_seg_{column}"].to_numpy(float)[common]
                    for key, column in segment_columns.items()
                }
                omega = {
                    key: kin[f"omega_seg_{column}"].to_numpy(float)[common]
                    for key, column in segment_columns.items()
                }
                alpha = {
                    key: kin[f"alpha_seg_{column}"].to_numpy(float)[common]
                    for key, column in segment_columns.items()
                }
                result = inverse_dynamics_2d(
                    tk, angles, omega, alpha, grf_x, grf_z, cop,
                    np.column_stack([kin["ax_tobillo"].to_numpy(float)[common],
                                     kin["ay_tobillo"].to_numpy(float)[common]]),
                    params, float(spec.get("contact_threshold_n", 20.0)))
                result["tracked_side"] = str(spec.get("tracked_side", "no especificado"))
                if force_mode == "assume_flight_no_grf":
                    result["external_force_mode"] = "GRF=0; fase aérea asumida"
                result_path = out / "data/processed/inverse_dynamics" / f"{kin_group}_inverse_dynamics.csv"
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result.to_csv(result_path, index=False)
                if force_mode == "assume_flight_no_grf":
                    status = "completed_assumed_flight"
                    message = (
                        f"Procesadas {len(result)} muestras suponiendo GRF=0 y fase aérea "
                        "en todo el registro; no válido para interpretar fase de apoyo."
                    )
                    warnings.append(
                        "M12 supone fase aérea (GRF=0) en todas las muestras; los resultados "
                        "no representan las cargas reales durante el apoyo."
                    )
                else:
                    status, message = "completed", f"Procesadas {len(result)} muestras sincronizadas."
    if status not in ("completed", "completed_assumed_flight"):
        warnings.append(message)
    return _write_status(out, status, message, warnings)
