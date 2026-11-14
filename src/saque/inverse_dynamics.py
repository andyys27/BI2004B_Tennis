"""M12: dinámica inversa planar de pie, pierna y muslo con Newton-Euler."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config


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


def run_inverse_dynamics(out_dir=".", config="config/pipeline.yaml", raw_dir="data/raw") -> dict:
    """Corre M12 solo cuando GRF, antropometría y reloj común están configurados."""
    cfg, out = load_config(config), Path(out_dir)
    spec = cfg.get("inverse_dynamics") or {}
    warnings = []
    status = "disabled"
    message = "M12 está desactivado; no se calcularon fuerzas ni torques."
    if spec.get("enabled"):
        grf_path = Path(spec.get("file", ""))
        if not grf_path.is_absolute():
            grf_path = Path(raw_dir) / grf_path
        if not spec.get("shared_clock_verified", False):
            status, message = "not_run", "Se requiere verificar la sincronización entre GRF y video."
        elif not grf_path.is_file():
            status, message = "not_run", f"No existe el archivo GRF configurado: {grf_path}"
        else:
            columns = spec["columns"]
            kin_group = spec["kinematics_group"]
            kin_path = out / "data/processed/kinematics" / f"{kin_group}_kinematics.csv"
            if not kin_path.is_file():
                raise FileNotFoundError(f"M12 requiere cinemática procesada: {kin_path}")
            kin = pd.read_csv(kin_path)
            grf = pd.read_csv(grf_path)
            needed = [columns[k] for k in ("time", "force_x", "force_z", "cop_x", "cop_z")]
            missing = [c for c in needed if c not in grf]
            if missing:
                raise ValueError(f"M12: faltan columnas GRF configuradas en {grf_path}: {missing}")
            tg = grf[columns["time"]].to_numpy(float)
            if not np.isfinite(tg).all() or np.any(np.diff(tg) <= 0):
                raise ValueError("M12: el tiempo de GRF debe ser finito y estrictamente creciente")
            tk = kin["time_s"].to_numpy(float)
            common = (tk >= tg[0]) & (tk <= tg[-1])
            if common.sum() < 3:
                raise ValueError("M12: GRF y cinemática no tienen al menos tres muestras en común")
            tk = tk[common]

            def interp(name):
                values = grf[columns[name]].to_numpy(float)
                if not np.isfinite(values).all():
                    raise ValueError(f"M12: la columna GRF {columns[name]!r} contiene valores no finitos")
                return np.interp(tk, tg, values)

            angles = {k: kin[f"theta_seg_{k}"].to_numpy(float)[common]
                      for k in ("foot", "shank", "thigh")}
            omega = {k: kin[f"omega_seg_{k}"].to_numpy(float)[common]
                     for k in ("foot", "shank", "thigh")}
            alpha = {k: kin[f"alpha_seg_{k}"].to_numpy(float)[common]
                     for k in ("foot", "shank", "thigh")}
            anthropometry = spec["anthropometry"]
            params = winter_parameters(float(anthropometry["height_m"]),
                                       float(anthropometry["body_mass_kg"]),
                                       float(anthropometry["foot_length_m"]))
            result = inverse_dynamics_2d(
                tk, angles, omega, alpha, interp("force_x"), interp("force_z"),
                np.column_stack([interp("cop_x"), interp("cop_z")]),
                np.column_stack([kin["ax_tobillo"].to_numpy(float)[common],
                                 kin["ay_tobillo"].to_numpy(float)[common]]),
                params, float(spec.get("contact_threshold_n", 20.0)))
            result_path = out / "data/processed/inverse_dynamics" / f"{kin_group}_inverse_dynamics.csv"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(result_path, index=False)
            status, message = "completed", f"Procesadas {len(result)} muestras sincronizadas."
    if status != "completed":
        warnings.append(message)
    table_dir = out / "reports/tables/inverse_dynamics"
    table_dir.mkdir(parents=True, exist_ok=True)
    status_df = pd.DataFrame([{"status": status, "message": message}])
    status_df.to_csv(table_dir / "status.csv", index=False)
    return {"status": status, "message": message, "warnings": warnings}
