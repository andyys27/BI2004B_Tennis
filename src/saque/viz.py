"""M13: generación de figuras diagnósticas y cinemáticas."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_time_coverage(catalog: pd.DataFrame, path: Path):
    required = {"archivo", "t_ini", "t_fin"}
    if not required.issubset(catalog.columns) or catalog.empty:
        return False
    data = catalog.dropna(subset=["t_ini", "t_fin"]).reset_index(drop=True)
    if data.empty:
        return False
    fig, ax = plt.subplots(figsize=(10, max(3, 0.32 * len(data))))
    for i, row in data.iterrows():
        ax.plot([row.t_ini, row.t_fin], [i, i], lw=5, solid_capstyle="butt")
    ax.set_yticks(range(len(data)), data["archivo"])
    ax.set_xlabel("Tiempo (s)")
    ax.set_title("Cobertura temporal de los archivos")
    ax.grid(axis="x", alpha=0.25)
    _save(fig, path)
    return True


def plot_sampling_intervals(files: list[Path], path: Path):
    deltas = []
    for file in files:
        data = pd.read_csv(file)
        if "time_s" not in data:
            continue
        t = data["time_s"].to_numpy(float)
        dt = np.diff(t)
        deltas.extend(dt[np.isfinite(dt) & (dt > 0)])
    if not deltas:
        return False
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.asarray(deltas) * 1000, bins="auto", color="#4472C4", alpha=0.85)
    ax.set_xlabel("Intervalo entre muestras (ms)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución del intervalo de muestreo")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, path)
    return True


def plot_kinematics(data: pd.DataFrame, path: Path, title: str):
    columns = {
        key: [c for c in data.columns if c.startswith(prefix)]
        for key, prefix in (("Posición angular", "theta_"),
                            ("Velocidad angular", "omega_"),
                            ("Aceleración angular", "alpha_"))
    }
    if not any(columns.values()) or "time_s" not in data:
        return False
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 8))
    units = ("rad", "rad/s", "rad/s²")
    for ax, (label, cols), unit in zip(axes, columns.items(), units):
        for col in cols:
            ax.plot(data["time_s"], data[col], label=col.removeprefix(
                "theta_").removeprefix("omega_").removeprefix("alpha_"))
        ax.set_ylabel(unit)
        ax.grid(alpha=0.2)
        if cols:
            ax.legend(loc="best", fontsize="small")
        else:
            ax.text(0.5, 0.5, f"Sin series de {label.lower()}", ha="center", va="center",
                    transform=ax.transAxes)
    axes[-1].set_xlabel("Tiempo (s)")
    fig.suptitle(title)
    _save(fig, path)
    return True


def plot_phases(phases: pd.DataFrame, events: pd.DataFrame, path: Path):
    if phases.empty:
        return False
    fig, ax = plt.subplots(figsize=(9, 2.8))
    for i, row in phases.iterrows():
        ax.barh(i, row["t_fin"] - row["t_ini"], left=row["t_ini"], height=0.55,
                label=row["fase"])
        ax.text((row["t_ini"] + row["t_fin"]) / 2, i, row["fase"],
                ha="center", va="center", fontsize=8)
    if not events.empty:
        for _, row in events.iterrows():
            ax.axvline(row["t"], color="black", linestyle="--", linewidth=0.7)
    ax.set_yticks([])
    ax.set_xlabel("Tiempo (s)")
    ax.set_title("Fases y eventos detectados (reloj de referencia M7)")
    ax.grid(axis="x", alpha=0.2)
    _save(fig, path)
    return True


def plot_jacobian(data: pd.DataFrame, path: Path):
    needed = {"time_s", "speed_J", "speed_directa"}
    if not needed.issubset(data.columns):
        return False
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(data["time_s"], data["speed_directa"], label="Directa")
    axes[0].plot(data["time_s"], data["speed_J"], label="Jacobiano", alpha=0.8)
    axes[0].set_xlabel("Tiempo (s)")
    axes[0].set_ylabel("Rapidez (m/s)")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].scatter(data["speed_directa"], data["speed_J"], s=8, alpha=0.6)
    lim = float(np.nanmax(data[["speed_directa", "speed_J"]].to_numpy()))
    axes[1].plot([0, lim], [0, lim], "k--", lw=1)
    axes[1].set_xlabel("Rapidez directa (m/s)")
    axes[1].set_ylabel("Rapidez Jacobiano (m/s)")
    axes[1].grid(alpha=0.2)
    fig.suptitle("Validación de velocidad de muñeca")
    _save(fig, path)
    return True


def plot_contributions(data: pd.DataFrame, path: Path):
    needed = {"time_s", "frac_hombro", "frac_codo"}
    if not needed.issubset(data.columns):
        return False
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(data["time_s"], data["frac_hombro"], label="Hombro")
    ax.plot(data["time_s"], data["frac_codo"], label="Codo")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Fracción proyectada")
    ax.set_title("Contribuciones articulares a la rapidez de muñeca")
    ax.legend()
    ax.grid(alpha=0.2)
    _save(fig, path)
    return True


def plot_wrist_trajectory(filtered: pd.DataFrame, kin: pd.DataFrame, path: Path,
                          flip_y: bool = True):
    if not {"x_muneca", "y_muneca"}.issubset(filtered.columns):
        return False
    x = filtered["x_muneca"].to_numpy(float)
    y = filtered["y_muneca"].to_numpy(float) * (-1 if flip_y else 1)
    speed = kin["speed_muneca"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(6, 6))
    points = ax.scatter(x, y, c=speed, s=12, cmap="viridis")
    ax.plot(x, y, color="gray", alpha=0.4, linewidth=0.7)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Trayectoria de muñeca coloreada por rapidez")
    fig.colorbar(points, ax=ax, label="Rapidez (m/s)")
    _save(fig, path)
    return True


def _inverse_dynamics_note(data: pd.DataFrame) -> str:
    notes = []
    for column in ("tracked_side", "external_force_mode"):
        if column in data:
            values = data[column].dropna().unique()
            if len(values) == 1:
                notes.append(str(values[0]))
            elif len(values) > 1:
                notes.append(f"{column} variable")
    if "tracking_valid" in data:
        valid = data["tracking_valid"].astype(bool)
        invalid_count = int((~valid).sum())
        if invalid_count:
            notes.append(f"{invalid_count} muestras enmascaradas por QC")
    return "; ".join(notes)


def _plot_inverse_dynamics(data: pd.DataFrame, path: Path, quantity: str) -> bool:
    if "T_ankle_Nm" in data:
        joints = {"ankle": "Tobillo", "knee": "Rodilla", "hip": "Cadera"}
    elif "T_shoulder_Nm" in data:
        joints = {"shoulder": "Hombro", "elbow": "Codo"}
    elif "T_hip_Nm" in data:
        joints = {"hip": "Cadera", "knee": "Rodilla"}
    else:
        return False
    if "time_s" not in data:
        return False
    columns = ([f"T_{joint}_Nm" for joint in joints]
               if quantity == "moments"
               else [f"{axis}_{joint}_N" for joint in joints for axis in ("Fx", "Fz")])
    if not set(columns).issubset(data.columns):
        return False
    time_s = data["time_s"].to_numpy(float)
    if not np.isfinite(time_s).all() or np.any(np.diff(time_s) <= 0):
        raise ValueError("M13: el tiempo de dinámica inversa debe ser finito y creciente")
    if np.isinf(data[columns].to_numpy(float)).any():
        raise ValueError("M13: las fuerzas/torques de dinámica inversa contienen infinitos")

    fig, axes = plt.subplots(len(joints), 1, sharex=True,
                             figsize=(9, 3.2 * len(joints)), squeeze=False)
    axes = axes[:, 0]
    for ax, (joint, label) in zip(axes, joints.items()):
        if quantity == "moments":
            ax.plot(time_s, data[f"T_{joint}_Nm"], color="#C00000", label="Momento neto")
            ax.axhline(0, color="black", linewidth=0.7)
            ax.set_ylabel("N·m")
        else:
            ax.plot(time_s, data[f"Fx_{joint}_N"], label="Fx")
            ax.plot(time_s, data[f"Fz_{joint}_N"], label="Fz")
            ax.axhline(0, color="black", linewidth=0.7)
            ax.set_ylabel("N")
            ax.legend(loc="best")
        ax.set_title(label, loc="left", fontsize="medium")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Tiempo (s)")
    title = ("Momentos articulares netos" if quantity == "moments"
             else "Fuerzas de reacción articulares")
    note = _inverse_dynamics_note(data)
    if note:
        title += f"\nSupuesto: {note}"
    fig.suptitle(title)
    _save(fig, path)
    return True


def plot_inverse_dynamics_moments(data: pd.DataFrame, path: Path) -> bool:
    """Grafica momentos articulares netos del tobillo, rodilla y cadera."""
    return _plot_inverse_dynamics(data, path, "moments")


def plot_inverse_dynamics_forces(data: pd.DataFrame, path: Path) -> bool:
    """Grafica componentes de las fuerzas de reacción articulares."""
    return _plot_inverse_dynamics(data, path, "forces")


def run_visualizations(out_dir=".", config=None) -> dict:
    """Genera las figuras posibles con los datos actuales e informa las faltantes."""
    out = Path(out_dir)
    figures = out / "reports/figures"
    generated, warnings = [], []
    catalog_path = out / "reports/tables/catalog/catalogo.csv"
    if catalog_path.exists() and plot_time_coverage(pd.read_csv(catalog_path),
                                                   figures / "time_coverage.png"):
        generated.append(figures / "time_coverage.png")
    else:
        warnings.append("M13 omitió cobertura temporal: catálogo ausente o sin rangos temporales.")

    sync_files = sorted((out / "data/interim/synchronized").glob("*.csv"))
    if plot_sampling_intervals(sync_files, figures / "sampling_intervals.png"):
        generated.append(figures / "sampling_intervals.png")
    else:
        warnings.append("M13 omitió histograma de muestreo: no hay series sincronizadas.")

    kin_dir = out / "data/processed/kinematics"
    for file in sorted(kin_dir.glob("*_kinematics.csv")):
        dest = figures / "kinematics" / f"{file.stem.removesuffix('_kinematics')}.png"
        if plot_kinematics(pd.read_csv(file), dest, file.stem):
            generated.append(dest)

    phase_path = out / "reports/tables/events/fases.csv"
    event_path = out / "reports/tables/events/eventos.csv"
    if phase_path.exists():
        phases = pd.read_csv(phase_path)
        events = pd.read_csv(event_path) if event_path.exists() else pd.DataFrame()
        dest = figures / "events/phases.png"
        if plot_phases(phases, events, dest):
            generated.append(dest)

    jac_dir = out / "data/processed/jacobian"
    for file in sorted(jac_dir.glob("*_jacobian.csv")):
        data = pd.read_csv(file)
        for plotter, suffix in ((plot_jacobian, "validation"), (plot_contributions, "contributions")):
            dest = figures / "jacobian" / f"{file.stem}_{suffix}.png"
            if plotter(data, dest):
                generated.append(dest)

    arm_filtered = out / "data/processed/filtered/posterior_brazo_filtered.csv"
    arm_kin = kin_dir / "posterior_brazo_kinematics.csv"
    if arm_filtered.exists() and arm_kin.exists():
        flip_y = bool((config or {}).get("flip_y", True))
        dest = figures / "trajectory/wrist.png"
        if plot_wrist_trajectory(pd.read_csv(arm_filtered), pd.read_csv(arm_kin), dest, flip_y):
            generated.append(dest)

    inverse_dir = out / "data/processed/inverse_dynamics"
    for file in sorted(inverse_dir.glob("*_inverse_dynamics.csv")):
        data = pd.read_csv(file)
        plots = (
            (plot_inverse_dynamics_moments, "joint_moments"),
            (plot_inverse_dynamics_forces, "joint_reaction_forces"),
        )
        for plotter, suffix in plots:
            dest = figures / "inverse_dynamics" / f"{file.stem}_{suffix}.png"
            if plotter(data, dest):
                generated.append(dest)
            else:
                warnings.append(
                    f"M13 omitió la gráfica {suffix}: faltan columnas en {file}."
                )
    if not list(kin_dir.glob("*_kinematics.csv")):
        warnings.append("M13 no encontró salidas cinemáticas para graficar.")
    return {"figures": generated, "warnings": warnings}
