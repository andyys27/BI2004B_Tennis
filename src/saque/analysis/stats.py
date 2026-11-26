"""M10: descriptivos, curvas normalizadas y asociaciones exploratorias."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import load_config


def describe_series(time_s, values, *, variable: str, trial: str, phase: str,
                    start_s: float, end_s: float) -> dict:
    """Resume una variable en una fase usando las muestras dentro de sus límites."""
    t, y = np.asarray(time_s, float), np.asarray(values, float)
    if t.ndim != 1 or y.ndim != 1 or t.size != y.size:
        raise ValueError("time_s y values deben ser vectores unidimensionales de igual longitud")
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("time_s debe ser finito y estrictamente creciente")
    if end_s < start_s:
        raise ValueError("end_s debe ser mayor o igual que start_s")
    mask = (t >= start_s) & (t <= end_s) & np.isfinite(y)
    if not mask.any():
        raise ValueError(f"No hay muestras finitas de {variable} en la fase {phase}")
    tp, yp = t[mask], y[mask]
    peak_i = int(np.argmax(np.abs(yp)))
    return {
        "trial": trial,
        "variable": variable,
        "fase": phase,
        "t_ini_s": float(tp[0]),
        "t_fin_s": float(tp[-1]),
        "n_muestras": int(len(yp)),
        "media": float(np.mean(yp)),
        "desviacion_estandar": float(np.std(yp, ddof=1)) if len(yp) > 1 else np.nan,
        "rom": float(np.max(yp) - np.min(yp)),
        "pico_firmado": float(yp[peak_i]),
        "pico_absoluto": float(abs(yp[peak_i])),
        "tiempo_pico_s": float(tp[peak_i]),
    }


def normalize_trials(trials: dict[str, tuple], *, start_s: float | None = None,
                     end_s: float | None = None, n_points: int = 101) -> pd.DataFrame:
    """Normaliza múltiples curvas sobre su intervalo común, sin extrapolar."""
    if n_points < 2:
        raise ValueError("n_points debe ser al menos 2")
    if not trials:
        return pd.DataFrame(columns=["trial", "percent", "value"])
    arrays = {}
    for label, (time_s, values) in trials.items():
        t, y = np.asarray(time_s, float), np.asarray(values, float)
        if t.ndim != 1 or y.ndim != 1 or t.size != y.size:
            raise ValueError(f"Serie {label}: tiempo y valores deben tener igual longitud")
        if t.size < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError(f"Serie {label}: time_s debe ser finito y estrictamente creciente")
        arrays[label] = (t, y)
    common_start = max(t[0] for t, _ in arrays.values()) if start_s is None else float(start_s)
    common_end = min(t[-1] for t, _ in arrays.values()) if end_s is None else float(end_s)
    if common_end <= common_start:
        raise ValueError("Las series no comparten un intervalo temporal positivo")
    grid = np.linspace(common_start, common_end, n_points)
    percent = np.linspace(0.0, 100.0, n_points)
    rows = []
    for label, (t, y) in arrays.items():
        finite = np.isfinite(y)
        if finite.sum() < 2 or grid[0] < t[finite][0] or grid[-1] > t[finite][-1]:
            raise ValueError(f"Serie {label}: no cubre el intervalo de normalización")
        vals = np.interp(grid, t[finite], y[finite])
        rows.extend({"trial": label, "percent": float(p), "value": float(v)}
                    for p, v in zip(percent, vals))
    return pd.DataFrame(rows)


def intertrial_variability(curves: pd.DataFrame, *, epsilon: float = 1e-12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula banda media±DE y RMSD de cada intento respecto a la media."""
    required = {"trial", "percent", "value"}
    if not required.issubset(curves.columns):
        raise ValueError(f"curves debe contener {sorted(required)}")
    wide = curves.pivot(index="percent", columns="trial", values="value").sort_index()
    mean = wide.mean(axis=1)
    sd = wide.std(axis=1, ddof=1)
    cv = 100.0 * sd / mean.abs().where(mean.abs() > epsilon)
    count = wide.count(axis=1)
    band = pd.DataFrame({"percent": mean.index, "media": mean.values,
                         "desviacion_estandar": sd.values, "cv_pct": cv.values,
                         "n_intentos": count.values})
    rmsd = np.sqrt(((wide.sub(mean, axis=0)) ** 2).mean(axis=0))
    attempt = pd.DataFrame({"trial": rmsd.index, "rmsd_vs_media": rmsd.values})
    return band, attempt


def bootstrap_spearman(x, y, *, n_boot: int = 5000, alpha: float = 0.05,
                       seed: int = 0) -> dict:
    """ρ de Spearman e IC percentil bootstrap; con n<3 informa que no es estimable."""
    if n_boot < 1:
        raise ValueError("n_boot debe ser positivo")
    if not 0 < alpha < 1:
        raise ValueError("alpha debe estar entre 0 y 1")
    xx, yy = np.asarray(x, float), np.asarray(y, float)
    if xx.ndim != 1 or yy.ndim != 1 or xx.size != yy.size:
        raise ValueError("x e y deben ser vectores unidimensionales de igual longitud")
    valid = np.isfinite(xx) & np.isfinite(yy)
    xx, yy = xx[valid], yy[valid]
    n = len(xx)
    if n < 3 or np.ptp(xx) == 0 or np.ptp(yy) == 0:
        return {"n": n, "rho": np.nan, "ic_inferior": np.nan, "ic_superior": np.nan,
                "muestras_bootstrap": 0, "nota": "se requieren >=3 pares no constantes"}
    rho = float(spearmanr(xx, yy).statistic)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.ptp(xx[idx]) and np.ptp(yy[idx]):
            samples.append(float(spearmanr(xx[idx], yy[idx]).statistic))
    if not samples:
        low = high = np.nan
    else:
        low, high = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"n": n, "rho": rho, "ic_inferior": float(low), "ic_superior": float(high),
            "muestras_bootstrap": len(samples),
            "nota": "exploratorio; IC bootstrap inestable con muestras pequeñas"}


def run_statistics(out_dir=".", config="config/pipeline.yaml") -> dict:
    """Genera tablas M10 a partir de series configuradas y fases M7 disponibles."""
    cfg, out = load_config(config), Path(out_dir)
    specs = (cfg.get("statistics") or {}).get("series") or {}
    kin_dir = out / "data/processed/kinematics"
    phase_path = out / "reports/tables/events/fases.csv"
    if phase_path.exists():
        phases = pd.read_csv(phase_path)
    else:
        phases = pd.DataFrame()
    if not specs:
        return {"descriptives": pd.DataFrame(), "warnings": ["M10 sin series configuradas"]}

    desc_rows, curve_frames, warnings = [], [], []
    for label, spec in specs.items():
        source = kin_dir / f"{spec['group']}_kinematics.csv"
        if not source.exists():
            warnings.append(f"M10 omitió {label}: no existe {source}")
            continue
        data = pd.read_csv(source)
        column = spec["column"]
        if column not in data:
            raise ValueError(f"M10: falta la columna {column!r} en {source}")
        t, y = data["time_s"].to_numpy(float), data[column].to_numpy(float)
        label_phases = phases if not phases.empty else pd.DataFrame(
            [{"fase": "registro_completo", "t_ini": float(t[0]), "t_fin": float(t[-1])}])
        for phase in label_phases.to_dict(orient="records"):
            start_s = max(float(phase["t_ini"]), float(t[0]))
            end_s = min(float(phase["t_fin"]), float(t[-1]))
            try:
                if end_s <= start_s:
                    raise ValueError("fase fuera del intervalo temporal de la serie")
                row = describe_series(t, y, variable=label, trial="intento_disponible",
                                      phase=phase["fase"], start_s=start_s, end_s=end_s)
            except ValueError as exc:
                warnings.append(f"M10 omitió {label}/{phase['fase']}: {exc}")
                continue
            row["grupo"] = spec["group"]
            row["n_intentos_disponibles"] = 1
            row["nota"] = "Una exportación por serie: no estima variabilidad entre intentos."
            desc_rows.append(row)
            finite = np.isfinite(y)
            grid = np.linspace(start_s, end_s, 101)
            if grid[0] < t[finite][0] or grid[-1] > t[finite][-1]:
                warnings.append(f"M10 omitió curva normalizada {label}/{phase['fase']}: "
                                "faltan muestras válidas en los extremos.")
                continue
            curve = pd.DataFrame({"trial": "intento_disponible",
                                  "percent": np.linspace(0.0, 100.0, 101),
                                  "value": np.interp(grid, t[finite], y[finite])})
            curve.insert(0, "variable", label)
            curve.insert(1, "fase", phase["fase"])
            curve_frames.append(curve)
    descriptives = pd.DataFrame(desc_rows)
    curves = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame(
        columns=["variable", "fase", "trial", "percent", "value"])
    variability = []
    for (variable, phase), group in curves.groupby(["variable", "fase"]):
        band, attempts = intertrial_variability(group[["trial", "percent", "value"]])
        variability.append({"variable": variable, "fase": phase, "n_intentos": int(group["trial"].nunique()),
                            "rmsd_max": float(attempts["rmsd_vs_media"].max()) if not attempts.empty else np.nan,
                            "nota": "variabilidad inter-intento no estimable con un solo intento"})
    variability_df = pd.DataFrame(variability)
    table_dir = out / "reports/tables/statistics"
    table_dir.mkdir(parents=True, exist_ok=True)
    descriptives.to_csv(table_dir / "descriptivos_por_fase.csv", index=False)
    curves.to_csv(table_dir / "curvas_normalizadas.csv", index=False)
    variability_df.to_csv(table_dir / "variabilidad_interintento.csv", index=False)
    if descriptives.empty or descriptives["trial"].nunique() < 2:
        warnings.append("M10: hay menos de dos intentos identificados; no se interpreta variabilidad inter-intento ni asociaciones.")
    if not (cfg.get("events") or {}).get("shared_clock_verified", False):
        warnings.append("M10: el reloj común entre series no está verificado; métricas por fase entre grupos son exploratorias.")
    return {"descriptives": descriptives, "curves": curves,
            "variability": variability_df, "warnings": warnings}
