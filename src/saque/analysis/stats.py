"""Descriptivos por fase, curvas normalizadas, variabilidad entre saques y asociaciones."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr



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
