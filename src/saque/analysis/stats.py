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


def joint_peaks(kin: pd.DataFrame, joints: list[str], t0: float, t1: float, t_ref: float,
                kinds=("omega", "alpha")) -> pd.DataFrame:
    """Pico absoluto de ω y α de cada articulación (`<kind>_interior_<j>`) en [t0, t1].

    Ignora huecos; `frac_valida` indica cuánta ventana tenía datos. t_rel en ms respecto a t_ref.
    """
    m = (kin["time_s"] >= t0) & (kin["time_s"] <= t1)
    t = kin.loc[m, "time_s"].to_numpy(float)
    rows = []
    for j in joints:
        for kind in kinds:
            col = f"{kind}_interior_{j}"
            if col not in kin:
                continue
            y = np.abs(kin.loc[m, col].to_numpy(float))
            ok = np.isfinite(y)
            if not ok.any():
                continue
            i = int(np.nanargmax(y))
            rows.append(dict(articulacion=j, magnitud=kind, pico_abs=float(y[i]),
                             unidad="rad/s" if kind == "omega" else "rad/s²",
                             t_rel_impacto_ms=1000 * (t[i] - t_ref), frac_valida=float(ok.mean())))
    return pd.DataFrame(rows)


def spearman_matrix(wide: pd.DataFrame, min_n: int = 4) -> pd.DataFrame:
    """ρ de Spearman y p para cada par de columnas de `wide` (filas = saques), formato largo.

    Descriptivo: con n = 5 un |ρ| = 0.9 tiene p ≈ 0.04 y uno de 0.7 no es distinguible de 0.
    """
    cols = [c for c in wide if wide[c].notna().sum() >= min_n and wide[c].nunique() > 1]
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            d = wide[[a, b]].dropna()
            if len(d) < min_n or d[a].nunique() < 2 or d[b].nunique() < 2:
                continue
            r = spearmanr(d[a], d[b])
            rows.append(dict(var_a=a, var_b=b, n=len(d), rho=float(r.statistic), p=float(r.pvalue)))
    return pd.DataFrame(rows)


def qc_summary(qc_log: pd.DataFrame) -> pd.DataFrame:
    """Muestras anuladas por vista y tipo de control, a partir de `qc_log`.

    Dos formatos de registro: resumen ("tronco: 148 fotogramas a NaN") o una fila por muestra
    (columna `indice`, p. ej. `salto_tracking`). Los registros de `huecos` describen la
    interpolación de huecos ya existentes, no anulaciones, y no se cuentan.
    """
    if qc_log.empty or "accion" not in qc_log:
        return pd.DataFrame(columns=["vista", "tipo", "muestras"])
    d = qc_log[qc_log["tipo"] != "huecos"]
    n = pd.to_numeric(d["accion"].astype(str).str.extract(r"(\d+)\s+(?:fotogramas|muestras)")[0],
                      errors="coerce")
    if "indice" in d:
        n = n.where(n.notna() | d["indice"].isna(), 1.0)          # una fila = una muestra
    d = d.assign(muestras=n).dropna(subset=["muestras"])
    return (d.groupby(["archivo", "tipo"])["muestras"].sum().reset_index()
            .rename(columns={"archivo": "vista"}))
