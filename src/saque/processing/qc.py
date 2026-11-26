"""Control de calidad del tracking: saltos aislados y huecos. Todo cambio queda en un log."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from .filtering import interpolate_short_gaps


class QCLog:
    """Acumula eventos de QC: archivo, indice, tipo, accion."""

    def __init__(self):
        self.rows: list[dict] = []

    def add(self, archivo, tipo, accion, indice=None, detalle=""):
        self.rows.append(dict(archivo=archivo, indice=indice, tipo=tipo, accion=accion, detalle=detalle))

    def to_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows, columns=["archivo", "indice", "tipo", "accion", "detalle"])
        df["indice"] = df["indice"].astype("Int64")
        return df


def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + 1e-12
    return 0.6745 * (x - med) / mad


def markers_in(df: pd.DataFrame) -> list[str]:
    return sorted({c[2:] for c in df.columns if c.startswith("x_") and f"y_{c[2:]}" in df.columns})


def noise_sigma(x: np.ndarray) -> float:
    """Desviación del ruido blanco estimada con segundas diferencias (robusta a movimiento suave).

    Var(Δ²x) = 6σ² para ruido blanco; se usa MAD para no contaminarse con saltos aislados.
    """
    d2 = np.diff(np.asarray(x, float), 2)
    return float(1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6.0))


def flag_outliers(df: pd.DataFrame, z_thr: float = 8.0, min_rel: float = 0.02) -> dict[str, np.ndarray]:
    """Muestras que se alejan de la mediana móvil (ventana 5): saltos de etiqueta aislados.

    Criterio doble: residual > z_thr * sigma_ruido  Y  residual > min_rel * rango de la coordenada.
    (sigma_ruido sale de segundas diferencias; el MAD del propio residual se degenera a 0.)
    """
    flags = {}
    for m in markers_in(df):
        x, y = df[f"x_{m}"].to_numpy(float), df[f"y_{m}"].to_numpy(float)
        if (~(np.isnan(x) | np.isnan(y))).sum() < 7:
            continue
        xi = pd.Series(x).interpolate(limit_direction="both").to_numpy()
        yi = pd.Series(y).interpolate(limit_direction="both").to_numpy()
        res = np.hypot(xi - median_filter(xi, 5, mode="nearest"), yi - median_filter(yi, 5, mode="nearest"))
        rng = max(np.ptp(xi), np.ptp(yi), 1e-12)
        sigma = max(noise_sigma(xi), noise_sigma(yi), 1e-12)
        bad = (res > z_thr * sigma) & (res > min_rel * rng)
        if bad.any():
            flags[m] = np.where(bad)[0]
    return flags


def mask_outliers(df: pd.DataFrame, name: str, log: QCLog, z_thr=8.0, min_rel=0.02) -> pd.DataFrame:
    df = df.copy()
    for m, idx in flag_outliers(df, z_thr, min_rel).items():
        for i in idx:
            log.add(name, "salto_tracking", f"marcador_{m}_a_NaN", int(i))
        df.loc[idx, [f"x_{m}", f"y_{m}"]] = np.nan
    return df


def fill_short_gaps(df: pd.DataFrame, name: str, log: QCLog, max_gap_s: float = 0.05) -> pd.DataFrame:
    """Interpola linealmente solo huecos NaN cortos; los largos quedan NaN y se registran."""
    df = df.copy()
    dt = float(np.median(np.diff(df["time_s"])))
    limit = max(1, math.ceil(max_gap_s / dt))
    cols = [c for c in df.columns if c.startswith(("x_", "y_"))]
    for c in cols:
        n_nan = int(df[c].isna().sum())
        if n_nan == 0:
            continue
        df[c] = interpolate_short_gaps(df[c].to_numpy(float), limit)
        left = int(df[c].isna().sum())
        log.add(name, "huecos", f"{c}: {n_nan - left} interpoladas, {left} sin resolver")
    return df
