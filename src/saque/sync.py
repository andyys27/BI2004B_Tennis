"""[M4] Sincronización por tiempo (nunca por índice de fila) y remuestreo a malla uniforme.

Convención de offsets:  t_global = t_raw + offset_s
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import correlate, correlation_lags

_META = ("time_s", "time_s_raw", "time_ms_raw")


def apply_offset(df: pd.DataFrame, offset_s: float) -> pd.DataFrame:
    df = df.copy()
    df["time_s"] = df["time_s"] + offset_s
    return df


def estimate_offset(t_a, a, t_b, b, fs: float = 120.0, max_lag_s: float | None = None) -> float:
    """Offset (s) a SUMAR a t_b para alinear la señal b con la señal a (xcorr).

    Evidencia numérica, no verdad: confirmar con un evento visible en el video.
    """
    t_a, a, t_b, b = map(lambda v: np.asarray(v, float), (t_a, a, t_b, b))
    lo, hi = min(t_a[0], t_b[0]), max(t_a[-1], t_b[-1])
    grid = np.arange(lo, hi, 1 / fs)

    def on_grid(t, y):
        y = y - np.nanmean(y)
        out = np.zeros_like(grid)
        m = (grid >= t[0]) & (grid <= t[-1])
        out[m] = np.interp(grid[m], t, y)
        return out

    ia, ib = on_grid(t_a, a), on_grid(t_b, b)
    cc = correlate(ia, ib, mode="full")
    lags = correlation_lags(len(ia), len(ib))
    if max_lag_s is not None:
        keep = np.abs(lags) <= max_lag_s * fs
        cc, lags = np.where(keep, cc, -np.inf), lags
    return float(lags[np.argmax(cc)] / fs)


def common_interval(dfs: list[pd.DataFrame]) -> tuple[float, float]:
    t0 = max(d["time_s"].iloc[0] for d in dfs)
    t1 = min(d["time_s"].iloc[-1] for d in dfs)
    if t1 <= t0:
        raise ValueError(f"Sin intervalo temporal común (t0={t0:.3f}, t1={t1:.3f}). Revisar offsets.")
    return float(t0), float(t1)


def to_uniform_grid(df: pd.DataFrame, fs: float = 120.0, t0=None, t1=None,
                    cols: list[str] | None = None) -> pd.DataFrame:
    """Spline cúbico sobre malla uniforme dentro de [t0, t1] (sin extrapolar)."""
    t = df["time_s"].to_numpy(float)
    t0 = t[0] if t0 is None else max(t0, t[0])
    t1 = t[-1] if t1 is None else min(t1, t[-1])
    n = int(np.floor((t1 - t0) * fs + 1e-9)) + 1
    grid = t0 + np.arange(n) / fs
    cols = cols or [c for c in df.columns if c not in _META and pd.api.types.is_numeric_dtype(df[c])]
    out = pd.DataFrame({"time_s": grid})
    for c in cols:
        ok = ~df[c].isna().to_numpy()
        if ok.sum() < 4:
            out[c] = np.nan
            continue
        out[c] = CubicSpline(t[ok], df.loc[ok, c].to_numpy(float))(grid)
    out.attrs.update(df.attrs)
    return out


def merge_on_grid(named: dict[str, pd.DataFrame], fs: float = 120.0) -> tuple[pd.DataFrame, dict]:
    """Une varias series de un grupo sobre su intervalo común. Devuelve (df, info)."""
    t0, t1 = common_interval(list(named.values()))
    merged, seen = None, set()
    for name, d in named.items():
        g = to_uniform_grid(d, fs, t0, t1)
        ren = {c: (c if c not in seen else f"{c}__{name}") for c in g.columns if c != "time_s"}
        seen.update(ren.values())
        g = g.rename(columns=ren)
        merged = g if merged is None else merged.merge(g, on="time_s", how="inner")
    return merged, dict(t0=t0, t1=t1, n=len(merged), fs=fs)
