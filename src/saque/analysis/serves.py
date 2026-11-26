"""Segmentación del video continuo en saques individuales y sincronización entre vistas.

1. Candidatos: máximos de altura de la muñeca de golpeo (uno por saque).
2. Impacto: fin de la caída de la pelota cerca de cada candidato (video/ball.py); si la pelota no
   se puede seguir, se usa como respaldo el máximo de rapidez de la muñeca cerca del candidato.
3. Otras vistas: se sincronizan con la de referencia por correlación cruzada de la altura de la
   cadera (los saltos de los saques se ven en ambas) y los impactos se trasladan con ese desfase.
Ventana de cada saque: [impacto - pre, impacto + post].
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags, find_peaks


def detect_impacts(t, wrist_height, wrist_speed, n_expected: int | None = None,
                   min_separation_s: float = 3.0, refine_window_s: float = 0.2) -> pd.DataFrame:
    """Un impacto por saque. Devuelve tabla ordenada en el tiempo con t_altura y t_impacto."""
    t = np.asarray(t, float)
    h = pd.Series(np.asarray(wrist_height, float)).interpolate(limit_direction="both").to_numpy()
    v = np.nan_to_num(np.asarray(wrist_speed, float), nan=0.0)
    fs = 1.0 / np.median(np.diff(t))
    peaks, props = find_peaks(h, distance=max(1, int(min_separation_s * fs)), prominence=0)
    if peaks.size == 0:
        return pd.DataFrame(columns=["saque", "t_altura", "t_impacto", "prominencia"])
    order = np.argsort(props["prominences"])[::-1]
    if n_expected:
        order = order[:n_expected]
    keep = np.sort(peaks[order])
    prom = dict(zip(peaks, props["prominences"]))
    rows, w = [], int(refine_window_s * fs)
    for k, p in enumerate(keep, start=1):
        lo, hi = max(0, p - w), min(len(t), p + w + 1)
        i = lo + int(np.argmax(v[lo:hi]))
        rows.append(dict(saque=k, t_altura=float(t[p]), t_impacto=float(t[i]),
                         prominencia=float(prom[p])))
    return pd.DataFrame(rows)


def serve_windows(impacts: pd.DataFrame, pre_s: float, post_s: float,
                  t_min: float, t_max: float) -> pd.DataFrame:
    """Ventana [t_ini, t_fin] por saque, recortada a la duración del registro."""
    w = impacts[["saque", "t_impacto"]].copy()
    w["t_ini"] = np.maximum(w["t_impacto"] - pre_s, t_min)
    w["t_fin"] = np.minimum(w["t_impacto"] + post_s, t_max)
    return w


def manual_impacts(times_s: list[float]) -> pd.DataFrame:
    return pd.DataFrame(dict(saque=range(1, len(times_s) + 1), t_altura=np.nan,
                             t_impacto=[float(x) for x in times_s], prominencia=np.nan))


def estimate_lag(t_ref, y_ref, t_other, y_other, fs: float, max_lag_s: float = 5.0) -> dict:
    """Desfase (s) tal que t_other = t_ref + lag, por correlación cruzada normalizada.

    Las series se llevan a una malla común de `fs` Hz; los huecos se rellenan con la media.
    """
    def grid(t, y):
        t, y = np.asarray(t, float), np.asarray(y, float)
        ok = np.isfinite(y)
        g = np.arange(t[0], t[-1], 1 / fs)
        v = np.interp(g, t[ok], y[ok])
        return (v - v.mean()) / (v.std() or 1.0)

    a, b = grid(t_ref, y_ref), grid(t_other, y_other)
    cc = correlate(b, a, mode="full") / min(len(a), len(b))
    lags = correlation_lags(len(b), len(a)) / fs
    keep = np.abs(lags) <= max_lag_s
    k = int(np.argmax(np.where(keep, cc, -np.inf)))
    lag = float(lags[k] + (t_other[0] - t_ref[0]))
    return dict(lag_s=lag, correlacion=float(cc[k]))


def shift_impacts(impacts: pd.DataFrame, lag_s: float) -> pd.DataFrame:
    out = impacts.copy()
    for c in ("t_impacto", "t_altura"):
        out[c] = out[c] + lag_s
    out["metodo"] = "sincronizado_desde_referencia"
    return out.drop(columns=[c for c in ("x_contacto_px", "y_contacto_px", "puntos_pelota")
                             if c in out])


def view_offset(ref: pd.DataFrame, other: pd.DataFrame) -> dict:
    """Offset (s) a SUMAR al reloj de `other` para alinearlo con `ref`, emparejando saques.

    Usa la mediana de las diferencias de impacto; la dispersión indica la calidad del ajuste.
    """
    n = min(len(ref), len(other))
    if n == 0:
        return dict(offset_s=np.nan, dispersion_s=np.nan, n=0)
    d = ref["t_impacto"].to_numpy()[:n] - other["t_impacto"].to_numpy()[:n]
    med = float(np.median(d))
    return dict(offset_s=med, dispersion_s=float(np.max(np.abs(d - med))), n=n)
