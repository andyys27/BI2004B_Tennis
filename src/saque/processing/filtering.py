"""Suavizado y derivación sobre malla uniforme."""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter


def butter_lp(x, fs: float, fc: float, order: int = 2) -> np.ndarray:
    """Butterworth pasa-bajas de fase cero (filtfilt => orden efectivo = 2*order)."""
    x = np.asarray(x, float)
    if np.isnan(x).any():
        raise ValueError("La señal contiene NaN; resolver en QC antes de filtrar.")
    if not 0 < fc < fs / 2:
        raise ValueError(f"fc={fc} fuera de (0, Nyquist={fs/2}).")
    b, a = butter(order, fc / (fs / 2), btype="low")
    padlen = min(3 * max(len(a), len(b)), len(x) - 1)
    return filtfilt(b, a, x, padlen=padlen)


def savgol(x, fs: float, window: int = 15, poly: int = 3, deriv: int = 0) -> np.ndarray:
    window += (window % 2 == 0)
    return savgol_filter(np.asarray(x, float), window, poly, deriv=deriv, delta=1 / fs)


def residual_analysis(x, fs: float, fcs) -> np.ndarray:
    """R(fc) = RMS(x - x_filtrada(fc)) (Winter)."""
    x = np.asarray(x, float)
    return np.array([np.sqrt(np.mean((x - butter_lp(x, fs, f)) ** 2)) for f in fcs])


def winter_cutoff(x, fs: float, fc_range=(4, 20), n_fc: int = 33, tail_frac: float = 0.35,
                  fallback: float = 10.0) -> dict:
    """Corte óptimo por análisis de residuos.

    Ajusta una recta a la cola (fc altas, dominada por ruido), extrapola a fc=0 (R0) y toma la fc
    donde la curva R(fc) cruza R0. Si el método no es aplicable, devuelve `fallback` y lo indica.
    """
    hi = min(fc_range[1], 0.9 * fs / 2)
    fcs = np.linspace(fc_range[0], hi, n_fc)
    R = residual_analysis(x, fs, fcs)
    k = max(3, int(n_fc * tail_frac))
    slope, r0 = np.polyfit(fcs[-k:], R[-k:], 1)
    out = dict(fcs=fcs, R=R, R0=float(r0), slope=float(slope), fc=fallback, metodo="fallback")
    if r0 > 0 and slope < 0:      # R decrece con fc; la cola (ruido) tiene pendiente negativa
        below = np.where(R <= r0)[0]
        if below.size and below[0] > 0:
            j = below[0]
            f0, f1, r_a, r_b = fcs[j - 1], fcs[j], R[j - 1], R[j]
            out["fc"] = float(f0 + (r_a - r0) / (r_a - r_b) * (f1 - f0)) if r_a != r_b else float(f1)
            out["metodo"] = "winter"
        elif below.size and below[0] == 0:
            out["fc"], out["metodo"] = float(fcs[0]), "winter_limite_inferior"
    return out


def check_uniform(t, rtol: float = 1e-6) -> float:
    dt = np.diff(np.asarray(t, float))
    if not np.allclose(dt, dt[0], rtol=rtol, atol=1e-9):
        raise ValueError("Malla temporal no uniforme: remuestrear antes de derivar (M4).")
    return float(dt[0])


def derivative(x, fs: float, order: int = 1) -> np.ndarray:
    """Derivada por diferencias centrales (np.gradient) sobre malla uniforme."""
    d = np.asarray(x, float)
    for _ in range(order):
        d = np.gradient(d, 1 / fs, edge_order=2)
    return d


def finite_runs(x) -> list[tuple[int, int]]:
    """Intervalos [ini, fin) de muestras finitas consecutivas."""
    ok = np.isfinite(np.asarray(x, float)).astype(int)
    edges = np.diff(np.r_[0, ok, 0])
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))


def butter_lp_gaps(x, fs: float, fc: float, order: int = 2, min_len: int = 12) -> np.ndarray:
    """Butterworth por tramos finitos: los huecos largos se conservan como NaN.

    Tramos más cortos que `min_len` muestras quedan NaN (no se pueden filtrar de forma fiable).
    """
    x = np.asarray(x, float)
    out = np.full_like(x, np.nan)
    for a, b in finite_runs(x):
        if b - a >= min_len:
            out[a:b] = butter_lp(x[a:b], fs, fc, order)
    return out


def longest_run(x) -> np.ndarray:
    """Segmento finito más largo (para elegir la frecuencia de corte de Winter)."""
    runs = finite_runs(x)
    if not runs:
        return np.array([])
    a, b = max(runs, key=lambda r: r[1] - r[0])
    return np.asarray(x, float)[a:b]


def short_gap_mask(x, max_gap: int) -> np.ndarray:
    """True en las muestras NaN que pertenecen a huecos internos de <= max_gap muestras."""
    nan = ~np.isfinite(np.asarray(x, float))
    edges = np.diff(np.r_[0, nan.astype(int), 0])
    out = np.zeros(nan.size, bool)
    for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
        if a > 0 and b < nan.size and b - a <= max_gap:
            out[a:b] = True
    return out


def interpolate_short_gaps(x, max_gap: int) -> np.ndarray:
    """Interpola linealmente SOLO los huecos internos completos de <= max_gap muestras.

    (pandas `interpolate(limit=n)` rellenaría también las primeras n muestras de huecos largos.)
    """
    x = np.asarray(x, float)
    fill = short_gap_mask(x, max_gap)
    if not fill.any():
        return x.copy()
    ok = np.isfinite(x)
    out = x.copy()
    idx = np.arange(x.size)
    out[fill] = np.interp(idx[fill], idx[ok], x[ok])
    return out
