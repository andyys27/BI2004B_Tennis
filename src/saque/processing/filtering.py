"""[M5] Suavizado y derivación sobre malla uniforme."""
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
