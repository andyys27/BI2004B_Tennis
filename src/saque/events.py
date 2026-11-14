"""[M7] Eventos, fases y secuencia próximo-distal.

El impacto se toma como PROXY (máximo de rapidez de muñeca) porque no se digitaliza raqueta ni pelota.
Si marcaste el fotograma de contacto en el video, pásalo en `manual={"impacto": t_s}`.
Todas las series deben compartir un reloj verificado (M4); si no, los eventos son exploratorios.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

Series = tuple  # (t, y)


def _sustained_first(mask: np.ndarray, n: int) -> int | None:
    """Primer índice donde `mask` es True durante n muestras seguidas."""
    run = 0
    for i, m in enumerate(mask):
        run = run + 1 if m else 0
        if run >= n:
            return i - n + 1
    return None


def detect_events(series: dict[str, Series], manual: dict | None = None, start_frac: float = 0.05,
                  end_frac: float = 0.20, sustain: int = 3, omega_margin_s: float = 0.05,
                  flex_onset_frac: float = 0.05) -> dict:
    """Detecta eventos E1–E6.

    series: 'wrist_speed' (obligatoria); opcionales 'knee_angle' (interior, rad), 'elbow_angle'
            (interior, rad) y 'elbow_omega' (rad/s, extensión positiva).
    Devuelve {evento: {"t": s, "metodo": str}} y la clave especial "_warnings".
    """
    manual = manual or {}
    t, v = (np.asarray(a, float) for a in series["wrist_speed"])
    pk = int(np.argmax(v))
    ev, warn = {}, []

    if "impacto" in manual:
        ev["impacto"] = dict(t=float(manual["impacto"]), metodo="manual")
        pk = int(np.argmin(np.abs(t - manual["impacto"])))
    else:
        ev["impacto"] = dict(t=float(t[pk]), metodo="proxy_max_rapidez_muneca")
    vmax = v[pk]
    if pk in (0, len(v) - 1):
        warn.append("impacto en el borde del registro: el pico de rapidez no es un máximo interior (¿recorte del clip?)")

    i0 = _sustained_first(v[: pk + 1] > start_frac * vmax, sustain)
    ev["inicio"] = dict(t=float(t[i0 if i0 is not None else 0]), metodo=f"rapidez>{start_frac:.0%}pico")
    if i0 is None:
        warn.append("inicio: no se halló umbral sostenido; se usó la primera muestra")

    after = _sustained_first(v[pk:] < end_frac * vmax, sustain)
    if after is None:
        ev["fin"] = dict(t=float(t[-1]), metodo="fin_de_registro")
        warn.append("fin: la rapidez no baja de 20% del pico; se usó el fin del registro")
    else:
        ev["fin"] = dict(t=float(t[pk + after]), metodo=f"rapidez<{end_frac:.0%}pico")

    t_in, t_imp = ev["inicio"]["t"], ev["impacto"]["t"]       # t_in = inicio del movimiento de la MUÑECA
    if "knee_angle" in series:
        # La carga de rodillas ocurre ANTES de que se mueva la muñeca: se busca desde el inicio del registro.
        tk, k = (np.asarray(a, float) for a in series["knee_angle"])
        m = tk <= t_imp
        if m.any():
            tm, km = tk[m], k[m]
            i_min = int(np.argmin(km))
            ev["max_flex_rodilla"] = dict(t=float(tm[i_min]), metodo="min_angulo_interior")
            stand = float(np.max(km[: i_min + 1]))                 # postura erguida previa
            thr = stand - flex_onset_frac * (stand - km[i_min])
            above = np.where(km[: i_min + 1] >= thr)[0]
            if above.size and stand - km[i_min] > 0:
                ev["inicio_flexion_rodilla"] = dict(
                    t=float(tm[above[-1]]), metodo=f"angulo<{1-flex_onset_frac:.0%}_del_rango_de_flexion")
        else:
            warn.append("rodilla: sin muestras antes del impacto (¿relojes distintos?)")
    if "elbow_omega" in series:
        to, w = (np.asarray(a, float) for a in series["elbow_omega"])
        m = (to >= t_in) & (to <= t_imp + omega_margin_s)
        if m.any():
            ev["pico_omega_codo"] = dict(t=float(to[m][np.argmax(w[m])]), metodo="max_omega_extension")
    if "elbow_angle" in series and "pico_omega_codo" in ev:
        te, e = (np.asarray(a, float) for a in series["elbow_angle"])
        m = (te >= t_in) & (te <= ev["pico_omega_codo"]["t"])
        if m.any():
            ev["max_flex_codo"] = dict(t=float(te[m][np.argmin(e[m])]), metodo="min_angulo_interior")

    if not (ev["inicio"]["t"] < ev["impacto"]["t"] <= ev["fin"]["t"]):
        warn.append("orden inválido: se esperaba inicio < impacto <= fin")
    if "max_flex_rodilla" in ev and ev["max_flex_rodilla"]["t"] > ev["impacto"]["t"]:
        warn.append("orden inválido: máxima flexión de rodilla posterior al impacto")
    ev["_warnings"] = warn
    return ev


def phase_table(ev: dict) -> pd.DataFrame:
    """Fases con los eventos disponibles: carga, aceleración, seguimiento."""
    start = "inicio_flexion_rodilla" if "inicio_flexion_rodilla" in ev else "inicio"
    pts = [("carga", start, "max_flex_rodilla"), ("aceleracion", "max_flex_rodilla", "impacto"),
           ("seguimiento", "impacto", "fin")]
    rows = []
    for name, a, b in pts:
        if a in ev and b in ev:
            rows.append(dict(fase=name, evento_ini=a, evento_fin=b, t_ini=ev[a]["t"], t_fin=ev[b]["t"],
                             duracion_ms=1000 * (ev[b]["t"] - ev[a]["t"])))
    if "max_flex_rodilla" not in ev and "inicio" in ev:
        rows.insert(0, dict(fase="inicio→impacto (sin evento de rodilla)", evento_ini="inicio",
                            evento_fin="impacto", t_ini=ev["inicio"]["t"], t_fin=ev["impacto"]["t"],
                            duracion_ms=1000 * (ev["impacto"]["t"] - ev["inicio"]["t"])))
    return pd.DataFrame(rows)


def time_normalize(t, y, t0: float, t1: float, n: int = 101) -> np.ndarray:
    """Lleva y(t) en [t0, t1] a n puntos (0–100 %)."""
    return np.interp(np.linspace(t0, t1, n), np.asarray(t, float), np.asarray(y, float))


def peak_times(series: dict[str, Series], window: tuple[float, float], use_abs: bool = True) -> dict:
    """Tiempo y amplitud del pico de cada serie dentro de `window`."""
    out = {}
    for label, (t, y) in series.items():
        t, y = np.asarray(t, float), np.asarray(y, float)
        m = (t >= window[0]) & (t <= window[1])
        if not m.any():
            continue
        yy = np.abs(y[m]) if use_abs else y[m]
        i = int(np.argmax(yy))
        out[label] = dict(t=float(t[m][i]), amp=float(yy[i]))
    return out


def sequence_metrics(peaks: dict, order: list[str]) -> dict:
    """Lags entre picos consecutivos (ms) y τ de Kendall frente al orden esperado (1 = orden perfecto).

    No se calculan razones de amplitud: las series tienen unidades distintas (rad/s vs m/s).
    """
    labels = [l for l in order if l in peaks]
    if len(labels) < 2:
        return dict(labels=labels, lags_ms=[], tau=None, nota="se requieren >=2 series con pico")
    times = [peaks[l]["t"] for l in labels]
    lags = [1000 * (times[i + 1] - times[i]) for i in range(len(times) - 1)]
    tau = kendalltau(np.arange(len(times)), times)[0] if len(times) >= 3 else (1.0 if lags[0] > 0 else -1.0)
    return dict(labels=labels, lags_ms=lags, tau=float(tau),
                nota="cadena incompleta" if len(labels) < len(order) else "")
