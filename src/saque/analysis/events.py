"""Eventos, fases y secuencia próximo-distal de un saque.

El impacto se toma como PROXY (máximo de rapidez de muñeca) porque no se rastrea raqueta ni pelota.
Si se marcó el fotograma de contacto en el video, pasarlo en `manual={"impacto": t_s}`.
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


def _last_still_end(v: np.ndarray, thr: float, n: int, stop: int) -> int | None:
    """Índice siguiente al final del último tramo de >= n muestras con v < thr antes de `stop`."""
    run = 0
    last = None
    for i in range(stop):
        run = run + 1 if v[i] < thr else 0
        if run >= n:
            last = i + 1
    return last if last is not None and last < stop else None


def detect_events(series: dict[str, Series], manual: dict | None = None, start_frac: float = 0.05,
                  end_frac: float = 0.20, sustain: int = 3, omega_margin_s: float = 0.05,
                  flex_onset_frac: float = 0.05, still_s: float = 0.1,
                  stand_window_s: float = 1.0) -> dict:
    """Detecta eventos E1–E6.

    series: 'wrist_speed' (obligatoria); opcionales 'knee_angle' (interior, rad), 'elbow_angle'
            (interior, rad) y 'elbow_omega' (rad/s, extensión positiva).
    Devuelve {evento: {"t": s, "metodo": str}} y la clave especial "_warnings".
    """
    manual = manual or {}
    t, v = (np.asarray(a, float) for a in series["wrist_speed"])
    if not np.isfinite(v).any():
        raise ValueError("wrist_speed no tiene muestras válidas")
    # Huecos (oclusión, QC) interpolados solo para los umbrales: un NaN nunca cumple "v < umbral".
    v = pd.Series(v).interpolate(limit_direction="both").to_numpy()
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


    after = _sustained_first(v[pk:] < end_frac * vmax, sustain)
    if after is None:
        ev["fin"] = dict(t=float(t[-1]), metodo="fin_de_registro")
        warn.append("fin: la rapidez no baja de 20% del pico; se usó el fin del registro")
    else:
        ev["fin"] = dict(t=float(t[pk + after]), metodo=f"rapidez<{end_frac:.0%}pico")

    t_imp = ev["impacto"]["t"]
    if "knee_angle" in series:
        # La carga de rodillas ocurre ANTES de que se mueva la muñeca: se busca desde el inicio del registro.
        tk, k = (np.asarray(a, float) for a in series["knee_angle"])
        m = tk <= t_imp
        if m.any():
            tm, km = tk[m], k[m]
            i_min = int(np.argmin(km))
            ev["max_flex_rodilla"] = dict(t=float(tm[i_min]), metodo="min_angulo_interior")
            # Postura erguida: máximo en el segundo previo a la máxima flexión (no en toda la ventana:
            # la rutina previa puede tener la rodilla más extendida y desplazar el umbral).
            j0 = int(np.searchsorted(tm, tm[i_min] - stand_window_s))
            seg = km[j0: i_min + 1]
            stand = float(np.nanmax(seg)) if np.isfinite(seg).any() else float(km[i_min])
            thr = stand - flex_onset_frac * (stand - km[i_min])
            above = j0 + np.where(km[j0: i_min + 1] >= thr)[0]
            if above.size and stand - km[i_min] > 0:
                ev["inicio_flexion_rodilla"] = dict(
                    t=float(tm[above[-1]]), metodo=f"angulo<{1-flex_onset_frac:.0%}_del_rango_de_flexion")
        else:
            warn.append("rodilla: sin muestras antes del impacto (¿relojes distintos?)")
    # Inicio: fin de la última quietud de la muñeca (>= still_s por debajo de start_frac·pico) antes
    # de que empiece la flexión de rodilla (o del impacto). Buscar hacia atrás evita tomar como
    # inicio la rutina previa (botar la pelota), que ya supera el umbral, y la pausa del armado.
    fs = 1.0 / np.median(np.diff(t)) if len(t) > 1 else 120.0
    n_still = max(sustain, int(round(still_s * fs)))
    t_stop = ev.get("inicio_flexion_rodilla", ev.get("max_flex_rodilla", ev["impacto"]))["t"]
    stop = int(np.searchsorted(t, t_stop))
    # Umbral con el pico real cerca del impacto (la rapidez en el fotograma marcado como impacto
    # puede ser bastante menor que el pico y daría un umbral demasiado bajo).
    near = (t >= t_imp - 0.15) & (t <= t_imp + 0.05)
    thr0 = start_frac * (float(np.nanmax(v[near])) if near.any() else vmax)
    i0 = _last_still_end(v, thr0, n_still, min(stop, pk))
    if i0 is None and 0 < stop <= pk and v[stop - 1] < thr0:
        # La muñeca sigue quieta cuando empiezan las rodillas: su movimiento empieza después.
        i0 = _last_still_end(v, thr0, n_still, pk)
    if i0 is not None:
        ev["inicio"] = dict(t=float(t[i0]), metodo=f"fin_quietud_muñeca(<{start_frac:.0%}pico)")
    else:
        i0 = _sustained_first(v[: pk + 1] > start_frac * vmax, sustain)
        ev["inicio"] = dict(t=float(t[i0 if i0 is not None else 0]), metodo=f"rapidez>{start_frac:.0%}pico")
        warn.append("inicio: no se halló quietud de la muñeca antes de la carga; se usó el primer "
                    "cruce del umbral (puede ser el borde de la ventana)")
    t_in = ev["inicio"]["t"]
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
    """Fases con los eventos disponibles: preparación (lanzamiento y armado), carga, aceleración y
    seguimiento."""
    start = "inicio_flexion_rodilla" if "inicio_flexion_rodilla" in ev else "inicio"
    pts = [("preparacion", "inicio", "inicio_flexion_rodilla"), ("carga", start, "max_flex_rodilla"), ("aceleracion", "max_flex_rodilla", "impacto"),
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


def peak_times(series: dict[str, Series], window: tuple[float, float], use_abs: bool = True,
               min_valid_frac: float = 0.0) -> dict:
    """Tiempo y amplitud del pico de cada serie dentro de `window` (ignora huecos NaN).

    `frac_valida` es la fracción de muestras finitas en la ventana; si es menor que
    `min_valid_frac` la serie se omite (pico poco fiable).
    """
    out = {}
    for label, (t, y) in series.items():
        t, y = np.asarray(t, float), np.asarray(y, float)
        m = (t >= window[0]) & (t <= window[1])
        if not m.any():
            continue
        yy = np.abs(y[m]) if use_abs else y[m]
        ok = np.isfinite(yy)
        frac = float(ok.mean())
        if not ok.any() or frac < min_valid_frac:
            continue
        i = int(np.nanargmax(yy))
        out[label] = dict(t=float(t[m][i]), amp=float(yy[i]), frac_valida=frac)
    return out


def observed_order(peaks: dict, labels: list[str], tie_ms: float = 1e-6) -> str:
    """Orden temporal observado de los picos; picos simultáneos se unen con '='."""
    items = sorted(labels, key=lambda l: peaks[l]["t"])
    out = items[:1]
    for prev, cur in zip(items, items[1:]):
        sep = " = " if 1000 * abs(peaks[cur]["t"] - peaks[prev]["t"]) <= tie_ms else " > "
        out.append(sep + cur)
    return "".join(out)


def sequence_metrics(peaks: dict, order: list[str], tie_ms: float = 1e-6) -> dict:
    """Orden observado, lags entre picos consecutivos del orden esperado (ms; negativo = invertido)
    y τ de Kendall frente al orden esperado (1 = orden perfecto, −1 = invertido).

    No se calculan razones de amplitud: las series tienen unidades distintas (rad/s vs m/s).
    """
    labels = [l for l in order if l in peaks]
    if len(labels) < 2:
        return dict(labels=labels, observed="", lags_ms=[], tau=None, n_invertidos=None,
                    nota="se requieren >=2 series con pico")
    times = [peaks[l]["t"] for l in labels]
    lags = [1000 * (times[i + 1] - times[i]) for i in range(len(times) - 1)]
    if len(set(times)) == 1:
        tau = float("nan")
    elif len(times) >= 3:
        tau = float(kendalltau(np.arange(len(times)), times)[0])
    else:
        tau = 1.0 if lags[0] > 0 else -1.0
    notas = []
    if len(labels) < len(order):
        notas.append("cadena incompleta: " + ", ".join(l for l in order if l not in peaks))
    ties = sum(abs(x) <= tie_ms for x in lags)
    if ties:
        notas.append(f"{ties} pico(s) simultáneo(s)")
    return dict(labels=labels, observed=observed_order(peaks, labels, tie_ms), lags_ms=lags, tau=tau,
                n_invertidos=int(sum(x < -tie_ms for x in lags)), nota="; ".join(notas))
