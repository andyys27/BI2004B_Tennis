"""Secuencia con huecos, orden observado, picos articulares, correlaciones y resumen de QC."""
import numpy as np
import pandas as pd
import pytest

from saque.analysis import events, stats


def test_peak_times_ignora_nan():
    t = np.linspace(0, 1, 101)
    y = np.sin(np.pi * t)            # pico en t = 0.5
    y[:10] = np.nan                  # antes: argmax devolvía el primer NaN
    pk = events.peak_times({"a": (t, y)}, (0, 1))
    assert pk["a"]["t"] == pytest.approx(0.5)
    assert np.isfinite(pk["a"]["amp"])
    assert pk["a"]["frac_valida"] == pytest.approx(0.9, abs=0.01)


def test_peak_times_omite_series_con_pocos_datos():
    t = np.linspace(0, 1, 11)
    y = np.full(11, np.nan)
    y[5] = 1.0
    assert "a" not in events.peak_times({"a": (t, y)}, (0, 1), min_valid_frac=0.5)


def test_orden_observado_no_es_el_esperado():
    peaks = {"a": {"t": 0.3}, "b": {"t": 0.1}, "c": {"t": 0.2}}
    sm = events.sequence_metrics(peaks, ["a", "b", "c"])
    assert sm["observed"] == "b > c > a"
    assert sm["n_invertidos"] == 1
    assert sm["lags_ms"] == pytest.approx([-200, 100])


def test_orden_con_empates_y_cadena_incompleta():
    peaks = {"a": {"t": 0.1}, "b": {"t": 0.1}}
    sm = events.sequence_metrics(peaks, ["a", "b", "c"])
    assert sm["observed"] == "a = b"
    assert "cadena incompleta" in sm["nota"] and "simultáneo" in sm["nota"]


def test_joint_peaks():
    t = np.linspace(0, 1, 101)
    kin = pd.DataFrame({"time_s": t, "omega_interior_codo": -5 * np.sin(np.pi * t),
                        "alpha_interior_codo": np.cos(np.pi * t)})
    jp = stats.joint_peaks(kin, ["codo", "rodilla"], 0, 1, t_ref=1.0)
    w = jp[jp["magnitud"] == "omega"].iloc[0]
    assert w["pico_abs"] == pytest.approx(5)
    assert w["t_rel_impacto_ms"] == pytest.approx(-500)
    assert set(jp["articulacion"]) == {"codo"}


def test_spearman_matrix():
    wide = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10], "c": [5, 4, 3, 2, 1],
                         "k": [1, 1, 1, 1, 1]})
    m = stats.spearman_matrix(wide).set_index(["var_a", "var_b"])
    assert m.loc[("a", "b"), "rho"] == pytest.approx(1)
    assert m.loc[("a", "c"), "rho"] == pytest.approx(-1)
    assert "k" not in set(m.reset_index()["var_a"]) | set(m.reset_index()["var_b"])


def test_qc_summary():
    log = pd.DataFrame({"archivo": ["lateral"] * 5,
                        "indice": [np.nan, np.nan, 10, 11, np.nan],
                        "tipo": ["longitud_segmento", "longitud_segmento", "salto_tracking",
                                 "salto_tracking", "huecos"],
                        "accion": ["tronco: 10 fotogramas a NaN", "brazo: 5 fotogramas a NaN",
                                   "marcador_codo_a_NaN", "marcador_codo_a_NaN",
                                   "x_codo: 4 interpoladas, 22 sin resolver"]})
    q = stats.qc_summary(log).set_index("tipo")["muestras"]
    assert q["longitud_segmento"] == 15
    assert q["salto_tracking"] == 2
    assert "huecos" not in q


def test_inicio_es_fin_de_la_quietud_no_el_borde():
    """Rutina previa (muñeca a 0.6 m/s) -> quietud -> lanzamiento/carga -> impacto."""
    fs = 120.0
    t = np.arange(0, 3.0, 1 / fs)
    v = np.where(t < 0.4, 0.6, 0.05)                         # rutina previa y luego quietud
    v = v + np.clip((t - 0.6) * 2, 0, None) * (t < 2.0)       # el gesto empieza en t = 0.6
    v = v + 8 * np.exp(-0.5 * ((t - 2.5) / 0.08) ** 2)        # impacto en 2.5
    knee = 2.8 - 1.0 * np.exp(-0.5 * ((t - 1.8) / 0.25) ** 2)  # flexión de rodilla después
    ev = events.detect_events({"wrist_speed": (t, v), "knee_angle": (t, knee)})
    # Fin de la quietud = cruce del umbral (5 % del pico) en la rampa que empieza en 0.6 s,
    # no el borde de la ventana (t = 0) ni la rutina previa.
    t_cross = t[np.argmax((t > 0.4) & (v > 0.05 * v.max()))]
    assert ev["inicio"]["t"] == pytest.approx(t_cross, abs=2 / fs)
    assert 0.6 < ev["inicio"]["t"] < 1.0
    assert ev["inicio"]["metodo"].startswith("fin_quietud")
