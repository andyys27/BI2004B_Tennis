"""Datos sintéticos con los encabezados y problemas documentados de los CSV de Kinovea.

Verdad conocida: brazo de 2 eslabones (hombro fijo) con θ1(t), θ2(t) analíticos.
"""
from pathlib import Path
import numpy as np
import pandas as pd

L1, L2 = 0.30, 0.28
SHOULDER = (0.50, 1.40)          # m (y hacia arriba)


def th1(t): return 1.0 + 0.6 * np.sin(2 * np.pi * 0.7 * t)
def th2(t): return 0.4 + 0.9 * np.sin(2 * np.pi * 0.9 * t + 0.5)
def w1(t): return 0.6 * 2 * np.pi * 0.7 * np.cos(2 * np.pi * 0.7 * t)
def a1(t): return -0.6 * (2 * np.pi * 0.7) ** 2 * np.sin(2 * np.pi * 0.7 * t)


def jittered_time(t_end, start=0.0, seed=0):
    """Intervalos alternando 0.008/0.009 s como en las exportaciones reales."""
    n = int(t_end / 0.00850) + 1
    dt = np.where(np.arange(n) % 2 == 0, 0.008, 0.009)
    return start + np.concatenate([[0], np.cumsum(dt[:-1])])


def arm_markers(t, noise=0.0015, seed=1, y_down=True):
    rng = np.random.default_rng(seed)
    sx, sy = SHOULDER
    ex, ey = sx + L1 * np.cos(th1(t)), sy + L1 * np.sin(th1(t))
    wx, wy = ex + L2 * np.cos(th2(t)), ey + L2 * np.sin(th2(t))
    out = {}
    for n, (x, y) in dict(homb=(np.full_like(t, sx), np.full_like(t, sy)), codo=(ex, ey), mun=(wx, wy)).items():
        yy = -y if y_down else y
        out[n] = (x + rng.normal(0, noise, t.size), yy + rng.normal(0, noise, t.size))
    return out


def write_raw(root: Path, seed=0):
    root = Path(root)
    (root / "posterior").mkdir(parents=True, exist_ok=True)
    (root / "lateral").mkdir(parents=True, exist_ok=True)

    # posterior: brazo (cm, con encabezados reales)
    t = jittered_time(2.955)
    m = arm_markers(t)
    pd.DataFrame({"time_s": t.round(3),
                  "x_mun_cm": m["mun"][0] * 100, "y_mun_cm": m["mun"][1] * 100,
                  "x_codo_cm": m["codo"][0] * 100, "y_codo_cm": m["codo"][1] * 100,
                  "x_homb_cm": m["homb"][0] * 100, "y_homb_cm": m["homb"][1] * 100}
                 ).to_csv(root / "posterior/cinematica_lineal_brazo_pos.csv", index=False)

    # lateral lineal: sin sufijo _cm, con tiempo duplicado (filas 185-186) y un salto de etiqueta
    t = jittered_time(4.523)
    m = arm_markers(t, seed=2)       # reutiliza geometría: hombro->cadera, codo->rodilla, muñeca->tobillo
    d = pd.DataFrame({"tiempo_s": t.round(3),
                      "x_cadera": m["homb"][0] * 100, "y_cadera": m["homb"][1] * 100,
                      "x_rodilla": m["codo"][0] * 100, "y_rodilla": m["codo"][1] * 100,
                      "x_tobillo": m["mun"][0] * 100, "y_tobillo": m["mun"][1] * 100})
    d.loc[300, ["x_tobillo", "y_tobillo"]] += 40          # salto de tracking aislado
    d.loc[185, "tiempo_s"] = d.loc[184, "tiempo_s"]        # tiempo repetido
    d.to_csv(root / "lateral/cinematica_lineal_lateral.csv", index=False)

    # posterior angular: rodilla (Tiempo (ms) + tiempo_s, 0.593-...), tobillo con origen distinto
    th = lambda tt: 120 + 25 * np.sin(2 * np.pi * 0.8 * tt)
    t_k = jittered_time(2.338, start=0.593)
    pd.DataFrame({"Tiempo (ms)": (t_k * 1000).round(0), "tiempo_s": t_k.round(3),
                  "Angulo_Rod": th(t_k), "angulo_MA_Rod": th(t_k), "v_Rod": 0.0, "a_Rod": 0.0}
                 ).to_csv(root / "posterior/cinematica_angular_rodilla.csv", index=False)
    pd.DataFrame({"tiempo_ms": (t_k * 1000).round(0), "tiempo_s": t_k.round(3),
                  "angulo_codo": th(t_k) / 2, "angulo_codo_clean": th(t_k) / 2, "v_codo": 0.0, "a_codo": 0.0}
                 ).to_csv(root / "posterior/cinematica_angular_brazo.csv", index=False)
    t_t = t_k - 0.592                                      # mismo clip, otro origen temporal
    pd.DataFrame({"tiempo_ms": (t_t * 1000).round(0), "tiempo_s": t_t.round(3),
                  "angulo_tob": 90 + 15 * np.sin(2 * np.pi * 0.8 * (t_t + 0.592)),
                  "angulo_MA_tob": 0.0, "v_tob": 0.0, "a_tob": 0.0}
                 ).to_csv(root / "posterior/cinematica_angular_tobillo.csv", index=False)
    # duplicado byte a byte fuera de data/
    (root / "posterior/cinematica_angular_rodilla_copia.csv").write_bytes(
        (root / "posterior/cinematica_angular_rodilla.csv").read_bytes())
    return root
