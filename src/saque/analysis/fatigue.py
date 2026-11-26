"""Indicadores cinemáticos de posible fatiga a lo largo de la serie de saques.

Para cada variable se calcula un valor por saque (pico, mínimo, etc.) dentro de la ventana
inicio-impacto y se resume su tendencia con el número de saque:
  - pendiente por regresión lineal (unidades por saque) y cambio porcentual primero->último;
  - ρ de Spearman con IC bootstrap (exploratorio; con n=5 los IC son muy amplios);
  - CV entre saques.
Una caída de picos de velocidad o un aumento de variabilidad es *compatible* con fatiga, pero
también con cambios de intención o técnica: no es una medida fisiológica de fatiga.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import linregress

from .stats import bootstrap_spearman

STATS = {
    "max": np.nanmax,
    "min": np.nanmin,
    "absmax": lambda y: np.nanmax(np.abs(y)),
    "mean": np.nanmean,
}


def per_serve_values(kin_by_serve: dict[int, pd.DataFrame], windows: pd.DataFrame,
                     variables: dict, quality_window_s: float = 0.2,
                     min_valid_frac: float = 0.8) -> pd.DataFrame:
    """Valor de cada variable por saque dentro de [t_ini, t_impacto + margen].

    `frac_valida_pre_impacto`: fracción de muestras finitas en los `quality_window_s` previos al
    impacto (donde ocurren los picos del brazo). Si es menor que `min_valid_frac`, el pico pudo
    caer dentro de un hueco: `valor` queda NaN (excluido de tendencias) y el cálculo original se
    conserva en `valor_sin_control`.
    """
    rows = []
    for _, w in windows.iterrows():
        kin = kin_by_serve.get(int(w["saque"]))
        if kin is None:
            continue
        m = (kin["time_s"] >= w["t_ini"]) & (kin["time_s"] <= w["t_fin_analisis"])
        q = (kin["time_s"] >= w["t_impacto"] - quality_window_s) & (kin["time_s"] <= w["t_impacto"])
        for name, spec in variables.items():
            col = spec["column"]
            y = kin.loc[m, col].to_numpy(float) if col in kin else np.array([])
            val = float(STATS[spec.get("stat", "max")](y)) if np.isfinite(y).any() else np.nan
            yq = kin.loc[q, col].to_numpy(float) if col in kin else np.array([])
            frac = float(np.isfinite(yq).mean()) if yq.size else np.nan
            ok = not (np.isfinite(frac) and frac < min_valid_frac)
            rows.append(dict(saque=int(w["saque"]), variable=name, valor=val if ok else np.nan,
                             valor_sin_control=val, unidad=spec.get("unit", ""), columna=col,
                             confianza=spec.get("confidence", "media"),
                             frac_valida_pre_impacto=frac,
                             excluido="" if ok else f"<{min_valid_frac:.0%} muestras válidas pre-impacto"))
    return pd.DataFrame(rows)


def trend_table(values: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    """Tendencia de cada variable con el número de saque."""
    rows = []
    for name, g in values.groupby("variable", sort=False):
        g = g.dropna(subset=["valor"]).sort_values("saque")
        x, y = g["saque"].to_numpy(float), g["valor"].to_numpy(float)
        row = dict(variable=name, unidad=g["unidad"].iloc[0] if len(g) else "", n=len(g),
                   confianza=g["confianza"].iloc[0] if "confianza" in g and len(g) else "")
        if len(g) >= 3 and np.ptp(y) > 0:
            lr = linregress(x, y)
            sp = bootstrap_spearman(x, y, n_boot=n_boot)
            first, last = y[0], y[-1]
            row.update(media=float(y.mean()), cv_pct=float(100 * y.std(ddof=1) / abs(y.mean()))
                       if y.mean() else np.nan,
                       pendiente_por_saque=float(lr.slope), r2=float(lr.rvalue ** 2),
                       p_valor_pendiente=float(lr.pvalue),
                       cambio_primero_ultimo_pct=float(100 * (last - first) / abs(first))
                       if first else np.nan,
                       rho_spearman=sp["rho"], ic95_inf=sp["ic_inferior"], ic95_sup=sp["ic_superior"])
        else:
            row.update(nota="menos de 3 saques válidos o valor constante")
        rows.append(row)
    return pd.DataFrame(rows)


def phase_duration_values(phases: pd.DataFrame,
                          fases: tuple[str, ...] = ("carga", "aceleracion")) -> pd.DataFrame:
    """Duración de fases por saque como variable de fatiga (no depende de derivadas).

    El seguimiento se excluye: su fin depende de que la ventana post-impacto alcance a cubrirlo.
    """
    phases = phases[phases["fase"].isin(fases)] if not phases.empty else phases
    if phases.empty:
        return pd.DataFrame()
    return pd.DataFrame(dict(saque=phases["saque"].astype(int),
                             variable="duracion_" + phases["fase"].astype(str),
                             valor=phases["duracion_ms"].astype(float), unidad="ms",
                             columna="eventos", confianza="alta",
                             frac_valida_pre_impacto=np.nan, excluido=""))


def fatigue_index(values: pd.DataFrame, performance_vars: list[str]) -> pd.DataFrame:
    """Índice simple: % de cambio del último tercio respecto al primero en variables de desempeño.

    Negativo = caída del desempeño hacia el final de la serie.
    """
    rows = []
    for name in performance_vars:
        g = values[values["variable"] == name].dropna(subset=["valor"]).sort_values("saque")
        if len(g) < 3:
            continue
        k = max(1, len(g) // 3)
        first, last = g["valor"].iloc[:k].mean(), g["valor"].iloc[-k:].mean()
        rows.append(dict(variable=name, primer_tercio=first, ultimo_tercio=last,
                         cambio_pct=float(100 * (last - first) / abs(first)) if first else np.nan,
                         saques_por_tercio=k))
    return pd.DataFrame(rows)
