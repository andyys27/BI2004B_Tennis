"""Reporte Markdown generado a partir de las tablas del pipeline."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _md(df: pd.DataFrame, cols: list[str] | None = None, floatfmt: str = ".3g") -> str:
    if df.empty:
        return "_sin datos_"
    d = df[[c for c in (cols or df.columns) if c in df]]
    head = "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n"
    fmt = lambda v: format(v, floatfmt) if isinstance(v, float) else str(v)  # noqa: E731
    return head + "\n".join("| " + " | ".join(fmt(v) for v in r) + " |" for r in d.itertuples(index=False))


def write_report(ctx, figures: list[Path]) -> Path:
    t = ctx.paths.tables
    rep = ctx.paths.reports
    rel = lambda p: Path(p).relative_to(rep).as_posix()  # noqa: E731
    fig = {Path(p).stem: rel(p) for p in figures}
    cal, serves, sync = _read(t / "calibracion.csv"), _read(t / "saques.csv"), _read(t / "sincronizacion.csv")
    src = _read(t / "fuente_marcadores.csv")
    trends, index = _read(t / "fatiga_tendencias.csv"), _read(t / "fatiga_indice.csv")
    seq, jac = _read(t / "sequence.csv"), _read(t / "jacobian_metrics.csv")
    phases = _read(t / "phases.csv")

    lines = [
        "# Reporte: análisis cinemático del saque kick (MediaPipe + estampitas)",
        "",
        "Generado automáticamente por `saque run`. Cinemática 2D por vista; todas las asociaciones "
        "y tendencias entre saques son exploratorias (n = 5 saques, una jugadora).",
        "",
        "## 1. Datos y calibración",
        _md(cal),
        "",
        "## 2. Origen de los marcadores",
        "Cada marcador usa el centroide de la estampita cuando se detecta y, si no, el landmark de "
        "MediaPipe corregido por el desfase mediano estampita-landmark.",
        "",
        _md(src[src["con_estampita"]] if "con_estampita" in src else src,
            ["vista", "marcador", "frac_estampita", "frac_pose", "frac_hueco", "estampitas_descartadas"]),
        "",
    ]
    if "fuente_marcadores" in fig:
        lines += [f"![fuentes]({fig['fuente_marcadores']})", ""]
    lines += ["## 3. Saques detectados", _md(serves, ["vista", "saque", "t_ini", "t_impacto", "t_fin"]), ""]
    if not sync.empty:
        lines += ["Sincronización entre vistas: t_vista = t_referencia + lag_s (por correlación de la altura de cadera). El control compara con los impactos que la vista detecta por su cuenta con la muñeca:", "",
                  _md(sync), ""]
    for k in ("deteccion_saques", "impactos_lateral", "impactos_posterior"):
        if k in fig:
            lines += [f"![{k}]({fig[k]})", ""]
    if not phases.empty:
        ph = phases.pivot_table(index=["vista", "saque"], columns="fase", values="duracion_ms").reset_index()
        lines += ["## 4. Fases (duración en ms)", _md(ph, floatfmt=".0f"), ""]
    if not seq.empty:
        lines += ["## 5. Secuencia próximo-distal", "τ de Kendall = 1 indica picos en el orden esperado.",
                  "", _md(seq), ""]
    if "curvas_por_saque" in fig:
        lines += ["## 6. Curvas por saque (inicio → impacto)", f"![curvas]({fig['curvas_por_saque']})", ""]
    lines += [
        "## 7. Indicadores de posible fatiga",
        "Pendiente por saque, cambio del primer al último saque y ρ de Spearman con IC95 % bootstrap. "
        "Con 5 saques los intervalos son amplios: una tendencia es una señal a confirmar, no una "
        "prueba de fatiga (también puede reflejar cambios de intención o técnica).",
        "",
        _md(trends, ["variable", "confianza", "unidad", "media", "cv_pct", "pendiente_por_saque",
                     "cambio_primero_ultimo_pct", "rho_spearman", "ic95_inf", "ic95_sup"]),
        "",
        "Confianza: *alta* = posiciones, ángulos y tiempos; *baja* = picos de velocidad angular del "
        "brazo, sensibles a oclusiones y escorzo 2D cerca del impacto (ver "
        "`tables/fatiga_valores_por_saque.csv`, columna `frac_valida_pre_impacto`).",
        "",
        "Índice de fatiga (último tercio vs primer tercio, variables de desempeño):",
        "",
        _md(index),
        "",
    ]
    if "fatiga_tendencias" in fig:
        lines += [f"![fatiga]({fig['fatiga_tendencias']})", ""]
    if not jac.empty:
        lines += ["## 8. Jacobiano v = J(θ)ω",
                  _md(jac, ["vista", "saque", "L_brazo", "cv_brazo", "L_antebrazo", "cv_antebrazo",
                            "rmse_rel_pico", "corr_speed", "pico_speed_directa"]), ""]
    for k, v in fig.items():
        if k.startswith(("jacobiano_", "dinamica_inversa_")):
            lines += [f"![{k}]({v})", ""]
    lines += [
        "## 9. Limitaciones",
        "- 2D por vista: los ángulos son proyecciones en el plano de cada cámara.",
        "- Escala: vara de 1 m corregida por profundidad suponiendo cámara nivelada a 1.30 m (ver "
        "estatura aparente en la tabla de calibración); escala fija por vista aunque la jugadora se "
        "mueve en profundidad. Los ángulos y tiempos no dependen de la escala.",
        "- El impacto es un proxy (máxima rapidez de muñeca cerca de su altura máxima); no se rastrean "
        "raqueta ni pelota.",
        "- Dinámica inversa exploratoria: GRF = 0 y muñeca sin carga externa.",
        "- MediaPipe pierde precisión en el brazo de golpeo a alta velocidad; las estampitas reducen "
        "el problema solo donde se detectan.",
    ]
    if ctx.warnings:
        lines += ["", "## Advertencias del pipeline", *[f"- {w}" for w in dict.fromkeys(ctx.warnings)]]
    path = rep / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
