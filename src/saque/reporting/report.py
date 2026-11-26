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
        if "fases" in fig:
            lines += [f"![fases]({fig['fases']})", ""]
    if not seq.empty:
        lines += ["## 5. Secuencia próximo-distal",
                  "Pico de |ω| (o rapidez de muñeca) de cada segmento entre el inicio y 50 ms después del "
                  "impacto, ignorando huecos; una serie con menos del 50 % de muestras válidas en la "
                  "ventana se omite. `orden_observado` ordena los picos en el tiempo ('=' = mismo "
                  "fotograma). `lags_ms` son las diferencias entre picos consecutivos del orden esperado "
                  "(negativo = invertido). τ de Kendall = 1 indica el orden esperado.",
                  "", _md(seq, ["vista", "saque", "orden_observado", "tau_kendall", "n_invertidos",
                                "lags_ms", "nota"]), ""]
        if "secuencia_cinematica" in fig:
            lines += [f"![secuencia]({fig['secuencia_cinematica']})", ""]
    if "curvas_por_saque" in fig:
        lines += ["## 6. Curvas por saque (inicio → impacto)", f"![curvas]({fig['curvas_por_saque']})", ""]
    jp = _read(t / "picos_articulares.csv")
    if "cinematica_articular" in fig or not jp.empty:
        lines += ["## 6b. Cinemática angular θ, ω, α por articulación (vista lateral)",
                  "Ángulos interiores (180° = articulación extendida), normalizados de inicio a impacto. "
                  "La banda gris es la media ± DE de los 5 saques. α es la segunda derivada del ángulo "
                  "filtrado y es la magnitud más ruidosa.", ""]
        if "cinematica_articular" in fig:
            lines += [f"![θωα]({fig['cinematica_articular']})", ""]
        if not jp.empty:
            g = jp.groupby(["vista", "articulacion", "magnitud", "unidad"]).agg(
                media=("pico_abs", "mean"), de=("pico_abs", "std"),
                t_rel_impacto_ms=("t_rel_impacto_ms", "median"),
                frac_valida=("frac_valida", "mean")).reset_index()
            lines += ["Picos de |ω| y |α| entre inicio e impacto (media ± DE de los saques; tiempo "
                      "mediano respecto al impacto):", "", _md(g), ""]
    vr = _read(t / "variabilidad_rmsd_articular.csv")
    if "variabilidad_saques" in fig or not vr.empty:
        lines += ["## 6c. Consistencia entre saques",
                  "DE entre saques a lo largo del gesto y RMSD de cada saque respecto a la curva media "
                  "(ángulos, °). Un saque con RMSD alto en todas las articulaciones fue técnicamente "
                  "distinto del resto.", ""]
        if not vr.empty:
            th = vr[vr["variable"].str.startswith("theta")]
            lines += [_md(th.pivot_table(index="variable", columns="trial", values="rmsd_vs_media")
                          .reset_index(), floatfmt=".1f"), ""]
        if "variabilidad_saques" in fig:
            lines += [f"![variabilidad]({fig['variabilidad_saques']})", ""]
    corr = _read(t / "correlaciones.csv")
    if not corr.empty:
        top = corr.reindex(corr["rho"].abs().sort_values(ascending=False).index).head(10)
        lines += ["## 6d. Asociaciones entre indicadores por saque",
                  "ρ de Spearman entre los indicadores de la sección 7 y las variables del centro de masa. "
                  "Con n = 5 solo |ρ| = 1 (o 0.9) llega a p < 0.05; se reporta como descripción, sin "
                  "corrección por comparaciones múltiples y sin interpretación causal.", "",
                  "Diez pares con mayor |ρ|:", "", _md(top), ""]
        if "correlaciones" in fig:
            lines += [f"![correlaciones]({fig['correlaciones']})", ""]
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
        if k.startswith(("jacobiano_", "dinamica_inversa_")) and k != "dinamica_inversa_picos":
            lines += [f"![{k}]({v})", ""]
    body, post = _read(t / "cuerpo.csv"), _read(t / "posturas.csv")
    rom, lev, idp = _read(t / "rom.csv"), _read(t / "palancas.csv"), _read(t / "picos_dinamica.csv")
    if not body.empty:
        lines += ["## 9. Cuerpo completo: centro de masa, salto y fuerza del suelo estimada",
                  "Fuerza vertical estimada F = m(a_CoM + g) (suma de ambos pies). Fin del impulso = "
                  "máxima velocidad vertical del CoM. El tiempo de vuelo no se reporta: con marcadores "
                  "2D la altura de los pies no es fiable.", "",
                  _md(body, ["vista", "saque", "altura_contacto_m", "elevacion_max_com_m",
                             "vy_max_com_m_s", "grf_pico_bw", "impulso_neto_Ns", "v_por_impulso_m_s",
                             "separacion_tobillos_inicio_m", "separacion_tobillos_carga_m",
                             "desplazamiento_x_com_m"]), ""]
        for k in ("com_fuerza_suelo", "trayectorias"):
            if k in fig:
                lines += [f"![{k}]({fig[k]})", ""]
    if not post.empty:
        th = post[post["variable"].str.startswith("theta")]
        tab = th.pivot_table(index=["vista", "variable"], columns="evento", values="valor",
                             aggfunc="mean").reset_index()
        lines += ["## 10. Ángulos en los eventos (media de los saques, °)", _md(tab, floatfmt=".0f"), ""]
        for k in ("posturas_lateral", "posturas_posterior", "plano_frontal_posterior"):
            if k in fig:
                lines += [f"![{k}]({fig[k]})", ""]
    if not rom.empty:
        r = rom.groupby(["vista", "variable"])["rom_deg"].agg(["mean", "std"]).reset_index()
        lines += ["## 11. Rango de movimiento inicio → impacto (°)", _md(r, floatfmt=".1f"), ""]
        if "rango_movimiento" in fig:
            lines += [f"![rom]({fig['rango_movimiento']})", ""]
    if not lev.empty:
        lv = lev.groupby(["evento", "articulacion"])[["brazo_palanca_m", "momento_peso_Nm"]].mean().reset_index()
        lines += ["## 12. Brazos de palanca del peso corporal (media de los saques)",
                  "Distancia horizontal de la articulación a la vertical por el CoM; momento estático "
                  "= m·g·brazo (sin fuerzas inerciales ni musculares).", "", _md(lv), ""]
        if "palancas_lateral" in fig:
            lines += [f"![palancas]({fig['palancas_lateral']})", ""]
    if not idp.empty:
        ip = idp.groupby(["parte", "variable", "unidad"])["pico_abs"].agg(["median", "min", "max"]).reset_index()
        lines += ["## 13. Picos de dinámica inversa y potencia articular (exploratorio)",
                  "Tren inferior con GRF estimada por el CoM, F = m(a_CoM + g), 50 % por pierna (pies "
                  "juntos en la carga) y centro de presión fijo a mitad del pie; picos entre el inicio "
                  "de la flexión de rodilla y el despegue (máx. velocidad vertical del CoM). Tren superior "
                  "con la muñeca libre (sin raqueta): los picos del hombro están inflados por la doble "
                  "derivada del brazo de golpeo. La potencia de cadera sale negativa en el empuje: el "
                  "tronco rota hacia atrás (arco del saque kick) contra un momento flexor estimado; "
                  "depende del supuesto del centro de presión.", "",
                  _md(ip, floatfmt=".1f"), ""]
        if "dinamica_inversa_picos" in fig:
            lines += [f"![picos]({fig['dinamica_inversa_picos']})", ""]
    filt, qc = _read(t / "filtros.csv"), _read(t / "qc_resumen.csv")
    if not filt.empty or not qc.empty:
        lines += ["## 13b. Filtrado y control de calidad",
                  "Butterworth de orden 2 sin desfase; la fc de cada marcador sale del análisis de "
                  "residuos de Winter (punto: fc elegida, donde R(fc) cruza la extrapolación del ruido).",
                  ""]
        if not filt.empty:
            lines += [_md(filt.groupby("vista")["fc_hz"].agg(["min", "median", "max"]).reset_index(),
                          floatfmt=".1f"), ""]
        if "filtrado_residuos" in fig:
            lines += [f"![residuos]({fig['filtrado_residuos']})", ""]
        if not qc.empty:
            lines += ["Muestras anuladas por cada control (suma sobre marcadores o segmentos):", "",
                      _md(qc, floatfmt=".0f"), ""]
        if "qc_resumen" in fig:
            lines += [f"![qc]({fig['qc_resumen']})", ""]
    lines += [
        "## 14. Limitaciones",
        "- 2D por vista: los ángulos son proyecciones en el plano de cada cámara.",
        "- Escala: vara de 1 m corregida por profundidad suponiendo cámara nivelada a 1.30 m (ver "
        "estatura aparente en la tabla de calibración); escala fija por vista aunque la jugadora se "
        "mueve en profundidad. Los ángulos y tiempos no dependen de la escala.",
        "- El impacto es un proxy (máxima rapidez de muñeca cerca de su altura máxima); no se rastrean "
        "raqueta ni pelota.",
        "- Dinámica inversa exploratoria: GRF estimada por el CoM (sin plataforma; reparto 50/50 "
        "entre pies y centro de presión supuestos) y muñeca sin carga externa.",
        "- MediaPipe pierde precisión en el brazo de golpeo a alta velocidad; las estampitas reducen "
        "el problema solo donde se detectan.",
    ]
    if ctx.warnings:
        lines += ["", "## Advertencias del pipeline", *[f"- {w}" for w in dict.fromkeys(ctx.warnings)]]
    path = rep / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
