"""M14: consolida tablas, limitaciones y decisiones en un reporte reproducible."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def run_report(out_dir=".", *, warnings=None) -> dict:
    """Escribe el resumen Markdown, el registro de decisiones y métricas de validación."""
    out = Path(out_dir)
    tables = out / "reports/tables"
    reports = out / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    warnings = list(warnings or [])

    events = _read_optional(tables / "events/eventos.csv")
    phases = _read_optional(tables / "events/fases.csv")
    stats = _read_optional(tables / "statistics/descriptivos_por_fase.csv")
    jacobian = _read_optional(tables / "jacobian/validacion.csv")
    inverse_status = _read_optional(tables / "inverse_dynamics/status.csv")
    sync = _read_optional(tables / "synchronization/decisiones_sync.csv")
    qc = _read_optional(tables / "qc/qc_log.csv")
    filters = _read_optional(tables / "filtering/filtros.csv")

    validation_rows = []
    for _, row in jacobian.iterrows():
        for key in ("rmse_speed", "rmse_rel_pico", "corr_speed", "cv_brazo", "cv_antebrazo"):
            if key in row and pd.notna(row[key]):
                validation_rows.append({"procedimiento": "M8_jacobiano", "metrica": key,
                                        "valor": row[key], "grupo": row.get("grupo", "")})
    for _, row in inverse_status.iterrows():
        validation_rows.append({"procedimiento": "M12_dinamica_inversa",
                                "metrica": row.get("status", "estado"),
                                "valor": row.get("message", "")})
    validation = pd.DataFrame(validation_rows, columns=["procedimiento", "metrica", "valor", "grupo"])
    report_tables = tables / "report"
    report_tables.mkdir(parents=True, exist_ok=True)
    validation.to_csv(report_tables / "metricas_validacion.csv", index=False)

    decisions = ["# Registro de decisiones y control de calidad", ""]
    if sync.empty:
        decisions.append("- No hay tabla de decisiones de sincronización.")
    else:
        for _, row in sync.iterrows():
            decisions.append(
                f"- **{row.get('grupo', '')} / {row.get('archivo', '')}:** "
                f"offset={row.get('offset_s', '')} s; estado={row.get('estado', '')}."
            )
    decisions += ["", f"- Eventos de QC registrados: {len(qc)}.",
                  f"- Decisiones de filtro registradas: {len(filters)}."]
    if warnings:
        decisions += ["", "## Advertencias", *[f"- {warning}" for warning in warnings]]
    (reports / "log_decisiones.md").write_text("\n".join(decisions) + "\n", encoding="utf-8")

    summary = [
        "# Reporte del análisis del saque",
        "",
        "Reporte generado automáticamente por el pipeline. Las métricas se interpretan "
        "considerando las limitaciones de sincronización, calibración y tamaño muestral.",
        "",
        "## Resultados disponibles",
        f"- Eventos detectados (M7): {len(events)}.",
        f"- Fases segmentadas (M7): {len(phases)}.",
        f"- Resúmenes por fase/intento (M10): {len(stats)}.",
        f"- Métricas de validación consolidadas (M14): {len(validation)}.",
    ]
    if inverse_status.empty:
        summary.append("- Dinámica inversa (M12): no se ejecutó; no hay estado de M12 registrado.")
    else:
        for _, row in inverse_status.iterrows():
            summary.append(f"- Dinámica inversa (M12): **{row.get('status', 'sin estado')}** — "
                           f"{row.get('message', '')}")
    summary += [
        "",
        "## Tablas y figuras",
        "- Eventos y fases: `tables/events/`.",
        "- Estadística descriptiva: `tables/statistics/`.",
        "- Validación del Jacobiano: `tables/jacobian/`.",
        "- Control de calidad y sincronización: `tables/qc/` y `tables/synchronization/`.",
        "- Métricas consolidadas: `tables/report/metricas_validacion.csv`.",
        "- Figuras: `figures/`.",
        "- Decisiones y advertencias: `log_decisiones.md`.",
        "",
        "## Interpretación",
        "Las asociaciones y comparaciones entre intentos son exploratorias. No se "
        "atribuyen causalidad ni riesgo de lesión a partir de cinemática 2D. Las fuerzas "
        "y torques solo se reportan si hay GRF sincronizada y parámetros antropométricos "
        "verificados para el participante.",
    ]
    if warnings:
        summary += ["", "## Advertencias del pipeline", *[f"- {warning}" for warning in warnings]]
    report_path = reports / "report.md"
    report_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    return {"report": report_path, "validation": validation,
            "decision_log": reports / "log_decisiones.md"}
