# saque_kick — pipeline cinemático 2D (M1–M8, M10, M12–M14)

Alcance: **solo la jugadora** (cinemática, cadena cinética, brazos de palanca, y torques/fuerzas cuando existan GRF).
La pelota **no** se analiza (M9 descartado). M11 (ML) no está implementado.

## Uso
```bash
pip install -e .            # o: pip install numpy pandas scipy matplotlib pyyaml pytest
# 1) copia tus CSV de Kinovea a data/raw/lateral/ y data/raw/posterior/ (nombres originales)
python scripts/run_pipeline.py                    # procesa lo verificado
python scripts/run_pipeline.py --use-hypothesis   # exploratorio: aplica offsets NO verificados
pytest                                            # suite unitaria y de integración
```

## Módulos
| Módulo | Archivo | Qué hace |
|---|---|---|
| M1 | `catalog.py` | Inventario, rango temporal, hash SHA-256, excluye duplicados |
| M2 | `io_kinovea.py` | Nombres canónicos (`X_cad_cm`→`x_cadera`), ms→s, cm→m, °→rad, `v_*/a_*` → `kin_*` |
| M3 | `qc.py` | Tiempo creciente (duplicados), saltos de tracking, huecos cortos; todo a `qc_log.csv` |
| M4 | `sync.py` | Offsets por archivo, xcorr, intervalo común, spline a malla uniforme (sin extrapolar) |
| M5 | `filtering.py` | Butterworth fase cero, corte por residuos (Winter), Savitzky-Golay, derivadas |
| M6 | `kinematics.py` | θ/ω/α de segmentos y articulaciones, velocidad lineal, comparación con Kinovea |
| M7 | `events.py` | Eventos, fases y secuencia próximo-distal |
| M8 | `jacobian.py` | Estimación/validación de velocidad de mano mediante `v = J(θ)ω` |
| M10 | `stats.py` | Descriptivos por fase, curvas normalizadas y variabilidad (exploratorio) |
| M12 | `inverse_dynamics.py` | Newton-Euler 2D; requiere GRF y reloj compartido verificado |
| M13 | `viz.py` | Figuras diagnósticas, cinemáticas, fases y Jacobiano |
| M14 | `report.py` | Reporte Markdown, registro de decisiones y validación |

## Salidas
`data/interim/synchronized/*_merged.csv` · `data/processed/{filtered,kinematics,jacobian,inverse_dynamics}/` ·
`reports/tables/{catalog,qc,synchronization,filtering,jacobian,events,statistics,inverse_dynamics,report}/` ·
`reports/figures/` · `reports/report.md`

## Antes de usar con datos reales
1. `flip_y` (config): Kinovea suele exportar y hacia abajo. Si tus ejes están calibrados hacia arriba, ponlo en `false`.
2. `coord_unit_overrides`: la cinemática lineal lateral no trae `_cm`; se asumió cm. **Confírmalo.**
3. `sync_offsets.yaml`: `offset_s: null` = pendiente. Esos grupos se omiten hasta verificar con el video.
4. `duplicate_time_policy`: define qué hacer con el 1.535 s repetido (por defecto, conserva la primera fila).
5. Convención de ángulo articular: `compare_with_kinovea` dice si tu ángulo coincide con el directo o el suplementario.
