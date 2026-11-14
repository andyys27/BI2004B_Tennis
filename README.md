# Análisis biomecánico 2D del saque kick de tenis (`saque`)

Pipeline reproducible en Python que convierte las exportaciones de **Kinovea** (video a 120 fps, vistas lateral y posterior) de tenistas del equipo representativo del Tec en series cinemáticas calibradas, sincronizadas y documentadas: ángulos, velocidades y aceleraciones articulares, eventos y fases del saque, secuencia próximo-distal, velocidad de la mano con el Jacobiano, estadística descriptiva, y una estimación **exploratoria** de fuerzas y momentos articulares por dinámica inversa 2D.

> **Alcance decidido:** el análisis es **solo de la jugadora**. La pelota (trayectoria, velocidad de salida, spin) está **fuera de alcance** y el módulo M9 fue descartado.

---

## 0. Estado del proyecto en una página

| Pregunta | Respuesta honesta |
|---|---|
| ¿El código corre? | Sí. Con datos sintéticos genera tablas y 11 figuras de punta a punta. |
| ¿Hay resultados científicos del saque? | **No todavía.** El repositorio no contiene los CSV de Kinovea ni salidas generadas con ellos. Las secciones de resultados reales (§9.2) son una plantilla por llenar. |
| ¿Están verificados los cálculos? | Los módulos M1–M8 y la dinámica inversa (M12) sí, contra soluciones analíticas y una implementación independiente (§9.1). |
| ¿Se pueden interpretar los torques? | **No como cargas reales del saque.** Faltan fuerzas externas y el modelo actual asume suelo ausente (inferior) y muñeca libre (superior). Ver §10, C1–C3. |
| ¿Se puede medir variabilidad entre intentos? | **No con el código actual:** no existe el concepto de sujeto/intento (§10, C4). |
| ¿Qué lo bloquea? | Offsets de sincronización sin verificar, definición de intentos, plano de análisis correcto por variable, y datos cinéticos. |

**Suite de pruebas:** 47 tests originales → 46 pasan, 1 falla por una diferencia de redacción (§10, M1). Se añadieron 4 tests independientes de dinámica inversa (todos pasan).

---

## 1. Contexto y objetivos

El saque kick depende de la coordinación entre extremidades inferiores, tronco y brazo de golpeo. La motivación a largo plazo es relacionar técnica, rendimiento y cargas biomecánicas para, a futuro, apoyar prevención de lesiones. El análisis de video 2D **no mide fuerzas internas ni demuestra causalidad**; es una primera herramienta de medición.

### Objetivos específicos y su cobertura

| # | Objetivo | Módulos | Estado |
|---|---|---|---|
| 1 | Digitalizar puntos anatómicos (y raqueta/pelota) | Kinovea (manual) | Parcial: no hay raqueta, tronco ni ángulo de hombro/cadera. Pelota descartada. |
| 2 | θ(t), ω(t), α(t) articulares | M2–M6 | Implementado; depende de sincronía y convenciones |
| 3 | Fases, timing y secuencia de coordinación | M7 | Implementado; cadena incompleta (faltan tronco y hombro) |
| 4 | Posturas clave, **brazos de palanca** y torques cualitativos | `levers.py` | **No integrado al pipeline** (solo funciones sueltas) |
| 5 | Velocidad de la pelota | — | **Descartado** |
| 6 | Velocidad de la mano con `v = J(θ)ω` | M8 | Implementado y validado |
| 7 | Asociaciones y consistencia entre intentos | M10 | **No operativo:** un solo intento por serie |
| 8 | Comparación con literatura | — | Pendiente (manual) |
| 9 | Camino hacia fuerzas y dinámica inversa | M12 | Implementado con supuestos fuertes; sin GRF |

### Hipótesis de trabajo

H0/H1 del contexto son hipótesis **direccionales de trabajo**, no hipótesis estadísticas nula/alternativa. Antes de inferir hay que formalizarlas (p. ej. H0 estadística: *no hay asociación entre el índice de secuencia próximo-distal y la velocidad pico de muñeca*) y definir el diseño muestral.

---

## 2. Datos de entrada

Archivos CSV de Kinovea (no son bases relacionales: una fila = una muestra temporal; sin identificadores de fotograma, sujeto ni ensayo). Ángulos en **grados**; sufijo `_cm` = centímetros; `ABS` = ángulo absoluto; `MA` = promedio móvil. Detalle completo en `docs/contexto_bases_datos.md`.

| Vista | Archivo | Filas | Contenido |
|---|---|---:|---|
| Lateral | `cinematica_lineal_lateral.csv` | 544 | x/y de cadera, rodilla, tobillo + v, a (**sin sufijo de unidad**) |
| Lateral | `cinematica_angular_lateral_rodilla.csv` / `PI1_..._rodilla.csv` | 537 | ángulo de rodilla (0–4.461 s) |
| Lateral | `cinematica_angular_lateral_tobillo.csv` / `PI1_..._tobillo.csv` | 138 | ángulo de tobillo (**desfasados 0.100 s**) |
| Posterior | `cinematica_angular_rodilla.csv` | 282 | rodilla (0.593–2.931 s) |
| Posterior | `cinematica_angular_tobillo.csv` | 282 | tobillo (0.000–2.340 s) |
| Posterior | `cinematica_angular_brazo.csv` | 282 | codo (0.593–2.931 s) |
| Posterior | `cinematica_posterior_PD.xlsx.csv` | 348 | pierna derecha: punta, tobillo, rodilla, cadera (cm) |
| Posterior | `cinematica_lineal_brazo_pos.csv` | 356 | muñeca, codo, hombro (cm) |
| Posterior | `cinematica_lineal_brazo_v_a.csv` | 356 | v, a del brazo (solo `tiempo_ms`; **no se usa**) |

### Problemas conocidos en los datos (ya tratados por el pipeline)

- **Muestreo no uniforme:** intervalos que alternan ≈ 0.008/0.009 s (≈ 120 Hz nominal). Se leen los tiempos reales, nunca `1/120`.
- **Tiempo duplicado:** en la cinemática lineal lateral las filas 185 y 186 tienen `1.535 s`.
- **Offsets temporales:** tobillo posterior vs. rodilla/codo (hipótesis ≈ 0.592 s, pues las duraciones coinciden: 2.340 vs. 2.338 s) y tobillo lateral (0.100 s).
- **Duplicado exacto** `posterior/cinematica_angular_rodilla.csv` (excluido por hash).
- **Encabezados irregulares:** `X_cad_cm` (mayúscula), `Tiempo (ms)`, `Angulo_Rod`.
- **Faltantes:** no hay fuerzas de plataforma, raqueta, tronco, ángulo de hombro ni de cadera.

---

## 3. Estructura del repositorio

```
.
├── config/
│   ├── pipeline.yaml            # toda la configuración (grupos, filtros, eventos, M10, M12)
│   └── sync_offsets.yaml        # offsets por archivo (verificado vs. hipótesis)
├── docs/                        # contextos del proyecto y documento maestro
├── scripts/run_pipeline.py      # orquesta M1→M14
├── src/saque/
│   ├── catalog.py      [M1]     # inventario, hash, duplicados
│   ├── io_kinovea.py   [M2]     # lectura y estandarización
│   ├── qc.py           [M3]     # control de calidad del tracking
│   ├── sync.py         [M4]     # offsets, xcorr, malla uniforme
│   ├── filtering.py    [M5]     # Butterworth, Winter, derivadas
│   ├── kinematics.py   [M6]     # θ, ω, α, velocidad lineal
│   ├── events.py       [M7]     # eventos, fases, secuencia
│   ├── jacobian.py     [M8]     # v = v_hombro + J(θ)ω
│   ├── stats.py        [M10]    # descriptivos, variabilidad, Spearman bootstrap
│   ├── inverse_dynamics.py [M12]# Newton-Euler 2D (inferior y superior)
│   ├── viz.py          [M13]    # figuras
│   ├── report.py       [M14]    # reporte y log de decisiones
│   ├── levers.py                # brazos de momento (no integrado)
│   ├── pipeline.py              # run() y run_events()
│   └── config.py
└── tests/                       # synth.py (datos sintéticos) + tests
```

> **M9 (pelota): descartado. M11 (ML exploratorio): no implementado** (opcional, §11).

Salidas generadas (carpetas creadas al correr):

```
data/interim/synchronized/    series remuestreadas a malla común
data/processed/{filtered,kinematics,jacobian,inverse_dynamics}/
reports/tables/{catalog,qc,synchronization,filtering,events,statistics,jacobian,inverse_dynamics,report}/
reports/figures/...           reports/report.md   reports/log_decisiones.md
```

---

## 4. Instalación y uso

```bash
pip install numpy pandas scipy matplotlib pyyaml pytest
# Copia tus CSV originales (sin editar) a data/raw/lateral/ y data/raw/posterior/

python scripts/run_pipeline.py                     # solo lo verificado
python scripts/run_pipeline.py --use-hypothesis    # exploratorio: aplica offsets NO verificados
pytest                                             # suite completa (datos sintéticos)
```

La corrida con `--use-hypothesis` deja todo offset no verificado marcado como `HIPOTESIS_NO_VERIFICADA` en `reports/tables/synchronization/decisiones_sync.csv`. Los grupos con offset pendiente se **omiten** si no se pasa esa bandera.

---

## 5. Flujo de datos

```
raw CSV → M1 catálogo → M2 estandarizar → M3 QC → M4 sync+remuestreo → M5 filtrar → M6 cinemática
                                                                                   │
                         ┌──────────────────────────┬──────────────────────────────┤
                         ▼                          ▼                              ▼
                  M7 eventos/fases           M8 Jacobiano                  M12 dinámica inversa*
                         └────────────┬─────────────┘                       (*supuestos fuertes)
                                      ▼
                           M10 estadística → M13 figuras → M14 reporte
```

Principios de diseño: (1) `data/raw` es de solo lectura; (2) **nunca se une por índice de fila**, solo por tiempo; (3) todo dato excluido o modificado queda en un log; (4) lo no verificado se **marca**, no se oculta.

---

## 6. Módulos

### M1 · `catalog.py` — Inventario
Recorre `data/raw`, calcula SHA-256, rango temporal, número de filas y NaN. Marca `duplicado_de` y `usar`. **Salida:** `catalog/catalogo.csv`. Los duplicados se excluyen del procesamiento y quedan en el log de QC.

### M2 · `io_kinovea.py` — Estandarización
- Tiempo → `time_s` (ms ÷ 1000); se conserva `time_s_raw`.
- Coordenadas `x_<marcador>`, `y_<marcador>` con alias (`cad→cadera`, `rod→rodilla`, `tob→tobillo`, `mun→muneca`, `homb→hombro`); `_cm → m`.
- Sin sufijo de unidad: unidad `raw` salvo `coord_unit_overrides` (hoy se asume **cm** para la lateral lineal; **debe confirmarse**).
- Ángulos: se conserva `angulo_*` (°) y se crea `angulo_*_rad`.
- `v_*`, `a_*` de Kinovea → `kin_v_*`, `kin_a_*`: **no se usan** (filtro y unidades no documentados); se recalculan.

### M3 · `qc.py` — Calidad del tracking
1. Tiempo estrictamente creciente. Duplicados según `duplicate_time_policy` (`first` | `last` | `mean`).
2. **Saltos de etiqueta aislados:** residual contra mediana móvil (ventana 5) con doble criterio: $r > z\,\hat\sigma$ y $r > 0.02\cdot\text{rango}$, con $\hat\sigma$ estimado por segundas diferencias:
$$\hat\sigma=\frac{1.4826\,\operatorname{med}|\Delta^2x-\operatorname{med}(\Delta^2x)|}{\sqrt6}$$
3. Interpolación lineal solo de huecos cortos (≤ `max_gap_s`); los largos quedan NaN y se registran.

**Salida:** `qc/qc_log.csv` (archivo, índice, tipo, acción) y `qc/time_report.csv`.

### M4 · `sync.py` — Sincronización y remuestreo
Convención: $t_{global}=t_{raw}+\text{offset\_s}$.
- `estimate_offset`: correlación cruzada entre señales que miden lo mismo (**evidencia, no verdad**).
- `common_interval`: intersección temporal (error si no existe).
- `to_uniform_grid`: spline cúbico a `resample_hz` **sin extrapolar**.
- `merge_on_grid`: une series de un grupo; columnas repetidas se renombran `col__archivo`.

Los offsets se guardan en `sync_offsets.yaml`: `offset_s` (verificado con video) o `hypothesis_s` (no verificado).

### M5 · `filtering.py` — Suavizado y derivadas
Butterworth pasa-bajas de **fase cero** (`filtfilt`, orden efectivo 4). Elección del corte por **análisis de residuos de Winter**:
$$R(f_c)=\sqrt{\tfrac1N\sum_i\big(x_i-\hat x_i(f_c)\big)^2}$$
Se ajusta una recta a la cola de $R(f_c)$ (fc altas, dominada por ruido), se extrapola a $f_c=0$ y la fc óptima es donde la curva cruza ese valor. Rango por defecto 4–20 Hz; si no es aplicable se usa `fallback_cutoff_hz`. Derivadas por diferencias centrales:
$$\dot x_i=\frac{x_{i+1}-x_{i-1}}{2\Delta t},\qquad \ddot x_i=\frac{x_{i+1}-2x_i+x_{i-1}}{\Delta t^2}$$
Rechaza NaN, fc fuera de Nyquist y mallas no uniformes. Cada decisión queda en `filtering/filtros.csv` (método: `winter`, `winter_limite_inferior`, `fijo`, `fallback`).

### M6 · `kinematics.py` — Cinemática
- **Ángulo absoluto** del vector proximal→distal (y hacia arriba; `flip_y` convierte desde y-abajo), desenvuelto con `np.unwrap`:
$$\theta=\operatorname{atan2}(y_d-y_p,\;x_d-x_p)$$
- **Ángulo interior** en $[0,\pi]$: $\operatorname{atan2}(|u\times v|,\,u\cdot v)$, y **ángulo con signo** (flexión/extensión).
- $\omega=\dot\theta$, $\alpha=\ddot\theta$ en rad, rad/s, rad/s².
- Velocidad y aceleración lineal por marcador: `vx, vy, speed, ax, ay`.
- `compare_with_kinovea` (RMSE vs. el ángulo exportado, directo o suplementario) está **implementado pero no se invoca en el pipeline** (§10, I8).

### M7 · `events.py` — Eventos, fases y secuencia
Eventos (todos reproducibles por algoritmo):

| Evento | Criterio |
|---|---|
| `inicio` | rapidez de muñeca > 5 % del pico, sostenida 3 muestras |
| `inicio_flexion_rodilla` | último instante antes de la máx. flexión con ángulo ≥ 95 % de la postura erguida |
| `max_flex_rodilla` | mínimo del ángulo interior de rodilla desde el inicio del registro hasta el impacto |
| `pico_omega_codo` | máx. ω de extensión de codo |
| `max_flex_codo` | mínimo del ángulo interior de codo previo al pico de ω |
| `impacto` | **proxy:** máx. rapidez de muñeca (o fotograma manual en `events.manual.impacto`) |
| `fin` | rapidez < 20 % del pico tras el impacto |

Fases: **carga** (`inicio_flexion_rodilla → max_flex_rodilla`), **aceleración** (`max_flex_rodilla → impacto`), **seguimiento** (`impacto → fin`).

**Secuencia próximo-distal:** tiempo de pico de cada eslabón (muslo → brazo → codo → muñeca), lags en ms y $\tau$ de Kendall frente al orden esperado ($\tau=1$ orden perfecto). **No** se calculan razones de amplitud (unidades distintas). Si faltan eslabones, la tabla dice `cadena incompleta`. Avisos automáticos: orden inválido, pico en el borde del registro, reloj compartido no verificado.

### M8 · `jacobian.py` — Velocidad de la mano
Brazo planar de 2 eslabones, ángulos absolutos $\theta_1$ (brazo) y $\theta_2$ (antebrazo):
$$\mathbf v=\mathbf v_{hombro}+\underbrace{\begin{bmatrix}-L_1\sin\theta_1&-L_2\sin\theta_2\\ L_1\cos\theta_1&L_2\cos\theta_2\end{bmatrix}}_{J(\boldsymbol\theta)}\begin{bmatrix}\omega_1\\ \omega_2\end{bmatrix}$$
`L₁`, `L₂` = mediana de la distancia hombro–codo y codo–muñeca. Reporta RMSE y correlación frente a la derivada directa, **CV de longitudes** (rigidez/perspectiva) y la **fracción** que aporta cada articulación a la rapidez de la mano.

> **Lectura honesta:** `J·ω` vs. derivada directa coinciden casi por construcción (las longitudes salen de los mismos marcadores). Lo informativo es el **CV de longitud** (`longitud_rigida = False` ⇒ tracking inestable o movimiento fuera del plano) y la **descomposición hombro/codo**.

### M10 · `stats.py` — Estadística
`describe_series` (media, DE, ROM, pico firmado/absoluto, tiempo del pico, por fase), `normalize_trials` (0–100 %, 101 puntos, sin extrapolar), `intertrial_variability` (banda media ± DE, CV %, RMSD por intento) y `bootstrap_spearman` (ρ con IC percentil; exige ≥ 3 pares no constantes).
**Hoy** `run_statistics` trata cada serie como un único intento (`trial="intento_disponible"`), por lo que la variabilidad inter-intento **no es estimable** (§10, C4).

### M12 · `inverse_dynamics.py` — Dinámica inversa 2D (Newton-Euler)
Resuelve de **distal a proximal**. Por segmento, con $\mathbf e=(\cos\theta,\sin\theta)$ del vector proximal→distal, aceleración de un punto a distancia $s$ del extremo proximal:
$$\mathbf a_{P+s}=\mathbf a_P+s\,(\alpha\,\mathbf e_\perp-\omega^2\mathbf e)$$
Newton y Euler sobre el centro de masa:
$$\sum\mathbf F=m(\mathbf a_{CoM}-\mathbf g),\qquad \sum M_{CoM}=I\alpha,\quad I=m\,(\rho\,l)^2$$
Fuerzas/momentos devueltos = los que la articulación **proximal ejerce sobre el segmento**, momento antihorario positivo.

**Dos variantes:**
1. **Miembro inferior** (pie–pierna–muslo → tobillo, rodilla, cadera). Modos: `measured_grf` (requiere GRF, COP y reloj verificado) y `assume_flight_no_grf` (GRF = 0 en todo el registro).
2. **Miembro superior** (brazo–antebrazo → codo, hombro) con **fuerza nula en muñeca**; mano, raqueta e impacto excluidos.

Antropometría de Winter para $M=58$ kg, $H=1.64$ m: pie 1.45 %, pierna 4.65 %, muslo 10 % de la masa; brazo 2.71 %, antebrazo 1.62 %; CoM y radios de giro según tablas de Winter.

### M13 · `viz.py` — Figuras
Cobertura temporal por archivo, histograma de muestreo, θ/ω/α por grupo, fases y eventos, validación del Jacobiano, contribuciones articulares, trayectoria de muñeca coloreada por rapidez, y fuerzas/momentos de dinámica inversa (el título incluye el supuesto de fuerza externa).

### M14 · `report.py` — Reporte
Escribe `report.md`, `log_decisiones.md` y `report/metricas_validacion.csv`, e incorpora advertencias. **Hoy solo cuenta resultados; no incluye valores numéricos** (§10, I6).

### `levers.py` — Brazos de palanca
`moment_arm`, `joint_moment`, `horizontal_offset_to_vertical_line`. Calculan $\tau=\mathbf r\times\mathbf F=r_xF_y-r_yF_x$. **No están conectadas al pipeline.**

---

## 7. Configuración (`config/pipeline.yaml`)

| Bloque | Qué controla |
|---|---|
| `resample_hz`, `flip_y` | malla común; conversión de y-abajo a y-arriba |
| `filter` | tipo (`butterworth`/`savgol`), orden, `cutoff_hz` (null = Winter), rango 4–20 Hz, fallback 10 Hz |
| `qc` | política de duplicados, umbrales de saltos, hueco máximo |
| `coord_unit_overrides` | unidad de coordenadas sin sufijo |
| `groups` | archivos a unir, `segments`, `joints`, `linear`, `jacobian`, `angle_cols` |
| `events` | series, impacto manual, secuencia y **`shared_clock_verified`** |
| `statistics` | series a resumir |
| `inverse_dynamics` / `upper_limb_inverse_dynamics` | habilitación, modo de fuerza, antropometría |

**Atención:** `inverse_dynamics.enabled` y `upper_limb_inverse_dynamics.enabled` vienen en `true`. Ver §10, C1 y C3.

---

## 8. Salidas y cómo leerlas

| Archivo | Qué contiene | Cómo interpretarlo |
|---|---|---|
| `qc/qc_log.csv` | cada dato excluido/modificado | Muchos eventos en un marcador = tracking pobre |
| `qc/time_report.csv` | n, t0, t1, Δt, fs efectiva | fs ≈ 120 Hz esperada; Δt irregular es normal |
| `synchronization/decisiones_sync.csv` | offsets aplicados y estado | Cualquier `HIPOTESIS_NO_VERIFICADA` ⇒ resultado exploratorio |
| `filtering/filtros.csv` | fc por marcador/serie y método | Muchos `winter_limite_inferior` o fc = 20 Hz ⇒ rango mal elegido |
| `kinematics/*_kinematics.csv` | θ, ω, α, v, a | θ en rad; ω pico ≈ momento de mayor velocidad angular |
| `events/eventos.csv`, `fases.csv` | tiempos de eventos y duración de fases | Impacto = proxy salvo `manual` |
| `events/secuencia.csv` | lags (ms) y τ | τ→1 y lags positivos ⇒ consistente con cadena próximo-distal |
| `jacobian/validacion.csv` | L, CV, RMSE, correlación | `longitud_rigida=False` ⇒ desconfiar del plano |
| `jacobian/*_jacobian.csv` | `frac_hombro`, `frac_codo` | Quién domina la velocidad de la mano en cada instante |
| `statistics/*.csv` | descriptivos por fase, curvas normalizadas | Con un intento: solo descriptivo |
| `inverse_dynamics/*.csv` | F y T articulares | **Solo bajo los supuestos declarados** (§10) |
| `report/metricas_validacion.csv` | métricas consolidadas | Revisar antes de citar cualquier resultado |

**Guía de lectura rápida:** θ(t) = postura y ROM; ω(t) = cuándo y cuán rápido gira; α(t) = cuándo se aplica momento neto (la señal más sensible al filtro); el cambio de signo de α marca el pico de ω. El torque articular de una fuerza vale $\tau=r_\perp F$: sin fuerza medida solo se discute **cómo cambia el brazo de palanca**, no la magnitud del torque.

---

## 9. Resultados

### 9.1 Resultados técnicos verificados (reproducibles)

**Pruebas automáticas.** 47 tests originales: 46 ✔, 1 ✘ (redacción del reporte, §10 M1). 4 tests nuevos de dinámica inversa independiente (`tests/test_inverse_dynamics_independent.py`): 4 ✔.

**Validación independiente de la dinámica inversa** (misma cadena, pero calculada por posiciones: CoM derivado numéricamente y Newton-Euler con vectores explícitos):

| Variable (miembro inferior) | Error relativo máx. vs. referencia |
|---|---:|
| Fx/Fz/T tobillo | 2.4 × 10⁻⁶ / 1.3 × 10⁻⁷ / 1.2 × 10⁻⁷ |
| Fx/Fz/T rodilla | 2.0 × 10⁻⁶ / 2.8 × 10⁻⁷ / 2.6 × 10⁻⁷ |
| Fx/Fz/T cadera | 2.3 × 10⁻⁶ / 1.0 × 10⁻⁶ / 3.5 × 10⁻⁷ |

Miembro superior: diferencias ≤ 2.2 × 10⁻³ N (picos de 20–44 N) y ≤ 7 × 10⁻⁴ N·m. Equilibrio estático con el COP en el centro de masa ponderado: momentos ≈ 10⁻¹⁵ N·m. Caída libre: fuerzas y momentos = 0. **Conclusión: las ecuaciones de M12 son correctas;** la limitación está en los supuestos y en los datos, no en el álgebra.

**Demostración sintética de punta a punta** (brazo de 2 eslabones con verdad conocida, ruido de 1.5 mm, muestreo irregular 8/9 ms): longitudes recuperadas 0.2999 m y 0.2800 m (verdad 0.30 y 0.28), CV 0.19 % y 0.26 %, correlación `J·ω` vs. directa 0.9993, RMSE 0.77 % del pico. ⚠ **Son números de datos sintéticos; no son resultados de las jugadoras.**

### 9.2 Resultados científicos — plantilla por llenar con datos reales

| Resultado | Tabla/figura fuente | Valor | Estado |
|---|---|---|---|
| fs efectiva y duración por archivo | `qc/time_report.csv` | — | pendiente |
| Offsets verificados con el video | `sync_offsets.yaml` | — | **pendiente** |
| fc de filtrado por marcador | `filtering/filtros.csv` | — | pendiente |
| Tiempos de eventos E1–E6 | `events/eventos.csv` | — | pendiente |
| Duración de fases (ms) | `events/fases.csv` | — | pendiente |
| Pico de ω de rodilla, codo (rad/s) y rapidez de muñeca (m/s) | `statistics/descriptivos_por_fase.csv` | — | pendiente |
| Lags próximo-distales y τ | `events/secuencia.csv` | — | pendiente (cadena incompleta) |
| CV de longitudes de brazo/antebrazo | `jacobian/validacion.csv` | — | pendiente |
| Fracción hombro/codo en la velocidad de mano | `jacobian/*_jacobian.csv` | — | pendiente |
| Variabilidad entre intentos (CV, RMSD) | `statistics/variabilidad_interintento.csv` | — | **no estimable hoy** |
| Momentos y fuerzas articulares | `inverse_dynamics/` | — | **no interpretables hoy** |

### 9.3 Lo que estos datos NO permiten concluir

Causalidad, riesgo de lesión, cargas tisulares, eficiencia del efecto (spin), ni transferencia próximo-distal "completa" (faltan tronco y hombro). Con ~5 intentos y número de jugadoras sin definir, toda asociación es **exploratoria**.

---

## 10. Revisión: errores y puntos de mejora

Cada punto indica **evidencia** (cómo se comprobó), **impacto** y **acción**. Severidad: **C** = invalida la interpretación si no se corrige; **I** = afecta la calidad de los resultados; **M** = menor/técnico.

### Críticos

**C1 · Dinámica inversa inferior sin GRF, activada por defecto.**
*Evidencia:* `config/pipeline.yaml` → `inverse_dynamics: enabled: true, force_mode: assume_flight_no_grf`. En ese modo $GRF=0$ en **todo** el registro. *Impacto:* durante el apoyo (carga e impulso del saque) las reacciones del suelo dominan las cargas; los momentos y fuerzas resultantes no representan nada físico en esa fase y pueden confundirse en una presentación. *Acción:* poner `enabled: false` por defecto; o restringir el cálculo a una ventana aérea definida por un evento de despegue (*foot-off*) marcado en video; o conseguir GRF y usar `measured_grf` con reloj verificado.

**C2 · Plano de análisis incorrecto para variables sagitales.**
*Evidencia:* `events.series.knee_angle` y `inverse_dynamics.kinematics_group` usan `posterior_pierna`. La vista posterior es el plano **frontal**; la flexión de rodilla y el modelo de dinámica inversa documentado son **sagitales** (X–Z). *Impacto:* el "ángulo de rodilla" posterior es una proyección que subestima/distorsiona la flexión, y el segmento *pie* (tobillo→punta) puede estar casi alineado con el eje de la cámara, lo que hace inestable su ángulo (el $\operatorname{atan2}$ de un vector proyectado muy corto amplifica ruido en ω y α). *Acción:* tomar flexión de rodilla/tobillo/cadera de la **vista lateral**; usar la posterior solo para variables frontales; verificar la longitud proyectada del vector del pie (si es pequeña, descartar ese segmento). Requiere resolver la sincronía entre vistas (C5).

**C3 · Dinámica inversa del brazo omite mano, raqueta e impacto.**
*Evidencia:* `force_mode: assume_zero_wrist_force`; la cadena termina en la muñeca. En una prueba sintética, añadir apenas 0.35 kg (mano + raqueta, **sin** impacto) en la muñeca aportó un término de pico 7.8 N·m frente a 14.1 N·m calculados en el hombro (≈ 55 %). *Impacto:* los momentos de hombro y codo estarían fuertemente subestimados; además, el movimiento del saque es 3D (rotación interna del hombro, pronación) y un modelo plano captura solo una proyección. *Acción:* tratar T_hombro/T_codo como **cotas inferiores exploratorias**, o no reportarlos; para reportarlos se requiere masa/inercia de mano+raqueta, punto de impacto y un modelo 3D.

**C4 · No existe el concepto de sujeto/intento.**
*Evidencia:* `stats.py` fija `trial="intento_disponible"`; no hay `trial_id`/`subject_id` en el código; los grupos se definen por ruta de archivo. *Impacto:* el objetivo 7 (consistencia entre cinco intentos) y la banda media ± DE no se pueden calcular; para analizar 5 saques habría que correr el pipeline 5 veces a mano. *Acción:* estructurar `data/raw/<sujeto>/<intento>/{lateral,posterior}/`, añadir `subject_id`, `trial_id`, `session` al catálogo y a todas las salidas, y alimentar `normalize_trials` con un intento por clave. Confirmar si los cinco saques son por jugadora o en total.

**C5 · Sincronía no verificada y resultados que dependen de ella.**
*Evidencia:* `sync_offsets.yaml` tiene `offset_s: null` (solo hipótesis 0.592 s y −0.100 s); `events.shared_clock_verified: false`. M7 mezcla series de `posterior_brazo` y `posterior_pierna`; M10 aplica fases del brazo a series de pierna; M12 exige reloj común con la GRF. *Impacto:* lags de la secuencia y fases entre grupos pueden estar desplazados por cientos de ms. *Acción:* elegir un **evento común visible** (contacto del pie, impacto) y registrar su número de fotograma en cada video; calcular offsets desde ahí; confirmar con `estimate_offset`; solo entonces poner `shared_clock_verified: true`.

### Importantes

**I1 · Longitudes inconsistentes entre modelos.**
*Evidencia:* el miembro superior usa longitudes **medidas** de los marcadores; el inferior usa longitudes **de Winter** (pierna 0.403 m, muslo 0.402 m, pie 0.250 m). En el inferior, la aceleración de la rodilla se obtiene de la longitud paramétrica, no del marcador de rodilla. *Acción:* usar longitudes medidas por sujeto (mediana) en ambos, y añadir una comprobación de consistencia: $\mathbf a_{rodilla}^{modelo}$ vs. $\mathbf a_{rodilla}^{marcador}$.

**I2 · Antropometría genérica.**
Los parámetros de Winter provienen de datos de cadáveres (mayormente masculinos) y se aplican a una atleta de 58 kg y 1.64 m; la longitud del pie (0.25 m) es un supuesto. *Acción:* parámetros por jugadora en un `subject.yaml` y evaluar tablas por sexo (p. ej. De Leva, 1996); reportar sensibilidad de los momentos a ±10 % en masa y longitud.

**I3 · Ruido amplificado y filtro sin análisis de sensibilidad.**
*Evidencia:* el análisis de sensibilidad ±2 Hz está en el diseño pero **no está implementado**. Con error de digitalización de 1, 2 y 3 mm a 120 Hz, el ruido sin filtrar es ≈ 0.08/0.17/0.25 m/s en velocidad y ≈ 35/71/106 m/s² en aceleración ($\sigma_v=\sigma/(\sqrt2\Delta t)$, $\sigma_a=\sqrt6\,\sigma/\Delta t^2$). Además, el rango de Winter 4–20 Hz puede suavizar picos reales de la muñeca y el transitorio del impacto (en la corrida sintética varios marcadores terminaron en el límite inferior de 4 Hz). *Acción:* repetir M5–M8 con fc ± 2 Hz y reportar la variación de los picos; si un pico de α cambia > ~15 %, interpretarlo solo cualitativamente; revisar `filtros.csv` y ampliar el rango si hay saturación en los bordes.

**I4 · Efectos de borde.**
*Evidencia:* en la figura de momentos del brazo hay transitorios claros en los primeros y últimos ~0.1–0.2 s (relleno de `filtfilt` y `np.gradient`); la detección de eventos ya avisó "pico en el borde del registro". *Impacto:* picos y estadísticas pueden quedar sobre artefactos. *Acción:* recortar un margen (≥ 0.1–0.2 s) al calcular picos/estadísticas o filtrar con extensión de señal; excluir bordes de M7 y M10.

**I5 · Perspectiva, escala y movimiento fuera de plano.**
La vara de 1 m calibra **un plano**; un punto que se acerca o aleja de la cámara cambia su escala (error de magnitud proporcional a la variación de profundidad relativa, p. ej. ≈ 10 % si la profundidad cambia 10 %). No están registradas la distancia ni la orientación de las cámaras. *Acción:* documentar posición de cámara, validar la escala con una distancia conocida en otro punto del plano y **enmascarar los instantes** donde el CV de longitudes supere un umbral (p. ej. 5 %), en vez de solo advertir.

**I6 · Reporte sin números y M10 poco informativo.**
`report.md` solo cuenta tablas; no resume picos, tiempos, lags ni métricas de calidad. *Acción:* añadir al reporte una tabla de resultados clave (picos, tiempos de eventos, duración de fases, lags, CV de longitudes, fc usadas) y un semáforo de calidad por grupo (offset verificado, QC, CV, fc en límite).

**I7 · Objetivo 4 sin entregable (brazos de palanca).**
`levers.py` no se invoca en ninguna parte. *Acción:* módulo `postures.py` que extraiga fotogramas en carga, máxima flexión/armado, impacto y seguimiento, dibuje el esquema segmentario y calcule el **brazo de palanca geométrico** (p. ej. distancia horizontal de la línea de gravedad a rodilla/cadera/hombro) con su lectura cualitativa; sin fuerza medida no se calcula magnitud de torque.

**I8 · Validación de ángulos de Kinovea sin ejecutar.**
`compare_with_kinovea` está implementada y probada, pero el pipeline **no la llama**; los grupos `*_angulos` usan los ángulos exportados sin contrastarlos con los recalculados desde coordenadas. *Acción:* invocarla donde existan ambas fuentes y registrar RMSE y convención (directa/suplementaria); si hay discrepancia > 2–3°, revisar definición de ángulo o marcadores.

**I9 · Frecuencia de corte distinta por marcador dentro de una misma cadena.**
Cada marcador se filtra con su propia fc, de modo que la longitud de un segmento deja de ser constante por artefactos del filtrado. *Acción:* usar una fc común por cadena (p. ej. el máximo o la mediana de los marcadores del segmento) y comprobar que el CV de longitudes no empeora.

**I10 · Convenciones sin verificar con el video.**
`flip_y: true` y la unidad `cm` de la cinemática lineal lateral son supuestos. *Acción:* verificar con un movimiento conocido (la muñeca **sube** en el lanzamiento: tras el `flip_y`, y debe aumentar) y con una distancia conocida de la escena.

### Menores / técnicos

**M1 · Un test falla (1 de 47).** `test_inverse_dynamics_runs_without_grf_under_explicit_flight_assumption` espera la frase *"sus fuerzas/torques no deben interpretarse como cargas reales durante el apoyo"*, pero `report.py` escribe *"…no representan las cargas reales durante el apoyo"*. Alinear el texto del reporte o la aserción.

**M2 · Panel vacío en figuras del brazo.** `_plot_inverse_dynamics` crea siempre `plt.subplots(3, 1)` aunque el brazo tenga 2 articulaciones; queda un tercer eje vacío. Cambiar a `plt.subplots(len(joints), 1, ...)`.

**M3 · Cobertura de pruebas desigual.** M10, M12 superior, M13 y M14 tienen pocas pruebas (sin prueba dedicada del miembro superior). Añadir `tests/test_inverse_dynamics_independent.py` (incluido) y pruebas de rutas de error.

**M4 · Bandera duplicada.** `shared_clock_verified` aparece en `events` y en `inverse_dynamics`; usar una única fuente de verdad (o heredar).

**M5 · Gráfica de cinemática saturada.** `plot_kinematics` superpone `theta_interior_*` y `theta_signed_*` de los mismos eslabones; separar por tipo.

**M6 · Higiene del repositorio.** `src/saque/__init__.py` aún dice "módulos M1-M6"; `docs/pipeline_saque_kick.md` conserva rutas y plan previos (M11 "no implementado" está anotado); mantener un solo documento rector (este README).

**M7 · Datos y ética.** Los CSV y videos de jugadoras no deben subirse sin consentimiento; documentar procedencia, consentimiento y anonimización, y fijar un `data/README` con la versión de Kinovea y la calibración usada.

**M8 · Política de tiempos duplicados.** `first` es arbitraria; confirmar en el video si el 1.535 s repetido es una muestra duplicada o un error de redondeo.

### Lo que sí está bien (verificado)

- M1–M8 del repositorio son idénticos a la versión probada y pasan sus pruebas.
- La dinámica inversa (inferior y superior) coincide con una implementación independiente (§9.1) y cumple los límites estático y de caída libre.
- El diseño es honesto: el pipeline **omite** grupos con offsets pendientes, **marca** lo no verificado, registra cada exclusión y etiqueta los supuestos de fuerza externa en tablas y figuras.
- Se calcula en SI con tiempos reales; no se une por índice de fila.

---

## 11. Hoja de ruta priorizada

**P0 — antes de presentar cualquier resultado final**
1. Verificar con el video los offsets de tobillo y definir un evento común entre vistas (C5).
2. Marcar manualmente el fotograma de **impacto** y de **despegue/contacto del pie** (`events.manual`; condiciona las fases y C1).
3. Definir sujeto/intento y reestructurar `data/raw` (C4); confirmar si son 5 intentos por jugadora.
4. Mover variables sagitales a la vista lateral (C2).
5. Desactivar por defecto M12 inferior y declarar M12 superior como exploratorio y de cota inferior (C1, C3).
6. Verificar `flip_y` y la unidad lateral (I10).
7. Corregir M1 y M2 para dejar la suite en verde.

**P1 — para fortalecer la calidad**
8. Sensibilidad de filtro ±2 Hz y recorte de bordes (I3, I4).
9. Longitudes medidas por sujeto y comprobación de consistencia de aceleraciones (I1).
10. Invocar `compare_with_kinovea` y enmascarar por CV de longitud (I8, I5).
11. Reporte con tabla de resultados y semáforo de calidad (I6).
12. Módulo de posturas y brazos de palanca (I7).

**P2 — opcional**
13. Digitalizar tronco y ángulo de hombro/cadera para completar la cadena próximo-distal.
14. Raqueta (velocidad de cabeza) para un impacto más fiable que el proxy de muñeca.
15. M11 (ML **exploratorio** si hay ≥ varias jugadoras): detección de anomalías de tracking, PCA funcional, DTW + clustering, regresión Ridge/PLS con validación *leave-one-out*. Con ~5 intentos no generaliza; es herramienta de QC y descubrimiento, no de predicción.
16. Con GRF sincronizada: `measured_grf`, torques normalizados (N·m/kg) y análisis de sensibilidad antropométrica.

---

## 12. Limitaciones globales

2D (proyecciones en el plano de cada cámara), cinemática sin cinética medida, muestra pequeña y de composición aún indefinida, modelo de cuerpo rígido (no se estiman tensiones ni deformaciones de tejidos) y rendimiento inferido solo desde la propia jugadora (sin pelota). Las asociaciones observadas no demuestran causalidad ni capacidad de prevenir lesiones.

---

## 13. Documentación relacionada

- `docs/contexto_proyecto_saque_kick.md` — objetivos, hipótesis, diseño de adquisición y alcance.
- `docs/contexto_bases_datos.md` — inventario de CSV, unidades y problemas de sincronía.
- `docs/contexto_dinamica_inversa.md` — modelo Newton-Euler y variables.
- `docs/pipeline_saque_kick.md` — documento maestro previo del pipeline (este README lo actualiza).
