# Pipeline de procesamiento: análisis cinemático del saque kick (video + MediaPipe)

> Documento maestro del pipeline (versión 0.2, 26-nov-2026). Contextos:
> [`contexto_proyecto_saque_kick.md`](./contexto_proyecto_saque_kick.md) (objetivos),
> [`contexto_bases_datos.md`](./contexto_bases_datos.md) (videos y CSV históricos) y
> [`contexto_dinamica_inversa.md`](./contexto_dinamica_inversa.md) (modelo Newton-Euler 2D).
> Convención interna: SI (m, s, rad), y hacia arriba. Todo parámetro vive en `config/pipeline.yaml`.

La versión anterior (M1–M14 sobre exportaciones de Kinovea de un solo saque) se reemplazó. Se
eliminaron los módulos que solo servían a Kinovea (`catalog`, `io_kinovea`, `sync` por offsets
manuales); los de cálculo (filtrado, cinemática, eventos, Jacobiano, estadística, dinámica inversa)
se conservaron y se adaptaron a datos con huecos.

---

## 1. Qué resuelve y qué no

**Sí:** pose 2D en cada fotograma de ambos videos; refinamiento con estampitas; detección automática
de los 5 saques y del impacto; sincronización entre vistas; θ/ω/α, fases, secuencia próximo-distal,
Jacobiano y dinámica inversa exploratoria por saque; tendencias entre saques (posible fatiga).

**No:** 3D (dos vistas 2D independientes); velocidad o spin de la pelota (solo se usa para fechar el
impacto); cargas reales (no hay GRF ni fuerzas en la raqueta); fatiga fisiológica; causalidad o riesgo
de lesión. Con 5 saques de una jugadora todo lo estadístico es **exploratorio**.

---

## 2. Etapas

```
video ──► [pose] MediaPipe + estampitas (caché parquet)
            │
            ▼
      [kinematics] fusión estampita/landmark → calibración (m) → QC → filtro → θ, ω, α, v
            │
            ▼
      [serves] candidatos (altura de muñeca) → impacto (fin de caída de la pelota)
               → ventanas por saque → sincronización de la otra vista (altura de cadera)
            │
            ▼
      [analysis] por saque: eventos, fases, secuencia, Jacobiano, dinámica inversa
                 entre saques: fatiga (tendencias) y variabilidad de curvas
            │
            ▼
      [report] figuras + reports/report.md
```

Cada etapa lee lo que dejó la anterior en disco: `saque run --stages kinematics serves analysis report`
repite todo menos MediaPipe (~5 min por video, la parte cara).

---

## 3. Pose y estampitas (`video/`)

### 3.1 Lectura de video (`reader.py`)
- ffmpeg (no `cv2.VideoCapture`): respeta la rotación del celular (videos verticales), decodifica
  HEVC/H.264 de forma uniforme y entrega RGB listo para MediaPipe.
- **Tiempo real por fotograma** (`frame_times`, pts de los paquetes). Los videos pierden fotogramas
  (0.19 s y 0.13 s tras el primero): usar `índice / 120` desfasaba todo el análisis ~0.19 s
  respecto del video y desalineaba esqueleto e imagen. La frecuencia real es ≈ 120.16 fps.
- `load_pose` lleva la serie a una **malla uniforme** `round(pts · fps)` e inserta filas NaN donde
  faltan fotogramas (quedan como huecos explícitos para QC y filtrado).

### 3.2 MediaPipe (`pose.py`)
Pose Landmarker *heavy* en modo VIDEO; 19 de los 33 landmarks con nombres en español
(`hombro_der`, `muneca_izq`, …). Se guarda x, y (px), visibilidad y, para los landmarks configurados,
el centroide de la estampita.

> `mediapipe>=0.10.30,<1.0`: las 1.x se cierran al cargar el modelo (Python 3.11 y 3.14).

### 3.3 Estampitas (`stickers.py`)
En una ventana de 90 px alrededor del landmark se segmenta en HSV (tono 22–48, saturación ≥ 110) y se
acepta la mancha más cercana con área 25–1500 px y circularidad ≥ 0.35. La saturación separa la
estampita de la playera (mismo tono, menos saturada); el área descarta pelotas y la playera.
Qué estampitas se usan se configura por vista (`views.<vista>.stickers`):

| Vista | Estampitas usadas | Motivo |
|---|---|---|
| lateral | hombro, codo, muñeca, cadera, rodilla, tobillo, punta (lado derecho) | Lado de golpeo hacia la cámara |
| posterior | cadera izq., rodilla izq., ambos tobillos | Las demás no se ven de espaldas |

### 3.4 Impacto con la pelota (`ball.py`)
La pelota se usa **solo para fechar el contacto**. Tras el lanzamiento reentra por arriba del cuadro y
cae con aceleración ≈ g; se sigue con predicción parabólica (ventana de 15 px) desde que aparece en la
parte alta (y < 450 px) e ignorando manchas fijas del fondo. **Impacto = último punto de la caída**:
el contacto saca a la pelota de la trayectoria. En los 5 saques el contacto ocurre 30–50 ms después
del máximo de altura de la muñeca (consistente) y la verificación visual muestra la pelota en las
cuerdas (`figures/impactos_lateral.png`). Si la pelota no se puede seguir, el respaldo es el máximo de
rapidez de la muñeca cerca del candidato (y queda una advertencia).

---

## 4. Marcadores y cinemática (`processing/`)

### 4.1 Fusión estampita / landmark (`markers.fuse_markers`)
1. Landmarks con visibilidad < 0.5 → hueco.
2. Detecciones de estampita con desfase al landmark atípico (> 4 MAD) → descartadas.
3. **Una estampita física pertenece a un solo marcador**: si dos la tomaron en el mismo fotograma
   (codo junto al hombro, muñeca en reposo junto a la cadera), se asigna al landmark esperado más
   cercano.
4. Huecos cortos de estampita (≤ 6 fotogramas) se interpolan. Cada cambio de fuente introduce un
   pequeño salto que la derivada convierte en pico espurio, así que se evita alternar entre fuentes.
5. Sin estampita: landmark + desfase mediano estampita-landmark.

La fuente de cada marcador por fotograma (`estampita` / `pose` / `hueco`) se guarda (`src_*`) y se
resume en `tables/fuente_marcadores.csv`.

### 4.2 Calibración (`method: ground_plane`)
La vara de 1 m da la escala **en su propio plano**, pero no está a la misma distancia de la cámara que
la jugadora. En la vista posterior la vara está sobre la línea de fondo, más lejos que la jugadora, y
sin corrección la estatura aparente salía 2.30 m. Se corrige con la altura conocida de la cámara
(H = 1.30 m), suponiendo una cámara nivelada (pinhole):

- horizonte: `y_h = y_base_vara − H · s_vara` (s en px/m);
- escala en el plano de la jugadora: `s_j = (y_pies − y_h) / H`, donde `y_pies` es el contacto con el
  suelo en la postura más erguida;
- origen: el suelo bajo sus pies, de modo que y es la altura sobre el suelo.

Control (estatura aparente = nariz–pie / 0.92): **lateral 1.58 m, posterior 1.59 m** frente a 1.64 m
reales. Las dos vistas coinciden y quedan 3.5 % abajo, lo que se atribuye a la postura (nunca del todo
erguida) y a una posible inclinación leve de la cámara. Si el error pasa de 10 % se emite una
advertencia. `method: pole` usa la escala de la vara sin corregir. Los extremos de la vara se miden
con `saque calibrate <vista>`. Los ángulos no dependen de la escala.

### 4.3 Control de calidad (todo queda en `tables/qc_log.csv`)
| Control | Criterio | Por qué |
|---|---|---|
| Longitud de segmento | salto entre fotogramas > 30 % de la mediana, o fuera de ±60 % | Un segmento rígido no cambia de golpe. En 2D sí cambia **gradualmente** por escorzo (el brazo a ~30 rad/s hasta ~25 % por fotograma), por eso el criterio es el salto. |
| Rapidez imposible | desplazamiento > 15 m/s entre fotogramas | Cambios de estampita o errores de MediaPipe |
| Saltos aislados | residuo frente a mediana móvil > 8σ y > 2 % del rango | Etiquetas erróneas puntuales |
| Huecos cortos | ≤ 0.05 s se interpolan; los largos quedan NaN | Se interpolan **solo** huecos completos cortos |
| ω implausible | \|ω\| > 60 rad/s (≈ 3400 °/s en el plano) → NaN | Error de marcador o escorzo extremo |

### 4.4 Filtrado y derivadas
Butterworth pasa-bajas de fase cero (orden 2, `filtfilt`). El corte se elige por marcador con el
análisis de residuos de Winter (4–20 Hz) sobre su tramo continuo más largo. El filtro se aplica **por
tramos** entre huecos y las derivadas son diferencias centrales. Se descartan **3 muestras junto a
cada hueco** porque los bordes del filtro y de la derivada producían picos espurios de ω.

### 4.5 Variables (`views.<vista>.segments/joints/linear`)
- Ángulo absoluto de segmento `atan2(Δy, Δx)` (desenvuelto por tramos), y ángulo interior y con signo
  de articulación. ω y α en rad/s y rad/s². Posición, velocidad y aceleración lineal por marcador.
- **Lateral (sagital aproximado):** tronco, brazo, antebrazo, muslo, pierna y pie, con ángulos de
  hombro (elevación brazo-tronco), codo, cadera, rodilla y tobillo.
- **Posterior (frontal):** líneas de hombros y de caderas, tronco, brazo y antebrazo, con codo y
  ambas rodillas.

---

## 5. Saques, eventos y sincronización (`analysis/serves.py`, `events.py`)

- **Candidatos**: los 5 máximos de altura de la muñeca de golpeo con mayor prominencia, separados
  al menos 3 s.
- **Impacto**: fin de la caída de la pelota (ver §3.4). Se puede corregir con
  `serves.manual_impacts_s`.
- **Ventana de cada saque**: de 3.0 s antes a 0.8 s después del impacto. El gesto empieza ~2.3 s
  antes; con 1.6 s quedaba fuera.
- **Sincronización**: correlación cruzada de la altura media de cadera. Los saltos de los saques se
  ven en las dos vistas: posterior = lateral + 0.64 s, con correlación 0.75. Como control, los
  impactos que la vista posterior detecta por su cuenta con la muñeca difieren 0.13 s en mediana.
- **Eventos**: inicio (fin de la última quietud de la muñeca, ≥ 0.1 s por debajo del 5 % del pico
  cerca del impacto, antes de la flexión de rodilla), inicio de flexión de rodilla (95 % del rango
  desde la postura erguida del segundo previo), máxima
  flexión de rodilla, pico de ω del codo, impacto y fin (rapidez < 20 % del pico). Fases:
  preparación (lanzamiento y armado), carga, aceleración y seguimiento.
- **Vista posterior**: no detecta sus propios eventos. Usa los de la lateral trasladados con el
  desfase entre vistas, porque su rapidez de muñeca nunca baja del umbral y el inicio caía en el
  borde de la ventana.
- **Secuencia próximo-distal** (vista lateral): pico de |ω| (o rapidez de muñeca) de muslo → tronco
  → brazo → codo → muñeca entre el inicio y `sequence_margin_s` después del impacto. Usa `nanargmax`,
  y una serie con menos de `sequence_min_valid_frac` (50 %) de muestras válidas se omite. Se reportan
  el orden observado (`=` = mismo fotograma), los desfases en el orden esperado (negativo = invertido),
  el número de inversiones y el τ de Kendall.
- **Picos articulares** (`stats.joint_peaks`): |ω| y |α| máximos de cada articulación entre el inicio
  y el impacto, con su tiempo respecto al impacto y la fracción de muestras válidas.
- **Curvas θ/ω/α** (`curvas_articulares.csv`): normalizadas de 0 a 100 % (inicio → impacto). Incluyen
  la banda media ± DE y el RMSD de cada saque respecto a la media.
- **Asociaciones** (`stats.spearman_matrix`): ρ de Spearman entre los indicadores por saque y las
  variables del CoM. Es descriptivo y no corrige por comparaciones múltiples.

---

## 6. Fatiga (`analysis/fatigue.py`)

Valor por saque de cada variable (`fatigue.variables`), además de la duración de la carga y de la
aceleración. El seguimiento se excluye porque la ventana posterior no siempre lo cubre. Para cada
variable se reporta:
- pendiente por saque (regresión lineal), R², p, cambio del primer al último saque;
- ρ de Spearman con IC95 % bootstrap;
- CV entre saques;
- índice simple: último tercio frente al primero, en variables de desempeño.

**Control de calidad del pico:** si menos del 80 % de las muestras en los 200 ms previos al impacto
son válidas, el pico de ese saque se excluye. El valor original se conserva en `valor_sin_control`.

**Confianza por variable:**

| Confianza | Variables | Por qué |
|---|---|---|
| alta | flexión máx. de rodilla, altura máx. de muñeca, duración de fases | Posiciones y tiempos, sin derivadas |
| media | rapidez pico de muñeca, elevación de hombro | Derivada de un marcador con estampita |
| baja | picos de ω de codo, hombro y tronco | Segunda fuente de error: oclusión del brazo y escorzo 2D cerca del impacto |

### Resultado actual (n = 5, exploratorio)
- Los picos de ω del brazo pasan el control solo en 3 de 5 saques. En los saques 1 y 4 el brazo de
  golpeo queda ocluido u oculto por MediaPipe cerca del impacto.
- Ningún indicador de confianza alta muestra una tendencia concluyente. La flexión máxima de rodilla
  sube levemente a lo largo de la serie (menos flexión, ρ = 0.7, p = 0.07), pero el valor de p cambió
  entre 0.01 y 0.07 al ajustar el QC: es una tendencia frágil.
- **Conclusión prudente:** con 5 saques no hay evidencia cinemática clara de fatiga. Para estudiarla
  hacen falta series más largas (por ejemplo, 20–30 saques) y, de preferencia, una medida
  independiente como la percepción de esfuerzo o la velocidad de la pelota.

---

## 6b. Cuerpo completo (`analysis/body.py`)

| Análisis | Método | Control / limitación |
|---|---|---|
| Centro de masa | Media ponderada de 15 segmentos (fracciones de Winter); pesos renormalizados si falta un segmento (se descarta si falta > 20 % de la masa) | De pie: CoM ≈ 1.0 m |
| Fuerza vertical del suelo | F = m(a_CoM + g), con el CoM filtrado a 6 Hz | De pie = 1.00–1.02 × peso. Cerca del impacto no es confiable (brazo) |
| Impulso de piernas | Desde la máxima flexión de rodilla hasta la máxima velocidad vertical del CoM: pico de fuerza, impulso neto y v = impulso/m | v por impulso frente a v medida |
| Elevación del CoM | Máximo en impacto ± 0.3 s menos la altura en la postura inicial | Lateral 0.14–0.18 m, posterior 0.21 m (escala) |
| Tiempo de vuelo | **No se reporta** | La altura 2D de los pies no es confiable (pie sin estampita, pierna trasera elevada, desplazamiento en profundidad) |
| Altura de contacto | Posición de la pelota al fin de su caída, convertida con la calibración | Solo en la vista donde se siguió la pelota |
| Base de apoyo | Separación horizontal entre tobillos al inicio y en la carga | |
| Ángulos por evento | Valor en inicio, inicio de carga, máxima flexión de rodilla, impacto y fin | |
| Rango de movimiento | Mínimo y máximo de inicio a impacto | Después del impacto la rotación del cuerpo vuelve ambiguos los ángulos 2D |
| Vista frontal | Inclinación de las líneas de hombros y caderas (periodo 180°, continuas por bloques), diferencia hombros − caderas e inclinación lateral del tronco | Se anulan cuando la línea proyectada mide menos del 50 % de su p95 (cuerpo de perfil) |
| Palancas | Distancia horizontal de cada articulación a la vertical por el CoM; momento estático = m·g·brazo | Cualitativo: sin fuerzas inerciales ni musculares |
| Potencia articular | P = M·(ω_distal − ω_proximal) sobre los momentos de la dinámica inversa | Exploratoria |

## 7. Jacobiano y dinámica inversa

- **Jacobiano** `v = v_hombro + J(θ)ω` (brazo planar de 2 eslabones, vista lateral). La validación
  frente a la derivada directa de la muñeca es casi una identidad. Lo informativo es el **CV de las
  longitudes** (≈ 0.2–0.26), que confirma el escorzo 2D: el brazo no se mueve en el plano de la
  cámara.
- **Dinámica inversa** (vista lateral, por saque):
  - tren superior: hombro y codo, con la muñeca libre;
  - tren inferior: pie, pierna y muslo con **GRF estimada por el centro de masa**
    (`inverse_dynamics.grf`). Las componentes son F_x = m·a_x,CoM y F_z = m·(a_z,CoM + g). Se
    asigna el 50 % a la pierna analizada, porque en la carga del saque los pies quedan juntos
    (0.01–0.03 m). El centro de presión se fija a mitad de camino entre el tobillo y la punta. Hay
    contacto cuando la GRF vertical de la pierna supera el 20 % de su parte del peso;
    `grf.mode: none` vuelve a GRF = 0.
  - Los picos se buscan entre el inicio y el impacto (+50 ms). En el tren inferior, la ventana va del
    inicio de la flexión de rodilla al despegue (máxima velocidad vertical del CoM). Antes, con los
    pies separados, el reparto 50/50 vertical no vale. Después, en el aire o en el aterrizaje sobre
    un pie, la GRF estimada es ruido del CoM.

  Las fracciones antropométricas de Winter están en la configuración y las longitudes salen de la
  mediana de cada saque. **Son estimaciones exploratorias:** la GRF no se mide, el reparto entre pies
  y el centro de presión son supuestos, no hay raqueta y hay escorzo. La potencia de cadera sale
  negativa durante el empuje: el tronco rota hacia atrás (el arco del saque kick) contra un momento
  flexor estimado. Puede ser control excéntrico real, pero depende del centro de presión supuesto.

---

## 8. Limitaciones y decisiones

La discusión sobre la idealización en segmentos rígidos frente al sólido continuo está en
[`solido_continuo.md`](solido_continuo.md).

1. Cada vista es una proyección 2D. Las rotaciones fuera del plano (rotación interna del hombro,
   pronación) no se miden y alteran los ángulos proyectados.
2. MediaPipe pierde el brazo de golpeo en algunos impactos. Las estampitas lo compensan solo donde
   se detectan, y los picos del brazo de los saques 1 y 4 se excluyen.
3. Escala con corrección de profundidad por cámara nivelada a 1.30 m (error de estatura ≈ −3.5 %); la
   jugadora se mueve en profundidad durante el saque y se usa una escala fija por vista.
4. El impacto se fecha con la pelota con una precisión de unos ±1–2 fotogramas (≈ 10–17 ms).
5. Los umbrales de QC (`qc.*`) son decisiones documentadas y los resultados del brazo dependen de
   ellos. Cualquier cambio debe registrarse y su efecto reportarse.
6. Una jugadora y 5 saques: sin inferencia poblacional.

---

## 9. Salidas

| Archivo | Contenido |
|---|---|
| `tables/calibracion.csv` | escala, estatura aparente, fps real, fotogramas perdidos |
| `tables/fuente_marcadores.csv` | % estampita / MediaPipe / hueco por marcador |
| `tables/saques.csv`, `sincronizacion.csv` | impactos, ventanas, método, desfase entre vistas |
| `tables/events.csv`, `phases.csv`, `sequence.csv`, `peaks.csv` | eventos y fases por saque |
| `tables/fatiga_*.csv` | valores por saque, tendencias, índice |
| `tables/curvas_normalizadas.csv`, `variabilidad_*.csv` | curvas 0–100 % y variabilidad entre saques |
| `tables/jacobian_metrics.csv`, `descriptives.csv`, `filtros.csv`, `qc_log.csv` | validación y trazabilidad |
| `tables/picos_articulares.csv`, `curvas_articulares.csv`, `variabilidad_*_articular.csv` | θ/ω/α por articulación y consistencia |
| `tables/correlaciones.csv` | ρ de Spearman entre indicadores por saque |
| `tables/residuos_winter.csv`, `qc_resumen.csv` | curvas R(fc) del filtrado y muestras anuladas por control |
| `data/processed/<vista>/saque_XX/` | cinemática, Jacobiano y dinámica inversa por saque |
| `figures/` | detección de saques, impactos con esqueleto, fuentes, fases, secuencia cinemática, θ/ω/α, variabilidad, correlaciones, curvas, fatiga, Jacobiano y su validación, residuos de Winter, QC, dinámica inversa |

## 10. Pruebas
`pytest`: 62 pruebas con datos sintéticos, sin video:
- estampitas y malla temporal con fotogramas perdidos;
- calibración, fusión y duplicados, y controles de QC;
- filtrado por tramos y recorte de bordes;
- cinemática frente a soluciones analíticas, eventos, Jacobiano y dinámica inversa en caída libre y
  en estática;
- detección de saques, desfase entre vistas y seguimiento de la pelota;
- fatiga;
- el pipeline completo de punta a punta.
