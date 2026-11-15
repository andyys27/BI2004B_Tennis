# Contexto de las bases de datos cinemáticas

## Propósito y alcance

Este documento describe los CSV disponibles para el análisis biomecánico del saque kick de tenis, obtenidos mediante seguimiento de marcadores en Kinovea desde una vista lateral y una vista posterior. Está pensado para acompañar a [`contexto_dinamica_inversa.md`](./contexto_dinamica_inversa.md) como contexto de entrada para construir un pipeline de análisis reproducible.

Los archivos son tablas CSV, no bases de datos relacionales. Cada fila representa una muestra temporal y cada columna una variable. Los datos corresponden a exportaciones de análisis cinemático: posición/coordenadas, ángulos y variables derivadas de velocidad y aceleración. No hay identificadores de fotograma, sujeto, ensayo ni evento que permitan vincular inequívocamente las exportaciones.

Se revisaron los archivos de `data/lateral/` y `data/posterior/`, además del CSV duplicado en `posterior/`. Todos tienen encabezados, filas de ancho consistente y valores no vacíos en todas las columnas observadas.

> **Unidades y definiciones:** los ángulos de las bases están en grados (°), no en radianes. `ABS` identifica el ángulo absoluto y `MA` el promedio móvil. `PI1` identifica la pierna izquierda y `PD` la pierna derecha. Los sufijos `_cm` indican centímetros; `tiempo_s` y `time_s` indican segundos, y `tiempo_ms` milisegundos. Las unidades de las variables lineales y el método/parámetros usados para el promedio móvil no están documentados en los CSV.

## Inventario de archivos y columnas

La cantidad indicada es el número de filas de datos, sin contar el encabezado. En los archivos listados, cada columna tiene ese mismo número de valores no vacíos.

### Vista lateral

| Archivo | Filas | Columnas y significado indicado por los nombres |
|---|---:|---|
| [`data/lateral/cinematica_lineal_lateral.csv`](./data/lateral/cinematica_lineal_lateral.csv) | 544 | `tiempo_s`; `x_cadera`, `y_cadera`; `x_rodilla`, `y_rodilla`; `x_tobillo`, `y_tobillo` (coordenadas de cadera, rodilla y tobillo); `v_cadera`, `v_rodilla`, `v_tobillo`; `a_cadera`, `a_rodilla`, `a_tobillo` (variables derivadas por punto). |
| [`data/lateral/cinematica_angular_lateral_rodilla.csv`](./data/lateral/cinematica_angular_lateral_rodilla.csv) | 537 | `tiempo_s`; `angulo_rod`; `ABS_angulo_rod`; `angulo_MA_rod`; `v_rod`; `a_rod`. |
| [`data/lateral/cinematica_angular_lateral_tobillo.csv`](./data/lateral/cinematica_angular_lateral_tobillo.csv) | 138 | `tiempo_s`; `angulo_tob`; `ABS_angulo_tob`; `angulo_MA_tob`; `v_tob`; `a_tob`. |
| [`data/lateral/PI1_tiempo_angulo_abs_rodilla.csv`](./data/lateral/PI1_tiempo_angulo_abs_rodilla.csv) | 537 | `tiempo_ms`; `tiempo_s`; `angulo_PI1`; `ABS_angulo_PI1`; `angulo_MA_PI1`; `v_clean_PI1`; `a_clean_PI1`. |
| [`data/lateral/PI1_tiempo_angulo_abs_tobillo.csv`](./data/lateral/PI1_tiempo_angulo_abs_tobillo.csv) | 138 | `tiempo_ms`; `tiempo_s`; `angulo_tob`; `ABS_angulo_tob`; `angulo_MA_tob`; `v_clean_tob`; `a_clean_tob`. |

En los archivos angulares laterales, `rod`/`tob` identifican rodilla/tobillo. Los archivos con `PI1` en el nombre corresponden a la pierna izquierda.

La cadena lineal lateral disponible incluye cadera, rodilla y tobillo, sin marcador de
punta del pie. M12 la asocia a PI1 y solo puede estimar rodilla/cadera bajo la hipótesis
de fuerza externa nula en tobillo; no produce una estimación de torque de tobillo.

### Vista posterior

| Archivo | Filas | Columnas y significado indicado por los nombres |
|---|---:|---|
| [`data/posterior/cinematica_angular_rodilla.csv`](./data/posterior/cinematica_angular_rodilla.csv) | 282 | `Tiempo (ms)`; `tiempo_s`; `Angulo_Rod`; `angulo_MA_Rod`; `v_Rod`; `a_Rod`. |
| [`data/posterior/cinematica_angular_tobillo.csv`](./data/posterior/cinematica_angular_tobillo.csv) | 282 | `tiempo_ms`; `tiempo_s`; `angulo_tob`; `angulo_MA_tob`; `v_tob`; `a_tob`. |
| [`data/posterior/cinematica_angular_brazo.csv`](./data/posterior/cinematica_angular_brazo.csv) | 282 | `tiempo_ms`; `tiempo_s`; `angulo_codo`; `angulo_codo_clean`; `v_codo`; `a_codo`. |
| [`data/posterior/cinematica_posterior_PD.xlsx.csv`](./data/posterior/cinematica_posterior_PD.xlsx.csv) | 348 | `time_s`; `x_punta_cm`, `y_punta_cm`; `x_tob_cm`, `y_tob_cm`; `x_rod_cm`, `y_rod_cm`; `X_cad_cm`, `y_cad_cm` (coordenadas de punta del pie, tobillo, rodilla y cadera); `v_punta`, `v_tob`, `v_rod`, `v_cad`; `a_punta`, `a_tob`, `a_rod`, `a_cad`. |
| [`data/posterior/cinematica_lineal_brazo_pos.csv`](./data/posterior/cinematica_lineal_brazo_pos.csv) | 356 | `time_s`; `x_mun_cm`, `y_mun_cm`; `x_codo_cm`, `y_codo_cm`; `x_homb_cm`, `y_homb_cm` (muñeca, codo y hombro). |
| [`data/posterior/cinematica_lineal_brazo_v_a.csv`](./data/posterior/cinematica_lineal_brazo_v_a.csv) | 356 | `tiempo_ms`; `v_mun`, `v_codo`, `v_homb`; `a_mun`, `a_codo`, `a_homb` (muñeca, codo y hombro). |

El archivo de coordenadas `cinematica_posterior_PD.xlsx.csv` corresponde a la pierna derecha. La grafía `X_cad_cm` (mayúscula) es el encabezado real y debe respetarse al leer el archivo.

### Archivo adicional fuera de `data/`

[`posterior/cinematica_angular_rodilla.csv`](./posterior/cinematica_angular_rodilla.csv) contiene 282 filas y los mismos encabezados que `data/posterior/cinematica_angular_rodilla.csv`. Los dos archivos son idénticos byte por byte; deben tratarse como una sola exportación y evitar contarlos como ensayos diferentes.

## Extensión temporal y muestreo observado

Los tiempos están representados con tres decimales en la mayoría de los archivos. Los intervalos consecutivos suelen alternar entre aproximadamente 0.008 y 0.009 s, compatibles con muestras cercanas a 120 Hz, pero no constituyen una frecuencia perfectamente uniforme. Los tiempos se deben leer de cada fila, no reconstruirse suponiendo una frecuencia fija.

| Archivo(s) | Muestras | Tiempo inicial–final |
|---|---:|---:|
| Lateral: cinemática angular de rodilla y `PI1` de rodilla | 537 | 0.000–4.461 s |
| Lateral: cinemática lineal | 544 | 0.000–4.523 s |
| Lateral: cinemática angular de tobillo | 138 | 2.538–3.678 s |
| Lateral: `PI1` de tobillo | 138 | 2.638–3.778 s |
| Posterior: ángulos de rodilla y codo | 282 cada uno | 0.593–2.931 s |
| Posterior: ángulo de tobillo | 282 | 0.000–2.340 s |
| Posterior: posición lineal de brazo | 356 | 0.000–2.955 s |
| Posterior: velocidad/aceleración de brazo | 356 | `tiempo_ms`: 0–2956 ms |
| Posterior: coordenadas y derivadas de pie/pierna | 348 | 0.000–2.889 s |

## ¿Están sincronizados los análisis?

**No se debe asumir que todas las exportaciones de una misma vista comparten el mismo origen temporal ni que pueden combinarse por número de fila.** Los CSV tienen marcas de tiempo, pero no incluyen número de fotograma ni un identificador de reloj global. La coincidencia de filas o de duración no demuestra sincronía.

### Vista lateral

- Los dos archivos de rodilla con 537 filas tienen exactamente la misma serie `tiempo_s` (0.000–4.461 s). Es posible alinearlos por tiempo; sus encabezados y variables, no obstante, no son idénticos.
- Los dos archivos de tobillo tienen 138 filas y cubren la misma duración, pero sus marcas de tiempo difieren en **0.100 s**: la cinemática angular va de 2.538 a 3.678 s y el archivo `PI1` de 2.638 a 3.778 s. No deben combinarse fila a fila sin investigar y resolver ese desfase. Confirmar con el video/fotogramas si representa un origen temporal distinto o un error de exportación.
- La cinemática lineal tiene 544 filas, frente a 537 para rodilla y 138 para tobillo; no corresponde directamente por índice. Su columna `tiempo_s` tampoco es estrictamente creciente: las filas 185 y 186 (numeración desde 1) tienen ambas `1.535 s`. Antes de interpolar o derivar, hay que resolver cómo tratar esas dos muestras repetidas.
- Las series de rodilla/lineal parten cerca de 0 s, mientras que las series de tobillo son recortes de una fase más breve. Pueden compartir el reloj del video, pero eso no queda demostrado solo por estos CSV.

### Vista posterior

- Las series angulares de rodilla y codo tienen 282 filas y marcas `tiempo_s` idénticas (0.593–2.931 s).
- La serie angular del tobillo también tiene 282 filas, pero usa el intervalo 0.000–2.340 s. Aunque el conteo coincide, su tiempo no coincide con rodilla/codo; probablemente hay que determinar el desplazamiento temporal a partir del video o de una referencia de fotograma.
- Las coordenadas de pierna/pie tienen 348 filas y las de brazo 356. Ambas comienzan cerca de 0 s, pero tienen distinto número de muestras y duración; no alinearlas por índice. El archivo separado de velocidad/aceleración del brazo tiene 356 filas, igual que el de posiciones, pero solo proporciona `tiempo_ms`; se requiere verificar su correspondencia exacta con `time_s` antes de unirlo.

### Recomendación para el pipeline

1. Conservar los tiempos exportados y normalizar sus nombres/unidades a una columna estándar (`time_s`).
2. Establecer el tiempo global por video —idealmente mediante número de fotograma o sincronización visual con un evento identificable— y registrar cualquier offset aplicado por archivo.
3. Validar que el tiempo sea creciente y resolver tiempos duplicados, como el de la cinemática lineal lateral.
4. Alinear series por tiempo mediante una regla documentada (interpolación/resampling, si corresponde), no por posición de fila. No extrapolar fuera del intervalo común sin justificarlo.
5. Antes de calcular magnitudes dinámicas, confirmar sentido de ejes, puntos anatómicos y método de suavizado/derivación. Como los ángulos de origen están en grados y la dinámica inversa descrita requiere radianes, convertir los ángulos a radianes y sus derivadas angulares a rad/s y rad/s² antes de utilizarlos.

## Relación con dinámica inversa y datos faltantes

La documentación de dinámica inversa requiere cinemática angular de segmentos (en radianes, y sus derivadas) y fuerzas de reacción del suelo (`GRF_x` y `vGRF`). Los CSV inventariados contienen variables cinemáticas, pero **no se encontró en estas carpetas un CSV de fuerzas de plataforma ni metadatos de sincronización con dichas fuerzas**. El pipeline debe localizar esa fuente por separado y sincronizarla con el video antes de calcular fuerzas y torques articulares.

Asimismo, estos archivos no especifican explícitamente qué columnas de ángulo corresponden a ángulos absolutos de segmentos frente a ángulos articulares. Esa correspondencia debe verificarse antes de alimentar las ecuaciones descritas en el documento de dinámica inversa.
