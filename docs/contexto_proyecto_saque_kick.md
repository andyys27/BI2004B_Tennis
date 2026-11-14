# Contexto general del proyecto: análisis biomecánico del saque kick

## Resumen ejecutivo

El proyecto busca caracterizar la cinemática del saque kick de tenis en jugadores adultos del equipo representativo mediante análisis de video 2D. Se utilizan dos vistas (lateral y posterior), tracking en Kinovea y un pipeline de procesamiento en **Python**. La actividad requiere completar la digitalización de puntos anatómicos, de la raqueta y de la pelota; el análisis se centra en posiciones y variables angulares a lo largo de las fases del saque, coordinación temporal entre segmentos, interpretación cualitativa de posturas y velocidad resultante de la pelota.

La motivación a largo plazo es comprender cómo la técnica y la secuencia de movimiento pueden relacionarse con el rendimiento y con las cargas biomecánicas, contribuyendo a futuro a estrategias de prevención de lesiones. El análisis cinemático de video, por sí solo, no mide fuerzas articulares ni demuestra que una técnica cause o prevenga lesiones. La estimación de fuerzas y torques sería una etapa posterior que requiere datos cinéticos adicionales y validación del modelo.

Este documento ofrece el macrocontexto del problema y sus objetivos. El detalle de los archivos y sus limitaciones está en [`contexto_bases_datos.md`](./contexto_bases_datos.md); el modelo propuesto de dinámica inversa está descrito en [`contexto_dinamica_inversa.md`](./contexto_dinamica_inversa.md).

## Problema y motivación

El saque kick se ejecuta numerosas veces durante un partido y depende de la coordinación de segmentos de las extremidades inferiores y superiores, el tronco y el brazo de golpeo. La repetición de patrones técnicos ineficientes podría asociarse con menor rendimiento o con cargas biomecánicas no deseadas en rodillas, tronco, hombro y codo.

Actualmente no se cuenta con un sistema de medición cuantitativo integrado que relacione la secuencia cinemática y los ángulos articulares con la velocidad/eficacia del saque y que, en una etapa futura, permita estudiar las cargas asociadas. Este proyecto plantea construir una primera herramienta de medición y análisis con video; no pretende diagnosticar lesiones ni cuantificar directamente el riesgo individual de lesión a largo plazo.

## Objetivo general

Caracterizar la cinemática del saque kick en jugadores adultos del equipo representativo mediante análisis de video con Kinovea y procesamiento en Python, digitalizando puntos anatómicos, de la raqueta y de la pelota para describir la evolución angular y posicional, la coordinación segmentaria y la relación con la velocidad de la pelota.

## Objetivos específicos

1. Digitalizar en cada vista puntos anatómicos relevantes, puntos de la raqueta y el centro de la pelota, documentando cómo se define cada punto y su sistema de coordenadas.
2. Obtener las posiciones angulares \(\theta(t)\), velocidades angulares \(\omega(t)\) y aceleraciones angulares \(\alpha(t)\) de las articulaciones que puedan medirse con los marcadores disponibles.
3. Segmentar el gesto en fases con eventos definidos de forma reproducible e identificar el timing de los movimientos y la secuencia temporal de coordinación entre segmentos.
4. Interpretar posturas estáticas clave, identificar brazos de palanca relevantes y discutir cualitativamente cómo la postura y la línea de acción de fuerzas podrían aumentar o reducir los torques articulares.
5. Obtener las coordenadas de la pelota y, si el seguimiento/calibración lo permiten, estimar su velocidad entre fotogramas.
6. Estimar la velocidad de la mano a partir de la cinemática, incluyendo la relación de velocidad de extremo con velocidades articulares mediante el modelo
   \[
   \mathbf{v} = J(\boldsymbol{\theta})\,\boldsymbol{\omega},
   \]
   donde la definición de articulaciones, coordenadas, Jacobiano y unidades debe establecerse y validarse para el modelo implementado.
7. Explorar asociaciones entre variables cinemáticas, velocidad de pelota y consistencia entre intentos, sin interpretarlas como causalidad.
8. Comparar las curvas obtenidas con rangos de la literatura que sean metodológicamente comparables, documentando las diferencias de población, técnica, plano y definición de cada variable.
9. Conectar conceptualmente el análisis de los segmentos corporales con la mecánica del sólido continuo y dejar preparado un camino de evolución hacia fuerzas y torques mediante datos cinéticos sincronizados y un modelo de dinámica inversa validado.

## Hipótesis de trabajo

- **H0 (propuesta):** una transferencia de impulso próximo-distal técnicamente adecuada durante el saque kick se asocia con mayor rendimiento y con un efecto más eficiente de la pelota.
- **H1 (propuesta):** las desviaciones técnicas y una coordinación segmentaria deficiente se asocian con mayor estrés biomecánico, menor eficacia del golpe y posibles factores de riesgo de lesión a largo plazo.

Estas son hipótesis direccionales de trabajo. En particular, la formulación de H0 no es una hipótesis nula estadística convencional (ausencia de asociación). Antes de una prueba inferencial se deberán definir hipótesis nula/alternativa, indicadores observables de transferencia próximo-distal, definición de rendimiento/efecto y el diseño muestral. Si no se mide giro o trayectoria de la pelota, no se podrá concluir directamente sobre la eficiencia del efecto kick.

## Diseño y adquisición de datos

### Participantes e intentos

- Población de interés: jugadores adultos del equipo representativo.
- Se han indicado **cinco iteraciones de saque** para el análisis actual. Falta precisar si son cinco intentos por jugador o cinco intentos en total; también debe registrarse el número de participantes.
- Los intentos deben identificarse de forma única (participante, sesión, intento) para poder medir variabilidad entre repeticiones sin confundirla con diferencias entre jugadores.

### Registro de video

- Dos cámaras: vista lateral y vista posterior.
- Frecuencia nominal indicada: 120 fps.
- Altura aproximada de cámara: 1.30 m.
- Vara de 1 m como referencia de escala.
- Los marcadores se siguen en Kinovea y las exportaciones se procesan posteriormente con Python.

La vara permite establecer una escala espacial por vista, pero no corrige automáticamente perspectiva o distorsión. Se debe documentar la posición/orientación de cámara, el plano de movimiento, la escala aplicada y la calidad de la calibración. Las dos vistas 2D no constituyen por sí mismas una reconstrucción 3D: cualquier fusión entre vistas requiere calibración y sincronización apropiadas.

La frecuencia de cámara es nominal. En las exportaciones existentes los intervalos temporales no son perfectamente uniformes y hay desfasajes entre algunas series. El procesamiento debe utilizar las marcas de tiempo verificadas, no asumir que cada fila está separada exactamente por \(1/120\) s.

## Variables de interés

| Variable planificada | Uso/definición que debe fijarse | Situación observada en los datos actuales |
|---|---|---|
| Cinemática angular articular \(\theta(t), \omega(t), \alpha(t)\) | Posición angular, velocidad y aceleración angular, con convención anatómica, signo, unidad y método de filtrado/derivación definidos. | Hay ángulos y variables `v`/`a` para algunas articulaciones, pero no para todas las variables previstas. Los CSV expresan los ángulos en grados; revisar la disponibilidad y sincronía por archivo en [`contexto_bases_datos.md`](./contexto_bases_datos.md). |
| Flexión de rodilla delantera | Ángulo articular y criterio para identificar la pierna delantera en cada intento. | Hay series angulares de rodilla y coordenadas de rodilla en algunas vistas. Debe confirmarse su correspondencia con la pierna delantera y con la definición anatómica elegida. |
| Inclinación del tronco | Orientación de un eje del tronco respecto a la vertical/horizontal o a otro segmento de referencia. | No hay una serie explícita de inclinación del tronco. Se requieren marcadores y una definición operacional del eje. |
| Elevación del hombro | Ángulo del brazo respecto al tronco/plano de referencia, no solo una coordenada del hombro. | Hay coordenadas de hombro, codo y muñeca en una exportación posterior; no hay una serie angular explícita de elevación del hombro. |
| Flexión del codo | Ángulo entre brazo y antebrazo, con convención de signo y rango definidos. | Existe una serie `angulo_codo` en la exportación posterior; debe verificarse que represente la definición requerida y la vista/ensayo correcto. |
| Flexión de cadera (\(\theta_2\)) | Ángulo relativo entre tronco/pelvis y muslo; especificar convención y lado. | Hay coordenadas de cadera y segmentos inferiores en archivos existentes, pero no una serie explícita de flexión de cadera lista para el modelo. Puede requerir puntos adicionales, sincronía y cálculo angular. |
| Posición de la mano \((x,y)\) | Trayectoria del punto anatómico (mano o muñeca) y sistema de coordenadas/calibración. | Hay coordenadas de muñeca (`x_mun_cm`, `y_mun_cm`) en una exportación posterior. Confirmar si la muñeca representa suficientemente la mano o si se debe rastrear otro punto. |
| Puntos de la raqueta | Coordenadas de puntos identificables (por ejemplo, extremo del mango y cabeza) para describir posición y orientación proyectadas. | No se identificaron coordenadas de raqueta en los CSV actuales; deben digitalizarse en Kinovea y añadirse con tiempos sincronizables. |
| Velocidad de la mano | Velocidad lineal del punto definido como mano, calculada con tiempos reales o mediante el Jacobiano validado. | Hay variables lineales derivadas para algunos puntos, pero sus unidades y sincronía deben verificarse antes de usarlas. |
| Posición de la pelota \((x,y)\) | Coordenadas del centro de la pelota en cada fotograma y en un sistema espacial calibrado. | No se identificaron coordenadas de pelota en los CSV actuales; deben digitalizarse y vincularse a fotogramas/tiempos. |
| Velocidad de la pelota | Cambio de posición del centro de la pelota por unidad de tiempo, en un sistema calibrado. | No se identificó una serie de trayectoria/velocidad de pelota en los CSV actuales; debe añadirse el seguimiento del balón y verificarse su resolución temporal. |
| Efecto/spin de la pelota | Rotación, trayectoria o indicador equivalente de efecto. | No aparece medido directamente en los datos actuales. Requiere un método de seguimiento adicional y una definición cuantitativa. |

Los ángulos disponibles en las bases están en **grados**. Las variables `ABS` representan ángulos absolutos y `MA` promedios móviles. Para las ecuaciones de dinámica inversa que requieren radianes habrá que convertir los ángulos y sus derivadas angulares, después de validar sus unidades y definiciones. Más detalles de encabezados, conteos y disponibilidad por archivo están en [`contexto_bases_datos.md`](./contexto_bases_datos.md).

### Interpretación estática y mecánica del sólido continuo

Para algunas posturas clave (por ejemplo, carga, máxima flexión/armado, contacto y seguimiento), se identificarán los segmentos, las articulaciones y los brazos de palanca visibles en la proyección 2D. El brazo de palanca de una fuerza es la distancia perpendicular entre su línea de acción y el eje/punto articular considerado; cualitativamente, una mayor distancia puede aumentar el momento si la fuerza y las demás condiciones se mantienen. La postura por sí sola no determina el torque: sin medir o estimar la fuerza externa no se calculará su magnitud ni se afirmará que una postura produce una carga específica. Esta parte es una interpretación mecánica cualitativa.

La conexión conceptual con la mecánica del sólido continuo consiste en reconocer que el cuerpo está formado por tejidos con masa distribuida y que, para el análisis cinemático/dinámico, se idealiza mediante segmentos y articulaciones. La distribución de masa se resume en parámetros como masa e inercia en un modelo de cuerpo rígido; deformaciones, tensiones y esfuerzos internos de los tejidos no se obtienen de estos videos ni forman parte de esta etapa.

## Pipeline analítico propuesto en Python

1. **Catalogar y validar entradas:** leer CSV, conservar los archivos originales, registrar participante/sesión/intento/vista, unidades, columnas, valores faltantes y procedencia.
2. **Calibrar y normalizar coordenadas:** convertir píxeles a unidades físicas con la referencia de 1 m, definir ejes y origen por cámara y documentar las limitaciones de perspectiva.
3. **Digitalizar puntos:** registrar puntos anatómicos, puntos de la raqueta y centro de la pelota en los fotogramas relevantes o a lo largo de la serie, con identificadores y tiempos consistentes.
4. **Sincronizar series:** comprobar tiempos y eventos entre exportaciones de una misma vista; resolver offsets y marcas repetidas con evidencia del video. No unir por índice de fila. Registrar cualquier corrección y conservar la marca temporal original.
5. **Controlar calidad del tracking:** revisar continuidad y plausibilidad de los puntos, oclusiones, saltos y errores de etiquetado. Definir explícitamente el tratamiento de muestras inválidas.
6. **Filtrar y derivar:** aplicar un método de suavizado documentado y reproducible; calcular velocidades/aceleraciones lineales y angulares a partir del tiempo real. Evitar derivar datos ruidosos sin evaluar el efecto del filtro.
7. **Calcular variables cinemáticas:** obtener posiciones y ángulos articulares y de la raqueta con convenciones explícitas; generar \(\theta(t)\), \(\omega(t)\), \(\alpha(t)\), posiciones y velocidades de mano.
8. **Segmentar el saque:** marcar eventos biomecánicos visibles y definir las fases y sus límites antes de comparar intentos. El impacto/contacto con la pelota puede servir como referencia temporal si se identifica con fiabilidad; analizar el timing y el orden de los eventos.
9. **Analizar posturas clave:** seleccionar fotogramas representativos, identificar brazos de palanca y elaborar la discusión cualitativa de momentos/torques, indicando qué fuerzas se desconocen.
10. **Analizar pelota:** obtener coordenadas por fotograma y, si hay escala, sincronía y resolución suficientes, estimar desplazamiento y velocidad. Informar incertidumbre y no confundir velocidad de salida con efecto/spin.
11. **Resumir y validar:** producir curvas de posición angular, velocidad y aceleración por intento/fase, indicadores de timing y consistencia; contrastar cálculos con inspección visual o mediciones de referencia e informar unidades, filtros, datos excluidos y limitaciones.
12. **Conectar con el modelo físico:** explicar conceptualmente la idealización de segmentos rígidos frente a un cuerpo continuo; no inferir tensiones/deformaciones tisulares sin mediciones y modelos adecuados.

El modelo \(\mathbf{v}=J(\boldsymbol{\theta})\boldsymbol{\omega}\) debe relacionar una cadena articular y un punto terminal definidos. La velocidad resultante de la mano estimada por este modelo no sustituye la medición independiente de la velocidad de la pelota.

## Alcance e interpretación

### Etapa actual: cinemática 2D

El resultado principal esperado es un conjunto de curvas temporales de posición angular, velocidad angular y aceleración angular, además de trayectorias digitalizadas de puntos anatómicos, raqueta y pelota, organizado por intento y fase. Se incluirán análisis de timing, consistencia y una interpretación cualitativa de brazos de palanca en posturas clave. La comparación entre cinco intentos debe tratarse como exploratoria, especialmente si el número de jugadores también es pequeño o todavía no está definido.

Las mediciones 2D describen proyecciones en el plano de cada cámara. No capturan necesariamente movimientos fuera del plano, fuerzas internas, momentos articulares, carga tisular ni evolución de lesiones. Las asociaciones observadas no demuestran causalidad ni permiten afirmar que el sistema previene lesiones a largo plazo.

### Etapa futura: fuerzas y dinámica inversa

La evolución hacia fuerzas y torques requiere, como mínimo, cinemática segmentaria sincronizada, parámetros antropométricos por participante y fuerzas externas (por ejemplo, reacción del suelo medida con plataforma). En los CSV inventariados no se encontró una serie de fuerzas de plataforma ni de velocidad de pelota. El documento [`contexto_dinamica_inversa.md`](./contexto_dinamica_inversa.md) describe un modelo 2D de extremidad inferior, pero sus supuestos, parámetros y aplicabilidad al saque kick deben verificarse antes de presentar sus salidas como estimaciones válidas.

Los parámetros antropométricos de un sujeto descritos allí no deben generalizarse a todos los participantes: deben obtenerse o justificarse para cada persona. Si se busca estudiar cargas de hombro, codo o tronco, el modelo de extremidad inferior no basta y hará falta definir modelos segmentarios y mediciones apropiadas para esas regiones.

## Limitaciones y decisiones pendientes

- Confirmar si los cinco intentos son por participante o en total y registrar cuántos jugadores participan.
- Definir qué lado realiza cada acción y cómo se identifica la pierna delantera.
- Precisar marcadores, convenciones de ángulos, sistemas de coordenadas y definición de cada fase.
- Añadir a Kinovea y exportar puntos de la raqueta y coordenadas de pelota; decidir qué puntos anatómicos se seguirán en cada vista.
- Obtener series sincronizadas de \(\theta(t)\), \(\omega(t)\) y \(\alpha(t)\) para todas las articulaciones que se incluyan en los resultados.
- Seleccionar posturas para la interpretación estática y documentar las fuerzas desconocidas; limitar las conclusiones de torque a lo cualitativo mientras no se midan fuerzas.
- Resolver los desfasajes temporales, diferencias de duración y tiempos repetidos ya detectados en los CSV existentes; no todas las series son directamente sincronizables.
- Confirmar las unidades de las variables derivadas y el método de filtrado usado en exportaciones previas.
- Incorporar y validar trayectoria de pelota para estimar velocidad; medir spin/efecto si se desea contrastar esa parte de la hipótesis.
- Asegurar la sincronización entre las vistas y evaluar oclusiones, perspectiva y error de seguimiento.
- Tratar la prevención de lesiones como motivación y aplicación futura; medir lesiones o cargas requiere un diseño específico y no puede inferirse directamente de curvas cinemáticas.
- Formalizar las hipótesis estadísticas y definir métricas de rendimiento/eficacia antes de probar asociaciones.

## Resultado esperado del proyecto

Un pipeline reproducible en Python que convierta videos y exportaciones de Kinovea en series cinemáticas calibradas, sincronizadas y documentadas; describa patrones por fase e intento; explore relaciones entre coordinación corporal y velocidad de la pelota cuando esta se mida; y deje claros los requisitos y límites para una futura extensión a dinámica inversa y análisis de cargas biomecánicas.
