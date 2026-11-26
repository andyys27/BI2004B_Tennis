# Del sólido continuo al modelo de segmentos rígidos

Este documento cubre el objetivo específico 10 del proyecto. Explica tres cosas:
- cómo se relaciona el análisis de segmentos corporales con la mecánica del sólido continuo;
- qué se pierde al idealizar el cuerpo;
- qué haría falta para pasar de las estimaciones actuales a cargas en los tejidos.

## 1. El cuerpo como medio continuo

Mecánicamente, el cuerpo es un **medio continuo deformable** con varios tejidos de propiedades muy
distintas:

| Tejido | Comportamiento mecánico | Rigidez aproximada |
|---|---|---|
| Hueso cortical | sólido elástico anisótropo, casi rígido a estas cargas | E ≈ 15–20 GPa |
| Tendón y ligamento | viscoelástico, no lineal; trabaja a tracción | E ≈ 1–2 GPa |
| Cartílago | poroelástico, bifásico (sólido + fluido) | E ≈ 1–15 MPa |
| Músculo | activo: genera fuerza, y su rigidez depende de la activación | variable |
| Grasa y piel | blandos, viscoelásticos | kPa |

En un continuo, cada punto material tiene su propio desplazamiento **u(x, t)**. Su estado se describe
con campos:
- **deformación** ε = ½(∇u + ∇uᵀ);
- **tensión** σ, relacionada con ε por la ley constitutiva de cada tejido;
- **equilibrio local**: ∇·σ + ρb = ρü en cada punto del volumen.

Resolver esas ecuaciones da las tensiones dentro de un tendón o del cartílago de la rodilla. Para
eso hacen falta la geometría de cada tejido (imagen médica), sus propiedades y las cargas en sus
fronteras. Se resuelve con elementos finitos.

## 2. La idealización usada: segmentos rígidos articulados

El pipeline no resuelve el continuo. Lo **idealiza** como una cadena de segmentos rígidos (pie,
pierna, muslo, tronco, brazo, antebrazo) unidos por articulaciones ideales (bisagras en 2D). La
idealización se apoya en dos hechos:

1. **El hueso es mucho más rígido que las cargas que recibe.** En el saque, la deformación del hueso
   es del orden de 0.1 % (≈ 1000 µε). Sobre una pierna de ~0.40 m eso es ≈ 0.4 mm, frente a
   desplazamientos de cientos de milímetros entre segmentos. Tratar cada segmento como rígido es un
   error despreciable para la cinemática.
2. **Las ecuaciones del continuo se integran sobre el volumen de cada segmento.** Al integrar el
   equilibrio local en todo el segmento, las tensiones internas se cancelan por pares (acción y
   reacción) y solo quedan las resultantes en sus fronteras. Esas son las ecuaciones de
   **Newton–Euler** que usa `analysis/inverse_dynamics.py`:

   - ΣF = m·a_CoM (traslación del centro de masa del segmento)
   - ΣM_CoM = I·α (rotación alrededor de su centro de masa)

   Toda la distribución de masa del continuo queda resumida en cuatro parámetros por segmento:
   **masa, posición del centro de masa, longitud y momento de inercia**. En el proyecto salen de las
   fracciones de Winter (`inverse_dynamics.anthropometry`).

El centro de masa del cuerpo completo (`analysis/body.py`) aplica la misma idea a todo el cuerpo: es
la media ponderada de los centros de masa de 15 segmentos. La fuerza estimada del suelo,
F = m(a_CoM + g), es la segunda ley de Newton para el sistema completo. Las fuerzas internas
(músculos y articulaciones) se cancelan y solo quedan el peso y la reacción del suelo.

## 3. Qué representan y qué no los resultados del proyecto

| Magnitud del proyecto | Lo que es en el continuo | Lo que NO es |
|---|---|---|
| Ángulo articular θ, ω, α | rotación relativa entre dos cuerpos rígidos | deformación de los tejidos |
| Momento articular neto (p. ej. rodilla ≈ 167 N·m en el empuje) | resultante de **todas** las tensiones que cruzan la articulación (músculos, ligamentos, contacto óseo), reducida a un momento | la fuerza de un músculo o la tensión en un ligamento |
| Fuerza articular neta | resultante de fuerzas en la frontera del segmento | la fuerza de contacto entre cartílagos (que es mayor, porque los músculos co-contraen) |
| Potencia articular | flujo neto de energía mecánica a través de la articulación | el trabajo de un músculo concreto |
| Brazo de palanca del peso | geometría de la línea de acción respecto al eje articular | el brazo de palanca de un tendón |

En particular:
- **Un mismo momento neto lo producen infinitas combinaciones de fuerzas musculares.** Es el
  problema de redundancia muscular. El momento neto no dice cuánto trabaja cada músculo ni cuánta
  co-contracción hay.
- **La fuerza de contacto en la rodilla es varias veces mayor que la fuerza neta**, porque los
  extensores y los flexores tiran a la vez. Las cargas en el cartílago no se pueden leer de estos
  resultados.
- **La tensión (N/m²) en un tejido** requiere además su geometría (área de sección, forma) y su ley
  constitutiva. Ninguna de las dos sale del video.

## 4. Supuestos de la idealización y su efecto en este proyecto

| Supuesto | Realidad | Efecto |
|---|---|---|
| Segmentos rígidos | los tejidos blandos se mueven respecto al hueso (masas oscilantes, artefacto de piel) | ruido en las aceleraciones; sobreestima los picos de α y de los momentos del brazo de golpeo |
| Articulaciones como bisagras en 2D | el hombro es una articulación esférica y la rodilla rueda y desliza | los ángulos 2D son proyecciones; el hombro no se puede modelar bien en un plano |
| Parámetros de Winter | proceden de cadáveres de varones adultos | error de ~5–10 % en masas e inercias para esta jugadora |
| Fuerza del suelo estimada por el CoM | sin plataforma: reparto 50/50 entre pies y centro de presión supuestos | los momentos del tren inferior son estimaciones exploratorias |
| Sin raqueta | la raqueta (~0.3 kg en el extremo) carga la muñeca | subestima los momentos del codo y el hombro cerca del impacto |

## 5. Camino hacia las cargas en los tejidos

De menor a mayor esfuerzo:

1. **Fuerzas externas medidas.** Plataforma de fuerza sincronizada (la GRF y el centro de presión
   dejan de ser supuestos) y raqueta instrumentada o modelada como un segmento más.
2. **Cinemática 3D.** Dos o más cámaras calibradas y sincronizadas, o captura óptica con marcadores,
   para modelar el hombro y la rotación del tronco fuera del plano.
3. **Modelo músculo-esquelético** (por ejemplo OpenSim). Reparte los momentos netos entre músculos por
   optimización o con EMG, y da las fuerzas musculares y de contacto articular.
4. **Modelo de elementos finitos** de la región de interés (rodilla, hombro o codo). Toma como
   condiciones de frontera las fuerzas del paso 3 y resuelve el continuo para obtener las tensiones
   y deformaciones en cartílago, ligamentos o tendones. Requiere imagen médica (RM o TC) del sujeto
   y propiedades de cada tejido.

Solo en el paso 4 se vuelve al sólido continuo con el que empieza este documento, y solo ahí se
podría hablar de carga en el tejido y, con estudios longitudinales, de riesgo de lesión. Este
proyecto llega a la etapa previa al paso 1: cinemática 2D y dinámica inversa con fuerzas externas
estimadas.
