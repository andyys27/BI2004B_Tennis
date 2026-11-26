# Contexto Teórico, Matemático y Algorítmico: Dinámica Inversa de Extremidad Inferior

Este documento describe la metodología analítica y computacional empleada para determinar las fuerzas internas y los torques articulares en la extremidad inferior (tobillo, rodilla y cadera) durante una maniobra dinámica con fase de contacto y fase aérea (brinco), utilizando un enfoque de **Dinámica Inversa en 2D (plano sagital)**.

> **Estado del modelo:** las ecuaciones cerradas de este documento son la formulación inicial y no deben confundirse con la implementación actual de M12. El módulo [`analysis/inverse_dynamics.py`](../src/saque/analysis/inverse_dynamics.py) usa balances vectoriales de Newton-Euler por segmento. La implementación actual (`run_serve_inverse_dynamics`, por saque) asume GRF=0 (vuelo) durante toda la ventana del saque, por lo que sus resultados no representan las cargas reales durante el apoyo. El tren superior es un cálculo 2D exploratorio que supone fuerza externa nula en muñeca y excluye mano, raqueta y pelota. Ninguno de estos modos sustituye la validación con las cargas externas relevantes medidas.

---

## 1. Fundamento del Proyecto

El objetivo de la **dinámica inversa** es calcular las fuerzas netas y los momentos (torques) musculares que actúan en cada articulación a partir de dos fuentes de datos experimentales:
1. **Cinemática:** Posición angular ($\theta$), velocidad angular ($\omega$) y aceleración angular ($\alpha$) de cada segmento corporal en el tiempo.
2. **Cinética externa:** Fuerzas de reacción del suelo ($GRF$) medidas mediante plataformas de fuerza.

El modelo sigue un enfoque recursivo **de distal a proximal** (de abajo hacia arriba): se inicia en el punto de contacto con el entorno (el pie) y se avanza secuencialmente hacia la pierna (rodilla) y el muslo (cadera), resolviendo las ecuaciones de equilibrio dinámico de Newton-Euler para cada cuerpo rígido.

---

## 2. Variables Involucradas

### A. Variables Cinemáticas y Cinéticas (Entrada desde Excel)
* $\theta_1, \theta_2, \theta_3$: Ángulos absolutos (en radianes) de los segmentos pie, pierna y muslo respecto a la horizontal.
* $\omega_1, \omega_2, \omega_3$ ($\dot{\theta}_i$): Velocidades angulares ($\text{rad/s}$) de cada segmento.
* $\alpha_1, \alpha_2, \alpha_3$ ($\ddot{\theta}_i$): Aceleraciones angulares ($\text{rad/s}^2$) de cada segmento.
* $GRF_x$: Componente horizontal (anteroposterior) de la fuerza de reacción del suelo ($\text{N}$).
* $vGRF$: Componente vertical de la fuerza de reacción del suelo ($\text{N}$).

### B. Parámetros Antropométricos y Geométricos (Constantes del Sujeto)
Estimados mediante las tablas antropométricas de **David A. Winter** para una atleta femenina de $1.64\text{ m}$ y masa total $M = 58.0\text{ kg}$:
* $l_1, l_2, l_3$: Longitudes anatómicas de pie, pierna y muslo ($\text{m}$).
* $m_1, m_2, m_3$: Masas de los segmentos ($\text{kg}$), calculadas como porcentaje del peso total ($1.45\%$, $4.65\%$ y $10.0\%$ respectivamente).
* $r_1, r_2, r_3$: Distancia desde la articulación proximal hasta el centro de masa (CoM) de cada segmento ($\text{m}$).
* $d_i$: Distancia complementaria desde el centro de masa hasta la articulación distal ($d_i = l_i - r_i$).
* $I_1, I_2, I_3$: Momentos de inercia de cada segmento respecto a su centro de masa ($\text{kg}\cdot\text{m}^2$), calculados mediante el radio de giro $\rho_i$ ($I_i = m_i \rho_i^2$).
* $h$: Altura efectiva del centro de presión en el pie ($\text{m}$).
* $g$: Aceleración gravitacional ($9.81\text{ m/s}^2$).

### C. Variables Incógnitas (Salida del Algoritmo)
* $F_{x1}, F_{z1}, T_1$: Fuerzas horizontal, vertical ($\text{N}$) y torque articular ($\text{N}\cdot\text{m}$) en el **tobillo**.
* $F_{x2}, F_{z2}, T_2$: Fuerzas horizontal, vertical ($\text{N}$) y torque articular ($\text{N}\cdot\text{m}$) en la **rodilla**.
* $F_{x3}, F_{z3}, T_3$: Fuerzas horizontal, vertical ($\text{N}$) y torque articular ($\text{N}\cdot\text{m}$) en la **cadera**.

---

## 3. Procedimiento Analítico y Ecuaciones Base

El análisis se rige por las ecuaciones fundamentales de movimiento de Newton-Euler en el plano $X-Z$:

$$\sum F_x = m \ddot{x}_{CoM} \qquad \sum F_z = m \ddot{z}_{CoM} \qquad \sum M_{CoM} = I \alpha$$

### Nivel 1: Pie y Articulación del Tobillo
Se evalúa en dos regímenes físicos dependiendo del contacto con el suelo:

1. **Fase de Apoyo ($vGRF > \text{umbral}$):**
   El pie interactúa con el suelo y transmite las fuerzas de reacción al tobillo (despreciando la masa del pie frente a la magnitud de la $GRF$ o considerándolo en equilibrio cuasi-estático):
   $$F_{x1} = GRF_x$$
   $$F_{z1} = vGRF$$
   $$T_1 = GRF_x \cdot l_1 - vGRF \cdot h$$

2. **Fase Aérea / Despegue ($vGRF \le \text{umbral}$):**
   Al perder contacto con la superficie, desaparecen las fuerzas externas:
   $$GRF_x = 0, \quad vGRF = 0 \implies F_{x1} = 0, \quad F_{z1} = 0, \quad T_1 = 0$$

---

### Nivel 2: Pierna y Articulación de la Rodilla
Se deriva la posición del centro de masa de la pierna respecto al origen para obtener sus aceleraciones lineales ($\ddot{x}_2, \ddot{z}_2$):

$$\ddot{x}_2 = -r_1 (\cos\theta_1 \omega_1^2 + \sin\theta_1 \alpha_1) - r_2 (\cos\theta_2 \omega_2^2 + \sin\theta_2 \alpha_2)$$
$$\ddot{z}_2 = -r_1 (\sin\theta_1 \omega_1^2 - \cos\theta_1 \alpha_1) - r_2 (\sin\theta_2 \omega_2^2 - \cos\theta_2 \alpha_2)$$

Aplicando el equilibrio de fuerzas y momentos sobre el segmento:

$$F_{x2} = F_{x1} - m_2 \ddot{x}_2$$
$$F_{z2} = F_{z1} - m_2 g - m_2 \ddot{z}_2$$
$$T_2 = T_1 + F_{x1} r_1 \cos\theta_1 + F_{z1} r_1 \sin\theta_1 - F_{z2} d_2 \sin\theta_1 + F_{x2} d_2 \cos\theta_1 - I_2 \alpha_2$$

---

### Nivel 3: Muslo y Articulación de la Cadera (Fin de Cadena)
Se calcula la aceleración lineal del centro de masa del muslo ($\ddot{x}_3, \ddot{z}_3$) propagando la cinemática desde el tobillo y la rodilla:

$$\ddot{x}_3 = -l_1 \cos\theta_1 \omega_1^2 - l_1 \sin\theta_1 \alpha_1 - l_2 \cos\theta_2 \omega_2^2 - l_2 \sin\theta_2 \alpha_2 - r_3 (\cos\theta_3 \omega_3^2 + \sin\theta_3 \alpha_3)$$
$$\ddot{z}_3 = -l_1 \sin\theta_1 \omega_1^2 + l_1 \cos\theta_1 \alpha_1 - l_2 \sin\theta_2 \omega_2^2 + l_2 \cos\theta_2 \alpha_2 - r_3 (\sin\theta_3 \omega_3^2 - \cos\theta_3 \alpha_3)$$

Al cerrar la cadena cinemática en la cadera (sin incluir el tronco), se obtienen las fuerzas y el torque neto ejercidos sobre la articulación coxofemoral:

$$F_{x3} = F_{x2} - m_3 \ddot{x}_3$$
$$F_{z3} = F_{z2} - m_3 g - m_3 \ddot{z}_3$$
$$T_3 = T_2 + F_{x2}(l_1\cos\theta_1 - l_2\cos\theta_2 + r_3\cos\theta_3) - F_{z2}(l_1\sin\theta_1 - l_2\sin\theta_2 + r_3\sin\theta_3) - I_3\alpha_3$$

---

## 4. Flujo Algorítmico Computacional

El algoritmo implementado sigue una estructura secuencial discreta evaluada instante por instante ($i = 1, 2, \dots, N$):

1. **Ingesta y Preprocesamiento:**
   * Lectura tabular de series temporales cinemáticas ($\theta, \omega, \alpha$) y cinéticas ($GRF$).
   * Inicialización de parámetros antropométricos escalados a la estatura y masa del sujeto.
2. **Preasignación de Memoria:**
   * Creación de vectores columna vacíos de tamaño $N \times 1$ para optimizar el tiempo de ejecución.
3. **Iteración Temporal (Bucle Principal):**
   * *Paso 3.1 (Detección de Evento):* Evaluación condicional sobre $vGRF(i)$ para discriminar entre fase de impulso/apoyo y fase de vuelo.
   * *Paso 3.2 (Resolución Distal):* Cálculo de reacciones en el tobillo ($F_{x1}, F_{z1}, T_1$).
   * *Paso 3.3 (Propagación Cinemática-Dinámica):* Sustitución de variables en las ecuaciones de aceleración de CoM y resolución de fuerzas/torques en rodilla ($F_{x2}, F_{z2}, T_2$) y cadera ($F_{x3}, F_{z3}, T_3$).
4. **Postprocesamiento y Visualización:**
   * Generación de curvas temporales de torques y fuerzas articulares para identificar picos de carga mecánica antes y después del despegue.