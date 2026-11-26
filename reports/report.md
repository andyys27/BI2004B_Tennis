# Reporte: análisis cinemático del saque kick (MediaPipe + estampitas)

Generado automáticamente por `saque run`. Cinemática 2D por vista; todas las asociaciones y tendencias entre saques son exploratorias (n = 5 saques, una jugadora).

## 1. Datos y calibración
| vista | metodo | m_por_px | px_por_m | origen_x_px | origen_y_px | estatura_aparente_m | estatura_real_m | fps | n_fotogramas | fotogramas_perdidos | frac_detectado |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lateral | ground_plane | 0.00167 | 599 | 443 | 1.78e+03 | 1.58 | 1.64 | 120 | 5414 | 24 | 0.996 |
| posterior | ground_plane | 0.00151 | 662 | 768 | 1.6e+03 | 1.59 | 1.64 | 120 | 5588 | 15 | 0.997 |

## 2. Origen de los marcadores
Cada marcador usa el centroide de la estampita cuando se detecta y, si no, el landmark de MediaPipe corregido por el desfase mediano estampita-landmark.

| vista | marcador | frac_estampita | frac_pose | frac_hueco | estampitas_descartadas |
|---|---|---|---|---|---|
| lateral | hombro_der | 0.962 | 0.0342 | 0.00405 | 318 |
| lateral | codo_der | 0.714 | 0.246 | 0.0397 | 776 |
| lateral | muneca_der | 0.801 | 0.144 | 0.0548 | 308 |
| lateral | cadera_der | 0.63 | 0.366 | 0.00441 | 78 |
| lateral | rodilla_der | 0.737 | 0.254 | 0.00938 | 252 |
| lateral | tobillo_der | 0.717 | 0.257 | 0.0256 | 629 |
| lateral | punta_der | 0.688 | 0.266 | 0.0458 | 598 |
| posterior | cadera_izq | 0.91 | 0.0871 | 0.00268 | 53 |
| posterior | rodilla_izq | 0.855 | 0.112 | 0.0328 | 275 |
| posterior | tobillo_izq | 0.641 | 0.318 | 0.0409 | 322 |
| posterior | tobillo_der | 0.515 | 0.455 | 0.0303 | 702 |

![fuentes](figures/fuente_marcadores.png)

## 3. Saques detectados
| vista | saque | t_ini | t_impacto | t_fin |
|---|---|---|---|---|
| lateral | 1 | 1.99 | 4.99 | 5.79 |
| lateral | 2 | 10.1 | 13.1 | 13.9 |
| lateral | 3 | 19.8 | 22.8 | 23.6 |
| lateral | 4 | 28.2 | 31.2 | 32 |
| lateral | 5 | 38.2 | 41.2 | 42 |
| posterior | 1 | 2.64 | 5.64 | 6.44 |
| posterior | 2 | 10.7 | 13.7 | 14.5 |
| posterior | 3 | 20.5 | 23.5 | 24.3 |
| posterior | 4 | 28.9 | 31.9 | 32.7 |
| posterior | 5 | 38.8 | 41.8 | 42.6 |

Sincronización entre vistas: t_vista = t_referencia + lag_s (por correlación de la altura de cadera). El control compara con los impactos que la vista detecta por su cuenta con la muñeca:

| vista | referencia | lag_s | correlacion_cadera | control_diferencia_muneca_mediana_s | control_dispersion_s |
|---|---|---|---|---|---|
| posterior | lateral | 0.642 | 0.751 | -0.132 | 0.183 |

![deteccion_saques](figures/deteccion_saques.png)

![impactos_lateral](figures/impactos_lateral.png)

![impactos_posterior](figures/impactos_posterior.png)

## 4. Fases (duración en ms)
| vista | saque | aceleracion | carga | preparacion | seguimiento |
|---|---|---|---|---|---|
| lateral | 1 | 475 | 524 | 1257 | 258 |
| lateral | 2 | 433 | 574 | 1356 | 241 |
| lateral | 3 | 441 | 541 | 1323 | 258 |
| lateral | 4 | 550 | 391 | 1348 | 316 |
| lateral | 5 | 533 | 383 | 1356 | 308 |
| posterior | 1 | 475 | 524 | 1257 | 258 |
| posterior | 2 | 433 | 574 | 1356 | 241 |
| posterior | 3 | 441 | 541 | 1323 | 258 |
| posterior | 4 | 550 | 391 | 1348 | 316 |
| posterior | 5 | 533 | 383 | 1356 | 308 |

![fases](figures/fases.png)

## 5. Secuencia próximo-distal
Pico de |ω| (o rapidez de muñeca) de cada segmento entre el inicio y 50 ms después del impacto, ignorando huecos; una serie con menos del 50 % de muestras válidas en la ventana se omite. `orden_observado` ordena los picos en el tiempo ('=' = mismo fotograma). `lags_ms` son las diferencias entre picos consecutivos del orden esperado (negativo = invertido). τ de Kendall = 1 indica el orden esperado.

| vista | saque | orden_observado | tau_kendall | n_invertidos | lags_ms | nota |
|---|---|---|---|---|---|---|
| lateral | 1 | brazo > tronco > codo > muneca > muslo | 0 | 2 | -441;-1490;1806;83 | nan |
| lateral | 2 | muslo = tronco > brazo = codo > muneca | 0.894 | 0 | 0;42;0;58 | 2 pico(s) simultáneo(s) |
| lateral | 3 | muslo > tronco > brazo = codo = muneca | 0.837 | 0 | 92;158;0;0 | 2 pico(s) simultáneo(s) |
| lateral | 4 | brazo = codo > muslo > muneca > tronco | -0.105 | 1 | 250;-258;0;108 | 1 pico(s) simultáneo(s) |
| lateral | 5 | tronco > muslo > codo > brazo > muneca | 0.6 | 2 | -208;266;-8;58 | nan |

![secuencia](figures/secuencia_cinematica.png)

## 6. Curvas por saque (inicio → impacto)
![curvas](figures/curvas_por_saque.png)

## 6b. Cinemática angular θ, ω, α por articulación (vista lateral)
Ángulos interiores (180° = articulación extendida), normalizados de inicio a impacto. La banda gris es la media ± DE de los 5 saques. α es la segunda derivada del ángulo filtrado y es la magnitud más ruidosa.

![θωα](figures/cinematica_articular.png)

Picos de |ω| y |α| entre inicio e impacto (media ± DE de los saques; tiempo mediano respecto al impacto):

| vista | articulacion | magnitud | unidad | media | de | t_rel_impacto_ms | frac_valida |
|---|---|---|---|---|---|---|---|
| lateral | cadera | alpha | rad/s² | 547 | 99.3 | -292 | 0.979 |
| lateral | cadera | omega | rad/s | 9.25 | 2.22 | -167 | 0.979 |
| lateral | codo | alpha | rad/s² | 3.6e+03 | 2.31e+03 | -158 | 0.966 |
| lateral | codo | omega | rad/s | 37.8 | 18.8 | -142 | 0.966 |
| lateral | hombro | alpha | rad/s² | 3.51e+03 | 2.42e+03 | -158 | 0.962 |
| lateral | hombro | omega | rad/s | 31.7 | 16.9 | -150 | 0.962 |
| lateral | rodilla | alpha | rad/s² | 394 | 138 | -100 | 0.994 |
| lateral | rodilla | omega | rad/s | 9.6 | 0.602 | -192 | 0.994 |
| lateral | tobillo | alpha | rad/s² | 838 | 263 | -41.9 | 0.994 |
| lateral | tobillo | omega | rad/s | 13.8 | 11 | -42 | 0.994 |
| posterior | codo | alpha | rad/s² | 680 | 301 | -59.5 | 0.466 |
| posterior | codo | omega | rad/s | 18.5 | 10.5 | -67.8 | 0.466 |
| posterior | rodilla | alpha | rad/s² | 358 | 64.7 | -143 | 0.971 |
| posterior | rodilla | omega | rad/s | 5.85 | 1.14 | -209 | 0.971 |
| posterior | rodilla_izq | alpha | rad/s² | 731 | 185 | -184 | 0.838 |
| posterior | rodilla_izq | omega | rad/s | 6.85 | 0.45 | -850 | 0.838 |

## 6c. Consistencia entre saques
DE entre saques a lo largo del gesto y RMSD de cada saque respecto a la curva media (ángulos, °). Un saque con RMSD alto en todas las articulaciones fue técnicamente distinto del resto.

| variable | saque_1 | saque_2 | saque_3 | saque_4 | saque_5 |
|---|---|---|---|---|---|
| theta_cadera | 4.3 | 3.1 | 2.8 | 2.8 | 2.2 |
| theta_codo | 10.5 | 6.3 | 7.0 | 8.2 | 5.5 |
| theta_hombro | 8.0 | 5.1 | 6.6 | 6.0 | 5.0 |
| theta_incl_tronco | 3.7 | 2.4 | 2.6 | 4.1 | 2.1 |
| theta_rodilla | 5.5 | 3.1 | 1.3 | 3.0 | 2.7 |
| theta_tobillo | 8.1 | 2.6 | 2.1 | 3.6 | 3.9 |

![variabilidad](figures/variabilidad_saques.png)

## 6d. Asociaciones entre indicadores por saque
ρ de Spearman entre los indicadores de la sección 7 y las variables del centro de masa. Con n = 5 solo |ρ| = 1 (o 0.9) llega a p < 0.05; se reporta como descripción, sin corrección por comparaciones múltiples y sin interpretación causal.

Diez pares con mayor |ρ|:

| var_a | var_b | n | rho | p |
|---|---|---|---|---|
| elevacion_max_com_m | altura_contacto_m | 5 | 0.9 | 0.0374 |
| duracion_aceleracion | elevacion_max_com_m | 5 | -0.9 | 0.0374 |
| duracion_aceleracion | duracion_carga | 5 | -0.9 | 0.0374 |
| altura_max_muneca | elevacion_max_com_m | 5 | 0.9 | 0.0374 |
| duracion_carga | elevacion_max_com_m | 5 | 0.8 | 0.104 |
| duracion_aceleracion | altura_contacto_m | 5 | -0.8 | 0.104 |
| altura_max_muneca | rapidez_pico_muneca | 5 | 0.8 | 0.104 |
| altura_max_muneca | duracion_aceleracion | 5 | -0.7 | 0.188 |
| flexion_max_rodilla | vy_max_com_m_s | 5 | 0.7 | 0.188 |
| rapidez_pico_muneca | grf_pico_bw | 5 | -0.7 | 0.188 |

![correlaciones](figures/correlaciones.png)

## 7. Indicadores de posible fatiga
Pendiente por saque, cambio del primer al último saque y ρ de Spearman con IC95 % bootstrap. Con 5 saques los intervalos son amplios: una tendencia es una señal a confirmar, no una prueba de fatiga (también puede reflejar cambios de intención o técnica).

| variable | confianza | unidad | media | cv_pct | pendiente_por_saque | cambio_primero_ultimo_pct | rho_spearman | ic95_inf | ic95_sup |
|---|---|---|---|---|---|---|---|---|---|
| rapidez_pico_muneca | media | m/s | 7.64 | 29.2 | -0.994 | -43.3 | -0.5 | -1 | 1 |
| omega_pico_codo | baja | rad/s | 48.6 | 28.8 | 0.396 | -6.98 | -0.5 | -1 | 1 |
| omega_pico_tronco | baja | rad/s | 4.75 | 4.56 | -0.129 | -8.42 | -1 | -1 | -1 |
| omega_pico_hombro | baja | rad/s | 40.3 | 37.3 | -3.75 | -29.6 | -0.5 | -1 | 1 |
| flexion_max_rodilla | alta | rad | 1.83 | 1.14 | 0.0112 | 3.22 | 0.7 | -0.667 | 1 |
| elevacion_max_hombro | media | rad | 2.83 | 12 | 0.00321 | -3.86 | -0.5 | -1 | 1 |
| altura_max_muneca | alta | m | 1.79 | 1.03 | -0.00538 | -0.699 | -0.6 | -1 | 1 |
| duracion_carga | alta | ms | 483 | 18.5 | -46.6 | -27 | -0.7 | -1 | 0.667 |
| duracion_aceleracion | alta | ms | 486 | 10.9 | 23.3 | 12.3 | 0.6 | -1 | 1 |

Confianza: *alta* = posiciones, ángulos y tiempos; *baja* = picos de velocidad angular del brazo, sensibles a oclusiones y escorzo 2D cerca del impacto (ver `tables/fatiga_valores_por_saque.csv`, columna `frac_valida_pre_impacto`).

Índice de fatiga (último tercio vs primer tercio, variables de desempeño):

| variable | primer_tercio | ultimo_tercio | cambio_pct | saques_por_tercio |
|---|---|---|---|---|
| rapidez_pico_muneca | 11.4 | 6.48 | -43.3 | 1 |
| omega_pico_codo | 58.6 | 54.5 | -6.98 | 1 |
| omega_pico_tronco | 5 | 4.57 | -8.42 | 1 |
| omega_pico_hombro | 55.8 | 39.3 | -29.6 | 1 |

![fatiga](figures/fatiga_tendencias.png)

## 8. Jacobiano v = J(θ)ω
| vista | saque | L_brazo | cv_brazo | L_antebrazo | cv_antebrazo | rmse_rel_pico | corr_speed | pico_speed_directa |
|---|---|---|---|---|---|---|---|---|
| lateral | 1 | 0.225 | 0.238 | 0.198 | 0.172 | 0.032 | 0.897 | 11.4 |
| lateral | 2 | 0.22 | 0.236 | 0.186 | 0.209 | 0.135 | 0.906 | 6.37 |
| lateral | 3 | 0.233 | 0.208 | 0.199 | 0.205 | 0.0956 | 0.848 | 8.29 |
| lateral | 4 | 0.232 | 0.222 | 0.2 | 0.189 | 0.0622 | 0.912 | 6.17 |
| lateral | 5 | 0.228 | 0.234 | 0.196 | 0.176 | 0.0892 | 0.867 | 6.48 |

![jacobiano_validacion](figures/jacobiano_validacion.png)

![jacobiano_lateral](figures/jacobiano_lateral.png)

![dinamica_inversa_tren_superior_lateral](figures/dinamica_inversa_tren_superior_lateral.png)

![dinamica_inversa_tren_inferior_lateral](figures/dinamica_inversa_tren_inferior_lateral.png)

## 9. Cuerpo completo: centro de masa, salto y fuerza del suelo estimada
Fuerza vertical estimada F = m(a_CoM + g) (suma de ambos pies). Fin del impulso = máxima velocidad vertical del CoM. El tiempo de vuelo no se reporta: con marcadores 2D la altura de los pies no es fiable.

| vista | saque | altura_contacto_m | elevacion_max_com_m | vy_max_com_m_s | grf_pico_bw | impulso_neto_Ns | v_por_impulso_m_s | separacion_tobillos_inicio_m | separacion_tobillos_carga_m | desplazamiento_x_com_m |
|---|---|---|---|---|---|---|---|---|---|---|
| lateral | 1 | 2.09 | 0.195 | 1.23 | 2.12 | 76.2 | 1.31 | 0.42 | 0.031 | 0.0134 |
| lateral | 2 | 2.14 | 0.196 | 1.91 | 2.71 | 123 | 2.12 | 0.425 | 0.00715 | 0.0575 |
| lateral | 3 | 2.18 | 0.215 | 1.95 | 2.89 | 127 | 2.19 | 0.426 | 0.00988 | 0.0762 |
| lateral | 4 | 2.09 | 0.172 | 1.87 | 3.64 | 128 | 2.21 | 0.447 | 0.0279 | 0.267 |
| lateral | 5 | 2.13 | 0.181 | 2.01 | 3.27 | 133 | 2.3 | 0.452 | 0.018 | 0.187 |
| posterior | 1 | nan | 0.162 | 1.26 | 2.26 | 0.548 | 0.00945 | 0.295 | 0.0914 | -0.161 |
| posterior | 2 | nan | 0.149 | 0.883 | 2.8 | 42.1 | 0.726 | nan | 0.0863 | -0.153 |
| posterior | 3 | nan | 0.169 | 0.909 | 2.61 | 69.4 | 1.2 | 0.31 | 0.0987 | -0.112 |
| posterior | 4 | nan | 0.16 | 0.998 | 2.28 | 70.6 | 1.22 | 0.3 | 0.157 | -0.245 |
| posterior | 5 | nan | 0.162 | 1.01 | 2.47 | 73 | 1.26 | 0.307 | 0.146 | -0.185 |

![com_fuerza_suelo](figures/com_fuerza_suelo.png)

![trayectorias](figures/trayectorias.png)

## 10. Ángulos en los eventos (media de los saques, °)
| vista | variable | fin | impacto | inicio | inicio_flexion_rodilla | max_flex_rodilla |
|---|---|---|---|---|---|---|
| lateral | theta_incl_tronco | 32 | 25 | -16 | -4 | 18 |
| lateral | theta_interior_cadera | 152 | 151 | 174 | 170 | 161 |
| lateral | theta_interior_codo | 130 | 163 | 173 | 124 | 54 |
| lateral | theta_interior_hombro | 22 | 123 | 67 | 73 | 41 |
| lateral | theta_interior_rodilla | 52 | 160 | 175 | 172 | 105 |
| lateral | theta_interior_tobillo | 117 | 152 | 130 | 112 | 116 |
| posterior | theta_incl_linea_caderas | nan | -13 | -21 | -23 | -35 |
| posterior | theta_incl_linea_hombros | 175 | 59 | 11 | -10 | -35 |
| posterior | theta_incl_tronco | 43 | 28 | 4 | -9 | -20 |
| posterior | theta_interior_codo | 176 | 162 | nan | nan | 48 |
| posterior | theta_interior_rodilla | 116 | 169 | 172 | 175 | 153 |
| posterior | theta_interior_rodilla_izq | 117 | 174 | 174 | 173 | 174 |
| posterior | theta_sep_hombros_caderas | nan | 67 | 31 | 14 | 0 |

![posturas_lateral](figures/posturas_lateral.png)

![posturas_posterior](figures/posturas_posterior.png)

![plano_frontal_posterior](figures/plano_frontal_posterior.png)

## 11. Rango de movimiento inicio → impacto (°)
| vista | variable | mean | std |
|---|---|---|---|
| lateral | theta_incl_tronco | 57.3 | 7.0 |
| lateral | theta_interior_cadera | 39.3 | 5.5 |
| lateral | theta_interior_codo | 135.9 | 11.1 |
| lateral | theta_interior_hombro | 165.0 | 15.0 |
| lateral | theta_interior_rodilla | 74.7 | 1.2 |
| lateral | theta_interior_tobillo | 72.7 | 5.8 |
| posterior | theta_incl_linea_caderas | 48.6 | 8.5 |
| posterior | theta_incl_linea_hombros | 100.5 | 4.6 |
| posterior | theta_incl_tronco | 53.0 | 1.2 |
| posterior | theta_interior_codo | 153.1 | 4.5 |
| posterior | theta_interior_rodilla | 33.1 | 5.4 |
| posterior | theta_interior_rodilla_izq | 16.5 | 4.3 |
| posterior | theta_sep_hombros_caderas | 74.8 | 4.2 |

![rom](figures/rango_movimiento.png)

## 12. Brazos de palanca del peso corporal (media de los saques)
Distancia horizontal de la articulación a la vertical por el CoM; momento estático = m·g·brazo (sin fuerzas inerciales ni musculares).

| evento | articulacion | brazo_palanca_m | momento_peso_Nm |
|---|---|---|---|
| impacto | cadera_der | -0.104 | -59.1 |
| impacto | cadera_izq | 0.0143 | 8.14 |
| impacto | rodilla_der | -0.0713 | -40.6 |
| impacto | rodilla_izq | -0.0394 | -22.4 |
| impacto | tobillo_der | -0.175 | -99.3 |
| impacto | tobillo_izq | -0.0367 | -20.9 |
| max_flex_rodilla | cadera_der | 0.0327 | 18.6 |
| max_flex_rodilla | cadera_izq | -0.0854 | -48.6 |
| max_flex_rodilla | rodilla_der | -0.192 | -109 |
| max_flex_rodilla | rodilla_izq | -0.208 | -118 |
| max_flex_rodilla | tobillo_der | 0.0162 | 9.21 |
| max_flex_rodilla | tobillo_izq | 0.00419 | 2.38 |

![palancas](figures/palancas_lateral.png)

## 13. Picos de dinámica inversa y potencia articular (exploratorio)
Tren inferior con GRF estimada por el CoM, F = m(a_CoM + g), 50 % por pierna (pies juntos en la carga) y centro de presión fijo a mitad del pie; picos entre el inicio de la flexión de rodilla y el despegue (máx. velocidad vertical del CoM). Tren superior con la muñeca libre (sin raqueta): los picos del hombro están inflados por la doble derivada del brazo de golpeo. La potencia de cadera sale negativa en el empuje: el tronco rota hacia atrás (arco del saque kick) contra un momento flexor estimado; depende del supuesto del centro de presión.

| parte | variable | unidad | median | min | max |
|---|---|---|---|---|---|
| tren_inferior | P_ankle_W | W | 98.9 | 52.7 | 133.7 |
| tren_inferior | P_hip_W | W | 523.6 | 388.0 | 1674.1 |
| tren_inferior | P_knee_W | W | 841.6 | 579.5 | 1555.8 |
| tren_inferior | T_ankle_Nm | N·m | 28.7 | 21.7 | 52.8 |
| tren_inferior | T_hip_Nm | N·m | 120.6 | 96.4 | 193.7 |
| tren_inferior | T_knee_Nm | N·m | 166.7 | 127.1 | 208.4 |
| tren_superior | P_elbow_W | W | 92.2 | 44.9 | 163.1 |
| tren_superior | P_shoulder_W | W | 2632.3 | 32.8 | 6643.3 |
| tren_superior | T_elbow_Nm | N·m | 4.4 | 3.5 | 15.1 |
| tren_superior | T_shoulder_Nm | N·m | 49.6 | 7.9 | 242.8 |

![picos](figures/dinamica_inversa_picos.png)

## 13b. Filtrado y control de calidad
Butterworth de orden 2 sin desfase; la fc de cada marcador sale del análisis de residuos de Winter (punto: fc elegida, donde R(fc) cruza la extrapolación del ruido).

| vista | min | median | max |
|---|---|---|---|
| lateral | 5.0 | 8.8 | 9.4 |
| posterior | 7.3 | 8.8 | 9.7 |

![residuos](figures/filtrado_residuos.png)

Muestras anuladas por cada control (suma sobre marcadores o segmentos):

| vista | tipo | muestras |
|---|---|---|
| lateral | longitud_segmento | 854 |
| lateral | omega_implausible | 22 |
| lateral | rapidez_imposible | 885 |
| lateral | salto_tracking | 1199 |
| posterior | longitud_segmento | 1424 |
| posterior | rapidez_imposible | 347 |
| posterior | salto_tracking | 936 |

![qc](figures/qc_resumen.png)

## 14. Limitaciones
- 2D por vista: los ángulos son proyecciones en el plano de cada cámara.
- Escala: vara de 1 m corregida por profundidad suponiendo cámara nivelada a 1.30 m (ver estatura aparente en la tabla de calibración); escala fija por vista aunque la jugadora se mueve en profundidad. Los ángulos y tiempos no dependen de la escala.
- El impacto es un proxy (máxima rapidez de muñeca cerca de su altura máxima); no se rastrean raqueta ni pelota.
- Dinámica inversa exploratoria: GRF estimada por el CoM (sin plataforma; reparto 50/50 entre pies y centro de presión supuestos) y muñeca sin carga externa.
- MediaPipe pierde precisión en el brazo de golpeo a alta velocidad; las estampitas reducen el problema solo donde se detectan.
