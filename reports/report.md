# Reporte: análisis cinemático del saque kick (MediaPipe + estampitas)

Generado automáticamente por `saque run`. Cinemática 2D por vista; todas las asociaciones y tendencias entre saques son exploratorias (n = 5 saques, una jugadora).

## 1. Datos y calibración
| vista | metodo | m_por_px | px_por_m | estatura_aparente_m | estatura_real_m | fps | n_fotogramas | fotogramas_perdidos | frac_detectado |
|---|---|---|---|---|---|---|---|---|---|
| lateral | ground_plane | 0.00167 | 599 | 1.58 | 1.64 | 120 | 5414 | 24 | 0.996 |
| posterior | ground_plane | 0.00151 | 662 | 1.59 | 1.64 | 120 | 5588 | 15 | 0.997 |

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
| lateral | 1 | 3.39 | 4.99 | 5.79 |
| lateral | 2 | 11.5 | 13.1 | 13.9 |
| lateral | 3 | 21.2 | 22.8 | 23.6 |
| lateral | 4 | 29.6 | 31.2 | 32 |
| lateral | 5 | 39.6 | 41.2 | 42 |
| posterior | 1 | 4.04 | 5.64 | 6.44 |
| posterior | 2 | 12.1 | 13.7 | 14.5 |
| posterior | 3 | 21.9 | 23.5 | 24.3 |
| posterior | 4 | 30.3 | 31.9 | 32.7 |
| posterior | 5 | 40.2 | 41.8 | 42.6 |

Sincronización entre vistas: t_vista = t_referencia + lag_s (por correlación de la altura de cadera). El control compara con los impactos que la vista detecta por su cuenta con la muñeca:

| vista | referencia | lag_s | correlacion_cadera | control_diferencia_muneca_mediana_s | control_dispersion_s |
|---|---|---|---|---|---|
| posterior | lateral | 0.642 | 0.751 | -0.132 | 0.183 |

![deteccion_saques](figures/deteccion_saques.png)

![impactos_lateral](figures/impactos_lateral.png)

![impactos_posterior](figures/impactos_posterior.png)

## 4. Fases (duración en ms)
| vista | saque | aceleracion | carga | seguimiento |
|---|---|---|---|---|
| lateral | 1 | 475 | 524 | 258 |
| lateral | 2 | 433 | 574 | 241 |
| lateral | 3 | 441 | 541 | 258 |
| lateral | 4 | 550 | 391 | 316 |
| lateral | 5 | 533 | 383 | 308 |
| posterior | 1 | 334 | 424 | 423 |
| posterior | 2 | 309 | 1290 | 432 |
| posterior | 3 | 292 | 424 | 398 |
| posterior | 4 | 284 | 433 | 390 |
| posterior | 5 | 268 | 441 | 398 |

## 5. Secuencia próximo-distal
τ de Kendall = 1 indica picos en el orden esperado.

| vista | saque | orden | tau_kendall | nota | lags_ms |
|---|---|---|---|---|---|
| lateral | 1 | muslo > tronco > brazo > codo > muneca | 0 | nan | -275;0;0;275 |
| lateral | 2 | muslo > tronco > brazo > codo > muneca | 0.949 | nan | 0;33;8;58 |
| lateral | 3 | muslo > tronco > brazo > codo > muneca | 0.738 | nan | 92;-42;200;0 |
| lateral | 4 | muslo > tronco > brazo > codo > muneca | 0.632 | nan | 0;0;0;100 |
| lateral | 5 | muslo > tronco > brazo > codo > muneca | 0.6 | nan | -208;250;141;-75 |

## 6. Curvas por saque (inicio → impacto)
![curvas](figures/curvas_por_saque.png)

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
| lateral | 1 | 0.205 | 0.255 | 0.188 | 0.203 | 0.0375 | 0.874 | 11.4 |
| lateral | 2 | 0.195 | 0.236 | 0.178 | 0.238 | 0.147 | 0.91 | 6.37 |
| lateral | 3 | 0.217 | 0.245 | 0.198 | 0.234 | 0.121 | 0.836 | 8.29 |
| lateral | 4 | 0.205 | 0.228 | 0.198 | 0.227 | 0.0713 | 0.903 | 6.17 |
| lateral | 5 | 0.206 | 0.227 | 0.193 | 0.196 | 0.111 | 0.853 | 6.48 |

![jacobiano_lateral](figures/jacobiano_lateral.png)

![dinamica_inversa_tren_superior_lateral](figures/dinamica_inversa_tren_superior_lateral.png)

![dinamica_inversa_tren_inferior_lateral](figures/dinamica_inversa_tren_inferior_lateral.png)

## 9. Limitaciones
- 2D por vista: los ángulos son proyecciones en el plano de cada cámara.
- Escala: vara de 1 m corregida por profundidad suponiendo cámara nivelada a 1.30 m (ver estatura aparente en la tabla de calibración); escala fija por vista aunque la jugadora se mueve en profundidad. Los ángulos y tiempos no dependen de la escala.
- El impacto es un proxy (máxima rapidez de muñeca cerca de su altura máxima); no se rastrean raqueta ni pelota.
- Dinámica inversa exploratoria: GRF = 0 y muñeca sin carga externa.
- MediaPipe pierde precisión en el brazo de golpeo a alta velocidad; las estampitas reducen el problema solo donde se detectan.
