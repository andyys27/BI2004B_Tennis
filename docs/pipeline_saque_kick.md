# Pipeline de procesamiento: análisis cinemático del saque kick (Tenis, Tec)

> Documento maestro del pipeline. Se apoya en tres contextos previos:
> `contexto_proyecto_saque_kick.md` (objetivos y alcance), `contexto_bases_datos.md` (inventario y problemas de los CSV de Kinovea) y `contexto_dinamica_inversa.md` (modelo Newton-Euler 2D).
> Lenguaje: Python 3.11+. Convención: SI en el cálculo interno (m, s, rad); los datos de entrada vienen en cm y grados.

---

## 0. Qué resuelve este pipeline (y qué no)

**Sí resuelve (etapa actual, cinemática 2D):**

1. Limpiar, sincronizar y validar las exportaciones de Kinovea (lateral y posterior).
2. Obtener θ(t), ω(t), α(t) articulares y posiciones/velocidades de mano y muñeca con un filtrado documentado.
3. Segmentar el saque en fases con eventos reproducibles y medir el **timing y la secuencia próximo-distal**.
4. Estimar la velocidad de la mano con el Jacobiano (`v = J(θ)ω`) y contrastarla con la medida directa.
5. (M9, pelota, descartado: el análisis es solo de la jugadora.)
6. Análisis exploratorio entre intentos: variabilidad, asociaciones, y un módulo extra de ML.
7. Dejar listo el módulo de dinámica inversa para cuando existan fuerzas externas.

**No resuelve (hay que decirlo explícito en el reporte):**

- No hay GRF en los CSV. M12 puede ejecutarse con GRF=0 bajo la hipótesis explícita de que todo el registro corresponde a fase aérea; sus cargas no representan la fase de apoyo.
- Causalidad ni riesgo de lesión. Con ~5 intentos todo lo estadístico es **exploratorio**.
- 3D. Son dos vistas 2D sin calibración conjunta.
- Pelota (velocidad, spin): fuera de alcance por decisión del proyecto.

---

## 1. Estructura del proyecto

```
saque_kick/
├── README.md
├── pyproject.toml                # numpy, pandas, scipy, matplotlib, scikit-learn, pyyaml, pytest
├── config/
│   ├── pipeline.yaml             # fs nominal, filtros, umbrales, rutas
│   ├── subject.yaml              # masa, estatura, lado dominante, pierna delantera
│   └── sync_offsets.yaml         # offsets por archivo (con evidencia del video)
├── data/
│   ├── raw/                      # CSV originales de Kinovea (NUNCA se editan)
│   │   ├── lateral/
│   │   └── posterior/
│   ├── interim/
│   │   └── synchronized/         # series estandarizadas y alineadas (time_s)
│   └── processed/
│       ├── filtered/             # coordenadas/ángulos después del filtro
│       ├── kinematics/           # θ, ω, α y cinemática lineal
│       ├── jacobian/             # velocidad de mano por J(θ)ω
│       └── inverse_dynamics/     # fuerzas y momentos; requiere GRF sincronizada
├── src/saque/
│   ├── __init__.py
│   ├── config.py                 # carga de YAML y dataclasses
│   ├── catalog.py                # [M1] inventario, hash, duplicados
│   ├── io_kinovea.py             # [M2] lectura y estandarización de columnas
│   ├── qc.py                     # [M3] control de calidad del tracking
│   ├── sync.py                   # [M4] sincronización y remuestreo
│   ├── filtering.py              # [M5] suavizado y derivación
│   ├── kinematics.py             # [M6] ángulos, velocidades, aceleraciones
│   ├── events.py                 # [M7] eventos y fases
│   ├── jacobian.py               # [M8] v = J(θ)ω
│   ├── stats.py                  # [M10] variabilidad, correlaciones, bootstrap
│   ├── ml.py                     # [M11] módulo extra de ML (no implementado)
│   ├── inverse_dynamics.py       # [M12] Newton-Euler (listo para GRF)
│   ├── viz.py                    # [M13] gráficas
│   └── report.py                 # [M14] tablas y reporte final
├── scripts/
│   ├── run_pipeline.py           # orquesta M1→M14
│   └── make_figures.py
├── notebooks/
│   └── 01_exploracion.ipynb
├── tests/
│   ├── test_filtering.py
│   ├── test_kinematics.py
│   ├── test_jacobian.py
│   └── test_inverse_dynamics.py
└── reports/
    ├── figures/
    └── tables/
        ├── catalog/
        ├── events/
        ├── filtering/
        ├── inverse_dynamics/
        ├── jacobian/
        ├── qc/
        ├── report/
        ├── statistics/
        └── synchronization/
```

**Regla de oro:** `data/raw` es de solo lectura. Todo lo demás se regenera corriendo `run_pipeline.py`.

**Flujo de datos:**

```
raw CSV ─► catalog ─► io_kinovea ─► qc ─► sync ─► filtering ─► kinematics
                                                                  │
            ┌─────────────────────────────────────────────────────┤
            ▼                  ▼                 ▼                ▼
         events ─► fases    jacobian          inverse_dynamics*
            └────────────┬─────┴─────────────────┘             (*requiere GRF)
                         ▼
                  stats ─► ml ─► viz ─► report
```

---

## 2. Configuración (`config/pipeline.yaml`)

```yaml
fs_nominal_hz: 120
resample_hz: 120            # malla uniforme común tras sincronizar
filter:
  type: butterworth         # butterworth | savgol
  order: 2                  # filtfilt duplica el orden efectivo (→ 4)
  cutoff_hz: null           # null = elegir por análisis de residuos (ver §5.2)
  cutoff_range_hz: [4, 20]
  savgol_window: 15
  savgol_poly: 3
qc:
  max_jump_cm: 15           # salto máximo plausible entre muestras consecutivas
  max_gap_s: 0.05
  zscore_speed: 6
events:
  min_prominence: 0.1
phases:                     # ver §7
  names: [preparacion, carga, aceleracion, impacto_seguimiento]
dynamics:
  vgrf_threshold_N: 20
```

---

## 3. Módulos

### M1 · `catalog.py` — Inventario y duplicados

**Qué hace:** recorre `data/raw`, calcula SHA-256 de cada archivo y produce una tabla (`archivo, vista, n_filas, columnas, t_ini, t_fin, hash`).

**Por qué:** el contexto detectó que `posterior/cinematica_angular_rodilla.csv` es idéntico byte a byte a `data/posterior/cinematica_angular_rodilla.csv`. Con el hash se descarta automáticamente y no se cuenta como segundo ensayo.

```python
import hashlib, pandas as pd
from pathlib import Path

def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def build_catalog(root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(root.rglob("*.csv")):
        df = pd.read_csv(p)
        tcol = next((c for c in df.columns if c.lower() in
                     ("tiempo_s", "time_s")), None)
        rows.append(dict(
            archivo=str(p.relative_to(root)), n=len(df),
            cols=list(df.columns), hash=sha256(p),
            t0=df[tcol].iloc[0] if tcol else None,
            t1=df[tcol].iloc[-1] if tcol else None))
    cat = pd.DataFrame(rows)
    cat["duplicado_de"] = cat.groupby("hash")["archivo"].transform("first")
    cat.loc[cat["duplicado_de"] == cat["archivo"], "duplicado_de"] = None
    return cat
```

**Salida:** `reports/tables/catalog/catalogo.csv`. Los duplicados se excluyen del procesamiento.

---

### M2 · `io_kinovea.py` — Lectura y estandarización

**Qué hace:** lee cada CSV y entrega un DataFrame con nombres canónicos y unidades SI.

| Problema en los datos | Tratamiento |
|---|---|
| Columnas de tiempo con nombres distintos (`tiempo_s`, `time_s`, `tiempo_ms`, `Tiempo (ms)`) | Mapear todo a `time_s` (ms → s ÷1000). **Conservar** el original en `time_s_raw`. |
| `X_cad_cm` con mayúscula | Diccionario de alias; se lee tal cual y se renombra a `x_cad`. |
| Coordenadas en cm | Convertir a m: `x_m = x_cm / 100`. |
| Ángulos en grados | Guardar `*_deg` y generar `*_rad`. |
| `ABS_`, `MA_`, `_clean` | Se conservan con prefijo, pero **no se usan como verdad**: el pipeline recalcula desde coordenadas cuando se puede (ver §5.3). |
| `cinematica_lineal_brazo_v_a.csv` solo tiene `tiempo_ms` | Se une con `cinematica_lineal_brazo_pos.csv` solo si pasa la validación de §M4. |

```python
ALIAS = {
    "tiempo_s": "time_s", "time_s": "time_s",
    "tiempo_ms": "time_ms", "Tiempo (ms)": "time_ms",
    "X_cad_cm": "x_cad_cm",
}

def read_kinovea(path):
    df = pd.read_csv(path).rename(columns=ALIAS)
    if "time_s" not in df and "time_ms" in df:
        df["time_s"] = df["time_ms"] / 1000.0
    df["time_s_raw"] = df["time_s"]
    for c in [c for c in df if c.endswith("_cm")]:
        df[c[:-3] + "_m"] = df[c] / 100.0
    return df
```

---

### M3 · `qc.py` — Control de calidad del tracking

**Qué hace (en orden):**

1. **Monotonicidad del tiempo.** Detecta `Δt ≤ 0`. En la cinemática lineal lateral hay dos filas con `1.535 s` (filas 185 y 186). Regla: se conserva la primera, se descarta la segunda, y las coordenadas se **promedian** solo si difieren menos que un umbral; se registra en un log.
2. **Gaps.** `Δt > max_gap_s` → marcar como hueco, no interpolar a ciegas.
3. **Saltos de posición.** `‖Δp‖ > max_jump_cm` entre muestras consecutivas indica salto de etiqueta de Kinovea.
4. **Velocidad atípica.** z-score robusto (mediana/MAD) de la rapidez.
5. **Frecuencia efectiva.** Histograma de `Δt` (esperado 0.008/0.009 s ≈ 120 Hz).

```python
import numpy as np

def qc_time(df, tol=1e-9):
    dt = np.diff(df["time_s"].to_numpy())
    flags = {"non_monotonic": np.where(dt <= tol)[0] + 1,
             "gaps": np.where(dt > 0.05)[0] + 1}
    return flags

def robust_z(x):
    med = np.median(x); mad = np.median(np.abs(x - med)) + 1e-12
    return 0.6745 * (x - med) / mad
```

**Salida:** `qc_log.csv` con `archivo, indice, tipo, accion`. Todo dato excluido queda registrado (requisito del contexto del proyecto).

---

### M4 · `sync.py` — Sincronización y remuestreo

**Principio:** **nunca unir por índice de fila.** Se une por tiempo, tras documentar offsets.

**Hechos de los datos que este módulo debe resolver:**

| Caso | Observación | Hipótesis a verificar en video |
|---|---|---|
| Lateral, tobillo: angular vs `PI1` | Mismas 138 muestras, tiempos desfasados **0.100 s** (2.538–3.678 vs 2.638–3.778) | Un archivo está recortado con otro origen. Verificar con un evento visible. |
| Posterior: tobillo vs rodilla/codo | 282 muestras en ambos; tobillo 0.000–2.340 s, rodilla/codo 0.593–2.931 s. Duraciones casi iguales (2.340 vs 2.338 s) → **offset ≈ 0.59 s** | Tobillo y rodilla/codo son el mismo clip con distinto origen de tiempo. |
| Lateral lineal (544) vs angular rodilla (537) | Distinta longitud, no unir por índice | Interpolar ambas a malla común. |
| Posterior: pierna (348) vs brazo (356) | Distinta longitud | Idem. |
| `brazo_v_a` vs `brazo_pos` | Ambos 356 filas, el primero solo `tiempo_ms` (0–2956) | Verificar que `time_ms/1000 ≈ time_s` fila a fila. |

**Método para estimar el offset (cuando hay una variable común):** correlación cruzada entre dos series que miden lo mismo (p. ej. ángulo de tobillo angular vs `ABS_angulo_tob` del PI1, o rapidez de codo):

```python
from scipy.signal import correlate, correlation_lags

def estimate_offset(t_a, a, t_b, b, fs=120):
    grid = np.arange(max(t_a[0], t_b[0]) - 1.0, min(t_a[-1], t_b[-1]) + 1.0, 1/fs)
    ia = np.interp(grid, t_a, a, left=np.nan, right=np.nan)
    ib = np.interp(grid, t_b, b, left=np.nan, right=np.nan)
    m = ~np.isnan(ia) & ~np.isnan(ib)
    ia, ib = ia[m] - ia[m].mean(), ib[m] - ib[m].mean()
    cc = correlate(ia, ib, mode="full")
    lag = correlation_lags(len(ia), len(ib))[np.argmax(cc)]
    return lag / fs   # segundos; signo: positivo = a va "adelante" de b
```

> El offset estimado por correlación es **evidencia**, no verdad. Se confirma con un evento en el video (p. ej. fotograma del impacto) y se guarda en `config/sync_offsets.yaml`:
> ```yaml
> posterior/cinematica_angular_tobillo.csv:   {offset_s: 0.592, evidencia: "frame 71, contacto pie", metodo: "video"}
> lateral/PI1_tiempo_angulo_abs_tobillo.csv:  {offset_s: -0.100, evidencia: "xcorr + video", metodo: "xcorr"}
> ```

**Remuestreo a malla uniforme:** spline cúbico sobre el intervalo **común** (sin extrapolar).

```python
from scipy.interpolate import CubicSpline

def to_uniform_grid(df, cols, fs=120, t0=None, t1=None):
    t = df["time_s"].to_numpy()
    t0 = t[0] if t0 is None else t0
    t1 = t[-1] if t1 is None else t1
    grid = np.arange(t0, t1, 1/fs)
    out = pd.DataFrame({"time_s": grid})
    for c in cols:
        out[c] = CubicSpline(t, df[c].to_numpy())(grid)
    return out
```

**Importante:** las dos vistas (lateral y posterior) **no** se sincronizan entre sí hasta tener un evento común visible en ambas (p. ej. el impacto). Hasta entonces se analizan por separado.

---

### M5 · `filtering.py` — Suavizado y derivación

**Por qué es crítico:** derivar dos veces amplifica el ruido (la aceleración sale del ruido de la posición × ω²). Las columnas `v_*`/`a_*` de Kinovea no documentan filtro ni unidades, así que el pipeline las **recalcula** desde posiciones.

**Método base:** Butterworth pasa-bajas de fase cero (`filtfilt`).

```python
from scipy.signal import butter, filtfilt

def butter_lp(x, fs, fc, order=2):
    b, a = butter(order, fc / (fs / 2), btype="low")
    return filtfilt(b, a, x, padlen=min(3 * max(len(a), len(b)), len(x) - 1))
```

Con `order=2` y `filtfilt` el filtro efectivo es de 4.º orden. El corte efectivo se corrige por el doble paso: a −3 dB se reduce ≈ 0.80·fc en el diseño de 2.º orden.

#### 5.2 Elección de la frecuencia de corte (análisis de residuos, Winter)

Para cada marcador, para un barrido `fc ∈ [4, 20] Hz`:

$$
R(f_c)=\sqrt{\frac{1}{N}\sum_{i=1}^{N}\left(x_i-\hat{x}_i(f_c)\right)^2}
$$

Se grafica `R(fc)`, se ajusta una recta a la zona lineal de ruido (fc alto) y se extrapola a `fc = 0`; el corte óptimo es la fc donde `R` iguala el valor de la intersección. Para el saque (movimientos rápidos de brazo) se espera una fc entre 8–15 Hz para el tren inferior y algo mayor para la muñeca; **documentar el valor por marcador**.

```python
def residual_analysis(x, fs, fcs):
    return np.array([np.sqrt(np.mean((x - butter_lp(x, fs, f))**2)) for f in fcs])
```

**Sensibilidad (obligatoria en el reporte):** repetir el análisis con fc ± 2 Hz y reportar cómo cambian los picos de ω y α. Si el pico de α cambia más de ~15 %, el valor de α no es confiable y se interpreta solo cualitativamente.

#### 5.3 Derivación

Sobre malla uniforme, diferencias centrales:

$$
\dot{x}_i=\frac{x_{i+1}-x_{i-1}}{2\,\Delta t},\qquad
\ddot{x}_i=\frac{x_{i+1}-2x_i+x_{i-1}}{\Delta t^{2}}
$$

```python
def derivative(x, fs):
    dt = 1/fs
    return np.gradient(x, dt)

def second_derivative(x, fs):
    return np.gradient(np.gradient(x, 1/fs), 1/fs)
```

Alternativa: `savgol_filter(x, w, p, deriv=1|2, delta=dt)` que filtra y deriva en un paso.

**Validación:** `tests/test_filtering.py` usa `x(t) = A sin(2π f t)`; la derivada analítica se compara con la numérica (error relativo < 1 %).

---

### M6 · `kinematics.py` — Ángulos, ω, α

#### 6.1 Ángulo absoluto de un segmento (respecto a la horizontal)

Para un segmento con extremo proximal `P=(x_p, y_p)` y distal `D=(x_d, y_d)`:

$$
\theta_{seg}=\operatorname{atan2}\!\left(y_d-y_p,\;x_d-x_p\right)
$$

Se aplica `np.unwrap` antes de derivar (si no, un salto de ±π produce picos falsos de ω).

#### 6.2 Ángulo articular relativo

Entre dos segmentos `u` (proximal) y `v` (distal) que comparten la articulación:

$$
\varphi=\operatorname{atan2}\!\big(u\times v,\;u\cdot v\big),\qquad
u\times v = u_x v_y-u_y v_x
$$

Esta forma conserva el signo (flexión/extensión), a diferencia de `arccos`, que solo da 0–180°.

#### 6.3 Velocidad y aceleración angular

$$
\omega=\frac{d\theta}{dt},\qquad \alpha=\frac{d^{2}\theta}{dt^{2}}
$$

Conversión: `θ[rad] = θ[°]·π/180` (las derivadas se calculan **ya en rad**).

```python
def abs_angle(px, py, dx, dy):
    return np.unwrap(np.arctan2(dy - py, dx - px))

def joint_angle(u, v):
    cross = u[:, 0]*v[:, 1] - u[:, 1]*v[:, 0]
    dot = (u*v).sum(axis=1)
    return np.unwrap(np.arctan2(cross, dot))
```

#### 6.4 Qué sale de los datos actuales

| Variable | Fuente | Vista | Notas |
|---|---|---|---|
| Flexión de rodilla | `angulo_rod` / coordenadas cadera-rodilla-tobillo | lateral | Confirmar pierna delantera |
| Tobillo | `angulo_tob` / coordenadas | lateral y posterior | Dos vistas, dos planos distintos |
| Codo | `angulo_codo` / coordenadas hombro-codo-muñeca | posterior | Plano proyectado |
| Posición/velocidad de muñeca | `x_mun_cm`, `y_mun_cm` | posterior | Proxy de la mano |
| **No disponible aún** | tronco, hombro (ángulo), cadera (flexión), raqueta, pelota | — | Hay que digitalizarlos |

**Check de consistencia:** comparar el ángulo recalculado desde coordenadas contra la columna de Kinovea (RMSE en grados). Si difieren más de ~2–3°, revisar convención de ángulo o puntos.

---

### M7 · `events.py` — Eventos y fases

Los eventos deben ser **reproducibles por algoritmo**, y luego **validados contra el video**.

#### 7.1 Eventos propuestos

| # | Evento | Criterio algorítmico | Variable |
|---|---|---|---|
| E1 | Inicio del movimiento | rapidez de muñeca supera 5 % de su pico, sostenido 3 muestras | `v_mun` |
| E2 | Máxima flexión de rodilla (carga/“trophy”) | mínimo de `θ_rodilla` (máxima flexión) antes del impacto | `θ_rod` |
| E3 | Máxima flexión de codo (armado, “racket drop”) | extremo local de `φ_codo` previo a E4 | `φ_codo` |
| E4 | Máxima velocidad angular de codo | `argmax ω_codo` | `ω_codo` |
| E5 | **Impacto (proxy)** | `argmax` de la rapidez de la muñeca (o fotograma marcado a mano) | `v_mun` |
| E6 | Fin del seguimiento | rapidez baja de nuevo a 20 % del pico | `v_mun` |

> **Impacto:** sin pelota ni raqueta digitalizadas, el máximo de rapidez de muñeca es solo un *proxy*. Lo correcto es marcar el fotograma de contacto en el video y guardarlo en `events_manual.csv`; el algoritmo lo compara contra el proxy y se reporta la diferencia en ms.

```python
from scipy.signal import find_peaks

def detect_events(t, wrist_speed, knee_angle, elbow_angle):
    ev = {}
    pk, _ = find_peaks(wrist_speed, prominence=0.1 * wrist_speed.max())
    ev["impacto"] = t[pk[np.argmax(wrist_speed[pk])]]
    i_imp = np.searchsorted(t, ev["impacto"])
    ev["max_flex_rodilla"] = t[np.argmin(knee_angle[:i_imp])]
    thr = 0.05 * wrist_speed.max()
    ev["inicio"] = t[np.argmax(wrist_speed > thr)]
    return ev
```

#### 7.2 Fases

| Fase | Límites | Interpretación |
|---|---|---|
| Preparación | E1 → inicio de flexión de rodillas | Postura inicial, balanceo |
| Carga | inicio flexión → E2 | Flexión de rodillas, armado |
| Aceleración | E2 → E5 | Extensión de piernas + rotación + extensión del codo, transferencia próximo-distal |
| Impacto y seguimiento | E5 → E6 | Desaceleración y recuperación |

**Normalización temporal:** cada intento se lleva a 0–100 % de la fase (o de E1→E6) con 101 puntos, para promediar/comparar curvas.

```python
def time_normalize(t, y, t0, t1, n=101):
    tn = np.linspace(t0, t1, n)
    return np.interp(tn, t, y)
```

#### 7.3 Secuencia próximo-distal (objetivo específico 3)

Orden esperado de picos de velocidad: **cadera/tronco → hombro → codo → muñeca** (los picos más distales ocurren después). Se calculan:

- Tiempos de pico `t_pk(i)` de cada segmento.
- Lags `Δ_{i→i+1} = t_pk(i+1) − t_pk(i)` en ms.
- Índice de secuencia (τ de Kendall entre orden esperado y observado): `τ = 1` secuencia perfecta.
- Ratio de amplificación de pico `ω_pk(distal)/ω_pk(proximal)`.

Con los datos actuales solo se puede evaluar parte de la cadena (rodilla → codo → muñeca); tronco y hombro requieren digitalizar más puntos. **No afirmar “transferencia próximo-distal correcta” con una cadena incompleta.**

---

### M8 · `jacobian.py` — Velocidad de la mano: `v = J(θ) ω`

Brazo planar de dos eslabones (hombro → codo → muñeca), con ángulos absolutos `θ₁` (brazo) y `θ₂` (antebrazo) respecto a la horizontal, longitudes `L₁`, `L₂`:

$$
\begin{aligned}
x_w &= x_s + L_1\cos\theta_1 + L_2\cos\theta_2\\
y_w &= y_s + L_1\sin\theta_1 + L_2\sin\theta_2
\end{aligned}
$$

$$
\mathbf{v}=
\begin{bmatrix}\dot x_w\\ \dot y_w\end{bmatrix}
=
\underbrace{\begin{bmatrix}
-L_1\sin\theta_1 & -L_2\sin\theta_2\\
\;\;L_1\cos\theta_1 & \;\;L_2\cos\theta_2
\end{bmatrix}}_{J(\boldsymbol\theta)}
\begin{bmatrix}\omega_1\\ \omega_2\end{bmatrix}
$$

(si el hombro se mueve, se suma su velocidad: `v = v_s + J ω`).

**Qué se espera:**
- `L₁`, `L₂` se estiman de las coordenadas (mediana de la distancia hombro–codo y codo–muñeca en todo el clip; debería ser casi constante: su CV es un check de calidad del tracking).
- `v_mano_J` (por Jacobiano) vs `v_mano_directa` (derivando `x_mun, y_mun`). Deben coincidir; si no, hay error de convención de ángulos, de sincronía, o de tracking.
- Métrica de validación: RMSE y correlación entre ambas rapideces.

```python
def jacobian_2link(th1, th2, L1, L2):
    J = np.empty((len(th1), 2, 2))
    J[:, 0, 0] = -L1*np.sin(th1); J[:, 0, 1] = -L2*np.sin(th2)
    J[:, 1, 0] =  L1*np.cos(th1); J[:, 1, 1] =  L2*np.cos(th2)
    return J

def hand_velocity(J, w1, w2):
    w = np.stack([w1, w2], axis=1)[..., None]
    return (J @ w)[..., 0]           # (N, 2)  [vx, vy]
```

**Análisis adicional útil:** descomponer la rapidez de la mano en la contribución de cada articulación (`J[:,:,0]·ω₁` vs `J[:,:,1]·ω₂`) para ver en qué instante domina el codo frente al hombro.

**Limitación:** la vista posterior proyecta el brazo en un plano; `v` es la velocidad proyectada, **no** la velocidad 3D de la raqueta. La velocidad de cabeza de raqueta en el impacto requiere puntos de raqueta.

---

### M9 · Pelota — **DESCARTADO**

El análisis se centra únicamente en la jugadora (cinemática, cadena cinética, brazos de palanca y, cuando existan GRF, fuerzas y torques). No se digitaliza ni se analiza la pelota; por lo tanto no hay módulo `ball.py`, ni velocidad de salida, ni spin. Las asociaciones de rendimiento se plantean con variables de la propia jugadora (p. ej. velocidad de la mano/muñeca).

---

### M10 · `stats.py` — Variabilidad y asociaciones (exploratorio)

Con ~5 intentos, **no hacer pruebas de hipótesis “de verdad”**. Sí:

1. **Descriptivos por fase e intento:** pico, tiempo al pico, rango (ROM), media. Tabla en `reports/tables/`.
2. **Variabilidad inter-intento:**
   - Coeficiente de variación: `CV = σ/μ · 100 %`
   - Banda de curvas normalizadas (media ± 1 DE).
   - Error típico entre intentos (RMSD de cada intento vs la media).
3. **Asociaciones entre variables de la jugadora** (p. ej. índice de secuencia próximo-distal vs velocidad pico de muñeca): Spearman `ρ`, siempre con IC por **bootstrap** y con la advertencia de `n`.
4. **Comparación con la literatura:** tabla con rango de referencia, población, plano y definición. Solo si son metodológicamente comparables; si no, se dice que no lo son.

```python
from scipy.stats import spearmanr

def boot_ci(x, y, n_boot=5000, alpha=0.05, rng=np.random.default_rng(0)):
    n = len(x); rs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if np.ptp(x[i]) == 0 or np.ptp(y[i]) == 0: continue
        rs.append(spearmanr(x[i], y[i])[0])
    return np.nanpercentile(rs, [100*alpha/2, 100*(1-alpha/2)])
```

> Con n = 5 el IC de `ρ` será enorme. Eso **es** el resultado honesto: se reporta como “tendencia observada, no concluyente”.

**Formalizar hipótesis (antes de inferir):** H0 estadística = “no hay asociación entre el índice de transferencia próximo-distal y la velocidad pico de la mano/muñeca”; H1 = “sí hay asociación”. Las H0/H1 direccionales del contexto del proyecto son hipótesis de trabajo, no nula/alternativa estadística.

La implementación actual escribe `reports/tables/statistics/descriptivos_por_fase.csv`,
`curvas_normalizadas.csv` y `variabilidad_interintento.csv`. Los CSV disponibles no
identifican intentos repetidos de forma independiente, por lo que el pipeline reporta
descriptivos por serie y no presenta CV inter-intento ni asociaciones como estimables
hasta que se incorporen IDs de intento y varias repeticiones.

---

### M11 · `ml.py` — Módulo extra (ML) 🔥

> **Advertencia honesta:** con 5 intentos (y quizá 1 sujeto) el ML **no puede aprender un modelo predictivo generalizable**. Lo valioso aquí es: (a) ML como herramienta de **QC y descubrimiento de patrones**, (b) demostrar una **arquitectura lista para escalar** a más tenistas, (c) evaluar con validación honesta (LOOCV). Todo se etiqueta “exploratorio”.

#### 11.1 Detección de anomalías en el tracking (útil desde el día 1)
`IsolationForest` o z-score robusto sobre `[rapidez, aceleración, Δángulo]` para marcar saltos de etiqueta de Kinovea. Reemplaza revisión manual cuadro por cuadro. Se valida contra los saltos marcados por QC.

```python
from sklearn.ensemble import IsolationForest
def anomaly_flags(features, contamination=0.01):
    return IsolationForest(contamination=contamination, random_state=0)\
        .fit_predict(features) == -1
```

#### 11.2 Detección automática de fases (change-point / clustering)
Segmentar la señal multivariada `[ω_rodilla, ω_codo, v_muñeca]` con `ruptures` (PELT/Binseg) y comparar los puntos de cambio con los eventos E1–E6 detectados por reglas. Medida: diferencia media en ms. Si coinciden, se gana confianza en las fases.

#### 11.3 PCA funcional de curvas normalizadas
Matriz `X ∈ R^{intentos × (variables·101)}`. PCA indica en qué parte de la fase y en qué variable difieren los intentos (cargas de PC1/PC2 graficadas sobre el % del gesto). Con 5 intentos hay máximo 4 componentes con varianza no nula: se interpreta PC1–PC2 **de forma descriptiva**.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
Z = StandardScaler().fit_transform(X)
pca = PCA(n_components=min(3, len(Z)-1)).fit(Z)
scores, evr = pca.transform(Z), pca.explained_variance_ratio_
```

#### 11.4 Similitud entre intentos: DTW + clustering jerárquico
Distancia DTW entre curvas (resuelve que un intento sea ligeramente más lento) → dendrograma. Responde “¿qué saques son técnicamente parecidos?” y detecta el intento atípico.

#### 11.5 Regresión de rendimiento con validación honesta (objetivo: velocidad pico de muñeca, no pelota)
Features interpretables (no curvas crudas): `ω_pk_codo`, lag cadera→muñeca, flexión máxima de rodilla, ROM de codo, `ω_pk_hombro`. Modelos **fuertemente regularizados**: Ridge o **PLS** (maneja p>n). Validación: **Leave-One-Out CV**, métricas `RMSE_LOO` y `Q²`. Comparar siempre contra un baseline “predecir la media”.

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))
pred = cross_val_predict(model, X_feat, y_wrist, cv=LeaveOneOut())
rmse = np.sqrt(np.mean((y_wrist - pred)**2))
q2 = 1 - np.sum((y_wrist - pred)**2) / np.sum((y_wrist - y_wrist.mean())**2)
```

Si `Q² ≤ 0`, el modelo no supera a la media: se reporta tal cual (resultado válido y honesto).

#### 11.6 Rutas de escalamiento (para presentar como “trabajo futuro”)
- Estimación de pose automática (MediaPipe/OpenPose) para digitalizar sin Kinovea y tener decenas de saques.
- Autoencoder 1D / modelos de series temporales cuando haya >50 intentos.
- Mixed-effects (jugador como efecto aleatorio) cuando haya varios jugadores.

---

### M12 · `inverse_dynamics.py` — Dinámica inversa (Newton-Euler 2D)

Implementa balances vectoriales de Newton-Euler por segmento como **módulo vectorizado** y testeado. Puede ejecutarse con GRF medida y sincronizada, o en el modo actual `assume_flight_no_grf`, que fija GRF=0 para todas las muestras y supone fase aérea.

#### 12.1 Parámetros antropométricos (Winter), sujeto 1.64 m, 58 kg

| Segmento | Masa (% M) | m (kg) | Long. (% H) | l (m) |
|---|---:|---:|---:|---:|
| Pie | 1.45 % | 0.841 | 15.2 % (long. pie) | ≈ 0.25 |
| Pierna | 4.65 % | 2.697 | 24.6 % | ≈ 0.403 |
| Muslo | 10.0 % | 5.800 | 24.5 % | ≈ 0.402 |

Centros de masa desde proximal (Winter): pie 0.50 l, pierna 0.433 l, muslo 0.433 l. Radios de giro respecto al CM: pie 0.475 l, pierna 0.302 l, muslo 0.323 l. `I = m ρ²`.

> Esos valores deben **verificarse con la tabla original** de Winter y con la tenista real; los parámetros no se generalizan entre participantes.

#### 12.2 Modelo implementado

El módulo usa la forma vectorial de Newton–Euler, calculando el centro de masa y su
aceleración para pie, pierna y muslo, y resolviendo fuerzas y momentos de distal a
proximal. Las fuerzas y momentos de acción-reacción se transmiten entre segmentos.
Los ángulos absolutos se miden desde el eje x positivo, con z vertical hacia arriba;
el momento positivo es antihorario.

`PD` identifica la pierna derecha y `PI1` la izquierda. Para la derecha, M12 usa la
cadena posterior cadera–rodilla–tobillo–punta del pie. Para la izquierda usa la cadena
de marcadores lateral cadera–rodilla–tobillo asociada a PI1; al no haber punto de pie,
solo calcula fuerzas/momentos en rodilla y cadera, suponiendo fuerza externa cero en
tobillo. Las muestras con longitudes segmentarias atípicas y las seis muestras vecinas
por lado (margen de 50 ms a 120 fps) se marcan inválidas para limitar la contaminación
de las derivadas; sus fuerzas/momentos quedan vacíos (NaN), no como picos. Ambos CSV
identifican el lado y el modo de carga supuesto.

En modo `measured_grf`, la fuerza de plataforma actúa en el centro de presión. El CSV
debe contener tiempo en segundos, `GRF_x` y `vGRF` en N y el vector
`(cop_x_from_ankle_m, cop_z_from_ankle_m)` en metros; se exige reloj compartido
verificado y se interpola solo en el intervalo común, sin extrapolar. En el modo
`assume_flight_no_grf` no se carga GRF ni CoP: se usa cero en todas las muestras y se
procesa el registro cinemático completo.

En el modo actual, los valores estimados solo pueden interpretarse bajo la hipótesis
de vuelo y **no son cargas reales de la fase de apoyo**. El CSV incluye
`external_force_mode` para identificarlo y el reporte advierte esta limitación. El
estado queda en `reports/tables/inverse_dynamics/status.csv`. En el modo medido,
contacto se detecta con `vGRF > contact_threshold_n`; durante el vuelo se anula la
fuerza externa, pero las fuerzas internas del tobillo no se fijan a cero. Los
parámetros antropométricos son valores iniciales y deben validarse para cada
participante.

La rama superior usa la vista posterior y los segmentos hombro–codo y codo–muñeca.
Calcula reacciones y momentos netos en codo y hombro, suponiendo fuerza externa cero
en muñeca. Estima longitudes de los marcadores filtrados y usa proporciones
antropométricas de referencia configuradas para masas, centros de masa y radios de
giro. La salida está en `posterior_brazo_inverse_dynamics.csv`; no calcula fuerzas
de la mano, raqueta ni pelota y no debe interpretarse como carga real del saque.

El código no implementa literalmente las fórmulas cerradas de
`contexto_dinamica_inversa.md`: estas se conservan como formulación preliminar y deben
compararse con el modelo vectorial antes de interpretarlas como resultados equivalentes.

#### 12.3 Riesgos conocidos (revisar antes de usarlo)

1. El modelo inicial fue planteado para un **brinco**. Su aplicación al **saque kick** debe justificarse: la configuración actual representa una sola cadena y un único contacto; no modela explícitamente apoyo bilateral, pelvis ni tronco.
2. Los parámetros antropométricos de Winter y las longitudes segmentarias deben verificarse para cada participante.
3. La sincronización entre GRF, video y centro de presión debe validarse; un archivo temporalmente alineado de forma incorrecta produce fuerzas/torques engañosos.
4. El modelo cierra en cadera sin tronco → no sirve para cargas de columna, hombro o codo.
5. Sin GRF no se puede identificar si una muestra corresponde a apoyo o vuelo; el modo actual asume vuelo en todo el registro y no debe usarse para interpretar el saque durante el apoyo.
6. La cadena izquierda no incluye pie ni GRF y supone fuerza externa cero en tobillo; solo entrega cadera y rodilla. La cadena derecha asume GRF=0 en todo el registro. Ninguna permite interpretar cargas reales de apoyo.
7. El tren superior es un modelo 2D de brazo y antebrazo que termina en la muñeca. Se supone fuerza externa nula en muñeca y se excluyen mano, raqueta y pelota; sus momentos no describen las cargas reales del golpe. Las masas, centros de masa y radios de giro son proporciones antropométricas de referencia, no mediciones individuales.

#### 12.4 Cómo validarlo sin GRF (tests)

- **Caída libre:** aceleración de tobillo igual a la gravedad, orientación constante y `GRF = 0` → fuerzas y momentos internos nulos.
- **Equilibrio estático simple:** segmentos verticales, sin aceleración y una fuerza vertical bajo el tobillo → momentos articulares nulos; la reacción se propaga según las masas segmentarias.
- El signo de las salidas se fija por el convenio de fuerzas/momentos sobre el segmento; antes de usar GRF real aún falta comparar con una segunda implementación independiente o una medición de referencia.

---

### M13 · `viz.py` — Gráficas

| # | Gráfica | Qué muestra | Qué se interpreta |
|---|---|---|---|
| G1 | Catálogo temporal (diagrama de barras de t_ini–t_fin por archivo) | Ventanas de cada CSV | Evidencia visual de offsets (tobillo vs rodilla) |
| G2 | Histograma de Δt | Muestreo efectivo | Frecuencia real ≈ 120 Hz; irregularidades |
| G3 | Análisis de residuos `R(fc)` por marcador | Elección del corte | fc elegida y sensibilidad |
| G4 | Crudo vs filtrado (posición, ω, α) | Efecto del filtro | α cruda es ruido; α filtrada es utilizable |
| G5 | θ(t), ω(t), α(t) por articulación con eventos y fases sombreadas | Cinemática angular | Picos, timing, ROM |
| G6 | Curvas normalizadas 0–100 % por intento + media ± DE | Consistencia | Variabilidad por fase |
| G7 | Secuencia próximo-distal: ω de cada segmento normalizado y marcas de pico | Coordinación | Orden de picos y lags en ms |
| G8 | Trayectoria de muñeca (x vs y) coloreada por rapidez | Trayectoria de la mano | Dónde se acelera y dónde está el pico |
| G9 | `v_mano_J` vs `v_mano_directa` | Validación del Jacobiano | Cercanía a y=x, RMSE |
| G10 | Contribución de cada articulación a la velocidad de mano | Descomposición | Dominio de codo vs hombro |
| G11 | Postura clave (stick figure) en E2, E3, E5 con brazos de palanca | Interpretación estática | Qué distancias perpendiculares crecen |
| G12 | (descartada: pelota) | — | — |
| G13 | PCA: scores + cargas; dendrograma DTW | Módulo ML | Intentos atípicos y dónde difieren |
| G14 | LOOCV: predicho vs observado (objetivo: velocidad de muñeca) | ML honesto | ¿Supera al baseline? |
| G15 | (Futuro) Torques T₁, T₂, T₃ y GRF vs tiempo | Dinámica inversa | Picos de carga antes/después del despegue |

Estilo: ejes con unidades, leyenda clara, misma escala entre intentos, eventos como líneas verticales.

La implementación genera las figuras que tengan entradas disponibles en
`reports/figures/`: cobertura temporal, intervalos de muestreo, cinemática por grupo,
fases, validación/contribuciones del Jacobiano, trayectoria de muñeca y, cuando existe
la salida M12, momentos netos por articulación y componentes de fuerzas de reacción en
`reports/figures/inverse_dynamics/`. Las figuras M12 muestran la hipótesis de fuerza
externa cuando está etiquetada en el CSV; con `GRF=0`, sus valores son solo exploratorios
y no representan las cargas reales durante el apoyo.

```python
import matplotlib.pyplot as plt

def plot_kinematics(t, theta_deg, omega, alpha, events, title):
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
    for a, y, lab in zip(ax, [theta_deg, omega, alpha],
                         ["θ (°)", "ω (rad/s)", "α (rad/s²)"]):
        a.plot(t, y); a.set_ylabel(lab)
        for name, te in events.items():
            a.axvline(te, ls="--", lw=.8, c="gray")
    ax[-1].set_xlabel("tiempo (s)"); fig.suptitle(title)
    return fig
```

---

### M14 · `report.py` — Tablas y reporte

Genera automáticamente:

- `tabla_eventos.csv`: tiempos de E1–E6 por intento.
- `tabla_picos.csv`: pico, tiempo al pico, ROM por variable y fase.
- `tabla_variabilidad.csv`: media, DE, CV por variable.
- `tabla_secuencia.csv`: lags próximo-distales y τ de Kendall.
- `tabla_validacion.csv`: RMSE de ángulos recalculados vs Kinovea, Jacobiano vs directa, sensibilidad al filtro.
- `log_decisiones.md`: todo offset, dato excluido, filtro y parámetro usado.

La implementación escribe `reports/report.md`, `reports/log_decisiones.md` y
`reports/tables/report/metricas_validacion.csv`, incorporando advertencias y estado de
M12 para que las limitaciones queden visibles.

---

## 4. Orden de ejecución (`scripts/run_pipeline.py`)

Al ejecutar `python scripts/run_pipeline.py --raw data/raw`, el punto de entrada corre M1–M6, M7 y M8 (para los grupos que declaran `jacobian`, actualmente `posterior_brazo`), y luego M10, M12, M13 y M14. M12 está activado en `assume_flight_no_grf`; para analizar cargas durante el apoyo se necesita cambiar a `measured_grf`, aportar GRF/CoP y verificar el reloj común. Las salidas relevantes son:

- M7: `reports/tables/events/` (`eventos.csv`, `fases.csv`, `picos_secuencia.csv`, `secuencia.csv`).
- M8: `data/processed/jacobian/<grupo>_jacobian.csv` y `reports/tables/jacobian/validacion.csv`.
- M10: `reports/tables/statistics/` (descriptivos por fase, curvas normalizadas y estado de variabilidad).
- M12: `posterior_pierna_inverse_dynamics.csv` (pierna derecha, marcador PD), `lateral_PI1_left_inverse_dynamics.csv` (pierna izquierda, cadera/rodilla) y `posterior_brazo_inverse_dynamics.csv`, bajo `data/processed/inverse_dynamics/`. Las tres cadenas tienen gráficas de fuerzas y momentos en `reports/figures/inverse_dynamics/`. Los estados se registran en `reports/tables/inverse_dynamics/`.
- M13: figuras bajo `reports/figures/`; M14: `reports/report.md`, `reports/log_decisiones.md` y tabla consolidada.

Las salidas intermedias y procesadas se separan por procedimiento: sincronización en `data/interim/synchronized/`; filtrado, cinemática y Jacobiano en sus respectivas carpetas bajo `data/processed/`. Los reportes tabulares también se agrupan por procedimiento.

M9 (pelota) está marcado como descartado y M11 no está implementado. M7/M10 advierten si la sincronización entre grupos no está verificada; sus comparaciones entre vistas/grupos deben considerarse exploratorias.

---

## 5. Interpretación de resultados (guía para el reporte)

### 5.1 Cómo leer θ, ω, α

- **θ(t):** postura. La rodilla delantera debe mostrar flexión durante la carga (mínimo de θ en E2) y luego extensión. El **ROM** (máx − mín) indica cuánta carga excéntrica/concéntrica hay.
- **ω(t):** velocidad de giro. El pico de ω de codo o de la rotación del tronco suele ocurrir justo antes del impacto.
- **α(t):** indica *cuándo* se aplica el torque neto. Un pico positivo grande antes del pico de ω corresponde a la fase de aceleración; el cambio de signo de α marca el pico de ω. Es la variable más sensible al filtro: interpretarla con cautela.

### 5.2 Cadena cinética

Si los picos de velocidad aparecen en orden proximal → distal con lags positivos y la amplitud del pico crece hacia la mano, es **consistente con** una transferencia de energía eficiente. Un pico distal que ocurre **antes** que el proximal, o lags ~0, sugiere que el segmento distal actúa “por su cuenta” (menor aprovechamiento de la cadena). Se afirma “consistente con”, no “demuestra”.

### 5.3 Jacobiano

Si `v_mano_J ≈ v_mano_directa` (correlación > 0.95 y RMSE pequeño frente al pico), el modelo de 2 eslabones describe bien el brazo en el plano proyectado. Diferencias sistemáticas apuntan a movimiento fuera de plano o a longitudes de segmento mal estimadas.

### 5.4 Posturas y brazos de palanca (cualitativo)

El torque articular de una fuerza `F` es `τ = r⊥ · F`, donde `r⊥` es la distancia perpendicular de la línea de acción al eje articular:

$$
\tau = \mathbf r\times\mathbf F = r_x F_y - r_y F_x
$$

Con la postura se puede decir si `r⊥` **crece o disminuye**, pero sin conocer `F` no se puede cuantificar `τ`. Ejemplo: en la carga, una rodilla más flexionada aleja la línea de acción de la GRF de la rodilla → mayor brazo de palanca → **potencialmente** mayor demanda del cuádriceps, a mismas condiciones externas.

### 5.5 Variabilidad

CV bajo en variables clave (< ~10 %) sugiere técnica consistente; CV alto sugiere variabilidad técnica, pero **con n = 5 y sin repetir por sujeto no se separa variabilidad intra-sujeto de diferencias entre jugadores.**

### 5.6 Torques (futuro)

Cuando existan GRF sincronizadas: picos de `T₂` (rodilla) durante el apoyo y el empuje suelen marcar la mayor demanda extensora; `T₁` (tobillo) caracteriza la flexión plantar en el despegue. Siempre se reportan **normalizados** (N·m/kg) para comparar entre atletas, y se aclara que son torques *netos*, no fuerza muscular individual.

---

## 6. Decisiones y riesgos abiertos (registrar en el reporte)

| # | Pendiente | Impacto | Acción |
|---|---|---|---|
| 1 | ¿Cinco intentos por jugador o en total? ¿Cuántos jugadores? | Define si hay estadística entre sujetos | Confirmar y registrar `subject_id`, `trial_id` |
| 2 | Offsets de tobillo (0.100 s lateral; ≈0.59 s posterior) | Series descuadradas si no se resuelven | Verificar en video y guardar en `sync_offsets.yaml` |
| 3 | Tiempo repetido 1.535 s (lateral lineal) | Derivadas infinitas si no se corrige | Regla de QC documentada |
| 4 | Unidades y método de `v_*`/`a_*` de Kinovea | No usables directamente | Recalcular desde posiciones |
| 5 | Convenciones de ángulo (absoluto vs articular) | Signos y rangos erróneos | Comparar recalculado vs Kinovea |
| 6 | No hay raqueta, tronco ni ángulo de hombro/cadera (la pelota queda fuera de alcance) | Objetivos 3, 4 y 6 parciales | Digitalizar en Kinovea |
| 7 | No hay GRF | Sin dinámica inversa real | Conseguir plataforma o declarar fuera de alcance |
| 8 | Sincronía entre vistas | No se puede fusionar lateral y posterior | Evento común (impacto) en ambos videos |
| 9 | Perspectiva y cámara a 1.30 m | Error de escala | Documentar posición y calidad de calibración |
| 10 | Revisión de signos en `T₂`, `T₃` | Torques erróneos | Test cruzado §12.4 |

---

## 7. Checklist de entrega para la clase

- [ ] Catálogo de archivos y duplicados descartados.
- [ ] Offsets verificados y registrados.
- [ ] Gráfica G4 (crudo vs filtrado) y G3 (elección de fc) justificando el filtro.
- [ ] θ, ω, α por articulación con eventos (G5) y curvas normalizadas por intento (G6).
- [ ] Secuencia próximo-distal con lags (G7) y la advertencia de cadena incompleta.
- [ ] Jacobiano validado (G9).
- [ ] Postura clave con brazos de palanca (G11) y discusión cualitativa.
- [ ] Módulo ML con advertencias de n y validación LOOCV (G13–G14).
- [ ] Sección de limitaciones (§0 y §6) y propuesta de trabajo futuro (GRF, tronco, raqueta, 3D).

---

## 8. Dependencias y reproducibilidad

```toml
[project]
dependencies = [
  "numpy", "pandas", "scipy", "matplotlib",
  "scikit-learn", "pyyaml", "ruptures", "tslearn",   # tslearn: DTW
  "pytest"
]
```

- Semillas fijas (`random_state=0`, `default_rng(0)`).
- Cada figura/tabla se genera por script, no a mano.
- Versionar `config/` y `log_decisiones.md`; `data/raw` solo lectura.