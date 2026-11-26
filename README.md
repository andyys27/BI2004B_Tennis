# Análisis cinemático 2D del saque kick (`saque`)

Pipeline en Python que analiza **los videos completos** de una jugadora del equipo representativo del
Tec (vistas lateral y posterior, 120 fps) con **MediaPipe Pose** + **seguimiento de las estampitas**
adheridas a las articulaciones. Detecta automáticamente los 5 saques, calcula θ/ω/α articulares,
fases, secuencia próximo-distal, velocidad de la mano con el Jacobiano, dinámica inversa exploratoria
y **indicadores de posible fatiga a lo largo de la serie**.

Sustituye al flujo anterior basado en exportaciones manuales de Kinovea de un solo saque
(los CSV se conservan en `data/raw/kinovea/` como referencia histórica; el pipeline ya no los usa).

## Uso rápido

```bash
python -m venv .env && source .env/bin/activate
pip install -e ".[dev]"          # requiere ffmpeg/ffprobe en el PATH

saque run                        # pipeline completo (MediaPipe ~5 min por video; queda en caché)
saque run --stages kinematics serves analysis report    # sin repetir MediaPipe
saque calibrate lateral          # fotograma con rejilla para medir la vara de 1 m
pytest                           # 41 pruebas con datos sintéticos (sin video)
```

Sin instalar: `python scripts/run_pipeline.py [--stages ...]`.

Los videos van en `data/raw/video/lateral.mp4` y `data/raw/video/posterior.mp4` (ignorados por git:
pesan >100 MB). El modelo `pose_landmarker_heavy.task` se descarga solo a `models/`.

> **MediaPipe:** se fija `mediapipe>=0.10.30,<1.0`. Las versiones 1.x se cierran al cargar el modelo
> (probado en Python 3.11 y 3.14); 0.10.35 funciona en Python 3.14.

## Estructura

```
config/pipeline.yaml       toda la configuración (sujeto, vistas, calibración, estampitas, QC, fatiga)
data/raw/video/            videos originales (solo lectura)
data/raw/kinovea/          exportaciones antiguas de Kinovea (histórico, no se usan)
data/interim/pose/         pose por fotograma de MediaPipe + estampitas (caché .parquet)
data/processed/<vista>/    marcadores (px y m), cinemática y una carpeta por saque
reports/                   report.md, tables/*.csv, figures/*.png
src/saque/
  video/       reader (ffmpeg), pose (MediaPipe), stickers, ball (fecha el impacto), extract
  processing/  markers (fusión + calibración + QC geométrico), qc, filtering, kinematics
  analysis/    serves, events, jacobian, inverse_dynamics, levers, stats, fatigue
  reporting/   viz, report
  pipeline.py  etapas: pose → kinematics → serves → analysis → report
  cli.py       comando `saque`
tests/         pruebas unitarias y de punta a punta con datos sintéticos
docs/          contexto del proyecto y documento maestro del pipeline
```

## Qué hace cada etapa

| Etapa | Qué hace |
|---|---|
| `pose` | MediaPipe en cada fotograma; busca la estampita alrededor de cada landmark configurado. Usa el **tiempo real** de cada fotograma (pts): los celulares pierden fotogramas. |
| `kinematics` | Fusiona estampita/landmark, calibra con la vara (m, y hacia arriba), QC (saltos, longitudes de segmento, rapidez imposible), Butterworth con corte de Winter por tramos, θ/ω/α y velocidades. |
| `serves` | Candidatos por altura de muñeca; **impacto = fin de la caída de la pelota** (vista lateral); la vista posterior se sincroniza por correlación de la altura de cadera. |
| `analysis` | Por saque: eventos, fases, secuencia, Jacobiano, dinámica inversa. Entre saques: fatiga y variabilidad. |
| `report` | Figuras y `reports/report.md`. |

Detalle de métodos, decisiones y limitaciones: [`docs/pipeline_saque_kick.md`](docs/pipeline_saque_kick.md).

## Estado de los resultados (corrida actual)

- **5 saques detectados** en ambas vistas; el impacto por la pelota se verificó visualmente
  (`reports/figures/impactos_lateral.png`: la pelota está en las cuerdas en los 5).
- **Sincronización**: la vista posterior va 0.64 s atrasada (correlación de cadera 0.75).
- **Estampitas** (vista lateral, lado de golpeo): hombro 96 %, muñeca 80 %, rodilla 74 %, tobillo 72 %,
  codo 71 % de los fotogramas; el resto usa MediaPipe corregido o queda como hueco.
- **Fatiga (exploratorio, n = 5)**: los indicadores de confianza alta (flexión de rodilla, altura de muñeca,
  duración de fases) **no muestran una tendencia concluyente**. Los picos de velocidad angular del brazo
  solo son válidos en 3 de 5 saques (oclusión del brazo cerca del impacto) y no sostienen conclusiones.
- **Calibración** con la vara de 1 m corregida por profundidad (altura de cámara 1.30 m): la vara y la
  jugadora no están a la misma distancia de la cámara. Estatura aparente 1.58 m (lateral) y 1.59 m
  (posterior) frente a 1.64 m reales (−3.5 %, atribuible a la postura nunca del todo erguida).

## Limitaciones

Cinemática 2D por vista (proyecciones), un solo sujeto, impacto fechado con la pelota pero sin medir su
velocidad, dinámica inversa con supuestos fuertes (GRF = 0, muñeca libre). Ver
[`docs/pipeline_saque_kick.md`](docs/pipeline_saque_kick.md) §8.
