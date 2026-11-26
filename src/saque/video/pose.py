"""Extracción de pose 2D con MediaPipe Pose Landmarker sobre el video completo.

Salida (una fila por fotograma, guardada en parquet):
  frame, time_s, detected
  x_<lm>, y_<lm>      posición en píxeles (origen arriba-izquierda, y hacia abajo)
  vis_<lm>            visibilidad de MediaPipe [0, 1]
  sx_<lm>, sy_<lm>    centroide de la estampita en px (NaN si no se encontró)

El tiempo es el pts real de cada fotograma (no índice / fps: los celulares pierden fotogramas).
Al cargar, la serie se lleva a una malla uniforme de 1/fps y los fotogramas perdidos quedan
como huecos (NaN), que el QC y el filtrado tratan explícitamente.
"""
from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .reader import VideoInfo, frame_times, iter_frames, probe
from .stickers import StickerParams, find_sticker

# Índices de MediaPipe Pose (33 puntos) -> nombre en español usado en todo el pipeline.
LANDMARKS: dict[str, int] = {
    "nariz": 0,
    "hombro_izq": 11, "hombro_der": 12,
    "codo_izq": 13, "codo_der": 14,
    "muneca_izq": 15, "muneca_der": 16,
    "indice_izq": 19, "indice_der": 20,
    "cadera_izq": 23, "cadera_der": 24,
    "rodilla_izq": 25, "rodilla_der": 26,
    "tobillo_izq": 27, "tobillo_der": 28,
    "talon_izq": 29, "talon_der": 30,
    "punta_izq": 31, "punta_der": 32,
}

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task")


def ensure_model(model_dir: Path, variant: str = "heavy") -> Path:
    """Descarga el modelo .task si no existe localmente."""
    path = Path(model_dir) / f"pose_landmarker_{variant}.task"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL.format(v=variant), tmp)
        tmp.rename(path)
    return path


def _landmarker(model_path: Path, pose_cfg: dict):
    from mediapipe.tasks.python import BaseOptions, vision
    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=float(pose_cfg["min_detection_confidence"]),
        min_tracking_confidence=float(pose_cfg["min_tracking_confidence"]),
    )
    return vision.PoseLandmarker.create_from_options(opts)


def extract_pose(info: VideoInfo, model_path: Path, pose_cfg: dict,
                 sticker_landmarks: list[str] | None = None, sticker_cfg: dict | None = None,
                 max_frames: int | None = None, progress_every: int = 600) -> pd.DataFrame:
    """Corre MediaPipe en todos los fotogramas y busca estampitas en los landmarks indicados."""
    import mediapipe as mp

    names = list(LANDMARKS)
    idx = np.array([LANDMARKS[n] for n in names])
    stk_names = [n for n in (sticker_landmarks or []) if n in LANDMARKS]
    stk_params = StickerParams.from_config(sticker_cfg or {})
    n_total = min(info.n_frames, max_frames or info.n_frames)

    xy = np.full((n_total, len(names), 3), np.nan)          # x_px, y_px, visibilidad
    stk = np.full((n_total, len(stk_names), 2), np.nan)
    frames = np.arange(n_total)
    t_start = time.time()
    with _landmarker(model_path, pose_cfg) as lm:
        for i, rgb in iter_frames(info, 0, n_total):
            ts_ms = int(round(i * 1000.0 / info.fps))
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
            if res.pose_landmarks:
                pts = res.pose_landmarks[0]
                arr = np.array([(p.x * info.width, p.y * info.height, p.visibility) for p in pts])
                xy[i] = arr[idx]
                for k, name in enumerate(stk_names):
                    j = names.index(name)
                    stk[i, k] = find_sticker(rgb, xy[i, j, 0], xy[i, j, 1], stk_params)[:2]
            if progress_every and i and i % progress_every == 0:
                rate = i / (time.time() - t_start)
                print(f"  [{info.path.name}] {i}/{n_total} fotogramas ({rate:.1f} fps)", flush=True)

    cols = {"frame": frames, "time_s": frames / info.fps,
            "detected": np.isfinite(xy[:, 0, 0])}
    for j, n in enumerate(names):
        cols[f"x_{n}"], cols[f"y_{n}"], cols[f"vis_{n}"] = xy[:, j, 0], xy[:, j, 1], xy[:, j, 2]
    for k, n in enumerate(stk_names):
        cols[f"sx_{n}"], cols[f"sy_{n}"] = stk[:, k, 0], stk[:, k, 1]
    df = pd.DataFrame(cols)
    df.attrs.update(fps=info.fps, width=info.width, height=info.height, video=info.path.name)
    return df


def run_extraction(video_path: Path, out_path: Path, model_path: Path, pose_cfg: dict,
                   sticker_landmarks: list[str], sticker_cfg: dict, force: bool = False,
                   max_frames: int | None = None) -> Path:
    """Extrae y guarda la pose de un video; reutiliza el parquet si ya existe (caché)."""
    out_path = Path(out_path)
    if out_path.is_file() and not force:
        return out_path
    info = probe(video_path)
    df = extract_pose(info, model_path, pose_cfg, sticker_landmarks, sticker_cfg, max_frames)
    df["fps"] = info.fps
    df["pts_s"] = frame_times(info)[: len(df)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


def effective_fps(pts: np.ndarray) -> float:
    """Frecuencia real: inversa del intervalo medio entre fotogramas consecutivos (sin saltos).

    La nominal (120) difiere ligeramente de la real (~120.16 fps en estos videos).
    """
    d = np.diff(np.asarray(pts, float))
    normal = d[(d > 0) & (d < 1.5 * np.median(d))]
    return float(1.0 / normal.mean())


def to_uniform_grid(df: pd.DataFrame, fps: float | None = None) -> pd.DataFrame:
    """Reindexa por ranura temporal round(pts * fps); fotogramas perdidos -> filas NaN."""
    fps = fps or effective_fps(df["pts_s"])
    slot = np.round(df["pts_s"].to_numpy(float) * fps).astype(int)
    if np.any(np.diff(slot) <= 0):
        raise ValueError("pts no crecientes o fotogramas duplicados en la misma ranura temporal")
    df = df.assign(slot=slot).set_index("slot").reindex(np.arange(slot[0], slot[-1] + 1))
    df["detected"] = df["detected"].fillna(False).astype(bool)
    df["fps"] = fps
    df["time_s"] = df.index.to_numpy() / fps
    df["frame"] = df["frame"].astype("Int64")      # NaN donde se perdió el fotograma
    return df.reset_index(drop=True)


def load_pose(path: Path, video_path: Path | None = None) -> pd.DataFrame:
    """Pose en malla uniforme. Si el parquet es anterior y no trae pts, los toma del video."""
    df = pd.read_parquet(path)
    if "pts_s" not in df:
        if video_path is None:
            raise ValueError(f"{path} no trae pts_s; pasar video_path para calcularlos")
        df["pts_s"] = frame_times(probe(video_path))[: len(df)]
        df.to_parquet(path, index=False)
    fps = effective_fps(df["pts_s"])
    df = to_uniform_grid(df, fps)
    df.attrs["fps"] = fps
    n_lost = int(df["frame"].isna().sum())
    df.attrs["fotogramas_perdidos"] = n_lost
    return df
