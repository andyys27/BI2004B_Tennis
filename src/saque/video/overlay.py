"""Video de control: esqueleto y marcadores dibujados sobre el video original.

Colores (BGR de OpenCV sobre RGB): naranja = estampita detectada, azul = MediaPipe,
sin punto = hueco. Arriba a la izquierda: tiempo, fotograma y número de saque si aplica.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .reader import frame_times, iter_frames, probe

BONES = [("hombro", "codo"), ("codo", "muneca"), ("hombro", "cadera"), ("cadera", "rodilla"),
         ("rodilla", "tobillo"), ("tobillo", "punta"), ("tobillo", "talon"), ("talon", "punta")]
CROSS = [("hombro_izq", "hombro_der"), ("cadera_izq", "cadera_der")]
ORANGE, BLUE, WHITE = (244, 114, 48), (37, 125, 224), (255, 255, 255)


def draw(img: np.ndarray, row: pd.Series) -> np.ndarray:
    """Dibuja esqueleto y marcadores de una fila de markers_px sobre una copia de la imagen."""
    out = img.copy()

    def pt(m):
        x, y = row.get(f"x_{m}"), row.get(f"y_{m}")
        return (int(x), int(y)) if pd.notna(x) and pd.notna(y) else None

    pairs = [(f"{a}_{s}", f"{b}_{s}") for s in ("izq", "der") for a, b in BONES] + CROSS
    for a, b in pairs:
        pa, pb = pt(a), pt(b)
        if pa and pb:
            cv2.line(out, pa, pb, WHITE, 3, cv2.LINE_AA)
    for c in [c for c in row.index if c.startswith("src_")]:
        p = pt(c[4:])
        if p and row[c] != "hueco":
            color = ORANGE if row[c] == "estampita" else BLUE
            cv2.circle(out, p, 9, WHITE, -1, cv2.LINE_AA)
            cv2.circle(out, p, 7, color, -1, cv2.LINE_AA)
    return out


def render_overlay(video_path: Path, markers_px: pd.DataFrame, out_path: Path,
                   start_s: float = 0.0, end_s: float | None = None, scale: float = 0.5,
                   serves: pd.DataFrame | None = None, slow: float = 1.0) -> Path:
    """Escribe un mp4 (H.264) con las marcas. `slow` = 4 reproduce 4x más lento."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("No se encontró ffmpeg en el PATH")
    info = probe(video_path)
    times = frame_times(info)
    end_s = times[-1] if end_s is None else min(end_s, times[-1])
    f0 = int(np.searchsorted(times, start_s))
    f1 = int(np.searchsorted(times, end_s))
    by_frame = markers_px.dropna(subset=["frame"]).set_index(markers_px["frame"].dropna().astype(int))
    w, h = int(info.width * scale) // 2 * 2, int(info.height * scale) // 2 * 2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", f"{info.fps / slow:.4f}", "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for i, img in iter_frames(info, f0, f1 - f0, times=times):
            frame = draw(img, by_frame.loc[i]) if i in by_frame.index else img.copy()
            t = times[i]
            label = f"t={t:6.2f}s  f={i}"
            if serves is not None:
                hit = serves[(serves["t_ini"] <= t) & (serves["t_fin"] >= t)]
                if len(hit):
                    s = hit.iloc[0]
                    label += f"  saque {int(s['saque'])}"
                    if abs(t - s["t_impacto"]) < 0.5 / info.fps * 2:
                        label += "  IMPACTO"
            cv2.putText(frame, label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 8, cv2.LINE_AA)
            cv2.putText(frame, label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, WHITE, 3, cv2.LINE_AA)
            small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            proc.stdin.write(np.ascontiguousarray(small).tobytes())
    finally:
        proc.stdin.close()
        proc.wait()
    return out_path
