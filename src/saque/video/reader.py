"""Lectura de video con ffmpeg.

Se usa ffmpeg (y no cv2.VideoCapture) porque:
- respeta el metadato de rotación de los celulares (los videos están grabados en vertical);
- decodifica HEVC/H.264 sin depender de cómo se compiló OpenCV;
- entrega fotogramas RGB contiguos listos para MediaPipe.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int          # ancho ya rotado (como se ve el video)
    height: int
    fps: float
    n_frames: int
    duration_s: float
    codec: str
    rotation: int

    @property
    def frame_bytes(self) -> int:
        return self.width * self.height * 3


def _require(tool: str) -> str:
    exe = shutil.which(tool)
    if exe is None:
        raise RuntimeError(f"No se encontró `{tool}` en el PATH; instala ffmpeg.")
    return exe


def probe(path) -> VideoInfo:
    """Metadatos del primer stream de video (dimensiones ya corregidas por rotación)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    cmd = [_require("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=width,height,r_frame_rate,nb_frames,codec_name:stream_side_data=rotation"
           ":format=duration", "-of", "json", str(path)]
    meta = json.loads(subprocess.run(cmd, capture_output=True, check=True, text=True).stdout)
    s = meta["streams"][0]
    rotation = int(next((d["rotation"] for d in s.get("side_data_list", []) if "rotation" in d), 0))
    w, h = int(s["width"]), int(s["height"])
    if abs(rotation) % 180 == 90:
        w, h = h, w
    fps = float(Fraction(s["r_frame_rate"]))
    duration = float(meta["format"]["duration"])
    n = int(s.get("nb_frames") or round(duration * fps))
    return VideoInfo(path, w, h, fps, n, duration, s["codec_name"], rotation)


def frame_times(info: VideoInfo) -> np.ndarray:
    """Tiempo de presentación real (s) de cada fotograma, desde el primero.

    Los celulares pueden perder fotogramas (p. ej. un salto de 0.19 s tras el primero): el tiempo
    real NO es índice / fps. Se usan los pts de los paquetes ordenados (orden de presentación).
    """
    cmd = [_require("ffprobe"), "-v", "error", "-select_streams", "v:0",
           "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(info.path)]
    out = subprocess.run(cmd, capture_output=True, check=True, text=True).stdout
    pts = np.sort(np.array([float(x) for x in out.split() if x and x != "N/A"]))
    return pts - pts[0]


def iter_frames(info: VideoInfo, start_frame: int = 0, n_frames: int | None = None,
                scale: float = 1.0, times: np.ndarray | None = None
                ) -> Iterator[tuple[int, np.ndarray]]:
    """Itera (índice_de_fotograma, imagen RGB uint8 HxWx3) desde `start_frame`.

    Para empezar a mitad del video se busca por el pts real del fotograma (`times`, de
    `frame_times`); sin él se calcula y se usa.
    """
    w, h = int(round(info.width * scale)) // 2 * 2, int(round(info.height * scale)) // 2 * 2
    cmd = [_require("ffmpeg"), "-v", "error", "-nostdin"]
    if start_frame:
        times = frame_times(info) if times is None else times
        cmd += ["-ss", f"{times[start_frame]:.6f}"]
    cmd += ["-i", str(info.path)]
    if n_frames is not None:
        cmd += ["-frames:v", str(int(n_frames))]
    if scale != 1.0:
        cmd += ["-vf", f"scale={w}:{h}"]
    cmd += ["-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    nbytes = w * h * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=nbytes * 2)
    try:
        i = start_frame
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            yield i, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            i += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def read_frame(info: VideoInfo, frame: int, times: np.ndarray | None = None) -> np.ndarray:
    """Un solo fotograma RGB por índice (útil para calibración y figuras)."""
    for _, img in iter_frames(info, frame, 1, times=times):
        return img.copy()
    raise IndexError(f"Fotograma {frame} fuera del video {info.path.name}")
