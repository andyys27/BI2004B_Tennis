"""Etapa 2: de la pose en píxeles a marcadores calibrados en metros.

1. Landmarks con visibilidad baja -> hueco (NaN).
2. Fusión estampita/landmark: si hay estampita válida se usa su centroide; si no, el landmark de
   MediaPipe desplazado por el desfase mediano estampita-landmark (evita saltos al cambiar de
   fuente). Detecciones con desfase atípico (MAD) se descartan como falsos positivos.
3. Calibración con la vara: escala m/px y origen en el extremo inferior; y hacia arriba.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..video.pose import LANDMARKS
from .filtering import interpolate_short_gaps


@dataclass(frozen=True)
class Calibration:
    """Transformación píxel -> metro a partir de una referencia de longitud conocida."""

    origin_px: tuple[float, float]
    m_per_px: float

    @classmethod
    def from_reference(cls, p_top, p_bottom, length_m: float = 1.0) -> "Calibration":
        p_top, p_bottom = np.asarray(p_top, float), np.asarray(p_bottom, float)
        dist = float(np.hypot(*(p_top - p_bottom)))
        if dist <= 0 or length_m <= 0:
            raise ValueError("La referencia de calibración debe tener longitud positiva")
        return cls(tuple(p_bottom), length_m / dist)

    @classmethod
    def from_config(cls, view_cfg: dict, markers_px: pd.DataFrame | None = None) -> "Calibration":
        """`method: pole` usa la escala de la vara; `ground_plane` la lleva al plano de la jugadora."""
        c = view_cfg["calibration"]
        method = c.get("method", "pole")
        if method == "pole":
            return cls.from_reference(c["p_top"], c["p_bottom"], float(c.get("length_m", 1.0)))
        if method == "ground_plane":
            if markers_px is None:
                raise ValueError("ground_plane requiere los marcadores en px")
            return cls.ground_plane(c["p_top"], c["p_bottom"], float(c.get("length_m", 1.0)),
                                    float(c["camera_height_m"]), standing_foot_y_px(markers_px))
        raise ValueError(f"Método de calibración desconocido: {method!r}")

    @classmethod
    def ground_plane(cls, p_top, p_bottom, length_m: float, camera_height_m: float,
                     foot_y_px: tuple[float, float]) -> "Calibration":
        """Escala en el plano de la jugadora a partir de la vara y la altura de la cámara.

        Cámara de pinhole nivelada a altura H: un punto del suelo a distancia Z se proyecta en
        y = y_h + f·H/Z y 1 m a esa distancia mide f/Z px. Con la vara (base y_b, s_v px/m):
        horizonte y_h = y_b − H·s_v. Con los pies de la jugadora en y_f: s_j = (y_f − y_h)/H.
        Corrige que la vara esté más cerca o más lejos de la cámara que la jugadora.
        Origen: x de la jugadora, y del suelo bajo sus pies.
        """
        pole = cls.from_reference(p_top, p_bottom, length_m)
        s_pole = 1.0 / pole.m_per_px
        y_horizon = float(p_bottom[1]) - camera_height_m * s_pole
        x_f, y_f = foot_y_px
        s_player = (y_f - y_horizon) / camera_height_m
        if s_player <= 0:
            raise ValueError("Los pies quedan por encima del horizonte: revisar altura de cámara/vara")
        return cls((float(x_f), float(y_f)), 1.0 / s_player)

    def to_m(self, x_px, y_px) -> tuple[np.ndarray, np.ndarray]:
        x0, y0 = self.origin_px
        return ((np.asarray(x_px, float) - x0) * self.m_per_px,
                (y0 - np.asarray(y_px, float)) * self.m_per_px)


def _robust_offset(d: np.ndarray, k: float) -> tuple[np.ndarray, np.ndarray]:
    """Desfase mediano (2,) y máscara de detecciones no atípicas."""
    ok = np.isfinite(d).all(axis=1)
    if ok.sum() < 5:
        return np.zeros(2), ok
    med = np.median(d[ok], axis=0)
    dist = np.hypot(*(d - med).T)
    mad = max(float(np.median(dist[ok])), 2.0)
    return med, ok & (dist <= k * mad)


def _fill_short(a: np.ndarray, max_gap: int) -> np.ndarray:
    """Interpola linealmente huecos internos de hasta `max_gap` filas (por columna)."""
    if max_gap <= 0:
        return a
    return np.column_stack([interpolate_short_gaps(a[:, k], max_gap) for k in range(a.shape[1])])


def fuse_markers(pose: pd.DataFrame, min_visibility: float = 0.5, max_offset_mad: float = 4.0,
                 duplicate_px: float = 10.0, sticker_gap_frames: int = 6,
                 sticker_markers: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Coordenadas en px por marcador con la fuente usada; devuelve (datos, resumen).

    Si una misma estampita quedó asignada a dos marcadores en un fotograma (sus ventanas de
    búsqueda se solapan, p. ej. hombro y codo con el brazo pegado al cuerpo, o la muñeca en reposo
    junto a la cadera), se asigna al marcador cuyo landmark esperado está más cerca; el otro usa
    el landmark de MediaPipe en ese fotograma. Los huecos cortos de la estampita (<= sticker_gap_frames)
    se interpolan en vez de saltar a MediaPipe: cada cambio de fuente introduce un pequeño salto de
    posición que la derivada convierte en un pico de velocidad.
    """
    out = pd.DataFrame({"frame": pose["frame"].to_numpy(), "time_s": pose["time_s"].to_numpy()})
    names = [n for n in LANDMARKS if f"x_{n}" in pose]
    lm, stk, offset = {}, {}, {}
    for name in names:
        a = pose[[f"x_{name}", f"y_{name}"]].to_numpy(float, copy=True)
        a[pose[f"vis_{name}"].to_numpy(float) < min_visibility] = np.nan
        lm[name] = a
        use = sticker_markers is None or name in sticker_markers
        if use and f"sx_{name}" in pose:
            raw = pose[[f"sx_{name}", f"sy_{name}"]].to_numpy(float, copy=True)
            off, good = _robust_offset(raw - pose[[f"x_{name}", f"y_{name}"]].to_numpy(float),
                                       max_offset_mad)
            raw[~good] = np.nan
            stk[name], offset[name] = raw, off
    # Una estampita física pertenece a un solo marcador: si dos la tomaron, se queda con el
    # marcador cuyo landmark (+ desfase) está más cerca; el otro vuelve a MediaPipe.
    duplicated = {n: np.zeros(len(pose), bool) for n in stk}
    expected = {n: lm[n] + offset[n] for n in stk}
    keys = list(stk)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            same = np.hypot(*(stk[a] - stk[b]).T) < duplicate_px
            if not same.any():
                continue
            da = np.hypot(*(stk[a] - expected[a]).T)
            db = np.hypot(*(stk[b] - expected[b]).T)
            a_wins = np.where(np.isfinite(da) & np.isfinite(db), da <= db, np.isfinite(da))
            duplicated[a] |= same & ~a_wins
            duplicated[b] |= same & a_wins
    summary = []
    for name in names:
        xy = lm[name].copy()
        source = np.where(np.isfinite(xy).all(axis=1), "pose", "hueco").astype(object)
        rejected = dup = 0
        if name in stk:
            raw = pose[[f"sx_{name}", f"sy_{name}"]].to_numpy(float)
            clean = stk[name].copy()
            clean[duplicated[name]] = np.nan
            clean = _fill_short(clean, sticker_gap_frames)
            good = np.isfinite(clean).all(axis=1)
            rejected = int((np.isfinite(raw).all(axis=1) & ~np.isfinite(stk[name]).all(axis=1)).sum())
            dup = int(duplicated[name].sum())
            xy = np.where(np.isfinite(xy), xy + offset[name], np.nan)
            xy[good] = clean[good]
            source[good] = "estampita"
        out[f"x_{name}"], out[f"y_{name}"] = xy[:, 0], xy[:, 1]
        out[f"src_{name}"] = source
        n = len(source)
        summary.append(dict(marcador=name, con_estampita=name in stk,
                            frac_estampita=float((source == "estampita").sum() / n),
                            frac_pose=float((source == "pose").sum() / n),
                            frac_hueco=float((source == "hueco").sum() / n),
                            estampitas_descartadas=rejected, estampitas_duplicadas=dup))
    return out, pd.DataFrame(summary)


def mask_segment_length(markers: pd.DataFrame, segments: dict, tol: float = 0.6,
                        jump_tol: float = 0.15, log=None, name: str = "") -> pd.DataFrame:
    """Huecos donde la longitud proyectada de un segmento cambia de forma no anatómica.

    En 2D la longitud proyectada varía *gradualmente* cuando el segmento gira fuera del plano
    (escorzo); un marcador mal ubicado (oclusión, estampita equivocada) la cambia de *golpe*.
    Se marcan los fotogramas con salto entre fotogramas > `jump_tol` de la mediana, o con
    longitud fuera de ±`tol` de la mediana (error grosero). Se anulan ambos extremos.
    """
    out = markers.copy()
    for seg, (a, b) in segments.items():
        if not {f"x_{a}", f"x_{b}"} <= set(out):
            continue
        L = np.hypot(out[f"x_{a}"] - out[f"x_{b}"], out[f"y_{a}"] - out[f"y_{b}"]).to_numpy(float)
        med = np.nanmedian(L)
        jump = np.abs(np.diff(L)) > jump_tol * med
        bad = (np.isfinite(L) & (np.abs(L / med - 1) > tol)) | np.r_[False, jump] | np.r_[jump, False]
        if bad.any():
            for m in (a, b):
                out.loc[bad, [f"x_{m}", f"y_{m}"]] = np.nan
            if log is not None:
                log.add(name, "longitud_segmento", f"{seg}: {int(bad.sum())} fotogramas a NaN",
                        detalle=f"salto >{jump_tol:.0%} o fuera de ±{tol:.0%} de {med:.3g}")
    return out


def to_metric(markers_px: pd.DataFrame, calib: Calibration) -> pd.DataFrame:
    """Convierte todas las columnas x_/y_ a metros (y hacia arriba)."""
    out = markers_px.copy()
    for c in [c for c in out if c.startswith("x_")]:
        m = c[2:]
        out[f"x_{m}"], out[f"y_{m}"] = calib.to_m(out[f"x_{m}"], out[f"y_{m}"])
    out.attrs["coord_unit"] = "m"
    return out


FOOT = ("talon_izq", "talon_der", "punta_izq", "punta_der")
NOSE_HEIGHT_FRACTION = 0.92     # altura de la nariz / estatura (antropometría aproximada)


def standing_foot_y_px(markers_px: pd.DataFrame) -> tuple[float, float]:
    """(x, y) en px del contacto con el suelo en la postura más erguida (p95 nariz-pie)."""
    cols = [f"y_{m}" for m in FOOT if f"y_{m}" in markers_px]
    foot = np.nanmax(markers_px[cols].to_numpy(float), axis=1)       # y crece hacia abajo
    span = foot - markers_px["y_nariz"].to_numpy(float)
    ok = np.isfinite(span)
    sel = ok & (span >= 0.98 * np.nanpercentile(span[ok], 95))
    x = np.nanmedian(markers_px.loc[sel, [c.replace("y_", "x_") for c in cols]].to_numpy(float))
    return float(x), float(np.nanmedian(foot[sel]))


def implied_height_m(markers_m: pd.DataFrame) -> float:
    """Estatura aparente: nariz -> pie en la postura más erguida / fracción de altura de la nariz.

    Control de la escala (y hacia arriba, en m). Subestima un poco si la jugadora nunca está
    completamente erguida (rodillas flexionadas, cabeza inclinada).
    """
    cols = [f"y_{m}" for m in FOOT if f"y_{m}" in markers_m]
    if "y_nariz" not in markers_m or not cols:
        return float("nan")
    span = (markers_m["y_nariz"] - markers_m[cols].min(axis=1)).to_numpy(float)
    return float(np.nanpercentile(span, 95) / NOSE_HEIGHT_FRACTION) if np.isfinite(span).any() else float("nan")


def mask_speed(markers: pd.DataFrame, fs: float, max_speed_m_s: float, log=None,
               name: str = "") -> pd.DataFrame:
    """Huecos en muestras cuyo desplazamiento entre fotogramas implica una rapidez imposible.

    Un salto de marcador (cambio de estampita, error de MediaPipe) produce un desplazamiento de
    ida y otro de vuelta; se anula la muestra entre ambos.
    """
    out = markers.copy()
    for c in [c for c in out if c.startswith("x_")]:
        m = c[2:]
        step = np.hypot(np.diff(out[f"x_{m}"]), np.diff(out[f"y_{m}"])) * fs
        fast = np.r_[False, step > max_speed_m_s] | np.r_[step > max_speed_m_s, False]
        if fast.any():
            out.loc[fast, [f"x_{m}", f"y_{m}"]] = np.nan
            if log is not None:
                log.add(name, "rapidez_imposible", f"{m}: {int(fast.sum())} muestras a NaN",
                        detalle=f">{max_speed_m_s} m/s")
    return out
