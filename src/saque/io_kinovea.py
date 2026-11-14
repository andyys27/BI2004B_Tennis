"""[M2] Lectura y estandarización de exportaciones de Kinovea.

Salida estándar:
  time_s, time_s_raw           tiempo en s (original conservado)
  x_<marcador>, y_<marcador>   coordenadas (m si el CSV traía _cm; si no, unidad 'raw')
  angulo_*  y  angulo_*_rad    ángulos en grados (original) y en radianes
  kin_v_*, kin_a_*             derivadas exportadas por Kinovea (NO confiables: se recalculan)
"""
from __future__ import annotations
import math
import re
import unicodedata
import pandas as pd

MARKER_ALIAS = {"cad": "cadera", "rod": "rodilla", "tob": "tobillo", "mun": "muneca",
                "homb": "hombro"}
_COORD = re.compile(r"^([xy])_(.+?)(_cm)?$")
_DERIV = re.compile(r"^(v|a)_")
TO_M = {"cm": 0.01, "m": 1.0, "mm": 0.001}


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return s.strip().lower().replace(" ", "_")


def standardize(df: pd.DataFrame, coord_unit: str | None = None) -> pd.DataFrame:
    """Normaliza nombres y unidades. `coord_unit` aplica a coordenadas sin sufijo _cm."""
    out = pd.DataFrame(index=df.index)
    raw = {c: _norm(c) for c in df.columns}

    ms_col = next((c for c, n in raw.items() if n in ("tiempo_(ms)", "tiempo_ms", "time_ms")), None)
    s_col = next((c for c, n in raw.items() if n in ("tiempo_s", "time_s")), None)
    if s_col is not None:
        out["time_s"] = df[s_col].astype(float)
        if ms_col is not None:
            out["time_ms_raw"] = df[ms_col].astype(float)
    elif ms_col is not None:
        out["time_s"] = df[ms_col].astype(float) / 1000.0
    else:
        raise ValueError("No se encontró columna de tiempo (tiempo_s/time_s/tiempo_ms).")
    out["time_s_raw"] = out["time_s"]

    used_unit = None
    for c, n in raw.items():
        if c in (ms_col, s_col):
            continue
        m = _COORD.match(n)
        if m and not _DERIV.match(n):
            xy, marker, has_cm = m.group(1), m.group(2), m.group(3)
            marker = MARKER_ALIAS.get(marker, marker)
            if has_cm:
                out[f"{xy}_{marker}"] = df[c].astype(float) * TO_M["cm"]
                used_unit = "m"
            elif coord_unit:
                out[f"{xy}_{marker}"] = df[c].astype(float) * TO_M[coord_unit]
                used_unit = "m"
            else:
                out[f"{xy}_{marker}"] = df[c].astype(float)
                used_unit = used_unit or "raw"
        elif _DERIV.match(n):
            out[f"kin_{n}"] = df[c].astype(float)
        elif n.startswith("angulo") or n.startswith("abs_angulo"):
            out[n] = df[c].astype(float)
            out[f"{n}_rad"] = df[c].astype(float) * math.pi / 180.0
        else:
            out[n] = df[c]
    out.attrs["coord_unit"] = used_unit
    return out


def read_kinovea(path, coord_unit: str | None = None) -> pd.DataFrame:
    df = standardize(pd.read_csv(path), coord_unit=coord_unit)
    df.attrs["source"] = str(path)
    return df
