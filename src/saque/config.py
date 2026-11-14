"""Carga de configuración YAML con valores por defecto."""
from __future__ import annotations
import copy
from pathlib import Path
import yaml

DEFAULTS: dict = {
    "resample_hz": 120,
    "flip_y": True,          # Kinovea exporta y hacia abajo; True => y_up = -y
    "filter": {
        "type": "butterworth", "order": 2,
        "cutoff_hz": None,                 # None => análisis de residuos (Winter)
        "cutoff_range_hz": [4, 20],
        "fallback_cutoff_hz": 10.0,
    },
    "qc": {
        "duplicate_time_policy": "first",  # first | last | mean
        "jump_z": 8.0, "jump_min_rel": 0.02,
        "max_gap_s": 0.05,
    },
    "coord_unit_overrides": {},            # {"ruta/archivo.csv": "cm"}
    "groups": {},
}


def _merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path) -> dict:
    p = Path(path)
    return (yaml.safe_load(p.read_text(encoding="utf-8")) or {}) if p.exists() else {}


def load_config(path="config/pipeline.yaml") -> dict:
    return _merge(DEFAULTS, load_yaml(path))
