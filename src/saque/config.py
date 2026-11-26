"""Configuración del pipeline: valores por defecto + YAML del proyecto.

Todas las rutas se resuelven respecto a la raíz del proyecto (carpeta que contiene `config/`),
de modo que el pipeline funciona igual sin importar desde dónde se ejecute.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "pipeline.yaml"

DEFAULTS: dict = {
    "subject": {"height_m": 1.64, "body_mass_kg": 58.0, "foot_length_m": 0.25,
                "hitting_side": "auto"},
    "views": {},
    "pose": {
        "model": "heavy",
        "model_dir": "models",
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5,
        "min_visibility": 0.5,
    },
    "stickers": {
        "enabled": True,
        "hsv_low": [25, 170, 90],
        "hsv_high": [45, 255, 255],
        "search_radius_px": 45,
        "min_area_px": 40,
        "max_area_px": 1500,
        "min_circularity": 0.55,
    },
    "serves": {
        "expected": 5,
        "min_separation_s": 3.0,
        "pre_impact_s": 1.6,
        "post_impact_s": 0.8,
        "manual_impacts_s": {},
    },
    "filter": {
        "type": "butterworth",
        "order": 2,
        "cutoff_hz": None,
        "cutoff_range_hz": [4, 20],
        "fallback_cutoff_hz": 10.0,
    },
    "qc": {"jump_z": 8.0, "jump_min_rel": 0.02, "max_gap_s": 0.05, "segment_length_tol": 0.6,
           "segment_jump_tol": 0.30,
           "max_marker_speed_m_s": 15.0, "max_omega_rad_s": 60.0},
    "kinematics": {},
    "events": {"start_frac": 0.05, "end_frac": 0.20, "sequence_margin_s": 0.05,
               "sequence_min_valid_frac": 0.5},
    "fatigue": {"variables": {}},
    "body": {"com_cutoff_hz": 6.0, "segments": None,
             "posture_events": ["inicio", "inicio_flexion_rodilla", "max_flex_rodilla", "impacto", "fin"],
             "lever_view": "lateral", "lever_events": ["max_flex_rodilla", "impacto"],
             "lever_joints": ["tobillo_der", "rodilla_der", "cadera_der"]},
    "inverse_dynamics": {"anthropometry": {}},
    "paths": {
        "raw_video": "data/raw/video",
        "pose": "data/interim/pose",
        "processed": "data/processed",
        "reports": "reports",
    },
}


def _merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_yaml(path) -> dict:
    p = Path(path)
    return (yaml.safe_load(p.read_text(encoding="utf-8")) or {}) if p.exists() else {}


def load_config(path=DEFAULT_CONFIG) -> dict:
    return _merge(DEFAULTS, load_yaml(path))


@dataclass(frozen=True)
class Paths:
    """Rutas absolutas del proyecto derivadas de la configuración."""

    root: Path
    raw_video: Path
    pose: Path
    processed: Path
    reports: Path
    models: Path

    @classmethod
    def from_config(cls, cfg: dict, root: Path | None = None) -> "Paths":
        root = Path(root or PROJECT_ROOT)
        p = cfg["paths"]
        return cls(root=root, raw_video=root / p["raw_video"], pose=root / p["pose"],
                   processed=root / p["processed"], reports=root / p["reports"],
                   models=root / cfg["pose"]["model_dir"])

    @property
    def tables(self) -> Path:
        return self.reports / "tables"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    def video(self, view: str, cfg: dict) -> Path:
        return self.raw_video / cfg["views"][view]["video"]
