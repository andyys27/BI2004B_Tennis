"""Etapa 1: extracción de pose de todas las vistas (la más costosa; resultado en caché)."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..config import Paths
from .pose import ensure_model, run_extraction


def _one(args):
    return run_extraction(*args)


def extract_all(cfg: dict, paths: Paths, views: list[str] | None = None,
                force: bool = False, parallel: bool = True) -> dict[str, Path]:
    """Extrae la pose de cada vista configurada; las vistas se procesan en paralelo."""
    model = ensure_model(paths.models, cfg["pose"]["model"])
    views = views or list(cfg["views"])
    jobs = {}
    for v in views:
        vcfg = cfg["views"][v]
        sticker_cfg = {**cfg["stickers"], **vcfg.get("sticker_params", {})}
        stickers = vcfg.get("stickers", []) if cfg["stickers"].get("enabled", True) else []
        jobs[v] = (paths.video(v, cfg), paths.pose / f"{v}.parquet", model, cfg["pose"],
                   stickers, sticker_cfg, force)
    if parallel and len(jobs) > 1:
        with ProcessPoolExecutor(len(jobs)) as ex:
            return dict(zip(jobs, ex.map(_one, jobs.values())))
    return {v: _one(a) for v, a in jobs.items()}
