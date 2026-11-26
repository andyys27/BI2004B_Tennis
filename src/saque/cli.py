"""Línea de comandos: `saque run`, `saque calibrate <vista>`.

  saque run                      # todo el pipeline (la pose queda en caché)
  saque run --stages kinematics serves analysis report
  saque run --force-pose         # repite MediaPipe
  saque calibrate lateral        # guarda un fotograma con rejilla para medir la vara
  saque overlay lateral          # video con esqueleto y marcas (todo el video)
  saque overlay lateral --serve 3 --slow 4   # solo el saque 3, 4x más lento
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from .config import DEFAULT_CONFIG, Paths, load_config  # noqa: E402
from .pipeline import STAGES, run  # noqa: E402


def _calibrate(cfg: dict, view: str, frame: int) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .video.reader import probe, read_frame

    paths = Paths.from_config(cfg)
    info = probe(paths.video(view, cfg))
    img = read_frame(info, frame)
    fig, ax = plt.subplots(figsize=(info.width / 150, info.height / 150), dpi=150)
    ax.imshow(img)
    ax.set_xticks(range(0, info.width, 50))
    ax.set_yticks(range(0, info.height, 50))
    ax.tick_params(labelsize=5)
    ax.grid(color="yellow", lw=0.3)
    c = cfg["views"][view].get("calibration")
    if c:
        ax.plot(*zip(c["p_top"], c["p_bottom"]), "o-", color="red", ms=4, lw=1)
    out = paths.figures / f"calibracion_{view}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _overlay(cfg: dict, a) -> Path:
    import pandas as pd

    from .video.overlay import render_overlay

    paths = Paths.from_config(cfg)
    px_path = paths.processed / a.view / "markers_px.parquet"
    if not px_path.exists():
        raise SystemExit("Primero corre `saque run` (falta markers_px.parquet).")
    px = pd.read_parquet(px_path)
    sv_path = paths.tables / "saques.csv"
    serves = pd.read_csv(sv_path) if sv_path.exists() else None
    if serves is not None:
        serves = serves[serves["vista"] == a.view]
    start, end, name = a.start, a.end, f"overlay_{a.view}"
    if a.serve is not None:
        if serves is None:
            raise SystemExit("No hay saques detectados; corre `saque run` primero.")
        s = serves[serves["saque"] == a.serve].iloc[0]
        start, end, name = s["t_ini"], s["t_fin"], f"overlay_{a.view}_saque{a.serve}"
    out = paths.reports / "videos" / f"{name}.mp4"
    return render_overlay(paths.video(a.view, cfg), px, out, start, end, a.scale, serves, a.slow)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="saque", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="ejecuta el pipeline")
    r.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    r.add_argument("--force-pose", action="store_true", help="repite la extracción de MediaPipe")
    c = sub.add_parser("calibrate", help="fotograma con rejilla para medir la vara de 1 m")
    c.add_argument("view")
    c.add_argument("--frame", type=int, default=150)
    o = sub.add_parser("overlay", help="video con esqueleto y marcas")
    o.add_argument("view")
    o.add_argument("--serve", type=int, help="solo ese saque (1-5)")
    o.add_argument("--start", type=float, default=0.0, help="inicio en s")
    o.add_argument("--end", type=float, default=None, help="fin en s")
    o.add_argument("--slow", type=float, default=1.0, help="factor de cámara lenta")
    o.add_argument("--scale", type=float, default=0.5, help="escala de salida (0.5 = 540x960)")
    a = ap.parse_args(argv)

    if a.cmd == "overlay":
        print(f"Video: {_overlay(load_config(a.config), a)}")
        return

    if a.cmd == "calibrate":
        print(f"Figura: {_calibrate(load_config(a.config), a.view, a.frame)}")
        return
    ctx = run(a.config, a.stages, a.force_pose)
    print(f"Etapas: {', '.join(a.stages)}")
    print(f"Tablas: {ctx.paths.tables}")
    print(f"Reporte: {ctx.paths.reports / 'report.md'}")
    for w in dict.fromkeys(ctx.warnings):
        print(f"Advertencia: {w}")


if __name__ == "__main__":
    main()
