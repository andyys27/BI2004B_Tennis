"""Prueba de punta a punta con pose sintética (sin video ni MediaPipe)."""
import numpy as np
import pandas as pd
import yaml

from saque.config import DEFAULT_CONFIG
from saque.pipeline import run
from tests import synth

IMPACTS = (3.0, 8.0, 13.0)


def _setup(tmp_path):
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    cfg["serves"].update(expected=len(IMPACTS), impact_method="wrist", min_separation_s=3.0)
    cfg["inverse_dynamics"]["enabled"] = False
    for v in cfg["views"].values():
        v["calibration"] = {"p_top": [540, 1300], "p_bottom": [540, 1800], "length_m": 1.0}
    cfg_path = tmp_path / "pipeline.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    pose_dir = tmp_path / "data/interim/pose"
    pose_dir.mkdir(parents=True)
    synth.pose_frame(IMPACTS, lost_frames=range(1, 24)).to_parquet(pose_dir / "lateral.parquet")
    # La vista posterior va 0.5 s "atrasada": mismo gesto, reloj desplazado.
    synth.pose_frame(tuple(t + 0.5 for t in IMPACTS), seed=3).to_parquet(pose_dir / "posterior.parquet")
    return cfg_path


def test_pipeline_end_to_end(tmp_path):
    cfg_path = _setup(tmp_path)
    ctx = run(cfg_path, stages=("kinematics", "serves", "analysis", "report"), root=tmp_path)
    tables = tmp_path / "reports/tables"

    cal = pd.read_csv(tables / "calibracion.csv").set_index("vista")
    assert cal.loc["lateral", "fotogramas_perdidos"] == 23

    sv = pd.read_csv(tables / "saques.csv")
    lat = sv[sv["vista"] == "lateral"].sort_values("saque")
    assert len(lat) == len(IMPACTS)
    assert np.allclose(lat["t_altura"], IMPACTS, atol=0.05)

    sync = pd.read_csv(tables / "sincronizacion.csv").iloc[0]
    assert abs(sync["lag_s"] - 0.5) < 0.02

    for name in ("fatiga_valores_por_saque", "fatiga_tendencias", "events", "phases"):
        assert (tables / f"{name}.csv").exists(), name
    assert (tmp_path / "reports/report.md").read_text().startswith("# Reporte")
    assert (tmp_path / "reports/figures/fatiga_tendencias.png").exists()
    assert (tmp_path / "data/processed/lateral/saque_01/kinematics.csv").exists()
    assert ctx.paths.root == tmp_path
