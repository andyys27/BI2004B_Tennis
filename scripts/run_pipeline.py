"""Uso: python scripts/run_pipeline.py --raw data/raw --use-hypothesis"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from saque.pipeline import run

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="data/raw")
ap.add_argument("--config", default="config/pipeline.yaml")
ap.add_argument("--offsets", default="config/sync_offsets.yaml")
ap.add_argument("--out", default=".")
ap.add_argument("--use-hypothesis", action="store_true",
                help="Aplica offsets NO verificados (solo exploratorio; quedan marcados en decisiones_sync.csv)")
a = ap.parse_args()
r = run(a.raw, a.config, a.offsets, a.out, a.use_hypothesis)
print(f"Grupos procesados: {list(r['results'])}")
print(f"Eventos de QC: {len(r['log'])}  (ver reports/tables/qc_log.csv)")
