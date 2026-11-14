"""Ejecuta M1–M8, M10 y M12–M14 desde las salidas procesadas."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from saque.config import load_config
from saque.inverse_dynamics import run_inverse_dynamics, run_upper_limb_inverse_dynamics
from saque.pipeline import run, run_events
from saque.report import run_report
from saque.stats import run_statistics
from saque.viz import run_visualizations

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="data/raw")
ap.add_argument("--config", default="config/pipeline.yaml")
ap.add_argument("--offsets", default="config/sync_offsets.yaml")
ap.add_argument("--out", default=".")
ap.add_argument("--use-hypothesis", action="store_true",
                help="Aplica offsets NO verificados (solo exploratorio; quedan marcados en decisiones_sync.csv)")
a = ap.parse_args()
r = run(a.raw, a.config, a.offsets, a.out, a.use_hypothesis)
e = run_events(a.out, a.config)
m = run_statistics(a.out, a.config)
id_result = run_inverse_dynamics(a.out, a.config, a.raw)
upper_id_result = run_upper_limb_inverse_dynamics(a.out, a.config)
v = run_visualizations(a.out, load_config(a.config))
warnings = (e["warnings"] + m["warnings"] + id_result["warnings"]
            + upper_id_result["warnings"] + v["warnings"])
report = run_report(a.out, warnings=warnings)
out = Path(a.out)
print(f"Grupos procesados: {list(r['results'])}")
print(f"Eventos de QC: {len(r['log'])}  (ver {out / 'reports/tables/qc/qc_log.csv'})")
print(f"Jacobianos validados: {len(r['jacobian'])}  (ver {out / 'data/processed/jacobian'})")
print(f"Eventos M7: {len(e['events'])}  (ver {out / 'reports/tables/events'})")
print(f"Resúmenes M10: {len(m['descriptives'])}  (ver {out / 'reports/tables/statistics'})")
print(f"Dinámica inversa M12: {id_result['status']} — {id_result['message']}")
print(f"Dinámica inversa M12 tren superior: {upper_id_result['status']} — {upper_id_result['message']}")
print(f"Figuras M13 generadas: {len(v['figures'])}  (ver {out / 'reports/figures'})")
print(f"Reporte M14: {report['report']}")
for warning in warnings:
    print(f"Advertencia: {warning}")
