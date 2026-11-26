"""Atajo equivalente a `saque run` sin instalar el paquete: python scripts/run_pipeline.py [...]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from saque.cli import main  # noqa: E402

main(["run", *sys.argv[1:]] if len(sys.argv) == 1 or sys.argv[1].startswith("-") else sys.argv[1:])
