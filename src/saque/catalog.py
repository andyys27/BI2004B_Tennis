"""[M1] Inventario de CSV: hash, rango temporal y detección de duplicados."""
from __future__ import annotations
import hashlib
from pathlib import Path
import pandas as pd

TIME_NAMES = ("tiempo_s", "time_s")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def infer_view(rel: str) -> str:
    low = rel.lower()
    return "lateral" if "lateral" in low else "posterior" if "posterior" in low else "desconocida"


def _time_range(df: pd.DataFrame):
    cols = {c.strip().lower(): c for c in df.columns}
    for n in TIME_NAMES:
        if n in cols:
            s = df[cols[n]]
            return float(s.iloc[0]), float(s.iloc[-1])
    for n in ("tiempo_ms", "tiempo (ms)"):
        if n in cols:
            s = df[cols[n]] / 1000.0
            return float(s.iloc[0]), float(s.iloc[-1])
    return None, None


def build_catalog(root) -> pd.DataFrame:
    root = Path(root)
    rows = []
    for p in sorted(root.rglob("*.csv")):
        df = pd.read_csv(p)
        rel = p.relative_to(root).as_posix()
        t0, t1 = _time_range(df)
        rows.append({
            "archivo": rel, "vista": infer_view(rel),
            "n_filas": len(df), "columnas": "|".join(df.columns),
            "t_ini": t0, "t_fin": t1, "nan_total": int(df.isna().sum().sum()),
            "hash": sha256(p),
        })
    cat = pd.DataFrame(rows)
    if cat.empty:
        return cat.assign(duplicado_de=None, usar=True)
    first = cat.groupby("hash")["archivo"].transform("first")
    cat["duplicado_de"] = first.where(first != cat["archivo"])
    cat["usar"] = cat["duplicado_de"].isna()
    return cat


def valid_files(cat: pd.DataFrame) -> list[str]:
    return cat.loc[cat["usar"], "archivo"].tolist()
