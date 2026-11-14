"""Orquestación M1→M6: catálogo → lectura → QC → sync → filtrado → cinemática."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from .catalog import build_catalog, valid_files
from .config import load_config, load_yaml
from .io_kinovea import read_kinovea
from .qc import QCLog, clean_time, fill_short_gaps, mask_outliers, time_report
from .sync import apply_offset, merge_on_grid
from .filtering import butter_lp, savgol, winter_cutoff
from .kinematics import compute_kinematics


def _auto_angle_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("angulo_") and c.split("__")[0].endswith("_rad")
            and "_ma_" not in c and "_clean" not in c]


def _filter_series(x, fs, fcfg, label, rows):
    """Filtra una serie; elige fc por Winter si no hay fc fija. Registra la decisión."""
    if fcfg.get("type") == "savgol":
        rows.append(dict(serie=label, fc_hz=None, metodo="savgol"))
        return savgol(x, fs, fcfg.get("savgol_window", 15), fcfg.get("savgol_poly", 3))
    if fcfg.get("cutoff_hz"):
        fc, metodo = float(fcfg["cutoff_hz"]), "fijo"
    else:
        w = winter_cutoff(x, fs, tuple(fcfg["cutoff_range_hz"]), fallback=fcfg["fallback_cutoff_hz"])
        fc, metodo = w["fc"], w["metodo"]
    rows.append(dict(serie=label, fc_hz=round(fc, 2), metodo=metodo))
    return butter_lp(x, fs, fc, fcfg["order"])


def run(raw_dir="data/raw", config="config/pipeline.yaml", offsets="config/sync_offsets.yaml",
        out_dir=".", use_hypothesis=False) -> dict:
    cfg = load_config(config)
    off_cfg = load_yaml(offsets)
    raw_dir, out = Path(raw_dir), Path(out_dir)
    (out / "data/interim").mkdir(parents=True, exist_ok=True)
    (out / "data/processed").mkdir(parents=True, exist_ok=True)
    (out / "reports/tables").mkdir(parents=True, exist_ok=True)
    fs, log = float(cfg["resample_hz"]), QCLog()

    # M1
    cat = build_catalog(raw_dir)
    cat.drop(columns="columnas").to_csv(out / "reports/tables/catalogo.csv", index=False)
    usable = set(valid_files(cat))
    for _, r in cat[~cat["usar"]].iterrows():
        log.add(r["archivo"], "archivo_duplicado", f"excluido (idéntico a {r['duplicado_de']})")

    decisions, filt_rows, time_rows, results = [], [], [], {}
    for gname, g in cfg["groups"].items():
        files = g["files"]
        missing = [f for f in files if f not in usable]
        if missing:
            log.add(gname, "grupo_omitido", f"archivos no disponibles: {missing}")
            continue

        # M2–M4 por archivo
        series, skip = {}, False
        for f in files:
            unit = cfg["coord_unit_overrides"].get(f)
            df = read_kinovea(raw_dir / f, coord_unit=unit)
            df = clean_time(df, f, log, cfg["qc"]["duplicate_time_policy"])             # M3
            df = mask_outliers(df, f, log, cfg["qc"]["jump_z"], cfg["qc"]["jump_min_rel"])
            df = fill_short_gaps(df, f, log, cfg["qc"]["max_gap_s"])
            time_rows.append(dict(archivo=f, **time_report(df)))
            entry = off_cfg.get(f, {})
            if entry.get("offset_s") is not None:                                         # M4
                off, estado = float(entry["offset_s"]), "verificado"
            elif entry.get("hypothesis_s") is not None:
                if not use_hypothesis:
                    log.add(gname, "grupo_omitido", f"offset sin verificar en {f} (usar --use-hypothesis)")
                    skip = True
                    break
                off, estado = float(entry["hypothesis_s"]), "HIPOTESIS_NO_VERIFICADA"
            else:
                off, estado = 0.0, "sin_entrada"
            decisions.append(dict(grupo=gname, archivo=f, offset_s=off, estado=estado))
            series[f] = apply_offset(df, off)
        if skip:
            continue
        merged, info = merge_on_grid({Path(k).stem: v for k, v in series.items()}, fs)
        merged.attrs["coord_unit"] = next((v.attrs.get("coord_unit") for v in series.values()
                                           if v.attrs.get("coord_unit")), None)
        decisions.append(dict(grupo=gname, archivo="(intervalo común)", offset_s=None,
                              estado=f"{info['t0']:.3f}-{info['t1']:.3f} s, n={info['n']}"))
        merged.to_csv(out / f"data/interim/{gname}_merged.csv", index=False)

        # M5: filtrado (coordenadas por marcador; ángulos por columna)
        filt = merged.copy()
        markers = sorted({c[2:] for c in merged if c.startswith("x_") and f"y_{c[2:]}" in merged})
        for m in markers:
            if merged[[f"x_{m}", f"y_{m}"]].isna().any().any():
                log.add(gname, "filtrado_omitido", f"marcador {m} con NaN")
                continue
            tmp = []
            fx = _filter_series(merged[f"x_{m}"].to_numpy(), fs, cfg["filter"], f"x_{m}", tmp)
            fy = _filter_series(merged[f"y_{m}"].to_numpy(), fs, cfg["filter"], f"y_{m}", tmp)
            if cfg["filter"].get("cutoff_hz") is None and cfg["filter"].get("type") != "savgol":
                fc = float(np.median([t["fc_hz"] for t in tmp]))      # una fc por marcador (x e y)
                fx, fy = (butter_lp(merged[f"{a}_{m}"].to_numpy(), fs, fc, cfg["filter"]["order"]) for a in "xy")
                tmp = [dict(serie=f"marcador_{m}", fc_hz=round(fc, 2), metodo=tmp[0]["metodo"])]
            filt[f"x_{m}"], filt[f"y_{m}"] = fx, fy
            filt_rows += [dict(grupo=gname, **t) for t in tmp]
        angle_cols = g.get("angle_cols", "auto")
        angle_cols = _auto_angle_cols(merged) if angle_cols == "auto" else angle_cols
        for c in angle_cols:
            tmp = []
            filt[c] = _filter_series(np.unwrap(merged[c].to_numpy()), fs, cfg["filter"], c, tmp)
            filt_rows += [dict(grupo=gname, **t) for t in tmp]
        filt.to_csv(out / f"data/processed/{gname}_filtered.csv", index=False)

        # M6: cinemática
        kin = compute_kinematics(filt, fs, g.get("segments"), g.get("joints"), g.get("linear"),
                                 angle_cols, cfg["flip_y"])
        kin.to_csv(out / f"data/processed/{gname}_kinematics.csv", index=False)
        results[gname] = kin

    log.to_df().to_csv(out / "reports/tables/qc_log.csv", index=False)
    pd.DataFrame(time_rows).to_csv(out / "reports/tables/time_report.csv", index=False)
    pd.DataFrame(filt_rows).to_csv(out / "reports/tables/filtros.csv", index=False)
    pd.DataFrame(decisions).to_csv(out / "reports/tables/decisiones_sync.csv", index=False)
    return dict(results=results, log=log.to_df(), catalog=cat)
