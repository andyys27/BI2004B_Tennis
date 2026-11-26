"""Orquestación del pipeline: video -> pose -> marcadores -> cinemática -> saques -> fatiga -> reporte.

Etapas (cada una lee lo que dejó la anterior en disco, así que se pueden repetir por separado):
  pose      MediaPipe + estampitas sobre los videos completos (caché en data/interim/pose)
  kinematics marcadores calibrados, QC, filtrado y θ/ω/α del registro completo por vista
  serves    detección de los saques, ventanas y sincronización entre vistas
  analysis  eventos, secuencia, Jacobiano y dinámica inversa por saque; fatiga y variabilidad
  report    figuras y reporte Markdown
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis.events import detect_events, peak_times, phase_table, sequence_metrics, time_normalize
from .analysis.fatigue import fatigue_index, per_serve_values, phase_duration_values, trend_table
from .analysis.inverse_dynamics import run_serve_inverse_dynamics
from .analysis.jacobian import jacobian_from_kinematics
from .analysis.serves import (detect_impacts, estimate_lag, manual_impacts, serve_windows,
                              shift_impacts, view_offset)
from .analysis.stats import describe_series, intertrial_variability
from .config import DEFAULT_CONFIG, Paths, load_config
from .processing.filtering import butter_lp_gaps, longest_run, winter_cutoff
from .processing.kinematics import compute_kinematics
from .processing.markers import (Calibration, fuse_markers, implied_height_m, mask_segment_length,
                                 mask_speed, to_metric)
from .processing.qc import QCLog, fill_short_gaps, markers_in, mask_outliers
from .video.pose import load_pose

STAGES = ("pose", "kinematics", "serves", "analysis", "report")


@dataclass
class Context:
    cfg: dict
    paths: Paths
    log: QCLog = field(default_factory=QCLog)
    warnings: list[str] = field(default_factory=list)

    @property
    def side(self) -> str:
        return self.cfg["subject"]["hitting_side"]

    @property
    def wrist(self) -> str:
        return f"muneca_{self.side}"

    def view_dir(self, view: str) -> Path:
        d = self.paths.processed / view
        d.mkdir(parents=True, exist_ok=True)
        return d

    def table(self, df: pd.DataFrame, name: str) -> Path:
        self.paths.tables.mkdir(parents=True, exist_ok=True)
        p = self.paths.tables / f"{name}.csv"
        df.to_csv(p, index=False)
        return p


# ---------------------------------------------------------------- etapa: cinemática por vista
def _filter_markers(df: pd.DataFrame, fs: float, fcfg: dict, view: str) -> tuple[pd.DataFrame, list]:
    """Butterworth por marcador; fc por análisis de residuos (una fc común a x e y)."""
    out, rows = df.copy(), []
    for m in markers_in(df):
        x, y = df[f"x_{m}"].to_numpy(float), df[f"y_{m}"].to_numpy(float)
        if fcfg.get("cutoff_hz"):
            fc, metodo = float(fcfg["cutoff_hz"]), "fijo"
        else:
            cands = [winter_cutoff(longest_run(s), fs, tuple(fcfg["cutoff_range_hz"]),
                                   fallback=fcfg["fallback_cutoff_hz"])
                     for s in (x, y) if len(longest_run(s)) > 60]
            fc = float(np.median([c["fc"] for c in cands])) if cands else fcfg["fallback_cutoff_hz"]
            metodo = cands[0]["metodo"] if cands else "fallback_sin_datos"
        out[f"x_{m}"] = butter_lp_gaps(x, fs, fc, fcfg["order"])
        out[f"y_{m}"] = butter_lp_gaps(y, fs, fc, fcfg["order"])
        rows.append(dict(vista=view, marcador=m, fc_hz=round(fc, 2), metodo=metodo))
    return out, rows


def _mask_implausible_omega(kin: pd.DataFrame, max_omega: float, log: QCLog, view: str) -> pd.DataFrame:
    """ω (y su α) por encima del límite de plausibilidad -> NaN (error de marcador o escorzo 2D)."""
    for c in [c for c in kin if c.startswith("omega_")]:
        bad = np.abs(kin[c].to_numpy(float)) > max_omega
        if bad.any():
            kin.loc[bad, [c, c.replace("omega_", "alpha_", 1)]] = np.nan
            log.add(view, "omega_implausible", f"{c}: {int(bad.sum())} muestras a NaN",
                    detalle=f"|ω|>{max_omega} rad/s")
    return kin


def process_view(ctx: Context, view: str) -> dict:
    cfg, vcfg = ctx.cfg, ctx.cfg["views"][view]
    pose = load_pose(ctx.paths.pose / f"{view}.parquet", ctx.paths.video(view, cfg))
    fs = pose.attrs["fps"]
    px, src = fuse_markers(pose, cfg["pose"]["min_visibility"], cfg["stickers"]["max_offset_mad"],
                           sticker_markers=vcfg.get("stickers", []))
    src.insert(0, "vista", view)
    calib = Calibration.from_config(vcfg, px)
    m = to_metric(px.drop(columns=[c for c in px if c.startswith("src_")]), calib)
    m = mask_segment_length(m, vcfg.get("segments", {}), cfg["qc"]["segment_length_tol"],
                            cfg["qc"]["segment_jump_tol"], ctx.log, view)
    m = mask_speed(m, fs, cfg["qc"]["max_marker_speed_m_s"], ctx.log, view)
    m = mask_outliers(m, view, ctx.log, cfg["qc"]["jump_z"], cfg["qc"]["jump_min_rel"])
    m = fill_short_gaps(m, view, ctx.log, cfg["qc"]["max_gap_s"])
    filt, frows = _filter_markers(m, fs, cfg["filter"], view)
    kin = compute_kinematics(filt, fs, vcfg.get("segments"), vcfg.get("joints"),
                             vcfg.get("linear"))
    kin = _mask_implausible_omega(kin, cfg["qc"]["max_omega_rad_s"], ctx.log, view)
    kin.insert(1, "frame", pose["frame"].to_numpy())
    d = ctx.view_dir(view)
    px.to_parquet(d / "markers_px.parquet", index=False)
    filt.to_parquet(d / "markers_filtered.parquet", index=False)
    kin.to_parquet(d / "kinematics.parquet", index=False)
    cal = dict(vista=view, metodo=vcfg["calibration"].get("method", "pole"),
               m_por_px=calib.m_per_px, px_por_m=1 / calib.m_per_px,
               estatura_aparente_m=implied_height_m(filt),
               estatura_real_m=cfg["subject"]["height_m"], fps=fs,
               n_fotogramas=int(pose["frame"].notna().sum()),
               fotogramas_perdidos=pose.attrs["fotogramas_perdidos"],
               frac_detectado=float(pose["detected"].mean()))
    return dict(sources=src, filters=frows, calibration=cal)


def stage_kinematics(ctx: Context) -> None:
    srcs, filt_rows, cals = [], [], []
    for view in ctx.cfg["views"]:
        r = process_view(ctx, view)
        srcs.append(r["sources"]), filt_rows.extend(r["filters"]), cals.append(r["calibration"])
    ctx.table(pd.concat(srcs, ignore_index=True), "fuente_marcadores")
    ctx.table(pd.DataFrame(filt_rows), "filtros")
    cal = pd.DataFrame(cals)
    ctx.table(cal, "calibracion")
    for _, r in cal.iterrows():
        err = abs(r["estatura_aparente_m"] / r["estatura_real_m"] - 1)
        if np.isfinite(err) and err > 0.10:
            ctx.warnings.append(
                f"Calibración {r['vista']}: estatura aparente {r['estatura_aparente_m']:.2f} m vs "
                f"{r['estatura_real_m']:.2f} m ({err:.0%}); revisar vara/altura de cámara: "
                "magnitudes lineales (m, m/s, N) con error de escala; ángulos no afectados.")
    ctx.table(ctx.log.to_df(), "qc_log")


# ---------------------------------------------------------------- etapa: saques
def load_kin(ctx: Context, view: str) -> pd.DataFrame:
    return pd.read_parquet(ctx.view_dir(view) / "kinematics.parquet")


def _mid_hip_height(kin: pd.DataFrame) -> np.ndarray:
    cols = [c for c in ("y_cadera_izq", "y_cadera_der") if c in kin]
    return kin[cols].mean(axis=1).to_numpy(float)


def _reference_impacts(ctx: Context, view: str, kin: pd.DataFrame) -> pd.DataFrame:
    """Impactos de la vista de referencia: candidatos por muñeca + fin de caída de la pelota."""
    from .video.ball import BallParams, ball_impact_time
    from .video.reader import frame_times, probe

    scfg = ctx.cfg["serves"]
    manual = (scfg.get("manual_impacts_s") or {}).get(view)
    if manual:
        return manual_impacts(manual).assign(metodo="manual")
    cand = detect_impacts(kin["time_s"], kin[f"y_{ctx.wrist}"], kin[f"speed_{ctx.wrist}"],
                          scfg["expected"], scfg["min_separation_s"])
    cand["metodo"] = "proxy_rapidez_muneca"
    if scfg.get("impact_method", "ball") != "ball":
        return cand
    info = probe(ctx.paths.video(view, ctx.cfg))
    times = frame_times(info)
    bp = BallParams.from_config(ctx.cfg["views"][view].get("ball", {}))
    for i, row in cand.iterrows():
        r = ball_impact_time(info, times, row["t_altura"], bp)
        if np.isfinite(r["t_impacto"]):
            cand.loc[i, ["t_impacto", "metodo"]] = [r["t_impacto"], r["metodo"]]
            cand.loc[i, "puntos_pelota"] = r["n_puntos"]
        else:
            ctx.warnings.append(f"{view} saque {row['saque']}: no se siguió la pelota; impacto = "
                                "proxy de rapidez de muñeca.")
    return cand


def stage_serves(ctx: Context) -> pd.DataFrame:
    scfg = ctx.cfg["serves"]
    ref = scfg["reference_view"]
    kins = {v: load_kin(ctx, v) for v in ctx.cfg["views"]}
    impacts = {ref: _reference_impacts(ctx, ref, kins[ref])}
    if len(impacts[ref]) != scfg["expected"]:
        ctx.warnings.append(f"{ref}: se detectaron {len(impacts[ref])} saques (esperados "
                            f"{scfg['expected']}); revisar `serves.manual_impacts_s`.")
    sync = []
    for view, kin in kins.items():
        if view == ref:
            continue
        lag = estimate_lag(kins[ref]["time_s"], _mid_hip_height(kins[ref]),
                           kin["time_s"], _mid_hip_height(kin), fs=120.0)
        impacts[view] = shift_impacts(impacts[ref], lag["lag_s"])
        # Control: impactos por muñeca detectados de forma independiente en esta vista.
        own = detect_impacts(kin["time_s"], kin[f"y_{ctx.wrist}"], kin[f"speed_{ctx.wrist}"],
                             scfg["expected"], scfg["min_separation_s"])
        check = view_offset(impacts[view], own)
        sync.append(dict(vista=view, referencia=ref, lag_s=lag["lag_s"],
                         correlacion_cadera=lag["correlacion"],
                         control_diferencia_muneca_mediana_s=check["offset_s"],
                         control_dispersion_s=check["dispersion_s"]))
        if lag["correlacion"] < 0.5:
            ctx.warnings.append(f"Sincronización {view}->{ref}: correlación baja "
                                f"({lag['correlacion']:.2f}); revisar.")
    rows = []
    for view, imp in impacts.items():
        kin = kins[view]
        win = serve_windows(imp, scfg["pre_impact_s"], scfg["post_impact_s"],
                            float(kin["time_s"].iloc[0]), float(kin["time_s"].iloc[-1]))
        rows.append(win.merge(imp, on=["saque", "t_impacto"]).assign(vista=view))
    table = pd.concat(rows, ignore_index=True)
    ctx.table(table, "saques")
    ctx.table(pd.DataFrame(sync), "sincronizacion")
    return table


# ---------------------------------------------------------------- etapa: análisis por saque
def _series(kin: pd.DataFrame, col: str):
    return (kin["time_s"].to_numpy(), kin[col].to_numpy()) if col in kin else None


def analyze_serve(ctx: Context, view: str, kin: pd.DataFrame, filt: pd.DataFrame,
                  w: pd.Series) -> dict:
    m = (kin["time_s"] >= w["t_ini"]) & (kin["time_s"] <= w["t_fin"])
    k, f = kin.loc[m].reset_index(drop=True), filt.loc[m].reset_index(drop=True)
    s = int(w["saque"])
    ser = {"wrist_speed": _series(k, f"speed_{ctx.wrist}"),
           "knee_angle": _series(k, "theta_interior_rodilla"),
           "elbow_angle": _series(k, "theta_interior_codo"),
           "elbow_omega": _series(k, "omega_interior_codo")}
    ser = {a: b for a, b in ser.items() if b is not None}
    ecfg = ctx.cfg["events"]
    ev = detect_events(ser, manual={"impacto": float(w["t_impacto"])},
                       start_frac=ecfg["start_frac"], end_frac=ecfg["end_frac"])
    for msg in ev.pop("_warnings"):
        ctx.warnings.append(f"{view} saque {s}: {msg}")
    events = pd.DataFrame([dict(vista=view, saque=s, evento=e, **v) for e, v in ev.items()])
    phases = phase_table(ev).assign(vista=view, saque=s)
    out = dict(events=events, phases=phases, kin=k, ev=ev)

    order = list((ecfg.get("sequence") or {}).keys())
    seq_series = {lab: _series(k, col) for lab, col in (ecfg.get("sequence") or {}).items()}
    seq_series = {a: b for a, b in seq_series.items() if b is not None}
    if order and len(seq_series) == len(order):          # solo con la cadena completa
        peaks = peak_times(seq_series, (ev["inicio"]["t"], ev["impacto"]["t"] + ecfg["sequence_margin_s"]))
        sm = sequence_metrics(peaks, order)
        out["peaks"] = pd.DataFrame([dict(vista=view, saque=s, serie=a,
                                          t_rel_impacto_ms=1000 * (b["t"] - ev["impacto"]["t"]),
                                          amplitud=b["amp"]) for a, b in peaks.items()])
        out["sequence"] = pd.DataFrame([dict(vista=view, saque=s, orden=" > ".join(sm["labels"]),
                                             tau_kendall=sm["tau"], nota=sm["nota"],
                                             lags_ms=";".join(f"{x:.0f}" for x in sm["lags_ms"]))])

    vcfg = ctx.cfg["views"][view]
    if vcfg.get("jacobian"):
        jt, met = jacobian_from_kinematics(k, f, **vcfg["jacobian"])
        out["jacobian"] = jt
        out["jacobian_metrics"] = pd.DataFrame([dict(vista=view, saque=s, **met)])
    idcfg = ctx.cfg["inverse_dynamics"]
    if idcfg.get("enabled") and idcfg.get("view") == view:
        out["inverse_dynamics"] = run_serve_inverse_dynamics(
            k, f, vcfg, ctx.cfg["subject"]["body_mass_kg"], idcfg["anthropometry"])
    return out


def _normalized_curves(results: dict[int, dict], variables: dict) -> pd.DataFrame:
    """Curvas inicio->impacto normalizadas a 0-100 % para comparar saques."""
    rows = []
    for s, r in results.items():
        t0, t1 = r["ev"]["inicio"]["t"], r["ev"]["impacto"]["t"]
        k = r["kin"]
        for name, spec in variables.items():
            col = spec["column"]
            if col not in k or t1 <= t0:
                continue
            ok = np.isfinite(k[col].to_numpy(float))
            if ok.sum() < 4:
                continue
            y = time_normalize(k["time_s"][ok], k[col][ok], t0, t1)
            rows += [dict(variable=name, trial=f"saque_{s}", percent=p, value=v)
                     for p, v in zip(np.linspace(0, 100, len(y)), y)]
    return pd.DataFrame(rows)


def stage_analysis(ctx: Context) -> dict:
    serves = pd.read_csv(ctx.paths.tables / "saques.csv")
    acc: dict[str, list] = {k: [] for k in ("events", "phases", "peaks", "sequence",
                                           "jacobian_metrics", "descriptives")}
    fat_view = ctx.cfg["fatigue"]["view"]
    fat_results = {}
    for view in ctx.cfg["views"]:
        kin = load_kin(ctx, view)
        filt = pd.read_parquet(ctx.view_dir(view) / "markers_filtered.parquet")
        for _, w in serves[serves["vista"] == view].iterrows():
            r = analyze_serve(ctx, view, kin, filt, w)
            s = int(w["saque"])
            sd = ctx.view_dir(view) / f"saque_{s:02d}"
            sd.mkdir(exist_ok=True)
            r["kin"].to_csv(sd / "kinematics.csv", index=False)
            if "jacobian" in r:
                r["jacobian"].to_csv(sd / "jacobian.csv", index=False)
            for name, df in (r.get("inverse_dynamics") or {}).items():
                df.to_csv(sd / f"dinamica_inversa_{name}.csv", index=False)
            for key in acc:
                if key in r:
                    acc[key].append(r[key])
            for _, ph in r["phases"].iterrows():
                for var, spec in ctx.cfg["fatigue"]["variables"].items():
                    if spec["column"] in r["kin"]:
                        try:
                            row = describe_series(r["kin"]["time_s"], r["kin"][spec["column"]],
                                                  variable=var, trial=f"saque_{s}", phase=ph["fase"],
                                                  start_s=ph["t_ini"], end_s=ph["t_fin"])
                            acc["descriptives"].append(pd.DataFrame([dict(vista=view, **row)]))
                        except ValueError:
                            pass
            if view == fat_view:
                fat_results[s] = r
    for key, frames in acc.items():
        if frames:
            ctx.table(pd.concat(frames, ignore_index=True), key)

    # Fatiga y variabilidad entre saques (vista configurada).
    variables = ctx.cfg["fatigue"]["variables"]
    win = serves[serves["vista"] == fat_view].copy()
    win["t_fin_analisis"] = win["t_impacto"] + ctx.cfg["events"]["sequence_margin_s"]
    starts = {s: r["ev"]["inicio"]["t"] for s, r in fat_results.items()}
    win["t_ini"] = win["saque"].map(starts).fillna(win["t_ini"])
    values = per_serve_values({s: r["kin"] for s, r in fat_results.items()}, win, variables,
                              min_valid_frac=ctx.cfg["fatigue"].get("min_valid_frac", 0.8))
    phases = pd.concat([r["phases"] for r in fat_results.values()], ignore_index=True) \
        if fat_results else pd.DataFrame()
    values = pd.concat([values, phase_duration_values(phases)], ignore_index=True)
    trends = trend_table(values)
    perf = [v for v in variables if v.startswith(("rapidez", "omega"))]
    index = fatigue_index(values, perf)
    ctx.table(values, "fatiga_valores_por_saque")
    ctx.table(trends, "fatiga_tendencias")
    ctx.table(index, "fatiga_indice")
    curves = _normalized_curves(fat_results, variables)
    bands, rmsd = [], []
    for var, g in curves.groupby("variable"):
        band, att = intertrial_variability(g[["trial", "percent", "value"]])
        bands.append(band.assign(variable=var)), rmsd.append(att.assign(variable=var))
    if bands:
        ctx.table(curves, "curvas_normalizadas")
        ctx.table(pd.concat(bands), "variabilidad_banda")
        ctx.table(pd.concat(rmsd), "variabilidad_rmsd")
    return dict(values=values, trends=trends, index=index)


# ---------------------------------------------------------------- ejecución
def run(config=DEFAULT_CONFIG, stages=STAGES, force_pose: bool = False, root=None) -> Context:
    cfg = load_config(config)
    ctx = Context(cfg, Paths.from_config(cfg, root))
    if "pose" in stages:
        from .video.extract import extract_all
        extract_all(cfg, ctx.paths, force=force_pose)
    if "kinematics" in stages:
        stage_kinematics(ctx)
    if "serves" in stages:
        stage_serves(ctx)
    if "analysis" in stages:
        stage_analysis(ctx)
    # Advertencias persistentes: una corrida parcial conserva las de etapas que no se repitieron.
    warn_path = ctx.paths.tables / "advertencias.csv"
    if "kinematics" not in stages and warn_path.exists():
        previous = pd.read_csv(warn_path)["advertencia"].dropna().tolist()
        ctx.warnings = list(dict.fromkeys(previous + ctx.warnings))
    if ctx.warnings or "kinematics" in stages:
        ctx.table(pd.DataFrame({"advertencia": list(dict.fromkeys(ctx.warnings))}), "advertencias")
    if "report" in stages:
        from .reporting.report import write_report
        from .reporting.viz import make_figures
        figs = make_figures(ctx)
        write_report(ctx, figs)
    return ctx
