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

from .analysis.body import (body_weight_levers, com_kinetics, com_serve_metrics, derived_angles,
                            joint_power, peak_table, range_of_motion, stance, values_at_events)
from .analysis.events import detect_events, peak_times, phase_table, sequence_metrics, time_normalize
from .analysis.fatigue import fatigue_index, per_serve_values, phase_duration_values, trend_table
from .analysis.inverse_dynamics import run_serve_inverse_dynamics
from .analysis.jacobian import jacobian_from_kinematics
from .analysis.serves import (detect_impacts, estimate_lag, manual_impacts, serve_windows,
                              shift_impacts, view_offset)
from .analysis.stats import (describe_series, intertrial_variability, joint_peaks, qc_summary,
                             spearman_matrix)
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
def _filter_markers(df: pd.DataFrame, fs: float, fcfg: dict, view: str,
                    residuals: list | None = None) -> tuple[pd.DataFrame, list]:
    """Butterworth por marcador; fc por análisis de residuos (una fc común a x e y).

    Si se pasa `residuals`, se le agregan las curvas R(fc) de cada coordenada (para graficarlas).
    """
    out, rows = df.copy(), []
    for m in markers_in(df):
        x, y = df[f"x_{m}"].to_numpy(float), df[f"y_{m}"].to_numpy(float)
        if fcfg.get("cutoff_hz"):
            fc, metodo = float(fcfg["cutoff_hz"]), "fijo"
        else:
            cands, coords = [], []
            for coord, s in (("x", x), ("y", y)):
                if len(longest_run(s)) > 60:
                    cands.append(winter_cutoff(longest_run(s), fs, tuple(fcfg["cutoff_range_hz"]),
                                               fallback=fcfg["fallback_cutoff_hz"]))
                    coords.append(coord)
            if residuals is not None:
                for coord, c in zip(coords, cands):
                    residuals += [dict(vista=view, marcador=m, coord=coord, fc_hz=f, residuo=r,
                                       R0=c["R0"], fc_elegida=c["fc"]) for f, r in zip(c["fcs"], c["R"])]
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
    resid: list = []
    filt, frows = _filter_markers(m, fs, cfg["filter"], view, resid)
    kin = compute_kinematics(filt, fs, vcfg.get("segments"), vcfg.get("joints"),
                             vcfg.get("linear"))
    kin = _mask_implausible_omega(kin, cfg["qc"]["max_omega_rad_s"], ctx.log, view)
    kin = derived_angles(kin, filt)
    ck = com_kinetics(filt, fs, cfg["subject"]["body_mass_kg"], cfg["body"]["com_cutoff_hz"],
                      cfg["body"].get("segments"))
    kin = pd.concat([kin, ck.drop(columns="time_s")], axis=1)
    kin.insert(1, "frame", pose["frame"].to_numpy())
    d = ctx.view_dir(view)
    px.to_parquet(d / "markers_px.parquet", index=False)
    filt.to_parquet(d / "markers_filtered.parquet", index=False)
    kin.to_parquet(d / "kinematics.parquet", index=False)
    cal = dict(vista=view, metodo=vcfg["calibration"].get("method", "pole"),
               m_por_px=calib.m_per_px, px_por_m=1 / calib.m_per_px,
               origen_x_px=calib.origin_px[0], origen_y_px=calib.origin_px[1],
               estatura_aparente_m=implied_height_m(filt),
               estatura_real_m=cfg["subject"]["height_m"], fps=fs,
               n_fotogramas=int(pose["frame"].notna().sum()),
               fotogramas_perdidos=pose.attrs["fotogramas_perdidos"],
               frac_detectado=float(pose["detected"].mean()))
    return dict(sources=src, filters=frows, calibration=cal, residuals=resid)


def stage_kinematics(ctx: Context) -> None:
    srcs, filt_rows, cals, resid = [], [], [], []
    for view in ctx.cfg["views"]:
        r = process_view(ctx, view)
        srcs.append(r["sources"]), filt_rows.extend(r["filters"]), cals.append(r["calibration"])
        resid.extend(r["residuals"])
    if resid:
        ctx.table(pd.DataFrame(resid), "residuos_winter")
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
    ctx.table(qc_summary(ctx.log.to_df()), "qc_resumen")


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
            cand.loc[i, ["x_contacto_px", "y_contacto_px"]] = [r["x_contacto_px"], r["y_contacto_px"]]
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
        peaks = peak_times(seq_series, (ev["inicio"]["t"], ev["impacto"]["t"] + ecfg["sequence_margin_s"]),
                           min_valid_frac=ecfg.get("sequence_min_valid_frac", 0.5))
        sm = sequence_metrics(peaks, order)
        out["peaks"] = pd.DataFrame([dict(vista=view, saque=s, serie=a,
                                          t_rel_impacto_ms=1000 * (b["t"] - ev["impacto"]["t"]),
                                          amplitud=b["amp"], frac_valida=b["frac_valida"])
                                     for a, b in peaks.items()])
        out["sequence"] = pd.DataFrame([dict(vista=view, saque=s,
                                             orden_esperado=" > ".join(order),
                                             orden_observado=sm["observed"],
                                             tau_kendall=sm["tau"], n_invertidos=sm["n_invertidos"],
                                             lags_ms=";".join(f"{x:.0f}" for x in sm["lags_ms"]),
                                             nota=sm["nota"])])

    vcfg = ctx.cfg["views"][view]
    jp = joint_peaks(k, list(vcfg.get("joints", {})), ev["inicio"]["t"],
                     ev["impacto"]["t"] + ecfg["sequence_margin_s"], ev["impacto"]["t"])
    if not jp.empty:
        out["picos_articulares"] = jp.assign(vista=view, saque=s)
    if vcfg.get("jacobian"):
        jt, met = jacobian_from_kinematics(k, f, **vcfg["jacobian"])
        out["jacobian"] = jt
        out["jacobian_metrics"] = pd.DataFrame([dict(vista=view, saque=s, **met)])
    idcfg = ctx.cfg["inverse_dynamics"]
    mass = ctx.cfg["subject"]["body_mass_kg"]
    if idcfg.get("enabled") and idcfg.get("view") == view:
        out["inverse_dynamics"] = run_serve_inverse_dynamics(k, f, vcfg, mass, idcfg["anthropometry"],
                                                              grf=idcfg.get("grf"))
        peaks_id, power = [], {}
        for part, res in out["inverse_dynamics"].items():
            pw = joint_power(res, k)
            power[part] = pw
            # Picos solo entre el inicio del saque y el impacto (+ margen): fuera de ahí están el
            # aterrizaje (un solo pie, no 50 %) y los bordes de la ventana.
            t0p, t1p = ev["inicio"]["t"], ev["impacto"]["t"] + ecfg["sequence_margin_s"]
            if part == "tren_inferior" and "vy_com" in k and "max_flex_rodilla" in ev:
                # Tren inferior: hasta el fin del impulso (máx. velocidad vertical del CoM = despegue).
                # Después la GRF estimada es ruido del CoM (brazo) y no hay apoyo.
                mm = (k["time_s"] >= ev["max_flex_rodilla"]["t"]) & (k["time_s"] <= ev["impacto"]["t"])
                if k.loc[mm, "vy_com"].notna().any():
                    t1p = float(k.loc[k.loc[mm, "vy_com"].idxmax(), "time_s"])
                # Desde el inicio de la flexión de rodilla: antes los pies están separados y el
                # reparto 50/50 vertical de la GRF no vale (cada pie tiene componente horizontal).
                t0p = ev.get("inicio_flexion_rodilla", ev["max_flex_rodilla"])["t"]
            for df, pref, unit in ((res, "T_", "N·m"), (pw, "P_", "W")):
                dw = df[(df["time_s"] >= t0p) & (df["time_s"] <= t1p)]
                pt = peak_table(dw, pref, unit)
                if not pt.empty:
                    peaks_id.append(pt.assign(parte=part, t_rel_impacto_s=pt["t"] - w["t_impacto"]))
        out["power"] = power
        if peaks_id:
            out["picos_dinamica"] = pd.concat(peaks_id, ignore_index=True).assign(vista=view, saque=s)

    # Cuerpo completo: CoM, fuerza del suelo estimada, base de apoyo, posturas, ROM, palancas.
    bcfg = ctx.cfg["body"]
    t_load = ev.get("max_flex_rodilla", {}).get("t", np.nan)
    body = com_serve_metrics(k, float(w["t_ini"]), t_load, float(w["t_impacto"]), mass)
    if {"x_tobillo_izq", "x_tobillo_der"} <= set(f):
        body["separacion_tobillos_inicio_m"] = stance(f, ev["inicio"]["t"])
        if np.isfinite(t_load):
            body["separacion_tobillos_carga_m"] = stance(f, t_load)
    out["cuerpo"] = pd.DataFrame([dict(vista=view, saque=s, **body)])
    times = {e: ev[e]["t"] for e in bcfg["posture_events"] if e in ev}
    cols = [c for c in k if c.startswith(("theta_interior_", "theta_incl_", "theta_sep_"))]
    out["posturas"] = values_at_events(k, times, cols + ["y_com", "grf_z_bw"]).assign(vista=view, saque=s)
    # ROM inicio→impacto: tras el impacto la rotación del cuerpo vuelve ambiguos los ángulos 2D.
    out["rom"] = range_of_motion(k, ev["inicio"]["t"], ev["impacto"]["t"], cols).assign(vista=view, saque=s)
    if view == bcfg.get("lever_view"):
        lev = [body_weight_levers(f, k, times[e], bcfg["lever_joints"], mass).assign(evento=e)
               for e in bcfg["lever_events"] if e in times]
        if lev:
            out["palancas"] = pd.concat(lev, ignore_index=True).assign(vista=view, saque=s)
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


def _add_contact_height(ctx: Context, body: pd.DataFrame, serves: pd.DataFrame) -> pd.DataFrame:
    """Altura del contacto (centro de la pelota al impacto) en m sobre el suelo, vista de referencia."""
    cal_path = ctx.paths.tables / "calibracion.csv"
    if "y_contacto_px" not in serves or not cal_path.exists():
        return body
    cal = pd.read_csv(cal_path).set_index("vista")
    rows = serves.dropna(subset=["y_contacto_px"])
    rows = rows[rows["metodo"] == "fin_caida_pelota"]          # solo donde se siguió la pelota
    if rows.empty:
        return body
    h = {(r["vista"], int(r["saque"])): (cal.loc[r["vista"], "origen_y_px"] - r["y_contacto_px"])
         * cal.loc[r["vista"], "m_por_px"] for _, r in rows.iterrows() if r["vista"] in cal.index}
    body["altura_contacto_m"] = [h.get((v, int(s)), np.nan) for v, s in zip(body["vista"], body["saque"])]
    return body


def stage_analysis(ctx: Context) -> dict:
    serves = pd.read_csv(ctx.paths.tables / "saques.csv")
    acc: dict[str, list] = {k: [] for k in ("events", "phases", "peaks", "sequence",
                                           "jacobian_metrics", "descriptives", "cuerpo",
                                           "posturas", "rom", "palancas", "picos_dinamica",
                                           "picos_articulares")}
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
            for name, df in (r.get("power") or {}).items():
                df.to_csv(sd / f"potencia_{name}.csv", index=False)
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
    if acc["cuerpo"]:
        acc["cuerpo"] = [_add_contact_height(ctx, pd.concat(acc["cuerpo"], ignore_index=True), serves)]
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
    # θ/ω/α de cada articulación (más inclinación del tronco), normalizadas inicio -> impacto.
    jvars = {}
    for j in ctx.cfg["views"][fat_view].get("joints", {}):
        for kind in ("theta", "omega", "alpha"):
            jvars[f"{kind}_{j}"] = {"column": f"{kind}_interior_{j}"}
    jvars["theta_incl_tronco"] = {"column": "theta_incl_tronco"}
    jcurves = _normalized_curves(fat_results, jvars)
    if not jcurves.empty:
        th = jcurves["variable"].str.startswith("theta")
        jcurves.loc[th, "value"] = np.degrees(jcurves.loc[th, "value"])
        ctx.table(jcurves, "curvas_articulares")
        jb, jr = [], []
        for var, g in jcurves.groupby("variable"):
            band, att = intertrial_variability(g[["trial", "percent", "value"]])
            jb.append(band.assign(variable=var)), jr.append(att.assign(variable=var))
        ctx.table(pd.concat(jb), "variabilidad_banda_articular")
        ctx.table(pd.concat(jr), "variabilidad_rmsd_articular")

    # Asociaciones entre indicadores por saque (Spearman; descriptivo con n = 5).
    wide = values.pivot_table(index="saque", columns="variable", values="valor")
    body_path = ctx.paths.tables / "cuerpo.csv"
    if body_path.exists():
        b = pd.read_csv(body_path)
        b = b[b["vista"] == fat_view].set_index("saque")
        keep = [c for c in ("elevacion_max_com_m", "vy_max_com_m_s", "grf_pico_bw",
                            "altura_contacto_m") if c in b]
        wide = wide.join(b[keep], how="left")
    corr = spearman_matrix(wide)
    if not corr.empty:
        ctx.table(corr, "correlaciones")
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
