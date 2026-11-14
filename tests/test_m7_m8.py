import numpy as np, pandas as pd, pytest
import subprocess
import sys
from pathlib import Path
from saque import inverse_dynamics, stats, viz
from saque import events, jacobian, levers, kinematics
from saque.pipeline import run, run_events
from saque.report import run_report
from tests import synth

FS = 120.0
t = np.arange(0, 3, 1 / FS)


def gauss(c, w=0.12): return np.exp(-0.5 * ((t - c) / w) ** 2)


# ---------------- M7 ----------------
def make_series(imp=1.5):
    return {"wrist_speed": (t, 8 * gauss(imp)),
            "knee_angle": (t, 2.6 - 0.9 * gauss(1.0, 0.15)),          # mínimo (máx. flexión) en 1.0 s
            "elbow_omega": (t, 30 * gauss(imp - 0.05, 0.07)),
            "elbow_angle": (t, 2.2 - 0.8 * gauss(1.2, 0.1) + 0.5 * gauss(imp, 0.1))}


def test_events_detected_at_expected_times():
    ev = events.detect_events(make_series(1.5))
    assert ev["impacto"]["t"] == pytest.approx(1.5, abs=1 / FS)
    assert ev["max_flex_rodilla"]["t"] == pytest.approx(1.0, abs=1 / FS)
    assert ev["pico_omega_codo"]["t"] == pytest.approx(1.45, abs=1 / FS)
    assert ev["max_flex_codo"]["t"] == pytest.approx(1.2, abs=2 / FS)
    assert ev["inicio"]["t"] < ev["impacto"]["t"] < ev["fin"]["t"]
    assert ev["inicio_flexion_rodilla"]["t"] < ev["max_flex_rodilla"]["t"]      # carga empieza antes que la muñeca
    assert ev["inicio_flexion_rodilla"]["t"] < ev["inicio"]["t"]
    assert ev["_warnings"] == []


def test_manual_impact_overrides_proxy():
    ev = events.detect_events(make_series(1.5), manual={"impacto": 1.55})
    assert ev["impacto"]["metodo"] == "manual" and ev["impacto"]["t"] == 1.55


def test_missing_optional_series_is_graceful():
    ev = events.detect_events({"wrist_speed": make_series()["wrist_speed"]})
    assert "max_flex_rodilla" not in ev
    ph = events.phase_table(ev)
    assert len(ph) >= 1 and (ph["duracion_ms"] > 0).all()


def test_phase_table_contiguous():
    ev = events.detect_events(make_series())
    ph = events.phase_table(ev).set_index("fase")
    assert ph.loc["carga", "t_ini"] < ph.loc["carga", "t_fin"]
    assert ph.loc["carga", "t_fin"] == ph.loc["aceleracion", "t_ini"]
    assert ph.loc["aceleracion", "t_fin"] == ph.loc["seguimiento", "t_ini"]


def test_no_end_threshold_warns():
    v = np.linspace(0, 1, t.size)           # nunca baja tras el pico
    ev = events.detect_events({"wrist_speed": (t, v)})
    assert any("fin" in w for w in ev["_warnings"])


def test_peak_at_record_edge_warns():
    ev = events.detect_events({"wrist_speed": (t, np.exp(-t))})
    assert any("borde" in w for w in ev["_warnings"])


def test_time_normalize_endpoints():
    y = np.sin(t)
    n = events.time_normalize(t, y, 0.5, 2.0)
    assert len(n) == 101 and n[0] == pytest.approx(np.sin(0.5)) and n[-1] == pytest.approx(np.sin(2.0), abs=1e-3)


def test_sequence_perfect_and_reversed():
    order = ["muslo", "brazo", "codo", "muneca"]
    good = {l: (t, gauss(1.0 + 0.1 * i)) for i, l in enumerate(order)}
    pk = events.peak_times(good, (0.5, 2.0))
    sm = events.sequence_metrics(pk, order)
    assert sm["tau"] == pytest.approx(1.0) and all(abs(l - 100) < 10 for l in sm["lags_ms"])
    bad = {l: (t, gauss(1.3 - 0.1 * i)) for i, l in enumerate(order)}
    assert events.sequence_metrics(events.peak_times(bad, (0.5, 2.0)), order)["tau"] == pytest.approx(-1.0)


def test_sequence_incomplete_chain_flagged():
    pk = events.peak_times({"brazo": (t, gauss(1.0)), "muneca": (t, gauss(1.2))}, (0.5, 2.0))
    sm = events.sequence_metrics(pk, ["muslo", "brazo", "codo", "muneca"])
    assert sm["nota"] == "cadena incompleta" and sm["lags_ms"][0] == pytest.approx(200, abs=10)


# ---------------- M8 ----------------
def test_jacobian_matrix_known_values():
    J = jacobian.jacobian_2link([0.0], [np.pi / 2], 0.3, 0.2)
    assert np.allclose(J[0], [[0.0, -0.2], [0.3, 0.0]])


def test_hand_velocity_pure_rotation():
    # eslabón 1 horizontal (θ1=0) girando 2 rad/s => v = L1*ω hacia +y
    J = jacobian.jacobian_2link([0.0], [0.0], 0.3, 0.2)
    v = jacobian.hand_velocity(J, [2.0], [0.0])
    assert np.allclose(v, [[0.0, 0.6]])


def _clean_pipeline_arm():
    t_ = np.arange(0.2, 2.8, 1 / FS)
    m = synth.arm_markers(t_, noise=0.0)
    d = pd.DataFrame({"time_s": t_})
    for k, n in dict(homb="hombro", codo="codo", mun="muneca").items():
        d[f"x_{n}"], d[f"y_{n}"] = m[k]
    kin = kinematics.compute_kinematics(
        d, FS, segments={"brazo": ["hombro", "codo"], "antebrazo": ["codo", "muneca"]},
        linear=["hombro", "codo", "muneca"], flip_y=True)
    # el Jacobiano trabaja en y-up: coordenadas para longitudes (la longitud no depende del signo de y)
    return kin, d


def test_jacobian_matches_direct_and_recovers_lengths():
    kin, d = _clean_pipeline_arm()
    jt, met = jacobian.jacobian_from_kinematics(kin, d)
    assert met["L_brazo"] == pytest.approx(synth.L1, abs=1e-6) and met["L_antebrazo"] == pytest.approx(synth.L2, abs=1e-6)
    # J·ω vs derivada directa difieren solo por discretización (diferencias centrales): <0.1 % del pico
    assert met["longitud_rigida"] and met["rmse_rel_pico"] < 1e-3 and met["corr_speed"] > 0.9999
    assert np.max(np.abs(jt["vx_J"] - kin["vx_muneca"])) < 2e-3 * met["pico_speed_directa"]


def test_contributions_sum_to_one_with_fixed_shoulder():
    kin, d = _clean_pipeline_arm()
    jt, _ = jacobian.jacobian_from_kinematics(kin, d)
    s = (jt["frac_hombro"] + jt["frac_codo"]).dropna()
    assert len(s) > 100 and np.allclose(s, 1.0, atol=1e-3)


def test_nonrigid_arm_flags_cv():
    kin, d = _clean_pipeline_arm()
    d2 = d.copy()
    d2["x_codo"] = d2["x_codo"] + 0.03 * np.sin(2 * np.pi * 1.0 * d2["time_s"])   # eslabón se "estira"
    _, met = jacobian.jacobian_from_kinematics(kin, d2)
    assert not met["longitud_rigida"] and "CV" in met["nota"]


# ---------------- levers ----------------
def test_moment_arm_and_moment():
    assert levers.moment_arm([0, 0], [1, 0], [0, 1])[0] == pytest.approx(1.0)       # F↑ a 1 m a la derecha
    assert levers.moment_arm([0, 0], [1, 0], [1, 0])[0] == pytest.approx(0.0)       # línea pasa por la articulación
    assert levers.joint_moment([0, 0], [0.4, 0], [0, 100])[0] == pytest.approx(40.0)
    assert levers.joint_moment([0, 0], [0.4, 0], [0, -100])[0] == pytest.approx(-40.0)
    assert levers.horizontal_offset_to_vertical_line([[0.1, 0.5]], 0.35)[0] == pytest.approx(0.25)


# ---------------- Integración ----------------
def test_pipeline_with_jacobian_and_events(tmp_path):
    raw = synth.write_raw(tmp_path / "raw")
    r = run(raw, "config/pipeline.yaml", "config/sync_offsets.yaml", tmp_path / "o", use_hypothesis=True)
    assert not r["jacobian"].empty and r["jacobian"].iloc[0]["longitud_rigida"]
    out = run_events(tmp_path / "o", "config/pipeline.yaml")
    assert "impacto" in set(out["events"]["evento"])
    assert any("RELOJ" in w for w in out["warnings"])
    assert (tmp_path / "o/reports/tables/events/eventos.csv").exists()
    assert (tmp_path / "o/data/processed/kinematics/posterior_brazo_kinematics.csv").exists()
    assert (tmp_path / "o/data/interim/synchronized/posterior_brazo_merged.csv").exists()


def test_stats_descriptives_and_normalized_variability():
    d = stats.describe_series([0, 1, 2], [1, -3, 2], variable="omega", trial="t1",
                              phase="carga", start_s=0, end_s=2)
    assert d["pico_absoluto"] == 3 and d["tiempo_pico_s"] == 1 and d["rom"] == 5
    curves = stats.normalize_trials({
        "t1": (t, gauss(1.0)),
        "t2": (t, gauss(1.1)),
    }, start_s=0.5, end_s=2.0)
    band, rmsd = stats.intertrial_variability(curves)
    assert len(band) == 101 and len(rmsd) == 2
    assert (rmsd["rmsd_vs_media"] > 0).all()


def test_bootstrap_spearman_is_reproducible_and_flags_small_n():
    result = stats.bootstrap_spearman([1, 2, 3, 4], [2, 4, 6, 8], n_boot=200)
    again = stats.bootstrap_spearman([1, 2, 3, 4], [2, 4, 6, 8], n_boot=200)
    assert result == again and result["rho"] == pytest.approx(1.0)
    assert stats.bootstrap_spearman([1, 2], [3, 4])["n"] == 2


def test_inverse_dynamics_freefall_and_static_contact():
    params = inverse_dynamics.winter_parameters(1.64, 58.0, 0.25)
    n = 5
    t_ = np.arange(n) / FS
    angles = {k: np.full(n, -np.pi / 2) for k in ("foot", "shank", "thigh")}
    omega = {k: np.zeros(n) for k in angles}
    alpha = {k: np.zeros(n) for k in angles}
    cop = np.zeros((n, 2))
    freefall = inverse_dynamics.inverse_dynamics_2d(
        t_, angles, omega, alpha, np.zeros(n), np.zeros(n), cop,
        np.tile([0.0, -params.gravity_m_s2], (n, 1)), params)
    force_cols = [c for c in freefall if c.startswith(("Fx_", "Fz_"))]
    torque_cols = [c for c in freefall if c.startswith("T_")]
    assert np.allclose(freefall[force_cols], 0) and np.allclose(freefall[torque_cols], 0)

    ground = np.full(n, params.foot.mass_kg * params.gravity_m_s2)
    supported = inverse_dynamics.inverse_dynamics_2d(
        t_, angles, omega, alpha, np.zeros(n), ground, cop, np.zeros((n, 2)), params,
        contact_threshold_n=1.0)
    assert supported["contact"].all()
    assert np.allclose(supported[["Fx_ankle_N", "Fz_ankle_N", "T_ankle_Nm"]], 0)
    assert np.allclose(supported[["T_knee_Nm", "T_hip_Nm"]], 0)


def test_inverse_dynamics_runs_only_with_configured_synced_grf(tmp_path):
    raw = tmp_path / "raw"
    (raw / "grf").mkdir(parents=True)
    out = tmp_path / "out"
    kin_dir = out / "data/processed/kinematics"
    kin_dir.mkdir(parents=True)
    time_s = np.arange(0, 0.1, 0.01)
    kin = {"time_s": time_s, "ax_tobillo": np.zeros_like(time_s),
           "ay_tobillo": np.zeros_like(time_s)}
    for segment in ("foot", "shank", "thigh"):
        kin[f"theta_seg_{segment}"] = np.full_like(time_s, -np.pi / 2)
        kin[f"omega_seg_{segment}"] = np.zeros_like(time_s)
        kin[f"alpha_seg_{segment}"] = np.zeros_like(time_s)
    pd.DataFrame(kin).to_csv(kin_dir / "posterior_pierna_kinematics.csv", index=False)
    pd.DataFrame({"time_s": time_s, "GRF_x": 0.0, "vGRF": 50.0,
                  "cop_x_from_ankle_m": 0.0, "cop_z_from_ankle_m": 0.0}).to_csv(
                      raw / "grf/grf.csv", index=False)
    config = tmp_path / "pipeline.yaml"
    config.write_text(
        "inverse_dynamics:\n"
        "  enabled: true\n"
        "  shared_clock_verified: true\n"
        "  file: grf/grf.csv\n"
        "  kinematics_group: posterior_pierna\n"
        "  contact_threshold_n: 20\n"
        "  columns: {time: time_s, force_x: GRF_x, force_z: vGRF, "
        "cop_x: cop_x_from_ankle_m, cop_z: cop_z_from_ankle_m}\n"
        "  anthropometry: {height_m: 1.64, body_mass_kg: 58, foot_length_m: 0.25}\n",
        encoding="utf-8")
    result = inverse_dynamics.run_inverse_dynamics(out, config, raw)
    assert result["status"] == "completed"
    output = pd.read_csv(out / "data/processed/inverse_dynamics/posterior_pierna_inverse_dynamics.csv")
    assert len(output) == len(time_s) and output["contact"].all()


def test_report_and_visualizations_write_outputs(tmp_path):
    kin_dir = tmp_path / "data/processed/kinematics"
    kin_dir.mkdir(parents=True)
    pd.DataFrame({"time_s": t, "theta_seg_brazo": np.sin(t),
                  "omega_seg_brazo": np.cos(t), "alpha_seg_brazo": -np.sin(t),
                  "vx_muneca": np.cos(t), "vy_muneca": np.sin(t),
                  "speed_muneca": np.ones(len(t))}).to_csv(
                      kin_dir / "posterior_brazo_kinematics.csv", index=False)
    result = viz.run_visualizations(tmp_path)
    assert any(p.name == "posterior_brazo.png" for p in result["figures"])
    report = run_report(tmp_path, warnings=["test warning"])
    assert report["report"].exists() and report["decision_log"].exists()
    assert "test warning" in report["report"].read_text()


def test_run_pipeline_entrypoint_generates_m7_and_m8_outputs(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    raw = synth.write_raw(tmp_path / "raw")
    out = tmp_path / "out"
    subprocess.run(
        [sys.executable, str(repo / "scripts/run_pipeline.py"),
         "--raw", str(raw), "--config", str(repo / "config/pipeline.yaml"),
         "--offsets", str(repo / "config/sync_offsets.yaml"), "--out", str(out),
         "--use-hypothesis"],
        cwd=repo, check=True, capture_output=True, text=True,
    )
    assert (out / "data/processed/jacobian/posterior_brazo_jacobian.csv").exists()
    assert (out / "data/processed/filtered/posterior_brazo_filtered.csv").exists()
    assert (out / "data/processed/kinematics/posterior_brazo_kinematics.csv").exists()
    assert (out / "data/interim/synchronized/posterior_brazo_merged.csv").exists()
    assert (out / "reports/tables/events/eventos.csv").exists()
    assert (out / "reports/tables/events/fases.csv").exists()
    assert (out / "reports/tables/jacobian/validacion.csv").exists()
    assert (out / "reports/tables/catalog/catalogo.csv").exists()
    assert (out / "reports/tables/statistics/descriptivos_por_fase.csv").exists()
    id_status = pd.read_csv(out / "reports/tables/inverse_dynamics/status.csv")
    assert id_status.loc[0, "status"] == "disabled"
    assert (out / "reports/report.md").exists()
    assert list((out / "reports/figures").rglob("*.png"))
