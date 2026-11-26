import numpy as np
import pandas as pd
import pytest

from saque.analysis import events, fatigue, inverse_dynamics, jacobian, levers, serves, stats
from saque.processing import kinematics
from tests import synth

FS = 120.0
t = np.arange(0, 3, 1 / FS)


def gauss(c, w=0.12):
    return np.exp(-0.5 * ((t - c) / w) ** 2)


# ---------------- eventos ----------------
def make_series(imp=1.5):
    return {"wrist_speed": (t, 8 * gauss(imp)),
            "knee_angle": (t, 2.6 - 0.9 * gauss(1.0, 0.15)),
            "elbow_omega": (t, 30 * gauss(imp - 0.05, 0.07)),
            "elbow_angle": (t, 2.2 - 0.8 * gauss(1.2, 0.1) + 0.5 * gauss(imp, 0.1))}


def test_events_detected_at_expected_times():
    ev = events.detect_events(make_series(1.5))
    assert ev["impacto"]["t"] == pytest.approx(1.5, abs=1 / FS)
    assert ev["max_flex_rodilla"]["t"] == pytest.approx(1.0, abs=1 / FS)
    assert ev["pico_omega_codo"]["t"] == pytest.approx(1.45, abs=1 / FS)
    assert ev["inicio"]["t"] < ev["impacto"]["t"] < ev["fin"]["t"]
    assert ev["_warnings"] == []


def test_manual_impact_overrides_proxy():
    ev = events.detect_events(make_series(1.5), manual={"impacto": 1.55})
    assert ev["impacto"]["metodo"] == "manual" and ev["impacto"]["t"] == 1.55


def test_phase_table_contiguous():
    ph = events.phase_table(events.detect_events(make_series())).set_index("fase")
    assert ph.loc["carga", "t_fin"] == ph.loc["aceleracion", "t_ini"]
    assert ph.loc["aceleracion", "t_fin"] == ph.loc["seguimiento", "t_ini"]


def test_sequence_perfect_and_reversed():
    peaks = {"a": {"t": 1.0, "amp": 1}, "b": {"t": 1.1, "amp": 1}, "c": {"t": 1.2, "amp": 1}}
    assert events.sequence_metrics(peaks, ["a", "b", "c"])["tau"] == pytest.approx(1.0)
    assert events.sequence_metrics(peaks, ["c", "b", "a"])["tau"] == pytest.approx(-1.0)


# ---------------- Jacobiano y palancas ----------------
def test_jacobian_matches_direct_and_recovers_lengths():
    tt = np.arange(0, 3, 1 / FS)
    df = synth.arm_markers(tt)
    kin = kinematics.compute_kinematics(df, FS, segments={"brazo": ["hombro", "codo"],
                                                          "antebrazo": ["codo", "muneca"]},
                                        linear=["hombro", "codo", "muneca"])
    out, met = jacobian.jacobian_from_kinematics(kin, df, shoulder="hombro", elbow="codo",
                                                 wrist="muneca")
    assert met["L_brazo"] == pytest.approx(synth.L1, rel=1e-3)
    assert met["rmse_rel_pico"] < 0.01 and met["longitud_rigida"]


def test_moment_arm_and_moment():
    assert levers.moment_arm([0, 0], [1, 0], [0, 1])[0] == pytest.approx(1.0)
    assert levers.joint_moment([0, 0], [0.5, 0], [0, 10])[0] == pytest.approx(5.0)


# ---------------- estadística ----------------
def test_bootstrap_spearman_reproducible_and_flags_small_n():
    a = stats.bootstrap_spearman([1, 2, 3, 4, 5], [2, 4, 5, 4, 6], n_boot=200, seed=1)
    b = stats.bootstrap_spearman([1, 2, 3, 4, 5], [2, 4, 5, 4, 6], n_boot=200, seed=1)
    assert a == b and a["n"] == 5
    assert np.isnan(stats.bootstrap_spearman([1, 2], [1, 2])["rho"])


def test_intertrial_variability_zero_for_identical_curves():
    curves = pd.DataFrame({"trial": ["a"] * 3 + ["b"] * 3, "percent": [0, 50, 100] * 2,
                           "value": [1.0, 2.0, 3.0] * 2})
    band, rmsd = stats.intertrial_variability(curves)
    assert np.allclose(band["desviacion_estandar"], 0) and np.allclose(rmsd["rmsd_vs_media"], 0)


# ---------------- dinámica inversa ----------------
ANTHRO = {f"{s}_{k}": v for s, vals in {
    "thigh": (0.10, 0.433, 0.323), "shank": (0.0465, 0.433, 0.302), "foot": (0.0145, 0.5, 0.475),
    "upper_arm": (0.0271, 0.436, 0.322), "forearm": (0.0162, 0.430, 0.303)}.items()
    for k, v in zip(("mass_fraction", "com_fraction", "gyration_fraction"), vals)}


def _seg(name, L):
    return inverse_dynamics._segment_from_anthropometry(L, 58.0, name, ANTHRO)


def test_inverse_dynamics_freefall_gives_zero_loads():
    limb = inverse_dynamics.LowerLimb(_seg("foot", 0.25), _seg("shank", 0.40), _seg("thigh", 0.40))
    n = 5
    tt = np.arange(n) / FS
    ang = {k: np.full(n, -np.pi / 2) for k in ("foot", "shank", "thigh")}
    zero = {k: np.zeros(n) for k in ang}
    res = inverse_dynamics.inverse_dynamics_2d(tt, ang, zero, zero, np.zeros(n), np.zeros(n),
                                               np.zeros((n, 2)), np.tile([0, -9.81], (n, 1)), limb)
    cols = [c for c in res if c.startswith(("Fx_", "Fz_", "T_"))]
    assert np.allclose(res[cols], 0)


def test_upper_limb_static_horizontal_arm():
    arm = inverse_dynamics.UpperLimb(_seg("upper_arm", 0.30), _seg("forearm", 0.25))
    n = 5
    tt = np.arange(n) / FS
    ang = {k: np.zeros(n) for k in ("upper_arm", "forearm")}
    zero = {k: np.zeros(n) for k in ang}
    res = inverse_dynamics.inverse_dynamics_upper_limb_2d(tt, ang, zero, zero, np.zeros((n, 2)),
                                                          np.zeros((n, 2)), arm)
    f = arm.forearm
    assert np.allclose(res["Fz_elbow_N"], f.mass_kg * 9.81)
    assert np.allclose(res["T_elbow_Nm"], f.mass_kg * 9.81 * f.com_from_proximal_m)


# ---------------- saques y sincronización ----------------
def test_detect_impacts_finds_expected_serves():
    tt = np.arange(0, 30, 1 / FS)
    imps = [4.0, 12.0, 20.0, 27.0]
    h = sum(np.exp(-0.5 * ((tt - c) / 0.15) ** 2) for c in imps) + 0.01 * np.sin(5 * tt)
    v = np.abs(np.gradient(h, 1 / FS))
    out = serves.detect_impacts(tt, h, v, n_expected=4, min_separation_s=3.0)
    assert len(out) == 4
    assert np.allclose(out["t_altura"], imps, atol=0.02)


def test_estimate_lag_recovers_known_shift():
    tt = np.arange(0, 20, 1 / FS)
    sig = lambda x: sum(np.exp(-0.5 * ((x - c) / 0.3) ** 2) for c in (3, 9, 15))  # noqa: E731
    lag = serves.estimate_lag(tt, sig(tt), tt, sig(tt - 0.65), fs=FS)
    assert lag["lag_s"] == pytest.approx(0.65, abs=1 / FS)
    assert lag["correlacion"] > 0.9


def test_serve_windows_clipped_to_record():
    imp = pd.DataFrame({"saque": [1, 2], "t_impacto": [0.5, 9.8]})
    w = serves.serve_windows(imp, 1.6, 0.8, 0.0, 10.0)
    assert w["t_ini"].iloc[0] == 0.0 and w["t_fin"].iloc[1] == 10.0


# ---------------- fatiga ----------------
def _values(y):
    return pd.DataFrame({"saque": range(1, len(y) + 1), "variable": "v", "valor": y,
                         "unidad": "u", "confianza": "alta"})


def test_trend_detects_monotonic_decline():
    tr = fatigue.trend_table(_values([10.0, 9.5, 9.0, 8.4, 8.0]), n_boot=200).iloc[0]
    assert tr["pendiente_por_saque"] == pytest.approx(-0.51, abs=0.02)
    assert tr["rho_spearman"] == pytest.approx(-1.0)
    assert tr["cambio_primero_ultimo_pct"] == pytest.approx(-20.0)


def test_fatigue_index_first_vs_last_third():
    idx = fatigue.fatigue_index(_values([10.0, 10.0, 9.0, 8.0, 8.0, 8.0]), ["v"]).iloc[0]
    assert idx["cambio_pct"] == pytest.approx(-20.0)


def test_per_serve_values_and_quality():
    tt = np.arange(0, 2, 1 / FS)
    kin = pd.DataFrame({"time_s": tt, "c": np.sin(tt)})
    kin.loc[(tt > 1.85) & (tt <= 1.9), "c"] = np.nan
    win = pd.DataFrame({"saque": [1], "t_ini": [0.0], "t_impacto": [1.9], "t_fin_analisis": [2.0]})
    spec = {"pk": {"column": "c", "stat": "max"}}
    v = fatigue.per_serve_values({1: kin}, win, spec, min_valid_frac=0.5).iloc[0]
    assert v["valor"] == pytest.approx(1.0, abs=1e-3)
    assert 0.6 < v["frac_valida_pre_impacto"] < 0.9
    strict = fatigue.per_serve_values({1: kin}, win, spec, min_valid_frac=0.95).iloc[0]
    assert np.isnan(strict["valor"]) and strict["valor_sin_control"] == pytest.approx(1.0, abs=1e-3)


def test_phase_durations_exclude_follow_through():
    ph = pd.DataFrame({"saque": [1, 1, 1], "fase": ["carga", "aceleracion", "seguimiento"],
                       "duracion_ms": [400.0, 300.0, 500.0]})
    out = fatigue.phase_duration_values(ph)
    assert set(out["variable"]) == {"duracion_carga", "duracion_aceleracion"}
