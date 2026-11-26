import numpy as np
import pandas as pd
import pytest

from saque.analysis import body

FS = 120.0


def _standing(n=240, z=0.0):
    """Cuerpo simétrico de pie (x=0), todos los marcadores desplazados verticalmente z(t)."""
    pts = {"nariz": (0, 1.55), "hombro_izq": (-0.18, 1.35), "hombro_der": (0.18, 1.35),
           "codo_izq": (-0.2, 1.08), "codo_der": (0.2, 1.08), "muneca_izq": (-0.2, 0.83),
           "muneca_der": (0.2, 0.83), "indice_izq": (-0.2, 0.75), "indice_der": (0.2, 0.75),
           "cadera_izq": (-0.1, 0.9), "cadera_der": (0.1, 0.9), "rodilla_izq": (-0.1, 0.48),
           "rodilla_der": (0.1, 0.48), "tobillo_izq": (-0.1, 0.08), "tobillo_der": (0.1, 0.08),
           "talon_izq": (-0.12, 0.02), "talon_der": (0.12, 0.02), "punta_izq": (-0.12, 0.02),
           "punta_der": (0.12, 0.02)}
    z = np.broadcast_to(np.asarray(z, float), (n,))
    d = {"time_s": np.arange(n) / FS}
    for m, (x, y) in pts.items():
        d[f"x_{m}"], d[f"y_{m}"] = np.full(n, float(x)), y + z
    return pd.DataFrame(d)


def test_com_is_centered_and_plausible():
    com = body.center_of_mass(_standing())
    assert np.allclose(com["x_com"], 0, atol=1e-9)
    assert 0.85 < com["y_com"].iloc[0] < 1.05
    assert np.allclose(com["com_frac_masa"], 1.0)


def test_missing_segment_renormalizes_weights():
    m = _standing()
    m[["x_codo_izq", "y_codo_izq"]] = np.nan
    com = body.center_of_mass(m)
    # sin codo se pierden brazo y antebrazo (la mano usa muñeca-índice)
    assert com["com_frac_masa"].iloc[0] == pytest.approx(1 - 0.028 - 0.016, abs=1e-6)
    assert np.isfinite(com["y_com"]).all()


def test_grf_is_body_weight_standing_and_zero_in_free_fall():
    n = 360
    st = body.com_kinetics(_standing(n), FS, 58.0)
    assert np.nanmedian(st["grf_z_bw"]) == pytest.approx(1.0, abs=1e-3)
    t = np.arange(n) / FS
    fall = body.com_kinetics(_standing(n, z=1.0 - 0.5 * 9.81 * (t - 1.5) ** 2), FS, 58.0, cutoff_hz=20)
    assert np.nanmedian(fall["grf_z_bw"].iloc[60:-60]) == pytest.approx(0.0, abs=0.02)


def test_com_serve_metrics_detects_rise():
    n = 360
    t = np.arange(n) / FS
    rise = 0.2 * np.exp(-0.5 * ((t - 2.0) / 0.15) ** 2)
    ck = body.com_kinetics(_standing(n, z=rise), FS, 58.0)
    met = body.com_serve_metrics(ck, 0.0, 1.5, 2.0, 58.0)
    assert met["elevacion_max_com_m"] == pytest.approx(0.2, abs=0.01)
    assert met["vy_max_com_m_s"] > 0 and met["t_fin_impulso_rel_s"] < 0


def test_line_angle_continuous_through_vertical_and_reanchored():
    a = np.r_[np.linspace(0.2, 1.7, 30), np.full(60, np.nan), np.linspace(-0.1, 0.1, 30)]
    wrapped = (a + np.pi / 2) % np.pi - np.pi / 2
    out = body._continuous_line_angle(wrapped, max_gap=36)
    assert np.all(np.diff(out[:30]) > 0)                 # cruza la vertical sin saltar
    assert abs(np.nanmedian(out[90:])) < 0.2             # bloque tras hueco largo reanclado


def test_body_weight_lever_sign_and_moment():
    m = _standing()
    ck = pd.DataFrame({"time_s": m["time_s"], "x_com": 0.05})
    lev = body.body_weight_levers(m, ck, 0.5, ["rodilla_der"], 58.0).iloc[0]
    assert lev["brazo_palanca_m"] == pytest.approx(0.05 - 0.1)
    assert lev["momento_peso_Nm"] == pytest.approx(58.0 * 9.81 * -0.05)


def test_joint_power_is_moment_times_relative_omega():
    t = np.arange(5) / FS
    idd = pd.DataFrame({"time_s": t, "T_elbow_Nm": 2.0})
    kin = pd.DataFrame({"time_s": t, "omega_seg_antebrazo": 5.0, "omega_seg_brazo": 1.0})
    assert np.allclose(body.joint_power(idd, kin)["P_elbow_W"], 8.0)


def test_range_of_motion_and_values_at_events():
    t = np.arange(0, 1, 1 / FS)
    kin = pd.DataFrame({"time_s": t, "theta_x": np.radians(90 + 30 * np.sin(2 * np.pi * t))})
    rom = body.range_of_motion(kin, 0, 1, ["theta_x"]).iloc[0]
    assert rom["rom_deg"] == pytest.approx(60, abs=0.5)
    ev = body.values_at_events(kin, {"a": 0.25}, ["theta_x"]).iloc[0]
    assert ev["valor"] == pytest.approx(120, abs=0.5)
