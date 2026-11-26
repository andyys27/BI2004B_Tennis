import numpy as np
import pandas as pd
import pytest

from saque.processing import filtering, kinematics, qc
from saque.processing.markers import (Calibration, fuse_markers, mask_segment_length, mask_speed,
                                      to_metric)
from tests import synth

FS = 120.0


# ---------------- calibración y fusión ----------------
def test_calibration_scale_origin_and_y_up():
    c = Calibration.from_reference((100, 200), (100, 700), 1.0)
    assert c.m_per_px == pytest.approx(1 / 500)
    x, y = c.to_m([100, 600], [700, 200])
    assert np.allclose(x, [0, 1]) and np.allclose(y, [0, 1])


def test_fuse_prefers_sticker_and_applies_offset_when_missing():
    n = 50
    pose = pd.DataFrame({"frame": np.arange(n), "time_s": np.arange(n) / FS,
                         "x_codo_der": np.full(n, 100.0), "y_codo_der": np.full(n, 200.0),
                         "vis_codo_der": np.ones(n),
                         "sx_codo_der": np.full(n, 108.0), "sy_codo_der": np.full(n, 195.0)})
    pose.loc[20:39, ["sx_codo_der", "sy_codo_der"]] = np.nan              # hueco largo de estampita
    out, summ = fuse_markers(pose, sticker_gap_frames=6)
    assert (out.loc[:19, "src_codo_der"] == "estampita").all()
    assert (out.loc[20:39, "src_codo_der"] == "pose").all()
    assert np.allclose(out.loc[20:39, "x_codo_der"], 108.0)              # landmark + desfase mediano
    assert summ.set_index("marcador").loc["codo_der", "frac_estampita"] == pytest.approx(0.6)


def test_fuse_interpolates_short_sticker_gaps():
    n = 40
    pose = pd.DataFrame({"frame": np.arange(n), "time_s": np.arange(n) / FS,
                         "x_muneca_der": np.full(n, 50.0), "y_muneca_der": np.full(n, 50.0),
                         "vis_muneca_der": np.ones(n),
                         "sx_muneca_der": np.linspace(60, 99, n), "sy_muneca_der": np.full(n, 52.0)})
    pose.loc[10:13, "sx_muneca_der"] = np.nan
    out, _ = fuse_markers(pose, sticker_gap_frames=6)
    assert (out["src_muneca_der"] == "estampita").all()
    assert np.allclose(out["x_muneca_der"], np.linspace(60, 99, n))


def test_duplicate_sticker_goes_to_closest_marker():
    n = 30
    pose = pd.DataFrame({"frame": np.arange(n), "time_s": np.arange(n) / FS})
    for m, (x, y) in {"hombro_der": (100, 100), "codo_der": (100, 160)}.items():
        pose[f"x_{m}"], pose[f"y_{m}"], pose[f"vis_{m}"] = float(x), float(y), 1.0
        pose[f"sx_{m}"], pose[f"sy_{m}"] = x + 3.0, y + 3.0
    # Brazo pegado al cuerpo: el landmark del codo se acerca al hombro y su búsqueda toma la
    # misma estampita (desfase coherente, así que no lo descarta el filtro de atípicos).
    pose.loc[10:12, ["x_codo_der", "y_codo_der"]] = [100.0, 100.0]
    pose.loc[10:12, ["sx_codo_der", "sy_codo_der"]] = [103.0, 103.0]
    pose.loc[10:12, ["x_codo_der", "y_codo_der"]] = [100.0, 104.0]       # un poco más lejos
    out, summ = fuse_markers(pose, sticker_gap_frames=0)
    assert (out.loc[10:12, "src_hombro_der"] == "estampita").all()       # gana el más cercano
    assert (out.loc[10:12, "src_codo_der"] == "pose").all()
    assert summ.set_index("marcador").loc["codo_der", "estampitas_duplicadas"] == 3
    assert summ.set_index("marcador").loc["hombro_der", "estampitas_duplicadas"] == 0


def test_segment_length_gate_flags_jumps_but_not_gradual_foreshortening():
    n = 100
    L = np.linspace(0.30, 0.18, n)                                        # escorzo gradual (-40 %)
    df = pd.DataFrame({"x_a": np.zeros(n), "y_a": np.zeros(n), "x_b": L, "y_b": np.zeros(n)})
    df.loc[50, "x_b"] = 0.05                                              # salto brusco
    out = mask_segment_length(df, {"s": ["a", "b"]}, tol=0.6, jump_tol=0.3)
    bad = out["x_b"].isna().to_numpy()
    assert bad[50] and bad.sum() <= 3


def test_speed_gate_removes_single_frame_jump():
    n = 60
    df = pd.DataFrame({"x_m": np.linspace(0, 0.5, n), "y_m": np.zeros(n)})
    df.loc[30, "y_m"] = 0.5                                               # 0.5 m en 1/120 s = 60 m/s
    out = mask_speed(df, FS, 15.0)
    assert out["x_m"].isna().sum() >= 1 and np.isnan(out.loc[30, "x_m"])
    assert out["x_m"].notna().sum() >= n - 3


def test_outlier_masked_and_short_gap_filled():
    t = np.arange(0, 2, 1 / FS)
    df = pd.DataFrame({"time_s": t, "x_m": np.sin(t), "y_m": np.cos(t)})
    df.loc[100, "x_m"] += 0.5
    log = qc.QCLog()
    out = qc.fill_short_gaps(qc.mask_outliers(df, "f", log), "f", log, max_gap_s=0.05)
    assert out["x_m"].notna().all()
    assert abs(out.loc[100, "x_m"] - np.sin(t[100])) < 0.01


# ---------------- filtrado ----------------
def test_filter_derivatives_match_analytic():
    t = np.arange(0, 4, 1 / FS)
    x = np.sin(2 * np.pi * 1.0 * t)
    xf = filtering.butter_lp(x, FS, 8.0)
    v = filtering.derivative(xf, FS)
    m = slice(30, -30)
    assert np.allclose(v[m], 2 * np.pi * np.cos(2 * np.pi * t)[m], atol=0.05)


def test_butter_with_gaps_keeps_gaps_and_filters_runs():
    x = np.sin(np.linspace(0, 6, 300))
    x[100:120] = np.nan
    y = filtering.butter_lp_gaps(x, FS, 10.0)
    assert np.isnan(y[100:120]).all() and np.isfinite(y[:100]).all() and np.isfinite(y[120:]).all()


def test_winter_cutoff_in_reasonable_band():
    t = np.arange(0, 4, 1 / FS)
    x = np.sin(2 * np.pi * 1.5 * t) + np.random.default_rng(0).normal(0, 0.01, t.size)
    fc = filtering.winter_cutoff(x, FS)["fc"]
    assert 3 <= fc <= 20


def test_filter_rejects_nan_and_bad_fc():
    with pytest.raises(ValueError):
        filtering.butter_lp([1.0, np.nan, 2.0], FS, 5)
    with pytest.raises(ValueError):
        filtering.butter_lp(np.ones(50), FS, 70)


# ---------------- cinemática ----------------
def test_segment_angle_and_derivatives_match_analytic():
    t = np.arange(0, 3, 1 / FS)
    df = synth.arm_markers(t)
    k = kinematics.compute_kinematics(df, FS, segments={"brazo": ["hombro", "codo"]})
    m = slice(10, -10)
    assert np.allclose(k["theta_seg_brazo"][m], synth.th1(t)[m], atol=1e-6)
    assert np.allclose(k["omega_seg_brazo"][m], synth.w1(t)[m], atol=0.01)
    assert np.allclose(k["alpha_seg_brazo"][m], synth.a1(t)[m], atol=0.3)


def test_interior_joint_angle_known_value():
    u, v = np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])
    assert kinematics.interior_angle(u, v)[0] == pytest.approx(np.pi / 2)


def test_unwrap_nan_does_not_propagate_gaps():
    a = np.r_[np.linspace(3.0, 3.1, 10), np.nan, np.linspace(-3.1, -3.0, 10)]
    out = kinematics.unwrap_nan(a)
    assert np.isnan(out[10]) and np.isfinite(out[11:]).all()


def test_trim_edges_masks_samples_next_to_gaps():
    ref = np.r_[np.ones(10), np.nan, np.ones(10)]
    d = kinematics.trim_edges(np.ones(21), ref, k=3)
    assert np.isnan(d[7:14]).all()
    assert np.isfinite(d[:7]).all() and np.isfinite(d[14:]).all()


def test_linear_speed_and_positions_in_output():
    t = np.arange(0, 2, 1 / FS)
    df = pd.DataFrame({"time_s": t, "x_p": np.cos(2 * np.pi * t), "y_p": np.sin(2 * np.pi * t)})
    k = kinematics.compute_kinematics(df, FS, linear=["p"])
    assert np.allclose(k["speed_p"][5:-5], 2 * np.pi, rtol=1e-3)
    assert np.allclose(k["y_p"], df["y_p"])


def test_to_metric_converts_all_markers():
    df = pd.DataFrame({"x_a": [100.0], "y_a": [700.0], "x_b": [600.0], "y_b": [200.0]})
    out = to_metric(df, Calibration.from_reference((100, 200), (100, 700), 1.0))
    assert out.loc[0, ["x_a", "y_a", "x_b", "y_b"]].tolist() == pytest.approx([0, 0, 1, 1])


def test_ground_plane_corrects_depth_of_reference():
    # Cámara nivelada a H=1.3 m, f=1000 px, horizonte en y=800. Vara a 4 m, jugadora a 2.5 m.
    f, H, yh = 1000.0, 1.3, 800.0
    proj = lambda h, Z: yh + f * (H - h) / Z                        # noqa: E731
    pole_top, pole_base = (500, proj(1.0, 4.0)), (500, proj(0.0, 4.0))
    feet_y = proj(0.0, 2.5)
    c = Calibration.ground_plane(pole_top, pole_base, 1.0, H, (600, feet_y))
    assert 1 / c.m_per_px == pytest.approx(f / 2.5)                  # px/m en el plano de la jugadora
    _, y = c.to_m([600], [proj(1.64, 2.5)])
    assert y[0] == pytest.approx(1.64)                               # altura sobre el suelo correcta
