import numpy as np, pandas as pd
import pytest
from pathlib import Path
from saque import catalog, io_kinovea, qc, sync, filtering, kinematics
from saque.pipeline import run
from tests import synth

FS = 120.0


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    return synth.write_raw(tmp_path_factory.mktemp("raw"))


# ---------------- M1 ----------------
def test_catalog_detects_duplicate(raw):
    cat = catalog.build_catalog(raw)
    dup = cat[~cat["usar"]]
    assert len(dup) == 1 and dup.iloc[0]["duplicado_de"] is not None
    assert cat["usar"].sum() == len(cat) - 1


def test_catalog_time_range_ms(raw):
    cat = catalog.build_catalog(raw).set_index("archivo")
    r = cat.loc["posterior/cinematica_angular_tobillo.csv"]
    assert r["t_ini"] == pytest.approx(0.0, abs=2e-3)      # tiempo_s redondeado a 3 decimales
    assert r["t_fin"] == pytest.approx(2.34, abs=0.03)


# ---------------- M2 ----------------
def test_io_aliases_and_units(raw):
    df = io_kinovea.read_kinovea(raw / "posterior/cinematica_lineal_brazo_pos.csv")
    assert {"x_muneca", "y_muneca", "x_codo", "x_hombro"} <= set(df.columns)
    assert df.attrs["coord_unit"] == "m"
    assert df["x_hombro"].mean() == pytest.approx(synth.SHOULDER[0], abs=0.01)   # cm -> m


def test_io_time_ms_and_angle_rad(raw):
    df = io_kinovea.read_kinovea(raw / "posterior/cinematica_angular_rodilla.csv")
    assert "angulo_rod" in df.columns and "angulo_rod_rad" in df.columns
    assert np.allclose(df["angulo_rod_rad"], np.radians(df["angulo_rod"]))
    assert {"kin_v_rod", "kin_a_rod"} <= set(df.columns)
    assert df["time_s"].iloc[0] == pytest.approx(0.593)


def test_io_unlabeled_units_flagged(raw):
    df = io_kinovea.read_kinovea(raw / "lateral/cinematica_lineal_lateral.csv")
    assert df.attrs["coord_unit"] == "raw"
    df2 = io_kinovea.read_kinovea(raw / "lateral/cinematica_lineal_lateral.csv", coord_unit="cm")
    assert df2.attrs["coord_unit"] == "m"


def test_io_requires_time():
    with pytest.raises(ValueError):
        io_kinovea.standardize(pd.DataFrame({"x_a_cm": [1, 2]}))


# ---------------- M3 ----------------
def test_duplicate_time_resolved(raw):
    df = io_kinovea.read_kinovea(raw / "lateral/cinematica_lineal_lateral.csv", coord_unit="cm")
    assert (np.diff(df["time_s"]) == 0).sum() == 1
    for pol in ("first", "last", "mean"):
        log = qc.QCLog()
        out = qc.clean_time(df, "f", log, pol)
        assert (np.diff(out["time_s"]) > 0).all() and len(out) == len(df) - 1
        assert log.to_df()["tipo"].eq("tiempo_duplicado").sum() == 1


def test_outlier_masked_and_gap_filled(raw):
    df = io_kinovea.read_kinovea(raw / "lateral/cinematica_lineal_lateral.csv", coord_unit="cm")
    log = qc.QCLog()
    df = qc.clean_time(df, "f", log)
    df = qc.mask_outliers(df, "f", log)
    assert df["x_tobillo"].isna().sum() == 1                       # solo el salto sembrado
    assert df["x_rodilla"].isna().sum() == 0 and df["x_cadera"].isna().sum() == 0   # sin falsos positivos
    df = qc.fill_short_gaps(df, "f", log)
    assert df["x_tobillo"].isna().sum() == 0
    assert log.to_df()["tipo"].eq("salto_tracking").sum() == 1


def test_no_false_positives_on_clean_signal():
    t = synth.jittered_time(2.9)
    m = synth.arm_markers(t)
    d = pd.DataFrame({"time_s": t, "x_mun": m["mun"][0], "y_mun": m["mun"][1]})
    assert qc.flag_outliers(d) == {}


# ---------------- M4 ----------------
def test_estimate_offset_recovers_known_shift():
    t = np.arange(0, 3, 1 / FS)
    sig = lambda x: np.sin(2 * np.pi * 0.8 * x) * np.exp(-((x - 1.5) ** 2))
    tb = t - 0.592                                   # b tiene origen distinto
    est = sync.estimate_offset(t, sig(t), tb, sig(t), fs=FS)
    assert est == pytest.approx(0.592, abs=1 / FS)


def test_common_interval_error():
    a = pd.DataFrame({"time_s": [0, 1.0]}); b = pd.DataFrame({"time_s": [2, 3.0]})
    with pytest.raises(ValueError):
        sync.common_interval([a, b])


def test_uniform_resample_accuracy():
    t = synth.jittered_time(2.0)
    d = pd.DataFrame({"time_s": t, "y": np.sin(2 * np.pi * 1.0 * t)})
    g = sync.to_uniform_grid(d, FS)
    assert np.allclose(np.diff(g["time_s"]), 1 / FS)
    assert np.max(np.abs(g["y"] - np.sin(2 * np.pi * g["time_s"]))) < 2e-3
    assert g["time_s"].iloc[0] >= t[0] and g["time_s"].iloc[-1] <= t[-1]     # sin extrapolar


# ---------------- M5 ----------------
def test_filter_derivatives_match_analytic():
    t = np.arange(0, 3, 1 / FS)
    rng = np.random.default_rng(0)
    x = np.sin(2 * np.pi * 1.5 * t) + rng.normal(0, 0.01, t.size)
    xf = filtering.butter_lp(x, FS, 8)
    v = filtering.derivative(xf, FS)
    v_true = 2 * np.pi * 1.5 * np.cos(2 * np.pi * 1.5 * t)
    sl = slice(30, -30)
    assert np.sqrt(np.mean((v - v_true)[sl] ** 2)) / np.max(np.abs(v_true)) < 0.05


def test_derivative_exact_on_polynomial():
    t = np.arange(0, 2, 1 / FS)
    x = 3 * t ** 2
    assert np.allclose(filtering.derivative(x, FS, 1), 6 * t, atol=1e-9)
    assert np.allclose(filtering.derivative(x, FS, 2)[2:-2], 6, atol=1e-6)


def test_winter_cutoff_in_reasonable_band():
    t = np.arange(0, 4, 1 / FS)
    rng = np.random.default_rng(1)
    x = np.sin(2 * np.pi * 2 * t) + 0.3 * np.sin(2 * np.pi * 5 * t) + rng.normal(0, 0.02, t.size)
    w = filtering.winter_cutoff(x, FS)
    assert w["metodo"].startswith("winter") and 4 <= w["fc"] <= 20


def test_filter_rejects_nan_and_bad_fc():
    with pytest.raises(ValueError):
        filtering.butter_lp([1, np.nan, 3, 4, 5, 6, 7, 8], FS, 5)
    with pytest.raises(ValueError):
        filtering.butter_lp(np.ones(50), FS, FS)


def test_non_uniform_grid_rejected():
    with pytest.raises(ValueError):
        filtering.check_uniform([0, 0.008, 0.017, 0.025])


# ---------------- M6 ----------------
def _clean_arm(noise=0.0, y_down=True):
    t = np.arange(0.2, 2.8, 1 / FS)
    m = synth.arm_markers(t, noise=noise, y_down=y_down)
    d = pd.DataFrame({"time_s": t})
    for k, n in dict(homb="hombro", codo="codo", mun="muneca").items():
        d[f"x_{n}"], d[f"y_{n}"] = m[k]
    return t, d


@pytest.mark.parametrize("y_down", [True, False])
def test_segment_angle_and_derivatives(y_down):
    t, d = _clean_arm(0.0, y_down)
    k = kinematics.compute_kinematics(d, FS, segments={"brazo": ["hombro", "codo"]}, flip_y=y_down)
    assert np.allclose(k["theta_seg_brazo"], synth.th1(t), atol=1e-9)
    sl = slice(3, -3)
    assert np.allclose(k["omega_seg_brazo"][sl], synth.w1(t)[sl], atol=5e-3)
    assert np.allclose(k["alpha_seg_brazo"][sl], synth.a1(t)[sl], atol=5e-2)


def test_interior_joint_angle_known_value():
    u = np.array([[1.0, 0.0]] * 3); v = np.array([[0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]])
    assert np.allclose(kinematics.interior_angle(u, v), [np.pi / 2, np.pi, 0.0])


def test_unwrap_avoids_false_spike():
    t = np.arange(0, 2, 1 / FS)
    th = 3.0 + 1.0 * t                       # cruza π
    v = np.stack([np.cos(th), np.sin(th)], axis=1)
    w = filtering.derivative(kinematics.abs_angle(v), FS)
    assert np.allclose(w, 1.0, atol=1e-6)


def test_linear_speed_matches_circle():
    t = np.arange(0, 2, 1 / FS)
    r, om = 0.5, 4.0
    d = pd.DataFrame({"time_s": t, "x_p": r * np.cos(om * t), "y_p": -r * np.sin(om * t)})
    k = kinematics.compute_kinematics(d, FS, linear=["p"], flip_y=True)
    assert np.allclose(k["speed_p"][3:-3], r * om, atol=1e-2)


def test_compare_with_kinovea_detects_convention():
    th = np.radians(np.linspace(30, 150, 50))
    assert kinematics.compare_with_kinovea(th, np.degrees(th))["convencion"] == "directo"
    assert kinematics.compare_with_kinovea(th, 180 - np.degrees(th))["convencion"] == "suplementario"


# ---------------- Pipeline end-to-end ----------------
def test_pipeline_end_to_end(raw, tmp_path):
    cfg = tmp_path / "pipeline.yaml"
    cfg.write_text(Path("config/pipeline.yaml").read_text())
    off = tmp_path / "off.yaml"; off.write_text(Path("config/sync_offsets.yaml").read_text())
    # sin hipótesis: grupo con offset pendiente debe omitirse
    r = run(raw, cfg, off, tmp_path / "o1", use_hypothesis=False)
    assert "posterior_brazo" in r["results"] and "lateral_pierna" in r["results"]
    assert "posterior_angulos" not in r["results"]
    assert (r["log"]["tipo"] == "grupo_omitido").any()
    # con hipótesis: se procesa y queda marcado
    r2 = run(raw, cfg, off, tmp_path / "o2", use_hypothesis=True)
    assert "posterior_angulos" in r2["results"]
    dec = pd.read_csv(tmp_path / "o2/reports/tables/synchronization/decisiones_sync.csv")
    assert dec["estado"].eq("HIPOTESIS_NO_VERIFICADA").any()
    # brazo: ω del segmento recuperada frente a la verdad analítica
    k = r["results"]["posterior_brazo"]
    t = k["time_s"].to_numpy()
    sl = slice(15, -15)
    err = np.sqrt(np.mean((k["omega_seg_brazo"].to_numpy()[sl] - synth.w1(t)[sl]) ** 2))
    assert err / np.max(np.abs(synth.w1(t))) < 0.10
    # ángulo codo (interior) coherente con θ2-θ1 de la verdad
    exp = np.abs(np.arctan2(np.sin(synth.th2(t) - synth.th1(t)), np.cos(synth.th2(t) - synth.th1(t))))
    # el vector hombro-codo apunta del codo al hombro: ángulo interior = π - |Δθ|
    assert np.max(np.abs(k["theta_interior_codo"].to_numpy()[sl] - (np.pi - exp)[sl])) < 0.05
    # el duplicado fue excluido
    assert (r["log"]["tipo"] == "archivo_duplicado").sum() == 1
