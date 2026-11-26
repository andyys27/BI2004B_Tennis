"""GRF estimada por el CoM en la dinámica inversa."""
import numpy as np
import pandas as pd
import pytest

from saque.analysis import inverse_dynamics as idm

G = 9.81
M = 60.0


def _static_leg(n=50, a_com_y=0.0):
    """Pierna vertical inmóvil: cadera (0, 0.9), rodilla (0, 0.5), tobillo (0, 0.1), punta (0.2, 0.1)."""
    t = np.arange(n) / 120
    mk = pd.DataFrame({"time_s": t})
    for name, (x, y) in {"cadera_der": (0, .9), "rodilla_der": (0, .5), "tobillo_der": (0, .1),
                         "punta_der": (.2, .1)}.items():
        mk[f"x_{name}"], mk[f"y_{name}"] = x, y
    kin = mk.copy()
    for seg, th in {"muslo": -np.pi / 2, "pierna": -np.pi / 2, "pie": 0.0}.items():
        kin[f"theta_seg_{seg}"], kin[f"omega_seg_{seg}"], kin[f"alpha_seg_{seg}"] = th, 0.0, 0.0
    kin["ax_tobillo_der"] = kin["ay_tobillo_der"] = 0.0
    kin["ax_com"], kin["ay_com"] = 0.0, a_com_y
    return kin, mk


VIEW = {"segments": {"muslo": ["cadera_der", "rodilla_der"], "pierna": ["rodilla_der", "tobillo_der"],
                     "pie": ["tobillo_der", "punta_der"]}}
ANTH = {'thigh_mass_fraction': 0.1, 'shank_mass_fraction': 0.0465, 'foot_mass_fraction': 0.0145, 'upper_arm_mass_fraction': 0.0271, 'forearm_mass_fraction': 0.0162, 'thigh_com_fraction': 0.433, 'shank_com_fraction': 0.433, 'foot_com_fraction': 0.5, 'upper_arm_com_fraction': 0.436, 'forearm_com_fraction': 0.43, 'thigh_gyration_fraction': 0.323, 'shank_gyration_fraction': 0.302, 'foot_gyration_fraction': 0.475, 'upper_arm_gyration_fraction': 0.322, 'forearm_gyration_fraction': 0.303}


def test_grf_estimada_de_pie():
    kin, mk = _static_leg()
    gx, gz, cop = idm.estimated_grf(kin, mk, "tobillo_der", "punta_der", M, share=0.5, cop_frac=0.5)
    assert np.allclose(gz, 0.5 * M * G) and np.allclose(gx, 0)
    assert np.allclose(cop, [[0.1, 0.0]] * len(kin))


def test_dinamica_inversa_con_grf_en_estatica():
    kin, mk = _static_leg()
    res = idm.run_serve_inverse_dynamics(kin, mk, VIEW, M, ANTH,
                                         grf={"mode": "com", "share": 0.5, "cop_frac": 0.0})["tren_inferior"]
    assert res["contact"].all()
    limb_mass = M * (ANTH["foot_mass_fraction"] + ANTH["shank_mass_fraction"])   # pie + pierna
    # En estática la fuerza vertical en la rodilla sostiene la mitad del peso menos pie y pierna:
    # la rodilla empuja hacia abajo sobre la pierna (signo negativo).
    assert res["Fz_knee_N"].iloc[0] == pytest.approx(-(0.5 * M * G - limb_mass * G), rel=0.02)
    # Con el CoP bajo el tobillo, el momento en el tobillo solo equilibra el peso del pie.
    foot_moment = M * ANTH["foot_mass_fraction"] * G * 0.2 * ANTH["foot_com_fraction"]
    assert abs(res["T_ankle_Nm"].iloc[0]) == pytest.approx(foot_moment, rel=0.01)


def test_dinamica_inversa_sin_grf_marca_vuelo():
    kin, mk = _static_leg()
    res = idm.run_serve_inverse_dynamics(kin, mk, VIEW, M, ANTH, grf={"mode": "none"})["tren_inferior"]
    assert not res["contact"].any()
    assert (res["external_force_mode"] == "GRF=0; fase aérea asumida").all()


def test_sin_contacto_en_vuelo():
    kin, mk = _static_leg(a_com_y=-G)              # caída libre: GRF = 0
    res = idm.run_serve_inverse_dynamics(kin, mk, VIEW, M, ANTH)["tren_inferior"]
    assert not res["contact"].any()

