"""Brazos de palanca (base de M12): distancia perpendicular de la línea de acción a una articulación.

τ = r × F = r_x F_y − r_y F_x   (positivo = antihorario, y hacia arriba)
"""
from __future__ import annotations
import numpy as np


def moment_arm(joint, point, force_dir) -> np.ndarray:
    """Brazo de momento firmado: componente z de (point − joint) × u, con u = dirección unitaria de F.

    joint, point: (N,2) o (2,); force_dir: (N,2) o (2,). Devuelve metros (mismas unidades que las coordenadas).
    """
    j, p, f = (np.atleast_2d(np.asarray(a, float)) for a in (joint, point, force_dir))
    r = p - j
    u = f / np.linalg.norm(f, axis=1, keepdims=True)
    return r[:, 0] * u[:, 1] - r[:, 1] * u[:, 0]


def joint_moment(joint, point, force) -> np.ndarray:
    """Momento de una fuerza `force` (N) aplicada en `point` respecto a `joint` (N·m)."""
    j, p, f = (np.atleast_2d(np.asarray(a, float)) for a in (joint, point, force))
    r = p - j
    return r[:, 0] * f[:, 1] - r[:, 1] * f[:, 0]


def horizontal_offset_to_vertical_line(joint, x_line) -> np.ndarray:
    """Brazo de palanca de una fuerza VERTICAL (p. ej. vGRF) con línea de acción en x = x_line."""
    return np.asarray(x_line, float) - np.atleast_2d(np.asarray(joint, float))[:, 0]
