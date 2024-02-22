import os
from math import cos, sin

import numba
import numpy as np
from numba import float64, complex128

from constants import PAULI_X, PAULI_Y, PAULI_Z

os.environ["OPENBLAS_NUM_THREADS"] = "2"


@numba.njit(float64[:, ::1](float64), cache=True)
def x_rot(phi: float) -> np.ndarray:
    """Rotation matrix for an x-axis rotation"""
    rotation_matrix = [[1.0, 0.0, 0.0],
                       [0.0, cos(phi), -sin(phi)],
                       [0.0, sin(phi), cos(phi)]]

    return np.array(rotation_matrix, dtype=np.float64)


@numba.njit(float64[:, ::1](float64), cache=True)
def y_rot(phi: float) -> np.ndarray:
    """Rotation matrix for a y-axis rotation"""
    rotation_matrix = [[cos(phi), 0.0, sin(phi)],
                       [0.0, 1.0, 0.0],
                       [-sin(phi), 0.0, cos(phi)]]

    return np.array(rotation_matrix, dtype=np.float64)


@numba.njit(float64[:, ::1](float64), cache=True)
def z_rot(phi: float) -> np.ndarray:
    """Rotation matrix for a z-axis rotation"""
    rotation_matrix = [[cos(phi), -sin(phi), 0.0],
                       [sin(phi), cos(phi), 0.0],
                       [0.0, 0.0, 1.0]]

    return np.array(rotation_matrix, dtype=np.float64)


@numba.njit((float64[:, ::1](float64, float64)),
            cache=True)
def arb_rot(phi: float, theta: float) -> np.ndarray:
    """Rotation matrix for arbitrary axis rotation"""
    r_x = np.array([[1.0, 0.0, 0.0],
                    [0.0, cos(phi), -sin(phi)],
                    [0.0, sin(phi), cos(phi)]])

    r_z = np.array([[cos(-theta), -sin(-theta), 0.0],
                    [sin(-theta), cos(-theta), 0.0],
                    [0.0, 0.0, 1.0]])

    r_arb = np.linalg.solve(r_z, np.eye(3)) @ r_x @ r_z

    return r_arb


@numba.njit(complex128[:, ::1](complex128[:, ::1], float64, float64[::1]), cache=True)
def quaternion_rotation(quaternion: np.ndarray, theta: float, axis_norm: np.ndarray = np.array([1.0, 0.0, 0])) -> np.ndarray:
    u_dot_sigma = axis_norm[0] * PAULI_X + axis_norm[1] * PAULI_Y + axis_norm[2] * PAULI_Z
    rotation = np.identity(2) * np.cos(theta / 2) + 1j * u_dot_sigma * np.sin(theta / 2)

    return rotation @ quaternion @ rotation.conj()
