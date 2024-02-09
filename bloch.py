from functools import lru_cache

import numba
import numpy as np
from numba import jit, float64
from math import cos, sin, exp


@jit(float64[:, ::1](float64), nopython=True)
def x_rot(phi: float) -> np.ndarray:
    """Rotation matrix for an x-axis rotation"""
    rotation_matrix = [[1.0, 0.0, 0.0],
                       [0.0, cos(phi), -sin(phi)],
                       [0.0, sin(phi), cos(phi)]]

    return np.array(rotation_matrix, dtype=np.float64)


@jit(float64[:, ::1](float64), nopython=True)
def y_rot(phi: float) -> np.ndarray:
    """Rotation matrix for a y-axis rotation"""
    rotation_matrix = [[cos(phi), 0.0, sin(phi)],
                       [0.0, 1.0, 0.0],
                       [-sin(phi), 0.0, cos(phi)]]

    return np.array(rotation_matrix, dtype=np.float64)


@jit(float64[:, ::1](float64), nopython=True)
def z_rot(phi: float) -> np.ndarray:
    """Rotation matrix for a z-axis rotation"""
    rotation_matrix = [[cos(phi), -sin(phi), 0.0],
                       [sin(phi), cos(phi), 0.0],
                       [0.0, 0.0, 1.0]]

    return np.array(rotation_matrix, dtype=np.float64)


@lru_cache
@jit((float64[:, ::1](float64, float64)), nopython=True)
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


@lru_cache
@jit(numba.types.Tuple((float64[:, ::1], float64[:]))(float64, float64, float64, float64), nopython=True)
def free_precession(delta_time: float, t1: float,
                    t2: float, df: float = 0) -> tuple[np.ndarray, np.ndarray]:
    phi = 2 * np.pi * df * delta_time
    r_z = np.array([[cos(phi), -sin(phi), 0.0],
                    [sin(phi), cos(phi), 0.0],
                    [0.0, 0.0, 1.0]])

    e1 = exp(-delta_time / t1)
    e2 = exp(-delta_time / t2)

    a = np.array([[e2, 0.0, 0.0],
                  [0.0, e2, 0.0],
                  [0.0, 0.0, e1]], dtype=float64) @ r_z

    b = np.array([0.0, 0.0, 1 - e1], dtype=float64)

    return a, b
