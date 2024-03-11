import os
from math import cos, sin, exp

import numba
import numpy as np
from numba import float64, complex128

from bloch.rotations import arb_rot, z_rot
from constants import GAMMA_RAD

os.environ["OPENBLAS_NUM_THREADS"] = "1"


@numba.njit(numba.types.Tuple((float64[:, ::1], float64[::1]))(float64, float64, float64, float64),
            cache=True)
def free_precession(delta_time: float, t1: float,
                    t2: float, df: float = 0) -> tuple[np.ndarray, np.ndarray]:
    phi = 2 * np.pi * df * delta_time
    r_z = np.array([[cos(phi), -sin(phi), 0.0],
                    [sin(phi), cos(phi), 0.0],
                    [0.0, 0.0, 1.0]])

    e1 = exp(-delta_time * 1e3 / t1)
    e2 = exp(-delta_time * 1e3 / t2)

    a = np.array([[e2, 0.0, 0.0],
                  [0.0, e2, 0.0],
                  [0.0, 0.0, e1]], dtype=float64) @ r_z

    b = np.array([0.0, 0.0, 1.0 - e1], dtype=float64)

    return a, b


@numba.njit((float64[:, :, ::1])(float64, float64, float64[::1], complex128[::1], float64),
            parallel=True, cache=True)
def non_selective_rot3d_matrix(t1: float, t2: float, df: np.ndarray,
                               rf_pulse: np.ndarray, delta_time: float = 1e-6):
    rf_waveform = rf_pulse * GAMMA_RAD * delta_time
    mag_rf = np.abs(rf_waveform)
    phase_rf = np.angle(rf_waveform)
    sim_length = len(rf_pulse)

    magnetisation = np.empty((sim_length, len(df), 3), dtype=np.float64)
    magnetisation[0, :, :] = np.array([0.0, 0.0, 1.0])
    for iso_num in numba.prange(len(df)):
        a, b = free_precession(delta_time, t1, t2, df[iso_num])
        for step in range(1, sim_length):
            # Free precession
            magnetisation[step, iso_num, :] = a @ magnetisation[step - 1, iso_num, :] + b

            # RF Rotation
            current_rot = mag_rf[step - 1], phase_rf[step - 1]
            rf_rotation = arb_rot(*current_rot)
            magnetisation[step, iso_num, :] = rf_rotation @ magnetisation[step, iso_num, :]

    return magnetisation


@numba.njit((float64[:, :, ::1])(float64, float64, float64[:, ::1], complex128[::1],
                                 float64[:], float64[:], float64[:], float64),
            parallel=True, cache=True)
def spatial_selective_rot3d_matrix(t1: float, t2: float, position: np.ndarray,
                                   rf_pulse: np.ndarray,
                                   grad_x: np.ndarray, grad_y: np.ndarray, grad_z: np.ndarray,
                                   delta_time: float = 1e-6):
    rf_waveform = rf_pulse * GAMMA_RAD * delta_time
    mag_rf = np.abs(rf_waveform)
    phase_rf = np.angle(rf_waveform)

    gradients = np.vstack((grad_x, grad_y, grad_z))
    gradient_waveform = gradients * GAMMA_RAD * delta_time
    sim_length = len(rf_pulse)

    magnetisation = np.empty((sim_length, len(position), 3), dtype=np.float64)
    magnetisation[0, :, :] = np.array([0.0, 0.0, 1.0])
    a, b = free_precession(delta_time, t1, t2, 0)

    for iso_num in numba.prange(len(position)):
        current_position = position[iso_num, :]
        for step in range(1, sim_length):
            # Free precession
            magnetisation[step, iso_num, :] = a @ magnetisation[step - 1, iso_num, :] + b

            # Gradient Rotation
            for grad_index, axis_position in enumerate(current_position):
                gradient_rotation = z_rot(axis_position * gradient_waveform[grad_index, step - 1].item())
                magnetisation[step, iso_num, :] = gradient_rotation @ magnetisation[step, iso_num, :]

            # RF Rotation
            current_rot = mag_rf[step - 1], phase_rf[step - 1]
            rf_rotation = arb_rot(*current_rot)
            magnetisation[step, iso_num, :] = rf_rotation @ magnetisation[step, iso_num, :]

    return magnetisation


@numba.njit((float64[:, :, ::1])(float64, float64, float64[:], float64[:, ::1], complex128[::1],
                                 float64[:], float64[:], float64[:], float64),
            parallel=True, cache=True)
def spectral_spatial_selective_rot3d_matrix(t1: float, t2: float, df: np.ndarray,
                                            position: np.ndarray, rf_pulse: np.ndarray,
                                            grad_x: np.ndarray, grad_y: np.ndarray, grad_z: np.ndarray,
                                            delta_time: float = 1e-6):
    rf_waveform = rf_pulse * GAMMA_RAD * delta_time
    mag_rf = np.abs(rf_waveform)
    phase_rf = np.angle(rf_waveform)

    gradients = np.vstack((grad_x, grad_y, grad_z))
    gradient_waveform = gradients * GAMMA_RAD * delta_time
    sim_length = len(rf_pulse)

    magnetisation = np.empty((sim_length, len(df), 3), dtype=np.float64)
    magnetisation[0, :, :] = np.array([0.0, 0.0, 1.0])
    for iso_num in numba.prange(len(df)):
        a, b = free_precession(delta_time, t1, t2, df[iso_num])
        current_position = position[iso_num, :]
        for step in range(1, sim_length):
            # Free precession
            magnetisation[step, iso_num, :] = a @ magnetisation[step - 1, iso_num, :] + b

            # Gradient Rotation
            for grad_index, axis_position in enumerate(current_position):
                gradient_rotation = z_rot(axis_position * gradient_waveform[grad_index, step - 1].item())
                magnetisation[step, iso_num, :] = gradient_rotation @ magnetisation[step, iso_num, :]

            # RF Rotation
            current_rot = mag_rf[step - 1], phase_rf[step - 1]
            rf_rotation = arb_rot(*current_rot)
            magnetisation[step, iso_num, :] = rf_rotation @ magnetisation[step, iso_num, :]

    return magnetisation
