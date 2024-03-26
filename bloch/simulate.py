from __future__ import annotations

import os
import time
from math import cos, sin, exp
from typing import Optional, TYPE_CHECKING

import numba
import numpy as np
from numba import float64, complex128

from bloch.rotations import arb_rot, z_rot
from constants import GAMMA_RAD

if TYPE_CHECKING:
    from sequence.rf_pulse import RFPulse
    from sequence.gradient import Gradient

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


def get_simulation_length(rf_pulse: Optional[np.ndarray | RFPulse], grad_x: Optional[np.ndarray | Gradient],
                          grad_y: Optional[np.ndarray | Gradient], grad_z: Optional[np.ndarray | Gradient],
                          delta_time: float) -> int:
    max_time = 0
    for waveform in [rf_pulse, grad_x, grad_y, grad_z]:
        if waveform is not None:
            if isinstance(waveform, np.ndarray):
                current_time = (len(waveform) - 1) * delta_time
            else:
                current_time = waveform.get_times(delta_time).max()

            max_time = max(max_time, current_time)

    return round(max_time / delta_time) + 1


def prepare_waveforms(simulation_length: int, rf_pulse: Optional[np.ndarray | RFPulse],
                      grad_x: Optional[np.ndarray | Gradient], grad_y: Optional[np.ndarray | Gradient],
                      grad_z: Optional[np.ndarray | Gradient],
                      delta_time: float):
    prepared_waveforms = []
    for waveform in [rf_pulse, grad_x, grad_y, grad_z]:
        if waveform is not None:
            prepared_waveform = waveform if isinstance(waveform, np.ndarray) else waveform.get_waveform(delta_time)
            prepared_waveform *= GAMMA_RAD * delta_time
        else:
            prepared_waveform = np.zeros(simulation_length)

        prepared_waveforms.append(prepared_waveform)

    rf_pulse, grad_x, grad_y, grad_z = prepared_waveforms

    return rf_pulse, grad_x, grad_y, grad_z


def run(t1: float, t2: float, simulation_style: str, df: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        rf_pulse: Optional[np.ndarray | RFPulse] = None, grad_x: Optional[np.ndarray | Gradient] = None,
        grad_y: Optional[np.ndarray | Gradient] = None, grad_z: Optional[np.ndarray | Gradient] = None,
        delta_time: float = 1e-5, init_magnetisation: np.ndarray = np.array([0.0, 0.0, 1.0]),
        verbose: bool = True):
    simulation_length = get_simulation_length(rf_pulse, grad_x, grad_y, grad_z, delta_time)
    rf_pulse, grad_x, grad_y, grad_z = prepare_waveforms(simulation_length, rf_pulse,
                                                         grad_x, grad_y, grad_z, delta_time)

    init_time = time.perf_counter()
    if simulation_style == 'non_selective':
        magnetisation = np.empty((simulation_length, len(df), 3), dtype=np.float64)
        magnetisation[0, :, :] = init_magnetisation
        magnetisation = non_selective_rot3d_matrix(magnetisation, t1, t2, df, rf_pulse, delta_time)
    elif simulation_style == 'spatial_selective':
        magnetisation = np.empty((simulation_length, position.shape[0], 3), dtype=np.float64)
        magnetisation[0, :, :] = init_magnetisation
        magnetisation = spatial_selective_rot3d_matrix(magnetisation, t1, t2, position, rf_pulse,
                                                       grad_x, grad_y, grad_z, delta_time)
    elif simulation_style == 'spectral_spatial':
        magnetisation = np.empty((simulation_length, len(df), 3), dtype=np.float64)
        magnetisation[0, :, :] = init_magnetisation

        original_shape = (df.shape, position.shape[0])
        df = np.tile(df, original_shape[1])
        position = np.repeat(position, original_shape[0], axis=0)

        magnetisation = spectral_spatial_selective_rot3d_matrix(magnetisation, t1, t2, df, position, rf_pulse,
                                                                grad_x, grad_y, grad_z,
                                                                delta_time)
    else:
        raise ValueError(f'Unknown simulation style: {simulation_style}!')

    end_time = time.perf_counter()

    if verbose:
        if df is not None:
            print(f"{round(end_time - init_time, 2)}s taken to simulate "
                  f"{simulation_length} time steps for {len(df)} isochromats! "
                  f"{round(simulation_length * len(df) / (end_time - init_time))} iterations per second!")
        elif position is not None:
            print(f"{round(end_time - init_time, 2)}s taken to simulate "
                  f"{simulation_length} time steps for {position.shape[0]} positions! "
                  f"{round(simulation_length * position.shape[0] / (end_time - init_time))} iterations per second!")

    return magnetisation


@numba.njit((float64[:, :, ::1])(float64[:, :, ::1], float64, float64, float64[::1], complex128[::1], float64),
            parallel=True, cache=True)
def non_selective_rot3d_matrix(magnetisation: np.ndarray, t1: float, t2: float, df: np.ndarray,
                               rf_pulse: np.ndarray, delta_time: float = 1e-6):
    mag_rf = np.abs(rf_pulse)
    phase_rf = np.angle(rf_pulse)

    for iso_num in numba.prange(len(df)):
        a, b = free_precession(delta_time, t1, t2, df[iso_num])
        for step in range(1, magnetisation.shape[0]):
            # Free precession
            magnetisation[step, iso_num, :] = a @ magnetisation[step - 1, iso_num, :] + b

            # RF Rotation
            current_rot = mag_rf[step - 1], phase_rf[step - 1]
            rf_rotation = arb_rot(*current_rot)
            magnetisation[step, iso_num, :] = rf_rotation @ magnetisation[step, iso_num, :]

    return magnetisation


@numba.njit((float64[:, :, ::1])(float64[:, :, ::1], float64, float64, float64[:, ::1], complex128[::1],
                                 float64[:], float64[:], float64[:], float64),
            parallel=True, cache=True)
def spatial_selective_rot3d_matrix(magnetisation: np.ndarray, t1: float, t2: float, position: np.ndarray,
                                   rf_pulse: np.ndarray,
                                   grad_x: np.ndarray, grad_y: np.ndarray, grad_z: np.ndarray,
                                   delta_time: float = 1e-6):
    mag_rf = np.abs(rf_pulse)
    phase_rf = np.angle(rf_pulse)

    gradients = np.vstack((grad_x, grad_y, grad_z))

    a, b = free_precession(delta_time, t1, t2, 0)
    for iso_num in numba.prange(len(position)):
        current_position = position[iso_num, :]
        for step in range(1, magnetisation.shape[0]):
            # Free precession
            magnetisation[step, iso_num, :] = a @ magnetisation[step - 1, iso_num, :] + b

            # Gradient Rotation
            for grad_index, axis_position in enumerate(current_position):
                gradient_rotation = z_rot(axis_position * gradients[grad_index, step - 1].item())
                magnetisation[step, iso_num, :] = gradient_rotation @ magnetisation[step, iso_num, :]

            # RF Rotation
            current_rot = mag_rf[step - 1], phase_rf[step - 1]
            rf_rotation = arb_rot(*current_rot)
            magnetisation[step, iso_num, :] = rf_rotation @ magnetisation[step, iso_num, :]

    return magnetisation


@numba.njit((float64[:, :, ::1])(float64[:, :, ::1], float64, float64, float64[:], float64[:, ::1], complex128[::1],
                                 float64[:], float64[:], float64[:], float64),
            parallel=True, cache=True)
def spectral_spatial_selective_rot3d_matrix(magnetisation: np.ndarray, t1: float, t2: float, df: np.ndarray,
                                            position: np.ndarray, rf_pulse: np.ndarray,
                                            grad_x: np.ndarray, grad_y: np.ndarray, grad_z: np.ndarray,
                                            delta_time: float = 1e-6):
    mag_rf = np.abs(rf_pulse)
    phase_rf = np.angle(rf_pulse)
    gradients = np.vstack((grad_x, grad_y, grad_z))

    magnetisation = np.empty((magnetisation.shape[0], len(df), 3), dtype=np.float64)
    magnetisation[0, :, :] = np.array([0.0, 0.0, 1.0])
    for iso_num in numba.prange(len(df)):
        a, b = free_precession(delta_time, t1, t2, df[iso_num])
        current_position = position[iso_num, :]
        for step in range(1, magnetisation.shape[0]):
            # Free precession
            magnetisation[step, iso_num, :] = a @ magnetisation[step - 1, iso_num, :] + b

            # Gradient Rotation
            for grad_index, axis_position in enumerate(current_position):
                gradient_rotation = z_rot(axis_position * gradients[grad_index, step - 1].item())
                magnetisation[step, iso_num, :] = gradient_rotation @ magnetisation[step, iso_num, :]

            # RF Rotation
            current_rot = mag_rf[step - 1], phase_rf[step - 1]
            rf_rotation = arb_rot(*current_rot)
            magnetisation[step, iso_num, :] = rf_rotation @ magnetisation[step, iso_num, :]

    return magnetisation
