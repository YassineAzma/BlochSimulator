import os

import numba
import numpy as np
from numba import float64, complex128

from constants import GAMMA_RAD

os.environ["OPENBLAS_NUM_THREADS"] = "2"


@numba.njit(complex128[:, ::1](complex128, float64, float64), cache=True)
def forward_slr_propagation(b1: float, off_resonance: float, delta_time: float) -> np.ndarray:
    absolute_b1 = np.abs(b1)
    c = np.cos(absolute_b1 / 2)

    if absolute_b1 == 0:
        s = 1j * (b1.real + 1j * b1.imag) * 0.5
    else:
        s = 1j * (b1.real + 1j * b1.imag) * np.sin(absolute_b1 / 2) / absolute_b1

    z = np.exp(1j * 2 * np.pi * off_resonance * delta_time)

    a = np.array([[c + 0j, np.conj(-s)], [s, c + 0j]], dtype=np.complex128)
    b = np.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1 / z]], dtype=np.complex128)

    return a @ b


@numba.njit(complex128[:, :, ::1](complex128[::1], float64[::1], float64), cache=True)
def forward_slr(rf_pulse: np.ndarray, df: np.ndarray, delta_time: float):
    rf = rf_pulse * GAMMA_RAD * delta_time
    sim_length = len(rf)
    coefficients = np.empty((sim_length, len(df), 2), dtype=np.complex128)
    coefficients[0, :, 0] = 1

    for iso_num in numba.prange(len(df)):
        current_df = df[iso_num]
        for index in range(1, len(rf)):
            prop_matrix = forward_slr_propagation(rf[index - 1].item(), current_df, delta_time)
            coefficients[index, iso_num, :] = prop_matrix @ coefficients[index - 1, iso_num, :]

    return coefficients


@numba.njit(numba.types.Tuple((complex128[:], complex128[:], complex128))(complex128[:], complex128[:]), cache=True)
def inverse_slr_propagation(alpha: np.ndarray, beta: np.ndarray) -> tuple[float, float, float]:
    phi = 2 * np.arctan2(np.abs(beta[0]), np.abs(alpha[0]))
    theta = np.angle(-1j * beta[0] * np.conj(alpha[0]))

    b1 = phi * np.exp(1j * theta)
    c = np.cos(abs(b1) / 2)
    if abs(b1) == 0:
        s = 0
    else:
        s = 1j * b1 / abs(b1) * np.sin(abs(b1) / 2)

    alpha_new = c * alpha + np.conj(s) * beta
    beta_new = -s * alpha + c * beta

    return alpha_new[1:], beta_new[1:], b1


# @numba.njit(complex128[::1](complex128[:], complex128[:]), cache=True)
def inverse_slr(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    sim_length = len(alpha)
    b1 = np.empty(sim_length, dtype=np.complex128)

    for index in range(sim_length - 1, -1, -1):
        c = np.sqrt(1 / (1 + np.abs(beta[index] / alpha[index]) ** 2))
        s = np.conj(c * beta[index] / alpha[index])

        theta = np.arctan2(np.abs(s), c)
        psi = np.angle(s)

        b1[index] = 2 * theta * np.exp(1j * psi)

        alpha_new = c * alpha + s * beta
        beta_new = -np.conj(s) * alpha + c * beta
        alpha = alpha_new[1: index + 1]
        beta = beta_new[:index]
        # alpha, beta, b1_i = inverse_slr_propagation(alpha, beta)
        # b1 = [b1_i] + b1

    return b1[::-1]
