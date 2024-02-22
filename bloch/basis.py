import os

import numba
import numpy as np
from numba import float64, complex128

from constants import PAULI_X, PAULI_Y, PAULI_Z

os.environ["OPENBLAS_NUM_THREADS"] = "2"


@numba.njit(complex128[:, ::1](float64[::1]), cache=True)
def magnetisation_to_quaternion(magnetisation: np.ndarray) -> np.ndarray:
    return 0.5 * (np.identity(2) + magnetisation[0] * PAULI_X +
                  magnetisation[1] * PAULI_Y + magnetisation[2] * PAULI_Z)


@numba.njit(float64[::1](complex128[:, ::1]), cache=True)
def quaternion_to_magnetisation(quaternion: np.ndarray) -> np.ndarray:
    mx = (quaternion[1, 0] + quaternion[0, 1]).real
    my = (quaternion[1, 0] - quaternion[0, 1]).imag
    mz = (quaternion[0, 0] - quaternion[1, 1]).real

    return np.array([mx, my, mz])


def cayley_klein_to_magnetisation(alpha: np.ndarray, beta: np.ndarray, mxy_absolute: bool = True) -> np.ndarray:
    mxy = 2 * beta * alpha.conj()
    mz = alpha * alpha.conj() - beta * beta.conj()

    if mxy_absolute:
        return np.array([np.abs(mxy), mz.real])
    else:
        return np.array([mxy.real, mxy.imag, mz])
