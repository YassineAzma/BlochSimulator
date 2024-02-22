import numpy as np

GAMMA = 42577478.518
GAMMA_RAD = 267522190.018

PAULI_X = np.array([[0, 1],
                    [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j],
                    [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0],
                    [0, -1]], dtype=complex)
