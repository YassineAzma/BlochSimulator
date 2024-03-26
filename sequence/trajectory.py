import numpy as np
from matplotlib import pyplot as plt

from constants import GAMMA
from sequence.gradient import Gradient
from sequence.object import generate_times


def constant_angle_spiral_trajectory(duration: float, k_max: float, cycles: float):
    times = np.arange(0, duration, 1e-6)
    kx = k_max * (1 - times / duration) * np.cos(2 * np.pi * cycles * times / duration)
    ky = k_max * (1 - times / duration) * np.sin(2 * np.pi * cycles * times / duration)

    return kx, ky


def spatial_frequency_to_gradient(spatial_frequency: np.ndarray, delta_time: float = 1e-6) -> Gradient:
    gradient_waveform = np.gradient(spatial_frequency, delta_time) / GAMMA
    gradient_amplitude = np.abs(gradient_waveform).max()

    duration = (len(spatial_frequency) - 1) * delta_time
    times = generate_times(delta_time, duration)

    gradient = Gradient(delta_time, times, gradient_waveform / gradient_amplitude)
    gradient.set_amplitude(gradient_amplitude)

    return gradient
