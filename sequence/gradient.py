import numpy as np
from matplotlib import pyplot as plt

from constants import GAMMA
from sequence.object import SequenceObject, generate_times


class Gradient(SequenceObject):
    def __init__(self, delta_time: float, times: np.ndarray, waveform: np.ndarray, **kwargs):
        super().__init__()
        self.delta_time = delta_time
        self.times = times
        self.waveform = waveform

        self.additional_info = kwargs

        self.amplitude = 1

    def __add__(self, other_pulse):
        comb_waveform = self.get_waveform() + other_pulse.get_waveform()

        return Gradient(self.delta_time, self.times, comb_waveform)

    def __sub__(self, other_pulse):
        comb_waveform = self.get_waveform() - other_pulse.get_waveform()

        return Gradient(self.delta_time, self.times, comb_waveform)

    def append(self, other_pulse, delay: float = 0.0) -> None:
        new_times, comb_data = self._append(other_pulse, delay)

        self.times = new_times
        self.waveform = comb_data

    def slew_rate(self, new_delta_time: float = None):
        if new_delta_time:
            slew_rate = np.gradient(self.get_waveform(new_delta_time), new_delta_time)
        else:
            slew_rate = np.gradient(self.get_waveform(), self.delta_time)

        return slew_rate

    def net_phase(self, new_delta_time: float = None):
        if new_delta_time:
            net_phase = np.cumsum(self.get_waveform(new_delta_time) * new_delta_time)
        else:
            net_phase = np.cumsum(self.get_waveform() * self.delta_time)

        return net_phase * 42.58e6 * 2 * np.pi

    def get_waveform(self, new_delta_time: float = None) -> np.ndarray:
        temp_waveform = self.waveform

        if new_delta_time:
            return self.amplitude * self.resample(self.delta_time, new_delta_time, temp_waveform)
        else:
            return self.amplitude * temp_waveform

    def display(self, title: str = None):
        plt.plot(1e3 * self.times, self.get_waveform() / self.amplitude)
        plt.xlabel('Time (ms)')
        plt.ylabel('Normalised Amplitude')
        plt.grid()

        plt.suptitle(title if title is not None else self.additional_info.get("title"))
        plt.show()


def calculate_excitation_amplitude(bandwidth: float, slice_thickness: float) -> float:
    return bandwidth / (GAMMA * slice_thickness)


def apply_eddy_currents(gradient: Gradient, amplitudes: np.ndarray,
                        rate_constants: np.ndarray) -> list[Gradient]:
    times = gradient.get_times()
    temp_waveform = gradient.get_waveform()
    slew_rate = gradient.slew_rate() * gradient.delta_time

    decay = np.sum(amplitudes[:, None] * np.exp(-rate_constants[:, None] * times), axis=0)
    eddy_waveform = np.convolve(-slew_rate, decay)

    new_times = generate_times(gradient.delta_time, (len(eddy_waveform) - 1) * gradient.delta_time)
    padded_waveform = np.concatenate([temp_waveform, np.zeros(len(eddy_waveform) - len(temp_waveform))])

    corrupted_gradient = Gradient(gradient.delta_time, new_times, padded_waveform + eddy_waveform)
    eddy_gradient = Gradient(gradient.delta_time, new_times, eddy_waveform)
    preemphasised_gradient = Gradient(gradient.delta_time, new_times, padded_waveform - eddy_waveform)

    return [corrupted_gradient, eddy_gradient, preemphasised_gradient]


def rect_gradient(duration: float, amplitude: float, delta_time: float) -> Gradient:
    times = generate_times(delta_time, duration)
    gradient = np.full(len(times), amplitude)

    return Gradient(delta_time, times, gradient)


def trapezium_gradient(ramp_time: float, duration: float, amplitude: float,
                       delta_time: float) -> Gradient:
    times = generate_times(delta_time, 2 * ramp_time + duration)
    ramp_length = round(ramp_time / delta_time)
    flat_length = round(duration / delta_time)

    rise_increment = amplitude / ramp_length
    rise_gradient = np.linspace(0, amplitude - rise_increment, ramp_length, dtype=np.float64)
    flat_gradient = np.full(flat_length + 1, amplitude)
    fall_gradient = np.linspace(amplitude - rise_increment, 0, ramp_length, dtype=np.float64)

    gradient = np.concatenate([rise_gradient, flat_gradient, fall_gradient])

    return Gradient(delta_time, times, gradient)


def periodic_trapezium_gradient(ramp_time: float, duration: float, amplitude: float,
                                num_lobes: int, delta_time: float) -> Gradient:
    desired_gradient = trapezium_gradient(ramp_time, duration, amplitude, delta_time)
    polarity_reversed_gradient = trapezium_gradient(ramp_time, duration, -amplitude, delta_time)

    gradient_lobes = [desired_gradient if i % 2 == 0 else polarity_reversed_gradient
                      for i in range(num_lobes)]

    final_gradient = Gradient(delta_time, desired_gradient.get_times(), desired_gradient.get_waveform())
    for lobe in gradient_lobes[1:]:
        final_gradient.append(lobe)

    return final_gradient


def sinusoidal_gradient(ramp_time: float, duration: float, amplitude: float,
                        delta_time: float) -> Gradient:
    times = generate_times(delta_time, 2 * ramp_time + duration)

    ramp_length = round(ramp_time / delta_time)
    flat_length = round(duration / delta_time)

    rise_ramp = np.sin(np.linspace(0, np.pi / 2, ramp_length)) * amplitude
    flat_gradient = np.full(flat_length + 1, amplitude)
    fall_ramp = np.sin(np.linspace(np.pi / 2, 0, ramp_length)) * amplitude

    gradient = np.concatenate([rise_ramp, flat_gradient, fall_ramp])

    return Gradient(delta_time, times, gradient)


def periodic_sinusoidal_gradient(ramp_time: float, duration: float, amplitude: float,
                                 num_lobes: int, delta_time: float) -> Gradient:
    desired_gradient = sinusoidal_gradient(ramp_time, duration, amplitude, delta_time)
    polarity_reversed_gradient = sinusoidal_gradient(ramp_time, duration, -amplitude, delta_time)

    gradient_lobes = [desired_gradient if i % 2 == 0 else polarity_reversed_gradient
                      for i in range(num_lobes)]

    final_gradient = Gradient(delta_time, desired_gradient.get_times(), desired_gradient.get_waveform())
    for lobe in gradient_lobes[1:]:
        final_gradient.append(lobe)

    return final_gradient


def constant_angle_spiral_gradients(duration: float, amplitude: float, cycles: float, delta_time: float = 1e-6):
    times = generate_times(delta_time, duration)

    gx = -(1 / (duration * GAMMA)) * (2 * np.pi * cycles * (1 - times / duration)
                                      * np.sin(2 * np.pi * cycles * times / duration) +
                                      np.cos(2 * np.pi * cycles * times / duration))

    gx = amplitude * gx / np.abs(gx).max()

    gy = 1 / (duration * GAMMA) * (2 * np.pi * cycles * (1 - times / duration)
                                   * np.cos(2 * np.pi * cycles * times / duration) -
                                   np.sin(2 * np.pi * cycles * times / duration))

    gy = amplitude * gy / np.abs(gy).max()

    return Gradient(delta_time, times, gx), Gradient(delta_time, times, gy)


def create(data: np.ndarray, delta_time: float) -> Gradient:
    duration = (len(data) - 1) * delta_time
    times = generate_times(delta_time, duration)

    gradient = Gradient(delta_time, times, data)

    return gradient


def gradient_to_spatial_frequency(gradient: Gradient, delta_time: float = 1e-6) -> np.ndarray:
    gradient_waveform = gradient.get_waveform(delta_time)

    spatial_frequency = GAMMA * np.cumsum(gradient_waveform) * delta_time

    return spatial_frequency
