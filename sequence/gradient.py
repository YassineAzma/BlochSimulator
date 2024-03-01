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

    def append(self, other_pulse, delay: float = 0.0):
        new_times, comb_data = self._append(other_pulse, delay)

        return Gradient(self.delta_time, new_times, comb_data)

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


def calculate_excitation_amplitude(bandwidth: float, slice_thickness: float) -> float:
    return bandwidth / (GAMMA * slice_thickness)


def apply_eddy_currents(gradient: Gradient, amplitudes: np.ndarray,
                        rate_constants: np.ndarray, new_delta_time: float = None) -> list[Gradient]:
    times = gradient.get_times(new_delta_time) if new_delta_time else gradient.get_times()
    temp_waveform = gradient.get_waveform(new_delta_time) if new_delta_time else gradient.get_waveform()
    slew_rate = gradient.slew_rate(new_delta_time) * (new_delta_time if new_delta_time else gradient.delta_time)
    delta_time = new_delta_time if new_delta_time else gradient.delta_time

    decay = np.sum(amplitudes[:, None] * np.exp(-rate_constants[:, None] * times), axis=0)
    eddy_waveform = np.convolve(-slew_rate, decay)

    new_times = generate_times(delta_time, (len(eddy_waveform) - 1) * delta_time)
    padded_waveform = np.concatenate([temp_waveform, np.zeros(len(eddy_waveform) - len(temp_waveform))])

    corrupted_gradient = Gradient(gradient.delta_time, new_times, padded_waveform + eddy_waveform)
    eddy_gradient = Gradient(new_delta_time, new_times, eddy_waveform)
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


def create(data: np.ndarray, delta_time: float) -> Gradient:
    times = generate_times(delta_time, (len(data) - 1) * delta_time)
    gradient = Gradient(delta_time, times, data)

    return gradient
