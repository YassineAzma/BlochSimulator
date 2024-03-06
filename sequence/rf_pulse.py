import math
import pprint

import numpy as np
from matplotlib import pyplot as plt

from bloch.simulate import non_selective_rot3d_matrix
from sequence.gradient import Gradient
from sequence.object import SequenceObject, generate_times


class RFPulse(SequenceObject):
    def __init__(self, delta_time: float, times: np.ndarray, waveform: np.ndarray, **kwargs):
        super().__init__()
        self.delta_time = delta_time
        self.times = times
        self.waveform = waveform

        self.additional_info = kwargs

        self.amplitude = 1
        self.delta_frequency = 0

    def __add__(self, other_pulse):
        comb_waveform = self.get_waveform() + other_pulse.get_waveform()
        comb_waveform = self.normalize(comb_waveform)

        return RFPulse(self.delta_time, self.times, comb_waveform)

    def __sub__(self, other_pulse):
        comb_waveform = self.get_waveform() - other_pulse.get_waveform()
        comb_waveform = self.normalize(comb_waveform)

        return RFPulse(self.delta_time, self.times, comb_waveform)

    def append(self, other_pulse, delay: float = 0.0):
        new_times, comb_data = self._append(other_pulse, delay)

        return RFPulse(self.delta_time, new_times, comb_data)

    def magnitude(self, new_delta_time: float = None) -> np.ndarray:
        return np.abs(self.get_waveform(new_delta_time))

    def phase(self, new_delta_time: float = None) -> np.ndarray:
        return np.angle(self.get_waveform(new_delta_time))

    def frequency_sweep(self, new_delta_time: float = None) -> np.ndarray:
        if new_delta_time:
            return np.gradient(np.unwrap(self.phase(new_delta_time))) / (2 * np.pi * new_delta_time)
        else:
            return np.gradient(np.unwrap(self.phase())) / (2 * np.pi * self.delta_time)

    def set_delta_frequency(self, delta_frequency: float):
        self.delta_frequency = delta_frequency

    def get_waveform(self, new_delta_time: float = None) -> np.ndarray:
        temp_waveform = self.waveform * np.exp(1j * 2 * np.pi * self.delta_frequency * self.times)
        if new_delta_time:
            return self.amplitude * self.resample(self.delta_time, new_delta_time, temp_waveform)

        return self.amplitude * temp_waveform

    def get_info(self) -> dict:
        power_integral = 1e3 * np.sum(np.abs(self.waveform) ** 2 * 1e-6)
        amplitude_integral = 1e3 * np.sqrt((np.sum(self.waveform.real * 1e-6)) ** 2 +
                                           (np.sum(self.waveform.imag * 1e-6)) ** 2)
        magnitude_integral = 1e3 * np.sum(np.abs(self.waveform) * 1e-6)
        normalised_rms = np.sqrt((1 / self.additional_info.get('duration')) * np.trapz(np.abs(self.waveform) ** 2,
                                                                                       dx=self.delta_time))
        time_bandwidth_product = self.additional_info.get('duration') * self.additional_info.get('bandwidth', 0.0)

        pulse_dict = {
            'Power Integral': f'{round(power_integral, 3)}uTms',
            'Amplitude Integral': f'{round(amplitude_integral, 3)}√uTms',
            'Magnitude Integral': f'{round(magnitude_integral, 3)}uTms',
            'Normalised RMS': f'{round(normalised_rms, 3)}uT',
            'Time-Bandwidth Product': f'{round(time_bandwidth_product, 2)}',
        }

        adiabatic_threshold = self.additional_info.get('adiabatic_threshold')
        if adiabatic_threshold is not None:
            pulse_dict['Adiabatic Threshold'] = f'{round(adiabatic_threshold * 1e6, 2)}uT'

        pprint.pprint(pulse_dict, sort_dicts=False)

        return pulse_dict

    def display(self, title: str = None):
        magnitude = self.magnitude()
        phase = self.phase()

        if phase is None:
            plt.figure()
        else:
            plt.subplots(2, 1)
            plt.subplot(2, 1, 1)

        plt.plot(1e3 * self.times, magnitude)
        plt.xlabel('Time (ms)') if phase is None else None
        plt.ylabel('Normalised Amplitude')
        plt.grid()

        if phase is not None:
            plt.subplot(2, 1, 2)
            plt.plot(1e3 * self.times, phase)
            plt.xlabel('Time (ms)')
            plt.ylabel('Phase (rad)')
            plt.grid()

        plt.suptitle(title if title is not None else self.additional_info.get("title"))
        plt.show()

    def get_optimal_amplitude(self, max_b1: float, desired_range: tuple[float, float],
                              flip_angle: float, delta_time: float = 1e-4, display: bool = False) -> float:

        num_amplitudes = int(max_b1 / 1e-6)
        amplitudes = np.linspace(1e-6, max_b1, num_amplitudes)

        initial_pulse = self.get_waveform(delta_time)
        mult_num_iso = math.ceil(np.max(np.abs([value for value in desired_range])) / 500.0)
        frequency_limit = mult_num_iso * 500

        off_resonance = np.linspace(-frequency_limit, frequency_limit, 100 + 10 * mult_num_iso)
        mxy_profiles = np.zeros((len(amplitudes), len(off_resonance)), dtype=np.float64)
        mz_profiles = np.zeros((len(amplitudes), len(off_resonance)), dtype=np.float64)

        for index, amplitude in enumerate(amplitudes):
            test_pulse = amplitude * initial_pulse
            magnetisation = non_selective_rot3d_matrix(t1=np.inf, t2=np.inf, df=off_resonance,
                                                       rf_pulse=test_pulse, delta_time=delta_time)
            mxy_profiles[index, :] = np.abs(magnetisation[-1, :, 0] + 1j * magnetisation[-1, :, 1])
            mz_profiles[index, :] = magnetisation[-1, :, 2]

        weighting = np.where(np.logical_and(desired_range[0] <= off_resonance, off_resonance <= desired_range[1]),
                             1, 0.5)

        alpha = np.tile(weighting, (len(amplitudes), 1))

        mz_desired = math.cos(flip_angle)
        mxy_desired = math.sin(flip_angle)

        inversion_score = np.sum((weighting * (mz_profiles - mz_desired)) ** 2, axis=1)
        excitation_score = np.sum((weighting * (mxy_profiles - mxy_desired)) ** 2, axis=1)

        ideal_amplitude = amplitudes[np.argmin(excitation_score + inversion_score)].item()

        if display:
            fig, _ = plt.subplots(1, 2, figsize=(6, 6), sharex=True, sharey=True)
            fig.text(0.5, 0.04, 'Off-Resonance (Hz)', ha='center')

            plt.subplot(1, 2, 1)
            plt.imshow(mxy_profiles, cmap='jet', origin='lower', vmin=-1, vmax=1,
                       aspect='auto', extent=(min(off_resonance), max(off_resonance),
                                              0, max_b1 * 1e6), alpha=alpha)
            plt.plot(off_resonance, [ideal_amplitude * 1e6] * len(off_resonance),
                     ls='--', color='white')
            # plt.colorbar()
            plt.ylabel('Amplitude (uT)')
            plt.title('$M_{xy}$')

            if ideal_amplitude > 0.5 * max_b1:
                shift = -2e-6
            else:
                shift = 1e-6

            plt.text(0, (ideal_amplitude + shift) * 1e6, 'Best = {:.1f} uT'.format(ideal_amplitude * 1e6),
                     color='white')

            plt.subplot(1, 2, 2)
            plt.imshow(mz_profiles, cmap='jet', origin='lower', vmin=-1, vmax=1,
                       aspect='auto', extent=(min(off_resonance), max(off_resonance),
                                              0, max_b1 * 1e6), alpha=alpha)
            plt.plot(off_resonance, [ideal_amplitude * 1e6] * len(off_resonance),
                     ls='--', color='white')

            plt.text(0, (ideal_amplitude + shift) * 1e6, 'Best = {:.1f} uT'.format(ideal_amplitude * 1e6),
                     color='white')

            plt.title('$M_{z}$')
            plt.colorbar()
            plt.show()

        return ideal_amplitude


# PULSES
def rect_pulse(duration: float, delta_time: float) -> RFPulse:
    times = generate_times(delta_time, duration)

    magnitude = np.ones_like(times)

    return RFPulse(delta_time, times, magnitude,
                   duration=duration)


# Shaped Pulses
def sinc_pulse(duration: float, bandwidth: float, delta_time: float,
               hamming: bool = False) -> RFPulse:
    times = generate_times(delta_time, duration)

    hamming_filter = (1 + np.cos(2 * np.pi * (times - duration / 2) / duration)) / 2 if hamming else 1
    magnitude = np.sinc(np.pi * bandwidth * (times - duration / 2)) * hamming_filter

    return RFPulse(delta_time, times, magnitude,
                   duration=duration, bandwidth=bandwidth,
                   title=f"Sinc Pulse ($T_p$ = {round(duration * 1e3, 2)}ms, "
                         f"BW = {bandwidth}Hz)")


def gaussian_pulse(duration: float, bandwidth: float, delta_time: float,
                   hamming: bool = True) -> RFPulse:
    times = generate_times(delta_time, duration)

    hamming_filter = (1 + np.cos(2 * np.pi * (times - duration / 2) / duration)) / 2 if hamming else 1
    magnitude = np.exp(-(np.pi * bandwidth * (times - duration / 2)) ** 2 / (4 * np.log(2))) * hamming_filter

    return RFPulse(delta_time, times, magnitude,
                   duration=duration, bandwidth=bandwidth,
                   title=f"Gaussian Pulse ($T_p$ = {round(duration * 1e3, 2)}ms, "
                         f"BW = {bandwidth}Hz)")


def hermite_pulse(duration: float, bandwidth: float, order: int, factors: list[float], delta_time: float) -> RFPulse:
    if order < 2 and order % 2 != 0:
        raise ValueError('Order must be at least 2 and even!')

    times = generate_times(delta_time, duration)

    hermite_envelope = np.zeros_like(times)
    for index, factor in enumerate(factors):
        hermite_envelope += factor * (1e3 * (times - duration / 2)) ** index

    hermite_envelope = np.clip(hermite_envelope, -1, 1)

    plt.plot(hermite_envelope)
    plt.show()

    gaussian_envelope = np.exp(-(np.pi * bandwidth * (times - duration / 2)) ** 2 / (4 * np.log(2)))
    magnitude = hermite_envelope * gaussian_envelope

    return RFPulse(delta_time, times, magnitude,
                   duration=duration, bandwidth=bandwidth,
                   envelope=hermite_envelope,
                   title=f"Hermite Pulse ($T_p$ = {round(duration * 1e3, 2)}ms, "
                         f"BW = {bandwidth}Hz, order = {order}, "
                         f"factors = {*factors,}")


# Adiabatic Pulses
def hypsec_pulse(duration: float, bandwidth: int,
                 empirical_factor: float, delta_time: float) -> RFPulse:
    times = generate_times(delta_time, duration)

    beta = np.pi * bandwidth / empirical_factor

    magnitude = 1 / np.cosh(beta * (times - duration / 2))
    frequency_sweep = -empirical_factor * beta * np.tanh(beta * (times - duration / 2))
    phase = np.cumsum(frequency_sweep) * delta_time

    adiabatic_threshold = beta * np.sqrt(empirical_factor) / (np.pi * 42.58e6)

    return RFPulse(delta_time, times, magnitude * np.exp(1j * phase),
                   duration=duration, bandwidth=bandwidth,
                   adiabatic_threshold=adiabatic_threshold,
                   title=f"HS Pulse ($T_p$ = {round(duration * 1e3, 2)}ms, "
                         f"BW = {bandwidth}Hz, μ = {empirical_factor}, "
                         f"β = {round(beta, 1)}s⁻¹)")


def foci_pulse(duration: float, bandwidth: int, empirical_factor: float,
               gradient_strength: float, max_gradient_strength: float,
               delta_time: float) -> tuple[RFPulse, Gradient]:

    times = generate_times(delta_time, duration)

    beta = np.pi * bandwidth / empirical_factor
    grad_factor = max_gradient_strength / gradient_strength
    amplitude = np.where(np.cosh(beta * (times - duration / 2)) < grad_factor, np.cosh(beta * (times - duration / 2)),
                         grad_factor)

    plt.plot(amplitude)
    plt.show()
    magnitude = amplitude / np.cosh(beta * (times - duration / 2))
    frequency_sweep = -empirical_factor * amplitude * beta * np.tanh(beta * (times - duration / 2))
    phase = np.cumsum(frequency_sweep) * delta_time

    rf_object = RFPulse(delta_time, times, magnitude * np.exp(1j * phase),
                        duration=duration, bandwidth=bandwidth,
                        title=f"FOCI Pulse ($T_p$ = {round(duration * 1e3, 2)}ms, "
                              f"BW = {bandwidth}Hz, μ = {empirical_factor}, "
                              f"β = {round(beta, 1)}s⁻¹)")

    grad_object = Gradient(delta_time, times, gradient_strength * amplitude)

    return rf_object, grad_object


def create(data: np.ndarray, delta_time: float) -> RFPulse:
    duration = (len(data) - 1) * delta_time
    times = generate_times(delta_time, duration)

    rf_pulse = RFPulse(delta_time, times, RFPulse.normalize(data),
                       duration=duration)

    return rf_pulse
