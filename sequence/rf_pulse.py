from __future__ import annotations

import math
import pprint
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from bloch.simulate import run
from constants import GAMMA_RAD
from sequence.gradient import Gradient
from sequence.object import SequenceObject, generate_times


def merge(rf_pulses: list[RFPulse], delta_time: float = 1e-6) -> RFPulse:
    for pulse in rf_pulses:
        if not isinstance(pulse, RFPulse):
            raise ValueError("All pulses must be of type RFPulse!")

    total_duration = rf_pulses[0].get_times().max()
    total_waveform = rf_pulses[0].get_waveform()
    for pulse in rf_pulses[1:]:
        total_duration += pulse.get_times()[1:].max()
        total_waveform = np.concatenate((total_waveform, pulse.get_waveform()[1:]))

    times = generate_times(delta_time, total_duration)

    return RFPulse(delta_time, times, total_waveform)


class RFPulse(SequenceObject):
    def __init__(self, delta_time: float, times: np.ndarray, waveform: np.ndarray, **kwargs):
        super().__init__()
        self.delta_time = delta_time
        self.times = times
        self.waveform = waveform

        self.additional_info = kwargs

        self.amplitude = 1
        self.delta_frequency = 0

    def __add__(self, other_pulse) -> 'RFPulse':
        comb_waveform = self.get_waveform() + other_pulse.get_waveform()
        comb_waveform = self.normalize(comb_waveform)

        return RFPulse(self.delta_time, self.times, comb_waveform)

    def __sub__(self, other_pulse) -> 'RFPulse':
        comb_waveform = self.get_waveform() - other_pulse.get_waveform()
        comb_waveform = self.normalize(comb_waveform)

        return RFPulse(self.delta_time, self.times, comb_waveform)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def append(self, other_pulse: RFPulse | list[RFPulse], delay: float = 0.0) -> None:
        if not isinstance(other_pulse, list):
            other_pulse = [other_pulse]

        for pulse in other_pulse:
            new_times, comb_data = self._append(pulse, delay)

            self.times = new_times
            self.waveform = comb_data
            self.amplitude = 1

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

        plt.plot(1e3 * self.times, magnitude / np.max(magnitude))
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

    def get_flip_angle(self):
        return np.sum(GAMMA_RAD * self.get_waveform() * self.delta_time).real

    def get_exact_amplitude(self, flip_angle: float) -> float:
        return self.amplitude * flip_angle / self.get_flip_angle()

    def get_optimal_amplitude(self, max_b1: float, desired_range: tuple[float, float],
                              flip_angle: float, delta_time: float = 1e-4, min_b1: float = 1e-6, display: bool = False) -> float:

        num_amplitudes = int((max_b1 - min_b1) / 1e-6)
        amplitudes = np.linspace(min_b1, max_b1, num_amplitudes + 1)

        initial_pulse = self.get_waveform(delta_time) / self.amplitude
        mult_num_iso = math.ceil(np.max(np.abs([value for value in desired_range])) / 500.0)
        frequency_limit = mult_num_iso * 500

        off_resonance = np.linspace(-frequency_limit, frequency_limit, 50 + 10 * mult_num_iso)
        mxy_profiles = np.zeros((len(amplitudes), len(off_resonance)), dtype=np.float64)
        mz_profiles = np.zeros((len(amplitudes), len(off_resonance)), dtype=np.float64)

        for index, amplitude in enumerate(amplitudes):
            test_pulse = amplitude * initial_pulse
            magnetisation = run(t1=np.inf, t2=np.inf, simulation_style='non_selective',
                                df=off_resonance,
                                rf_pulse=test_pulse, delta_time=delta_time,
                                verbose=False)
            mxy_profiles[index, :] = np.abs(magnetisation[-1, :, 0] + 1j * magnetisation[-1, :, 1])
            mz_profiles[index, :] = magnetisation[-1, :, 2]

        weighting = np.where(np.logical_and(desired_range[0] <= off_resonance, off_resonance <= desired_range[1]),
                             1, 0.5)

        alpha = np.tile(weighting, (len(amplitudes), 1))

        mz_desired = math.cos(flip_angle)
        mxy_desired = math.sin(flip_angle)

        inversion_score = np.sum((weighting * (mz_profiles - mz_desired)) ** 2, axis=1)
        excitation_score = np.sum((weighting * (mxy_profiles - mxy_desired)) ** 2, axis=1)

        if flip_angle <= np.pi / 2:
            excitation_score *= 4

        ideal_amplitude = amplitudes[np.argmin(excitation_score + inversion_score)].item()

        if display:
            fig, _ = plt.subplots(1, 2, figsize=(6, 6), sharex=True, sharey=True)
            fig.text(0.5, 0.04, 'Isochromat Frequency (Hz)', ha='center')

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


def binomial_pulse(subpulse_duration: float, interpulse_delay: float, binomial_order: int,
                   delta_time: float, flip_angle: float) -> RFPulse:
    binomial_coefficients = [math.comb(binomial_order, k) for k in range(binomial_order + 1)]

    pulses = []
    for index, coeff in enumerate(binomial_coefficients):
        pulse = rect_pulse(subpulse_duration, delta_time)
        if index != len(binomial_coefficients) - 1:
            pulse.zero_pad(interpulse_delay)

        amplitude = pulse.get_exact_amplitude(coeff / sum(binomial_coefficients) * flip_angle)
        pulse.set_amplitude(amplitude)
        pulses.append(pulse)

    binomial_pulse = merge(pulses, delta_time=delta_time)

    return binomial_pulse


# Shaped Pulses
def sinc_pulse(duration: float, bandwidth: float, delta_time: float = 1e-6,
               hamming: bool = False) -> RFPulse:
    times = generate_times(delta_time, duration)

    hamming_filter = (1 + np.cos(2 * np.pi * (times - duration / 2) / duration)) / 2 if hamming else 1
    magnitude = np.sinc(bandwidth * (times - duration / 2)) * hamming_filter

    return RFPulse(delta_time, times, magnitude,
                   duration=duration, bandwidth=bandwidth,
                   title=f"Sinc Pulse ($T_p$ = {round(duration * 1e3, 2)}ms, "
                         f"BW = {bandwidth}Hz)")


def gaussian_pulse(duration: float, bandwidth: float, delta_time: float = 1e-6,
                   hamming: bool = True) -> RFPulse:
    times = generate_times(delta_time, duration)

    hamming_filter = (1 + np.cos(2 * np.pi * (times - duration / 2) / duration)) / 2 if hamming else 1
    magnitude = np.exp(-(np.pi * bandwidth * (times - duration / 2)) ** 2 / (4 * np.log(2))) * hamming_filter

    return RFPulse(delta_time, times, magnitude,
                   duration=duration, bandwidth=bandwidth,
                   title=f"Gaussian Pulse ($T_p$ = {round(duration * 1e3, 2)}ms, "
                         f"BW = {bandwidth}Hz)")


def hermite_pulse(duration: float, bandwidth: float, order: int, factors: list[float],
                  delta_time: float = 1e-6) -> RFPulse:
    if order < 2 and order % 2 != 0:
        raise ValueError('Order must be at least 2 and even!')

    times = generate_times(delta_time, duration)

    hermite_envelope = np.zeros_like(times)
    for index, factor in enumerate(factors):
        hermite_envelope += factor * (1e3 * (times - duration / 2)) ** index

    hermite_envelope = np.clip(hermite_envelope, -1, 1)

    # plt.plot(hermite_envelope)
    # plt.show()

    gaussian_envelope = np.exp(-(np.pi * bandwidth * (times - duration / 2)) ** 2 / (4 * np.log(2)))
    magnitude = hermite_envelope * gaussian_envelope

    return RFPulse(delta_time, times, magnitude,
                   duration=duration, bandwidth=bandwidth,
                   envelope=hermite_envelope,
                   title=f"Hermite Pulse ($T_p$ = {round(duration * 1e3, 2)}ms, "
                         f"BW = {bandwidth}Hz, order = {order}, "
                         f"factors = {*factors,}")


# Adiabatic Pulses
def hypsec_pulse(duration: float, bandwidth: Optional[int] = None,
                 empirical_factor: Optional[float] = None, beta: Optional[float] = None,
                 delta_time: float = 1e-6) -> RFPulse:
    num_viable_parameters = len([param for param in [bandwidth, empirical_factor, beta] if param is not None])

    if num_viable_parameters != 2:
        raise AttributeError(f'{num_viable_parameters} parameters provided! '
                             f'Please provide two of the three parameters: empirical_factor, beta, and bandwidth!')

    times = generate_times(delta_time, duration)

    if beta is None:
        beta = np.pi * bandwidth / empirical_factor
    elif empirical_factor is None:
        empirical_factor = np.pi * bandwidth / beta
    else:
        bandwidth = empirical_factor * beta / np.pi

    magnitude = 1 / np.cosh(beta * (times - duration / 2))
    frequency_sweep = -empirical_factor * beta * np.tanh(beta * (times - duration / 2))
    phase = np.cumsum(frequency_sweep) * delta_time

    adiabatic_threshold = beta * np.sqrt(empirical_factor) / (np.pi * 42.58e6)

    return RFPulse(delta_time, times, magnitude * np.exp(1j * phase),
                   duration=duration, bandwidth=bandwidth,
                   adiabatic_threshold=adiabatic_threshold,
                   title=f"HS Pulse ($T_p$ = {round(duration * 1e3, 2)}ms, "
                         f"BW = {bandwidth}Hz, μ = {round(empirical_factor, 2)}, "
                         f"β = {round(beta, 1)}s⁻¹)")


def bir4_pulse(duration: float, bandwidth: float, beta: float, flip_angle: float, delta_time: float = 1e-6) -> RFPulse:
    times = generate_times(delta_time, duration)

    def b1(times: np.ndarray) -> np.ndarray:
        return 1 / np.cosh(beta * 4 * times / duration)

    def phase(times: np.ndarray) -> np.ndarray:
        return np.pi * bandwidth * duration / (2 * beta) * np.log(np.cosh(beta * 4 * times / duration))

    first_segment = (times <= duration / 4)
    first_magnitude_segment = np.where(first_segment,  b1(times), 0)
    first_phase_segment = np.where(first_segment, phase(times), 0)

    second_segment = (times >= duration / 4) & (times <= duration / 2)
    second_magnitude_segment = np.where(second_segment, b1(duration / 2 - times), first_magnitude_segment)
    second_phase_segment = np.where(second_segment, phase(duration / 2 - times) + np.pi + flip_angle / 2, first_phase_segment)

    third_segment = (times >= duration / 2) & (times <= 3 * duration / 4)
    third_magnitude_segment = np.where(third_segment, b1(times - duration / 2), second_magnitude_segment)
    third_phase_segment = np.where(third_segment, phase(times - duration / 2) + np.pi + flip_angle / 2, second_phase_segment)

    fourth_segment = (times >= 3 * duration / 4)
    fourth_magnitude_segment = np.where(fourth_segment, b1(duration - times), third_magnitude_segment)
    fourth_phase_segment = np.where(fourth_segment, phase(duration - times), third_phase_segment)

    magnitude = fourth_magnitude_segment
    phase = fourth_phase_segment

    return RFPulse(delta_time, times, magnitude * np.exp(1j * phase),
                   duration=duration, bandwidth=bandwidth,
                   title=f"BIR4 Pulse ($T_p$ = {round(duration * 1e3, 2)}ms, "
                         f"BW = {bandwidth}Hz, β = {round(beta, 1)}, "
                         f"θ = {round(flip_angle * 180 / np.pi, 1)}°)")


def foci_pulse(duration: float, gradient_strength: float, max_gradient_strength: float, bandwidth: Optional[int] = None,
               empirical_factor: Optional[float] = None, beta: Optional[float] = None,
               delta_time: float = 1e-6) -> tuple[RFPulse, Gradient]:
    num_viable_parameters = len([param for param in [bandwidth, empirical_factor, beta] if param is not None])

    if num_viable_parameters != 2:
        raise AttributeError(f'{num_viable_parameters} parameters provided! '
                             f'Please provide two of the three parameters: empirical_factor, beta, and bandwidth!')

    if beta is None:
        beta = np.pi * bandwidth / empirical_factor
    elif empirical_factor is None:
        empirical_factor = np.pi * bandwidth / beta
    else:
        bandwidth = empirical_factor * beta / np.pi

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
