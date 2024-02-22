import numpy as np

import bloch.simulate
import sequence
import visualise
from pulse_design import slr
from scipy import signal

from sequence.rf_pulse import RFPulse


class SLRDesign:
    """
    Parameter Relations for the Shinnar-Le Roux Selective Excitation Pulse Design Algorithm
    John Pauly, Member, IEEE, Patrick Le Roux, Dwight Nishimura, Member, IEEE, and
    Albert Macovski, Fellow, IEEE
    """

    def __init__(self, passband_ripple: float, stopband_ripple: float,
                 time_bandwidth_product: float, pulse_type: str):
        self.passband_ripple = passband_ripple
        self.stopband_ripple = stopband_ripple
        self.time_bandwidth_product = time_bandwidth_product
        self.pulse_type = pulse_type
        self.bsf, self.delta_pass, self.delta_stop = self.ripple_terms()

    def ripple_terms(self) -> tuple[float, float, float]:
        def case_0(s_1, s_2):
            return s_1, s_2

        def case_1(s_1, s_2):
            return np.sqrt(s_1 / 2), s_2 / np.sqrt(2)

        def case_2(s_1, s_2):
            return s_1 / 8, np.sqrt(s_2 / 2)

        def case_3(s_1, s_2):
            return s_1 / 4, np.sqrt(s_2)

        def case_4(s_1, s_2):
            return s_1 / 2, np.sqrt(s_2)

        table = {
            'small_tip': case_0,
            'excitation': case_1,
            'inversion': case_2,
            'refocusing': case_3,
            'saturation': case_4,
        }

        delta_pass, delta_stop = table.get(self.pulse_type)(self.passband_ripple, self.stopband_ripple)
        bsf = 1 if self.pulse_type not in ['excitation', 'saturation'] else np.sqrt(1 / 2)

        return bsf, delta_pass, delta_stop

    def fir_filter_performance(self, phase_type: str) -> float:
        a = [5.309e-3, 7.114e-2, -4.761e-1, -2.66e-3, -5.941e-1, -4.278e-1]
        match phase_type.lower():
            case 'linear' | 'quadratic':
                delta_pass = self.delta_pass
                delta_stop = self.delta_stop
            case 'minimum':
                delta_pass = 2 * self.delta_stop
                delta_stop = self.delta_pass ** 2 / 2
            case _:
                raise ValueError('Phase type must be "linear" or "minimum"')

        l1 = np.log10(delta_pass)
        l2 = np.log10(delta_stop)

        performance = (a[0] * l1 ** 2 + a[1] * l1 + a[2]) * l2 + (a[3] * l1 ** 2 + a[4] * l1 + a[5])
        performance *= 0.5 if phase_type.lower() == 'minimum' else 1

        return performance

    def recover_alpha(self, num_samples: int, oversampling_factor: int, beta: np.ndarray) -> np.ndarray:
        beta_z = np.fft.fft(beta, n=oversampling_factor * 2 * num_samples)
        absolute_alpha = np.sqrt(1 - (beta_z * np.conj(beta_z)))
        log_alpha = np.log(np.abs(absolute_alpha))
        fft_log_alpha = np.fft.fft(log_alpha)
        fft_log_alpha[1: oversampling_factor * num_samples] *= 2
        fft_log_alpha[oversampling_factor * num_samples + 1:] *= 0
        log_alpha = np.fft.ifft(fft_log_alpha)

        alpha_z = np.exp(log_alpha)
        alpha = np.fft.fft(alpha_z, n=oversampling_factor * 2 * num_samples)[:num_samples]
        alpha /= num_samples * oversampling_factor * 2

        return alpha[::-1]

    def design_pulse(self, n_samples: int, duration: float, oversampling_factor: int = 8,
                     phase_type: str = 'linear') -> RFPulse:
        bandwidth = self.time_bandwidth_product / duration
        sampling_frequency = 1 / (duration / n_samples)
        performance = self.fir_filter_performance(phase_type)

        fractional_transition_width = performance / self.time_bandwidth_product
        transition_bandwidth = bandwidth * fractional_transition_width

        pass_frequency = (bandwidth - transition_bandwidth) / 2
        stop_frequency = (bandwidth + transition_bandwidth) / 2

        if phase_type.lower() == 'minimum':
            frequency_bands = [0, pass_frequency, stop_frequency, 0.5 * sampling_frequency]
            gain_bands = [1, 1, 0, 0]
            weights = [1 / self.delta_pass, 1 / self.delta_stop]

            beta = signal.firls(2 * n_samples - 1, frequency_bands, gain_bands, weight=weights, fs=sampling_frequency)
            beta = signal.minimum_phase(beta)
        elif phase_type.lower() == 'linear':
            frequency_bands = [0, pass_frequency, stop_frequency, 0.5 * sampling_frequency]
            gain_bands = [1, 1, 0, 0]
            weights = [1 / self.delta_pass, 1 / self.delta_stop]

            beta = self.bsf * signal.firls(n_samples, frequency_bands, gain_bands, weight=weights,
                                           fs=sampling_frequency)
        else:
            raise ValueError('Phase type must be "linear" or "minimum"')

        alpha = self.recover_alpha(n_samples, oversampling_factor, beta)
        waveform = slr.inverse_slr(alpha, beta)

        pulse = sequence.rf.create(waveform, duration / len(waveform))

        return pulse


bandwidth = 3000
duration = 2e-3
time_bandwidth_product = bandwidth * duration

test = SLRDesign(0.001, 0.01, time_bandwidth_product, 'excitation')
pulse = test.design_pulse(255, duration, 2, 'linear')
pulse_amplitude = pulse.get_optimal_amplitude(30e-6, (-1000.0, 1000.0), np.pi / 2, display=True)
pulse.set_amplitude(pulse_amplitude)
pulse.get_info()

isochromats = np.linspace(-2500, 2500, 500)
magnetisation = bloch.simulate.non_selective_rot3d_matrix(t1=np.inf, t2=np.inf, df=isochromats, position=,
                                                          rf_pulse=pulse.get_waveform(1e-5), gradients=,
                                                          delta_time=1e-5)

animation = visualise.non_selective_animation(pulse, magnetisation, isochromats, 1e-5, magnetisation.shape[0],
                                              play=True, phase_mode=0, save_path=None)
