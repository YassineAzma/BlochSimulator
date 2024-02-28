import time

from matplotlib import pyplot as plt

import bloch
import numpy as np
import sequence
import visualise
from scipy import optimize


def main():
    isochromats = np.linspace(-1000, 1000, 500)
    positions = np.zeros((len(isochromats), 3))
    positions[:, 2] = np.linspace(-5e-3, 5e-3, 500)
    gradient_x = sequence.gradient.rect_gradient(7.68e-3, 0, 1e-6)
    gradient_y = sequence.gradient.rect_gradient(7.68e-3, 0, 1e-6)
    pulse_1, gradient_z = sequence.rf.foci_pulse(7.68e-3, 2000, 3.9, 3e-3, 1e-6)
    pulse_1.get_info()
    pulse_1.display()
    pulse_1.set_delta_frequency(2.89 * 42.58 * 0)
    pulse_amplitude = pulse_1.get_optimal_amplitude(30e-6, (-2000.0, 2000.0),
                                                    np.pi, display=True)
    pulse_1.set_amplitude(pulse_amplitude)

    delta_time = 1e-5

    init_time = time.perf_counter()
    sel_magnetisation = bloch.simulate.selective_rot3d_matrix(t1=np.inf, t2=np.inf, position=positions,
                                                              rf_pulse=pulse_1.get_waveform(delta_time),
                                                              grad_x=gradient_x.get_waveform(delta_time),
                                                              grad_y=gradient_y.get_waveform(delta_time),
                                                              grad_z=gradient_z.get_waveform(delta_time),
                                                              delta_time=delta_time)
    non_sel_magnetisation = bloch.simulate.non_selective_rot3d_matrix(t1=np.inf, t2=np.inf, df=isochromats,
                                                                      rf_pulse=pulse_1.get_waveform(delta_time),
                                                                      delta_time=delta_time)
    end_time = time.perf_counter()
    sim_length, num_iso, _ = sel_magnetisation.shape
    print(f"Time taken = {round(end_time - init_time, 2)}s to "
          f"simulate {sim_length} time steps for {num_iso} isochromats!")

    visualise.non_selective_animation(rf_pulse=pulse_1, magnetisation=non_sel_magnetisation, df=isochromats,
                                      delta_time=delta_time,
                                      play=True, repeat=False, phase_mode=1, save_path=None)
    visualise.selective_animation(rf_pulse=pulse_1,
                                  grad_x=gradient_x, grad_y=gradient_y, grad_z=gradient_z,
                                  magnetisation=sel_magnetisation, positions=positions, delta_time=delta_time,
                                  play=True, repeat=False, phase_mode=1, save_path=None)
    # visualise.pulse_time_efficiency(isochromats, magnetisation, delta_time, True)

    desired_frequencies = [0.9, 1.3, 1.59, 2.02, 2.25, 2.77, 4.7, 5.31]
    desired_frequencies = [(frequency - 4.7) * 42.58 * 2.89 for frequency in desired_frequencies]

    beta = np.zeros(len(desired_frequencies))
    for index, frequency in enumerate(desired_frequencies):
        idx = np.argmin(np.abs(isochromats - frequency))
        mz = non_sel_magnetisation[-1, idx, 2]
        efficiency = np.round((1 - mz) / 2, 4)
        beta[index] = efficiency

    print(beta)


if __name__ == '__main__':
    main()
