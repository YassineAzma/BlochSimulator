import time

import bloch
import numpy as np
import sequence
from visualise import non_selective_animation


def main():
    isochromats = np.linspace(-1000, 1000, 500)
    positions = np.zeros((len(isochromats), 3))
    pulse_1 = sequence.rf.hypsec_pulse(5e-3, 2400, 5, 1e-6)
    pulse_1.get_info()
    pulse_1.display()
    pulse_1.set_delta_frequency(2.89 * 42.58 * -2.3)
    pulse_amplitude = pulse_1.get_optimal_amplitude(30e-6, (-2000.0, 2000.0),
                                                    np.pi, display=True)
    pulse_1.set_amplitude(pulse_amplitude)

    delta_time = 1e-5

    init_time = time.perf_counter()
    magnetisation = bloch.simulate.non_selective_rot3d_matrix(t1=np.inf, t2=23.34, df=isochromats,
                                                              rf_pulse=pulse_1.get_waveform(delta_time),
                                                              delta_time=delta_time)
    end_time = time.perf_counter()
    sim_length, num_iso, _ = magnetisation.shape
    print(f"Time taken = {round(end_time - init_time, 2)}s to "
          f"simulate {sim_length} time steps for {num_iso} isochromats!")

    animation = non_selective_animation(test_pulse, magnetisation, isochromats, delta_time, play=True,
                                        repeat=False, phase_mode=0, save_path=None)


if __name__ == '__main__':
    main()
