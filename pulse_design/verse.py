from matplotlib import pyplot as plt

import sequence.rf_pulse
import numpy as np

from sequence.object import generate_times

test_pulse = sequence.rf_pulse.sinc_pulse(4e-3, 2400, 1e-6)
test_pulse.display()


def squeeze_pulse(rf_pulse: sequence.rf_pulse.RFPulse, amplitude_max: float = 30e-6, min_duration: float = 1e-3,
                  delta_time: float = 1e-6):

    max_amplitude = rf_pulse.get_exact_amplitude(np.pi / 2)
    rf_pulse.set_amplitude(max_amplitude)
    original_duration = rf_pulse.get_times().max()
    current_duration = rf_pulse.get_times().max()
    iter = 1.0

    new_rf_pulse = rf_pulse.__copy__()
    while (max_amplitude <= amplitude_max) and (current_duration >= min_duration):
        new_delta_time = (current_duration - 0.1e-3) / original_duration * delta_time
        new_duration = current_duration - 0.1e-3
        times = generate_times(new_delta_time, new_duration)

        new_rf_pulse.times = times
        new_rf_pulse.delta_time = new_delta_time
        max_amplitude = new_rf_pulse.get_exact_amplitude(np.pi / 2)
        current_duration = times.max()

        print(1e3 * current_duration, 1e3 * max_amplitude)
        iter += 1


squeeze_pulse(test_pulse, 30e-6, 1e-3, 1e-6)
