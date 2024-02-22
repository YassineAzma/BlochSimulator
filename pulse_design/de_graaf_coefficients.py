import numpy as np

import bloch
import visualise
from sequence.object import generate_times
from sequence.rf_pulse import RFPulse

de_graaf_SLR_pulses = {
    'Excitation':
        {
            'Linear':
                {
                    6:
                        {
                            'A': [0.16248152, -0.33473283, 0.31861783, -0.15415863, 0.02225950, -0.00511679,
                                  0.00156740, -0.00088433, 0.00013945, -0.00008296, 0.00005558, 0.00006721,
                                  0.00010308, 0.00012230, 0.00014492, 0.00015219, 0.00016416, 0.00017416,
                                  0.00018167, 0.00018828, 0.00019865],
                            'B': [0] * 20
                        },
                    12: {
                            'A': [0.07926787, -0.15568177, 0.15832894, -0.16016458, 0.16450203, -0.16193649,
                                  0.08371448, -0.01839556, 0.00737875, -0.00405617, 0.00266805, -0.00158901,
                                  0.00105189, -0.00065938, 0.00036897, -0.00026906, 0.00008059, -0.00011846,
                                  -0.00003266, -0.00008225, -0.00005674],
                            'B': [0] * 20
                        },
                    18: {
                            'A': [0.05078526, -0.10383633, 0.10238266, -0.10380703, 0.10432680, -0.]
                    }
                },
            'Refocused':
                {

                }
        }

}


def generate_de_graaf_pulse(pulse_type: str, phase_type: str, r_value: int, duration: float) -> RFPulse:
    coefficient_dict = de_graaf_SLR_pulses[pulse_type][phase_type][r_value]
    a = coefficient_dict['A']
    b = coefficient_dict['B']

    times = generate_times(1e-6, duration)
    waveform = a[0]

    for index, (a_value, b_value) in enumerate(zip(a[1:], b[1:])):
        waveform += (a_value * np.cos(2 * np.pi * (index + 1) * times / duration) +
                     b_value * np.sin(2 * np.pi * (index + 1) * times / duration))

    rf_object = RFPulse(1e-6, times, waveform,
                        duration=duration)

    return rf_object


test = generate_de_graaf_pulse('Excitation', 'Linear', 6, 3e-3)
amplitude = test.get_optimal_amplitude(30e-6, (-1000.0, 1000.0), np.pi / 2, display=True)
test.set_amplitude(amplitude)

isochromats = np.linspace(-2500, 2500, 500)
magnetisation = bloch.simulate.non_selective_rot3d_matrix(t1=np.inf, t2=np.inf, df=isochromats, position=,
                                                          rf_pulse=test.get_waveform(1e-5), gradients=, delta_time=1e-5)

animation = visualise.non_selective_animation(test, magnetisation, isochromats, 1e-5, magnetisation.shape[0], play=True,
                                              phase_mode=0, save_path=None)
