import numpy as np

import bloch
import sequence
import visualise
from vendor.siemens import extract_rf_pulse

# NON-SELECTIVE EXCITATION EXAMPLE

# Import RF waveform from text file exported from IDEA simulator
sinc_waveform = extract_rf_pulse('sinc_pulse.txt', 0)

# Convert the numpy array into an RF pulse object for pulse amplitude optimisation to achieve a 90 degree excitation.
# Siemens uses a convention of 1us for each time interval, so we use a delta time of 1e-6.

sinc_pulse = sequence.rf.create(sinc_waveform, 1e-6)
pulse_amplitude = sinc_pulse.get_optimal_amplitude(30e-6, (-1000.0, 1000.0), np.pi / 2, display=True)
sinc_pulse.get_info()
sinc_pulse.set_amplitude(pulse_amplitude)

# Prepare Bloch simulation
# Rather than sampling the RF waveform every second, we can speed up the simulation by sampling every 10us without
# changing the resulting frequency profile appreciably.

delta_time = 1e-5
rf_waveform = sinc_pulse.get_waveform(delta_time)
off_resonances = np.linspace(-3000, 3000, 500)

# The magnetisation returned has dimensions (number of time steps, number of off resonances, 3)
magnetisation = bloch.simulate.non_selective_rot3d_matrix(t1=np.inf, t2=np.inf, df=off_resonances,
                                                          rf_pulse=sinc_pulse.get_waveform(1e-5), delta_time=1e-5)

visualise.animate(magnetisation, delta_time, 'non_selective', off_resonances=off_resonances,
                  rf_pulse=sinc_pulse, repeat=True)
