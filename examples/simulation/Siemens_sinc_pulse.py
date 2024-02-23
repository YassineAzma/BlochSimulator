import numpy as np
from matplotlib import pyplot as plt

import bloch
import sequence
from utilities import extract_rf_pulse

# NON-SELECTIVE EXCITATION EXAMPLE

# Import RF waveform from text file exported from IDEA simulator
rf_pulse = extract_rf_pulse('sinc_pulse.txt', 0)

# Convert the numpy array into an RF pulse object for pulse amplitude optimisation to achieve a 90 degree excitation.
# Siemens uses a convention of 1us for each time interval, so we use a delta time of 1e-6.

pulse = sequence.rf.create(rf_pulse, 1e-6)
pulse_amplitude = pulse.get_optimal_amplitude(30e-6, (-1000.0, 1000.0), np.pi / 2, display=True)
pulse.set_amplitude(pulse_amplitude)

# Prepare Bloch simulation
# Rather than sampling the RF waveform every second, we can speed up the simulation by sampling every 10us without
# changing the resulting frequency profile appreciably.

delta_time = 1e-5
rf_waveform = pulse.get_waveform(delta_time)
off_resonances = np.linspace(-3000, 3000, 500)

# The magnetisation returned has dimensions (number of time steps, number of off resonances, 3)
magnetisation = bloch.simulate.non_selective_rot3d_matrix(t1=np.inf, t2=np.inf, df=off_resonances,
                                                          rf_pulse=pulse.get_waveform(1e-5), delta_time=1e-5)

# Visualisation
end_mxy = np.abs(magnetisation[-1, :, 0] + 1j * magnetisation[-1, :, 1])
end_mz = magnetisation[-1, :, 2]
plt.plot(off_resonances, end_mxy, label='$M_{xy}$', color='black')
plt.plot(off_resonances, end_mz, label='$M_{z}$', color='red')
plt.xlabel('Off-Resonance Frequency (Hz)')
plt.ylabel('Magnetisation')
plt.title('Frequency Profile')
plt.ylim([-0.05, 1.05])
plt.legend()
plt.grid()
plt.show()
