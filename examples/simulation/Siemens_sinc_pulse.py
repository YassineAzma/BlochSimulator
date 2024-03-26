import numpy as np

import bloch
import sequence
import visualise
from vendor.siemens import extract_rf_pulse

# NON-SELECTIVE SINC EXCITATION PULSE EXAMPLE

# Import RF waveform from text file exported from Siemens IDEA simulator
sinc_waveform = extract_rf_pulse('sinc_pulse.txt', 0)

# Convert the numpy array into an RF pulse object for pulse amplitude optimisation to achieve a 90 degree excitation.
# Siemens uses a convention of 1us for the dwell time in exported txt files, so we use a delta time of 1e-6.
sinc_pulse = sequence.rf.create(sinc_waveform, 1e-6)

# Perform pulse amplitude determination via fast non-selective Bloch simulations across a desired frequency range.
estimated_amplitude = sinc_pulse.get_optimal_amplitude(30e-6, (-200.0, 200.0), np.pi / 2, display=True)

# For non-phase modulated pulses, such as a sinc pulse, we can compute the pulse amplitude to achieve a flip angle
# analytically.
exact_amplitude = sinc_pulse.get_exact_amplitude(np.pi / 2)
print(estimated_amplitude, exact_amplitude)

# Print pulse information. This will be displayed in the terminal.
sinc_pulse.get_info()

# Set the pulse amplitude
sinc_pulse.set_amplitude(exact_amplitude)

# Prepare Bloch simulation
# Rather than sampling the RF waveform every second, we can speed up the simulation by sampling every 10us without
# changing the resulting frequency profile appreciably.
delta_time = 1e-5
rf_waveform = sinc_pulse.get_waveform(delta_time)
off_resonances = np.linspace(-3000, 3000, 500)

# The magnetisation returned has dimensions (number of time steps, number of off resonances, 3)
magnetisation = bloch.simulate.run(t1=np.inf, t2=np.inf, simulation_style='non_selective', df=off_resonances,
                                   rf_pulse=sinc_pulse.get_waveform(1e-5), delta_time=1e-5)

# Display the magnetisation dynamics as an animation
visualise.animate(magnetisation, delta_time, 'non_selective', off_resonances=off_resonances,
                  rf_pulse=sinc_pulse, repeat=True)
