import numpy as np

import bloch
import sequence
import visualise
from sequence.gradient import calculate_excitation_amplitude

# SELECTIVE C-FOCI INVERSION PULSE EXAMPLE

# Generate C-FOCI pulse and accompanying gradient modulation.
grad_strength = calculate_excitation_amplitude(2000, 5e-3)
foci_pulse, gradient_z = sequence.rf.foci_pulse(7.68e-3, grad_strength, 25e-3, empirical_factor=3.9,
                                                bandwidth=2000, delta_time=1e-6)

# Print pulse information. This will be displayed in the terminal.
foci_pulse.get_info()

# Display pulse waveforms
foci_pulse.display()

# As the pulse has a phase modulation, we empirically determine the optimal amplitude via fast Bloch simulations. As the
# phase varies rapidly with time, we use a relatively slow delta time.
pulse_amplitude = foci_pulse.get_optimal_amplitude(30e-6, (-2000.0, 2000.0),
                                                   np.pi, delta_time=5e-5, display=True)

# Set the pulse amplitude
foci_pulse.set_amplitude(pulse_amplitude)

# Prepare Bloch simulation
# Rather than sampling the RF waveform every second, we can speed up the simulation by sampling every 10us without
# changing the resulting frequency profile appreciably.
delta_time = 1e-5

# Isochromat positions in z
num_isochromats = 500
positions = np.zeros((num_isochromats, 3))
positions[:, 2] = np.linspace(-15e-3, 15e-3, num_isochromats)

sel_magnetisation = bloch.simulate.run(t1=np.inf, t2=np.inf, simulation_style='spatial_selective', position=positions,
                                       rf_pulse=foci_pulse.get_waveform(delta_time),
                                       grad_z=gradient_z.get_waveform(delta_time),
                                       delta_time=delta_time)

# Display the magnetisation dynamics as an animation
visualise.animate(magnetisation=sel_magnetisation, delta_time=delta_time, simulation_style='1d_selective',
                  positions=positions, position_axis=2, rf_pulse=foci_pulse,
                  grad_z=gradient_z,
                  repeat=False, save_path=None)
