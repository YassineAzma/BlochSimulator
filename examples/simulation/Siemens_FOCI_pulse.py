import numpy as np
from matplotlib import pyplot as plt

import bloch
import sequence
import visualise
from sequence.gradient import calculate_excitation_amplitude

# SELECTIVE INVERSION EXAMPLE

# Load data using numpy.loadtxt, skipping lines starting with '#'
magnitude = np.loadtxt("../../test_files/FOCI_ABS.txt", comments='#')
phase = np.loadtxt("../../test_files/FOCI_PHS.txt", comments='#')
gradient = np.loadtxt('../../test_files/FOCI_AHPE_FCT.txt', comments='#')

# Extract times and data columns
times = (magnitude[:, 0] - min(magnitude[:, 0])) * 1e-6
mag_waveform = magnitude[:, 1]
phase_waveform = phase[:, 1]

gradient_waveform = gradient[:, 1]

gx = sequence.gradient.rect_gradient(10.23e-3, 0, 1e-5)
gy = sequence.gradient.rect_gradient(10.23e-3, 0, 1e-5)
gz = sequence.gradient.create(gradient[:, 1] * (3.7e-3 / max(gradient[:, 1])), 1e-5)

# Convert the numpy array into an RF pulse object for pulse amplitude optimisation to achieve a 90 degree excitation.
rf_pulse = sequence.rf.create(mag_waveform * np.exp(1j * phase_waveform), 1e-5)
pulse_amplitude = rf_pulse.get_optimal_amplitude(30e-6, (-1000.0, 1000.0), np.pi, display=True)
rf_pulse.set_amplitude(pulse_amplitude)
rf_pulse.set_delta_frequency(0)

# Prepare Bloch simulation
# Rather than sampling the RF waveform every second, we can speed up the simulation by sampling every 10us without
# changing the resulting frequency profile appreciably.

delta_time = 1e-5
rf_waveform = rf_pulse.get_waveform(delta_time)
off_resonances = np.linspace(-1000, 1000, 500)
positions = np.zeros((len(off_resonances), 3))
positions[:, 2] = np.linspace(-15e-3, 15e-3, 500)

# The magnetisation returned has dimensions (number of time steps, number of off resonances, 3)
magnetisation = bloch.simulate.spatial_selective_rot3d_matrix(t1=np.inf, t2=np.inf, position=positions,
                                                              rf_pulse=rf_waveform, grad_x=gx.get_waveform(delta_time),
                                                              grad_y=gy.get_waveform(delta_time), grad_z=gz.get_waveform(delta_time),
                                                              delta_time=delta_time)
visualise.selective_animation(rf_pulse, gx,
                              gy, gz,
                              magnetisation, positions, 1e-5,
                              play=True, phase_mode=0, save_path=None)
