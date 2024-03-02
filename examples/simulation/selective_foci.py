import time
import bloch
import numpy as np
import sequence
import visualise
from sequence.gradient import calculate_excitation_amplitude

num_isochromats = 500
positions = np.zeros((num_isochromats, 3))
positions[:, 2] = np.linspace(-15e-3, 15e-3, num_isochromats)

gradient_x = sequence.gradient.rect_gradient(10.24e-3, 0, 1e-6)
gradient_y = sequence.gradient.rect_gradient(10.24e-3, 0, 1e-6)
grad_strength = calculate_excitation_amplitude(2000, 5e-3)
foci_pulse, gradient_z = sequence.rf.foci_pulse(10.24e-3, 2000, 2.5,
                                                grad_strength, 25e-3, 1e-6)
foci_pulse.get_info()
foci_pulse.display()
pulse_amplitude = foci_pulse.get_optimal_amplitude(30e-6, (-2000.0, 2000.0),
                                                   np.pi, delta_time=5e-5, display=True)
foci_pulse.set_amplitude(pulse_amplitude)
foci_pulse.set_delta_frequency(42.58 * 2.89 * 0)

delta_time = 1e-5

init_time = time.perf_counter()
sel_magnetisation = bloch.simulate.selective_rot3d_matrix(t1=np.inf, t2=np.inf, position=positions,
                                                          rf_pulse=foci_pulse.get_waveform(delta_time),
                                                          grad_x=gradient_x.get_waveform(delta_time),
                                                          grad_y=gradient_y.get_waveform(delta_time),
                                                          grad_z=gradient_z.get_waveform(delta_time),
                                                          delta_time=delta_time)
end_time = time.perf_counter()
sim_length, num_iso, _ = sel_magnetisation.shape
print(f"Time taken = {round(end_time - init_time, 2)}s to "
      f"simulate {sim_length} time steps for {num_iso} isochromats!")

visualise.selective_animation(rf_pulse=foci_pulse,
                              grad_x=gradient_x, grad_y=gradient_y, grad_z=gradient_z,
                              magnetisation=sel_magnetisation, positions=positions, delta_time=delta_time,
                              play=True, repeat=False, phase_mode=1, save_path=None)
