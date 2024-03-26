import numpy as np

import bloch.simulate
import sequence
import visualise

binomial_order = 5
interpulse_duration = 0.15e-3
subpulse_duration = 0.15e-3
total_duration = round((binomial_order * 2 + 1) * subpulse_duration, 6)

slice_thickness = 5e-3
gradient_z = sequence.gradient.periodic_sinusoidal_gradient(0.0e-3, subpulse_duration, 20e-3, binomial_order * 2 + 1,
                                                            1e-6)

binomial_pulse = sequence.rf.binomial_pulse(subpulse_duration, interpulse_duration, binomial_order, 1e-6, np.pi / 2)
binomial_pulse.display()

off_resonances = np.linspace(-2000, 2000, 50)
positions = np.zeros((150, 3))
z = np.linspace(-15e-3, 15e-3, 150)
positions[:, 2] = z

magnetisation = bloch.simulate.run(np.inf, np.inf, 'spectral_spatial', df=off_resonances, position=positions,
                                   rf_pulse=binomial_pulse.get_waveform(1e-5),
                                   grad_z=gradient_z.get_waveform(1e-5),
                                   delta_time=1e-5)

visualise.animations.animate(magnetisation, 1e-5, 'spectral_spatial', position_axis=2,
                             positions=positions, off_resonances=off_resonances,
                             rf_pulse=binomial_pulse, grad_z=gradient_z, repeat=False)
