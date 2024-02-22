import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.animation import FuncAnimation

from sequence import rf


def non_selective_animation(rf_pulse: rf.RFPulse, magnetisation: np.ndarray,
                            df: np.ndarray, delta_time: float,
                            play: bool = True, repeat: bool = False, phase_mode: int = 0, save_path: str = None):
    mxy = np.abs(magnetisation[:, :, 0] + 1j * magnetisation[:, :, 1])
    mz = magnetisation[:, :, 2]

    sim_length = magnetisation.shape[0]

    # Set up the figure and axis
    fig, ax = plt.subplots(2, 2, figsize=(10, 6))
    plt.ioff()

    # Magnitude subplot
    ax[0, 0].plot(1e3 * rf_pulse.get_times(delta_time), 1e6 * rf_pulse.magnitude(delta_time))
    mag_marker, = ax[0, 0].plot(1e3 * rf_pulse.get_times(delta_time)[0],
                                1e6 * rf_pulse.magnitude(delta_time)[0], color='r', marker='o')
    ax[0, 0].set(ylabel='Amplitude (uT)',
                 title=rf_pulse.additional_info.get("title"))
    ax[0, 0].grid()

    # Gradient subplot
    ax[0, 1].set_axis_off()

    if phase_mode == 0:
        phase_data = rf_pulse.phase(delta_time)
    else:
        phase_data = rf_pulse.frequency_sweep(delta_time)

    ax[1, 0].plot(1e3 * rf_pulse.get_times(delta_time), phase_data)
    phase_marker, = ax[1, 0].plot(1e3 * rf_pulse.get_times(delta_time)[0],
                                  phase_data[0], color='r', marker='o')
    ax[1, 0].set(xlabel='Time (ms)',
                 ylabel='Frequency Sweep (Hz)' if phase_mode == 1 else 'Phase (rad)')
    ax[1, 0].grid()

    mxy_line, = ax[1, 1].plot(df, mxy[0, :], color='k', label='$|M_{xy}|$')
    mz_line = ax[1, 1].plot(df, mz[0, :], color='r', label='$M_z$')[0]

    ax[1, 1].set(xlabel='Isochromat Frequency (Hz)',
                 ylabel='Magnetisation',
                 ylim=[-1.1, 1.1])
    box = ax[1, 1].get_position()
    ax[1, 1].set_position([box.x0, box.y0, box.width * 0.75, box.height])
    ax[1, 1].legend(loc='center left', framealpha=0.3, bbox_to_anchor=(0, -0.25),
                    fancybox=True, shadow=False, ncol=1)
    ax[1, 1].grid()
    plt.tight_layout()

    # Function to update the plot for each frame
    def update(frame):
        total_time = sim_length * delta_time
        nearest_time = frame * total_time / num_frames
        nearest_frame = np.argmin(np.abs(nearest_time - rf_pulse.get_times(delta_time)))

        mag_marker.set_xdata([1e3 * rf_pulse.get_times(delta_time)[nearest_frame]])
        mag_marker.set_ydata([1e6 * rf_pulse.magnitude(delta_time)[nearest_frame]])

        phase_marker.set_xdata([1e3 * rf_pulse.get_times(delta_time)[nearest_frame]])
        phase_marker.set_ydata([phase_data[nearest_frame]])

        mxy_line.set_ydata(mxy[nearest_frame, :])
        mz_line.set_ydata(mz[nearest_frame, :])

        return mag_marker, phase_marker, mxy_line, mz_line

    # Create the animation
    num_frames = 1000
    bloch_animation = FuncAnimation(fig, update, frames=num_frames, interval=1,
                                    blit=True, repeat=repeat)
    if save_path:
        bloch_animation.save(save_path, writer="ffmpeg")

    if play:
        plt.show()

    return animation
