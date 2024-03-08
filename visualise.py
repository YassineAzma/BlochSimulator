from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.animation import FuncAnimation

from sequence import rf, gradient

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


def selective_animation(rf_pulse: rf.RFPulse, grad_x: gradient.Gradient,
                        grad_y: gradient.Gradient, grad_z: gradient.Gradient,
                        magnetisation: np.ndarray, positions: np.ndarray, delta_time: float,
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

    # Gradient subplots
    ax[0, 1].plot(1e3 * grad_x.get_times(delta_time), 1e3 * grad_x.get_waveform(delta_time), color='k')
    ax[0, 1].plot(1e3 * grad_y.get_times(delta_time), 1e3 * grad_y.get_waveform(delta_time), color='r')
    ax[0, 1].plot(1e3 * grad_z.get_times(delta_time), 1e3 * grad_z.get_waveform(delta_time), color='b')

    gx_marker, = ax[0, 1].plot(1e3 * grad_x.get_times(delta_time)[0],
                                1e3 * grad_x.get_waveform(delta_time)[0], color='k', marker='o',
                               label='$G_x$')
    gy_marker, = ax[0, 1].plot(1e3 * grad_y.get_times(delta_time)[0],
                                1e3 * grad_y.get_waveform(delta_time)[0], color='r', marker='o',
                               label='$G_y$')
    gz_marker, = ax[0, 1].plot(1e3 * grad_z.get_times(delta_time)[0],
                                1e3 * grad_z.get_waveform(delta_time)[0], color='b', marker='o',
                               label='$G_z$')

    ax[0, 1].legend(loc='upper left', framealpha=0.3)

    ax[0, 1].set(xlabel='Time (ms)',
                 ylabel='Gradient (mT/m)',
                 title='Gradient Waveforms')
    ax[0, 1].grid()

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

    mxy_line, = ax[1, 1].plot(1e3 * positions[:, 2], mxy[0, :], color='k', label='$|M_{xy}|$')
    mz_line = ax[1, 1].plot(1e3 * positions[:, 2], mz[0, :], color='r', label='$M_z$')[0]

    ax[1, 1].set(xlabel='Position (mm)',
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

        gx_marker.set_xdata([1e3 * grad_x.get_times(delta_time)[nearest_frame]])
        gx_marker.set_ydata([1e3 * grad_x.get_waveform(delta_time)[nearest_frame]])

        gy_marker.set_xdata([1e3 * grad_y.get_times(delta_time)[nearest_frame]])
        gy_marker.set_ydata([1e3 * grad_y.get_waveform(delta_time)[nearest_frame]])

        gz_marker.set_xdata([1e3 * grad_z.get_times(delta_time)[nearest_frame]])
        gz_marker.set_ydata([1e3 * grad_z.get_waveform(delta_time)[nearest_frame]])

        phase_marker.set_xdata([1e3 * rf_pulse.get_times(delta_time)[nearest_frame]])
        phase_marker.set_ydata([phase_data[nearest_frame]])

        mxy_line.set_ydata(mxy[nearest_frame, :])
        mz_line.set_ydata(mz[nearest_frame, :])

        return mag_marker, phase_marker, gx_marker, gy_marker, gz_marker, mxy_line, mz_line

    # Create the animation
    num_frames = 1000
    bloch_animation = FuncAnimation(fig, update, frames=num_frames, interval=1,
                                    blit=True, repeat=repeat)
    if save_path:
        bloch_animation.save(save_path, writer="ffmpeg")

    if play:
        plt.show()

    return animation


def pulse_time_efficiency(off_resonances: np.ndarray, magnetisation: np.ndarray,
                          delta_time: float, is_inversion: bool):
    mxy = np.abs(magnetisation[:, :, 0] + 1j * magnetisation[:, :, 1])
    mz = magnetisation[:, :, 2]

    def iterate(is_inversion: bool, ratio: float) -> (np.ndarray, np.ndarray):
        first_moves = np.zeros(magnetisation.shape[1])
        final_moves = np.zeros(magnetisation.shape[1])
        for index in range(magnetisation.shape[1]):
            mxy_values = mxy[:, index]
            mz_values = mz[:, index]

            if is_inversion:
                possible_first_moves = np.where(mz_values < ratio)[0]
                possible_final_moves = np.where(mz_values < -ratio)[0]
                remains_inverted = mz_values[-1] < -ratio

                if not remains_inverted:
                    possible_final_moves = []
            else:
                possible_first_moves = np.where(mxy_values > 1 - ratio)[0]
                possible_final_moves = np.where(mxy_values > ratio)[0]
                remains_excited = mxy_values[-1] > ratio

                if not remains_excited:
                    possible_final_moves = []

            first_move = np.nan if len(possible_first_moves) == 0 else possible_first_moves[0]
            final_move = np.nan if len(possible_final_moves) == 0 else possible_final_moves[0]

            first_moves[index] = first_move
            final_moves[index] = final_move

        return first_moves, final_moves

    if is_inversion is False:
        first, final = iterate(is_inversion=False, ratio=0.95)
    else:
        first, final = iterate(is_inversion=True, ratio=0.95)

    fig, _ = plt.subplots(1, 2, figsize=(6, 6), sharex=True, sharey=False)
    fig.text(0.5, 0.04, 'Off-Resonance (Hz)', ha='center')
    plt.subplot(1, 2, 1)
    plt.ylabel('Time (ms)')

    if not is_inversion:
        plt.imshow(mxy, cmap='jet', origin='lower', vmin=-1, vmax=1,
                   aspect='auto', extent=(min(off_resonances), max(off_resonances),
                                          0, magnetisation.shape[0] * delta_time * 1e3))

        plt.plot(off_resonances, delta_time * 1e3 * first, color='white', ls='--')
        plt.text(np.median(off_resonances), (first[len(first) // 2] - len(first) / 20) * delta_time * 1e3,
                 '$M_{xy}$ > 0.05', color='white', horizontalalignment='center')

        plt.plot(off_resonances, delta_time * 1e3 * final, color='white', ls='--')
        plt.text(np.median(off_resonances), (final[len(final) // 2] + len(final) / 40) * delta_time * 1e3,
                 '$M_{xy}$ > 0.95', color='white', horizontalalignment='center')

        plt.title('$M_{xy}$')
    else:
        plt.imshow(mz, cmap='jet', origin='lower', vmin=-1, vmax=1,
                   aspect='auto', extent=(min(off_resonances), max(off_resonances),
                                          0, magnetisation.shape[0] * delta_time * 1e3))

        plt.plot(off_resonances, delta_time * 1e3 * first, color='white', ls='--')
        plt.text(np.median(off_resonances), (first[len(first) // 2] - len(first) / 20) * delta_time * 1e3,
                 '$M_{z}$ > 0.95', color='white', horizontalalignment='center')

        plt.plot(off_resonances, delta_time * 1e3 * final, color='white', ls='--')
        plt.text(np.median(off_resonances), (final[len(final) // 2] + len(final) / 40) * delta_time * 1e3,
                 '$M_{z}$ < -0.95', color='white', horizontalalignment='center')

        plt.title('$M_{z}$')

    plt.subplot(1, 2, 2)
    plt.plot(off_resonances, delta_time * 1e3 * (final - first), color='k', label='Time Efficiency')
    plt.ylabel('Time (ms)')
    plt.axhline(np.nanmean(final - first) * delta_time * 1e3, ls='--', label='Mean Time')
    plt.grid()
    plt.xlim([min(off_resonances), max(off_resonances)])
    title = 'Time from $M_{xy}$ > 0.05 to $M_{xy}$ > 0.95' if not is_inversion else 'Time from $M_{z}$ < 0.95 to $M_{z}$ < -0.95'
    plt.title(title)
    plt.legend()
    plt.show()


def frequency_profile(off_resonances: np.ndarray, magnetisation: np.ndarray):
    end_mxy = np.abs(magnetisation[-1, :, 0] + 1j * magnetisation[-1, :, 1])
    end_mz = magnetisation[-1, :, 2]

    plt.plot(off_resonances, end_mxy, label='$M_{xy}$', color='black')
    plt.plot(off_resonances, end_mz, label='$M_{z}$', color='red')
    plt.xlabel('Isochromat Frequency (Hz)')
    plt.ylabel('Magnetisation')
    plt.title('Frequency Profile')
    plt.ylim([-0.05, 1.05])
    plt.legend()
    plt.grid()
    plt.show()
