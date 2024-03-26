from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from sequence import rf, gradient


def get_mxy(magnetisation: np.ndarray) -> np.ndarray:
    return np.abs(magnetisation[:, :, 0] + 1j * magnetisation[:, :, 1])


def get_mz(magnetisation: np.ndarray) -> np.ndarray:
    return magnetisation[:, :, 2]


def get_waveform_data(rf_pulse: rf.RFPulse, grad_x: Optional[gradient.Gradient],
                      grad_y: Optional[gradient.Gradient], grad_z: Optional[gradient.Gradient],
                      delta_time) -> tuple[list, list, list]:
    titles = ['$|RF|$ (uT)', '$∠RF$ (rad)']
    times = [rf_pulse.get_times(), rf_pulse.get_times()]
    waveforms = [1e6 * np.abs(rf_pulse.get_waveform()), rf_pulse.phase()]

    grad_titles = ['$G_{X}$ (mT/m)', '$G_{Y}$ (mT/m)', '$G_{Z}$ (mT/m)']
    for index, grad_object in enumerate([grad_x, grad_y, grad_z]):
        if grad_object is not None:
            titles.append(grad_titles[index])
            times.append(grad_object.get_times())
            waveforms.append(1e3 * grad_object.get_waveform())

    return titles, times, waveforms


def create_waveform_frames(titles: list, times: list, waveforms: list):
    fig, axes = plt.subplots(len(waveforms), 2, figsize=(12, 8))

    for i in range(len(waveforms)):
        axes[i, 0].plot(1e3 * times[i], waveforms[i], lw=0.9, color='b')
        axes[i, 0].set_title(titles[i], x=-0.1, y=0.35, rotation=0, ha='right', fontsize=11)
        axes[i, 0].axhline(0, lw=0.375, color='k')
        if i == len(waveforms) - 1:
            axes[i, 0].set_xlabel('Time (ms)', fontsize=12)
        else:
            axes[i, 0].set_xticks([])

        axes[i, 1].set_axis_off()

    return fig, axes


def create_magnetisation_frame(mxy: np.ndarray, mz: np.ndarray, x_axis: str, x_data: np.ndarray):
    ax = plt.subplot(1, 2, 2)
    if x_axis == 'df':
        x_label = 'Isochromat Frequency (Hz)'
    elif x_axis == 'pos':
        x_label = 'Position (mm)'
        x_data *= 1e3
    else:
        raise ValueError("x_axis must be 'df' or 'pos'")

    mxy_line, = ax.plot(x_data, mxy[0, :], lw=2, color='k', label='$|M_{xy}|$')
    mz_line, = ax.plot(x_data, mz[0, :], lw=2, color='r', label='$M_z$')
    ax.grid()

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylim([-1.1, 1.1])
    ax.legend()
    ax.set_title('Magnetisation')

    return mxy_line, mz_line


def create_progress_lines(num_lines: int, times: list, axes: np.ndarray[plt.Axes]) -> list:
    progress_lines = []
    for i in range(num_lines):
        progress_line = axes[i, 0].axvline(1e3 * times[i][0], lw=1, color='g')
        progress_lines.append(progress_line)

    return progress_lines


def save_animation(anim: FuncAnimation, save_path: str):
    anim.save(save_path, fps=60, codec='h264', dpi=300)


def animate(magnetisation: np.ndarray, delta_time: float, simulation_style: str,
            off_resonances: Optional[np.ndarray] = None,
            position_axis: Optional[Union[int, tuple[int]]] = None,
            positions: Optional[np.ndarray] = None,
            rf_pulse: Optional[rf.RFPulse] = None,
            grad_x: Optional[gradient.Gradient] = None,
            grad_y: Optional[gradient.Gradient] = None,
            grad_z: Optional[gradient.Gradient] = None,
            repeat: bool = False, num_frames: int = 500, save_path: str = None):
    if simulation_style == 'relaxation':
        anim = None
    elif simulation_style == 'non_selective':
        anim = non_selective_animation(magnetisation, delta_time, off_resonances, rf_pulse, repeat, num_frames)
    elif simulation_style == '1d_selective':
        anim = slice_selective_animation(magnetisation, delta_time, position_axis, positions,
                                         rf_pulse, grad_x, grad_y, grad_z, repeat, num_frames)
    elif simulation_style == '2d_selective':
        anim = None
    elif simulation_style == 'spectral_spatial':
        anim = spectral_spatial_animation(magnetisation, delta_time, off_resonances, position_axis, positions,
                                          rf_pulse, grad_x, grad_y, grad_z, repeat, num_frames)
    else:
        raise ValueError("Simulation style invalid! "
                         "Must be 'relaxation', non_selective', '1d_selective', '2d_selective' or 'spectral_spatial'")

    if save_path is not None:
        save_animation(anim, save_path)


def non_selective_animation(magnetisation: np.ndarray, delta_time: float, off_resonances: np.ndarray,
                            rf_pulse: rf.RFPulse, repeat: bool = True, num_frames: int = 500):
    titles, times, waveforms = get_waveform_data(rf_pulse, None, None, None, delta_time)

    fig, axes = create_waveform_frames(titles, times, waveforms)

    mxy = get_mxy(magnetisation)
    mz = get_mz(magnetisation)

    mxy_line, mz_line = create_magnetisation_frame(mxy, mz, 'df', off_resonances)

    progress_lines = create_progress_lines(len(waveforms), times, axes)

    total_length = magnetisation.shape[0]
    frame_indices = np.linspace(0, total_length, num_frames, endpoint=False)
    nearest_times = np.floor(frame_indices) * delta_time

    def update(frame):
        nearest_time = nearest_times[frame]
        nearest_frame = np.argmin(np.abs(nearest_time - rf_pulse.get_times(delta_time)))

        for line in progress_lines:
            line.set_xdata(1e3 * nearest_time)

        mxy_line.set_ydata(mxy[nearest_frame, :])
        mz_line.set_ydata(mz[nearest_frame, :])

        return progress_lines + [mxy_line, mz_line]

    anim = FuncAnimation(fig, update, frames=num_frames, interval=1,
                         blit=True, repeat=repeat)
    plt.show()

    return anim


def slice_selective_animation(magnetisation: np.ndarray, delta_time: float, axis: int, positions: np.ndarray,
                              rf_pulse: rf.RFPulse,
                              grad_x: Optional[gradient.Gradient], grad_y: Optional[gradient.Gradient],
                              grad_z: Optional[gradient.Gradient],
                              repeat: bool = True, num_frames: int = 500):
    titles, times, waveforms = get_waveform_data(rf_pulse, grad_x, grad_y, grad_z, delta_time)

    fig, axes = create_waveform_frames(titles, times, waveforms)

    mxy = get_mxy(magnetisation)
    mz = get_mz(magnetisation)
    slice_positions = positions[:, axis]
    mxy_line, mz_line = create_magnetisation_frame(mxy, mz, 'pos', slice_positions)

    progress_lines = create_progress_lines(len(waveforms), times, axes)

    total_length = magnetisation.shape[0]
    frame_indices = np.linspace(0, total_length, num_frames, endpoint=False)
    nearest_times = np.floor(frame_indices) * delta_time

    def update(frame):
        nearest_time = nearest_times[frame]
        nearest_frame = np.argmin(np.abs(nearest_time - rf_pulse.get_times(delta_time)))

        for line in progress_lines:
            line.set_xdata(1e3 * nearest_time)

        mxy_line.set_ydata(mxy[nearest_frame, :])
        mz_line.set_ydata(mz[nearest_frame, :])

        return progress_lines + [mxy_line, mz_line]

    anim = FuncAnimation(fig, update, frames=num_frames, interval=1,
                         blit=True, repeat=repeat)
    plt.show()

    return anim


def spectral_spatial_animation(magnetisation: np.ndarray, delta_time: float, off_resonances: np.ndarray,
                               axis: int, positions: np.ndarray,
                               rf_pulse: rf.RFPulse,
                               grad_x: Optional[gradient.Gradient], grad_y: Optional[gradient.Gradient],
                               grad_z: Optional[gradient.Gradient],
                               repeat: bool = True, num_frames: int = 500):
    titles, times, waveforms = get_waveform_data(rf_pulse, grad_x, grad_y, grad_z, delta_time)

    fig, axes = create_waveform_frames(titles, times, waveforms)

    mxy = get_mxy(magnetisation)
    mxy = mxy.reshape(magnetisation.shape[0], len(positions[:, axis]), len(off_resonances))
    mz = get_mz(magnetisation)
    mz = mz.reshape(magnetisation.shape[0], len(positions[:, axis]), len(off_resonances))

    plot_extent = (min(off_resonances), max(off_resonances),
                   1e3 * np.min(positions[:, axis]), 1e3 * np.max(positions[:, axis]))
    plt.subplot(2, 2, 2)
    mxy_distribution = plt.imshow(mxy[0, :, :], vmin=0, vmax=1, cmap='viridis',
                                  aspect='auto', extent=plot_extent)
    cbar = plt.colorbar()
    cbar.set_label('$M_{xy}$', rotation=0)
    plt.gca().set_xticklabels([])

    plt.subplot(2, 2, 4)
    mz_distribution = plt.imshow(mz[0, :, :], vmin=-1, vmax=1, cmap='viridis',
                                 aspect='auto', extent=plot_extent)
    cbar = plt.colorbar()
    cbar.set_label('$M_{z}$', rotation=0)
    plt.subplots_adjust(hspace=0.1)
    plt.xlabel('Isochromat Frequency (Hz)', fontsize=12)
    fig.text(0.5, 0.5, 'Position (mm)', va='center', rotation='vertical', fontsize=12)

    progress_lines = create_progress_lines(len(waveforms), times, axes)

    total_length = magnetisation.shape[0]
    frame_indices = np.linspace(0, total_length, num_frames, endpoint=False)
    nearest_times = np.floor(frame_indices) * delta_time

    def update(frame):
        nearest_time = nearest_times[frame]
        nearest_frame = np.argmin(np.abs(nearest_time - rf_pulse.get_times(delta_time)))

        for line in progress_lines:
            line.set_xdata(1e3 * nearest_time)

        mxy_distribution.set_data(mxy[nearest_frame, :, :])
        mz_distribution.set_data(mz[nearest_frame, :, :])

        return progress_lines + [mxy_distribution, mz_distribution]

    anim = FuncAnimation(fig, update, frames=num_frames, interval=1,
                         blit=True, repeat=repeat)

    plt.show()

    return anim
