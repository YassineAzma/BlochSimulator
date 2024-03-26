import numpy as np
from matplotlib import pyplot as plt


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
    mxy_abs = np.abs(magnetisation[-1, :, 0] + 1j * magnetisation[-1, :, 1])
    mxy_angle = np.angle(magnetisation[-1, :, 0] + 1j * magnetisation[-1, :, 1])
    end_mz = magnetisation[-1, :, 2]

    fig, ax = plt.subplots()

    ax.plot(off_resonances, mxy_abs, label='$M_{xy}$', color='black')
    ax.plot(off_resonances, end_mz, label='$M_{z}$', color='red')
    ax.set_ylabel('Magnetisation')
    ax.set_ylim([-0.05, 1.05])

    ax2 = ax.twinx()
    ax2.plot(off_resonances, mxy_angle, label='$∠M_{xy}$', color='black', ls='--', lw=0.5)
    ax2.set_ylabel('Phase (rad)')
    ax2.set_ylim([-3 * np.pi / 2, 3 * np.pi / 2])

    plt.xlabel('Isochromat Frequency (Hz)')
    plt.title('Frequency Profile')
    ax.legend()
    ax2.legend()
    plt.grid()

    fig.tight_layout()
    plt.show()
