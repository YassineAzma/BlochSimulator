import numpy as np


def get_full_width_arbitrary(off_resonance: np.ndarray, magnetisation: np.ndarray,
                             threshold: float, profile: str = 'excitation') -> float:
    if profile == 'excitation':
        excitation_profile = np.abs(magnetisation[-1, :, 0] + 1j * magnetisation[-1, :, 1])
    elif profile == 'inversion':
        excitation_profile = -magnetisation[-1, :, 2]
    else:
        raise ValueError("Profile must be 'excitation' or 'inversion'")

    indices = np.where(excitation_profile > threshold)[0]
    if len(indices) == 0:
        return 0
    else:
        return off_resonance[indices[-1]].item() - off_resonance[indices[0]].item()


def get_full_width_half_maximum(off_resonance: np.ndarray, magnetisation: np.ndarray) -> float:
    return get_full_width_arbitrary(off_resonance, magnetisation, 0.5)


def profile_analysis(df: np.ndarray, magnetisation: np.ndarray, passband_ripple: float, stopband_ripple: float):
    mxy = np.abs(magnetisation[-1, :, 0] + 1j * magnetisation[-1, :, 1])
    mxy_angle = np.angle(magnetisation[-1, :, 0] + 1j * magnetisation[-1, :, 1])

    def passband() -> tuple[tuple[float, float], float, float]:
        passband = np.where(mxy > (1 - passband_ripple))[0]
        if len(passband) == 0:
            return (0, 0), 0, 0

        passband_phase_gradient = np.mean(np.gradient(np.unwrap(mxy_angle[passband])))
        passband_frequencies = (df[passband[0]].item(), df[passband[-1]].item())
        passband_diff = np.diff(passband)
        rippled_points = np.where(passband_diff > 1, passband_diff, 0).sum() * (df[1] - df[0])

        return passband_frequencies, rippled_points, passband_phase_gradient

    def transition_band(passband_end: float):
        passband_end_index = np.where(df == passband_end)[0]
        if len(passband_end_index) == 0:
            return np.inf, np.inf
        else:
            passband_end_index = passband_end_index[0]

        transition_region = mxy[passband_end_index:]
        transition_begin = np.where(transition_region < 0.95)[0]
        if len(transition_begin) == 0:
            return np.inf, np.inf

        gradient = np.gradient(transition_region)
        transition_end = np.where((gradient < 0) & (transition_region < 0.05))[0]

        if len(transition_end) == 0:
            return (np.inf, np.inf)

        transition_frequencies = (df[passband_end_index - 1 + transition_begin[0]].item(),
                                  df[passband_end_index - 1 + transition_end[0]].item())

        return transition_frequencies

    def stopband(transition_end: float):
        if transition_end == np.inf:
            return (0, 0), 0

        transition_end_index = np.where(df == transition_end)[0][0]
        stopband = mxy[transition_end_index:]
        if len(stopband) == 0:
            return (0, 0), 0

        stopband_frequencies = (df[transition_end_index - 1].item(),
                                df[transition_end_index - 1 + len(stopband)].item())

        rippled_points = np.where(stopband > stopband_ripple, 1, 0).sum() * (df[1] - df[0])

        return stopband_frequencies, rippled_points

    passband_frequencies, pass_ripples, passband_phase_gradient = passband()
    transition_frequencies = transition_band(passband_frequencies[1])
    stopband_frequencies, stop_ripples = stopband(transition_frequencies[1])

    fwhm = get_full_width_half_maximum(df, magnetisation)
    profile_dict = {
        'FWHM': fwhm,
        'Passband': passband_frequencies,
        'Passband Phase Gradient': passband_phase_gradient,
        'Passband Ripples': pass_ripples,
        'Transition Band': transition_frequencies,
        'Stopband': stopband_frequencies,
        'Stopband Ripples': stop_ripples
    }

    return profile_dict
