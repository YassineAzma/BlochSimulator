from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from constants import GAMMA
from sequence.rf_pulse import RFPulse


# IDEA Simulation text file import
def _extract_column(file_path, column_name) -> Optional[np.ndarray]:
    """
    Extract a specific column from a text file.

    Parameters:
    - file_path (str): Path to the text file.
    - column_name (str): Name of the column to extract.

    Returns:
    - column_data (numpy.ndarray): Extracted column data.
    """

    # Read the file and find the column names
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extract column names from the header
    header = lines[8].strip().split('\t')

    # Find the index of the desired column
    column_index = None
    for i, header_item in enumerate(header):
        if column_name in header_item:
            column_index = i
            break

    if column_index is None:
        print(f"Column '{column_name}' not found in the file.")
        return None

    # Extract column data
    data_lines = lines[10:]

    column_data = []
    for line in data_lines:
        column_data.append(float(line.strip().split('\t')[column_index]))

    return np.array(column_data)


def extract_rf_pulse(file_path: str, channel: int = 0, save: bool = False) -> np.ndarray:
    rf_envelope = _extract_column(file_path, f'RF-Signal (ch. {channel}, 1H, 123.2 MHz)')
    rf_phase = _extract_column(file_path, f'RF-Signal Phase (ch. {channel}, 1H, 123.2 MHz)')
    nco_phase = _extract_column(file_path, 'Numeric Crystal Oscillator 1 Phase')

    plt.plot(nco_phase)
    plt.show()
    b1_pulse = rf_envelope / np.max(rf_envelope) * np.exp(1j * np.deg2rad(rf_phase + nco_phase))
    # b1_pulse = nco_phase / (np.max(nco_phase) * np.exp(1j * np.deg2rad(rf_envelope + rf_phase)))
    stripped_file_path = file_path.rstrip('.txt')
    if save:
        np.save(f'{stripped_file_path}_b1.npy', b1_pulse)

    return b1_pulse


def extract_gradients(file_path: str, save: bool = False) -> np.ndarray:
    x_gradient = _extract_column(file_path, 'X Gradient (GPA 0)')
    y_gradient = _extract_column(file_path, 'Y Gradient (GPA 0)')
    z_gradient = _extract_column(file_path, 'Z Gradient (GPA 0)')

    gradients = np.array([x_gradient, y_gradient, z_gradient])

    stripped_file_path = file_path.rstrip('.txt')
    if save:
        np.save(f'{stripped_file_path}_gradient.npy', gradients)

    return gradients


def calculate_ref_grad(pulse_duration: float, bandwidth: float) -> float:
    tbp = pulse_duration * bandwidth

    return tbp / (GAMMA * 5.12e-3 * 10e-3)


def pulse_to_pta(pulse: RFPulse, family_name: str, pulse_name: str, file_name: str, ref_grad: float,
                 comment: str = None) -> None:
    waveform = pulse.get_waveform(1e-6)[1:-1]
    normalized_waveform = waveform / np.abs(waveform).max()

    amplitude_integral = np.sqrt(np.sum(normalized_waveform.real) ** 2 +
                                 np.sum(normalized_waveform.imag) ** 2)
    power_integral = np.sum(normalized_waveform.real ** 2 + normalized_waveform.imag ** 2)
    absolute_integral = np.sum(np.abs(normalized_waveform))
    with open(f'{file_name}.pta', 'w') as file:
        # Write header information
        file.write(f'PULSENAME: {family_name}.{pulse_name}\n')
        file.write(f'COMMENT: {comment}\n')
        file.write(f'REFGRAD: {1000 * ref_grad}\n')
        file.write(f'MINSLICE: 1.000000000\n')
        file.write(f'MAXSLICE: 400.000000000\n')
        file.write(f'AMPINT: {amplitude_integral}\n')
        file.write(f'POWERINT: {power_integral}\n')
        file.write(f'ABSINT: {absolute_integral}\n')
        file.write('\n')
        for index, pulse_sample in enumerate(normalized_waveform):
            magnitude = np.abs(pulse_sample)
            phase = np.angle(pulse_sample)
            file.write(f'{round(magnitude, 9)} {round(phase, 9)} \t ; ({index}) \n')
