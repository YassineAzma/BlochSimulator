from typing import Optional

import numpy as np


# IDEA Simulation text file import
def extract_column(file_path, column_name) -> Optional[np.ndarray]:
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
    rf_envelope = extract_column(file_path, f'RF-Signal (ch. {channel}, 1H, 123.2 MHz)')
    rf_phase = extract_column(file_path, f'RF-Signal Phase (ch. {channel}, 1H, 123.2 MHz)')
    nco_phase = extract_column(file_path, 'Numeric Crystal Oscillator 1 Phase')

    b1_pulse = rf_envelope / np.max(rf_envelope) * np.exp(1j * np.deg2rad(rf_phase + nco_phase))

    stripped_file_path = file_path.rstrip('.txt')
    if save:
        np.save(f'{stripped_file_path}_b1.npy', b1_pulse)

    return b1_pulse


def extract_gradients(file_path: str, save: bool = False) -> np.ndarray:
    x_gradient = extract_column(file_path, 'X Gradient (GPA 0)')
    y_gradient = extract_column(file_path, 'Y Gradient (GPA 0)')
    z_gradient = extract_column(file_path, 'Z Gradient (GPA 0)')

    gradients = np.array([x_gradient, y_gradient, z_gradient])

    stripped_file_path = file_path.rstrip('.txt')
    if save:
        np.save(f'{stripped_file_path}_gradient.npy', gradients)

    return gradients
