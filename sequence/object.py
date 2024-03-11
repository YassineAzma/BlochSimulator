import numpy as np


class SequenceObject:
    def __init__(self):
        self.delta_time = None
        self.times = None
        self.waveform = None
        self.amplitude = None

    def __add__(self, other_object):
        raise NotImplementedError

    def __sub__(self, other_object):
        raise NotImplementedError

    def _append(self, other_object, delay: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        new_times = generate_times(self.delta_time, self.times.max() + delay + other_object.times.max())

        length_of_delay = int(delay / 1e-6)
        padding = np.zeros(length_of_delay)
        comb_data = np.concatenate([self.get_waveform(), padding, other_object.get_waveform()[1:]])

        return new_times, comb_data

    def zero_pad(self, delay: float, is_after: bool = True):
        new_times = generate_times(self.delta_time, self.get_times().max() + delay)

        length_of_delay = int(delay / 1e-6)
        padding = np.zeros(length_of_delay)
        if is_after:
            comb_data = np.concatenate([self.get_waveform(), padding])
        else:
            comb_data = np.concatenate([padding, self.get_waveform()])

        self.times = new_times
        self.waveform = comb_data / self.amplitude



    @staticmethod
    def normalize(data: np.ndarray):
        return data / np.max(np.abs(data))

    @staticmethod
    def resample(old_dt: float, new_dt: float, data: np.ndarray) -> np.ndarray:
        max_length = len(data)
        new_length = round(max_length * old_dt / new_dt) + 1

        old_time = np.linspace(0, max_length, max_length, dtype=np.int64) * old_dt
        new_time = np.linspace(0, max_length, new_length, dtype=np.int64) * old_dt

        return np.interp(new_time, old_time, data)

    def get_times(self, new_delta_time: float = None) -> np.ndarray:
        if new_delta_time:
            return self.resample(self.delta_time, new_delta_time, self.times)

        return self.times

    def get_waveform(self, new_delta_time: float = None) -> np.ndarray:
        raise NotImplementedError

    def get_info(self):
        raise NotImplementedError

    def set_amplitude(self, amplitude: float):
        self.amplitude = amplitude

    def crop(self):
        """TO DO"""
        zero_amplitude = np.where(np.abs(self.get_waveform()) == 0)
        before = np.where(np.diff(zero_amplitude) > 1)
        print(before)


def generate_times(delta_time: float, duration: float) -> np.ndarray:
    total_steps = round(duration / delta_time)

    return np.linspace(0, total_steps, total_steps + 1, dtype=np.int64) * delta_time
