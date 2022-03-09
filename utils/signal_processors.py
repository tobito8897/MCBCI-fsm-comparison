#!/usr/bin/python3.7
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats as stats
from pyeeg.entropy import samp_entropy
from pyeeg.hjorth_mobility_complexity import hjorth


class Events2Label():

    def __init__(self, data_len: tuple, f_samp: int, events: list):
        self.data_len = data_len
        self.f_samp = f_samp
        self.events = events
        self.label_array = np.zeros(self.data_len)

    def get_label_array(self):
        """
        Generate array from seizure events list.
        0 = Background
        1 = Seizure
        Array has same shape as eeg array
        """
        for event in self.events:
            self.add_event(event[0], event[1])

    def add_event(self, start: int, stop: int):
        """
        Add ones to seizure period
        """
        start_from = round(int(start)*self.f_samp)
        stop_till = round(int(stop)*self.f_samp)
        self.label_array[start_from: stop_till] = 1


def select_monopolar_channels(needed_channels: list, current_channels: dict,
                              data: np.array) -> np.array:
    output_data = np.zeros((len(needed_channels), data.shape[1]))
    for index, channel_name in current_channels.items():
        if channel_name not in needed_channels:
            continue
        new_index = needed_channels.index(channel_name)
        output_data[new_index, :] = data[(int(index)-1), :]

    return output_data


def convert_to_bipolar(bipolar_montage: list, current_channels: dict,
                       data: np.array) -> np.array:
    current_channels = {v.lower(): k for k, v in current_channels.items()}
    output_data = np.zeros((len(bipolar_montage), data.shape[1]))
    for index, (ch1, ch2) in enumerate(bipolar_montage):
        ch1 = int(current_channels[ch1.lower()])
        ch2 = int(current_channels[ch2.lower()])
        output_data[index, :] = data[ch1, :] - data[ch2, :]
    return output_data


def filter_butterworth(f: int, order: int, btype: str, fs: int,
                       data: np.array) -> np.array:
    filtered_data = []
    b, a = scipy.signal.butter(order, f, btype=btype, fs=fs)

    for index in range(data.shape[0]):
        _data = scipy.signal.lfilter(b, a, data[index, :])
        _data = _data.reshape((1, len(_data)))
        filtered_data.append(_data)

    return np.concatenate(filtered_data)


def filter_notch(f: int, fs: int, data: np.array) -> np.array:
    filtered_data = []
    b, a = scipy.signal.iirnotch(f, 30, fs=fs)
    for index in range(int(data.shape[0])):
        _data = scipy.signal.lfilter(b, a, data[index, :])
        _data = _data.reshape((1, len(_data)))
        filtered_data.append(_data)

    return np.concatenate(filtered_data)


def scaler(scale: int, data: np.array) -> np.array:
    return scale*data


def peak_frequency(f_array: np.array, p_array: np.array) -> float:
    index = np.argmax(p_array)
    return f_array[index]


def median_frequency(f_array: np.array, p_array: np.array) -> float:
    p_sum = 0
    for p in p_array:
        p_sum += p

    tmp_p_sum = 0
    for f, p in zip(f_array, p_array):
        tmp_p_sum += p
        if tmp_p_sum > p_sum/2:
            return f


def variance(data: np.array) -> float:
    return np.var(data)


def rms(data: np.array) -> float:
    return np.sqrt(np.mean(data**2))


def skewness(data: np.array) -> float:
    return stats.skew(data)


def kurtosis(data: np.array) -> float:
    return stats.kurtosis(data)


def zerocrossing(data: np.array) -> int:
    crosses = np.where(np.diff(np.sign(data)))[0]
    return len(crosses)


def sampleentropy(data: np.array) -> int:
    entropy = samp_entropy(data, 2, 0.2*np.std(data))
    return entropy


def range_val(data: np.array) -> int:
    _range = np.max(data) - np.min(data)
    return _range


def mean(data: np.array) -> int:
    _mean = np.mean(data) - np.min(data)
    return _mean


def sdeviation(data: np.array) -> int:
    _sdeviation = np.std(data)
    return _sdeviation


def hjorth_params(data: np.array) -> int:
    complexity, mobility = hjorth(data)
    return complexity, mobility


def interquartile_range(data: np.array) -> int:
    Q1 = np.percentile(data, 25, interpolation="midpoint")
    Q3 = np.percentile(data, 75, interpolation="midpoint")
    return Q3 - Q1


def absolute_median_deviation(data: np.array) -> int:
    amd = stats.median_absolute_deviation(data)
    return amd


def min_val(data: np.array) -> int:
    return np.min(data)


def jaccard_similarity(set_1: set, set_2: set) -> set:
    s1 = set(set_1)
    s2 = set(set_2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))
