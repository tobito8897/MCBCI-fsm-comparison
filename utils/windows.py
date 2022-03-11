#!/usr/bin/python3.7

import numpy as np


def get_windowsdata(data: np.array, labels: np.array, windows_len: int,
                    ictal_overlap: int) -> np.array:
    global_ictal_windows = []
    global_noictal_windows = []
    current_index = 0

    while(current_index < (len(labels) - (windows_len+1))):
        windows_mean = np.mean(labels[current_index: current_index+windows_len])

        if windows_mean == 0:
            global_noictal_windows.append(data[:, current_index: current_index + windows_len])
            current_index += windows_len
        elif windows_mean == 1:
            global_ictal_windows.append(data[:, current_index: current_index + windows_len])
            current_index += ictal_overlap
        else:
            current_index += 50

    return global_noictal_windows, global_ictal_windows


def no_random_selection(windows: list, selected: int) -> np.array:
    return windows[:selected]


def features_selection(features_map: dict, band: str, channels: list,
                       parameter: str) -> list:
    if band:
        bands = [features_map["bands"].index(band)]
    else:
        bands = list(range(len(features_map["bands"])))

    if parameter:
        parameters = [features_map["features"].index(parameter)]
    else:
        parameters = list(range(len(features_map["features"])))

    if channels:
        channels = [features_map["channels"].index(c) for c in channels]
    else:
        channels = list(range(len(features_map["channels"])))

    len_band = len(features_map["channels"])*len(features_map["features"])
    len_channel = len(features_map["features"])

    indices_to_keep = []
    for band in bands:
        for channel in channels:
            for parameter in parameters:
                indices_to_keep.append(len_band*band + len_channel*channel + parameter)

    return indices_to_keep


def patient_selection(file_names: list, patients: list):
    if not patients:
        return file_names

    return [f for f in file_names
            for patient in patients if patient in f]


def list_of_list_to_array(data: list) -> np.array:
    new_array = np.array([np.array(x) for x in data])
    return new_array


def get_index_features_map(features_map: dict, importances: list,
                           features_to_keep: int, invert: bool = False,
                           blacklist: list = []) -> list:
    index_map = {}
    real_counter, counter = 0, 0
    for band in features_map["bands"]:
        for channel in features_map["channels"]:
            for feature in features_map["features"]:
                if real_counter in blacklist:
                    real_counter += 1
                    continue
                index_map[counter] = "{}_{}_{}".format(band, channel, feature)
                counter += 1
                real_counter += 1

    importances = ((a, index_map[a], b) for a, b in enumerate(importances))

    if invert:
        importances = sorted(importances, key=lambda x: x[2], reverse=False)
    else:
        importances = sorted(importances, key=lambda x: x[2], reverse=True)

    if features_to_keep:
        indices = [i[0] for i in importances[:features_to_keep]]
        names = [i[1] for i in importances[:features_to_keep]]
        values = [i[2] for i in importances[:features_to_keep]]
    else:
        indices = [i[0] for i in importances]
        names = [i[1] for i in importances]
        values = [i[2] for i in importances]

    return indices, names, values
