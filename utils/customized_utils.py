#!/usr/bin/python3.7
__author__ = "Sergio E Sanchez Hdez"

import re
import pickle
from glob import glob
from collections import defaultdict
import numpy as np
import scipy.signal
from .signal_processors import Events2Label, filter_butterworth, filter_notch,\
                               peak_frequency, median_frequency,\
                               variance, rms, skewness, kurtosis,\
                               zerocrossing, sampleentropy, mean, \
                               min_val, range_val, sdeviation
from .basic_plotters import eeg_plot_plus_label, eeg_psd_plot,\
                            get_top_plot, get_feature_elimination_plot,\
                            plot_confusion_matrix, get_rfe_comparison_plot,\
                            get_scatter_plot_acc, get_scatter_plot_friedman,\
                            plot_jaccard_matrix, plot_jaccard_matrix_chb_vs_siena
from .windows import list_of_list_to_array,\
                     get_index_features_map
from .file_managers import list_dir, read_pickles, read_pickle
from .ml_pipeline import generate_labels
from .stats import test_5x2_ftest
from .parameters import methods_short_name
from sklearn.metrics import accuracy_score


def fast_eeg_plot_plus_label(file_name: str, period: tuple,
                             interest_channels: list, eeg: np.array,
                             meta: np.array, save):
    """Plot eeg recording plus label (0: background, 1:seizures)"""
    try:
        patient = file_name.split("/")[-2]
        fs = meta[patient]["fs"]
    except Exception:
        patient = file_name.split("/")[-1].split("_")[0]
        fs = meta[patient]["fs"]
    file_name = file_name.split("/")[-1]
    file_detls = meta[patient]["files"][file_name]

    channel_names = [file_detls["channels"][c]
                     for c in interest_channels]
    title = "Registro EEG, paciente: {}".format(patient)
    e2l = Events2Label(eeg.shape[1], f_samp=fs, events=file_detls["seizures"])
    e2l.get_label_array()

    eeg_plot_plus_label(eeg, e2l.label_array, channels=interest_channels,
                        channel_names=channel_names, title=title,
                        period=period, fs=fs, save=save)


def fast_eeg_plot_psd(file_name: str, period: tuple, interest_channels: list,
                      eeg: np.array, meta: np.array, save):
    """Plots Power Spectrum Density of EEG recording"""
    try:
        patient = file_name.split("/")[-2]
        fs = meta[patient]["fs"]
    except Exception:
        patient = file_name.split("/")[-1].split("_")[0]
        fs = meta[patient]["fs"]
    file_name = file_name.split("/")[-1]
    fs = meta[patient]["fs"]
    file_detls = meta[patient]["files"][file_name]

    channel_names = [file_detls["channels"][c]
                     for c in interest_channels]
    title = "Densidad Espectral de Potencia, paciente: {}".format(patient)

    eeg_psd_plot(eeg, channels=interest_channels, channel_names=channel_names,
                 title=title, period=period, fs=fs, save=save)


def pipeline_filter(pipeline, data: np.array):
    for filter in pipeline:
        if filter["type"] == "notch":
            data = filter_notch(filter["f"], filter["fs"], data)
        else:
            data = filter_butterworth(filter["f"], filter["order"],
                                      filter["type"], filter["fs"],
                                      data)

    return data


def bank_filter(bank, data: np.array) -> list:
    filtered_signals = []
    for filter in bank:
        _data = filter_butterworth(filter["f"], filter["order"],
                                   filter["type"], filter["fs"],
                                   data)
        filtered_signals.append(_data)
    return filtered_signals


def calculate_full_features_1(data: np.array, fs: int) -> np.array:
    features = []
    for index in range(data.shape[0]):
        fxx, psd = scipy.signal.periodogram(data[index],
                                            fs=fs)
        peak_freq = peak_frequency(fxx, psd)
        med_freq = median_frequency(fxx, psd)
        var_value = variance(data[index])
        rms_value = rms(data[index])
        skew_value = skewness(data[index])
        kurt_value = kurtosis(data[index])
        zero_value = zerocrossing(data[index])
        range_value = range_val(data[index])

        features.extend([peak_freq, med_freq, var_value,
                         rms_value, skew_value, kurt_value,
                         zero_value, range_value])

    return features


def calculate_full_features_2(data: np.array, fs: int) -> np.array:
    features = []
    for index in range(data.shape[0]):
        fxx, psd = scipy.signal.periodogram(data[index],
                                            fs=fs)
        peak_freq = peak_frequency(fxx, psd)
        med_freq = median_frequency(fxx, psd)
        sampen_value = sampleentropy(data[index])
        mean_value = mean(data[index])
        skew_value = skewness(data[index])
        kurt_value = kurtosis(data[index])
        min_value = min_val(data[index])
        std_value = sdeviation(data[index])

        features.extend([peak_freq, med_freq, sampen_value,
                         mean_value, skew_value, kurt_value,
                         min_value, std_value])

    return features


def retrieve_index_from_pickle(file_name: str, indices: list):
    selected_data = []
    with open(file_name, 'rb') as f:
        windows = pickle.load(f)

    for window in windows:
        for index in indices:
            selected_data.append(window[index])

    return selected_data


def retrieve_index_from_pickles(file_names: list, indices: list):
    selected_data = []
    for file in file_names:
        selected_data.extend(retrieve_index_from_pickle(file, indices))

    return selected_data


def prepare_data_for_ml(ictal_directory: str,
                        noictal_directory: str) -> tuple:
    ictal_files = list_dir(ictal_directory)
    noictal_files = list_dir(noictal_directory)

    ictal_data = read_pickles(ictal_files)
    noictal_data = read_pickles(noictal_files)

    ictal_data = list_of_list_to_array(ictal_data)
    noictal_data = list_of_list_to_array(noictal_data)

    data = np.concatenate((ictal_data, noictal_data))
    labels = generate_labels(len(ictal_data), len(noictal_data))

    return data, labels


def get_representative_slices(directory: str, classifiers: list,
                              explainers: list, start: int, end: int,
                              num_slices: int):
    """
    Get slices having a representative change in the accuracy mean value,
    accuracy of using 1 single feature is not considered
    """
    directory = directory + "/Stats/"
    filenames = list_dir(directory)
    global_accuracies = []

    for classifier in classifiers:
        for explainer in explainers:
            local_accuracies = []
            _filenames = [f for f in filenames
                          if classifier + "_" + explainer in f]
            num_features = [int(f.split("/")[-1].split(".")[0].split("_")[3])
                            for f in _filenames]

            for file in _filenames:
                predictions = read_pickle(file)
                acc = accuracy_score(np.concatenate(predictions["real"]),
                                     np.around(np.concatenate(predictions["predicted"])))
                local_accuracies.append(acc)

            num_features, local_accuracies = zip(*sorted(zip(num_features,
                                                             local_accuracies),
                                                 reverse=True))
            crop_num_features, crop_local_accuracies = [], []
            for x, y in zip(num_features, local_accuracies):
                if x <= start and x >= end:
                    crop_num_features.append(x)
                    crop_local_accuracies.append(y)

            global_accuracies.append(crop_local_accuracies)

    global_accuracies = np.array(global_accuracies)
    acc_mean = np.mean(global_accuracies, axis=0)
    acc_boundaries = np.linspace(acc_mean[0], acc_mean[-1], num_slices)
    print("Mean accuracy\n", acc_mean)
    print("Num Features\n", crop_num_features)
    print("Slices Accuracy Boundaries\n", acc_boundaries)

    num_feat_boundaries = []
    for acc_b in acc_boundaries:
        min_diff = 1
        nearest_idx = 0
        for idx, acc_m in enumerate(acc_mean):
            if min_diff >= abs(acc_b-acc_m):
                min_diff = abs(acc_b-acc_m)
                nearest_idx = idx
        num_feat_boundaries.append(crop_num_features[nearest_idx])
    print("Slices Num Features Boundaries\n", num_feat_boundaries)


def get_top_chart(directory: str, features_map: dict, features_to_plot: int,
                  classifier: str, explainer: str, save: str = None):
    invert = False
    if explainer == "reciprocalranking":
        invert = True

    top_file = directory + "/TopFeatures/top_%s_%s.pickle" % (classifier,
                                                              explainer)
    p = read_pickle(top_file)
    _, names, values = get_index_features_map(features_map, p,
                                              features_to_plot,
                                              invert=invert)
    print(names)
    get_top_plot(names, values, save)


def get_feature_elimination_chart(directory: str, classifier: str,
                                  explainer: str, save: str = None):

    directory = directory + "/Stats/"
    filenames = [f for f in list_dir(directory)
                 if classifier + "_" + explainer in f]
    content_list = [read_pickle(f) for f in filenames if explainer in f]
    real = [c["real"] for c in content_list]
    predicted = [[np.around(x) for x in c["predicted"]] for c in content_list]
    used_features = [int(f.split("/")[-1].split(".")[0].split("_")[3])
                     for f in filenames]

    get_feature_elimination_plot(used_features, real, predicted, save)


def get_rfe_comparison_chart(directory: str, classifiers: list,
                             explainers: list, save: str = None):
    directory = directory + "/Stats/"
    global_data = {}

    for classifier in classifiers:
        data = []

        for explainer in explainers:
            filenames = [f for f in list_dir(directory)
                         if classifier + "_" + explainer in f]
            content_list = [read_pickle(f) for f in filenames if explainer in f]
            real = [np.concatenate(c["real"]) for c in content_list]
            predicted = [np.around(np.concatenate(c["predicted"])) for c in content_list]
            used_features = [int(f.split("/")[-1].split(".")[0].split("_")[3])
                             for f in filenames]
            data.append({"real": real, "predicted": predicted,
                         "used_features": used_features,
                         "explainer": methods_short_name[explainer]})
        global_data[classifier] = data
    get_rfe_comparison_plot(global_data, save)


def confusion_matrix(directory: str, classes: list, top_features: int,
                     classifier: str, explainer: str, save: str = None):
    stats_file = directory + "/Stats/stats_%s_%s_%s.pickle" % (classifier,
                                                               explainer,
                                                               top_features)

    p = read_pickle(stats_file)
    real = np.concatenate(p["real"])
    predicted = np.concatenate(p["predicted"])

    plot_confusion_matrix(real, np.around(predicted),
                          classes, top_features, save=save)


def get_accuracy_distribution(directory: str, features_slices: list,
                              imp_model_first: bool = True, save: str = None):
    label_slices = {}
    for slice in features_slices:
        files = []
        labels = []
        models = set()
        explainers = set()

        pattern = directory + f"/Stats/stats_*_{slice}.pickle"
        files.extend(glob(pattern))

        for file in files:
            _file = read_pickle(file)
            name = file.split("_")[1:3]
            name[0] = methods_short_name[name[0]]
            name[1] = methods_short_name[name[1]]
            models.add(name[0])
            explainers.add(name[1])
            _file["name"] = name
            _file["predicted"] = [np.around(x) for x in _file["predicted"]]
            labels.append(_file)
        label_slices[slice] = labels

    get_scatter_plot_acc(label_slices, list(models), list(explainers),
                         imp_model_first, save)


def get_friedman_chart(directory: str, features_slices: list,
                       save: str = None):

    label_slices = {}
    for slice in features_slices:
        files = []
        labels = defaultdict(list)

        pattern = directory + f"/Stats/stats_*_{slice}.pickle"
        files = sorted(glob(pattern))

        for file in files:
            _file = read_pickle(file)
            name = file.split("_")[1:3]
            _file["predicted"] = [np.around(x) for x in _file["predicted"]]
            name[1] = methods_short_name[name[1]]
            labels[name[1]].append(_file)
        label_slices[slice] = dict(labels)

    get_scatter_plot_friedman(label_slices, save)


def get_jaccard_matrix(directory: str, features_map: dict,
                       features_slices: list, classifier: str,
                       explainers: list, save: str = None):
    sets_slices = {}
    for slice in features_slices:
        ranking_sets = {}

        for e in explainers:
            pattern = directory + f"/TopFeatures/top_{classifier}_{e}.pickle"
            file = glob(pattern)[0]
            importances = read_pickle(file)
            indices, names, ranking = get_index_features_map(features_map,
                                                             importances,
                                                             slice)
            e = methods_short_name[e]
            ranking_sets[e] = names
        sets_slices[slice] = dict(ranking_sets)

    plot_jaccard_matrix(sets_slices, save)


def get_jaccard_matrix_chb_vs_siena(chb_dir: str, siena_dir: str,
                                    features_map: dict, features_slices: list,
                                    classifier: str, explainers: list,
                                    save: str = None):
    chb_slices = {}
    siena_slices = {}

    for slice in features_slices:
        ranking_sets = {}
        for e in explainers:
            pattern = chb_dir + f"/TopFeatures/top_{classifier}_{e}.pickle"
            file = glob(pattern)[0]
            importances = read_pickle(file)
            _, names, _ = get_index_features_map(features_map,
                                                 importances,
                                                 slice)
            e = methods_short_name[e]
            ranking_sets[e] = names
        chb_slices[slice] = dict(ranking_sets)

    for slice in features_slices:
        ranking_sets = {}
        for e in explainers:
            pattern = siena_dir + f"/TopFeatures/top_{classifier}_{e}.pickle"
            file = glob(pattern)[0]
            importances = read_pickle(file)
            _, names, _ = get_index_features_map(features_map,
                                                 importances,
                                                 slice)
            e = methods_short_name[e]
            ranking_sets[e] = names
        siena_slices[slice] = dict(ranking_sets)

    plot_jaccard_matrix_chb_vs_siena(chb_slices, siena_slices, save)


def execute_test_5x2_ftest(directory: str, classification_method: str,
                           fsm1: str, fsm2: str, num_features: str):
    directory += "/Stats/"
    filenames = list_dir(directory)
    pattern1 = classification_method + "_" + fsm1 + "_" + num_features
    pattern2 = classification_method + "_" + fsm2 + "_" + num_features

    fsm1 = [x for x in filenames if pattern1 in x][0]
    fsm2 = [x for x in filenames if pattern2 in x][0]

    fsm1 = read_pickle(fsm1)
    fsm2 = read_pickle(fsm2)

    test_5x2_ftest(fsm1["real"], [np.around(x) for x in fsm1["predicted"]],
                   fsm2["real"], [np.around(x) for x in fsm2["predicted"]])
