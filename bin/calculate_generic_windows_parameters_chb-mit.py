#!/usr/bin/python3.7
"""
Usage:
    calculate_generic_windows_parameters_chb-mit.py (--feature_set=<fs>) [--patient=<p>] 

Options:
    --patient=<p>         Patient used for evaluation
    --feature_set=<fs>    1 or 2
"""
# 1.- Read EDF Files
# 2.- Keep needed channels based on channels.txt
# 3.- Scale EEG signal (original data is in Volts)
# 4.- Filter out drift (<0.5Hz)
# 5.- Generate X number of windows of Y length per each category
#    ["ictal", "noictal"]
# 6.- Get bands from EEG (delta, theta, alpha, beta)
# 7.- Calculate parameters for each window
#     feature_set_1 = (peak_freq, median_freq, variance, rms, skewness, kurtosis, Zero Crossing, Range)
#     feature_set_1 = (peak_freq, median_freq, skewness, kurtosis, SampEn, min value, mean, std
# 8.- Split 0.1 data for grid search and 0.9 for rest of analysis process
# 9.- Save data
# Note: check features_map.json for understanding features order.

import sys
import logging
from docopt import docopt
sys.path.append("../../Src")
from utils import settings
from utils.signal_processors import Events2Label, select_monopolar_channels,\
                                    scaler
from utils.customized_utils import pipeline_filter, bank_filter,\
                                   calculate_full_features_1,\
                                   calculate_full_features_2
from utils.file_managers import get_json_content, get_text_content,\
                                get_edf_content, write_pickle
from utils.windows import get_windowsdata, no_random_selection


OPTS = docopt(__doc__)
settings = settings["chb-mit"]
base_path = settings["database"]
output_path = settings["windows"].format(OPTS["--feature_set"])
metadata_file = settings["metadata_file_json"]
channels_file = settings["channels"]
part_for_grid = settings["part_for_grid"]


filt_pipeline = [{"f": 60, "fs": 256, "type": "notch"},                       # Filter 60Hz
                 {"f": 120, "order": 2, "fs": 256, "type": "lowpass"},        # Not needed
                 {"f": 0.5, "order": 2, "fs": 256, "type": "highpass"}]       # Filter low freq drifts
filt_bank = [{"f": (0.5, 30), "order": 2, "fs": 256, "type": "bandpass"},     # Full signal
             {"f": (0.5, 4), "order": 2, "fs": 256, "type": "bandpass"},      # Delta band
             {"f": (4, 8), "order": 2, "fs": 256, "type": "bandpass"},        # Theta band
             {"f": (8, 12), "order": 2, "fs": 256, "type": "bandpass"},       # Alpha band
             {"f": (12, 25), "order": 2, "fs": 256, "type": "bandpass"}]      # Beta band


meta = get_json_content(metadata_file)
channels = get_text_content(channels_file)
if OPTS["--feature_set"] == "1":
    calculate_full_features = calculate_full_features_1
else:
    calculate_full_features = calculate_full_features_2


for patient, data in meta.items():
    for file, details in list(data["files"].items()):
        if OPTS["--patient"] and OPTS["--patient"] not in file:
            continue
        try:
            eeg = get_edf_content(base_path + file)
            eeg = select_monopolar_channels(channels, details["channels"], eeg)
            eeg = scaler(settings["gain"]/settings["units"], eeg)
            eeg = pipeline_filter(filt_pipeline, eeg)
            e2l = Events2Label(eeg.shape[1], f_samp=data["fs"],
                               events=details["seizures"])
            e2l.get_label_array()
            labels = e2l.label_array
            windows_noictal, windows_ictal = get_windowsdata(eeg, labels,
                                                             settings["length"],
                                                             settings["ictal_sample_step"])

            if len(windows_noictal) > len(windows_ictal)*10:
                windows_noictal = no_random_selection(windows_noictal,
                                                      len(windows_ictal)*10)
            logging.info("File=%s, Ictal window=%s, No Ictal windows=%s", file,
                         len(windows_ictal), len(windows_noictal))

            windows_noictal = [bank_filter(filt_bank, s)
                               for s in windows_noictal]
            windows_ictal = [bank_filter(filt_bank, s)
                             for s in windows_ictal]

            windows_noictal = [[calculate_full_features(band, data["fs"])
                               for band in window]
                               for window in windows_noictal]
            windows_ictal = [[calculate_full_features(band, data["fs"])
                             for band in window]
                             for window in windows_ictal]
            windows_noictal = [sum(window, []) for window in windows_noictal]
            windows_ictal = [sum(window, []) for window in windows_ictal]

            limit_idx_ictal = int(len(windows_ictal)*settings["part_for_grid"])
            limit_idx_noictal = int(len(windows_noictal)*settings["part_for_grid"])
            logging.info("File=%s, Ictal window=%s, No Ictal windows=%s", file,
                         len(windows_ictal), len(windows_noictal))

            # Keep X number of windows apart for grid search
            windows_noictal_g = windows_noictal[:limit_idx_noictal]
            windows_ictal_g = windows_ictal[:limit_idx_ictal]
            windows_noictal_c = windows_noictal[limit_idx_noictal:]
            windows_ictal_c = windows_ictal[limit_idx_ictal:]

            write_pickle(output_path + "GridSearch/NoIctal/" + file.replace("edf",
                                                                            "pickle"),
                         windows_noictal_g)
            write_pickle(output_path + "GridSearch/Ictal/" + file.replace("edf",
                                                                          "pickle"),
                         windows_ictal_g)
            write_pickle(output_path + "Common/NoIctal/" + file.replace("edf",
                                                                        "pickle"),
                         windows_noictal_c)
            write_pickle(output_path + "Common/Ictal/" + file.replace("edf",
                                                                      "pickle"),
                         windows_ictal_c)
            print("Processed file " + file)
        except Exception as e:
            logging.error(e)
