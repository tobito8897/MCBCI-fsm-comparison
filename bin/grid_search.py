#!/usr/bin/python3.7
"""
Usage:
    grid_search.py (--feature_set=<fs>)

Options:
    --feature_set=<fs>    1 or 2
"""
# 1.- Read EDF Files
# 2.- Keep needed channels based on channels.txt
# 3.- Scale EEG signal (original data is in Volts)
# 4.- Filter out drift (<0.5Hz)
# 5.- Generate X number of windows of Y length per each category
#     ["ictal", "noictal"]
# 6.- Get bands from EEG (delta, theta, alpha, beta)
# 7.- Calculate parameters for each window
#     feature_set_1 = (peak_freq, median_freq, variance, rms, skewness, kurtosis, Zero Crossing, Range)
#     feature_set_1 = (peak_freq, median_freq, skewness, kurtosis, SampEn, min value, mean, std)
# 8.- Split 0.1 data for grid search and 0.9 for rest of analysis process
# 9.- Save data
# Note: check features_map.json for understanding features order.

import sys
import logging
import numpy as np
from docopt import docopt
sys.path.append("../../Src")
from utils import settings
from utils.ml_pipeline import grid_search
from utils.customized_utils import prepare_data_for_ml
from utils.parameters import *


OPTS = docopt(__doc__)
settings = settings["chb-mit"]
input_path = settings["windows"].format(OPTS["--feature_set"])
ictal_path = input_path + "/GridSearch/Ictal/"
noictal_path = input_path + "/GridSearch/NoIctal/"
features_map_file = settings["features_map_file"]
output_path = settings["grid_search"].format(OPTS["--feature_set"])


data, labels = prepare_data_for_ml(ictal_path, noictal_path)

for model, args in list(grid_params.items()) + ["ann"]:
    logging.info("Shape of input data: %s", data.shape)
    grid_search(model, 3, data, labels, output_path, args)
    logging.info("Execution completed for model: %s", model)
