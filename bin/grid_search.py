#!/usr/bin/python3.7
"""
Usage:
    grid_search.py (--feature_set=<fs>)

Options:
    --feature_set=<fs>    1 or 2
"""
# 1.- Read EDF Files
# 2.- Read the list of parameters to test from configuration file
# 3.- Test each parameter combination by using Cross Validation
# 4.- Ssave the results
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
