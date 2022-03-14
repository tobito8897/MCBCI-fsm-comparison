#!/usr/bin/python3.7
"""
Usage:
    ml_analysis.py (--db=<db>) (--feature_set=<fs>) (--model=<ex>)

Options:
    --feature_set=<fs>    1 or 2
    --db=<db>             siena or chb-mit
    --model=<ex>          tree, svm, knn, forest or ann
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
from docopt import docopt
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, ".."))
from utils.ml_pipeline import k_folds_stratified_nn, get_es_callback
from utils.customized_utils import prepare_data_for_ml
from utils.file_managers import get_json_content
from utils.explainers import FeaturesImportance
from utils.file_managers import write_pickle, read_pickle
from utils.parameters import *
from utils import settings


OPTS = docopt(__doc__)
settings = settings[OPTS["--db"]]
model = OPTS["--model"]
explainers = ["tree", "svm", "lime", "shap", "embeddedrandomforest",
              "reciprocalranking"]
input_path = os.path.join(current_dir, settings["windows"].format(OPTS["--feature_set"]))
features_map_file = os.path.join(current_dir, settings["features_map_file"].format(OPTS["--feature_set"]))
top_directory = os.path.join(current_dir, settings["top"].format(OPTS["--feature_set"]))
stats_directory = os.path.join(current_dir, settings["stats"].format(OPTS["--feature_set"]))
ictal_path = input_path + "/Common/Ictal/"
noictal_path = input_path + "/Common/NoIctal/"

train_kwargs["ann"]["callbacks"] = get_es_callback()
features_map = get_json_content(features_map_file)
data, labels = prepare_data_for_ml(ictal_path, noictal_path,
                                   remove_indexes=settings["blacklisted_features"][OPTS["--feature_set"]])
data[data == np.inf] = 0
data[data == -np.inf] = 0
data = pd.DataFrame(data)
data = data.fillna(0)
data = data.to_numpy()

up_limit = data.shape[1] - data.shape[1] % 50
top_features = [data.shape[1]] + list(range(up_limit, 0, -50)) + [25, 12, 6, 1]
print(data.shape)
print(top_features)
if OPTS["--feature_set"] == 1:
    start_kwargs = start_kwargs_fs1
else:
    start_kwargs = start_kwargs_fs2

for explainer in explainers:
    top_filename = top_directory + "/top_%s_%s.pickle" % (model, explainer)
    FImp = FeaturesImportance(data, labels, 200,
                              top_directory, features_map,
                              settings["blacklisted_features"][OPTS["--feature_set"]])
    importances = FImp(explainer, model, top_features[0], start_kwargs[model],
                       train_kwargs[model])

    write_pickle(top_filename, importances)

    importances = read_pickle(top_filename)
    for top in top_features:
        k_folds_stratified_nn(model, repetitions, data, labels,
                              stats_directory, features_map, top,
                              explainer, importances, start_kwargs[model],
                              train_kwargs[model],
                              settings["blacklisted_features"][OPTS["--feature_set"]])
        logging.info("Iteration completed, explainer%s=, number of features=%s",
                     explainer, top)
