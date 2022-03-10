#!/usr/bin/python3.7
"""
Usage:
    correlation_analysis.py (--db=<db>) (--feature_set=<fs>)

Options:
    --feature_set=<fs>    1 or 2
"""
# 1.- Read EDF processed files
# 2.- Calculate the Pearson correlation coefficient for each pair of
#     features
# 3.- Calculate the mean correlation of every feature against the rest
#     of features
# 4.- Keep combination having correlation over 0.95. 
#     Of the pair of features, the one having the 
#     largest mean correlation will be removed
# 4.- Save the results
import os
import sys
import numpy as np
import pandas as pd
from docopt import docopt
from collections import defaultdict
from scipy.stats import pearsonr
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, ".."))
from utils import settings
from utils.file_managers import get_json_content
from utils.customized_utils import prepare_data_for_ml
from utils.windows import get_index_features_map
from utils.file_managers import write_pickle
from utils.parameters import *


OPTS = docopt(__doc__)
settings = settings[OPTS["--db"]]
input_path = os.path.join(current_dir, settings["windows"].format(OPTS["--feature_set"]))
ictal_path = input_path + "/Common/Ictal/"
noictal_path = input_path + "/Common/NoIctal/"
features_map_file = os.path.join(current_dir, settings["features_map_file"].format(OPTS["--feature_set"]))
output_path = os.path.join(current_dir, settings["correlation"].format(OPTS["--feature_set"]))


data, _ = prepare_data_for_ml(ictal_path, noictal_path)
data[data == np.inf] = 0
data[data == -np.inf] = 0
data = pd.DataFrame(data)
data = data.fillna(0)
data = data.to_numpy()
features_map = get_json_content(features_map_file)
dummy_importances = np.arange(data.shape[1])

indexes, names, _ = get_index_features_map(features_map, dummy_importances,
                                           None, invert=True)


pearson_coeffs = []
mean_pearson_coeffs = defaultdict(int)
for index_1 in indexes:
    for index_2 in indexes:
        if index_1 == index_2:
            continue
        coeff, _ = pearsonr(data[:, index_1], data[:, index_2])
        coeff = np.abs(coeff.tolist())
        pearson_coeffs.append([index_1, names[index_1], index_2,
                               names[index_2], coeff])
        mean_pearson_coeffs[index_1] += coeff/len(indexes)
        mean_pearson_coeffs[index_2] += coeff/len(indexes)

pearson_coeffs = np.vstack(pearson_coeffs)
pearson_coeffs = pd.DataFrame(pearson_coeffs, columns=["Index_1", "Name_1",
                                                       "Index_2", "Name_2",
                                                       "Pearson_coeff"])
pearson_coeffs["Pearson_coeff"] = pd.to_numeric(pearson_coeffs["Pearson_coeff"])
write_pickle(output_path, pearson_coeffs)

filtered_results = pearson_coeffs[pearson_coeffs["Pearson_coeff"] > 0.95]
filtered_results = filtered_results.sort_values(["Pearson_coeff"])
filtered_results = filtered_results.to_dict(orient="records")

safe_features = set()
blacklisted_features = set()

for record in filtered_results:
    if mean_pearson_coeffs[record["Index_1"]] > mean_pearson_coeffs[record["Index_2"]]:
        print(record)
        if record["Index_1"] not in safe_features:
            blacklisted_features.add(record["Index_1"])
            if record["Index_2"] not in blacklisted_features:
                safe_features.add(record["Index_2"])
    else:
        if record["Index_2"] not in safe_features:
            blacklisted_features.add(record["Index_2"])
            if record["Index_1"] not in blacklisted_features:
                safe_features.add(record["Index_1"])

print("Safe features:")
print(safe_features)
print("Blacklisted features:")
print(list(blacklisted_features))
print(safe_features.intersection(blacklisted_features))