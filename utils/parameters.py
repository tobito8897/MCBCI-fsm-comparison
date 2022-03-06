#!/usr/bin/python3.7

# ML Analysis
from os import terminal_size


max_features = 990
instances_to_explain = 500
repetitions = 5
folds = 2
top_features = [990] + list(range(950, 0, -50)) + [25, 12, 6, 1]

start_kwargs = {"ann": {},
                "tree": {"min_samples_split": 4,
                         "random_state": 1},
                "svm": {"C": 2,
                        "gamma": "scale",
                        "kernel": "rbf",
                        "random_state": 1},
                "knn": {"metric": "manhattan",
                        "n_neighbors": 10,
                        "weights": "uniform"},
                "forest": {"bootstrap": False,
                           "min_samples_split": 8,
                           "n_estimators": 30,
                           "random_state": 1}
                }
train_kwargs = {"ann": {"steps_per_epoch": 5,
                        "epochs": 200,
                        "shuffle": True,
                        "validation_split": 0.2},
                "tree": {},
                "svm": {},
                "knn": {},
                "forest": {}
                }

# Parameters grid search
grid_params = {"tree": {"random_state": [1],
                        "min_samples_split": (2, 4, 8, 16)},
               "svm": {"random_state": [1],
                       "kernel": ("linear", "rbf"),
                       "C": (0.1, 0.5, 1, 2, 10),
                       "gamma": ("scale", "auto")},
               "knn": {"n_neighbors": (3, 5, 10, 15),
                       "weights": ("uniform", "distance"),
                       "metric": ("euclidean", "manhattan")},
               "forest": {"random_state": [1],
                          "n_estimators": (30, 50, 100, 150),
                          "min_samples_split": (2, 4, 8, 16),
                          "bootstrap": (True, False)}
               }


bands = ["full", "delta", "theta", "alpha", "beta"]
features = ["peak_freq", "median_freq", "var", "rms", "skew", "kurt", "std",
            "zero_cross", "sample_entr"]
channels = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3",
            "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4",
            "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8",
            "P8-O2", "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9",
            "FT9-FT10", "FT10-T8"]

methods_short_name = {"ann": "ANN",
                      "tree": "DT",
                      "svm": "SVM",
                      "knn": "KNN",
                      "forest": "RF",
                      "lime": "LIME",
                      "shap": "SHAP",
                      "reciprocalranking": "RR",
                      "embeddedrandomforest": "ERF"}


methods_color = {"ANN": "deeppink",
                 "DT": "dodgerblue",
                 "SVM": "darkorange",
                 "KNN": "yellow",
                 "RF": "black",
                 "LIME": "green",
                 "SHAP": "red",
                 "RR": "sienna",
                 "ERF": "darkorchid"}
