#!/usr/bin/python3.7
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from .models import Classifiers
from .file_managers import write_pickle
from .windows import get_index_features_map


def generate_labels(*args) -> np.array:
    class_number = 0
    labels = []

    for index, num_instances in enumerate(args):
        labels.append([class_number]*num_instances)
        class_number += 1

    return np.concatenate(labels)


def get_es_callback(patience: int = 50):
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto',
                                          verbose=1, patience=patience)
    return es


def grid_search(model_name: int, folds: int, X: np.array,
                Y: np.array, directory: str, grid_params: dict):
    np.random.seed(1)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    classifier = Classifiers(model_name)
    model = classifier(0, {})
    clf = GridSearchCV(model, grid_params, cv=folds, verbose=3)
    clf.fit(X, Y)
    grid_dir = directory + "/grid_%s.pickle" % (model_name)
    write_pickle(grid_dir, clf.cv_results_)


def k_folds_stratified_nn(model_name: int, repetitions: int,
                          X: np.array, Y: np.array, directory: str,
                          features_map: dict, top: str, explainer: str,
                          importances: list, start_kwargs: dict,
                          train_kwargs: dict, blacklist: list):
    np.random.seed(1)
    tf.random.set_seed(1)
    rng = np.random.RandomState(1)

    invert = False
    if explainer == "reciprocalranking":
        invert = True

    y_real = []
    y_predicted = []

    classifier = Classifiers(model_name)
    indices, _, _ = get_index_features_map(features_map,
                                           importances,
                                           top, invert=invert,
                                           blacklist=blacklist)

    for _ in range(repetitions):
        randint = rng.randint(low=0, high=32767)
        X_1, X_2, y_1, y_2 = train_test_split(X, Y, test_size=0.5,
                                              random_state=randint)
        for X_train, X_test, y_train, y_test in zip((X_1, X_2),
                                                    (X_2, X_1),
                                                    (y_1, y_2),
                                                    (y_2, y_1)):

            model = classifier(top, start_kwargs)
            X_train = X_train[:, indices]
            X_test = X_test[:, indices]
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model.fit(X_train, y_train, **train_kwargs)

            results = model.predict(X_test)
            y_real.append(y_test)
            y_predicted.append(results)

            tf.keras.backend.clear_session()
            del model

    stats_dir = directory + "/stats_%s_%s_%s.pickle" % (model_name,
                                                        explainer,
                                                        top)
    stats_data = {"real": y_real,
                  "predicted": y_predicted}
    write_pickle(stats_dir, stats_data)
