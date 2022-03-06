#!/usr/bin/python3.7
from glob import glob
import numpy as np
import tensorflow as tf
from lime.lime_tabular import LimeTabularExplainer
from shap import KernelExplainer, kmeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.preprocessing import MinMaxScaler
from .models import Classifiers
from .file_managers import read_pickle
from .windows import get_index_features_map, random_selection


def reciprocal_ranking(importances, features_map):
    rankings = []
    for imp in importances:
        ranking, names, _ = get_index_features_map(features_map,
                                                   imp,
                                                   None,
                                                   invert=True)
        rankings.append(ranking)

    rec_ranking = []
    mean_ranking = []
    std_ranking = []

    for a in range(len(rankings[0])):
        _reciprocal = []
        _rank = []
        for ranking in rankings:
            _reciprocal.append(1/(ranking.index(a)+1))
            _rank.append(ranking.index(a))
        rec_ranking.append(1/np.sum(_reciprocal))
        mean_ranking.append(np.mean(_rank))
        std_ranking.append(np.std(_rank))

    return rec_ranking, mean_ranking, std_ranking


class FeaturesImportance():

    def __init__(self, X: np.array, Y: np.array, num_instances: int,
                 top_directory: str, features_map: dict):
        np.random.seed(1)
       
        scaler = MinMaxScaler()
        X_train, X_exp, y_train, y_exp = train_test_split(X, Y,
                                                          test_size=0.50,
                                                          random_state=1,
                                                          stratify=Y)
        X_train = scaler.fit_transform(X_train)
        X_exp = scaler.transform(X_exp)
        self.X_train = X_train
        self.Y_train = y_train
        self.X_explain = X_exp
        self.Y_explain = y_exp
        self.top_directory = top_directory
        self.features_map = features_map
        self.instances = num_instances

    def __call__(self, explainer: str, model_name: str, num_features: str,
                 start_kwargs: dict, train_kwargs: dict):
        np.random.seed(1)
        tf.random.set_seed(1)
        classifier = Classifiers(model_name)
        model = classifier(num_features, start_kwargs)

        if explainer == "tree":
            tree_model = DecisionTreeClassifier(random_state=1)
            tree_model.fit(self.X_explain, self.Y_explain)
            return np.abs(tree_model.feature_importances_)

        if explainer == "svm":
            svm_model = SVC(kernel="linear", random_state=1)
            svm_model.fit(self.X_explain, self.Y_explain)
            return np.abs(svm_model.coef_[0])

        elif explainer == "lime":
            model.fit(self.X_train, self.Y_train, **train_kwargs)

            X_explain = self.X_explain[:self.instances, :]
            explainer = LimeTabularExplainer(np.array(X_explain),
                                             discretize_continuous=False,
                                             mode="regression",
                                             random_state=1)
            importances = np.zeros((X_explain.shape))
            for idx in range(X_explain.shape[0]):
                exp = explainer.explain_instance(X_explain[idx, :],
                                                 model.predict,
                                                 num_features=X_explain.shape[1])
                exp = sorted(exp.as_list(), key=lambda x: int(x[0]))
                exp = [a[1] for a in exp]
                importances[idx, :] = exp
            return np.mean(np.abs(importances), axis=0)

        elif explainer == "shap":
            X_explain = self.X_explain[:self.instances, :]
            data = kmeans(X_explain, 50)

            model.fit(self.X_train, self.Y_train, **train_kwargs)

            explainer_model = KernelExplainer(model.predict, data)
            shap_values = explainer_model.shap_values(X_explain)
            return np.mean(np.abs(np.vstack(shap_values)), axis=0)
        
        elif explainer == "embeddedrandomforest":
            forest = RandomForestClassifier(10, random_state=1)
            forest.fit(self.X_train, self.Y_train)
            PI = permutation_importance(forest, self.X_explain, self.Y_explain,
                                        n_repeats=20,
                                        random_state=1)
            return np.abs(PI.importances_mean)
        
        elif explainer == "reciprocalranking":
            directory = self.top_directory + f"/TopFeatures/top_{model_name}_*.pickle"
            files = glob(directory)
            importances = []

            for file in files:
                if "reciprocalranking" not in file:
                    importances.append(read_pickle(file))

            ranking, _, _ = reciprocal_ranking(importances, self.features_map)
            return ranking

        else:
            raise Exception("Not valid algorithm")
