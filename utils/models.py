#!/usr/bin/python3.7
import os
from tensorflow.keras.utils import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# Necesario para usar Theano como backend
os.environ["MKL_THREADING_LAYER"] = "GNU"


def Net(input_dim: int) -> Model:
    """
    Creation of model object, size (width, length)
    """
    model = Sequential()
    model.add(Dense(100, input_dim=input_dim, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    plot_model(model, to_file="../../Images/ANNModel.png", show_shapes=True,
               show_layer_names=True, dpi=300)
    return model


class Classifiers():

    def __init__(self, model: str):
        self.model = model

    def __call__(self, top_features, start_kwargs):
        if self.model == "ann":
            return Net(top_features, **start_kwargs)
        elif self.model == "tree":
            return DecisionTreeClassifier(**start_kwargs)
        elif self.model == "svm":
            return SVC(**start_kwargs)
        elif self.model == "knn":
            return KNeighborsClassifier(**start_kwargs)
        elif self.model == "forest":
            return RandomForestClassifier(**start_kwargs)
        else:
            raise Exception("Not valid model")
