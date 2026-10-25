"""
CS23334 Fundamentals of Data Science Lab - Experiment 18
Experiment to understand KNN algorithm for a given dataset (Iris.csv).
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report


def ensure_iris_csv(file_path="Iris.csv"):
    """Create Iris.csv matching Experiment 18 schema if not present."""
    if not os.path.exists(file_path):
        iris = load_iris()
        variety_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
        df = pd.DataFrame(
            iris.data,
            columns=["sepal.length", "sepal.width", "petal.length", "petal.width"]
        )
        df["variety"] = [variety_map[t] for t in iris.target]
        df.to_csv(file_path, index=False)


def main():
    ensure_iris_csv("Iris.csv")

    df = pd.read_csv('Iris.csv')
    df.info()
    print("\nVariety counts:\n", df.variety.value_counts())
    print("\nHead:\n", df.head())

    features = df.iloc[:, :-1].values
    label = df.iloc[:, 4].values

    xtrain, xtest, ytrain, ytest = train_test_split(
        features, label, test_size=0.2, random_state=1
    )

    model_KNN = KNeighborsClassifier(n_neighbors=5)
    model_KNN.fit(xtrain, ytrain)

    print("Train Accuracy:", model_KNN.score(xtrain, ytrain))
    print("Test Accuracy:", model_KNN.score(xtest, ytest))

    cm = confusion_matrix(label, model_KNN.predict(features))
    print("\nConfusion Matrix:\n", cm)

    print("\nClassification Report:\n", classification_report(label, model_KNN.predict(features)))


if __name__ == "__main__":
    main()
