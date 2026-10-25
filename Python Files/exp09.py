"""
CS23334 Fundamentals of Data Science Lab - Experiment 9
Understanding Matplotlib & Seaborn Library Use Cases in Data Science (Iris Dataset).
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


def ensure_iris_dataset_csv(file_path="Iris_Dataset.csv"):
    """Create Iris_Dataset.csv with columns SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, variety."""
    if not os.path.exists(file_path):
        iris = load_iris()
        variety_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
        df = pd.DataFrame(
            iris.data,
            columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        )
        df["variety"] = [variety_map[t] for t in iris.target]
        df.to_csv(file_path, index=False)


def main():
    ensure_iris_dataset_csv("Iris_Dataset.csv")

    data = pd.read_csv('Iris_Dataset.csv')

    print(data.info())
    print(data.describe())
    print(data.value_counts('variety'))

    sns.countplot(x='variety', data=data)
    plt.show()

    dummies = pd.get_dummies(data.variety)
    FinalDataset = pd.concat([dummies, data.iloc[:, [0, 1, 2, 3]]], axis=1)
    print(FinalDataset.head())

    sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', hue='variety', data=data)
    plt.show()

    sns.scatterplot(x='PetalLengthCm', y='PetalWidthCm', hue='variety', data=data)
    plt.show()

    sns.pairplot(data, hue='variety', height=3)
    plt.show()

    sns.FacetGrid(data, hue='variety', height=5).map(sns.histplot, 'PetalLengthCm').add_legend()
    plt.show()

    sns.FacetGrid(data, hue='variety', height=5).map(sns.histplot, 'PetalWidthCm').add_legend()
    plt.show()

    sns.FacetGrid(data, hue='variety', height=5).map(sns.histplot, 'SepalLengthCm').add_legend()
    plt.show()

    sns.FacetGrid(data, hue='variety', height=5).map(sns.histplot, 'SepalWidthCm').add_legend()
    plt.show()


if __name__ == "__main__":
    main()
