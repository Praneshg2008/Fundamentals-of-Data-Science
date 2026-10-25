"""
CS23334 Fundamentals of Data Science Lab - Experiment 5
Data Collection and Initial Exploration (Diabetes Dataset).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def ensure_diabetes_csv(file_path="diabetes.csv"):
    """Generate sample diabetes.csv if not present."""
    if not os.path.exists(file_path):
        np.random.seed(42)
        n = 768
        df = pd.DataFrame({
            "Pregnancies": np.random.randint(0, 17, n),
            "Glucose": np.random.randint(50, 200, n),
            "BloodPressure": np.random.randint(40, 122, n),
            "SkinThickness": np.random.randint(0, 99, n),
            "Insulin": np.random.randint(0, 846, n),
            "BMI": np.round(np.random.uniform(18.0, 67.1, n), 1),
            "DiabetesPedigreeFunction": np.round(np.random.uniform(0.078, 2.42, n), 3),
            "Age": np.random.randint(21, 81, n),
            "Outcome": np.random.choice([0, 1], size=n, p=[0.65, 0.35])
        })
        # Set first 5 rows to match the standard Pima Indians Diabetes dataset preview
        first_five = pd.DataFrame({
            "Pregnancies": [6, 1, 8, 1, 0],
            "Glucose": [148, 85, 183, 89, 137],
            "BloodPressure": [72, 66, 64, 66, 40],
            "SkinThickness": [35, 29, 0, 23, 35],
            "Insulin": [0, 0, 0, 94, 168],
            "BMI": [33.6, 26.6, 23.3, 28.1, 43.1],
            "DiabetesPedigreeFunction": [0.627, 0.351, 0.672, 0.167, 2.288],
            "Age": [50, 31, 32, 21, 33],
            "Outcome": [1, 0, 1, 0, 1]
        })
        df.iloc[:5] = first_five
        df.to_csv(file_path, index=False)


def main():
    ensure_diabetes_csv("diabetes.csv")

    # 1. Data Collection
    diabetes_df = pd.read_csv("diabetes.csv")
    print(diabetes_df.head())

    # 2. Initial Exploration
    print(diabetes_df.info())
    print(diabetes_df.describe())

    diabetes_df.hist(bins=50, figsize=(20, 15))
    plt.show()

    sns.pairplot(diabetes_df)
    plt.show()


if __name__ == "__main__":
    main()
