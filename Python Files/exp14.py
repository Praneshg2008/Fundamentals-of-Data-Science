"""
CS23334 Fundamentals of Data Science Lab - Experiment 14
Feature Scaling & Preprocessing Techniques in Data Science using Scikit-Learn.
"""

import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler


def ensure_preprocess_csv(file_path="pre-process_datasample.csv"):
    """Create pre-process_datasample.csv matching Experiments 14 & 15 if not present."""
    if not os.path.exists(file_path):
        data = {
            "Country": [
                "France", "Spain", "Germany", "Spain", "Germany",
                "France", "Spain", "France", np.nan, "France"
            ],
            "Age": [44.0, 27.0, 30.0, 38.0, 40.0, 35.0, np.nan, 48.0, 50.0, 37.0],
            "Salary": [
                72000.0, 48000.0, 54000.0, 61000.0, np.nan,
                58000.0, 52000.0, 79000.0, 83000.0, 67000.0
            ],
            "Purchased": ["No", "Yes", "No", "No", "Yes", "Yes", "No", "Yes", "No", "Yes"]
        }
        pd.DataFrame(data).to_csv(file_path, index=False)


def main():
    ensure_preprocess_csv("pre-process_datasample.csv")

    df = pd.read_csv("pre-process_datasample.csv")
    print("Initial DataFrame:\n", df)

    df['Country'] = df['Country'].fillna(df['Country'].mode()[0])

    features = df.iloc[:, :-1].values
    label = df.iloc[:, -1].values

    age = SimpleImputer(strategy='mean', missing_values=np.nan)
    Salary = SimpleImputer(strategy='mean', missing_values=np.nan)

    age.fit(features[:, [1]])
    Salary.fit(features[:, [2]])

    features[:, [1]] = age.transform(features[:, [1]])
    features[:, [2]] = Salary.transform(features[:, [2]])
    print("\nImputed features:\n", features)

    oh = OneHotEncoder(sparse_output=False)
    Country = oh.fit_transform(features[:, [0]])
    print("\nOne-Hot Encoded Country:\n", Country)

    final_set = np.concatenate((Country, features[:, [1, 2]]), axis=1)
    print("\nFinal feature set:\n", final_set)

    sc = StandardScaler()
    sc.fit(final_set)
    feat_standard_scaler = sc.transform(final_set)
    print("\nStandardScaler output:\n", feat_standard_scaler)

    mms = MinMaxScaler(feature_range=(0, 1))
    mms.fit(final_set)
    feat_minmax_scaler = mms.transform(final_set)
    print("\nMinMaxScaler output:\n", feat_minmax_scaler)


if __name__ == "__main__":
    main()
