"""
CS23334 Fundamentals of Data Science Lab - Experiment 15
Data Preprocessing Pipeline & Encoding in Data Science.
"""

import os
import numpy as np
import pandas as pd


def ensure_preprocess_csv(file_path="pre-process_datasample.csv"):
    """Create pre-process_datasample.csv if not present."""
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
    df.info()

    print("\nCountry mode:", df['Country'].mode()[0])
    print("Type of mode():", type(df['Country'].mode()))

    df['Country'] = df['Country'].fillna(df['Country'].mode()[0])
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Salary'] = df['Salary'].fillna(round(df['Salary'].mean()))
    print("\nImputed DataFrame:\n", df)

    dummies = pd.get_dummies(df['Country'])
    print("\nDummy variables:\n", dummies)

    updated_dataset = pd.concat([dummies, df.iloc[:, [1, 2, 3]]], axis=1)
    updated_dataset['Purchased'] = updated_dataset['Purchased'].replace(['No', 'Yes'], [0, 1])
    print("\nFinal Encoded Dataset:\n", updated_dataset)


if __name__ == "__main__":
    main()
