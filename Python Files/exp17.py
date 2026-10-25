"""
CS23334 Fundamentals of Data Science Lab - Experiment 17
Experiment to understand Linear Regression for a given dataset (Salary_data.csv).
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def ensure_salary_csv(file_path="Salary_data.csv"):
    """Create standard 30-row Salary_data.csv if not present."""
    if not os.path.exists(file_path):
        data = {
            "YearsExperience": [
                1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.2, 3.7,
                3.9, 4.0, 4.0, 4.1, 4.5, 4.9, 5.1, 5.3, 5.9, 6.0,
                6.8, 7.1, 7.9, 8.2, 8.7, 9.0, 9.5, 9.6, 10.3, 10.5
            ],
            "Salary": [
                39343, 46205, 37731, 43525, 39891, 56642, 60150, 54445, 64445, 57189,
                63218, 55794, 56957, 57081, 61111, 67938, 66029, 83088, 81363, 93940,
                91738, 98273, 101302, 113812, 109431, 105582, 116969, 112635, 122391, 121872
            ]
        }
        pd.DataFrame(data).to_csv(file_path, index=False)


def main():
    ensure_salary_csv("Salary_data.csv")

    df = pd.read_csv('Salary_data.csv')
    print(df.head())
    df.info()

    df.dropna(inplace=True)
    df.info()
    print(df.describe())

    features = df.iloc[:, [0]].values
    label = df.iloc[:, [1]].values

    x_train, x_test, y_train, y_test = train_test_split(
        features, label, test_size=0.2, random_state=0
    )

    model = LinearRegression()
    model.fit(x_train, y_train)

    print("Train R^2 Score:", model.score(x_train, y_train))
    print("Test R^2 Score:", model.score(x_test, y_test))
    print("Coefficient (slope):", model.coef_)
    print("Intercept:", model.intercept_)

    with open('SalaryPred.model', 'wb') as f:
        pickle.dump(model, f)

    with open('SalaryPred.model', 'rb') as f:
        model = pickle.load(f)

    if sys.stdin.isatty():
        user_val = input("Enter Years of Experience (default 44): ").strip()
        yr_of_exp = float(user_val) if user_val else 44.0
    else:
        yr_of_exp = 44.0

    yr_of_exp_NP = np.array([[yr_of_exp]])
    Salary = model.predict(yr_of_exp_NP)

    print("Estimated Salary for {} years of experience is {}: ".format(yr_of_exp, Salary))


if __name__ == "__main__":
    main()
