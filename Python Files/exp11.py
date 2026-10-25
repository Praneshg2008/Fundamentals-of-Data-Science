"""
CS23334 Fundamentals of Data Science Lab - Experiment 11
Data Frame Use Cases & Operations in Data Science using Pandas.
"""

import os
import numpy as np
import pandas as pd


def ensure_csv_files():
    """Create 50_Startups.csv and employee.csv if not present."""
    if not os.path.exists("50_Startups.csv"):
        np.random.seed(0)
        startups_data = {
            "R&D Spend": [165349.20, 162597.70, 153441.51, 144372.41, 142107.34]
            + list(np.round(np.linspace(135000, 10000, 45), 2)),
            "Administration": [136897.80, 151377.59, 101145.55, 118671.85, 91391.77]
            + list(np.round(np.random.uniform(80000, 160000, 45), 2)),
            "Marketing Spend": [471784.10, 443898.53, 407934.54, 383199.62, 366168.42]
            + list(np.round(np.linspace(350000, 20000, 45), 2)),
            "State": ["New York", "California", "Florida", "New York", "Florida"]
            + ["New York", "California", "Florida"] * 15,
            "Profit": [192261.83, 191792.06, 191050.39, 182901.99, 166187.94]
            + list(np.round(np.linspace(160000, 40000, 45), 2))
        }
        pd.DataFrame(startups_data).to_csv("50_Startups.csv", index=False)

    if not os.path.exists("employee.csv"):
        emp_data = {
            "emp id": [1, 2, 3, 4, 5, 6, 7],
            "name": [
                "SREE VARSSINI K S",
                "SREEMATHI B",
                "SREYA G",
                "SREYASKARI MULLAPUDI",
                "SRI AKASH U G",
                "SRI HARSHAVARDHANAN R",
                "SRI HARSHAVARDHANAN R"
            ],
            "salary": [5000, 6000, 7000, 5000, 8000, 3000, 6000]
        }
        pd.DataFrame(emp_data).to_csv("employee.csv", index=False)


def main():
    ensure_csv_files()

    list_data = [[1, 'Smith', 50000], [2, 'Jones', 60000]]
    df = pd.DataFrame(list_data)
    print(df)

    df.columns = ['Empd', 'Name', 'Salary']
    print(df)
    df.info()

    df_startups = pd.read_csv("50_Startups.csv")
    df_startups.info()
    print(df_startups.head())
    print(df_startups.tail())

    df_emp = pd.read_csv("employee.csv")
    print(df_emp.head())
    print(df_emp.tail())
    df_emp.info()

    print("\ndf_emp.salary:\n", df_emp.salary)
    print("type(df_emp.salary):", type(df_emp.salary))

    print("Mean:", df_emp.salary.mean())
    print("Median:", df_emp.salary.median())
    print("Mode:\n", df_emp.salary.mode())
    print("Variance:", df_emp.salary.var())
    print("Std Dev:", df_emp.salary.std())

    print("\nDescribe:\n", df_emp.describe())
    print("\nDescribe (include='all'):\n", df_emp.describe(include='all'))

    empCol = df_emp.columns
    emparray = df_emp.values

    employee_DF = pd.DataFrame(emparray, columns=empCol)
    print("\nReconstructed employee_DF:\n", employee_DF)


if __name__ == "__main__":
    main()
