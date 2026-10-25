"""
CS23334 Fundamentals of Data Science Lab - Experiment 13
Procedure to Handle Missing Values & Inappropriate Data in a Dataset (Hotel_Dataset.csv).
"""

import os
import numpy as np
import pandas as pd


def ensure_hotel_dataset_csv(file_path="Hotel_Dataset.csv"):
    """Create Hotel_Dataset.csv matching Experiment 13 if not present."""
    if not os.path.exists(file_path):
        data = {
            "CustomerID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10],
            "Age_Group": [
                "20-25", "30-35", "25-30", "20-25", "35+",
                "35+", "35+", "20-25", "25-30", "25-30", "30-35"
            ],
            "Rating(1-5)": [4, 5, 6, -1, 3, 3, 4, 7, 2, 2, 5],
            "Hotel": [
                "Ibis", "LemonTree", "RedFox", "LemonTree", "Ibis",
                "Ibys", "RedFox", "LemonTree", "Ibis", "Ibis", "RedFox"
            ],
            "FoodPreference": [
                "veg", "Non-Veg", "Veg", "Veg", "Vegetarian",
                "Non-Veg", "Vegetarian", "Veg", "Non-Veg", "Non-Veg", "non-Veg"
            ],
            "Bill": [1300, 2000, 1322, 1234, 989, 1909, 1000, 2999, 3456, 3456, 6755],
            "NoOfPax": [2, 3, 2, 2, 2, 2, -1, -10, 3, 3, 4],
            "EstimatedSalary": [
                40000, 59000, 30000, 120000, 45000,
                122220, 21122, 345673, -99999, -99999, 87777
            ],
            "Age_Group.1": [
                "20-25", "30-35", "25-30", "20-25", "35+",
                "35+", "35+", "20-25", "25-30", "25-30", "30-35"
            ]
        }
        pd.DataFrame(data).to_csv(file_path, index=False)


def main():
    ensure_hotel_dataset_csv("Hotel_Dataset.csv")

    df = pd.read_csv("Hotel_Dataset.csv")
    print("Initial DataFrame:\n", df)
    print("\nDuplicated rows:\n", df.duplicated())
    df.info()

    df.drop_duplicates(inplace=True)
    index = np.array(list(range(0, len(df))))
    df.set_index(index, inplace=True)
    df.drop(['Age_Group.1'], axis=1, inplace=True)
    print("\nAfter dropping duplicates & redundant column:\n", df)

    # Replace invalid negative / out-of-bounds values with NaN
    df.loc[df['CustomerID'] < 0, 'CustomerID'] = np.nan
    df.loc[df['Bill'] < 0, 'Bill'] = np.nan
    df.loc[df['EstimatedSalary'] < 0, 'EstimatedSalary'] = np.nan
    df.loc[(df['Rating(1-5)'] < 1) | (df['Rating(1-5)'] > 5), 'Rating(1-5)'] = np.nan
    df.loc[(df['NoOfPax'] < 1) | (df['NoOfPax'] > 20), 'NoOfPax'] = np.nan

    print("\nUnique Age_Group:", df['Age_Group'].unique())
    print("Unique Hotel:", df['Hotel'].unique())

    # Standardize categorical strings
    df['Hotel'] = df['Hotel'].replace(['Ibys'], 'Ibis')
    print("Unique FoodPreference:", df['FoodPreference'].unique())
    df['FoodPreference'] = df['FoodPreference'].replace(['Vegetarian', 'veg'], 'Veg')
    df['FoodPreference'] = df['FoodPreference'].replace(['non-Veg'], 'Non-Veg')

    # Impute missing values
    df['EstimatedSalary'] = df['EstimatedSalary'].fillna(round(df['EstimatedSalary'].mean()))
    df['NoOfPax'] = df['NoOfPax'].fillna(round(df['NoOfPax'].median()))
    df['Rating(1-5)'] = df['Rating(1-5)'].fillna(round(df['Rating(1-5)'].median()))
    df['Bill'] = df['Bill'].fillna(round(df['Bill'].mean()))

    print("\nCleaned & Imputed DataFrame:\n", df)


if __name__ == "__main__":
    main()
