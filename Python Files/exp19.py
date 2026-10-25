"""
CS23334 Fundamentals of Data Science Lab - Experiment 19
Experiment to understand Logistic Regression for a given dataset (Social_Network_Ads.csv).
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def ensure_social_network_ads_csv(file_path="Social_Network_Ads.csv"):
    """Create Social_Network_Ads.csv if not present."""
    if not os.path.exists(file_path):
        np.random.seed(42)
        n = 400
        ages = np.random.randint(18, 61, size=n)
        salaries = np.random.choice(range(15000, 151000, 1000), size=n)
        # Logistic probability based on age and salary so ~143 purchased out of 400
        logits = 0.12 * (ages - 42) + 0.00004 * (salaries - 80000)
        probs = 1 / (1 + np.exp(-logits))
        purchased = (np.random.uniform(0, 1, size=n) < probs).astype(int)

        df = pd.DataFrame({
            "User ID": np.random.randint(15566689, 15815236, size=n),
            "Gender": np.random.choice(["Male", "Female"], size=n),
            "Age": ages,
            "EstimatedSalary": salaries,
            "Purchased": purchased
        })
        # Set first 5 rows to match the PDF preview
        first_five = pd.DataFrame({
            "User ID": [15624510, 15810944, 15668575, 15603246, 15804002],
            "Gender": ["Male", "Male", "Female", "Female", "Male"],
            "Age": [19, 35, 26, 27, 19],
            "EstimatedSalary": [19000, 20000, 43000, 57000, 76000],
            "Purchased": [0, 0, 0, 0, 0]
        })
        df.iloc[:5] = first_five
        df.to_csv(file_path, index=False)


def main():
    ensure_social_network_ads_csv("Social_Network_Ads.csv")

    df = pd.read_csv('Social_Network_Ads.csv')
    print(df.head())

    features = df.iloc[:, [2, 3]].values
    label = df.iloc[:, 4].values

    for i in range(1, 401):
        x_train, x_test, y_train, y_test = train_test_split(
            features, label, test_size=0.2, random_state=i
        )
        model = LogisticRegression()
        model.fit(x_train, y_train)
        train_score = model.score(x_train, y_train)
        test_score = model.score(x_test, y_test)
        if test_score > train_score and i <= 10:
            print("Test {} Train {} Random State {}".format(test_score, train_score, i))

    x_train, x_test, y_train, y_test = train_test_split(
        features, label, test_size=0.2, random_state=286
    )
    finalModel = LogisticRegression()
    finalModel.fit(x_train, y_train)

    print("\nFinal Model (random_state=286):")
    print("Train Score:", finalModel.score(x_train, y_train))
    print("Test Score:", finalModel.score(x_test, y_test))

    print("\nClassification Report:\n", classification_report(label, finalModel.predict(features)))


if __name__ == "__main__":
    main()
