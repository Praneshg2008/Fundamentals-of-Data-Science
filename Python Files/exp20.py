"""
CS23334 Fundamentals of Data Science Lab - Experiment 20
Experiment to understand K-Means clustering algorithm for a given dataset (Mall_Customers.csv).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


def ensure_mall_customers_csv(file_path="Mall_Customers.csv"):
    """Create 200-row Mall_Customers.csv with 5 natural clusters if not present."""
    if not os.path.exists(file_path):
        np.random.seed(42)
        # 5 customer segments in (Annual Income, Spending Score)
        centers = [
            (25, 20),  # Low income, low spending
            (25, 80),  # Low income, high spending
            (55, 50),  # Medium income, medium spending
            (85, 82),  # High income, high spending
            (88, 18)   # High income, low spending
        ]
        incomes, scores = [], []
        for inc_c, sc_c in centers:
            incomes.extend(np.clip(np.random.normal(inc_c, 7, 40), 15, 137).astype(int))
            scores.extend(np.clip(np.random.normal(sc_c, 8, 40), 1, 99).astype(int))

        df = pd.DataFrame({
            "CustomerID": range(1, 201),
            "Gender": np.random.choice(["Male", "Female"], size=200),
            "Age": np.random.randint(18, 70, size=200),
            "Annual Income (k$)": incomes,
            "Spending Score (1-100)": scores
        })
        first_five = pd.DataFrame({
            "CustomerID": [1, 2, 3, 4, 5],
            "Gender": ["Male", "Male", "Female", "Female", "Female"],
            "Age": [19, 21, 20, 23, 31],
            "Annual Income (k$)": [15, 15, 16, 16, 17],
            "Spending Score (1-100)": [39, 81, 6, 77, 40]
        })
        df.iloc[:5] = first_five
        df.to_csv(file_path, index=False)


def main():
    ensure_mall_customers_csv("Mall_Customers.csv")

    df = pd.read_csv('Mall_Customers.csv')
    df.info()
    print(df.head())

    sns.pairplot(df)
    plt.show()

    features = df.iloc[:, [3, 4]].values

    model = KMeans(n_clusters=5, random_state=42, n_init=10)
    model.fit(features)

    Final = df.iloc[:, [3, 4]].copy()
    Final['label'] = model.predict(features)
    print("\nClustered Data Head:\n", Final.head())

    sns.set_style("whitegrid")
    sns.FacetGrid(Final, hue="label", height=8) \
        .map(plt.scatter, "Annual Income (k$)", "Spending Score (1-100)") \
        .add_legend()
    plt.show()

    features_el = df.iloc[:, [2, 3, 4]].values
    wcss = []
    for i in range(1, 10):
        model_el = KMeans(n_clusters=i, random_state=42, n_init=10)
        model_el.fit(features_el)
        wcss.append(model_el.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 10), wcss, marker='o')
    plt.title('Elbow Method (WCSS vs Number of Clusters)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS (Inertia)')
    plt.show()


if __name__ == "__main__":
    main()
