"""
CS23334 Fundamentals of Data Science Lab - Experiment 16
Experiment to understand EDA - Quantitative and Qualitative Analysis (Seaborn 'tips' dataset).
"""

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    tips = sns.load_dataset('tips')
    print(tips.head())

    sns.displot(tips.total_bill, kde=True)
    plt.show()

    sns.displot(tips.total_bill, kde=False)
    plt.show()

    sns.jointplot(x=tips.tip, y=tips.total_bill)
    plt.show()

    sns.jointplot(x=tips.tip, y=tips.total_bill, kind="reg")
    plt.show()

    sns.jointplot(x=tips.tip, y=tips.total_bill, kind="hex")
    plt.show()

    sns.pairplot(tips)
    plt.show()

    print("\nTime value counts:\n", tips.time.value_counts())

    sns.pairplot(tips, hue='time')
    plt.show()

    sns.pairplot(tips, hue='day')
    plt.show()

    plt.figure(figsize=(6, 5))
    sns.heatmap(tips.corr(numeric_only=True), annot=True)
    plt.show()

    plt.figure(figsize=(6, 5))
    sns.boxplot(y=tips.total_bill)
    plt.show()

    plt.figure(figsize=(6, 5))
    sns.boxplot(y=tips.tip)
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.countplot(y=tips.day)
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.countplot(y=tips.sex)
    plt.show()

    plt.figure(figsize=(6, 6))
    tips.sex.value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.show()

    plt.figure(figsize=(6, 4))
    tips.sex.value_counts().plot(kind='bar')
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.countplot(y=tips[tips.time == 'Dinner']['day'])
    plt.show()


if __name__ == "__main__":
    main()
