"""
CS23334 Fundamentals of Data Science Lab - Experiment 12
Procedure to Identify and Detect Outliers in a Dataset using the IQR Method.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def outDetection(array):
    sorted(array)
    Q1, Q3 = np.percentile(array, [25, 75])
    IQR = Q3 - Q1
    lr = Q1 - (1.5 * IQR)
    ur = Q3 + (1.5 * IQR)
    return lr, ur


def main():
    array = np.array([27, 50, 44, 6, 58, 61, 23, 86, 67, 20, 75, 7, 79, 61, 90, 54])
    print("Array:", array)
    print("Mean:", array.mean())
    print("25th percentile:", np.percentile(array, 25))
    print("50th percentile:", np.percentile(array, 50))
    print("75th percentile:", np.percentile(array, 75))
    print("100th percentile:", np.percentile(array, 100))

    lr, ur = outDetection(array)
    print("Lower & Upper bounds (lr, ur):", (lr, ur))

    sns.displot(array)
    plt.title("Histogram of Array Data (displot)")
    plt.show()

    sns.histplot(array, kde=True, stat="density")
    plt.title("Distribution with KDE Curve")
    plt.show()

    new_array = array[(array > lr) & (array < ur)]
    print("Filtered new_array:", new_array)

    sns.displot(new_array)
    plt.title("Histogram of Filtered Array")
    plt.show()

    lr1, ur1 = outDetection(new_array)
    print("Re-verified bounds (lr1, ur1):", (lr1, ur1))

    final_array = new_array[(new_array > lr1) & (new_array < ur1)]
    print("Final array:", final_array)

    sns.histplot(final_array, kde=True, stat="density")
    plt.title("Final Array Distribution with KDE")
    plt.show()


if __name__ == "__main__":
    main()
