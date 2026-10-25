"""
CS23334 Fundamentals of Data Science Lab - Experiment 8
Conduct Z-Test for Given Data.
"""

import numpy as np
import scipy.stats as stats


def main():
    sample_data = np.array([
        152, 148, 151, 149, 147, 153, 150, 148, 152, 149,
        151, 150, 149, 152, 151, 148, 150, 152, 149, 150,
        148, 153, 151, 150, 149, 152, 148, 151, 150, 153
    ])

    population_mean = 150

    sample_mean = np.mean(sample_data)
    sample_std = np.std(sample_data, ddof=1)

    n = len(sample_data)

    z_statistic = (sample_mean - population_mean) / (sample_std / np.sqrt(n))

    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_statistic)))

    print(f"Sample Mean: {sample_mean:.2f}")
    print(f"Z-Statistic: {z_statistic:.4f}")
    print(f"P-Value: {p_value:.4f}")

    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis: The average weight is significantly different from 150 grams.")
    else:
        print("Fail to reject the null hypothesis: There is no significant difference in average weight from 150 grams.")


if __name__ == "__main__":
    main()
