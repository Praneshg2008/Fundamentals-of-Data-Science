"""
CS23334 Fundamentals of Data Science Lab - Experiment 3
Data Visualization using Line Plot (3.a), Bar Chart (3.b), and Pie Chart (3.c).
"""

import matplotlib.pyplot as plt
import numpy as np


def exp_3a_line_plot():
    """Exp 3.a: Conduct an experiment to show data visualization using line plot."""
    print("--- Exp 3.a: Line Plot (INDIA Vs SRILANKA) ---")
    Overs = list(range(5, 51, 5))
    Indian_Score = [30, 55, 90, 129, 165, 200, 239, 270, 310, 350]
    Srilankan_Score = [25, 70, 90, 120, 140, 170, 195, 220, 255, 279]

    plt.figure(figsize=(8, 5))
    plt.plot(Overs, Indian_Score, color="green", label="INDIA")
    plt.plot(Overs, Srilankan_Score, color="red", label="SRILANKA")
    plt.title("INDIA Vs SRILANKA")
    plt.xlabel("Overs")
    plt.ylabel("Score")
    plt.legend(loc="center right")
    plt.show()


def exp_3b_bar_chart():
    """Exp 3.b: Conduct an experiment to show data visualization using bar chart."""
    print("--- Exp 3.b: Bar Chart (Comparison of HSC Percentage) ---")
    Names = ['SHREE', 'DEV', 'KEERTHI', 'PRIYA', 'SHAN', 'KUMARAN']
    xaxis = np.arange(len(Names))
    Percentage_hsc = [96, 91, 94, 75, 45, 81]

    plt.figure(figsize=(8, 5))
    plt.bar(Names, Percentage_hsc)
    plt.xticks(xaxis, Names, rotation=45)
    plt.xlabel("Names of Pupil")
    plt.ylabel("Percentage")
    plt.title("Comparison of HSC Percentage", fontsize=20, color="green")
    plt.tight_layout()
    plt.show()


def exp_3c_pie_chart():
    """Exp 3.c: Conduct an experiment to show data visualization using pie chart."""
    print("--- Exp 3.c: Pie Chart (Election Results) ---")
    labels = ['CANDIDATE 1', 'CANDIDATE 2', 'CANDIDATE 3', 'CANDIDATE 4']
    Votes = [315, 130, 245, 210]
    colors = ['green', 'yellow', 'red', 'orange']
    explode = (0.2, 0, 0, 0)

    plt.figure(figsize=(7, 7))
    plt.pie(Votes, labels=labels, colors=colors, explode=explode, autopct='%0.2f%%')
    plt.title('Election Results')
    plt.show()


if __name__ == "__main__":
    exp_3a_line_plot()
    exp_3b_bar_chart()
    exp_3c_pie_chart()
