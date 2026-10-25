"""
CS23334 Fundamentals of Data Science Lab - Experiment 2
Upload and Analyze the dataset given in CSV format and perform data
preprocessing and visualization.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def ensure_sample_csv(file_path="sales_data.csv"):
    """Create sample sales_data.csv matching Experiment 2 output if not present."""
    if not os.path.exists(file_path):
        data = {
            "Date": pd.date_range("2023-01-01", periods=16, freq="D").strftime("%Y-%m-%d"),
            "Product": [
                "Product A", "Product B", "Product A", "Product C",
                "Product B", "Product A", "Product C", "Product B",
                "Product A", "Product C", "Product B", "Product A",
                "Product C", "Product B", "Product A", "Product C"
            ],
            "Sales": [
                200, 150, 220, 300,
                180, 210, 320, 160,
                230, 310, 190, 240,
                330, 170, 250, 340
            ],
            "Quantity": [4, 3, 5, 6, 4, 5, 7, 3, 6, 7, 4, 6, 8, 3, 7, 8],
            "Region": [
                "North", "South", "North", "East",
                "West", "North", "East", "South",
                "North", "East", "West", "North",
                "East", "South", "North", "East"
            ]
        }
        pd.DataFrame(data).to_csv(file_path, index=False)


def main():
    file_path = "sales_data.csv"
    ensure_sample_csv(file_path)

    # Load the data into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Display the first few rows of the DataFrame
    print(df.head())

    # Check for missing values
    print(df.isnull().sum())

    # Fill or drop missing values if necessary
    df["Sales"] = df["Sales"].fillna(df["Sales"].mean())
    df.dropna(subset=["Product", "Quantity", "Region"], inplace=True)

    # Summary statistics
    print(df.describe())

    # Group by product and calculate the total sales and quantity
    product_summary = df.groupby("Product").agg({
        "Sales": "sum",
        "Quantity": "sum"
    }).reset_index()
    print(product_summary)

    # Bar plot of total sales by product
    plt.figure(figsize=(10, 6))
    plt.bar(product_summary["Product"], product_summary["Sales"])
    plt.xlabel("Product")
    plt.ylabel("Total Sales")
    plt.title("Total Sales by Product")
    plt.show()

    # Line plot of sales over time
    df["Date"] = pd.to_datetime(df["Date"])
    sales_over_time = df.groupby("Date").agg({"Sales": "sum"}).reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(sales_over_time["Date"], sales_over_time["Sales"])
    plt.xlabel("Date")
    plt.ylabel("Total Sales")
    plt.title("Sales Over Time")
    plt.show()

    # Pivot table to analyze sales by region and product
    pivot_table = df.pivot_table(
        values="Sales",
        index="Region",
        columns="Product",
        aggfunc="sum",
        fill_value=0
    )
    print(pivot_table)

    # Correlation matrix
    correlation_matrix = df.corr(numeric_only=True)
    print(correlation_matrix)

    # Heatmap of the correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()


if __name__ == "__main__":
    main()
