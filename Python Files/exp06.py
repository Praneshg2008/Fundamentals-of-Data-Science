"""
CS23334 Fundamentals of Data Science Lab - Experiment 6
Time Series Analysis of Website Traffic.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


def ensure_website_traffic_csv(file_path="website_traffic.csv"):
    """Create sample website_traffic.csv with trend, seasonality, and anomalies if not present."""
    if not os.path.exists(file_path):
        np.random.seed(42)
        dates = pd.date_range(start="2019-09-01", periods=365, freq="D")
        trend = np.linspace(100, 300, 365)
        seasonality = 30 * np.sin(2 * np.pi * np.arange(365) / 7)
        noise = np.random.normal(0, 15, 365)
        page_views = trend + seasonality + noise

        # Inject a few anomalies (spikes)
        anomaly_indices = [50, 120, 200, 260, 310, 350]
        for idx in anomaly_indices:
            page_views[idx] += 1500

        df = pd.DataFrame({
            "Date": dates,
            "Page_Views": np.round(page_views, 2),
            "Unique_Visitors": np.round(page_views * 0.7, 0),
            "Bounce_Rate": np.round(np.random.uniform(30, 60, 365), 2)
        })
        df.to_csv(file_path, index=False)


def main():
    ensure_website_traffic_csv("website_traffic.csv")

    # Load dataset
    data = pd.read_csv('website_traffic.csv', parse_dates=['Date'], index_col='Date')
    data.index.freq = 'D'

    # Time Series Decomposition
    decomposition = seasonal_decompose(data['Page_Views'], model='additive')
    decomposition.plot()
    plt.show()

    # Moving Average
    data['Page_Views_MA'] = data['Page_Views'].rolling(window=30).mean()
    data[['Page_Views', 'Page_Views_MA']].plot()
    plt.title('Page Views with Moving Average')
    plt.show()

    # Anomaly Detection
    data['Page_Views_Z'] = (
        (data['Page_Views'] - data['Page_Views'].mean()) / data['Page_Views'].std()
    )
    anomalies = data[data['Page_Views_Z'].abs() > 3]
    print(f"Detected {len(anomalies)} anomalies:")
    print(anomalies[['Page_Views', 'Page_Views_Z']])

    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data['Page_Views'], label='Page Views')
    plt.scatter(anomalies.index, anomalies['Page_Views'], color='red', label='Anomalies')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
