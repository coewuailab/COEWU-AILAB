import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path
import json
import re
import matplotlib.pyplot as plt


if __name__ == "__main__":
    ##########################################################################################################
    # Step 8: Clean the data, remove outliers, and save the cleaned data
    file_path = "data/processed_data_selected_features_1min.csv"
    df = pd.read_csv(file_path, index_col='Timestamp', parse_dates=True)

    # Ensure the index is datetime
    df.index = pd.to_datetime(df.index, errors='coerce')

    print(df.describe().to_string(), "\n\n")

    # Drop outliers based on piera_PM25 values over 6000
    df_cleaned = df[df['piera_PM25'] <= 6000].copy()

    # Get daily 75th percentile thresholds
    daily_threshold = df_cleaned['piera_PM25'].resample('D').quantile(0.90)

    # Remove outliers based on daily 75th percentile in same day
    df_cleaned = df_cleaned[df_cleaned['piera_PM25'] <= daily_threshold[df_cleaned.index.date].values]

    print(df_cleaned.describe().to_string(), "\n\n")

    # List of Dates that will be removed
    dates_to_remove = [
        '2025-01-16',
        '2025-03-05 08:30:00',
        '2025-03-18 15:00:00',
        '2025-03-19',
        '2025-03-20 13:00:00',
        '2025-03-27 13:30:00',
        '2025-03-28 12:30:00',
        '2025-03-30 11:30:00',
        '2025-03-31',
        '2025-04-03 08:30:00',
        '2025-04-26 08:30:00',
        '2025-04-27',
        '2025-04-28',
        '2025-04-29',
        '2025-05-04',
        '2025-05-16 12:00:00',
        '2025-05-17',
        '2025-05-19',
        '2025-05-23 12:30:00',
        '2025-06-01',
    ]

    # If the date is in the index, remove it the whole day,
    # if the date and time are in the index, remove it only that hour to the end of the day
    for date in dates_to_remove:
        if ' ' in date:  # If it contains time
            date_time = pd.to_datetime(date)
            df_cleaned = df_cleaned[df_cleaned.index < date_time]
        else:  # If it is just a date
            date_only = pd.to_datetime(date).date()
            df_cleaned = df_cleaned[df_cleaned.index.date != date_only]

    # Save the cleaned DataFrame to a new CSV file
    output_file_path = "data/processed_data_selected_features_1min_cleaned.csv"
    df_cleaned.to_csv(output_file_path, index=True)

    print(df_cleaned.describe().to_string(), "\n\n")

    for day, group in df_cleaned.groupby(df_cleaned.index.date):
        plt.figure(figsize=(14, 7))
        group['piera_PM25'].plot(label='piera_PM25', color='blue')
        group['pmnova_PM25'].plot(label='pmnova_PM25', color='orange')
        group['pmnova_PM25_calibrated'].plot(label='pmnova_PM25_calibrated', color='green')
        plt.title(f'Daily PM2.5 Levels on {day}')
        plt.xlabel('Timestamp')
        plt.ylabel('PM2.5 Concentration')
        plt.legend()
        plt.grid()

        # Save the plot as a PNG file
        # Check if the directory exists, if not create it
        import os

        os.makedirs("data/plots", exist_ok=True)
        output_file = f"data/plots/pm25_daily_{day}.png"
        plt.savefig(output_file)
        plt.close()

