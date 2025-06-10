import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def calibrate_PC01(df):
    # Define bins and calibration factors
    pm25_bins = [0, 25, 50, 75, 100, 125, 150, 175, 200, np.inf]
    calibration_bins = [1, 5, 10, 15, 20, 25, 30, 35, 40]  # Ensure len(labels) = len(bins)-1

    # Apply pd.cut with proper edge handling and STORE the result in DataFrame
    df['calibration_factor'] = pd.cut(
        df['piera_PM25'],
        bins=pm25_bins,
        labels=calibration_bins,
        right=False,
        include_lowest=True
    ).astype(float)

    df['piera_PC01'] = df['piera_PC01'] / 1000 * df['calibration_factor']

    # Drop the 'calibration_factor' column if not needed
    df.drop(columns=['calibration_factor'], inplace=True)

    return df

def calibrate_sds011_industrial_auto(pm25_raw, temp_C, humidity_percent, current_hour):
    #     # 1. Auto-detect industrial activity periods
    #     # current_hour = datetime.datetime.now().hour
    is_active_hour = (18 <= current_hour <= 22) or (4 <= current_hour <= 7)
    # print(f'Industrial active hours: {is_active_hour} (Current hour: {current_hour})')

    # Risk logic: High if (humid OR cold) AND active hours
    transparency_risk = "high" if ((humidity_percent > 80 or temp_C < 15) and is_active_hour) else "low"

    # 2. Industrial-specific base multipliers (higher for coarse PM)
    range_multiplier = next(
        (mult for limit, mult in [(10, 1.2), (20, 1.6), (30, 2.0), (40, 2.5), (50, 3.0), (float('inf'), 4.0)] if
         pm25_raw <= limit), 1.0)

    # 3. Industrial environmental adjustments
    env_factor = (1.0 - 0.003 * max(0, humidity_percent - 60)) * (1.0 - 0.005 * (temp_C - 20))

    # 4. Industrial transparency compensation
    transparency_boost = 2.0 if transparency_risk == "high" else 1.0  # Stronger boost for industrial plumes

    # 5. Final calculation
    calibrated_pm = pm25_raw * range_multiplier * env_factor * transparency_boost
    calibrated_pm = min(1000, max(0, round(calibrated_pm, 1)))

    # Diagnostic info
    # print(
    #     f"{temp_C} °C | {humidity_percent}% | A: {int(is_active_hour)} | R: {transparency_risk[0]} | x{range_multiplier * env_factor * transparency_boost:.1f}")

    return calibrated_pm

if __name__ == '__main__':
    # ########################################################################################################
    # # # Step 4: Preprocess the combined data, delete blank columns and rename columns,
    # # output is processed_data.csv
    #
    # file_path = "data/rawdata/combined_all_data.csv"
    # df = pd.read_csv(file_path)
    #
    # columns_to_drop = [
    #     'pmnova_PC01', 'pmnova_PC03', 'pmnova_PC05',
    #     'pmnova_PC10', 'pmnova_PC100', 'pmnova_PC25', 'pmnova_PC50',
    #     'pmnova_PM01', 'pmnova_PM03', 'pmnova_PM05',
    #     'pmnova_PM100', 'pmnova_PM50', 'piera_IndoorHumidity',
    #     'piera_IndoorTemperature'
    # ]
    # df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    #
    # # Rename Unnamed: 0 to 'Timestamp', pmnova_IndoorTemperature to 'IndoorTemperature',
    # # pmnova_IndoorHumidity to 'IndoorHumidity'
    # df.rename(columns={
    #     '0': 'Timestamp',
    #     'pmnova_IndoorTemperature': 'IndoorTemperature',
    #     'pmnova_IndoorHumidity': 'IndoorHumidity'
    # }, inplace=True)
    #
    # # Convert 'Timestamp' to datetime
    # df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    # # Set 'Timestamp' as index
    # df.set_index('Timestamp', inplace=True)
    #
    # print(df.columns, "\n\n")
    # print(df.describe().to_string(), "\n\n")
    #
    # # Save the modified DataFrame to a new CSV file
    # output_file_path = "data/processed_data.csv"
    # df.to_csv(output_file_path, index=True)

    ######################################################################################################
    # # Step 5: Process the data, separate the data with and without df_PM10, df_PM1_0, df_PM2_5 columns,
    # output is processed_data_clean_without_df.csv and processed_data_clean_with_df.csv
    #
    # file_path = "data/processed_data.csv"
    # df = pd.read_csv(file_path)
    #
    # # Set Timestamp as index
    # df.set_index('Timestamp', inplace=True)
    #
    # # Convert index to datetime
    # df.index = pd.to_datetime(df.index, errors='coerce')
    #
    # df_with_df = df.copy()
    # df_without_df = df.copy()
    #
    # # Remove columns that contain df_PM10	df_PM1_0	df_PM2_5
    # df_without_df = df_without_df.loc[:, ~df_without_df.columns.str.contains('df_PM10|df_PM1_0|df_PM2_5')]
    #
    # # Drop Nan rows
    # df_with_df.dropna(inplace=True)
    # df_without_df.dropna(inplace=True)
    #
    # # Save the modified DataFrame to a new CSV file
    # output_file_path = "data/processed_data_clean_without_df.csv"
    # df_without_df.to_csv(output_file_path, index=True)
    #
    # output_file_path_with_df = "data/processed_data_clean_with_df.csv"
    # df_with_df.to_csv(output_file_path_with_df, index=True)

    ######################################################################################################
    # # Step 6: Select features, calibrate PM2.5 and PM0.1,
    # output is processed_data_selected_features_5sec.csv and processed_data_selected_features_1min.csv

    file = "data/processed_data_clean_without_df.csv"
    df = pd.read_csv(file, index_col='Timestamp', parse_dates=True)

    selected_columns = [
        'piera_PC01', 'piera_PC25', 'piera_PM25',
        'pmnova_PM25', 'piera_PC10', 'piera_PM10',
        'pmnova_PM10', 'IndoorHumidity', 'IndoorTemperature'
    ]

    df_selected = df[selected_columns].copy()

    # Calibrate PC0.1
    df_selected = calibrate_PC01(df_selected)

    # Calibrate PM2.5 for piera and pmnova
    for index, row in df_selected.iterrows():
        print(index)

        pm25_raw = row['pmnova_PM25']
        temp_C = row['IndoorTemperature']
        humidity_percent = row['IndoorHumidity']
        current_hour = index.hour

        calibrated_pm = calibrate_sds011_industrial_auto(pm25_raw, temp_C, humidity_percent, current_hour)
        df_selected.at[index, 'pmnova_PM25_calibrated'] = calibrated_pm

    print(df_selected.describe().to_string(), "\n\n")

    # Save the modified DataFrame to a new CSV file
    output_file_path = "data/processed_data_selected_features_5sec.csv"
    df_selected.to_csv(output_file_path, index=True)

    # Corr matrix
    corr_matrix = df_selected.corr()
    print("Correlation Matrix:\n", corr_matrix, "\n\n")

    # Plot the correlation matrix

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Matrix of Selected Features')
    plt.tight_layout()
    plt.show()
    plt.savefig('data/plots/correlation_matrix_selected_features.png')
    plt.close()

    # Resample the DataFrame to daily frequency, taking the mean of 1-min
    df_mean = df_selected.resample('1min').mean()
    df_mean['piera_PC25'] = df_selected['piera_PC25'].resample('1min').min()
    df_mean['piera_PM25'] = df_selected['piera_PM25'].resample('1min').min()
    df_mean['piera_PC10'] = df_selected['piera_PC10'].resample('1min').min()
    df_mean['piera_PM10'] = df_selected['piera_PM10'].resample('1min').min()
    df_mean['pmnova_PM25'] = df_selected['pmnova_PM25'].resample('1min').max()
    df_mean['pmnova_PM10'] = df_selected['pmnova_PM10'].resample('1min').max()
    df_mean['pmnova_PM25_calibrated'] = df_selected['pmnova_PM25_calibrated'].resample('1min').max()

    corr_matrix = df_mean.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Matrix of 1-min Mean Features')
    plt.tight_layout()
    plt.show()
    plt.savefig('data/plots/correlation_matrix_1min_mean_features.png')
    plt.close()

    # Save the modified DataFrame to a new CSV file
    output_file_path = "data/processed_data_selected_features_1min.csv"

    # Drop rows with NaN values
    df_mean.dropna(inplace=True)
    df_mean.to_csv(output_file_path, index=True)

    ######################################################################################################
    # # Step 7: Plot line graphs of piera_PM25, pmnova_PM25, and pmnova_PM25_calibrated in each day to review the data
    #
    # # Plot line graphs of piera_PM25, pmnova_PM25 in each day
    file_path = "data/processed_data_selected_features_1min.csv"
    df_mean = pd.read_csv(file_path, index_col='Timestamp', parse_dates=True)

    # Ensure the index is datetime
    df_mean.index = pd.to_datetime(df_mean.index, errors='coerce')
    df_mean.dropna(inplace=True)

    for day, group in df_mean.groupby(df_mean.index.date):
        plt.figure(figsize=(14, 7))
        group['piera_PM25'].plot(label='piera_PM25', color='blue')
        group['pmnova_PM25'].plot(label='pmnova_PM25', color='orange')
        group['pmnova_PM25_calibrated'].plot(label='pmnova_PM25_calibrated', color='green')
        plt.title(f'Daily PM2.5 Levels on {day}')
        plt.xlabel('Timestamp')
        plt.ylabel('PM2.5 Concentration')
        plt.legend()
        plt.grid()
        plt.show()
        #save picture in plots folder
        plt.savefig(f'data/plots/daily_pm25_levels_{day}.png')
        plt.close()


