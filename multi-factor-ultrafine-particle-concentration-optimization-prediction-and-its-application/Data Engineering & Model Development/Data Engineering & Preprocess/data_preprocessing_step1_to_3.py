import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path
import json
import re

if __name__ == "__main__":
    # ##################################################################################
    # # Step 1: Read all .json files in the data folder "data/rawdata" and save them as CSV files
    # # output is data/rawdata/csv/*.csv
    # # # Read all .json files in the data folder "data/rawdata"
    data_folder = Path("data/rawdata")
    data_files = list(data_folder.glob("*.json"))
    data_frames_piera = []
    data_frames_nova = []
    data_frames_dfrobot = []

    for file in data_files:
        print(f"Processing file: {file.name}")
        data = json.loads(Path(file).read_text())
        df = pd.DataFrame.from_dict(data, orient='index')

        # If there is a 'Timestamp' or 'timestamp' column, set it as index '2025-04-01 06:59:55'
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
            # Set to UTC timezone to Bangkok timezone
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Bangkok')
            df.rename(columns={'timestamp': 'Timestamp'}, inplace=True)
        else:
            print(f"No valid timestamp found in {file.name}. Skipping this file.")
            continue

        # Set the 'Timestamp' column as index
        df.set_index('Timestamp', inplace=True)

        # Save the DataFrame to a CSV file to "data/rawdata/csv"
        data_sub_folder = Path("data/rawdata/csv")
        output_file = data_sub_folder / f"{file.stem}.csv"

        df.to_csv(output_file)

    print("All JSON files have been processed and saved as CSV files.\n")
    #
    ##################################################################################
    # # Step 2: Read all .csv files in the data folder "data/rawdata/csv" and combine them into one DataFrame
    # # output is combined_piera_data.csv, combined_nova_data.csv, combined_dfrobot_data.csv
    # # # Read all .csv files in the data folder "data/rawdata/csv"
    # data_folder = Path("data/rawdata/csv")
    # data_files = list(data_folder.glob("*.csv"))
    #
    # data_frames_piera = []
    # data_frames_nova = []
    # data_frames_dfrobot = []
    #
    # for file in data_files:
    #     print(f"Processing file: {file.name}")
    #     df = pd.read_csv(file, index_col='Timestamp', parse_dates=True)
    #
    #     # Convert index to datetime if not already
    #     if not pd.api.types.is_datetime64_any_dtype(df.index):
    #         df.index = pd.to_datetime(df.index, errors='coerce')
    #
    #     prefix = re.split(r'(\d+)', file.stem, maxsplit=1)[0] + '_'
    #     df.columns = [prefix + col for col in df.columns]
    #
    #     # Check if the file.name contains specific prefixes
    #     if file.name.startswith('piera'):
    #         data_frames_piera.append(df)
    #     elif file.name.startswith('pmnova'):
    #         data_frames_nova.append(df)
    #     elif file.name.startswith('df'):
    #         data_frames_dfrobot.append(df)
    #
    # combined_data_frames_piera = pd.concat(data_frames_piera, axis=0)
    # print(combined_data_frames_piera.head().to_string(), '\n')
    # # replace '-' with '/' in the Timestamp column
    # output_file = Path("data/rawdata/combined_piera_data.csv")
    # combined_data_frames_piera.to_csv(output_file)
    # print(f"Combined data is saved to {output_file}")
    #
    # combined_data_frames_nova = pd.concat(data_frames_nova, axis=0)
    # print(combined_data_frames_nova.head().to_string(), '\n')
    # output_file = Path("data/rawdata/combined_nova_data.csv")
    # combined_data_frames_nova.to_csv(output_file)
    #
    # combined_data_frames_dfrobot = pd.concat(data_frames_dfrobot, axis=0)
    # print(combined_data_frames_dfrobot.head().to_string(), '\n')
    # output_file = Path("data/rawdata/combined_dfrobot_data.csv")
    # combined_data_frames_dfrobot.to_csv(output_file)

    ##################################################################################
    # Step 3: Read the combined CSV files and resample them to 5 seconds, then combine all dataframes into one
    # output is combined_all_data.csv that may contain NaN values
    # df_piera = pd.read_csv(
    #     Path("data/rawdata/combined_piera_data.csv"),
    #     index_col='Timestamp',
    #     parse_dates=True
    # )
    # df_nova = pd.read_csv(
    #     Path("data/rawdata/combined_nova_data.csv"),
    #     index_col='Timestamp',
    #     parse_dates=True
    # )
    # df_dfrobot = pd.read_csv(
    #     Path("data/rawdata/combined_dfrobot_data.csv"),
    #     index_col='Timestamp',
    #     parse_dates=True
    # )
    #
    # # # Convert datetime format
    # df_nova.index = df_nova.index.astype(str).str.extract(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')[0]
    # df_nova.index = df_nova.index.str.replace('-', '/', regex=False)
    #
    # df_dfrobot.index = df_dfrobot.index.astype(str).str.extract(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')[0]
    # df_dfrobot.index = df_dfrobot.index.str.replace('-', '/', regex=False)
    #
    # df_piera.index = df_piera.index.astype(str).str.extract(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')[0]
    # df_piera.index = df_piera.index.str.replace('-', '/', regex=False)
    #
    # df_nova.index = pd.to_datetime(df_nova.index, format='%Y/%m/%d %H:%M:%S', errors='coerce')
    # df_dfrobot.index = pd.to_datetime(df_dfrobot.index, format='%Y/%m/%d %H:%M:%S', errors='coerce')
    # df_piera.index = pd.to_datetime(df_piera.index, format='%Y/%m/%d %H:%M:%S', errors='coerce')
    #
    # print(df_piera.head())
    # print(df_nova.head())
    # print(df_dfrobot.head())
    #
    # df_piera = df_piera.resample('5s').mean()
    # df_nova = df_nova.resample('5s').mean()
    # df_dfrobot = df_dfrobot.resample('5s').mean()
    #
    # # Combine all dataframes into one
    # combined_df = pd.concat([df_piera, df_nova, df_dfrobot], axis=1)
    #
    # print(combined_df.head().to_string(), '\n')
    #
    # # Save the combined DataFrame to a CSV file
    # output_file = Path("data/rawdata/combined_all_data.csv")
    # combined_df.to_csv(output_file)
    #
