import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

# Load the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


def predict(model, scalerX, scalerY):
    data = load_data('processed_data_selected_features_1min_cleaned(Completed).csv')
    data = data[['IndoorHumidity', 'IndoorTemperature', 'pmnova_PM25_calibrated','pmnova_PM10', 'daytype', 'Time', 'piera_PC01', 'time_range']]
    # Rename feature columns to match the training set
    data.rename(columns={'pmnova_PM25_calibrated': 'PM2.5', 'pmnova_PM10': 'PM10'}, inplace=True)

    # Label encode categorical variables
    data['daytype'] = data['daytype'].astype('category').cat.codes
    data['time_range'] = data['time_range'].astype('category').cat.codes

    # Humidity,Temperature,PM2.5,PM10,daytype,time_range
    x_col = ['IndoorHumidity', 'IndoorTemperature', 'PM2.5', 'PM10', 'daytype', 'time_range']
    data = data[x_col + ['piera_PC01']]

    # Scale the features
    data[x_col] = scalerX.transform(data[x_col])
    data['piera_PC01'] = scalerY.transform(data[['piera_PC01']])

    # Prepare features and target variable
    X = data.loc[:, x_col]
    y = data['piera_PC01']

    # Predict using the trained model
    y_pred = model.predict(X)

    # Scale back the predictions
    y_pred_scaled = scalerY.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y = scalerY.inverse_transform(y.values.reshape(-1, 1)).flatten()

    # RMSE, MAE, R^2
    rmse = root_mean_squared_error(y, y_pred_scaled)
    mae = mean_absolute_error(y, y_pred_scaled)
    r2 = r2_score(y, y_pred_scaled)
    print(f"RMSE: {rmse}, MAE: {mae}, R^2: {r2}")

if __name__ == "__main__":
    # Load the model and scalar from the previous run if available
    try:
        import joblib
        model = joblib.load('XGBoost Regression_model(PIERA).pkl')
        scalerX = joblib.load('XGBoost Regression_scaler_X(PIERA).pkl')
        scalerY = joblib.load('XGBoost Regression_scaler_y(PIERA).pkl')
    
        predict(model, scalerX, scalerY)
    except FileNotFoundError:
        model = None
        scaler = None
