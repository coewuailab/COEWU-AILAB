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

def classify_pm(data):
    # Classify PM2.5 and PM10 into categories
    data = pd.cut(data, bins=[0, 1000, 10000, 20000, np.inf], labels=[1, 2, 3, 4])
    return data

def train_model():
    data = load_data('processed_data_selected_features_1min_cleaned(Completed).csv')
    # print(data[['PM10nova', 'PM10']].describe().to_string())
    data = data.rename(columns={'piera_PM25': 'PM2.5', 'piera_PM10': 'PM10'})
    x_col = ['IndoorHumidity', 'IndoorTemperature', 'PM2.5', 'PM10', 'daytype', 'time_range']

    # Label encode categorical variables
    data['daytype'] = data['daytype'].astype('category').cat.codes
    data['time_range'] = data['time_range'].astype('category').cat.codes

    # Select the relevant columns for regression in data
    data = data[x_col + ['piera_PC01']]
    print(data.describe().to_string())


    # Drop rows with NaN values in the target variable
    # MinMax scale the numerical features and target variable
    scalerX = MinMaxScaler()
    scalerY = MinMaxScaler()
    data[x_col] = scalerX.fit_transform(data[x_col])
    data['piera_PC01'] = scalerY.fit_transform(data[['piera_PC01']])

    # Prepare features and target variable
    X = data.loc[:, x_col]
    y = data['piera_PC01']

    # Split the dataset into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train = X
    y_train = y
    X_test = X
    y_test = y
    print(X_train.head().to_string())

    models = {
        'Linear Regression': LinearRegression(),
        'XGBoost Regression': XGBRegressor(n_estimators=100, random_state=42),
        'Random Forest Regression': RandomForestRegressor(n_estimators=100, random_state=42),
        'MLP Regression': MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
        'K-Neighbors Regression': KNeighborsRegressor(n_neighbors=5),
        'Gradient Boosting Regression': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        # Fit the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Scale back the predictions
        y_pred_scaled = scalerY.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_test_scaled = scalerY.inverse_transform(y_test.values.reshape(-1, 1)).flatten()

        # Evaluate the model
        rmse = root_mean_squared_error(y_test_scaled, y_pred_scaled)
        r2 = r2_score(y_test_scaled, y_pred_scaled)
        mae = mean_absolute_error(y_test_scaled, y_pred_scaled)

        print(f"{model_name} RMSE: {rmse}, R^2: {r2}", f"MAE: {mae}")

        # Classify y_pred_scaled and y_test_scaled
        y_pred_classified = classify_pm(y_pred_scaled)
        y_test_classified = classify_pm(y_test_scaled)

        # Evaluate classification accuracy
        accuracy = np.mean(y_test_classified == y_pred_classified)
        print(f"{model_name} Classification Accuracy: {accuracy}")
        
        # # Save x, y_test_scaled, y_pred to CSV for further analysis
        results_df = pd.DataFrame({
            'y_test_scaled': y_test_scaled,
            'y_pred_scaled': y_pred_scaled,
            'y_test_classified': y_test_classified,
            'y_pred_classified': y_pred_classified
        })
        results_df.to_csv(f'{model_name}_results(PIERA).csv', index=False)

        # Save the model (PTH file)
        import joblib
        joblib.dump(model, f'{model_name}_model(PIERA).pkl')
        # Save the scaler
        joblib.dump(scalerX, f'{model_name}_scaler_X(PIERA).pkl')
        joblib.dump(scalerY, f'{model_name}_scaler_y(PIERA).pkl')

        # Save feature names and order
        with open(f'{model_name}_features(PIERA).txt', 'w') as f:
            f.write(','.join(x_col))

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
    # Train the model
    train_model()





