import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
from optuna.trial import TrialState
import warnings
import json
from pathlib import Path

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class PMPredictionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate, rnn_type='lstm'):
        super(PMPredictionRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                              batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        else:
            self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                              batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        if self.rnn_type == 'lstm':
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)

        out = self.dropout(out[:, -1, :])  # Get the last time step output
        out = self.fc(out)
        return out


def prepare_data(file_path, sequence_length=24):
    raw_df = pd.read_csv(file_path, index_col='Time', parse_dates=True)
    raw_df.index = pd.to_datetime(raw_df.index, errors='coerce')

    cat_cols = ['daytype', 'time_range']
    for col in cat_cols:
        raw_df[col] = raw_df[col].astype('category').cat.codes

    log_columns = ['piera_PC01', 'piera_PM25', 'piera_PM10']
    for col in log_columns:
        raw_df[col] = np.log1p(raw_df[col])

    features = ['piera_PM25', 'piera_PM10', 'IndoorHumidity', 'IndoorTemperature'] + cat_cols
    target = 'piera_PC01'
    df = raw_df[features + [target]].copy()

    data = df.values
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length, :-1])
        targets.append(data[i + sequence_length, -1])

    return np.array(sequences), np.array(targets).reshape(-1, 1), df.shape[1] - 1


def calculate_metrics(true, pred):
    true = np.expm1(true)
    pred = np.expm1(pred)
    return {
        'mae': mean_absolute_error(true, pred),
        'rmse': np.sqrt(mean_squared_error(true, pred)),
        'r2': r2_score(true, pred)
    }


def save_trial_results(study, file_path):
    """Save all trial results to a CSV file."""
    trials = study.trials
    results = []

    for trial in trials:
        if trial.state != TrialState.COMPLETE:
            continue

        row = {
            'trial_number': trial.number,
            'value': trial.value,
            'params': json.dumps(trial.params),
            'train_mae': trial.user_attrs['train_metrics']['mae'],
            'train_rmse': trial.user_attrs['train_metrics']['rmse'],
            'train_r2': trial.user_attrs['train_metrics']['r2'],
            'val_mae': trial.user_attrs['val_metrics']['mae'],
            'val_rmse': trial.user_attrs['val_metrics']['rmse'],
            'val_r2': trial.user_attrs['val_metrics']['r2'],
            'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
            'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None
        }
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(file_path, index=False)

def objective(trial, sequences, targets, input_size):
    params = {
        'sequence_length': trial.suggest_int('sequence_length', 6, 48, step=6),
        'hidden_size': trial.suggest_int('hidden_size', 32, 256, step=32),
        'num_layers': trial.suggest_int('num_layers', 1, 3),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'rnn_type': trial.suggest_categorical('rnn_type', ['lstm', 'gru', 'rnn']),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    }

    tscv = TimeSeriesSplit(n_splits=5)
    fold_metrics = {
        'train': {'mae': [], 'rmse': [], 'r2': []},
        'val': {'mae': [], 'rmse': [], 'r2': []}
    }

    for fold, (train_idx, val_idx) in enumerate(tscv.split(sequences)):

        if len(train_idx) < params['sequence_length'] or len(val_idx) < params['sequence_length']:
            continue

        train_seq, train_tgt = sequences[train_idx], targets[train_idx]
        val_seq, val_tgt = sequences[val_idx], targets[val_idx]

        train_dataset = TensorDataset(torch.FloatTensor(train_seq), torch.FloatTensor(train_tgt))
        val_dataset = TensorDataset(torch.FloatTensor(val_seq), torch.FloatTensor(val_tgt))

        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PMPredictionRNN(
            input_size=input_size,
            hidden_size=params['hidden_size'],
            output_size=1,
            num_layers=params['num_layers'],
            dropout_rate=params['dropout_rate'],
            rnn_type=params['rnn_type']
        ).to(device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0

        # Training loop with metrics tracking
        for epoch in range(300):
            model.train()
            train_preds, train_true = [], []
            train_loss = 0
            for batch_seq, batch_tgt in train_loader:
                batch_seq, batch_tgt = batch_seq.to(device), batch_tgt.to(device)
                optimizer.zero_grad()
                outputs = model(batch_seq)
                loss = criterion(outputs, batch_tgt)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_preds.extend(outputs.detach().cpu().numpy())
                train_true.extend(batch_tgt.cpu().numpy())

            # Calculate training metrics
            train_metrics = calculate_metrics(np.array(train_true), np.array(train_preds))

            # Validation
            model.eval()
            val_preds, val_true = [], []
            val_loss = 0
            with torch.no_grad():
                for batch_seq, batch_tgt in val_loader:
                    batch_seq, batch_tgt = batch_seq.to(device), batch_tgt.to(device)
                    outputs = model(batch_seq)
                    val_loss += criterion(outputs, batch_tgt).item()
                    val_preds.extend(outputs.cpu().numpy())  # Explicitly move to CPU
                    val_true.extend(batch_tgt.cpu().numpy())  # Explicitly move to CPU

            val_metrics = calculate_metrics(np.array(val_true), np.array(val_preds))

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_train_metrics = train_metrics
                best_val_metrics = val_metrics
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Store best metrics for this fold
        for metric in ['mae', 'rmse', 'r2']:
            fold_metrics['train'][metric].append(best_train_metrics[metric])
            fold_metrics['val'][metric].append(best_val_metrics[metric])

    # Calculate average metrics across folds
    avg_metrics = {
        'train': {k: np.mean(v) for k, v in fold_metrics['train'].items()},
        'val': {k: np.mean(v) for k, v in fold_metrics['val'].items()}
    }

    # Store metrics in trial user attributes
    trial.set_user_attr("train_metrics", avg_metrics['train'])
    trial.set_user_attr("val_metrics", avg_metrics['val'])

    return avg_metrics['val']['mae']


def main():
    file_path = 'processed_data_selected_features_1min_cleaned(Completed).csv'
    sequences, targets, input_size = prepare_data(file_path)

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
    )

    study.optimize(
        lambda trial: objective(trial, sequences, targets, input_size),
        n_trials=50,
        n_jobs=-1,
        show_progress_bar=True
    )

    # Save study results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Save all trial results to CSV
    save_trial_results(study, results_dir / "all_trial_results.csv")

    # Save best trial info
    best_trial = study.best_trial
    best_params = best_trial.params
    best_metrics = {
        'train': best_trial.user_attrs['train_metrics'],
        'val': best_trial.user_attrs['val_metrics']
    }

    with open(results_dir / "best_trial.json", "w") as f:
        json.dump({
            'params': best_params,
            'metrics': best_metrics
        }, f, indent=4)

    print("\nBest trial results:")
    print(f"Validation MAE: {best_metrics['val']['mae']:.4f}")
    print(f"Validation RMSE: {best_metrics['val']['rmse']:.4f}")
    print(f"Validation R2: {best_metrics['val']['r2']:.4f}")
    print(f"\nTraining MAE: {best_metrics['train']['mae']:.4f}")
    print(f"Training RMSE: {best_metrics['train']['rmse']:.4f}")
    print(f"Training R2: {best_metrics['train']['r2']:.4f}")

    # Train final model with best params
    sequences, targets, input_size = prepare_data(file_path, best_params['sequence_length'])

    # Split into train and test (80-20)
    split_idx = int(0.8 * len(sequences))
    train_seq, test_seq = sequences[:split_idx], sequences[split_idx:]
    train_tgt, test_tgt = targets[:split_idx], targets[split_idx:]

    train_dataset = TensorDataset(torch.FloatTensor(train_seq), torch.FloatTensor(train_tgt))
    test_dataset = TensorDataset(torch.FloatTensor(test_seq), torch.FloatTensor(test_tgt))

    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PMPredictionRNN(
        input_size=input_size,
        hidden_size=best_params['hidden_size'],
        output_size=1,
        num_layers=best_params['num_layers'],
        dropout_rate=best_params['dropout_rate'],
        rnn_type=best_params['rnn_type']
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )
    criterion = nn.MSELoss()

    # Training loop with metrics tracking
    best_model = None
    best_val_loss = float('inf')
    train_metrics_history = []
    val_metrics_history = []

    for epoch in range(100):
        model.train()
        train_preds, train_true = [], []
        train_loss = 0
        for batch_seq, batch_tgt in train_loader:
            batch_seq, batch_tgt = batch_seq.to(device), batch_tgt.to(device)
            optimizer.zero_grad()
            outputs = model(batch_seq)
            loss = criterion(outputs, batch_tgt)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy())  # Move to CPU
            train_true.extend(batch_tgt.cpu().numpy())  # Move to CPU

        train_metrics = calculate_metrics(np.array(train_true), np.array(train_preds))
        train_metrics_history.append(train_metrics)

        # Validation on test set
        model.eval()
        test_preds, test_true = [], []
        test_loss = 0
        with torch.no_grad():
            for batch_seq, batch_tgt in test_loader:
                batch_seq, batch_tgt = batch_seq.to(device), batch_tgt.to(device)
                outputs = model(batch_seq)
                test_loss += criterion(outputs, batch_tgt).item()
                test_preds.extend(outputs.cpu().numpy())  # Move to CPU
                test_true.extend(batch_tgt.cpu().numpy())  # Move to CPU

        test_metrics = calculate_metrics(np.array(test_true), np.array(test_preds))
        val_metrics_history.append(test_metrics)

        print(f"\nEpoch {epoch + 1}:")
        print(f"Train - MAE: {train_metrics['mae']:.4f}, RMSE: {train_metrics['rmse']:.4f}, R2: {train_metrics['r2']:.4f}")
        print(f"Test - MAE: {test_metrics['mae']:.4f}, RMSE: {test_metrics['rmse']:.4f}, R2: {test_metrics['r2']:.4f}")

        # Save best model
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            best_model = model.state_dict()

    # Save final metrics and model
    final_metrics = {
        'train': train_metrics_history[-1],
        'test': val_metrics_history[-1],
        'best_val_loss': best_val_loss,
        'history': {
            'train': train_metrics_history,
            'test': val_metrics_history
        }
    }

    # Save metrics history to CSV files
    train_metrics_df = pd.DataFrame(train_metrics_history)
    val_metrics_df = pd.DataFrame(val_metrics_history)
    train_metrics_df.to_csv(results_dir / "final_train_metrics_history.csv", index=False)
    val_metrics_df.to_csv(results_dir / "final_val_metrics_history.csv", index=False)

    with open(results_dir / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=4)

    torch.save(best_model, results_dir / "best_pm_prediction_model.pth")
    print("\nFinal model and metrics saved in 'results' directory")


if __name__ == "__main__":
    main()