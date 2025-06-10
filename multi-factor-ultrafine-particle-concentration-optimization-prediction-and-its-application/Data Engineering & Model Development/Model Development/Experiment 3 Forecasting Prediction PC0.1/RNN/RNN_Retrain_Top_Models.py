"""
ปรับปรุงโค้ดสำหรับการเทรนซ้ำโมเดล RNN และลดความซับซ้อนของโครงสร้างไฟล์
โดยปรับให้มีโครงสร้างตามที่ต้องการ
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
from datetime import datetime
import copy
import shutil
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from tqdm import tqdm

# ตั้งค่า seed สำหรับการทำให้ผลลัพธ์คงที่
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

# ตรวจสอบอุปกรณ์ที่ใช้ (CPU หรือ GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"ใช้อุปกรณ์: {device}")

# นำเข้าคลาสและฟังก์ชันที่จำเป็นจากไฟล์เดิม
# สมมติว่า import จากไฟล์เดิมที่ชื่อ RNN.py
from RnnModelBaysian import (
    RNNModel, EarlyStopping, prepare_data_expanding_window,
    create_expanding_window_split, train_and_evaluate_model_expanding,
    plot_results_expanding, plot_combined_results, check_gpu_usage, check_model_on_device,
    default_converter, MAX_EXPANDING_WINDOWS, EARLY_STOP_PATIENCE
)

def plot_average_losses(all_windows_results, model_name, plots_dir, hyperparams, show_plot=False):
    """
    พล็อตกราฟค่าเฉลี่ยของ Training Loss และ Validation Loss จากทุก window
    """
    # ตรวจสอบว่ามีข้อมูล loss หรือไม่
    if not all_windows_results or 'train_losses' not in all_windows_results[0]:
        print("ไม่พบข้อมูล loss สำหรับการพล็อต")
        return None

    # หาความยาวสูงสุดของ epochs จากทุก window
    max_epochs = max([len(result['train_losses']) for result in all_windows_results])

    # เตรียมอาเรย์สำหรับเก็บค่าเฉลี่ย
    avg_train_losses = np.zeros(max_epochs)
    avg_val_losses = np.zeros(max_epochs)
    counts = np.zeros(max_epochs)

    # รวมค่า loss จากทุก window
    for result in all_windows_results:
        train_losses = result['train_losses']
        val_losses = result['val_losses']
        n_epochs = len(train_losses)

        # รวมค่า loss และนับจำนวน
        for i in range(n_epochs):
            avg_train_losses[i] += train_losses[i]
            avg_val_losses[i] += val_losses[i]
            counts[i] += 1

    # คำนวณค่าเฉลี่ย
    for i in range(max_epochs):
        if counts[i] > 0:
            avg_train_losses[i] /= counts[i]
            avg_val_losses[i] /= counts[i]

    # ใช้เฉพาะค่าที่มีข้อมูล (counts > 0)
    valid_epochs = np.where(counts > 0)[0]
    if valid_epochs.size == 0:
        print("ไม่พบข้อมูลสำหรับการพล็อต")
        return None

    max_valid_epoch = valid_epochs[-1] + 1

    # ตัดข้อมูลให้ใช้เฉพาะช่วงที่มีค่า
    avg_train_losses = avg_train_losses[:max_valid_epoch]
    avg_val_losses = avg_val_losses[:max_valid_epoch]
    epochs = np.arange(1, max_valid_epoch + 1)

    # สร้างกราฟ
    plt.figure(figsize=(15, 10))

    # พล็อตค่าเฉลี่ย losses
    plt.plot(epochs, avg_train_losses, 'b-', linewidth=2, label='Avg. Training Loss')
    plt.plot(epochs, avg_val_losses, 'r-', linewidth=2, label='Avg. Validation Loss')

    plt.title(f'Average Training and Validation Losses Across All Windows', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=12)

    # ข้อมูลเพิ่มเติม
    info_text = (
        f"Model: {model_name}\n"
        f"Windows: {len(all_windows_results)}\n"
        f"Hyperparameters:\n"
        f"  Sequence Length: {hyperparams['sequence_length']}\n"
        f"  Hidden Size: {hyperparams['hidden_size']}\n"
        f"  Num Layers: {hyperparams['num_layers']}\n"
        f"  Dropout Rate: {hyperparams['dropout_rate']:.4f}\n"
        f"  Learning Rate: {hyperparams['learning_rate']:.6f}\n"
        f"  Optimizer: {hyperparams.get('optimizer_type', 'Adam')}"
    )

    plt.figtext(0.02, 0.02, info_text, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.7))

    plt.tight_layout()

    # บันทึกรูปภาพ
    plot_path = os.path.join(plots_dir, f"{model_name}_average_losses.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()

    return plot_path

class RetrainTopModels:
    """
    คลาสสำหรับการเทรนโมเดล RNN ที่ดีที่สุดซ้ำหลายรอบ
    เพื่อวิเคราะห์ความเสถียรของโมเดลและความแปรปรวนของผลลัพธ์
    ปรับปรุงโครงสร้างไฟล์ให้มีความเรียบง่ายมากขึ้น
    """

    def __init__(self,
                 top_models_summary_path,
                 data_path,
                 output_dir=None,
                 num_retrain=10,
                 num_top_models=10):
        """
        ตั้งค่าเริ่มต้นสำหรับการเทรนซ้ำ

        Args:
            top_models_summary_path (str): พาธของไฟล์ all_models_summary.json
            data_path (str): พาธของไฟล์ CSV ที่ใช้ในการเทรน
            output_dir (str, optional): โฟลเดอร์สำหรับเก็บผลลัพธ์
            num_retrain (int): จำนวนรอบที่ต้องการเทรนซ้ำแต่ละโมเดล
            num_top_models (int): จำนวนโมเดลที่ดีที่สุดที่ต้องการเทรนซ้ำ
        """
        self.top_models_summary_path = top_models_summary_path
        self.data_path = data_path
        self.num_retrain = num_retrain
        self.num_top_models = num_top_models

        # กำหนดโฟลเดอร์สำหรับเก็บผลลัพธ์
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"PM01_RNN_Retrain_Top{num_top_models}_{timestamp}"
        else:
            self.output_dir = output_dir

        # สร้างโฟลเดอร์สำหรับเก็บผลลัพธ์
        os.makedirs(self.output_dir, exist_ok=True)

        # โหลดข้อมูลโมเดลที่ดีที่สุด
        with open(top_models_summary_path, 'r') as f:
            self.top_models_info = json.load(f)

        # เตรียมข้อมูลสำหรับการเทรน
        self.prepare_data()

    def prepare_data(self):
        """
        เตรียมข้อมูลพื้นฐานสำหรับการเทรนโมเดล
        """
        # กำหนดคุณลักษณะที่ใช้ในการเทรน
        self.numeric_features = ['PM2.5nova', 'PM10nova', 'Humidity', 'Temperature']
        self.categorical_features = ['daytype', 'time_range']

        # กำหนดค่าคงที่สำหรับการเทรนโมเดล
        self.fixed_params = {
            'output_size': 1  # ค่าที่ต้องการทำนาย (PM0.1)
        }

        # โหลดข้อมูลดิบ
        self.raw_data = pd.read_csv(self.data_path)
        print(f"โหลดข้อมูลเรียบร้อย: {self.raw_data.shape} แถว")

        # สำหรับตัวแปรที่จำเป็นต้องใช้ในระดับคลาส ให้กำหนดค่าเริ่มต้น
        self.input_size = None
        self.numeric_scaler = None
        self.label_encoders = None
        self.y_scaler = None

    def retrain_all_models(self):
        """
        เทรนโมเดลที่ดีที่สุดซ้ำหลายรอบ
        """
        # เลือกโมเดลที่ดีที่สุดตามจำนวนที่กำหนด
        top_models = self.top_models_info['models'][:self.num_top_models]

        # เตรียมตัวแปรสำหรับเก็บผลลัพธ์
        all_retrain_results = []

        # วนลูปเทรนแต่ละโมเดล
        for model_idx, model_info in enumerate(top_models):
            model_rank = model_idx + 1
            print(f"\n{'=' * 50}")
            print(f"กำลังเทรนโมเดลอันดับที่ {model_rank}/{self.num_top_models}")
            print(f"{'=' * 50}")

            # สร้างโฟลเดอร์สำหรับเก็บผลลัพธ์ของโมเดลนี้
            model_dir = os.path.join(self.output_dir, f"model_rank_{model_rank}")
            os.makedirs(model_dir, exist_ok=True)

            # สร้างโฟลเดอร์สำหรับเก็บผลการเทรน
            plots_dir = os.path.join(model_dir, "plots")
            retrain_logs_dir = os.path.join(model_dir, "retrain_logs")
            retrain_models_dir = os.path.join(model_dir, "retrain_models")

            os.makedirs(plots_dir, exist_ok=True)
            os.makedirs(retrain_logs_dir, exist_ok=True)
            os.makedirs(retrain_models_dir, exist_ok=True)

            # เทรนโมเดลซ้ำตามจำนวนที่กำหนด
            model_results = self.retrain_single_model(model_info, model_rank, model_dir)

            # เก็บผลลัพธ์
            all_retrain_results.append({
                'model_rank': model_rank,
                'model_info': model_info,
                'retrain_results': model_results,
                'model_dir': model_dir
            })

            # บันทึกผลลัพธ์ของโมเดลนี้
            self.save_model_results(model_results, model_info, model_rank, model_dir)

        # บันทึกผลลัพธ์ทั้งหมด
        self.create_custom_summary(all_retrain_results)

        return all_retrain_results

    def retrain_single_model(self, model_info, model_rank, model_dir):
        """
        เทรนโมเดลที่กำหนดซ้ำหลายรอบ

        Args:
            model_info (dict): ข้อมูลของโมเดล
            model_rank (int): อันดับของโมเดล
            model_dir (str): โฟลเดอร์สำหรับเก็บผลลัพธ์

        Returns:
            list: ผลลัพธ์การเทรนซ้ำแต่ละรอบ
        """
        # ดึง hyperparameters ของโมเดล
        hyperparams = model_info['hyperparams']

        # แสดง hyperparameters ก่อนเริ่มเทรน
        print(f"\n{'=' * 80}")
        print(f"Hyperparameters ของโมเดลอันดับที่ {model_rank}:")
        print(f"{'=' * 80}")
        print(f"Sequence Length: {hyperparams['sequence_length']}")
        print(f"Hidden Size: {hyperparams['hidden_size']}")
        print(f"Number of Layers: {hyperparams['num_layers']}")
        print(f"Dropout Rate: {hyperparams['dropout_rate']:.4f}")
        print(f"Batch Size: {hyperparams['batch_size']}")
        print(f"Learning Rate: {hyperparams['learning_rate']:.6f}")
        print(f"Optimizer: {hyperparams.get('optimizer_type', 'Adam')}")
        print(f"Number of Epochs: {hyperparams['num_epochs']}")
        print(f"{'=' * 80}\n")

        # เตรียมข้อมูลตาม sequence length ของโมเดลนี้
        print(f"กำลังเตรียมข้อมูลโดยใช้ sequence_length: {hyperparams['sequence_length']}")
        data_args = prepare_data_expanding_window(
            self.data_path,
            self.numeric_features,
            self.categorical_features,
            hyperparams['sequence_length'],
            initial_train_ratio=0.3,
            val_ratio=0.1,
            expand_step=0.1
        )

        # แยกข้อมูลจาก data_args
        (X_encoded, y_scaled, y_raw, numeric_scaler,
         label_encoders, y_scaler, input_size, all_data) = data_args

        # สร้างโฟลเดอร์สำหรับเก็บผลลัพธ์แต่ละรอบ
        retrain_logs_dir = os.path.join(model_dir, "retrain_logs")
        retrain_models_dir = os.path.join(model_dir, "retrain_models")
        plots_dir = os.path.join(model_dir, "plots")

        os.makedirs(retrain_logs_dir, exist_ok=True)
        os.makedirs(retrain_models_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        # เก็บผลลัพธ์ของแต่ละรอบ
        retrain_results = []

        # วนลูปเทรนซ้ำตามจำนวนที่กำหนด
        for retrain_idx in range(self.num_retrain):
            print(f"\nกำลังเทรนโมเดลอันดับที่ {model_rank} รอบที่ {retrain_idx + 1}/{self.num_retrain}")

            # กำหนด seed แบบสุ่มสำหรับแต่ละรอบ (แต่แบบที่ทำซ้ำได้)
            retrain_seed = RANDOM_SEED + retrain_idx
            torch.manual_seed(retrain_seed)
            np.random.seed(retrain_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(retrain_seed)

            # เทรนโมเดลสำหรับรอบนี้
            retrain_result = self.train_model_once(
                hyperparams, model_rank, retrain_idx + 1,
                retrain_models_dir, retrain_logs_dir, plots_dir,
                all_data, input_size, y_scaler  # ส่งข้อมูลและตัวแปรที่เกี่ยวข้องเพิ่มเติม
            )

            retrain_results.append(retrain_result)

        return retrain_results

    def train_model_once(self, hyperparams, model_rank, retrain_idx, models_dir, logs_dir, plots_dir,
                         all_data, input_size, y_scaler):
        """
        เทรนโมเดลหนึ่งรอบด้วย hyperparameters ที่กำหนด

        Args:
            hyperparams (dict): พารามิเตอร์ของโมเดล
            model_rank (int): อันดับของโมเดล
            retrain_idx (int): รอบที่เทรนซ้ำ
            models_dir (str): โฟลเดอร์สำหรับเก็บโมเดล
            logs_dir (str): โฟลเดอร์สำหรับเก็บ log
            plots_dir (str): โฟลเดอร์สำหรับเก็บกราฟ
            all_data (tuple): ข้อมูลทั้งหมดที่เตรียมด้วย sequence length เฉพาะของโมเดลนี้
            input_size (int): ขนาดของ input
            y_scaler (object): ตัวแปลงสเกลของ target

        Returns:
            dict: ผลลัพธ์การเทรน
        """
        # แสดงรายละเอียดการเทรนแต่ละรอบ
        print(f"\n{'-' * 80}")
        print(f"เทรนโมเดลอันดับที่ {model_rank} รอบที่ {retrain_idx}")
        print(f"Seed: {RANDOM_SEED + retrain_idx - 1}")
        print(f"เริ่มเทรน: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'-' * 80}")

        # สร้างชื่อโมเดล
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"RNN_Rank{model_rank}_Retrain{retrain_idx}_{timestamp}"

        # เทรนโมเดลด้วยพารามิเตอร์นี้
        all_windows_results = []
        total_training_time = 0

        for window_idx in range(MAX_EXPANDING_WINDOWS):
            # สร้าง data loader สำหรับแต่ละหน้าต่าง
            train_loader, val_loader, test_loader, data_sizes = create_expanding_window_split(
                all_data, window_idx, hyperparams['sequence_length'], hyperparams['batch_size']
            )

            # ตรวจสอบว่ายังมีข้อมูลเพียงพอหรือไม่
            if train_loader is None:
                print(f"ไม่มีข้อมูลเพียงพอสำหรับหน้าต่างที่ {window_idx + 1}")
                break

            # สร้างโมเดล RNN
            model = RNNModel(
                input_size,  # ใช้ input_size ที่ได้จากการเตรียมข้อมูลเฉพาะของโมเดลนี้
                hyperparams['hidden_size'],
                hyperparams['num_layers'],
                hyperparams['output_size'],
                hyperparams['dropout_rate']
            ).to(device)

            # สร้าง optimizer ตามประเภทที่กำหนด
            if 'optimizer_type' in hyperparams:
                if hyperparams['optimizer_type'] == 'Adam':
                    optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
                elif hyperparams['optimizer_type'] == 'AdamW':
                    optimizer = optim.AdamW(model.parameters(), lr=hyperparams['learning_rate'])
                elif hyperparams['optimizer_type'] == 'SGD':
                    optimizer = optim.SGD(model.parameters(), lr=hyperparams['learning_rate'], momentum=0.9)
                else:
                    # ค่าเริ่มต้น
                    optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
            else:
                # กรณีไม่มีค่า optimizer_type ในพารามิเตอร์ (ใช้กับโมเดลเก่า)
                optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])

            criterion = nn.MSELoss()

            # เทรนและประเมินโมเดล - ใช้ y_scaler แทน self.y_scaler
            (early_stopping, train_losses, val_losses,
             train_metrics, val_metrics, test_metrics, training_time,
             train_predictions, train_actuals, val_predictions, val_actuals,
             test_predictions, test_actuals, data_sizes, stopped_epoch) = train_and_evaluate_model_expanding(
                model, train_loader, val_loader, test_loader, criterion, optimizer,
                hyperparams['num_epochs'], EARLY_STOP_PATIENCE,
                data_sizes, y_scaler, window_idx  # เปลี่ยนจาก self.y_scaler เป็น y_scaler
            )

            # เก็บผลลัพธ์
            window_result = {
                'window_index': window_idx,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'data_sizes': data_sizes,
                'train_predictions': train_predictions.tolist() if isinstance(train_predictions, np.ndarray) else [],
                'train_actuals': train_actuals.tolist() if isinstance(train_actuals, np.ndarray) else [],
                'val_predictions': val_predictions.tolist() if isinstance(val_predictions, np.ndarray) else [],
                'val_actuals': val_actuals.tolist() if isinstance(val_actuals, np.ndarray) else [],
                'test_predictions': test_predictions.tolist() if isinstance(test_predictions, np.ndarray) else [],
                'test_actuals': test_actuals.tolist() if isinstance(test_actuals, np.ndarray) else [],
                'stopped_epoch': stopped_epoch
            }

            all_windows_results.append(window_result)
            total_training_time += training_time

            # พล็อตค่าเฉลี่ย losses
            if len(all_windows_results) > 0:
                avg_losses_plot_path = plot_average_losses(
                    all_windows_results, f"model_round{retrain_idx}", plots_dir, hyperparams, show_plot=False
                )
            else:
                avg_losses_plot_path = None

            # บันทึกโมเดลและ log
            # บันทึกโมเดล
            model_filename = f"RNN_Rank{model_rank}_Retrain{retrain_idx}.pth"
            model_path = os.path.join(models_dir, model_filename)
            torch.save(model.state_dict(), model_path)

            # บันทึก log
            json_log_filename = f"retrain_{retrain_idx}_log.json"
            json_log_file = os.path.join(logs_dir, json_log_filename)

            # คำนวณค่าเฉลี่ยของเมตริกทั้งหมด
            avg_train_metrics = {
                metric: float(np.mean([r['train_metrics'][metric] for r in all_windows_results]))
                for metric in ['mse', 'rmse', 'mae', 'r2']
            }

            avg_val_metrics = {
                metric: float(np.mean([r['val_metrics'][metric] for r in all_windows_results]))
                for metric in ['mse', 'rmse', 'mae', 'r2']
            }

            avg_test_metrics = {
                metric: float(np.mean([r['test_metrics'][metric] for r in all_windows_results]))
                for metric in ['mse', 'rmse', 'mae', 'r2']
            }

            # คำนวณค่าเฉลี่ยเมตริกสำหรับการแสดงผล
            if len(all_windows_results) > 0:
                avg_test_rmse = np.mean([r['test_metrics']['rmse'] for r in all_windows_results])
                avg_test_mae = np.mean([r['test_metrics']['mae'] for r in all_windows_results])
                avg_test_r2 = np.mean([r['test_metrics']['r2'] for r in all_windows_results])

                # คำนวณ SD ของเมตริก
                std_test_rmse = np.std([r['test_metrics']['rmse'] for r in all_windows_results])
                std_test_mae = np.std([r['test_metrics']['mae'] for r in all_windows_results])
                std_test_r2 = np.std([r['test_metrics']['r2'] for r in all_windows_results])
            else:
                avg_test_rmse = float('nan')
                avg_test_mae = float('nan')
                avg_test_r2 = float('nan')
                std_test_rmse = float('nan')
                std_test_mae = float('nan')
                std_test_r2 = float('nan')

            # สร้างข้อมูลสำหรับบันทึกลงไฟล์ log
            log_data = {
                "model_name": model_name,
                "expanding_window": True,
                "hyperparameters": {
                    "input_size": input_size,
                    "output_size": hyperparams['output_size'],
                    "hidden_size": hyperparams['hidden_size'],
                    "num_layers": hyperparams['num_layers'],
                    "dropout_rate": hyperparams['dropout_rate'],
                    "batch_size": hyperparams['batch_size'],
                    "learning_rate": hyperparams['learning_rate'],
                    "num_epochs": hyperparams['num_epochs'],
                    "sequence_length": hyperparams['sequence_length']
                },
                "total_training_time": total_training_time,
                "average_evaluation": {
                    "train": avg_train_metrics,
                    "val": avg_val_metrics,
                    "test": avg_test_metrics
                },
                "window_results": all_windows_results
            }

            # บันทึกไฟล์ log
            with open(json_log_file, 'w') as f:
                json.dump(log_data, f, indent=4, default=default_converter)

            # สร้างข้อมูลสรุปของการเทรนรอบนี้
            retrain_summary = {
                'model_rank': model_rank,
                'retrain_index': retrain_idx,
                'model_name': model_name,
                'hyperparams': hyperparams,
                'avg_metrics': {
                    'avg_test_rmse': float(avg_test_rmse),
                    'avg_test_mae': float(avg_test_mae),
                    'avg_test_r2': float(avg_test_r2),
                    'std_test_rmse': float(std_test_rmse),
                    'std_test_mae': float(std_test_mae),
                    'std_test_r2': float(std_test_r2)
                },
                'training_time': total_training_time,
                'model_path': model_path,
                'log_file': json_log_file,
                'avg_losses_plot_path': avg_losses_plot_path,
                'n_windows': len(all_windows_results)
            }

            print(f"\nโมเดลอันดับที่ {model_rank} รอบที่ {retrain_idx} - ผลลัพธ์:")
            print(f"R²: {avg_test_r2:.4f} ± {std_test_r2:.4f}")
            print(f"RMSE: {avg_test_rmse:.4f} ± {std_test_rmse:.4f}")
            print(f"MAE: {avg_test_mae:.4f} ± {std_test_mae:.4f}")

            return retrain_summary

    def save_model_results(self, model_results, model_info, model_rank, model_dir):
            """
            บันทึกผลลัพธ์การเทรนซ้ำของโมเดล

            Args:
                model_results (list): ผลลัพธ์การเทรนซ้ำแต่ละรอบ
                model_info (dict): ข้อมูลของโมเดล
                model_rank (int): อันดับของโมเดล
                model_dir (str): โฟลเดอร์สำหรับเก็บผลลัพธ์
            """
            # สร้างข้อมูลสรุปของโมเดลนี้
            model_summary = {
                'model_rank': model_rank,
                'original_model_info': model_info,
                'retrain_results': model_results,
                'num_retrain': self.num_retrain
            }

            # คำนวณค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานของเมตริกจากทุกรอบ
            rmse_values = [result['avg_metrics']['avg_test_rmse'] for result in model_results]
            mae_values = [result['avg_metrics']['avg_test_mae'] for result in model_results]
            r2_values = [result['avg_metrics']['avg_test_r2'] for result in model_results]

            model_summary['metrics_stats'] = {
                'rmse': {
                    'mean': float(np.mean(rmse_values)),
                    'std': float(np.std(rmse_values)),
                    'min': float(np.min(rmse_values)),
                    'max': float(np.max(rmse_values)),
                    'values': rmse_values
                },
                'mae': {
                    'mean': float(np.mean(mae_values)),
                    'std': float(np.std(mae_values)),
                    'min': float(np.min(mae_values)),
                    'max': float(np.max(mae_values)),
                    'values': mae_values
                },
                'r2': {
                    'mean': float(np.mean(r2_values)),
                    'std': float(np.std(r2_values)),
                    'min': float(np.min(r2_values)),
                    'max': float(np.max(r2_values)),
                    'values': r2_values
                }
            }

            # บันทึกข้อมูลสรุปของโมเดลนี้
            summary_file = os.path.join(model_dir, "model_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(model_summary, f, indent=4, default=default_converter)

    def create_custom_summary(self, all_retrain_results):
                    """
                    สร้างไฟล์สรุปผลการเทรนซ้ำในรูปแบบที่ต้องการ (รวมทุกโมเดลทุกรอบ)

                    Args:
                        all_retrain_results (list): ผลลัพธ์การเทรนซ้ำของทุกโมเดล
                    """
                    # เตรียมข้อมูลสำหรับไฟล์ JSON และ CSV
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    # สร้างข้อมูลสำหรับไฟล์ JSON
                    json_data = {
                        "timestamp": timestamp,
                        "num_models": self.num_top_models,
                        "num_retrain": self.num_retrain,
                        "models": []
                    }

                    # สร้างข้อมูลสำหรับไฟล์ CSV
                    csv_data = []

                    # สำหรับแต่ละโมเดล
                    for result in all_retrain_results:
                        model_rank = result['model_rank']
                        model_info = result['model_info']
                        retrain_results = result['retrain_results']

                        # เก็บข้อมูลพื้นฐานของโมเดล
                        model_data = {
                            "model_rank": model_rank,
                            "hyperparams": model_info['hyperparams'],
                            "retrain_runs": []
                        }

                        # สำหรับแต่ละรอบการเทรน
                        for retrain_idx, run in enumerate(retrain_results):
                            # เก็บข้อมูลแต่ละรอบการเทรน
                            run_data = {
                                "retrain_index": retrain_idx + 1,
                                "avg_metrics": run['avg_metrics'],
                                "training_time": run['training_time'],
                                "model_path": run['model_path'],
                                "log_file": run['log_file']
                            }

                            model_data["retrain_runs"].append(run_data)

                            # เก็บข้อมูลสำหรับไฟล์ CSV
                            csv_row = {
                                "model_rank": model_rank,
                                "retrain_index": retrain_idx + 1,
                                "sequence_length": model_info['hyperparams']['sequence_length'],
                                "hidden_size": model_info['hyperparams']['hidden_size'],
                                "num_layers": model_info['hyperparams']['num_layers'],
                                "dropout_rate": model_info['hyperparams']['dropout_rate'],
                                "learning_rate": model_info['hyperparams']['learning_rate'],
                                "batch_size": model_info['hyperparams']['batch_size'],
                                "optimizer": model_info['hyperparams'].get('optimizer_type', 'Adam'),
                                "rmse": run['avg_metrics']['avg_test_rmse'],
                                "mae": run['avg_metrics']['avg_test_mae'],
                                "r2": run['avg_metrics']['avg_test_r2'],
                                "std_rmse": run['avg_metrics']['std_test_rmse'],
                                "std_mae": run['avg_metrics']['std_test_mae'],
                                "std_r2": run['avg_metrics']['std_test_r2'],
                                "training_time": run['training_time']
                            }

                            csv_data.append(csv_row)

                        # คำนวณค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานของเมตริกจากทุกรอบ
                        rmse_values = [r['avg_metrics']['avg_test_rmse'] for r in retrain_results]
                        mae_values = [r['avg_metrics']['avg_test_mae'] for r in retrain_results]
                        r2_values = [r['avg_metrics']['avg_test_r2'] for r in retrain_results]

                        model_data['metrics_stats'] = {
                            'rmse': {
                                'mean': float(np.mean(rmse_values)),
                                'std': float(np.std(rmse_values)),
                                'min': float(np.min(rmse_values)),
                                'max': float(np.max(rmse_values))
                            },
                            'mae': {
                                'mean': float(np.mean(mae_values)),
                                'std': float(np.std(mae_values)),
                                'min': float(np.min(mae_values)),
                                'max': float(np.max(mae_values))
                            },
                            'r2': {
                                'mean': float(np.mean(r2_values)),
                                'std': float(np.std(r2_values)),
                                'min': float(np.min(r2_values)),
                                'max': float(np.max(r2_values))
                            }
                        }

                        json_data["models"].append(model_data)

                    # บันทึกข้อมูลลงไฟล์ JSON
                    json_file = os.path.join(self.output_dir, "custom_retrain_summary.json")
                    with open(json_file, 'w') as f:
                        json.dump(json_data, f, indent=4, default=default_converter)

                    # บันทึกข้อมูลลงไฟล์ CSV
                    csv_file = os.path.join(self.output_dir, "custom_retrain_summary.csv")
                    df = pd.DataFrame(csv_data)
                    df.to_csv(csv_file, index=False)

                    print(f"\nบันทึกไฟล์สรุปผลที่:")
                    print(f"- JSON: {json_file}")
                    print(f"- CSV: {csv_file}")

                    return json_file, csv_file