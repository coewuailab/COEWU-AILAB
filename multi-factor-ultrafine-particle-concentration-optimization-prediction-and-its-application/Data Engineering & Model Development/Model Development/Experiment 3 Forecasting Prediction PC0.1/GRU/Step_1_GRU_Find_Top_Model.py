"""
การเทรนโมเดล GRU ด้วย PyTorch สำหรับทำนายค่า PM0.1 พร้อมระบบ Bayesian Hyperparameter Tuning
ใช้เทคนิค Expanding Window Split สำหรับข้อมูล Time Series
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import time
import os
import json
import copy
from datetime import datetime
import shutil  # สำหรับคัดลอกไฟล์

# เพิ่มไลบรารีสำหรับ Bayesian Optimization
from skopt import BayesSearchCV
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective, plot_evaluations
import skopt

# ตั้งค่า seed สำหรับการทำให้ผลลัพธ์คงที่
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

# ตรวจสอบอุปกรณ์ที่ใช้ (CPU หรือ GPU)
# ตรวจสอบว่า CUDA พร้อมใช้งานไหม
print(f"CUDA available: {torch.cuda.is_available()}")

# ตรวจสอบจำนวน GPU ที่มี
print(f"GPU count: {torch.cuda.device_count()}")

# ตรวจสอบชื่อของ GPU ที่กำลังใช้
if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.get_device_name()}")

# ตรวจสอบว่าโมเดลของคุณอยู่บน GPU หรือไม่
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ตรวจสอบหน่วยความจำที่ใช้
if torch.cuda.is_available():
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")
    print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1024 ** 2:.2f} MB")

# ตรวจสอบอุปกรณ์ที่ใช้ (CPU หรือ GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"ใช้อุปกรณ์: {device}")

# ตั้งค่า global parameters
EARLY_STOP_PATIENCE = 20  # จำนวน epochs ที่รอก่อนหยุดการเทรนถ้าไม่มีการปรับปรุง
MAX_EXPANDING_WINDOWS = 5  # จำนวนหน้าต่างสูงสุดที่จะใช้ในเทคนิค Expanding Window
TOP_MODELS_TO_SAVE = 10  # จำนวนโมเดลที่ดีที่สุดที่จะบันทึกไว้

# เพิ่มฟังก์ชัน default เพื่อแปลง NumPy types
def default_converter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif 'RandomState' in obj.__class__.__name__:
        return obj.__getstate__() if hasattr(obj, '_getstate_') else str(obj)
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

# Early Stopping Class
class EarlyStopping:
    """Early stopping เพื่อหยุดการเทรนหากค่า validation loss ไม่ดีขึ้นหลังจากผ่านไป patience epochs"""

    def __init__(self, patience=20, delta=0, verbose=True):
        """
        Args:
            patience (int): จำนวน epochs สำหรับรอก่อนหยุดการเทรน
            delta (float): ค่าผลต่างขั้นต่ำที่ถือว่าเป็นการปรับปรุง
            verbose (bool): ใช้แสดงข้อความหรือไม่
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.best_model_state = None
        self.stopped_epoch = 0  # เพิ่มตัวแปรเพื่อเก็บ epoch ที่หยุด

    def __call__(self, val_loss, model, epoch):  # เพิ่มพารามิเตอร์ epoch
        score = -val_loss  # ยิ่งค่า loss ต่ำ ยิ่งดี

        if self.best_score is None:
            # ครั้งแรก
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            # ไม่มีการปรับปรุงเพียงพอ
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                self.stopped_epoch = epoch + 1  # บันทึก epoch ที่หยุด
        else:
            # มีการปรับปรุง
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''บันทึก model เมื่อ validation loss ลดลง'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        # บันทึก state_dict ของโมเดลที่ดีที่สุด
        self.best_model_state = copy.deepcopy(model.state_dict())
        self.val_loss_min = val_loss

# เพิ่มฟังก์ชันสำหรับตรวจสอบการใช้งาน GPU
def check_gpu_usage():
    """
    ตรวจสอบการใช้งาน GPU และแสดงข้อมูลการใช้หน่วยความจำ

    Returns:
        dict: ข้อมูลการใช้งาน GPU
    """
    gpu_info = {
        'available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count(),
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
        'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        'memory_allocated': 0,
        'memory_reserved': 0,
        'memory_free': 0,
        'memory_total': 0
    }

    if gpu_info['available']:
        # ข้อมูลหน่วยความจำ GPU ที่ใช้และสำรองไว้ (MB)
        gpu_info['memory_allocated'] = torch.cuda.memory_allocated() / (1024 ** 2)
        gpu_info['memory_reserved'] = torch.cuda.memory_reserved() / (1024 ** 2)

        # ข้อมูลหน่วยความจำทั้งหมดและที่ยังว่างอยู่ (ถ้ามี nvidia-smi)
        try:
            import subprocess
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.total,memory.free', '--format=csv,nounits,noheader'])
            total, free = map(int, result.decode('utf-8').split(','))
            gpu_info['memory_total'] = total
            gpu_info['memory_free'] = free
        except:
            # หากไม่สามารถเรียกใช้ nvidia-smi ได้ จะใช้ข้อมูลจาก PyTorch เท่านั้น
            pass

    # แสดงข้อมูล
    print("\n===== ข้อมูลการใช้งาน GPU =====")
    print(f"GPU พร้อมใช้งาน: {gpu_info['available']}")

    if gpu_info['available']:
        print(f"จำนวน GPU: {gpu_info['device_count']}")
        print(f"GPU ปัจจุบัน: {gpu_info['current_device']}")
        print(f"ชื่ออุปกรณ์: {gpu_info['device_name']}")
        print(f"หน่วยความจำที่ใช้: {gpu_info['memory_allocated']:.2f} MB")
        print(f"หน่วยความจำที่สำรอง: {gpu_info['memory_reserved']:.2f} MB")

        if gpu_info['memory_total'] > 0:
            print(f"หน่วยความจำทั้งหมด: {gpu_info['memory_total']} MB")
            print(f"หน่วยความจำที่ว่าง: {gpu_info['memory_free']} MB")
            print(
                f"เปอร์เซ็นต์การใช้งาน: {((gpu_info['memory_total'] - gpu_info['memory_free']) / gpu_info['memory_total']) * 100:.2f}%")

    print("===============================\n")

    return gpu_info


# ฟังก์ชันสำหรับตรวจสอบว่า tensor อยู่บน GPU หรือไม่
def check_model_on_device(model, data_sample=None):
    """
    ตรวจสอบว่าโมเดลและข้อมูลอยู่บนอุปกรณ์ (CPU/GPU) เดียวกันหรือไม่

    Args:
        model: โมเดลที่ต้องการตรวจสอบ
        data_sample: ตัวอย่างข้อมูลที่จะส่งเข้าโมเดล (optional)

    Returns:
        dict: ข้อมูลเกี่ยวกับอุปกรณ์ที่โมเดลและข้อมูลอยู่
    """
    device_info = {
        'model_device': next(model.parameters()).device,
        'data_device': None,
        'same_device': None,
        'using_gpu': str(next(model.parameters()).device).startswith('cuda')
    }

    if data_sample is not None:
        device_info['data_device'] = data_sample.device
        device_info['same_device'] = (device_info['model_device'] == device_info['data_device'])

    print("\n===== ข้อมูลอุปกรณ์ของโมเดล =====")
    print(f"โมเดลอยู่บน: {device_info['model_device']}")
    if data_sample is not None:
        print(f"ข้อมูลอยู่บน: {device_info['data_device']}")
        print(f"อยู่บนอุปกรณ์เดียวกัน: {device_info['same_device']}")
    print(f"ใช้งาน GPU: {device_info['using_gpu']}")
    print("==================================\n")

    return device_info


# =================== GRU Model ===================
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # ชั้น GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        # ชั้น Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # ชั้น Fully Connected สำหรับเอาต์พุตสุดท้าย
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # กำหนดค่าเริ่มต้นของ hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # ส่งข้อมูลผ่านชั้น GRU (GRU ไม่ต้องใช้ cell state เหมือน GRU)
        out, _ = self.gru(x, h0)

        # เลือกผลลัพธ์สุดท้ายจากลำดับ
        out = out[:, -1, :]

        # ส่งผ่านชั้น Dropout
        out = self.dropout(out)

        # ส่งผ่านชั้น Fully Connected
        out = self.fc(out)

        return out


# =================== Data Processing ===================
def prepare_data_expanding_window(file_path, numeric_features, categorical_features, sequence_length,
                                  initial_train_ratio=0.3, val_ratio=0.1, expand_step=0.1):
    """
    เตรียมข้อมูลสำหรับการเทรนโมเดลด้วยเทคนิค Expanding Window Split

    Args:
        file_path (str): พาธของไฟล์ CSV
        numeric_features (list): รายการคุณลักษณะตัวเลข
        categorical_features (list): รายการคุณลักษณะแบบ categorical
        sequence_length (int): ความยาวของลำดับเวลา
        initial_train_ratio (float): สัดส่วนของข้อมูลสำหรับการเทรนในรอบแรก (default: 0.5)
        val_ratio (float): สัดส่วนของข้อมูลสำหรับการตรวจสอบ (default: 0.2)
        expand_step (float): ขนาดของการขยายหน้าต่างในแต่ละรอบ (default: 0.1)

    Returns:
        tuple: ข้อมูลที่ต้องการสำหรับการเทรนโมเดลและประเมินผล
    """
    # โหลดข้อมูลจากไฟล์ CSV
    df = pd.read_csv(file_path)
    print("โหลดข้อมูลเรียบร้อย")
    print(f"จำนวนข้อมูลทั้งหมด: {df.shape[0]} แถว, {df.shape[1]} คอลัมน์")

    # แปลงคอลัมน์ Time เป็น datetime
    df['Time'] = pd.to_datetime(df['Time'])

    # กรองเฉพาะข้อมูลในช่วงเวลาที่กำหนด (7-17 น.)
    df = df[(df['Time'].dt.hour >= 7) & (df['Time'].dt.hour < 17)].reset_index(drop=True)
    print(f"จำนวนข้อมูลหลังกรองตามช่วงเวลา: {df.shape[0]} แถว")

    # แก้ไขปัญหาค่า null (ถ้ามี) - ใช้ forward fill
    df = df.ffill()

    # เรียงข้อมูลตามเวลา (เพื่อให้แน่ใจว่าข้อมูลเรียงตามลำดับเวลา)
    df = df.sort_values('Time').reset_index(drop=True)

    # เตรียมข้อมูลเพื่อทำ encoding
    X_numeric = df[numeric_features].values
    X_categorical = df[categorical_features].values

    # สร้าง encoder สำหรับข้อมูล categorical
    label_encoders = []
    X_categorical_encoded = []

    for col in X_categorical.T:  # วนตามแต่ละคอลัมน์
        le = LabelEncoder()
        encoded_col = le.fit_transform(col)
        X_categorical_encoded.append(encoded_col)
        label_encoders.append(le)

    # สร้าง scaler สำหรับข้อมูลตัวเลข
    numeric_scaler = MinMaxScaler()
    numeric_scaler.fit(X_numeric)

    # สกัดข้อมูล target
    y_data = df['PC0.1_Calibrate'].values.reshape(-1, 1) #เปลี่ยนเป็น PC0.1 ที่ Calibrate แล้ว

    # สร้าง scaler สำหรับข้อมูล y
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler.fit(y_data)

    # ปรับเสกลข้อมูลตัวเลข
    X_numeric_scaled = numeric_scaler.transform(X_numeric)

    # เอนโค้ดข้อมูล categorical (เก็บไว้ใช้ต่อ)
    X_categorical_encoded = np.column_stack(X_categorical_encoded)

    # รวมข้อมูลตัวเลขและข้อมูล categorical ที่เอนโค้ดแล้ว
    X_encoded = np.concatenate([X_numeric_scaled, X_categorical_encoded], axis=1)

    # ปรับเสกลข้อมูล y
    y_scaled = y_scaler.transform(y_data)

    # หาค่า input_size
    input_size = X_encoded.shape[1]
    print(f"จำนวน feature หลังการเอนโค้ด: {input_size}")

    # กำหนดจุดตัดสำหรับ expanding window split
    total_size = len(df)
    initial_train_size = int(total_size * initial_train_ratio)
    val_size = int(total_size * val_ratio)
    expand_size = int(total_size * expand_step)

    # ข้อมูลเริ่มต้น
    current_train_size = initial_train_size
    test_start = current_train_size + val_size

    print(f"เริ่มต้น: จำนวนข้อมูลฝึกอบรม = {current_train_size}, " +
          f"จำนวนข้อมูลตรวจสอบ = {val_size}, " +
          f"จำนวนข้อมูลทดสอบ = {total_size - test_start}")

    # ข้อมูลทั้งหมดที่ไม่ได้ปรับเสกล (เก็บไว้สำหรับการวิเคราะห์)
    all_data = {
        'X_encoded': X_encoded,
        'y_scaled': y_scaled,
        'y_raw': y_data,
        'initial_train_size': initial_train_size,
        'val_size': val_size,
        'expand_size': expand_size,
        'total_size': total_size
    }

    return (X_encoded, y_scaled, y_data, numeric_scaler, label_encoders,
            y_scaler, input_size, all_data)


def create_expanding_window_split(all_data, window_index, sequence_length, batch_size):
    """
    สร้าง DataLoader สำหรับการเทรนและการทดสอบด้วยเทคนิค Expanding Window

    Args:
        all_data (dict): ข้อมูลทั้งหมดที่ได้จาก prepare_data_expanding_window
        window_index (int): ดัชนีของหน้าต่าง (เริ่มจาก 0)
        sequence_length (int): ความยาวของลำดับเวลา
        batch_size (int): ขนาดของแบทช์

    Returns:
        tuple: DataLoader สำหรับการเทรนและการทดสอบ
    """
    X_encoded = all_data['X_encoded']
    y_scaled = all_data['y_scaled']

    initial_train_size = all_data['initial_train_size']
    val_size = all_data['val_size']
    expand_size = all_data['expand_size']

    # คำนวณขนาดของข้อมูลฝึกอบรมสำหรับหน้าต่างปัจจุบัน
    current_train_size = initial_train_size + (window_index * expand_size)

    # กำหนดช่วงข้อมูล
    train_end = current_train_size
    val_start = train_end
    val_end = val_start + val_size
    test_start = val_end

    # ตรวจสอบว่าวงหน้าต่างการทดสอบไม่เกินขนาดข้อมูล
    if test_start >= all_data['total_size']:
        return None, None, None, None

    # แยกข้อมูลตามช่วง
    X_train = X_encoded[:train_end]
    y_train = y_scaled[:train_end]

    X_val = X_encoded[val_start:val_end]
    y_val = y_scaled[val_start:val_end]

    X_test = X_encoded[test_start:]
    y_test = y_scaled[test_start:]

    # สร้าง sequences
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)

    print(f"Window {window_index + 1}: " +
          f"Train={X_train_seq.shape[0]}, Val={X_val_seq.shape[0]}, Test={X_test_seq.shape[0]}")

    # แปลงข้อมูลเป็น PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
    y_train_tensor = torch.FloatTensor(y_train_seq).to(device)

    X_val_tensor = torch.FloatTensor(X_val_seq).to(device)
    y_val_tensor = torch.FloatTensor(y_val_seq).to(device)

    X_test_tensor = torch.FloatTensor(X_test_seq).to(device)
    y_test_tensor = torch.FloatTensor(y_test_seq).to(device)

    # สร้าง DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    data_sizes = {
        'train': X_train_seq.shape[0],
        'val': X_val_seq.shape[0],
        'test': X_test_seq.shape[0],
        'window_index': window_index,
        'train_start': 0,
        'train_end': train_end,
        'val_start': val_start,
        'val_end': val_end,
        'test_start': test_start,
    }

    return train_loader, val_loader, test_loader, data_sizes


def create_sequences(x_data, y_data, seq_length):
    """
    สร้างข้อมูลแบบลำดับเวลา (sequences) จากข้อมูลที่มี
    โดยใช้ข้อมูล t-sequence_length ถึง t-1 เพื่อทำนายค่าที่ t
    """
    x_seq, y_seq = [], []
    for i in range(len(x_data) - seq_length + 1):  # +1 เพราะเราต้องการทำนายค่าที่ t ไม่ใช่ t+1
        # เลือกข้อมูลตั้งแต่ i ถึง i+seq_length-1 (คือ t-3, t-2, t-1 เมื่อ seq_length=3)
        x_seq.append(x_data[i:i + seq_length])
        # ใช้ค่าที่ i+seq_length-1 (คือค่าที่ t ที่ต้องการทำนาย)
        y_seq.append(y_data[i + seq_length - 1])
    return np.array(x_seq), np.array(y_seq)


# =================== Training and Evaluation ===================
def train_and_evaluate_model_expanding(model, train_loader, val_loader, test_loader, criterion, optimizer,
                                       num_epochs, patience, data_sizes, y_scaler, window_index):
    """
    เทรนและประเมินโมเดล GRU โดยใช้เทคนิค Expanding Window

    Returns:
        tuple: ผลการเทรนและประเมินโมเดล
    """
    # ตรวจสอบว่าโมเดลอยู่บน GPU หรือไม่
    if torch.cuda.is_available():
        device_info = check_model_on_device(model)
        if not device_info['using_gpu']:
            print(f"คำเตือน: GPU พร้อมใช้งานแต่โมเดลอยู่บน CPU ในหน้าต่างที่ {window_index + 1}")
            model = model.to(device)
            print(f"ย้ายโมเดลไปยัง {device} แล้ว")

    train_losses = []
    val_losses = []

    # ตรวจสอบหน่วยความจำ GPU ก่อนเริ่มเทรน (ถ้ามี)
    if torch.cuda.is_available():
        print(f"\nการใช้งาน GPU ก่อนเริ่มเทรนหน้าต่างที่ {window_index + 1}:")
        gpu_info_before_train = check_gpu_usage()

    # Early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # เริ่มจับเวลา
    start_time = time.time()

    # วนลูปตามจำนวน epochs
    for epoch in range(num_epochs):
        # =================== โหมดเทรน ===================
        model.train()
        train_loss = 0.0

        for i, (inputs, targets) in enumerate(train_loader):
            # ตรวจสอบว่าข้อมูลอยู่บนอุปกรณ์เดียวกับโมเดลหรือไม่
            if i == 0 and torch.cuda.is_available():  # เช็คเฉพาะ batch แรก
                if inputs.device != next(model.parameters()).device:
                    print(f"คำเตือน: ข้อมูลอยู่บน {inputs.device} แต่โมเดลอยู่บน {next(model.parameters()).device}")
                    inputs = inputs.to(device)
                    targets = targets.to(device)

            # ล้างค่า gradient
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # คำนวณค่า loss
            loss = criterion(outputs, targets)

            # Backward pass และอัปเดตค่าพารามิเตอร์
            loss.backward()
            optimizer.step()

            # สะสมค่า loss
            train_loss += loss.item() * inputs.size(0)

        # คำนวณค่าเฉลี่ย loss สำหรับ epoch นี้
        train_loss = train_loss / data_sizes['train']
        train_losses.append(train_loss)

        # =================== โหมดตรวจสอบ (Validation) ===================
        model.eval()
        val_loss = 0.0

        with torch.no_grad():  # ไม่คำนวณ gradient ในโหมดทดสอบ
            for inputs, targets in val_loader:
                # ตรวจสอบว่าข้อมูลอยู่บนอุปกรณ์เดียวกับโมเดลหรือไม่
                if torch.cuda.is_available() and inputs.device != next(model.parameters()).device:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                # Forward pass
                outputs = model(inputs)

                # คำนวณค่า loss
                loss = criterion(outputs, targets)

                # สะสมค่า loss
                val_loss += loss.item() * inputs.size(0)

        # คำนวณค่าเฉลี่ย loss สำหรับข้อมูลตรวจสอบ
        val_loss = val_loss / data_sizes['val']
        val_losses.append(val_loss)

        # แสดงผลลัพธ์
        print(
            f'Window {window_index + 1}, Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # ตรวจสอบการใช้งาน GPU ทุก 20 epochs (หรือตามที่เหมาะสม)
        if torch.cuda.is_available() and (epoch + 1) % 20 == 0:
            print(f"\nการใช้งาน GPU หลัง Epoch {epoch + 1}:")
            check_gpu_usage()

        # ตรวจสอบ early stopping
        early_stopping(val_loss, model, epoch)  # ส่งค่า epoch ด้วย
        if early_stopping.early_stop:
            print(f"Early stopping ที่ epoch {epoch + 1}")
            # โหลดค่า state_dict ที่ดีที่สุด
            model.load_state_dict(early_stopping.best_model_state)
            break

    # เวลาที่ใช้ในการเทรน
    training_time = time.time() - start_time

    # บันทึกค่า epoch ที่หยุด
    stopped_epoch = early_stopping.stopped_epoch if early_stopping.early_stop else num_epochs

    # ตรวจสอบหน่วยความจำ GPU หลังเทรนเสร็จ (ถ้ามี)
    if torch.cuda.is_available():
        print(f"\nการใช้งาน GPU หลังเทรนเสร็จหน้าต่างที่ {window_index + 1}:")
        gpu_info_after_train = check_gpu_usage()

        # แสดงความเปลี่ยนแปลงของหน่วยความจำ
        print(
            f"\nหน่วยความจำที่ใช้เพิ่มขึ้น: {gpu_info_after_train['memory_allocated'] - gpu_info_before_train['memory_allocated']:.2f} MB")

    # =================== ประเมินผลโมเดล ===================
    model.eval()

    # เตรียมตัวแปรสำหรับเก็บค่าทำนายและค่าจริง
    train_predictions = []
    train_actuals = []
    val_predictions = []
    val_actuals = []
    test_predictions = []
    test_actuals = []

    # ทำนายชุดข้อมูลฝึกอบรม
    with torch.no_grad():
        for inputs, targets in train_loader:
            # ตรวจสอบว่าข้อมูลอยู่บนอุปกรณ์เดียวกับโมเดลหรือไม่
            if torch.cuda.is_available() and inputs.device != next(model.parameters()).device:
                inputs = inputs.to(device)
                targets = targets.to(device)

            outputs = model(inputs)
            train_predictions.append(outputs.cpu().numpy())
            train_actuals.append(targets.cpu().numpy())

    # ทำนายชุดข้อมูลตรวจสอบ
    with torch.no_grad():
        for inputs, targets in val_loader:
            # ตรวจสอบว่าข้อมูลอยู่บนอุปกรณ์เดียวกับโมเดลหรือไม่
            if torch.cuda.is_available() and inputs.device != next(model.parameters()).device:
                inputs = inputs.to(device)
                targets = targets.to(device)

            outputs = model(inputs)
            val_predictions.append(outputs.cpu().numpy())
            val_actuals.append(targets.cpu().numpy())

    # ทำนายชุดข้อมูลทดสอบ
    with torch.no_grad():
        for inputs, targets in test_loader:
            # ตรวจสอบว่าข้อมูลอยู่บนอุปกรณ์เดียวกับโมเดลหรือไม่
            if torch.cuda.is_available() and inputs.device != next(model.parameters()).device:
                inputs = inputs.to(device)
                targets = targets.to(device)

            outputs = model(inputs)
            test_predictions.append(outputs.cpu().numpy())
            test_actuals.append(targets.cpu().numpy())

    # แปลงลิสต์เป็น array - เพิ่มการตรวจสอบว่าลิสต์ไม่ว่าง
    if len(train_predictions) > 0:
        train_predictions = np.vstack(train_predictions)
        train_actuals = np.vstack(train_actuals)

        # แปลงกลับเป็นสเกลเดิม
        train_predictions = y_scaler.inverse_transform(train_predictions)
        train_actuals = y_scaler.inverse_transform(train_actuals)

        # คำนวณค่าเมตริกต่างๆ
        train_metrics = {
            'mse': mean_squared_error(train_actuals, train_predictions),
            'rmse': np.sqrt(mean_squared_error(train_actuals, train_predictions)),
            'mae': mean_absolute_error(train_actuals, train_predictions),
            'r2': r2_score(train_actuals, train_predictions)
        }
    else:
        # กรณีไม่มีข้อมูล training
        train_predictions = np.array([])
        train_actuals = np.array([])
        train_metrics = {
            'mse': float('nan'),
            'rmse': float('nan'),
            'mae': float('nan'),
            'r2': float('nan')
        }

    if len(val_predictions) > 0:
        val_predictions = np.vstack(val_predictions)
        val_actuals = np.vstack(val_actuals)

        # แปลงกลับเป็นสเกลเดิม
        val_predictions = y_scaler.inverse_transform(val_predictions)
        val_actuals = y_scaler.inverse_transform(val_actuals)

        # คำนวณค่าเมตริกต่างๆ
        val_metrics = {
            'mse': mean_squared_error(val_actuals, val_predictions),
            'rmse': np.sqrt(mean_squared_error(val_actuals, val_predictions)),
            'mae': mean_absolute_error(val_actuals, val_predictions),
            'r2': r2_score(val_actuals, val_predictions)
        }
    else:
        # กรณีไม่มีข้อมูล validation
        val_predictions = np.array([])
        val_actuals = np.array([])
        val_metrics = {
            'mse': float('nan'),
            'rmse': float('nan'),
            'mae': float('nan'),
            'r2': float('nan')
        }

    if len(test_predictions) > 0:
        test_predictions = np.vstack(test_predictions)
        test_actuals = np.vstack(test_actuals)

        # แปลงกลับเป็นสเกลเดิม
        test_predictions = y_scaler.inverse_transform(test_predictions)
        test_actuals = y_scaler.inverse_transform(test_actuals)

        # คำนวณค่าเมตริกต่างๆ
        test_metrics = {
            'mse': mean_squared_error(test_actuals, test_predictions),
            'rmse': np.sqrt(mean_squared_error(test_actuals, test_predictions)),
            'mae': mean_absolute_error(test_actuals, test_predictions),
            'r2': r2_score(test_actuals, test_predictions)
        }
    else:
        # กรณีไม่มีข้อมูล test
        test_predictions = np.array([])
        test_actuals = np.array([])
        test_metrics = {
            'mse': float('nan'),
            'rmse': float('nan'),
            'mae': float('nan'),
            'r2': float('nan')
        }

    print(f"หน้าต่างที่ {window_index + 1} - เทรนเสร็จสิ้นในเวลา {training_time:.2f} วินาที")
    print(
        f"ผลลัพธ์ RMSE: Train = {train_metrics['rmse']:.4f}, Val = {val_metrics['rmse']:.4f}, Test = {test_metrics['rmse']:.4f}")

    return (early_stopping, train_losses, val_losses,
            train_metrics, val_metrics, test_metrics, training_time,
            train_predictions, train_actuals, val_predictions, val_actuals,
            test_predictions, test_actuals, data_sizes, stopped_epoch)


def plot_results_expanding(train_losses, val_losses, test_actuals, test_predictions,
                           model_name, plots_dir, window_index, hyperparams, metrics, stopped_epoch, show_plot=False):
    """
    พล็อตกราฟผลลัพธ์การเทรนและการทำนายสำหรับเทคนิค Expanding Window
    """
    plt.figure(figsize=(15, 10))

    # พล็อต loss
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'GRU Model Loss - Window {window_index + 1}')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # พล็อตผลการทำนาย
    plt.subplot(2, 2, 2)
    plt.plot(test_actuals[:200], label='Actual Data')
    plt.plot(test_predictions[:200], label='Predicted Data')
    plt.title(f'Prediction vs Actual - Window {window_index + 1}')
    plt.xlabel('Time')
    plt.ylabel('PM0.1 Value')
    plt.legend()

    # พล็อตกราฟการกระจาย (Scatter plot)
    plt.subplot(2, 2, 3)
    plt.scatter(test_actuals, test_predictions, alpha=0.5)
    plt.plot([min(test_actuals), max(test_actuals)], [min(test_actuals), max(test_actuals)], 'r--')
    plt.title(f'Actual vs Predicted - Window {window_index + 1}')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    # แสดงข้อมูลเพิ่มเติม
    plt.subplot(2, 2, 4)
    plt.axis('off')
    info_text = (
        f"Window {window_index + 1} - Hyperparameters & Metrics:\n\n"
        f"Sequence Length: {hyperparams['sequence_length']}\n"
        f"Hidden Size: {hyperparams['hidden_size']}\n"
        f"Num Layers: {hyperparams['num_layers']}\n"
        f"Dropout Rate: {hyperparams['dropout_rate']}\n"
        f"Learning Rate: {hyperparams['learning_rate']}\n"
        f"Optimizer: {hyperparams.get('optimizer_type', 'Adam')}\n"  # เพิ่มบรรทัดนี้
        f"Stopped at Epoch: {stopped_epoch}\n\n"
        f"Test Metrics:\n"
        f"RMSE: {metrics['rmse']:.4f}\n"
        f"MAE: {metrics['mae']:.4f}\n"
        f"R²: {metrics['r2']:.4f}"
    )
    plt.text(0.1, 0.5, info_text, fontsize=10, va='center')

    plt.tight_layout()

    # บันทึกรูปภาพโดยใช้ชื่อโมเดลและหน้าต่าง
    plot_path = os.path.join(plots_dir, f"{model_name}_window{window_index + 1}_results.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()

    return plot_path


def plot_combined_results(all_windows_results, model_name, plots_dir, hyperparams, show_plot=False):
    """
    พล็อตกราฟเปรียบเทียบผลลัพธ์จากหลายหน้าต่าง
    """
    # เตรียมข้อมูลสำหรับพล็อต
    window_indices = []
    train_rmse = []
    val_rmse = []
    test_rmse = []
    train_r2 = []
    val_r2 = []
    test_r2 = []
    stopped_epochs = []

    # เพิ่มตัวแปรเก็บค่า losses ของแต่ละ window
    all_train_losses = []
    all_val_losses = []

    for result in all_windows_results:
        window_idx = result['window_index']
        window_indices.append(window_idx + 1)  # เริ่มที่ 1 สำหรับการแสดงผล
        train_rmse.append(result['train_metrics']['rmse'])
        val_rmse.append(result['val_metrics']['rmse'])
        test_rmse.append(result['test_metrics']['rmse'])
        train_r2.append(result['train_metrics']['r2'])
        val_r2.append(result['val_metrics']['r2'])
        test_r2.append(result['test_metrics']['r2'])
        stopped_epochs.append(result['stopped_epoch'])

        # เก็บค่า losses ของแต่ละ window
        if 'train_losses' in result:
            all_train_losses.append(result['train_losses'])
        if 'val_losses' in result:
            all_val_losses.append(result['val_losses'])

    plt.figure(figsize=(15, 15))  # ปรับขนาดให้ใหญ่ขึ้น

    # พล็อต RMSE ของแต่ละหน้าต่าง
    plt.subplot(3, 2, 1)
    plt.plot(window_indices, train_rmse, 'o-', label='Training RMSE')
    plt.plot(window_indices, val_rmse, 's-', label='Validation RMSE')
    plt.plot(window_indices, test_rmse, '^-', label='Test RMSE')
    plt.title('RMSE Across Windows')
    plt.xlabel('Window')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.legend()

    # พล็อต R² ของแต่ละหน้าต่าง
    plt.subplot(3, 2, 2)
    plt.plot(window_indices, train_r2, 'o-', label='Training R²')
    plt.plot(window_indices, val_r2, 's-', label='Validation R²')
    plt.plot(window_indices, test_r2, '^-', label='Test R²')
    plt.title('R² Across Windows')
    plt.xlabel('Window')
    plt.ylabel('R²')
    plt.grid(True)
    plt.legend()

    # พล็อต Stopped Epochs
    plt.subplot(3, 2, 3)
    plt.plot(window_indices, stopped_epochs, 'o-', color='purple')
    plt.title('Early Stopping Epochs')
    plt.xlabel('Window')
    plt.ylabel('Epoch')
    plt.grid(True)

    # เพิ่มการพล็อต Training Losses และ Validation Losses ของทุก window
    if all_train_losses and all_val_losses:
        plt.subplot(3, 2, 4)

        # พล็อต losses ของแต่ละ window
        for i, (t_losses, v_losses) in enumerate(zip(all_train_losses, all_val_losses)):
            max_epoch = len(t_losses)
            epochs = list(range(1, max_epoch + 1))

            plt.plot(epochs, t_losses, '-', label=f'Train Loss (Window {i + 1})', alpha=0.7)
            plt.plot(epochs, v_losses, '--', label=f'Val Loss (Window {i + 1})', alpha=0.7)

        plt.title('Training & Validation Losses Across Windows')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

    # แสดงข้อมูลเพิ่มเติม
    plt.subplot(3, 2, 5)  # ปรับตำแหน่ง subplot
    plt.axis('off')
    avg_test_rmse = np.mean(test_rmse)
    avg_test_r2 = np.mean(test_r2)
    avg_stopped_epoch = np.mean(stopped_epochs)

    info_text = (
        f"Expanding Window Results Summary\n\n"
        f"Windows: {len(window_indices)}\n"
        f"Average Test RMSE: {avg_test_rmse:.4f}\n"
        f"Average Test R²: {avg_test_r2:.4f}\n"
        f"Average Stopped Epoch: {avg_stopped_epoch:.1f}\n\n"
        f"Hyperparameters:\n"
        f"Sequence Length: {hyperparams['sequence_length']}\n"
        f"Hidden Size: {hyperparams['hidden_size']}\n"
        f"Num Layers: {hyperparams['num_layers']}\n"
        f"Dropout Rate: {hyperparams['dropout_rate']}\n"
        f"Learning Rate: {hyperparams['learning_rate']}\n"
        f"Optimizer: {hyperparams.get('optimizer_type', 'Adam')}"
    )
    plt.text(0.1, 0.5, info_text, fontsize=10, va='center')

    plt.tight_layout()

    # บันทึกรูปภาพ
    plot_path = os.path.join(plots_dir, f"{model_name}_combined_results.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()

    return plot_path


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

def save_model_and_logs_expanding(model, optimizer, train_loss, val_loss, input_size, output_size,
                                 hyperparams, numeric_scaler, label_encoders, y_scaler,
                                 all_windows_results, training_time, model_name, models_dir, log_dir):
    """
    บันทึกโมเดล GRU และข้อมูลการเทรนสำหรับเทคนิค Expanding Window
    """
    # บันทึกโมเดล
    model_path = os.path.join(models_dir, f"{model_name}_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'hyperparameters': {
            'input_size': input_size,
            'output_size': output_size,
            'hidden_size': hyperparams['hidden_size'],
            'num_layers': hyperparams['num_layers'],
            'dropout_rate': hyperparams['dropout_rate'],
            'sequence_length': hyperparams['sequence_length']
        },
        'numeric_scaler': numeric_scaler,
        'label_encoders': label_encoders,
        'y_scaler': y_scaler
    }, model_path)

    # สรุปผลการเทรนจากทุกหน้าต่าง
    window_summaries = []
    for result in all_windows_results:
        window_summary = {
            'window_index': result['window_index'],
            'train_metrics': result['train_metrics'],
            'val_metrics': result['val_metrics'],
            'test_metrics': result['test_metrics'],
            'data_sizes': {
                'train': result['data_sizes']['train'],
                'val': result['data_sizes']['val'],
                'test': result['data_sizes']['test'],
                'train_start': result['data_sizes']['train_start'],
                'train_end': result['data_sizes']['train_end'],
                'val_start': result['data_sizes']['val_start'],
                'val_end': result['data_sizes']['val_end'],
                'test_start': result['data_sizes']['test_start'],
            },
            'stopped_epoch': result['stopped_epoch']
        }
        window_summaries.append(window_summary)

    # คำนวณค่าเฉลี่ยเมตริก
    avg_train_metrics = {
        'mse': np.mean([r['train_metrics']['mse'] for r in all_windows_results]),
        'rmse': np.mean([r['train_metrics']['rmse'] for r in all_windows_results]),
        'mae': np.mean([r['train_metrics']['mae'] for r in all_windows_results]),
        'r2': np.mean([r['train_metrics']['r2'] for r in all_windows_results])
    }

    avg_val_metrics = {
        'mse': np.mean([r['val_metrics']['mse'] for r in all_windows_results]),
        'rmse': np.mean([r['val_metrics']['rmse'] for r in all_windows_results]),
        'mae': np.mean([r['val_metrics']['mae'] for r in all_windows_results]),
        'r2': np.mean([r['val_metrics']['r2'] for r in all_windows_results])
    }

    avg_test_metrics = {
        'mse': np.mean([r['test_metrics']['mse'] for r in all_windows_results]),
        'rmse': np.mean([r['test_metrics']['rmse'] for r in all_windows_results]),
        'mae': np.mean([r['test_metrics']['mae'] for r in all_windows_results]),
        'r2': np.mean([r['test_metrics']['r2'] for r in all_windows_results])
    }

    # บันทึกข้อมูลการเทรน
    json_log_file = os.path.join(log_dir, f"{model_name}_training_log_expanding.json")
    log_file = os.path.join(log_dir, f"{model_name}_training_log_expanding.txt")

    training_info = {
        "model_name": model_name,
        "expanding_window": True,
        "hyperparameters": {
            "input_size": input_size,
            "output_size": output_size,
            "hidden_size": hyperparams['hidden_size'],
            "num_layers": hyperparams['num_layers'],
            "dropout_rate": hyperparams['dropout_rate'],
            "batch_size": hyperparams['batch_size'],
            "learning_rate": hyperparams['learning_rate'],
            "num_epochs": hyperparams['num_epochs'],
            "sequence_length": hyperparams['sequence_length']
        },
        "total_training_time": training_time,
        "average_evaluation": {
            "train": avg_train_metrics,
            "val": avg_val_metrics,
            "test": avg_test_metrics
        },
        "window_results": window_summaries
    }

    # บันทึก log เป็นไฟล์ JSON
    with open(json_log_file, 'w') as f:
        json.dump(training_info, f, indent=4, default=default_converter)

    # บันทึก log ในรูปแบบข้อความ
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("========== GRU Model Training Log (Expanding Window) ==========\n\n")
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Sequence Length: {hyperparams['sequence_length']}\n")
        f.write("\nHyperparameters:\n")
        f.write(f"- Input Size: {input_size}\n")
        f.write(f"- Output Size: {output_size}\n")
        f.write(f"- Hidden Size: {hyperparams['hidden_size']}\n")
        f.write(f"- Number of Layers: {hyperparams['num_layers']}\n")
        f.write(f"- Dropout Rate: {hyperparams['dropout_rate']}\n")
        f.write(f"- Batch Size: {hyperparams['batch_size']}\n")
        f.write(f"- Learning Rate: {hyperparams['learning_rate']}\n")
        f.write(f"- Number of Epochs: {hyperparams['num_epochs']}\n")
        f.write(f"\nTotal Training Time: {training_time:.2f} seconds\n")

        # ข้อมูลเฉลี่ย
        f.write("\nAverage Evaluation Results:\n")
        f.write("Train Set:\n")
        f.write(f"- MSE: {avg_train_metrics['mse']:.4f}\n")
        f.write(f"- RMSE: {avg_train_metrics['rmse']:.4f}\n")
        f.write(f"- MAE: {avg_train_metrics['mae']:.4f}\n")
        f.write(f"- R^2: {avg_train_metrics['r2']:.4f}\n")

        f.write("\nValidation Set:\n")
        f.write(f"- MSE: {avg_val_metrics['mse']:.4f}\n")
        f.write(f"- RMSE: {avg_val_metrics['rmse']:.4f}\n")
        f.write(f"- MAE: {avg_val_metrics['mae']:.4f}\n")
        f.write(f"- R^2: {avg_val_metrics['r2']:.4f}\n")

        f.write("\nTest Set:\n")
        f.write(f"- MSE: {avg_test_metrics['mse']:.4f}\n")
        f.write(f"- RMSE: {avg_test_metrics['rmse']:.4f}\n")
        f.write(f"- MAE: {avg_test_metrics['mae']:.4f}\n")
        f.write(f"- R^2: {avg_test_metrics['r2']:.4f}\n")

        # ข้อมูลแต่ละหน้าต่าง
        f.write("\n========== Individual Window Results ==========\n")
        for i, window in enumerate(window_summaries):
            f.write(f"\nWindow {window['window_index'] + 1}:\n")
            f.write(f"Train Size: {window['data_sizes']['train']}\n")
            f.write(f"Validation Size: {window['data_sizes']['val']}\n")
            f.write(f"Test Size: {window['data_sizes']['test']}\n")
            f.write(f"Stopped at Epoch: {window['stopped_epoch']}\n")

            f.write("Train Metrics:\n")
            f.write(f"- MSE: {window['train_metrics']['mse']:.4f}\n")
            f.write(f"- RMSE: {window['train_metrics']['rmse']:.4f}\n")
            f.write(f"- MAE: {window['train_metrics']['mae']:.4f}\n")
            f.write(f"- R^2: {window['train_metrics']['r2']:.4f}\n")

            f.write("Validation Metrics:\n")
            f.write(f"- MSE: {window['val_metrics']['mse']:.4f}\n")
            f.write(f"- RMSE: {window['val_metrics']['rmse']:.4f}\n")
            f.write(f"- MAE: {window['val_metrics']['mae']:.4f}\n")
            f.write(f"- R^2: {window['val_metrics']['r2']:.4f}\n")

            f.write("Test Metrics:\n")
            f.write(f"- MSE: {window['test_metrics']['mse']:.4f}\n")
            f.write(f"- RMSE: {window['test_metrics']['rmse']:.4f}\n")
            f.write(f"- MAE: {window['test_metrics']['mae']:.4f}\n")
            f.write(f"- R^2: {window['test_metrics']['r2']:.4f}\n")

    return model_path, json_log_file, log_file


# =================== Bayesian Hyperparameter Tuning Functions ===================
def objective_function(all_data, input_size, y_scaler, fixed_params, tuning_dir, sequence_length, hidden_size,
                       num_layers, dropout_rate, batch_size, learning_rate, num_epochs, optimizer_type):
    """
    Objective function สำหรับการทำ Bayesian Optimization

    ฟังก์ชันนี้จะรับค่าพารามิเตอร์จาก Bayesian optimization และคืนค่า RMSE เฉลี่ยจากการเทรนโมเดลด้วย expanding window
    """

    # แปลงค่าพารามิเตอร์ที่ได้รับจากการ optimize
    sequence_length = int(sequence_length)
    hidden_size = int(hidden_size)
    num_layers = int(num_layers)
    batch_size = int(batch_size)
    num_epochs = int(num_epochs)

    # สร้างชุดพารามิเตอร์สำหรับการเทรนโมเดล
    hyperparams = {
        'sequence_length': sequence_length,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'dropout_rate': dropout_rate,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'output_size': fixed_params['output_size'],
        'optimizer_type': optimizer_type  # เพิ่ม optimizer_type
    }

    print(f"\n========== ทดสอบพารามิเตอร์ ==========")
    print(f"Sequence Length: {sequence_length}")
    print(f"Hidden Size: {hidden_size}")
    print(f"Num Layers: {num_layers}")
    print(f"Dropout Rate: {dropout_rate:.4f}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate:.6f}")
    print(f"Num Epochs: {num_epochs}")
    print(f"Optimizer Type: {optimizer_type}")  # เพิ่มการแสดงผล optimizer_type

    # ทดลองเทรนโมเดลด้วย expanding window
    all_windows_results = []

    for window_idx in range(MAX_EXPANDING_WINDOWS):
        # สร้าง data loader สำหรับแต่ละหน้าต่าง
        train_loader, val_loader, test_loader, data_sizes = create_expanding_window_split(
            all_data, window_idx, sequence_length, batch_size
        )

        # ตรวจสอบว่ายังมีข้อมูลเพียงพอหรือไม่
        if train_loader is None:
            print(f"ไม่มีข้อมูลเพียงพอสำหรับหน้าต่างที่ {window_idx + 1}")
            break

        # สร้างโมเดล GRU
        model = GRUModel(
            input_size,
            hidden_size,
            num_layers,
            fixed_params['output_size'],
            dropout_rate
        ).to(device)

        # สร้าง optimizer ตามประเภทที่กำหนด
        if optimizer_type == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        elif optimizer_type == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            # ค่าเริ่มต้น
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        criterion = nn.MSELoss()

        # เทรนและประเมินโมเดล
        (early_stopping, train_losses, val_losses,
         train_metrics, val_metrics, test_metrics, training_time,
         train_predictions, train_actuals, val_predictions, val_actuals,
         test_predictions, test_actuals, data_sizes, stopped_epoch) = train_and_evaluate_model_expanding(
            model, train_loader, val_loader, test_loader, criterion, optimizer,
            num_epochs, EARLY_STOP_PATIENCE,
            data_sizes, y_scaler, window_idx
        )

        # เก็บผลลัพธ์
        window_result = {
            'window_index': window_idx,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'stopped_epoch': stopped_epoch
        }

        all_windows_results.append(window_result)

    # คำนวณค่าเฉลี่ย RMSE และ R² สำหรับการ optimize
    if len(all_windows_results) > 0:
        avg_test_rmse = np.mean([r['test_metrics']['rmse'] for r in all_windows_results])
        avg_test_r2 = np.mean([r['test_metrics']['r2'] for r in all_windows_results])
        print(f"ค่าเฉลี่ย Test RMSE: {avg_test_rmse:.4f}, R²: {avg_test_r2:.4f}")

        # บันทึกผลการทดลองใน log
        log_result = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'hyperparams': hyperparams,
            'avg_test_rmse': float(avg_test_rmse),
            'avg_test_r2': float(avg_test_r2),
            'n_windows': len(all_windows_results)
        }

        # บันทึก log ลงไฟล์
        log_file = os.path.join(tuning_dir, "bayesian_tuning_log.json")
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = {'trials': []}

        logs['trials'].append(log_result)

        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=4, default=default_converter)

        return avg_test_rmse
    else:
        print("ไม่มีผลลัพธ์จากการเทรน")
        return float('inf')  # ถ้าไม่มีผลลัพธ์ให้คืนค่า RMSE เป็นอนันต์


def run_bayesian_optimization(data_args, fixed_params, tuning_dir, n_calls=120):
    """
    ทำการ Bayesian Hyperparameter Tuning สำหรับโมเดล GRU

    Args:
        data_args (tuple): ข้อมูลที่ใช้ในการเทรนโมเดล
        fixed_params (dict): พารามิเตอร์ที่กำหนดค่าคงที่
        tuning_dir (str): โฟลเดอร์สำหรับเก็บผลลัพธ์การ tuning
        n_calls (int): จำนวนครั้งที่จะเรียกใช้ objective function

    Returns:
        list: รายการผลลัพธ์การ tuning เรียงตามค่า RMSE ที่ดีที่สุด
    """
    os.makedirs(tuning_dir, exist_ok=True)

    # แยกข้อมูลจาก data_args
    (X_encoded, y_scaled, y_raw, numeric_scaler, label_encoders,
     y_scaler, input_size, all_data) = data_args

    space = [
        Integer(3, 30, name='sequence_length'),
        Integer(16, 512, name='hidden_size'),
        Integer(1, 5, name='num_layers'),
        Real(0.0, 0.2, name='dropout_rate'),
        Integer(8, 256, name='batch_size'),
        Real(0.0001, 0.001, name='learning_rate'),
        Integer(50, 200, name='num_epochs'),
        Categorical(['Adam', 'AdamW', 'SGD'], name='optimizer_type')  # เพิ่ม optimizer type
    ]

    # สร้าง objective function แบบ wrapper ที่ใช้กับ gp_minimize
    @use_named_args(space)
    def objective_wrapper(sequence_length, hidden_size, num_layers, dropout_rate, batch_size, learning_rate,
                          num_epochs, optimizer_type):
        return objective_function(
            all_data, input_size, y_scaler, fixed_params, tuning_dir,
            sequence_length, hidden_size, num_layers, dropout_rate, batch_size, learning_rate, num_epochs,
            optimizer_type
        )

    # เริ่มการ optimize
    print("\n========== เริ่มการทำ Bayesian Hyperparameter Tuning ==========")
    print(f"จำนวนการเรียกใช้ objective function: {n_calls}")

    # ทำการ optimize
    result = gp_minimize(
        objective_wrapper,
        space,
        n_calls=n_calls,
        random_state=RANDOM_SEED,
        n_random_starts=20,  # จำนวนจุดสุ่มเริ่มต้น
        verbose=True
    )

    # บันทึกผลลัพธ์การ optimize
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(tuning_dir, f"bayesian_optimization_result_{timestamp}.json")

    # แปลงผลลัพธ์เป็น dictionary
    result_dict = {
        'x': result.x,
        'fun': float(result.fun),
        'x_iters': [list(x) for x in result.x_iters],
        'func_vals': [float(y) for y in result.func_vals],
        'space': [str(dim) for dim in result.space.dimensions],
        'random_state': result.random_state
    }

    # บันทึกผลลัพธ์ลงไฟล์
    with open(result_file, 'w') as f:
        json.dump(result_dict, f, indent=4, default=default_converter)

    # สร้างกราฟแสดงผลการ optimize
    plots_dir = os.path.join(tuning_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # กราฟแสดงการลู่เข้า
    plt.figure(figsize=(10, 6))
    plot_convergence(result)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"convergence_plot_{timestamp}.png"), dpi=300)
    plt.close()

    # กราฟแสดง importance ของแต่ละพารามิเตอร์
    plot_objective(result)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"objective_plot_{timestamp}.png"), dpi=300)
    plt.close()

    # กราฟแสดงการประเมินผล
    plot_evaluations(result)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"evaluations_plot_{timestamp}.png"), dpi=300)
    plt.close()

    # อ่านผลการทดลองทั้งหมดจาก log
    log_file = os.path.join(tuning_dir, "bayesian_tuning_log.json")
    with open(log_file, 'r') as f:
        logs = json.load(f)

    # เรียงผลลัพธ์ตาม R² (จากมากไปน้อย) และ RMSE (จากน้อยไปมาก) เป็นเกณฑ์รอง
    sorted_trials = sorted(logs['trials'],
                           key=lambda x: (-x['avg_test_r2'], x['avg_test_rmse']))

    # บันทึกผลลัพธ์ที่ดีที่สุด
    best_params = {
        'sequence_length': int(result.x[0]),
        'hidden_size': int(result.x[1]),
        'num_layers': int(result.x[2]),
        'dropout_rate': result.x[3],
        'batch_size': int(result.x[4]),
        'learning_rate': result.x[5],
        'num_epochs': int(result.x[6]),
        'optimizer_type': result.x[7]  # เพิ่ม optimizer_type
    }

    print("\n========== ผลการ Bayesian Hyperparameter Tuning ==========")
    print(f"พารามิเตอร์ที่ดีที่สุด: {best_params}")
    print(f"ค่า RMSE ที่ดีที่สุด: {result.fun:.4f}")

    # ดึง TOP_MODELS_TO_SAVE อันดับแรกเพื่อใช้ในการเทรนรอบที่ 2
    top_models = sorted_trials[:TOP_MODELS_TO_SAVE]

    # บันทึกข้อมูลโมเดลที่ดีที่สุด TOP_MODELS_TO_SAVE ตัว
    with open(os.path.join(tuning_dir, f"top_{TOP_MODELS_TO_SAVE}_models.json"), 'w') as f:
        json.dump(top_models, f, indent=4, default=default_converter)

    return top_models


def train_top_models(data_args, fixed_params, top_models, base_dir):
    """
    เทรนโมเดลที่ดีที่สุด TOP_MODELS_TO_SAVE ตัวอีกครั้งพร้อมบันทึกข้อมูลละเอียด

    Args:
        data_args (tuple): ข้อมูลที่ใช้ในการเทรนโมเดล
        fixed_params (dict): พารามิเตอร์ที่กำหนดค่าคงที่
        top_models (list): รายการข้อมูลโมเดลที่ดีที่สุด
        base_dir (str): โฟลเดอร์หลักสำหรับเก็บผลลัพธ์

    Returns:
        list: รายการผลลัพธ์การเทรนโมเดลที่ดีที่สุด
    """
    # สร้างโฟลเดอร์สำหรับเก็บโมเดลที่ดีที่สุด
    top_models_dir = os.path.join(base_dir, "top_models")
    os.makedirs(top_models_dir, exist_ok=True)

    # แยกข้อมูลจาก data_args
    (X_encoded, y_scaled, y_raw, numeric_scaler, label_encoders,
     y_scaler, input_size, all_data) = data_args

    # เตรียมตัวแปรสำหรับเก็บผลลัพธ์
    final_results = []

    # เทรนโมเดลที่ดีที่สุดทั้งหมด
    for i, model_info in enumerate(top_models):
        rank = i + 1
        params = model_info['hyperparams']

        # สร้างโฟลเดอร์สำหรับโมเดลนี้
        model_rank_dir = os.path.join(top_models_dir, f"rank_{rank}")
        models_dir = os.path.join(model_rank_dir, "models")
        log_dir = os.path.join(model_rank_dir, "logs")
        plots_dir = os.path.join(model_rank_dir, "plots")

        os.makedirs(model_rank_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        # สร้างชื่อโมเดล
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"GRU_Rank{rank}_seq{params['sequence_length']}_h{params['hidden_size']}_l{params['num_layers']}_{timestamp}"

        print(f"\n========== เทรนโมเดลอันดับที่ {rank} ==========")
        print(f"Hyperparameters: {params}")

        # เทรนโมเดลด้วยพารามิเตอร์นี้
        all_windows_results = []
        total_training_time = 0

        for window_idx in range(MAX_EXPANDING_WINDOWS):
            # สร้าง data loader สำหรับแต่ละหน้าต่าง
            train_loader, val_loader, test_loader, data_sizes = create_expanding_window_split(
                all_data, window_idx, params['sequence_length'], params['batch_size']
            )

            # ตรวจสอบว่ายังมีข้อมูลเพียงพอหรือไม่
            if train_loader is None:
                print(f"ไม่มีข้อมูลเพียงพอสำหรับหน้าต่างที่ {window_idx + 1}")
                break

            # สร้างโมเดล GRU
            model = GRUModel(
                input_size,
                params['hidden_size'],
                params['num_layers'],
                params['output_size'],
                params['dropout_rate']
            ).to(device)

            # สร้าง optimizer ตามประเภทที่กำหนด
            if 'optimizer_type' in params:
                if params['optimizer_type'] == 'Adam':
                    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
                elif params['optimizer_type'] == 'AdamW':
                    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'])
                elif params['optimizer_type'] == 'SGD':
                    optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=0.9)
                else:
                    # ค่าเริ่มต้น
                    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
            else:
                # กรณีไม่มีค่า optimizer_type ในพารามิเตอร์ (ใช้กับโมเดลเก่า)
                optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

            criterion = nn.MSELoss()

            # เทรนและประเมินโมเดล
            (early_stopping, train_losses, val_losses,
             train_metrics, val_metrics, test_metrics, training_time,
             train_predictions, train_actuals, val_predictions, val_actuals,
             test_predictions, test_actuals, data_sizes, stopped_epoch) = train_and_evaluate_model_expanding(
                model, train_loader, val_loader, test_loader, criterion, optimizer,
                params['num_epochs'], EARLY_STOP_PATIENCE,
                data_sizes, y_scaler, window_idx
            )

            # พล็อตผลลัพธ์ของแต่ละหน้าต่าง
            plot_path = plot_results_expanding(
                train_losses, val_losses, test_actuals, test_predictions,
                model_name, plots_dir, window_idx, params, test_metrics, stopped_epoch, show_plot=False
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
                'stopped_epoch': stopped_epoch,
                'plot_path': plot_path
            }

            all_windows_results.append(window_result)
            total_training_time += training_time

        # พล็อตผลรวมของทุกหน้าต่าง
        if len(all_windows_results) > 0:
            combined_plot_path = plot_combined_results(
                all_windows_results, model_name, plots_dir, params, show_plot=False
            )

            # เพิ่มการเรียกใช้ฟังก์ชันสำหรับพล็อตค่าเฉลี่ย losses
            avg_losses_plot_path = plot_average_losses(
                all_windows_results, model_name, plots_dir, params, show_plot=False
            )

        # บันทึกโมเดลและ log
        model_path, json_log_file, log_file = save_model_and_logs_expanding(
            model, optimizer,
            train_losses[-1] if len(train_losses) > 0 else 0,
            val_losses[-1] if len(val_losses) > 0 else 0,
            input_size, params['output_size'],
            params,
            numeric_scaler, label_encoders, y_scaler,
            all_windows_results,
            total_training_time, model_name,
            models_dir, log_dir
        )

        # คำนวณค่าเฉลี่ยเมตริกสำหรับการแสดงผล
        avg_test_rmse = np.mean([r['test_metrics']['rmse'] for r in all_windows_results])
        avg_test_r2 = np.mean([r['test_metrics']['r2'] for r in all_windows_results])

        # สร้างข้อมูลสรุปของโมเดลนี้
        model_summary = {
            'rank': rank,
            'model_name': model_name,
            'hyperparams': params,
            'avg_metrics': {
                'avg_test_rmse': float(avg_test_rmse),
                'avg_test_r2': float(avg_test_r2)
            },
            'training_time': total_training_time,
            'model_path': model_path,
            'log_file': log_file,
            'json_log_file': json_log_file,
            'combined_plot_path': combined_plot_path if len(all_windows_results) > 0 else None,
            'avg_losses_plot_path': avg_losses_plot_path if 'avg_losses_plot_path' in locals() else None,
            # เพิ่มบรรทัดนี้
            'n_windows': len(all_windows_results)
        }

        final_results.append(model_summary)

        # บันทึกข้อมูลสรุปลงไฟล์
        summary_file = os.path.join(model_rank_dir, "model_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(model_summary, f, indent=4, default=default_converter)

    # เรียงลำดับโมเดลใหม่ตาม R² (จากมากไปน้อย) และ RMSE (จากน้อยไปมาก) เป็นเกณฑ์รอง
    final_results_sorted = sorted(final_results,
                                  key=lambda x: (-x['avg_metrics']['avg_test_r2'], x['avg_metrics']['avg_test_rmse']))

    # บันทึกข้อมูลสรุปรวมของโมเดลทั้งหมด (เรียงลำดับใหม่)
    summary_file = os.path.join(top_models_dir, "all_models_summary.json")
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'models': final_results_sorted
        }, f, indent=4, default=default_converter)

    # สร้างไฟล์สรุปการจัดอันดับใหม่
    reranked_summary_file = os.path.join(top_models_dir, "reranked_models_summary.json")

    # เตรียมข้อมูลสรุปการจัดอันดับใหม่
    reranked_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'original_ranking': [{'original_rank': model['rank'],
                              'model_name': model['model_name'],
                              'r2': model['avg_metrics']['avg_test_r2'],
                              'rmse': model['avg_metrics']['avg_test_rmse']}
                             for model in final_results],
        'new_ranking': [{'new_rank': i + 1,
                         'original_rank': model['rank'],
                         'model_name': model['model_name'],
                         'r2': model['avg_metrics']['avg_test_r2'],
                         'rmse': model['avg_metrics']['avg_test_rmse']}
                        for i, model in enumerate(final_results_sorted)]
    }

    # บันทึกสรุปการจัดอันดับใหม่
    with open(reranked_summary_file, 'w') as f:
        json.dump(reranked_data, f, indent=4, default=default_converter)

    # แสดงผลการจัดอันดับใหม่
    print("\n========== การจัดอันดับโมเดลใหม่ตาม R² และ RMSE ==========")
    for i, model in enumerate(final_results_sorted):
        print(f"อันดับใหม่ #{i + 1}: อันดับเดิม #{model['rank']}, {model['model_name']}")
        print(f"    R² = {model['avg_metrics']['avg_test_r2']:.4f}, RMSE = {model['avg_metrics']['avg_test_rmse']:.4f}")

    return final_results_sorted


# =================== Main Execution Function ===================
def run_bayesian_tuning_and_train_top_models(csv_file_path, fixed_params=None, n_calls=120):
    """
    ฟังก์ชันหลักสำหรับรันการ Bayesian Hyperparameter Tuning และเทรนโมเดลที่ดีที่สุด

    Args:
        csv_file_path (str): พาธของไฟล์ CSV
        fixed_params (dict, optional): พารามิเตอร์ที่กำหนดค่าคงที่
        n_calls (int): จำนวนครั้งที่จะเรียกใช้ objective function
    """
    if fixed_params is None:
        fixed_params = {
            'output_size': 1,  # ค่าที่ต้องการทำนาย (PM0.1)
        }

    # สร้างไดเรกทอรีสำหรับเก็บผลลัพธ์
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = "PM01_GRU_Bayesian_Tuning"
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    tuning_dir = os.path.join(run_dir, "tuning")

    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(tuning_dir, exist_ok=True)

    # รายชื่อคุณลักษณะ (feature names)
    numeric_features = ['PM2.5nova', 'PM10nova', 'Humidity', 'Temperature']
    categorical_features = ['daytype', 'time_range']

    # เตรียมข้อมูล
    max_seq_length = 30  # กำหนดค่า sequence_length สูงสุดที่จะใช้ทดลอง
    data_args = prepare_data_expanding_window(
        csv_file_path,
        numeric_features,
        categorical_features,
        max_seq_length,
        initial_train_ratio=0.3,
        val_ratio=0.1,
        expand_step=0.1
    )

    # ทำ Bayesian Hyperparameter Tuning
    top_models = run_bayesian_optimization(
        data_args,
        fixed_params,
        tuning_dir,
        n_calls=n_calls
    )

    # เทรนโมเดลที่ดีที่สุด TOP_MODELS_TO_SAVE ตัว
    final_results = train_top_models(
        data_args,
        fixed_params,
        top_models,
        run_dir
    )

    print("\n========== เสร็จสิ้นการเทรนโมเดลทั้งหมด ==========")
    print(f"ผลลัพธ์ถูกบันทึกที่: {run_dir}")

    return final_results


# =================== Example Usage ===================
if __name__ == "__main__":
    # ตั้งค่าคงที่
    fixed_params = {
        'output_size': 1  # ค่าที่ต้องการทำนาย (PM0.1)
    }

    # รันการทำ Bayesian hyperparameter tuning และเทรนโมเดลที่ดีที่สุด
    file_path = 'calibrated_data.csv'  # แก้ไขเป็นพาธของไฟล์ของคุณ

    # จำนวนครั้งในการเรียกใช้ objective function สำหรับการหาค่าที่ดีที่สุด
    n_calls = 120  # ปรับตามความเหมาะสม (ยิ่งมากยิ่งแม่นยำแต่ใช้เวลานาน)

    results = run_bayesian_tuning_and_train_top_models(file_path, fixed_params, n_calls=n_calls)