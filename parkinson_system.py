# -*- coding: utf-8 -*-
"""Parkinson Rehabilitation System with Finger Angles and Arduino Control"""

import numpy as np
import pandas as pd
from scipy import signal
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from serial import Serial  # 正確導入 Serial

# ==================== 數據生成與加載模塊 ====================
def generate_base_dataset(samples=30000):
    """生成基準臨床數據集（100Hz採樣，5分鐘數據）"""
    np.random.seed(42)
    fs = 100
    t = np.linspace(0, 300, samples)
    tremor_freq = 4 + np.random.normal(0, 0.5)
    finger_angle = 90 + 10 * signal.sawtooth(2 * np.pi * tremor_freq * t)
    finger_angle += np.random.normal(0, 2, samples)
    acceleration = 0.8 * np.sin(2 * np.pi * 0.3 * t) * np.exp(-0.005*t)
    acceleration += 0.1 * np.random.randn(samples)
    emg_bursts = np.zeros(samples)
    for i in range(0, samples, 2000):
        burst = 0.5 * np.abs(signal.hilbert(np.random.randn(500)))
        emg_bursts[i:i+500] = burst
    emg = 0.4 * np.abs(signal.hilbert(np.random.randn(samples))) + emg_bursts
    labels = np.where(
        (np.std(finger_angle) > 8) &
        (np.mean(emg) > 0.45) &
        (np.max(acceleration) < 1.2),
        1, 0
    )
    df = pd.DataFrame({
        'timestamp': t,
        'finger_angle': finger_angle,
        'acceleration': acceleration,
        'emg': emg,
        'parkinson_label': labels
    })
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/generated_base_data.csv", index=False)
    print("基準數據集已生成，包含樣本數:", len(df))
    return df

def load_csv_dataset(file_path):
    """加載 CSV 數據集"""
    df = pd.read_csv(file_path)
    print("CSV 數據集已加載，包含樣本數:", len(df))
    print("列名:", df.columns.tolist())
    return df

def combine_datasets(generated_df, csv_df):
    """合併生成數據和 CSV 數據"""
    combined_df = pd.concat([generated_df, csv_df], ignore_index=True)
    combined_df['timestamp'] = combined_df['timestamp'].fillna(method='ffill')
    combined_df['parkinson_label'] = combined_df['parkinson_label'].fillna(0)
    print("合併後的數據集總樣本數:", len(combined_df))
    print("標籤分佈:\n", combined_df['parkinson_label'].value_counts())
    return combined_df

# ==================== 特徵工程模塊 ====================
def kinematic_feature_engineering(df):
    """運動學特徵增強"""
    df['angle_velocity'] = np.gradient(df['finger_angle'], df['timestamp'])
    df['angle_acceleration'] = np.gradient(df['angle_velocity'], df['timestamp'])
    freqs, psd = signal.welch(df['emg'], fs=100, nperseg=512)
    df['emg_peak_freq'] = freqs[np.argmax(psd)]
    df['emg_psd_ratio'] = psd[(freqs > 10) & (freqs < 35)].sum() / psd.sum()
    features = [
        'finger_angle', 'acceleration', 'emg',
        'angle_velocity', 'angle_acceleration',
        'emg_peak_freq', 'emg_psd_ratio'
    ]
    for feat in features:
        df[feat] = df[feat].replace([np.inf, -np.inf], np.nan).fillna(df[feat].mean())
    df[features] = (df[features] - df[features].mean()) / df[features].std()
    df[features] = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    df['rolling_angle_var'] = df['finger_angle'].rolling(window=100, center=True).var().fillna(0)
    final_features = [
        'finger_angle', 'acceleration', 'emg',
        'angle_velocity', 'angle_acceleration',
        'emg_peak_freq', 'emg_psd_ratio',
        'rolling_angle_var', 'timestamp',
        'parkinson_label'
    ]
    print("特徵工程後數據檢查:\n", df[final_features].isna().sum())
    return df[final_features]

# ==================== 模型架構 ====================
class PretrainedBioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(8, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.lstm = nn.LSTM(
            input_size=8,
            hidden_size=64,
            bidirectional=True,
            num_layers=2,
            batch_first=True
        )

    def forward(self, x):
        cnn_feat = self.cnn(x).squeeze(-1)
        lstm_input = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_input)
        lstm_feat = lstm_out[:, -1, :]
        return torch.cat([cnn_feat, lstm_feat], dim=1)

class TransferLearningModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PretrainedBioEncoder()
        self._initialize_weights()
        for param in list(self.encoder.parameters())[:4]:
            param.requires_grad = False
        self.adapter = nn.Sequential(
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 5)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.encoder(x)
        angles = self.adapter(features)
        return torch.sigmoid(angles) * 90

# ==================== 數據預處理 ====================
class ParkinsonDataset(Dataset):
    def __init__(self, df, seq_length=100):
        self.data = df.drop(columns=['timestamp', 'parkinson_label']).values
        self.labels = df['parkinson_label'].values
        self.seq_length = seq_length
        if self.data.shape[1] != 8:
            raise ValueError(f"輸入特徵數應為8，當前為{self.data.shape[1]}")

    def __len__(self):
        return len(self.data) // self.seq_length

    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length
        seq = self.data[start:end].T
        label = int(self.labels[start:end].mean() > 0.5)
        return torch.FloatTensor(seq), torch.tensor(label, dtype=torch.long)

# ==================== Arduino 數據收集與處理 ====================
def collect_emg_from_arduino():
    """從 Arduino 收集10秒的EMG信號數據"""
    try:
        ser = Serial('COM9', 115200, timeout=1)
        time.sleep(2)
        ser.write(b"START_EMG\n")
        print("已發送 START_EMG 命令")
        
        emg_data = []
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                if line == "END_OF_EMG":
                    print("收到結束標記，EMG數據收集完成")
                    break
                if line.startswith("EMG:"):
                    try:
                        emg_value = float(line.split("EMG:")[1])
                        emg_data.append(emg_value)
                    except ValueError:
                        print(f"無效EMG數據: {line}")
        
        ser.close()
        print(f"共收集到 {len(emg_data)} 個 EMG 數據點")
        return np.array(emg_data)
    except Exception as e:
        print(f"收集 EMG 數據時出錯: {str(e)}")
        return np.array([])

def collect_potentiometer_from_arduino():
    """從 Arduino 收集10秒的電位器數據（五個手指）"""
    try:
        ser = Serial('COM9', 115200, timeout=1)
        time.sleep(2)
        ser.write(b"START_POT\n")
        print("已發送 START_POT 命令")
        
        pot_data = []
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                if line == "END_OF_POT":
                    print("收到結束標記，電位器數據收集完成")
                    break
                if line.startswith("POT:"):
                    try:
                        values = list(map(float, line.split("POT:")[1].split(',')))
                        if len(values) == 5:
                            pot_data.append(values)
                        else:
                            print(f"電位器數據格式錯誤: {line}")
                    except ValueError:
                        print(f"無效電位器數據: {line}")
        
        ser.close()
        print(f"共收集到 {len(pot_data)} 個電位器數據點")
        return np.array(pot_data)  # 返回 shape: (n_samples, 5)
    except Exception as e:
        print(f"收集電位器數據時出錯: {str(e)}")
        return np.array([])

def potentiometer_to_angle(value, min_value=0, max_value=1023, min_angle=0, max_angle=90):
    """將電位器值轉換為角度"""
    return min_angle + (value - min_value) * (max_angle - min_angle) / (max_value - min_value)

def process_combined_data(emg_data, pot_data):
    """處理EMG和電位器數據，生成模型輸入"""
    # 時間戳假設為10秒，每100ms一個樣本
    t = np.arange(0, 10, 0.1)
    
    # 確保數據長度匹配
    min_length = min(len(emg_data), len(pot_data))
    emg_data = emg_data[:min_length]
    pot_data = pot_data[:min_length]
    t = t[:min_length]
    
    # 將電位器數據轉換為角度（使用中指作為代表）
    pot_angles = np.apply_along_axis(potentiometer_to_angle, 1, pot_data)
    middle_finger_angle = pot_angles[:, 2]  # 中指索引為2
    
    # 創建DataFrame
    combined_df = pd.DataFrame({
        'timestamp': t,
        'finger_angle': middle_finger_angle,
        'acceleration': np.zeros(min_length),  # 無加速度數據
        'emg': emg_data,
        'parkinson_label': np.zeros(min_length)
    })
    return combined_df

# ==================== 訓練與推理流程 ====================
def main():
    device = torch.device("cpu")
    print(f"使用設備: {device}")

    # 生成並訓練模型
    print("\n===== 正在生成數據 =====")
    generated_df = generate_base_dataset()

    print("\n===== 正在加載 CSV 數據 =====")
    csv_file_path = "data/base_data.csv"
    if not os.path.exists(csv_file_path):
        print(f"請將 base_data.csv 放入 {csv_file_path} 路徑")
        print("您可以從以下連結下載: https://drive.google.com/uc?id=1XWg7weCeZHIvUSVGX3TP_U23b5c70mAB")
        return
    csv_df = load_csv_dataset(csv_file_path)

    print("\n===== 正在合併數據 =====")
    combined_df = combine_datasets(generated_df, csv_df)

    print("\n===== 正在處理特徵 =====")
    processed_df = kinematic_feature_engineering(combined_df)
    print("處理後的特徵維度:", processed_df.shape)

    print("\n===== 正在劃分數據集 =====")
    train_size = int(0.8 * len(processed_df))
    train_dataset = ParkinsonDataset(processed_df.iloc[:train_size])
    val_dataset = ParkinsonDataset(processed_df.iloc[train_size:])
    print(f"訓練集樣本數: {len(train_dataset)}")
    print(f"驗證集樣本數: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    print("\n===== 正在初始化模型 =====")
    model = TransferLearningModel().to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-5
    )
    criterion = nn.MSELoss()

    print("\n===== 開始訓練 =====")
    best_loss = float('inf')
    os.makedirs("models", exist_ok=True)
    for epoch in range(20):
        model.train()
        total_loss = 0.0
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            target_angles = torch.rand(outputs.shape) * 90
            target_angles = target_angles.to(device)
            loss = criterion(outputs, target_angles)
            if torch.isnan(loss):
                print("警告：損失值為 nan，檢查輸入數據")
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                target_angles = torch.rand(outputs.shape) * 90
                target_angles = target_angles.to(device)
                val_loss += criterion(outputs, target_angles).item()

        val_loss /= len(val_loader)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/20 | 平均損失: {avg_loss:.4f} | 驗證損失: {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, "models/best_model.pth")
            print("新的最佳模型已保存")

    # 1. 采集EMG數據
    print("\n===== 采集EMG數據 =====")
    emg_data = collect_emg_from_arduino()
    if len(emg_data) == 0:
        print("未接收到EMG數據，退出")
        return

    # 2. 采集電位器數據
    print("\n===== 采集電位器數據 =====")
    pot_data = collect_potentiometer_from_arduino()
    if len(pot_data) == 0:
        print("未接收到電位器數據，退出")
        return

    # 3. 處理數據並生成模型輸入
    print("\n===== 處理EMG和電位器數據 =====")
    combined_df = process_combined_data(emg_data, pot_data)
    processed_df = kinematic_feature_engineering(combined_df)

    # 4. 模型推理
    dataset = ParkinsonDataset(processed_df)
    loader = DataLoader(dataset, batch_size=1)
    checkpoint_path = "models/best_model.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("最佳模型權重加載成功")
    model.eval()
    with torch.no_grad():
        inputs, _ = next(iter(loader))
        inputs = inputs.to(device)
        angles = model(inputs).cpu().numpy()[0]

    # 5. 生成訓練結果
    finger_names = ['拇指', '食指', '中指', '無名指', '小指']
    training_result = {
        '手指鍛煉角度': [f"{finger_names[i]}: {int(angles[i])} 度" for i in range(5)],
        '注意事項': [
            "訓練前後進行10分鐘熱敷/冷敷",
            "每個動作間隔休息2分鐘",
            "如出現疼痛或疲勞立即停止"
        ]
    }
    print("\n===== 生成的訓練結果 =====")
    for key, value in training_result.items():
        if isinstance(value, list):
            print(f"- {key}:")
            for item in value:
                print(f"  * {item}")
        else:
            print(f"- {key}: {value}")

    # 6. 發送結果到Arduino
    send_to_arduino(angles)

def send_to_arduino(angles):
    """將角度數據發送到 Arduino"""
    try:
        print("正在嘗試連接到 COM9...")
        ser = Serial('COM9', 115200, timeout=1)  # 使用與Arduino一致的波特率
        time.sleep(2)
        ser.flushInput()
        angle_str = "ANGLES:" + ",".join(map(str, angles)) + "\n"
        print(f"準備發送的數據: {angle_str.strip()}")
        ser.write(angle_str.encode('utf-8'))
        print("數據發送成功！")
        
        time.sleep(2)
        for _ in range(10):
            if ser.in_waiting > 0:
                response = ser.readline().decode('utf-8').strip()
                print(f"Arduino 回應: {response}")
            time.sleep(0.1)
        ser.close()
    except Exception as e:
        print(f"發送到 Arduino 時出錯: {str(e)}")

if __name__ == "__main__":
    main()