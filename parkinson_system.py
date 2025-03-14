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
import serial

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
            nn.Linear(64, 5)  # 輸出5個值，對應5隻手指的角度
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
        return torch.sigmoid(angles) * 90  # 限制角度在 0-90 度

# ==================== 數據預處理 ====================
class ParkinsonDataset(Dataset):
    def __init__(self, df, seq_length=3000):
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

# ==================== 訓練流程 ====================
def main():
    device = torch.device("cpu")
    print(f"使用設備: {device}")

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
    os.makedirs("models", exist_ok=True)  # 新增：確保 models 目錄存在
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
                print("Inputs:", inputs)
                print("Outputs:", outputs)
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

    print("\n===== 生成康復方案並發送到 Arduino =====")
    test_data = {
        'timestamp': np.linspace(0, 300, 3000),
        'finger_angle': 85 + 10 * np.sin(2 * np.pi * 5 * np.linspace(0, 1, 3000)),
        'acceleration': 0.6 * np.exp(-0.005 * np.linspace(0, 300, 3000)),
        'emg': 0.7 * np.abs(np.random.randn(3000)),
        'parkinson_label': 1
    }
    plan = predict_rehabilitation_plan(test_data, device)
    print("\n生成的帕金森手部訓練方案：")
    for key, value in plan.items():
        if isinstance(value, list):
            print(f"- {key}:")
            for item in value:
                print(f"  * {item}")
        else:
            print(f"- {key}: {value}")

# ==================== 推理模塊 ====================
def predict_rehabilitation_plan(input_data, device):
    """生成每隻手指的彎曲角度並發送到 Arduino"""
    try:
        model = TransferLearningModel().to(device)
        checkpoint_path = "models/best_model.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("最佳模型權重加載成功")
        else:
            print("未找到最佳模型權重，使用隨機初始化的模型")

        model.eval()
        processed_data = kinematic_feature_engineering(pd.DataFrame(input_data))
        dataset = ParkinsonDataset(processed_data)
        loader = DataLoader(dataset, batch_size=1)

        with torch.no_grad():
            inputs, _ = next(iter(loader))
            inputs = inputs.to(device)
            angles = model(inputs).cpu().numpy()[0]
            if np.isnan(angles).any():
                raise ValueError("模型輸出包含 NaN")

        finger_names = ['拇指', '食指', '中指', '無名指', '小指']
        plan = {
            '手指鍛煉角度': [f"{finger_names[i]}: {int(angles[i])} 度" for i in range(5)],
            '注意事項': [
                "訓練前後進行10分鐘熱敷/冷敷",
                "每個動作間隔休息2分鐘",
                "如出現疼痛或疲勞立即停止"
            ]
        }

        send_to_arduino(angles)
        return plan

    except Exception as e:
        print(f"生成方案時出錯: {str(e)}")
        return {"error": "無法生成訓練方案"}

def send_to_arduino(angles):
    try:
        print("正在尝试连接 COM3...")
        ser = serial.Serial('COM3', 9600, timeout=1)
        time.sleep(2)  # 等待 Arduino 初始化
        angle_str = ",".join(map(str, angles.astype(int))) + "\n"
        print(f"准备发送的数据: {angle_str.strip()}")  # 打印发送的数据
        ser.write(angle_str.encode('utf-8'))
        ser.close()
        print("数据发送成功！")
    except Exception as e:
        print(f"发送到 Arduino 时出错: {str(e)}")

if __name__ == "__main__":
    main()