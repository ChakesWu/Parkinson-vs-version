# -*- coding: utf-8 -*-
"""數據生成與處理模塊"""

import numpy as np
import pandas as pd
from scipy import signal
import os

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

if __name__ == "__main__":
    generated_df = generate_base_dataset()
    csv_df = load_csv_dataset("data/base_data.csv")
    combined_df = combine_datasets(generated_df, csv_df)
    processed_df = kinematic_feature_engineering(combined_df)
    processed_df.to_csv("data/processed_data.csv", index=False)
    print("處理後的數據已保存到 data/processed_data.csv")