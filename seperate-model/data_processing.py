# -*- coding: utf-8 -*-
"""數據生成與處理模塊 (集成NinaPro DB2 + UNICAMP數據)"""

import numpy as np
import pandas as pd
from scipy import signal
import os
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

# 1. NinaPro DB2數據加載
def load_ninapro_data(data_root="E:\\parkinson\\dataset", subject_range=(1, 10)):
    """
    加載NinaPro DB2數據集
    :param data_root: 數據集根目錄
    :param subject_range: 受試者範圍 (起始, 結束+1)
    """
    all_data = []
    print(f"開始加載NinaPro DB2數據，路徑: {data_root}...")
    
    # 定義可能的目錄前綴
    possible_prefixes = ["DB2_s", "DB22_s"]
    
    for subject in range(subject_range[0], subject_range[1]):  # 從1到9
        subject_dir = None
        found = False
        
        # 嘗試不同的目錄前綴
        for prefix in possible_prefixes:
            test_dir = os.path.join(data_root, f"{prefix}{subject}")
            if os.path.exists(test_dir):
                subject_dir = test_dir
                found = True
                print(f"找到受試者 {subject} 目錄: {subject_dir}")
                break
        
        if not found:
            print(f"警告: 受試者{subject}目錄不存在，跳過")
            continue
            
        # 處理每個練習 (E1, E2, E3)
        for exercise in range(1, 4):  # 練習1-3
            # 文件命名格式: S3_E1_A1.mat
            file_name = f"S{subject}_E{exercise}_A1.mat"
            file_path = os.path.join(subject_dir, file_name)
            
            if not os.path.exists(file_path):
                print(f"文件不存在: {file_path}")
                continue
            
            try:
                print(f"加載文件: {file_path}")
                # 加載MAT文件
                data = sio.loadmat(file_path)
                
                # 提取數據 - 根據實際數據結構調整
                # NinaPro DB2的emg數據有12列：前8個是EMG，後4個是加速度計
                if 'emg' in data:
                    emg_full = data['emg']  # 完整的12通道數據
                    emg = emg_full[:, :8]   # 只取前8個EMG通道
                elif 'data' in data:
                    # 有些版本數據存儲在'data'字段中
                    emg_full = data['data']
                    emg = emg_full[:, :8]   # 只取前8個EMG通道
                else:
                    print(f"警告: {file_path} 中未找到emg或data字段")
                    continue
                
                # 提取stimulus標籤
                if 'stimulus' in data:
                    stimulus = data['stimulus'].flatten()
                elif 'restimulus' in data:
                    stimulus = data['restimulus'].flatten()
                else:
                    print(f"警告: {file_path} 中未找到stimulus字段")
                    stimulus = np.zeros(len(emg))
                
                # 創建DataFrame - 只包含8個EMG通道
                df = pd.DataFrame(emg, columns=[f'emg_{i}' for i in range(1, 9)])
                df['stimulus'] = stimulus
                df['subject'] = subject
                df['exercise'] = exercise
                df['repetition'] = 1  # 所有文件都是A1
                
                # 計算時間戳 - 假設採樣率為2000Hz
                sampling_rate = 2000.0
                df['timestamp'] = np.arange(len(df)) / sampling_rate
                
                all_data.append(df)
                print(f"已加載: {file_path} | 樣本數: {len(df)}")
                
            except Exception as e:
                print(f"加載錯誤 {file_path}: {str(e)}")
                import traceback
                traceback.print_exc()
    
    if not all_data:
        raise ValueError("未加載到任何NinaPro數據，請檢查路徑")
    
    ninapro_df = pd.concat(all_data, ignore_index=True)
    
    # 添加健康人標籤 (0=健康)
    ninapro_df['parkinson_label'] = 0
    ninapro_df['data_source'] = 'ninapro'
    
    print(f"NinaPro數據加載完成! 總樣本數: {len(ninapro_df)}")
    return ninapro_df

# 2. 生成UNICAMP帕金森數據 (模擬)
def generate_unicamp_data(num_patients=15, samples_per_patient=30000):
    """生成符合UNICAMP特性的帕金森EMG數據"""
    print("開始生成UNICAMP模擬數據...")
    all_data = []
    
    for patient in range(1, num_patients + 1):
        # 生成時間序列 (5分鐘數據，100Hz採樣)
        t = np.linspace(0, 300, samples_per_patient)
        
        # 帕金森特有震顫 (4-6Hz)
        tremor_freq = 4.5 + np.random.normal(0, 0.3)
        tremor = 0.7 * np.sin(2 * np.pi * tremor_freq * t)
        
        # 肌肉強直 (持續肌電活動)
        rigidity = 0.5 * (1 + np.sin(2 * np.pi * 0.1 * t))
        
        # 運動遲緩 (不規則爆發)
        bradykinesia_bursts = np.zeros(samples_per_patient)
        burst_intervals = np.random.randint(1500, 3000, size=20)
        burst_starts = np.cumsum(burst_intervals)
        
        for start in burst_starts:
            if start + 500 < samples_per_patient:
                burst = 0.8 * np.abs(signal.hilbert(np.random.randn(500)))
                bradykinesia_bursts[start:start + 500] = burst
        
        # 組合EMG信號 (8通道)
        emg_data = np.zeros((samples_per_patient, 8))
        for ch in range(8):
            # 基礎EMG + 帕金森特徵
            base_emg = 0.4 * np.abs(signal.hilbert(np.random.randn(samples_per_patient)))
            # 不同通道有不同強度
            channel_emg = base_emg + (0.6 * tremor) + (0.5 * rigidity) + (0.7 * bradykinesia_bursts)
            emg_data[:, ch] = channel_emg
        
        # 創建DataFrame
        df = pd.DataFrame(emg_data, columns=[f'emg_{i}' for i in range(1, 9)])
        df['tremor_freq'] = tremor_freq
        df['rigidity_level'] = rigidity
        df['bradykinesia'] = bradykinesia_bursts
        df['medication_state'] = np.random.choice([0, 1], size=samples_per_patient, p=[0.3, 0.7])  # 0=停藥, 1=服藥
        df['patient_id'] = patient
        df['timestamp'] = t
        df['parkinson_label'] = 1  # 帕金森標籤
        df['data_source'] = 'unicamp'
        
        all_data.append(df)
        print(f"已生成患者 {patient} 數據 | 樣本數: {samples_per_patient}")
    
    unicamp_df = pd.concat(all_data, ignore_index=True)
    print(f"UNICAMP模擬數據生成完成! 總樣本數: {len(unicamp_df)}")
    return unicamp_df

# 3. 數據合併與特徵工程
# 修改 combine_and_process 函数
def combine_and_process(ninapro_df, unicamp_df):
    print("开始合併數據集...")
    
    # 统一受试者标识列名
    ninapro_df = ninapro_df.rename(columns={'subject': 'subject_id'})
    unicamp_df = unicamp_df.rename(columns={'patient_id': 'subject_id'})
    
    # 添加受试者类型标签
    ninapro_df['subject_type'] = 'healthy'
    unicamp_df['subject_type'] = 'parkinson'
    
    # 选择共同列 + 新增的重要列
    essential_cols = ['subject_id', 'subject_type', 'parkinson_label', 
                      'timestamp', 'data_source']
    emg_cols = [f'emg_{i}' for i in range(1, 9)]
    
    combined_df = pd.concat([
        ninapro_df[essential_cols + emg_cols], 
        unicamp_df[essential_cols + emg_cols]
    ], ignore_index=True)
    
    print(f"合併完成! 總樣本數: {len(combined_df)}")
    
    # 修复这里的括号问题
    healthy_count = len(combined_df[combined_df['parkinson_label'] == 0])
    parkinson_count = len(combined_df[combined_df['parkinson_label'] == 1])
    
    print(f"健康樣本: {healthy_count}")
    print(f"帕金森樣本: {parkinson_count}")
    
    print("開始特徵工程...")
    
    # 1. 計算EMG統計特徵 (优化内存使用)
    window_size = 100
    for i in range(1, 9):
        col = f'emg_{i}'
        combined_df[f'{col}_mean'] = combined_df.groupby('subject_id')[col].transform(
            lambda x: x.rolling(window_size, min_periods=1).mean()
        )
        combined_df[f'{col}_std'] = combined_df.groupby('subject_id')[col].transform(
            lambda x: x.rolling(window_size, min_periods=1).std()
        )
        combined_df[f'{col}_diff'] = combined_df.groupby('subject_id')[col].diff()
    
    # 2. 頻域特徵 (优化计算效率)
    def calc_spectral_features(series):
        if len(series) < 100:  # 确保足够样本
            return 0.0, 0.0
        freqs = np.fft.rfftfreq(len(series), d=0.01)
        fft_vals = np.abs(np.fft.rfft(series))
        if len(freqs) == 0:
            return 0.0, 0.0
        dominant_freq = freqs[np.argmax(fft_vals)]
        tremor_band = (freqs >= 4) & (freqs <= 6)
        power_ratio = np.sum(fft_vals[tremor_band]) / np.sum(fft_vals) if np.sum(fft_vals) > 0 else 0
        return dominant_freq, power_ratio

    # 使用高效的分组应用
    spectral_features = combined_df.groupby('subject_id')['emg_1'].rolling(
        window=500, min_periods=100
    ).apply(
        lambda x: calc_spectral_features(x)[1],  # 只获取震颤功率比
        raw=False
    ).reset_index(level=0, drop=True)
    
    combined_df['tremor_power_ratio'] = spectral_features.reindex(combined_df.index, fill_value=0)
    
    # 3. 填充缺失值
    combined_df.fillna(method='ffill', inplace=True)
    combined_df.fillna(0, inplace=True)
    
    # 4. 特徵選擇
    features = emg_cols + [
        'emg_1_mean', 'emg_1_std', 'emg_1_diff',
        'emg_2_mean', 'emg_2_std', 'emg_2_diff',
        'tremor_power_ratio'
    ]
    
    target = 'parkinson_label'
    
    print(f"特徵工程完成! 最終特徵數: {len(features)}")
    return combined_df[features + [target, 'timestamp', 'data_source', 'subject_id']]

# 4. 保存處理後的數據
def save_processed_data(df, output_dir="data/processed"):
    """保存處理後的數據集"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存完整數據集
    full_path = os.path.join(output_dir, "full_dataset.csv")
    df.to_csv(full_path, index=False)
    print(f"完整數據集已保存至: {full_path}")
    
    # 保存Tensor格式 (供PyTorch使用)
    tensor_path = os.path.join(output_dir, "parkinson_dataset.pt")
    
    # 提取特徵和標籤
    features = df.drop(columns=['parkinson_label', 'timestamp', 'data_source']).values
    labels = df['parkinson_label'].values
    
    # 標準化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 保存為PyTorch Dataset
    class ParkinsonDataset(Dataset):
        def __init__(self, features, labels):
            self.features = torch.FloatTensor(features)
            self.labels = torch.LongTensor(labels)
            
        def __len__(self):
            return len(self.features)
            
        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]
    
    dataset = ParkinsonDataset(features_scaled, labels)
    torch.save({
        'dataset': dataset,
        'scaler': scaler,
        'feature_names': list(df.drop(columns=['parkinson_label', 'timestamp', 'data_source']).columns)
    }, tensor_path)
    
    print(f"PyTorch數據集已保存至: {tensor_path}")

# 主函數
def main():
    # 步驟1: 加載NinaPro數據 (9個受試者)
    ninapro_df = load_ninapro_data(data_root="E:\\parkinson\\dataset", subject_range=(1, 10))
    
    # 步驟2: 生成UNICAMP數據
    unicamp_df = generate_unicamp_data(num_patients=15)
    
    # 步驟3: 合併與處理
    processed_df = combine_and_process(ninapro_df, unicamp_df)
    
    # 步驟4: 保存數據
    save_processed_data(processed_df)
    
    print("所有數據處理完成!")

if __name__ == "__main__":
    main()