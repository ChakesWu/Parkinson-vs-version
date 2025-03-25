# -*- coding: utf-8 -*-
"""訓練模塊"""

import pandas as pd
import torch
import torch.nn as nn  # Add this line to import torch.nn
from torch.utils.data import DataLoader
import os
from model import TransferLearningModel, ParkinsonDataset

def train_model():
    device = torch.device("cpu")
    print(f"使用設備: {device}")

    # 加載處理後的數據
    processed_df = pd.read_csv("data/processed_data.csv")
    print("處理後的特徵維度:", processed_df.shape)

    # 劃分數據集
    train_size = int(0.8 * len(processed_df))
    train_dataset = ParkinsonDataset(processed_df.iloc[:train_size])
    val_dataset = ParkinsonDataset(processed_df.iloc[train_size:])
    print(f"訓練集樣本數: {len(train_dataset)}")
    print(f"驗證集樣本數: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # 初始化模型
    model = TransferLearningModel().to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-5
    )
    criterion = nn.MSELoss()  # Now this will work with nn imported

    # 訓練循環
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

if __name__ == "__main__":
    train_model()