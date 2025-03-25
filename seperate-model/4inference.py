# -*- coding: utf-8 -*-
"""推理與 Arduino 控制模塊"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from serial import Serial
import time
from model import TransferLearningModel, ParkinsonDataset
from data_processing import kinematic_feature_engineering

def collect_data_from_arduino():
    """從 Arduino 收集10秒的彎曲角度數據"""
    try:
        ser = Serial('COM6', 9600, timeout=1)
        time.sleep(2)
        ser.write(b"START\n")
        print("已發送 START 命令")

        data_received = False
        while not data_received:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                if line.startswith("DATA:"):
                    data_str = line[5:]
                    angles = list(map(float, data_str.split(',')[:-1]))
                    print(f"接收到 {len(angles)} 個角度數據")
                    data_received = True
        ser.close()
        return angles
    except Exception as e:
        print(f"從 Arduino 收集數據時出錯: {str(e)}")
        return []

def generate_training_plan(angles):
    """根據彎曲角度數據生成訓練方案"""
    avg_angle = sum(angles) / len(angles)
    bending_speed = (max(angles) - min(angles)) / 10.0

    normal_speed = 15.0
    normal_max_angle = 90.0

    if bending_speed < 5.0:
        increment = 5
    elif bending_speed < 10.0:
        increment = 10
    else:
        increment = 15

    base_angle = min(avg_angle + increment, normal_max_angle)
    servo_angles = [
        int(base_angle * 0.5), int(base_angle * 0.7), int(base_angle * 0.9),
        int(base_angle * 0.7), int(base_angle * 0.5)
    ]
    return servo_angles

def send_to_arduino(angles):
    """將角度數據發送到 Arduino"""
    try:
        ser = Serial('COM6', 9600, timeout=1)
        time.sleep(2)
        ser.flushInput()
        angle_str = "ANGLES:" + ",".join(map(str, angles)) + "\n"
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

def predict_rehabilitation_plan(input_data=None, use_arduino=False):
    """生成每隻手指的彎曲角度並發送到 Arduino"""
    device = torch.device("cpu")
    model = TransferLearningModel().to(device)
    checkpoint_path = "models/best_model.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("最佳模型權重加載成功")
    else:
        print("未找到最佳模型權重，使用隨機初始化的模型")

    model.eval()
    if use_arduino:
        angles = collect_data_from_arduino()
        if not angles:
            return {"error": "未接收到 Arduino 數據"}
        servo_angles = generate_training_plan(angles)
    else:
        processed_data = kinematic_feature_engineering(pd.DataFrame(input_data))
        dataset = ParkinsonDataset(processed_data)
        loader = DataLoader(dataset, batch_size=1)
        with torch.no_grad():
            inputs, _ = next(iter(loader))
            inputs = inputs.to(device)
            servo_angles = model(inputs).cpu().numpy()[0]
            if np.isnan(servo_angles).any():
                raise ValueError("模型輸出包含 NaN")

    finger_names = ['拇指', '食指', '中指', '無名指', '小指']
    plan = {
        '手指鍛煉角度': [f"{finger_names[i]}: {int(servo_angles[i])} 度" for i in range(5)],
        '注意事項': [
            "訓練前後進行10分鐘熱敷/冷敷",
            "每個動作間隔休息2分鐘",
            "如出現疼痛或疲勞立即停止"
        ]
    }
    send_to_arduino(servo_angles)
    return plan

if __name__ == "__main__":
    # 使用 Arduino 數據生成方案
    plan = predict_rehabilitation_plan(use_arduino=True)
    print("\n生成的帕金森手部訓練方案：")
    for key, value in plan.items():
        if isinstance(value, list):
            print(f"- {key}:")
            for item in value:
                print(f"  * {item}")
        else:
            print(f"- {key}: {value}")