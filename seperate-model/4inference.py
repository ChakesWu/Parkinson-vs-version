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

com = 'COM3'  # 替換為你的 Arduino 端口號

def collect_data_from_arduino():
    """從Arduino採集帶基準值的數據"""
    try:
        ser = Serial(com, 9600, timeout=2)
        ser.flushInput()
        
        # 增加初始化延遲以確保Arduino準備好
        time.sleep(3)
        
        # 發送啟動命令
        ser.write(b"START\n")
        print("已發送啟動命令")

        baseline = []  # 基準值記錄
        angle_series = []  # 存儲所有數據點
        final_angles = []  # 最終角度
        
        start_time = time.time()
        while (time.time() - start_time) < 30:  # 超時30秒
            line = ser.readline().decode('utf-8').strip()
            print(f"收到數據: '{line}'")  # 記錄所有接收到的數據
            
            if line.startswith("BASELINE:"):
                try:
                    baseline = list(map(float, line.split(':')[1].split(',')))
                    print(f"基準值: {baseline}")
                except ValueError as e:
                    print(f"基準值解析錯誤: {e}, 數據: {line}")
            elif line.startswith("DATA,"):
                try:
                    values = list(map(int, line.split(',')[1:6]))
                    angle_series.append(values)
                    final_angles = values  # 更新最終角度
                    print(f"角度數據: {values}")
                except ValueError as e:
                    print(f"角度數據解析錯誤: {e}, 數據: {line}")
            elif line == "END":
                print("數據接收完成")
                break
            else:
                print(f"未知數據格式: {line}")

        ser.close()
        if not angle_series:
            print("未接收到任何角度數據")
            return {'baseline': baseline, 'angle_series': [], 'final_angles': []}
        if not baseline:
            print("警告: 未接收到基準值，將使用最終角度繼續")
        if not final_angles:
            print("警告: 未設置最終角度，將使用最後一組角度數據")
            final_angles = angle_series[-1] if angle_series else []
        
        print(f"共收集到基準值: {len(baseline)} 個, 角度數據組: {len(angle_series)} 組")
        return {
            'baseline': baseline,
            'angle_series': angle_series,
            'final_angles': final_angles
        }
    
    except Exception as e:
        print(f"數據採集錯誤: {str(e)}")
        return {'baseline': [], 'angle_series': [], 'final_angles': []}

def generate_training_plan(angles):
    """根據彎曲角度數據生成訓練方案"""
    if not angles:
        print("警告: 角度數據為空，使用默認角度")
        angles = [45] * 50  # 默認值以避免錯誤
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

def generate_professional_plan(baseline, final_angles):
    """生成專業級手部康復方案，根據手指彎曲情況選擇一個計劃"""
    fingers = ['拇指', '食指', '中指', '無名指', '小指']
    
    # 計算彎曲幅度比例 (0~100%)
    if not baseline or not final_angles:
        print("警告: 基準值或最終角度缺失，使用默認方案")
        flex_ratio = [50] * 5  # 默認50%屈曲度
        avg_flex = 50
    else:
        flex_ratio = [
            min((final - base)/1023*100, 100)
            for base, final in zip(baseline, final_angles)
        ]
        avg_flex = sum(flex_ratio) / len(flex_ratio)  # 平均屈曲度
        print(f"平均屈曲度: {avg_flex:.2f}%")  # 調試輸出
    
    # 根據平均屈曲度選擇一個訓練計劃（10個區間）
    if avg_flex < 10:
        plan = {
            '名稱': '被動關節活動',
            '強度': '被動活動度70%',
            '動作': '治療師輔助下進行全手關節的輕柔屈伸',
            '次數': '每個關節5次 ×2循環',
            '禁忌': '避免任何不適或疼痛'
        }
    elif avg_flex < 20:
        plan = {
            '名稱': '掌指關節滑動',
            '強度': '被動活動度80%',
            '動作': '治療師輔助下進行單個手指的屈曲-伸展全範圍運動',
            '次數': '每個關節5次 ×2循環',
            '禁忌': '避免疼痛範圍內操作'
        }
    elif avg_flex < 30:
        plan = {
            '名稱': '波浪式屈伸',
            '強度': '30%力度',
            '動作': '從尾指到食指依次彎曲，形成波浪運動',
            '次數': '10次循環 ×3組',
            '節奏': '4秒/次',
            '作用': '改善手指分離運動能力'
        }
    elif avg_flex < 40:
        plan = {
            '名稱': '指尖對捏訓練',
            '強度': '拇指-食指50%力度',
            '動作': '模擬捏硬幣動作，保持精準控制',
            '次數': '15次/組 ×3組',
            '組間休息': '60秒',
            '注意事項': '強調動作準確性而非速度'
        }
    elif avg_flex < 50:
        plan = {
            '名稱': '基礎等長收縮',
            '強度': '60% MVE',
            '動作': '五指同時握球狀維持',
            '次數': '3組×30秒',
            '組間休息': '90秒',
            '進階標準': '能輕鬆完成時換用硬質握力器'
        }
    elif avg_flex < 60:
        plan = {
            '名稱': '彈性帶抗阻伸指',
            '強度': '黃色阻力帶(2-4磅)',
            '動作': '五指同時對抗彈性帶阻力伸展',
            '次數': '12次 ×3組',
            '角度控制': '伸展終末端保持2秒'
        }
    elif avg_flex < 70:
        plan = {
            '名稱': '圓柱體抓握',
            '強度': f'{int(avg_flex*0.8)}% 最大屈曲度',
            '動作': '全手握直徑5cm圓柱，保持功能性抓握姿勢',
            '次數': '維持1分鐘 ×3組',
            '器材': '使用有紋理表面的防滑圓柱'
        }
    elif avg_flex < 80:
        plan = {
            '名稱': '模擬擰瓶蓋',
            '強度': '三指(拇/食/中)70%力度',
            '動作': '旋轉直徑3cm的調節鈕',
            '次數': '順時針/逆時針各10圈 ×2組',
            '設備': '使用可調節阻力的訓練裝置'
        }
    elif avg_flex < 90:
        plan = {
            '名稱': '紋理辨識抓握',
            '強度': '自然力度',
            '動作': '閉眼狀態下抓握不同紋理的物體並辨識',
            '次數': '8種物體 ×3輪',
            '作用': '增強本體感覺輸入'
        }
    else:  # avg_flex >= 90
        plan = {
            '名稱': '漸進式握持耐力',
            '強度': '40% MVC',
            '動作': '持續握持適應性物體',
            '次數': '從1分鐘開始，每3天增加15秒',
            '目標': '最終達到持續5分鐘握持'
        }
    
    return plan

def send_to_arduino(angles):
    """將角度數據發送到 Arduino"""
    try:
        ser = Serial(com, 9600, timeout=1)
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
    # 使用OpenVINO進行推理加速
    from openvino.tools.pytorch import convert_model
    from openvino.runtime import Core
    
    # 加載原始PyTorch模型
    model = TransferLearningModel()
    checkpoint_path = "models/best_model.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("最佳模型權重加載成功")
    else:
        print("未找到最佳模型權重，使用隨機初始化的模型")
    
    model.eval()
    
    # 轉換為OpenVINO格式
    ov_model = convert_model(model, example_input=torch.randn(1, 5, 100))  # 假設輸入形狀為(1,5,100)
    
    # 初始化OpenVINO核心
    core = Core()
    compiled_model = core.compile_model(ov_model, "GPU")  # 使用Intel GPU
    baseline = []
    final_angles = []
    servo_angles = []

    if use_arduino:
        data = collect_data_from_arduino()
        if not data['angle_series']:
            print("錯誤: 未接收到角度數據，無法生成方案")
            return {"error": "未接收到 Arduino 數據"}
        # 扁平化時間序列數據
        all_angles = [angle for reading in data['angle_series'] for angle in reading]
        servo_angles = generate_training_plan(all_angles[-50:])
        baseline = data['baseline']
        final_angles = data['final_angles']
    else:
        processed_data = kinematic_feature_engineering(pd.DataFrame(input_data))
        dataset = ParkinsonDataset(processed_data)
        loader = DataLoader(dataset, batch_size=1)
        with torch.no_grad():
            inputs, _ = next(iter(loader))
            # 使用OpenVINO進行推理
            inputs_np = inputs.numpy()  # 轉換為numpy數組
            results = compiled_model([inputs_np])[0]
            servo_angles = results[0]  # 獲取第一個輸出
            if np.isnan(servo_angles).any():
                raise ValueError("模型輸出包含 NaN")

    finger_names = ['拇指', '食指', '中指', '無名指', '小指']
    basic_plan = {
        '彎曲角度': [f"{finger_names[i]}: {int(servo_angles[i])} 度" for i in range(5)],
        '注意事項': [
            "訓練前後進行10分鐘熱敷/冷敷",
            "每個動作間隔休息2分鐘",
            "如出現疼痛或疲勞立即停止"
        ]
    }

    professional_plan = {}
    if use_arduino:
        # 顯示角度分析
        if baseline and final_angles:
            print("\n=== 手指角度分析 ===")
            for i, name in enumerate(finger_names):
                print(f"{name}: 初始基準={baseline[i]:.1f} → 最終角度={final_angles[i]}")
        else:
            print("警告: 缺少基準值或最終角度，無法顯示角度分析")
        
        # 生成專業方案
        professional_plan = generate_professional_plan(baseline, final_angles)

    # 構建最終輸出
    result = {
        '基礎訓練方案': basic_plan,
        '專業進階方案': professional_plan
    }

    send_to_arduino(servo_angles)
    return result

if __name__ == "__main__":
    # 使用 Arduino 數據生成方案
    plan = predict_rehabilitation_plan(use_arduino=True)
    print("\n生成的帕金森手部訓練方案：")
    for key, value in plan.items():
        print(f"- {key}:")
        if key == '基礎訓練方案':
            for sub_key, sub_value in value.items():
                print(f"  * {sub_key}:")
                for item in sub_value:
                    print(f"    - {item}")
        elif key == '專業進階方案':
            if value:
                print("  * 訓練計劃:")
                for plan_key, plan_value in value.items():
                    print(f"    - {plan_key}: {plan_value}")
            else:
                print("  * 無專業方案（數據不足）")
