# -*- coding: utf-8 -*-
"""模型定義模塊"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset

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