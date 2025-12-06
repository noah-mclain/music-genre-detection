import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import librosa
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )

    def foward(self, lstm_output):
        attention_weights = self.attention(lstm_output)
        attention_weights = attention_weights.squeeze(-1)

        context = torch.bmm(
            attention_weights.unsqueeze(1), # (batch_size, 1, seq_len)
            lstm_output # (batch_size, seq_len, hidden_dim)
        )

        context = context.squeeze(1) # (batch_size, hidden_dim)
        return context, attention_weights
    
class CNNLSTMAttentionModel(nn.Module):
    def __init__(self, num_genres=10, num_lstm_layers=2, hidden_dim=128, dropout=0.5):
        super(CNNLSTMAttentionModel, self).__init__()

        self.num_genres = num_genres
        self.hidden_dim = hidden_dim

        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout(dropout),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout(dropout),

            # BLock 3
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout(dropout),
        )

        self.cnn_output_dim = 128 * 16 * 16

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True
        )

        lstm_output_dim = hidden_dim * 2 # bidirectional

        # Classification layer
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_genres)
        )
        
    def foward(self, x, return_attention=False):
        batch_size = x.size(0)

        # CNN feature extraction
        cnn_out = self.cnn(x) # (batch_size, 128, 16, 16)

        # Reshape for LSTM: flatten spatial dimensions
        cnn_out = cnn_out.view(batch_size, -1, self.cnn_output_dim)
        lstm_out, (h_c, c_n) = self.lstm(cnn_out) # (batch_size, seq_len, hidden_dim * 2)
        # Attention mechanism
        context, attention_weights = self.attention(lstm_out) # (batch_size, hidden_dim * 2)
        # Classification
        logits = self.classifier(context) # (batch_size, num_genres
        
        if return_attention:
            return logits, attention_weights
        return logits