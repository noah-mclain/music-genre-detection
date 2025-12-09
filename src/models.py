import torch
import torch.nn as nn
from typing import Union, Tuple


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        self.scale = self.head_dim**-0.5

    def forward(self, lstm_output: torch.Tensor) -> tuple:
        batch_size = lstm_output.shape[0]
        seq_len = lstm_output.shape[1]

        Q = self.query(lstm_output)  # (batch, seq_len, hidden_dim)
        K = self.key(lstm_output)
        V = self.value(lstm_output)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # shape: (batch_size, num_heads, seq_len, head_dim)

        scores = (
            torch.matmul(Q, K.transpose(-2, 1)) * self.scale
        )  # (batch_size, num_heads, seq_len, seq_len)
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply attention weights to values and concatinate heads
        context = torch.matmul(
            attention_weights, V
        )  # (batch_size, num_heads, seq_len, head_dim)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.hidden_dim)

        context = self.fc_out(context)
        # Global average pooling over time
        context = torch.mean(context, dim=1)  # (batch_size, hidden_dim)

        return context, attention_weights


class CNNLSTMAttentionModel(nn.Module):
    def __init__(
        self,
        num_genres: int = 10,
        num_lstm_layers: int = 2,
        hidden_dim: int = 128,
        num_attention_heads: int = 4,
        dropout: float = 0.5,
    ):
        super(CNNLSTMAttentionModel, self).__init__()

        self.num_genres = num_genres
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.dropout_rate = dropout

        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout2d(dropout),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout2d(dropout),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout2d(dropout),
            # Block 4
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout2d(dropout),
        )

        self.cnn_output_dim = (
            256 * 8 * 8
        )  # Assuming input mel-spectrogram size is (1, 128, 130)

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True,
        )

        lstm_output_dim = hidden_dim * 2  # bidirectional

        self.attention = TemporalAttention(
            lstm_output_dim, num_heads=num_attention_heads
        )

        # Classification layer
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_genres),
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = x.size(0)

        # CNN feature extraction
        cnn_out = self.cnn(x)  # (batch_size, 128, 16, 16)

        # Reshape for LSTM: flatten spatial dimensions
        cnn_out = cnn_out.view(batch_size, -1, self.cnn_output_dim)
        lstm_out, _ = self.lstm(cnn_out)  # (batch_size, seq_len, hidden_dim * 2)

        # Attention mechanism
        context, attention_weights = self.attention(
            lstm_out
        )  # (batch_size, hidden_dim * 2)

        # Classification
        logits = self.classifier(context)  # (batch_size, num_genres)

        if return_attention:
            return logits, attention_weights
        return logits

    def get_model_summary(self) -> dict:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "num_genres": self.num_genres,
            "hidden_dim": self.hidden_dim,
            "lstm_layers": self.num_lstm_layers,
            "dropout_rate": self.dropout_rate,
        }
