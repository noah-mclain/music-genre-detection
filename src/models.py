import logging
from typing import Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


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
            # torch.matmul(Q, K.transpose(-2, 1)) * self.scale
            torch.matmul(Q, K.transpose(-1, -2))
            * self.scale
        )  # (batch_size, num_heads, seq_len, seq_len)
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply attention weights to values and concatinate heads
        context = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_len, head_dim)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.hidden_dim)

        context = self.fc_out(context)
        # context = torch.matmul(attention_weights.mean(dim=[1,2], keepdim=True), context)
        # Global average pooling over time
        # context = context.squeeze(1)
        context = torch.mean(context, dim=1)  # (batch_size, hidden_dim)

        return context, attention_weights


class CNNLSTMAttentionModel(nn.Module):
    def __init__(
        self,
        num_genres: int = 10,
        num_lstm_layers: int = 1,
        hidden_dim: int = 128,
        num_attention_heads: int = 8,
        dropout: float = 0.3,
        lstm_dropout: float = 0.3,
        classifier_dropout: float = 0.45,
    ):
        super(CNNLSTMAttentionModel, self).__init__()

        self.num_genres = num_genres
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.dropout_rate = dropout

        self.cnn = nn.Sequential(
            # Block 1
            # 1, 32 -> start
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
            # nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # nn.Dropout2d(dropout),
        )

        # 256 * 16 * 16
        self.lstm_input_dim = 128 * 16  # Adjusted for 3 CNN blocks

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if num_lstm_layers > 1 else 0,
            bidirectional=True,
        )

        lstm_output_dim = hidden_dim * 2  # bidirectional
        self.lstm_dropout = nn.Dropout(lstm_dropout)

        logger.info(f"Model Configuration:")
        logger.info(f"  CNN output shape: (B, 128, 16, 16)")
        logger.info(f"  LSTM input_size: {self.lstm_input_dim} (128 * 16)")
        logger.info(f"  LSTM hidden_size: {hidden_dim}")
        logger.info(f"  LSTM output_dim (bidirectional): {lstm_output_dim}")
        logger.info(f"  Temporal sequence length: 16")

        self.attention = TemporalAttention(lstm_output_dim, num_heads=num_attention_heads)

        # Classification layer
        self.classifier = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(classifier_dropout),
            nn.Linear(lstm_output_dim, num_genres),
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = x.size(0)

        # CNN feature extraction
        cnn_out = self.cnn(x)  # (batch_size, 128, 16, 16)
        logger.debug(f"CNN output shape: {cnn_out.shape}")

        # Reshape for LSTM: flatten spatial dimensions
        B, C, H, W = cnn_out.shape
        cnn_out = cnn_out.permute(0, 2, 1, 3)  # (B, H, C, W)
        cnn_out = cnn_out.reshape(B, H, C * W)  # (B, 16, 2048)
        logger.debug(f"LSTM input shape: {cnn_out.shape}")

        lstm_out, _ = self.lstm(cnn_out)
        logger.debug(f"LSTM output shape: {lstm_out.shape}")
        lstm_out = self.lstm_dropout(lstm_out)
        # cnn_out = cnn_out.view(batch_size, -1, self.cnn_output_dim)
        # lstm_out, _ = self.lstm(cnn_out)  # (batch_size, seq_len, hidden_dim * 2)
        # batch_size, channels, height, width = cnn_out.shape
        # seq_len = height * width  # 64 timesteps
        # cnn_out = cnn_out.view(batch_size, seq_len, channels)  # (batch, 64, 256)
        # lstm_out, _ = self.lstm(cnn_out)  # (batch, 64, 256)
        # # Attention mechanism
        context, attention_weights = self.attention(lstm_out)  # (batch_size, hidden_dim * 2)
        logger.debug(f"Attention context shape: {context.shape}")

        # # Classification
        logits = self.classifier(context)  # (batch_size, num_genres)
        logger.debug(f"Logits shape: {logits.shape}")

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


class SimpleModel(nn.Module):
    def __init__(
        self,
        num_genres: int = 10,
        num_lstm_layers: int = 2,
        hidden_dim: int = 128,
        num_attention_heads: int = 4,
        dropout: float = 0.5,
    ):
        super(SimpleModel, self).__init__()

        self.num_genres = num_genres
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.dropout_rate = dropout

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, num_genres)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.global_pooling(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out
