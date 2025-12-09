import pytest
import torch
import torch.nn as nn

from src.models import CNNLSTMAttentionModel, TemporalAttention


class TestTemporalAttention:
    def test_initialization(self) -> None:
        attention = TemporalAttention(hidden_dim=128, num_heads=4)
        assert attention.hidden_dim == 128
        assert attention.num_heads == 4
        assert attention.head_dim == 32  # 128 / 4

    def test_forward_pass(self, sample_batch: torch.Tensor) -> None:
        batch_size = 2
        seq_len = 10
        hidden_dim = 128

        lstm_output = torch.randn(batch_size, seq_len, hidden_dim)
        attention = TemporalAttention(hidden_dim=hidden_dim, num_heads=4)

        context, weights = attention(lstm_output)

        assert context.shape == (batch_size, hidden_dim)
        assert weights.shape == (batch_size, 4, seq_len, seq_len)

        def test_invalid_hidden_dim(self) -> None:
            with pytest.raises(AssertionError):
                TemporalAttention(hidden_dim=127, num_heads=4)

        def test_output_normalization(self) -> None:
            batch_size = 2
            seq_len = 10
            hidden_dim = 128

            lstm_output = torch.randn(batch_size, seq_len, hidden_dim)
            attention = TemporalAttention(hidden_dim=hidden_dim, num_heads=4)

            _, weights = attention(lstm_output)

            # Sum of attention weights for each query should be ~1 for each query
            weight_sum = weights.sum(dim=-1)
            assert torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-6)


class TestCNNLSTMAttentionModel:
    def test_model_initialization(self) -> None:
        model = CNNLSTMAttentionModel(num_genres=10)
        assert model.num_genres == 10
        assert model.hidden_dim == 128
        assert model.num_lstm_layers == 2

    def test_model_forward_pass(self, sample_batch: torch.Tensor) -> None:
        model = CNNLSTMAttentionModel(num_genres=10)
        logits = model(sample_batch)

        assert logits.shape == (8, 10)

    def test_model_forward_pass_with_attention(
        self, sample_batch: torch.Tensor
    ) -> None:
        model = CNNLSTMAttentionModel(num_genres=10)
        logits, attn_weights = model(sample_batch, return_attention=True)

        assert logits.shape == (8, 10)
        assert attn_weights is not None

    def test_model_summary(self) -> None:
        model = CNNLSTMAttentionModel(num_genres=10)
        summary = model.get_model_summary()

        assert "total_params" in summary
        assert "trainable_params" in summary
        assert "num_genres" in summary
        assert summary["num_genres"] == 10

    def test_model_eval_mode(self, sample_batch: torch.Tensor) -> None:
        model = CNNLSTMAttentionModel(num_genres=10)
        model.eval()

        with torch.no_grad():
            logits = model(sample_batch)

        assert logits.shape == (8, 10)

    def test_different_genre_counts(self, sample_batch: torch.Tensor) -> None:
        for num_genres in [5, 10, 20]:
            model = CNNLSTMAttentionModel(num_genres=10)
            optimizer = torch.optim.Adam(model.parameters())

            logits = model(sample_batch)
            labels = torch.randint(0, 10, (8,))

            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            loss.backward()

            # Check that gradients are non-zero
            for param in model.parameters():
                if param.grad is not None:
                    assert param.grad.abs().sum() > 0

    def test_model_device_compatibility(self) -> None:
        model = CNNLSTMAttentionModel(num_genres=10)

        sample = torch.randn(2, 1, 128, 130)
        logits = model(sample)
        assert logits.shape == (2, 10)

        # Test NVIDIA GPU if available
        if torch.cuda.is_available():
            model_cuda = model.to("cuda")
            sample_cuda = sample.to("cuda")
            logits_cuda = model_cuda(sample_cuda)
            assert logits.device.type == "cuda"

        # Test MPS (Apple Silicon) if available
        if torch.backends.mps.is_available():
            model_mps = model.to("mps")
            sample_mps = sample.to("mps")
            logits_mps = model_mps(sample_mps)
            assert logits.device.type == "mps"


class TestModelOutputs:
    def test_output_shape(self, sample_batch: torch.Tensor) -> None:
        model = CNNLSTMAttentionModel(num_genres=10)
        logits = model(sample_batch)
        assert logits.dim() == 2
        assert logits.size(0) == sample_batch.size(0)

    def test_output_is_logits(self, sample_batch: torch.Tensor) -> None:
        model = CNNLSTMAttentionModel(num_genres=10)
        logits = model(sample_batch)

        row_sum = logits.sum(dim=1)
        assert not torch.allclose(row_sum, torch.ones(8), atol=1e-4)

    def test_deterministic_eval(self, sample_batch: torch.Tensor) -> None:
        model = CNNLSTMAttentionModel(num_genres=10)
        model.eval()

        with torch.no_grad():
            out1 = model(sample_batch)
            out2 = model(sample_batch)

        assert torch.allclose(out1, out2)
