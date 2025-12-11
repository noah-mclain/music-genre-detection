import pytest
import torch
from pathlib import Path

@pytest.fixture
def sample_batch() -> torch.Tensor:
    return torch.randn(8, 1, 128, 130) # (batch_size, channel, n_mels, seg_len)

@pytest.fixture
def data_dir() -> Path:
    return Path("./src/Data/genres_original")

@pytest.fixture
def cache_dir() -> Path:
    return Path("./preprocessed_cache")