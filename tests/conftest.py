from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def sample_batch() -> torch.Tensor:
    return torch.randn(8, 1, 128, 130)  # (batch_size, channel, n_mels, seg_len)


@pytest.fixture
def data_dir() -> Path:
    return Path("./src/Data/genres_original")


@pytest.fixture
def cache_dir() -> Path:
    return Path("./preprocessed_cache")


@pytest.fixture
def sample_mel_spec() -> np.ndarray:
    return np.random.randn(128, 130).astype(np.float32)


@pytest.fixture
def sample_audio() -> np.ndarray:
    return np.random.randn(22050).astype(np.float32)
