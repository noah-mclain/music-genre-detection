import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference_utils import AudioAugmentation, AudioProcessor


class TestAudioProcessor:
    def test_to_tensor_basic(self, sample_mel_spec: np.ndarray) -> None:
        tensor = AudioProcessor.to_tensor(sample_mel_spec)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 1, 128, 130)  # (batch, channel, n_mels, seg_len)

    def test_to_tensor_no_channel(self, sample_mel_spec: np.ndarray) -> None:
        tensor = AudioProcessor.to_tensor(sample_mel_spec, add_channel=False, add_batch=True)

        assert tensor.shape == (128, 130)

    def test_to_tensor_no_batch(self, sample_mel_spec: np.ndarray) -> None:
        tensor = AudioProcessor.to_tensor(sample_mel_spec, add_channel=True, add_batch=False)

        assert tensor.shape == (1, 128, 130)

    def test_to_tensor_device_cpu(self, sample_mel_spec: np.ndarray) -> None:
        tensor = AudioProcessor.to_tensor(sample_mel_spec, device="cpu")

        assert tensor.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    def test_to_tensor_device_cuda(self, sample_mel_spec: np.ndarray) -> None:
        tensor = AudioProcessor.to_tensor(sample_mel_spec, device="cuda")

        assert tensor.device.type == "cuda"

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
    def test_to_tensor_device_mps(self, sample_mel_spec: np.ndarray) -> None:
        tensor = AudioProcessor.to_tensor(sample_mel_spec, device="mps")

        assert tensor.device.type == "mps"

    def tesr_to_tensor_dtype(self, sample_mel_spec: np.ndarray) -> None:
        tensor = AudioProcessor.to_tensor(sample_mel_spec)

        assert tensor.dtype == torch.float32


class TestAudioAugmentation:
    def test_add_noise(self, sample_audio: np.ndarray) -> None:
        augmented = AudioAugmentation.add_noise(sample_audio)

        assert augmented.shape == sample_audio.shape
        assert not np.allclose(augmented, sample_audio)

    def test_add_noise_factor(self, sample_audio: np.ndarray) -> None:
        augmented1 = AudioAugmentation.add_noise(sample_audio, noise_factor=0.001)
        augmented2 = AudioAugmentation.add_noise(sample_audio, noise_factor=0.01)

        diff1 = np.mean((augmented1 - sample_audio) ** 2)
        diff2 = np.mean((augmented2 - sample_audio) ** 2)

        assert diff2 > diff1

    def test_random_augment(self, sample_audio: np.ndarray) -> None:
        augmented = AudioAugmentation.random_augment(sample_audio)

        assert augmented.shape == sample_audio.shape

    def test_augmentation_preserves_shape(self, sample_audio: np.ndarray) -> None:
        original_shape = sample_audio.shape

        augmented_noise = AudioAugmentation.add_noise(sample_audio)
        assert augmented_noise.shape == original_shape

        augmented_pitch = AudioAugmentation.pitch_shift(sample_audio, sr=22050)
        assert augmented_pitch.shape == original_shape

        augmented_stretch = AudioAugmentation.time_stretch(sample_audio)
        assert augmented_stretch.shape == original_shape

    def test_multiple_augmentations(self, sample_audio: np.ndarray) -> None:
        augmented = sample_audio
        for _ in range(5):
            augmented = AudioAugmentation.random_augment(augmented)

        assert augmented.shape == sample_audio.shape


class TestAudioProcessorIntegration:
    def test_mel_spec_normalization(self) -> None:
        mel_spec = np.random.randn(128, 130).astype(np.float32)

        normalized = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)

        assert np.abs(normalized.mean()) < 1e-6
        assert np.abs(normalized.std() - 1.0) < 1e-6

    def test_tensor_conversion_pipeline(self, sample_mel_spec: np.ndarray) -> None:
        tensor = AudioProcessor.to_tensor(
            sample_mel_spec, add_channel=True, add_batch=True, device="cpu"
        )

        assert tensor.dim() == 4
        assert tensor.shape[0] == 1  # batch size
        assert tensor.shape[1] == 1  # channel size
        assert tensor.shape[2] == 128  # n_mels
        assert tensor.shape[3] == 130  # seg_len
        assert tensor.device.type == "cpu"

    def test_device_compatibility(self, sample_mel_spec: np.ndarray) -> None:
        devices_to_test = ["cpu"]
        if torch.cuda.is_available():
            devices_to_test.append("cuda")

        if torch.backends.mps.is_available():
            devices_to_test.append("mps")

        for device in devices_to_test:
            tensor = AudioProcessor.to_tensor(sample_mel_spec, device=device)
            assert tensor.device.type == device
