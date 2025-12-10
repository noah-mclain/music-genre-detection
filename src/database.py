import torch
from torch.utils.data import Dataset
import numpy as np
import librosa
import os
from pathlib import Path
from typing import Tuple, List, Optional
import logging

from inference_utils import AudioProcessor

logger = logging.getLogger(__name__)


class GTZANDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        sr: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        duration: float = 30.0,
        segment_length: int = 130,
        augment: bool = True,
        cache_spectrograms: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        self.segment_length = segment_length
        self.augment = augment
        self.cache_spectrograms = cache_spectrograms

        # Discover genres and map to indices
        self.genres = sorted(
            [
                d
                for d in os.listdir(self.data_dir)
                if os.path.isdir(os.path.join(self.data_dir, d))
            ]
        )
        self.genre_to_idx = {g: idx for idx, g in enumerate(self.genres)}

        self.audio_files = []
        self.labels = []

        # Load audio file paths and labels
        for genre in self.genres:
            genre_dir = self.data_dir / genre
            audio_paths = (
                list(genre_dir.glob("*.au"))
                + list(genre_dir.glob("*.wav"))
                + list(genre_dir.glob("*.mp3"))
            )

            for audio_path in audio_paths:
                self.audio_files.append(str(audio_path))
                self.labels.append(self.genre_to_idx[genre])

            logger.info(
                f"Loaded GTZAN dataset: {len(self.audio_files)} files, {len(self.genres)} genres"
            )

        # Cache spectrograms if required
        if cache_spectrograms:
            self.spec_cache = {}
        else:
            self.spec_cache = None

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            audio_path = self.audio_files[index]
            label = self.labels[index]

            mel_spectrogram = self._extract_mel_spectrogram(audio_path)
            spec_tensor = torch.FloatTensor(mel_spectrogram).unsqueeze(0)

            label_tensor = torch.tensor(label, dtype=torch.long)
            return spec_tensor, label_tensor
        except Exception as e:
            logger.error(f"Error loading data at index {index}: {e}")
            raise IndexError(f"Could not load sample at index: {index}")

    def _extract_mel_spectrogram(self, audio_path: str) -> np.ndarray:
        # Check cache
        if self.spec_cache is not None and audio_path in self.spec_cache:
            return self.spec_cache[audio_path]

        # Use shared utility
        mel_spec = AudioProcessor.extract_mel_spectrogram(
            audio_path,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            duration=self.duration,
            segment_length=self.segment_length,
        )

        # Cache if enabled
        if self.spec_cache is not None:
            self.spec_cache[audio_path] = mel_spec

        return mel_spec

    def __len__(self):
        return len(self.audio_files)

    def _augment_spectrogram(self, mel_spec: np.ndarray) -> np.ndarray:
        if not self.augment:
            return mel_spec

        # Time masking
        if np.random.rand() < 0.5:
            max_mask_width = 50
            mask_width = np.random.randint(0, max_mask_width)
            mask_start = np.random.randint(0, mel_spec.shape[1] - mask_width)
            mel_spec = mel_spec.copy()
            mel_spec[:, mask_start : mask_start + mask_width] = mel_spec.mean()

        # Frequency masking
        if np.random.rand() < 0.5:
            max_mask_height = 10
            mask_height = np.random.randint(0, max_mask_height)
            mask_start = np.random.randint(0, mel_spec.shape[0] - mask_height)
            mel_spec = mel_spec.copy()
            mel_spec[mask_start : mask_start + mask_height, :] = mel_spec.mean()

        return mel_spec

    def get_num_genres(self) -> int:
        return len(self.genres)

    def get_genre_names(self) -> List[str]:
        return self.genres

    def get_dataset_info(self) -> dict:
        genre_counts = {}
        for label in self.labels:
            genre = self.genres[label]
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

        return {
            "total_samples": len(self.audio_files),
            "num_genres": len(self.genres),
            "genres": self.genres,
            "genre_distribution": genre_counts,
            "sample_rate": self.sr,
            "n_mels": self.n_mels,
            "duration_seconds": self.duration,
            "segment_length": self.segment_length,
            "augmentation": self.augment,
            "caching": self.spec_cache is not None,
        }
