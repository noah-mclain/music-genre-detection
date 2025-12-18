import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .inference_utils import AudioAugmentationTemporal, AudioProcessor

logger = logging.getLogger(__name__)


class GTZANDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        cache_dir: str = "src/preprocessed_cache",
        sr: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        duration: float = 30.0,
        segment_length: int = 130,
        augment: bool = True,
        preprocess_all: bool = True,
        # cache_spectrograms: bool = False,
    ):
        logger.info(f"Initializing GTZANDataset from {data_dir}")

        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        logger.debug(f"Cache directory: {self.cache_dir}")

        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        self.segment_length = segment_length
        self.augment = augment
        # self.cache_spectrograms = cache_spectrograms

        # Discover genres and map to indices
        logger.info("Discovering genres...")
        self.genres = sorted(
            [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        )
        logger.info(f"Found genres: {self.genres}")
        self.genre_to_idx = {g: idx for idx, g in enumerate(self.genres)}

        self.audio_files = []
        self.labels = []

        # Load audio file paths and labels
        logger.info("Loading audio file paths...")
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
                logger.debug(f"Genre '{genre}': {len(self.audio_files)} files")

        logger.info(
            f"Loaded GTZAN dataset: {len(self.audio_files)} files, {len(self.genres)} genres"
        )

        # Preprocess if requested
        if preprocess_all:
            self._preprocess_all()

    def _preprocess_all(self):
        metadata_file = self.cache_dir / "metadata.pkl"
        if metadata_file.exists():
            logger.info("Preprocessed cache found, skipping preprocessing")
            return

        logger.info("Preprocessing all audio files (this will take a few minutes)...")
        try:
            from tqdm import tqdm

            use_tqdm = True
        except ImportError:
            logger.warning("tqdm not installed, progress bar disabled")
            use_tqdm = False

        iterator = (
            tqdm(
                enumerate(self.audio_files),
                desc="Preprocessing",
                total=len(self.audio_files),
            )
            if use_tqdm
            else enumerate(self.audio_files)
        )

        for idx, audio_path in iterator:
            cache_file = self.cache_dir / f"spec_{idx}.npy"
            if not cache_file.exists():
                try:
                    mel_spec = self._extract_mel_spectrogram(audio_path)
                    np.save(cache_file, mel_spec)
                except Exception as e:
                    logger.error(f"Error preprocessing {audio_path}: {e}")
                    continue

        try:
            with open(metadata_file, "wb") as f:
                pickle.dump(
                    {
                        "genres": self.genres,
                        "genre_to_idx": self.genre_to_idx,
                        "num_files": len(self.audio_files),
                        "labels": self.labels,
                    },
                    f,
                )
            logger.info(f"Preprocessing complete! Cached {len(self.audio_files)} spectrograms")
            logger.info(f"Cache size: ~{len(self.audio_files) * 0.2:.1f}MB")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            raise

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            audio_path = self.audio_files[index]
            label = self.labels[index]

            # If augmenting (TRAINING), load raw audio and apply temporal augmentation
            if self.augment:
                # Load raw audio
                y_audio = AudioProcessor.load_audio(
                    audio_path, sr=self.sr, duration=self.duration, mono=True
                )

                # Apply TEMPORAL augmentation (time-stretch, pitch-shift, noise)
                y_audio = AudioAugmentationTemporal.apply_random(y_audio, sr=self.sr)

                # Extract spectrogram from augmented audio
                mel_spec = AudioProcessor.extract_mel_spectrogram(
                    y_audio=y_audio,  # Pass augmented audio
                    sr=self.sr,
                    n_mels=self.n_mels,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    duration=self.duration,
                    segment_length=self.segment_length,
                )
                logging.debug(f"Applied temporal augmentation to {audio_path}")

                # Also apply spectrogram-level augmentation (masking)
                mel_spec = self._augment_spectrogram(mel_spec)

            else:
                # Validation: use cache (no augmentation needed)
                cache_file = self.cache_dir / f"spec_{index}.npy"
                mel_spec = np.load(cache_file)
                logging.debug(f"Loaded cached spectrogram for {audio_path}")

            # Convert to PyTorch tensors
            spec_tensor = torch.FloatTensor(mel_spec).unsqueeze(0)
            label_tensor = torch.tensor(label, dtype=torch.long)

            return spec_tensor, label_tensor.squeeze()

        except Exception as e:
            logger.error(f"Error in __getitem__: {e}")
            raise

    def _extract_mel_spectrogram(self, audio_path: str) -> np.ndarray:
        try:
            mel_spec = AudioProcessor.extract_mel_spectrogram(
                audio_path,
                sr=self.sr,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                duration=self.duration,
                segment_length=self.segment_length,
            )
            return mel_spec
        except Exception as e:
            logger.error(f"Error extracting mel-spectrogram from {audio_path}: {e}")
            raise

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
        genre_counts: Dict[str, int] = {}
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
            "n_fft": self.n_fft,
            "duration_seconds": self.duration,
            "segment_length": self.segment_length,
            "augmentation": self.augment,
            "caching": "preprocessed_cache",
        }
