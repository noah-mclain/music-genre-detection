import torch
from torch.utils.data import Dataset
import numpy as np
import librosa
import os
from pathlib import Path
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

class GTZANDataset(Dataset):
    def __init__(self, data_dir: str, sr: int = 22050, 
                 n_mels: int = 128, n_fft: int = 2048, hop_length: int = 512, 
                 duration: float = 30.0, segment_length: int = 130, augment: bool = True, cache_spectrograms: bool = False):
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
            [d for d in os.listdir(self.data_dir) 
             if os.path.isdir(os.path.join(self.data_dir, d))]
        )
        self.genre_to_idx = {g: idx for idx, g in enumerate(self.genres)}

        self.audio_files = []
        self.labels = []

        # Load audio file paths and labels
        for genre in self.genres:
            genre_dir = self.data_dir / genre
            audio_paths = list(genre_dir.glob("*.au")) + list(genre_dir.glob("*.wav")) + list(genre_dir.glob("*.mp3"))

            for audio_path in audio_paths:
                self.audio_files.append(str(audio_path))
                self.labels.append(self.genre_to_idx[genre])

            logger.info(f"Loaded GTZAN dataset: {len(self.audio_files)} files, {len(self.genres)} genres")

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

    def _load_audio(self, audio_path: str) -> np.ndarray:
        try:
            y_audio, _ = librosa.load(audio_path, duration=self.duration, sr=self.sr, mono=True)

            # Pad to fixed length
            expected_samples = int(self.sr * self.duration)
            if len(y_audio) < expected_samples:
                y_audio = np.pad(y_audio, (0, expected_samples - len(y_audio)), mode='constant')
            else:
                y_audio = y_audio[:expected_samples]

            return y_audio
        except Exception as e:
            logger.error(f"Error loading {audio_path}: {e}")
            raise e

    def _extract_mel_spectrogram(self, audio_path: str) -> np.ndarray:
        try:
            if self.spec_cache is not None and audio_path in self.spec_cache:
                return self.spec_cache[audio_path]

            y_audio = self._load_audio(audio_path)

            # mfccs = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=20)
            spectrogram = librosa.feature.melspectrogram(y=y_audio,
                                                        sr=self.sr,
                                                        n_mels=self.n_mels,
                                                        n_fft=self.n_fft,
                                                        hop_length=self.hop_length)


            spectrogram_log_scale = librosa.power_to_db(spectrogram, ref=np.max)

            # Use segment_length if set, otherwise calculate based on duration and hop_length
            if hasattr(self, 'segment_length'):
                target_width = self.segment_length
            else:
                target_width = int((self.sr * self.duration) / self.hop_length) + 1

            current_width = spectrogram_log_scale.shape[1]
            if current_width < target_width:
                padding = target_width - current_width
                spectrogram_log_scale = np.pad(spectrogram_log_scale, [(0, 0), (0, padding)], mode='constant')
            else:
                spectrogram_log_scale = spectrogram_log_scale[:, :target_width]

            # Normalize the spectrogram
            spectrogram_log_scale = (spectrogram_log_scale - spectrogram_log_scale.mean()) / (spectrogram_log_scale.std() + 1e-8)

            if self.spec_cache is not None:
                self.spec_cache[audio_path] = spectrogram_log_scale

            return spectrogram_log_scale
        except Exception as e:
            print(f"Error extracting spectrogram: {e}")
            raise e
        
    def __len__(self):
        return len(self.audio_files)

    def _augment_spectrogram(self, mel_spec: np.ndarray) -> np.ndarray:
        if not self.augment:
            return mel_spec
        
        # Time masking
        if np.random.rand() < 0.5:
            max_mask_width = 50
            mask_width = np.random.randint(0, max_mask_width)
            mask_start = np.random.randint(0, mel_spec[1] - mask_width)
            mel_spec = mel_spec.copy()
            mel_spec[:, mask_start:mask_start + mask_width] = mel_spec.mean()

        # Frequency masking
        if np.random.rand() < 0.5:
            max_mask_height = 10
            mask_height = np.random.randint(0, max_mask_height)
            mask_start = np.random.randint(0, mel_spec.shape[0] - mask_height)
            mel_spec = mel_spec.copy()
            mel_spec[mask_start:mask_start + mask_height, :] = mel_spec.mean()

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
    
class AudioAugmentation:
    @staticmethod
    def pitch_shift(y_audio:np.ndarray, sr: int, steps: int = 2) -> np.ndarray:
        return librosa.effects.pitch_shift(y_audio, sr=sr, n_steps=steps)
    
    @staticmethod
    def time_stretch(y_audio: np.ndarray, rate: float = 0.9) -> np.ndarray:
        return librosa.effects.time_stretch(y_audio, rate=rate)
    
    @staticmethod
    def add_noise(y_audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
        noise = np.random.normal(0, noise_factor, len(y_audio))
        return y_audio + noise
    
    @staticmethod
    def random_augment(y_audio: np.ndarray, sr: int = 22050) -> np.ndarray:
        augmentations = []

        if np.random.rand() < 0.33:
            augmentations.append(
                AudioAugmentation.pitch_shift(y_audio, sr, steps=np.random.randint(-2, 3))
            )
        if np.random.rand() < 0.33:
            augmentations.append(
                AudioAugmentation.time_stretch(y_audio, rate=np.random.uniform(0.9, 1.1))
            )
        if np.random.rand() < 0.33:
            augmentations.append(
                AudioAugmentation.add_noise(y_audio, noise_factor=0.005)
            )
        if augmentations:
            return augmentations[0]
        return y_audio