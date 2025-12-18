import logging
import os

import librosa
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.models import CNNLSTMAttentionModel

logger = logging.getLogger(__name__)


class AudioProcessor:
    @staticmethod
    def load_audio(
        audio_path: str, sr: int = 22050, duration: float = 30.0, mono: bool = True
    ) -> np.ndarray:
        try:
            y_audio, _ = librosa.load(audio_path, duration=duration, sr=sr, mono=True)

            # Pad to fixed length
            expected_samples = int(sr * duration)
            if len(y_audio) < expected_samples:
                y_audio = np.pad(y_audio, (0, expected_samples - len(y_audio)), mode="constant")
            else:
                y_audio = y_audio[:expected_samples]

            return y_audio
        except Exception as e:
            logger.error(f"Error loading {audio_path}: {e}")
            raise e

    @staticmethod
    def extract_mel_spectrogram(
        audio_path: str = None,
        y_audio: np.ndarray = None,
        sr: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        duration: float = 30.0,
        segment_length: int = 130,
        normalize: bool = True,
    ) -> np.ndarray:
        try:
            if y_audio is None:
                y_audio = AudioProcessor.load_audio(audio_path, sr=sr, duration=duration, mono=True)

            # mfccs = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=20)
            spectrogram = librosa.feature.melspectrogram(
                y=y_audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
            )

            # Convert to dB scale
            spectrogram_log_scale = librosa.power_to_db(spectrogram, ref=np.max)

            if normalize:
                spectrogram_log_scale = (spectrogram_log_scale - spectrogram_log_scale.mean()) / (
                    spectrogram_log_scale.std() + 1e-8
                )

            # Pad to fixed length
            current_width = spectrogram_log_scale.shape[1]
            target_width = segment_length

            if current_width < target_width:
                padding = target_width - current_width
                spectrogram_log_scale = np.pad(
                    spectrogram_log_scale,
                    [(0, 0), (0, padding)],
                    mode="constant",
                )
            else:
                spectrogram_log_scale = spectrogram_log_scale[:, :target_width]
            return spectrogram_log_scale  # (n_mels, segment_length)
        except Exception as e:
            print(f"Error extracting spectrogram: {e}")
            raise e

    @staticmethod
    def to_tensor(
        mel_spec: np.ndarray,
        add_channel: bool = True,
        add_batch: bool = True,
        device: str = "cpu",
    ) -> torch.Tensor:
        tensor = torch.FloatTensor(mel_spec)

        if add_channel:
            tensor = tensor.unsqueeze(0)  # (1, n_mels, seg_len)
        if add_batch:
            tensor = tensor.unsqueeze(0)  # (1, 1, n_mels, seg_len)
        return tensor.to(device)


class AudioAugmentation:
    @staticmethod
    def pitch_shift(y_audio: np.ndarray, sr: int, steps: int = 2) -> np.ndarray:
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
            augmentations.append(AudioAugmentation.add_noise(y_audio, noise_factor=0.005))
        if augmentations:
            return augmentations[0]
        return y_audio


class AudioAugmentationTemporal:
    @staticmethod
    def time_stretch(y_audio: np.ndarray, sr: int = 22050) -> np.ndarray:
        rate = np.random.uniform(0.8, 1.2)
        try:
            return librosa.effects.time_stretch(y_audio, rate=rate)
        except:
            return y_audio

    @staticmethod
    def pitch_shift(y_audio: np.ndarray, sr: int = 22050) -> np.ndarray:
        n_steps = np.random.randint(-4, 5)
        try:
            return librosa.effects.pitch_shift(y_audio, sr=sr, n_steps=n_steps)
        except:
            return y_audio

    @staticmethod
    def add_noise(y_audio: np.ndarray, snr_db: float = 20) -> np.ndarray:
        signal_power = np.mean(y_audio**2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(y_audio))
        return y_audio + noise

    @staticmethod
    def apply_random(y_audio: np.ndarray, sr: int = 22050) -> np.ndarray:
        augs = [
            lambda: AudioAugmentationTemporal.time_stretch(y_audio, sr),
            lambda: AudioAugmentationTemporal.pitch_shift(y_audio, sr),
            lambda: AudioAugmentationTemporal.add_noise(y_audio),
        ]
        n_apply = np.random.randint(2, 5)
        selected = np.random.choice(len(augs), n_apply, replace=True)
        result = y_audio.copy()
        for idx in selected:
            try:
                result = augs[idx]()
            except:
                pass
        return result


class GenreClassifier:
    def __init__(
        self,
        model_path: str,
        genre_mapping: dict,
        sample_rate: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        duration: float = 30.0,
        segment_length: int = 130,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CNNLSTMAttentionModel(num_genres=len(genre_mapping))
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.genre_mapping = genre_mapping
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        self.segment_length = segment_length

        self.max_width = int((sample_rate * duration) / hop_length) + 1

    def predict(self, file_path: str):
        try:
            mel_spec = AudioProcessor.extract_mel_spectrogram(
                file_path,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                duration=self.duration,
                segment_length=self.segment_length,
            )

            x = AudioProcessor.to_tensor(mel_spec, device=self.device)

            with torch.no_grad():
                logits = self.model(x)

            pred_idx = torch.argmax(logits, dim=1).item()
            return self.genre_mapping[pred_idx]
        except Exception as e:
            logger.error(f"Prediction error for {file_path}: {e}")
            raise

    # Predict multiple files in a batch
    def predict_batch(self, file_paths: list) -> list:
        predictions = []
        for file_path in file_paths:
            try:
                pred = self.predict(file_path)
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Failed to predict {file_path}: {e}")
                predictions.append(None)

        return predictions


def audio_generator(
    data_path: str,
    genres: list,
    batch_size: int = 32,
    test_size: float = 0.2,
    random_state: int = 42,
):
    all_files = []
    all_labels = []

    # Collect all file paths and labels
    for genre in genres:
        genre_dir = os.path.join(data_path, genre)

        if not os.path.exists(genre_dir):
            logger.warning(f"Genre directory not found: {genre_dir}")
            continue

        for file in os.listdir(genre_dir):
            if file.endswith((".wav", ".mp3")):
                all_files.append(os.path.join(genre_dir, file))
                all_labels.append(genre)

    if not all_files:
        raise ValueError(f"No audio files found in {data_path}")

    # Encode labels once
    le = LabelEncoder()
    y_encoded = le.fit_transform(all_labels)

    # Split file paths and labels into train and validation sets
    train_files, val_files, train_labels, val_labels = train_test_split(
        all_files,
        y_encoded,
        test_size=test_size,
        stratify=y_encoded,
        random_state=random_state,
    )

    def generator(files: list, labels: np.ndarray, batch_size: int):
        while True:
            indices = np.arange(len(files))
            np.random.shuffle(indices)

            for start in range(0, len(files), batch_size):
                batch_indices = indices[start : start + batch_size]
                X_batch, y_batch = [], []

                for i in batch_indices:
                    try:
                        mel_spec = AudioProcessor.extract_mel_spectrogram(
                            files[i],
                            sr=22050,
                            n_mels=128,
                            n_fft=2048,
                            hop_length=512,
                            duration=30.0,
                            segment_length=130,
                        )

                        X_batch.append(mel_spec)
                        y_batch.append(labels[i])
                    except Exception as e:
                        logger.warning(f"Failed to load {files[i]}: {e}")
                        continue
                if X_batch and y_batch:
                    yield np.array(X_batch), np.array(y_batch)

    return (
        generator(train_files, train_labels, batch_size),
        generator(val_files, val_labels, batch_size),
        le,
    )
