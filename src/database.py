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
    def __init__(self, data_dir: str, sample_rate: int = 22050, 
                 n_mels: int = 128, n_fft: int = 2048, hop_length: int = 512, 
                 duration: float = 30.0, segment_length: int = 130, augement: bool = True, cache_spectograms: bool = False):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        self.segment_length = segment_length
        self.augement = augement
        self.cache_spectograms = cache_spectograms

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
            audio_paths = list(genre_dir.glob(".*au")) + list(genre_dir.glob("*.wav"))

            for audio_path in audio_paths:
                self.audio_files.append(str(audio_path))
                self.labels.append(self.genre_to_idx[genre])

            logger.info(f"Loaded GTZAN dataset: {len(self.audio_files)} files, {len(self.genres)} genres")

        # Cache spectrograms if required
        if cache_spectograms:
            self.spec_cache = {}
        else:
            self.spec_cache = None

    def __getitem__(self, index):
        try:
            if self.cache_spectograms and index in self.spec_cache:
                spectogram_log_scale = self.spec_cache[index]
            
            else:
                y_audio, sr = librosa.load(self.audio_files[index], duration=self.duration, sr=self.sample_rate, mono=True)  
                # mfccs = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=20)
                spectogram = librosa.feature.melspectrogram(y=y_audio,
                                                            sr=sr,
                                                            n_mels=self.n_mels,
                                                            n_fft=self.n_fft,
                                                            hop_length=self.hop_length)


                spectogram_log_scale = librosa.power_to_db(spectogram, ref=np.max)

                # time_size = spectogram_log_scale.shape[1]
                # max_width = int((self.sample_rate * self.duration) / self.hop_length) + 1

                # if time_size < max_width:
                #     spectogram_log_scale = np.pad(spectogram_log_scale, [(0, 0), (0, max_width - time_size)], mode="constant")

                # else:
                #     spectogram_log_scale = spectogram_log_scale[:, :max_width]

                if self.cache_spectograms:
                    self.spec_cache[index] = spectogram_log_scale

            mfcc_tensor = torch.Tensor(spectogram_log_scale, dtype=torch.float32).unsqueeze(0)
            label_tensor = torch.tensor(self.labels[index], dtype=torch.long)

            return mfcc_tensor, label_tensor

        except:
            raise IndexError