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
                 n_mels: int = 128, n_fft: int = 2048, hop_lenght: int = 512, 
                 duration: float = 30.0, segment_length: int = 130, augement: bool = True, cache_spectograms: bool = False):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_lenght
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
