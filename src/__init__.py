__version__ = "1.0.0"
__author__ = "Omar Khaled, Nada Ayman"
__license__ = "MIT"
__description__ = "Music Genre Classification using CNN-LSTM-Attention"

from .inference_utils import (
    AudioProcessor,
    AudioAugmentation,
    GenreClassifier,
    audio_generator,
)

from .models import TemporalAttention, CNNLSTMAttentionModel

from .database import GTZANDataset

__all__ = [
    # Audio Processing
    "AudioProcessor",
    "AudioAugmentation",
    "GenreClassifier",
    "audio_generator",
    # Model Components
    "TemporalAttention",
    "CNNLSTMAttentionModel",
    # Dataset
    "GTZANDataset",
]

import logging

logger = logging.getLogger(__name__)
logger.debug(f"Loaded {__description__} v{__version__} by {__author__}")
