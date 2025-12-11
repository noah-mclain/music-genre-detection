__version__ = "1.0.0"
__author__ = "Omar Khaled, Nada Ayman"
__license__ = "MIT"
__description__ = "Music Genre Classification using CNN-LSTM-Attention"

from .database import GTZANDataset
from .GradCam import GradCAM
from .inference_utils import (
    AudioAugmentation,
    AudioProcessor,
    GenreClassifier,
    audio_generator,
)
from .mock_classifier import MockGenreClassifier
from .models import CNNLSTMAttentionModel, TemporalAttention

__all__ = [
    # Audio Processing
    "AudioProcessor",
    "AudioAugmentation",
    "GenreClassifier",
    "audio_generator",
    # Mock Classifier
    "MockGenreClassifier",
    # Model Components
    "TemporalAttention",
    "CNNLSTMAttentionModel",
    # Dataset
    "GTZANDataset",
    # Grad-CAM
    "GradCAM",
]

import logging

logger = logging.getLogger(__name__)
logger.debug(f"Loaded {__description__} v{__version__} by {__author__}")
