import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import logging
import random

import matplotlib.pyplot as plt
import torch

from src.database import GTZANDataset
from src.inference_utils import AudioProcessor
from src.models import CNNLSTMAttentionModel

from src.GradCam import GradCAM
from src.IG import compute_ig, visualize_integrated_gradients

logger = logging.getLogger(__name__)


def visualize_cam(mel_spec, cam, class_idx: int):
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec, origin="lower", aspect="auto", cmap="gray")
    plt.imshow(cam, origin="lower", aspect="auto", cmap="jet", alpha=0.5)
    plt.title(f"Predicted Class: {class_idx}")
    plt.colorbar(label="Grad-CAM intensity")
    plt.show()


def run_explainable_ai():
    # Loading the dataset and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = GTZANDataset("src/Data/genres_original")
    num_classes = 10
    model = CNNLSTMAttentionModel(num_genres=num_classes)
    model.load_state_dict(torch.load("models/nn_lstm_attention.pth", map_location=device))
    model.to(device)
    model.eval()

    random_audio_idx = random.randint(0, len(dataset) - 1)
    audio_path = dataset.audio_files[random_audio_idx]
    genre_label = dataset.labels[random_audio_idx]
    genre_name = dataset.genres[genre_label]
    logger.info(f"Selected audio file: {audio_path}, Genre: {genre_name}")

    mel_spec = AudioProcessor.extract_mel_spectrogram(audio_path)
    input_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Choosing target CNN layer for GradCam
    target_layer = "cnn.3"
    grad_cam = GradCAM(model, target_layer)

    # Forward pass to get predicted class
    with torch.no_grad():
        output = model(input_tensor)
        class_idx = torch.argmax(output, dim=1).item()

    # Generate Grad-CAM heatmap
    cam = grad_cam.generate(input_tensor, class_idx)
    visualize_cam(mel_spec, cam, class_idx)

    # IntegratedGradients
    spec_tensor, label_tensor = dataset[random_audio_idx]
    mel_np = spec_tensor.squeeze().numpy()

    # forward pass
    input_batch = spec_tensor.unsqueeze(0).to(device)
    logits = model(input_batch)
    predicted_class = torch.argmax(logits, dim=1).item()
    attributions = compute_ig(model, input_batch, predicted_class)
    visualize_integrated_gradients(
        mel_np,
        attributions,
        title=f"Integrated Gradients Attribution (Predicted: {predicted_class})",
    )


if __name__ == "__main__":
    run_explainable_ai()
