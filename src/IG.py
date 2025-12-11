import logging
import os

import matplotlib.pyplot as plt
import torch
from captum.attr import IntegratedGradients

from .database import GTZANDataset
from .models import CNNLSTMAttentionModel

logger = logging.getLogger(__name__)

#Configuration
MODEL_PATH = "cnn_lstm_attention.pth"
AUDIO_INDEX = 0                
SAMPLE_RATE = 22050
N_MELS = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load Model
def load_model(model_path):
    model = CNNLSTMAttentionModel(
        num_genres=10
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.load_state_dict(torch.load("cnn_lstm_attention.pth", map_location=device))
    model.to(device)
    model.eval()

    return model

#Compute gradients
def compute_ig(model, input_tensor, target_class):
    ig = IntegratedGradients(model)

    baseline = torch.zeros_like(input_tensor)

    attributions = ig.attribute(
        inputs=input_tensor,
        baselines=baseline,
        target=target_class,
        n_steps=50 
    )

    attributions = attributions.squeeze().detach().cpu().numpy()

    return attributions

#visualize spectrogram + IG attributions
def visualize_integrated_gradients(mel_spec, attributions, title=""):
    plt.figure(figsize=(15, 8))

    # Original mel-spectrogram
    plt.subplot(2, 1, 1)
    plt.title("Original Mel-Spectrogram")
    plt.imshow(mel_spec, aspect="auto", origin="lower")
    plt.colorbar()

    # IG heatmap
    plt.subplot(2, 1, 2)
    plt.title(title)
    plt.imshow(attributions, aspect="auto", origin="lower")
    plt.colorbar()

    plt.tight_layout()
    plt.show()


def run_integrated_gradients():
    logger.info("Loading dataset...")
    dataset = GTZANDataset("E:\Deep Learning\music-genre-detection\Data\genres_original")
    spec_tensor, label_tensor = dataset[AUDIO_INDEX]

    mel_np = spec_tensor.squeeze().numpy()

    logger.info("Loading model...")
    model = load_model(MODEL_PATH)

    logger.info("Running forward pass...")
    input_batch = spec_tensor.unsqueeze(0).to(device)

    logits = model(input_batch)
    predicted_class = torch.argmax(logits, dim=1).item()

    logger.info(f"Predicted label: {predicted_class}")
    logger.info(f"Actual label: {label_tensor.item()}")

    logger.info("Computing gradients")
    attributions = compute_ig(model, input_batch, predicted_class)

    logger.info("Visualizing")
    visualize_integrated_gradients(
        mel_np,
        attributions,
        title=f"Integrated Gradients Attribution (Predicted: {predicted_class})"
    )


    logger.info("Done")


if __name__ =="__main__":
    run_integrated_gradients()

 
