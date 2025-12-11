import matplotlib.pyplot as plt
import torch

from src.database import GTZANDataset
from src.inference_utils import AudioProcessor
from src.models import CNNLSTMAttentionModel

from .GradCam import GradCAM
from .IG import AUDIO_INDEX, N_MELS, SAMPLE_RATE, compute_ig, visualize_integrated_gradients


def visualize_cam(mel_spec, cam, class_idx: int):
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec, origin="lower", aspect="auto", cmap="gray")
    plt.imshow(cam, origin="lower", aspect="auto", cmap="jet", alpha=0.5)
    plt.title(f"Predicted Class: {class_idx}")
    plt.colorbar(label="Grad-CAM intensity")
    plt.show()


def run_explainable_ai():
    #Loading the dataset and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = GTZANDataset("E:\Deep Learning\music-genre-detection\Data\genres_original")
    num_classes = 10 
    model = CNNLSTMAttentionModel(num_genres=num_classes)
    model.load_state_dict(torch.load("cnn_lstm_attention.pth", map_location=device))
    model.to(device)
    model.eval()
    
    audio_path = "E:\Deep Learning\music-genre-detection\Data\genres_original\\reggae\\reggae.00014.wav"
    mel_spec = AudioProcessor.extract_mel_spectrogram(audio_path)
    input_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    #Choosing target CNN layer for GradCam
    target_layer = "cnn.3"
    grad_cam = GradCAM(model, target_layer)   

    # Forward pass to get predicted class
    with torch.no_grad():
        output = model(input_tensor)
        class_idx = torch.argmax(output, dim=1).item()

    # Generate Grad-CAM heatmap
    cam = grad_cam.generate(input_tensor, class_idx)
    visualize_cam(mel_spec, cam, class_idx)

    #IntegratedGradients
    spec_tensor, label_tensor = dataset[AUDIO_INDEX]
    mel_np = spec_tensor.squeeze().numpy()

    #forward pass
    input_batch = spec_tensor.unsqueeze(0).to(device)
    logits = model(input_batch)
    predicted_class = torch.argmax(logits, dim=1).item()
    attributions = compute_ig(model, input_batch, predicted_class)
    visualize_integrated_gradients(
        mel_np,
        attributions,
        title=f"Integrated Gradients Attribution (Predicted: {predicted_class})"
    )



if __name__ == "__main__":
    run_explainable_ai()