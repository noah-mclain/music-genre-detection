import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
from models import CNNLSTMAttentionModel
from inference_utils import AudioProcessor

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                break

    def generate(self, input_tensor: torch.Tensor, class_idx: int):
        self.model.eval()
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32)

        for i, w in enumerate(pooled_grads):
            cam += w * self.activations[0, i, :, :]

        cam = np.maximum(cam.cpu().numpy(), 0)
        cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    num_classes = 10 
    model = CNNLSTMAttentionModel(num_genres=num_classes)
    model.load_state_dict(torch.load("cnn_lstm_attention.pth", map_location=device))
    model.to(device)
    model.eval()

    # Load an example audio file
    audio_path = "E:\Deep Learning\music-genre-detection\Data\genres_original\\reggae\\reggae.00014.wav"
    mel_spec = AudioProcessor.extract_mel_spectrogram(audio_path)
    input_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Choose the target convolutional layer for Grad-CAM
    target_layer = "cnn.3"
    grad_cam = GradCAM(model, target_layer)

    # Forward pass to get predicted class
    with torch.no_grad():
        output = model(input_tensor)
        class_idx = torch.argmax(output, dim=1).item()

    # Generate Grad-CAM heatmap
    cam = grad_cam.generate(input_tensor, class_idx)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec, origin="lower", aspect="auto", cmap="gray")
    plt.imshow(cam, origin="lower", aspect="auto", cmap="jet", alpha=0.5)
    plt.title(f"Predicted Class: {class_idx}")
    plt.colorbar(label="Grad-CAM intensity")
    plt.show()

if __name__ == "__main__":
    main()
