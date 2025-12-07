from flask import Flask, request, render_template
import torch
import librosa
import numpy as np

class WebUI:
    def __init__ (self, model_path, genres, host = "127.0.0.1", port = 8080):
        self.app = Flask(__name__)
        self.model = torch.load(model_path)
        self.model.eval()
        self.genres = genres
        self.host = host
        self.port = port
        self.app.add_url_rule("/", "index", self.index, methods=["GET", "POST"])

    def extract_mfcc(self, file_path, n_mfcc=20, max_len=215, duration=30):
        y, sr = librosa.load(file_path, sr=22050, duration=duration)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        if mfcc.shape[1] < max_len:
            mfcc.shape = np.pad(mfcc, ((0,0) , (0,max_len - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]
        mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    def index(self):
        if request.method == "POST":
            if "audio_file" not in request.files:
                return "No file uploaded"
            file = request.files["audio_file"]
            file_path = f"temp_{file.filename}"

            mfcc_tensor = self.extract_mfcc(file_path)
            with torch.no_grad():
                output = self.model(mfcc_tensor)
                pred_idx = torch.argmax(output, dim=1).item()
                pred_genre = self.genres[pred_idx]

            return render_template("index.html", genre=pred_genre)
        return render_template("index.html", genre=None)
    
    def run(self):
        self.app.run(host=self.host, port=self.port, debug=True)
    
