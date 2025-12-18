import atexit
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"

if USE_MOCK:
    from src.mock_classifier import MockGenreClassifier as GenreClassifier
else:
    from src.inference_utils import AudioProcessor, GenreClassifier


class Config:
    # Flask settings
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
    DEBUG = os.getenv("FLASK_DEBUG", False)

    # Audio settings
    SUPPORTED_FORMATS = {"mp3", "wav", "m4a", "flac"}
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = "uploaded_files"

    # Model settings
    MODEL_PATH = os.getenv("MODEL_PATH", "models/cnn_lstm_attention.pth")
    SAMPLE_RATE = 22050
    N_MELS = 128
    DURATION = 30
    N_FFT = 2048
    HOP_LENGTH = 512
    SEGMENT_LENGTH = 130

    # Genre mapping
    GENRE_MAPPING = {
        0: "Blues",
        1: "Classical",
        2: "Country",
        3: "Disco",
        4: "Hip-Hop",
        5: "Jazz",
        6: "Metal",
        7: "Pop",
        8: "Reggae",
        9: "Rock",
    }

    # Server settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 5001))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


class MusicGenreClassifierApp:
    def __init__(self, config: Config):
        self.config = config
        self.app = Flask(__name__, template_folder="templates", static_folder="static")
        self._register_routes()

        self.app.config.from_object(config)
        self.prediction_cache = {}
        CORS(self.app)

        # Create upload folder if it doesn't exist
        Path(self.config.UPLOAD_FOLDER).mkdir(exist_ok=True)

        # Initialize the genre classifier
        logger.info("Loading genre classifier model...")
        try:
            self.classifier = GenreClassifier(
                model_path=self.config.MODEL_PATH,
                genre_mapping=self.config.GENRE_MAPPING,
                sample_rate=self.config.SAMPLE_RATE,
                n_mels=self.config.N_MELS,
                n_fft=self.config.N_FFT,
                hop_length=self.config.HOP_LENGTH,
                duration=self.config.DURATION,
                segment_length=self.config.SEGMENT_LENGTH,
            )
            logger.info(f"✅ Model loaded successfully on device: {self.classifier.device}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise e

        # Register routes
        self._register_routes()

    def _register_routes(self):
        self.app.add_url_rule("/", "index", self.index, methods=["GET"])
        self.app.add_url_rule("/upload", "upload", self.upload, methods=["POST"])
        self.app.add_url_rule("/health", "health", self.health, methods=["GET"])
        self.app.add_url_rule("/api/genres", "get_genres", self.get_genres, methods=["GET"])
        self.app.add_url_rule("/predict", "predict", self.predict, methods=["POST"])

    def index(self):
        try:
            return render_template("index.html")
        except Exception as e:
            logger.error(f"Error loading index page: {e}")
            return jsonify({"error": "Failed to load page"}), 500

    def health(self):
        return (
            jsonify(
                {
                    "status": "healthy",
                    "model": "loaded",
                    "device": self.classifier.device,
                    "genres": len(self.config.GENRE_MAPPING),
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            200,
        )

    def get_genres(self):
        return (
            jsonify(
                {
                    "genres": list(self.config.GENRE_MAPPING.values()),
                    "count": len(self.config.GENRE_MAPPING),
                }
            ),
            200,
        )

    def predict(self):
        try:
            # Validate request
            if "file" not in request.files:
                return jsonify({"error": "No file provided"}), 400

            file = request.files["file"]

            if file.filename == "":
                return jsonify({"error": "No file selected"}), 400

            if not self._is_valid_audio_file(file.filename):
                return (
                    jsonify({"error": "Invalid file format. Supported: MP3, WAV, M4A, FLAC"}),
                    400,
                )

            if file.content_length > self.config.MAX_FILE_SIZE:
                return jsonify({"error": "File size exceeds 50MB limit"}), 400

            # Save temporary file
            secure_name = secure_filename(file.filename)
            temp_path = os.path.join(self.config.UPLOAD_FOLDER, f"temp_{secure_name}")
            file.save(temp_path)
            
            if secure_name in self.prediction_cache:
                logger.info(f"Cache hit for {file.filename}")
                cached_result = self.prediction_cache[secure_name]
                return jsonify({
                    "success" : True,
                    "filename" : file.filename,
                    **cached_result
                }), 200


            try:
                # Extract features and predict
                logger.info(f"Processing file: {secure_name}")

                mel_spec = AudioProcessor.extract_mel_spectrogram(
                    temp_path,
                    sr=self.config.SAMPLE_RATE,
                    n_mels=self.config.N_MELS,
                    n_fft=self.config.N_FFT,
                    hop_length=self.config.HOP_LENGTH,
                    duration=self.config.DURATION,
                    segment_length=self.config.SEGMENT_LENGTH,
                )

                # Convert to tensor
                x = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                x = x.to(self.classifier.device)

                # Get predictions
                with torch.no_grad():
                    logits = self.classifier.model(x)
                    probabilities = torch.softmax(logits, dim=1)
                    pred_idx = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0, pred_idx].item() * 100

                # Get all predictions
                all_predictions = {}
                for idx, genre in self.config.GENRE_MAPPING.items():
                    all_predictions[genre] = float(probabilities[0, idx].item())

                # Sort by confidence
                sorted_predictions = dict(
                    sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
                )
                result = {
                    "genre": self.config.GENRE_MAPPING[pred_idx],
                    "confidence" : confidence,
                    "predictions" : sorted_predictions
                }
                self.prediction_cache[file.filename] = result


                logger.info(
                    f"Prediction: {self.config.GENRE_MAPPING[pred_idx]} ({confidence:.2f}%)"
                )

                return (
                    jsonify(
                        {
                            "success": True,
                            "filename": secure_name,
                            **result
                        }
                    ),
                    200,
                )

            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    def upload(self):
        try:
            if "files" not in request.files:
                return jsonify({"error": "No files provided"}), 400

            files = request.files.getlist("files")
            results = []

            for file in files:
                if file and file.filename:
                    filename = secure_filename(file.filename)

                    # Validate file format
                    if not self._is_valid_audio_file(filename):
                        results.append(
                            {
                                "filename": filename,
                                "success": False,
                                "error": "Invalid file format",
                            }
                        )
                        continue

                    filepath = os.path.join(self.config.UPLOAD_FOLDER, filename)
                    file.save(filepath)

                    try:
                        # ============================================
                        # MOCK VERSION
                        # ============================================
                        # logger.info(f"Processing file: {filename}")
                        # genre, confidence, predictions = self.classifier.predict(
                        #     filepath
                        # )
                        # # Sort by confidence
                        # sorted_predictions = dict(
                        #     sorted(
                        #         predictions.items(), key=lambda x: x[1], reverse=True
                        #     )
                        # )

                        # results.append(
                        #     {
                        #         "filename": filename,
                        #         "success": True,
                        #         "genre": genre,
                        #         "confidence": float(confidence),
                        #         "predictions": sorted_predictions,
                        #     }
                        # )
                        # logger.info(f"{filename}: {genre}")

                        # ============================================
                        # PRODUCTION VERSION
                        # ============================================
                        logger.info(f"Processing file: {filename}")

                        # Extract mel-spectrogram
                        mel_spec = AudioProcessor.extract_mel_spectrogram(
                            filepath,
                            sr=self.config.SAMPLE_RATE,
                            n_mels=self.config.N_MELS,
                            n_fft=self.config.N_FFT,
                            hop_length=self.config.HOP_LENGTH,
                            duration=self.config.DURATION,
                            segment_length=self.config.SEGMENT_LENGTH,
                        )

                        # Convert to tensor
                        x = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                        x = x.to(self.classifier.device)

                        # Get predictions
                        with torch.no_grad():
                            logits = self.classifier.model(x)
                            probabilities = torch.softmax(logits, dim=1)
                            pred_idx = torch.argmax(probabilities, dim=1).item()
                            confidence = probabilities[0, pred_idx].item() * 100

                        # Get all predictions
                        all_predictions = {}
                        for idx, genre in self.config.GENRE_MAPPING.items():
                            all_predictions[genre] = float(probabilities[0, idx].item())

                        # Sort by confidence
                        sorted_predictions = dict(
                            sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
                        )

                        results.append(
                            {
                                "filename": filename,
                                "success": True,
                                "genre": self.config.GENRE_MAPPING[pred_idx],
                                "confidence": confidence,
                                "predictions": sorted_predictions,
                            }
                        )
                        logger.info(f"{filename}: {self.config.GENRE_MAPPING[pred_idx]}")

                    except Exception as e:
                        logger.error(f"Error processing {filename}: {e}")
                        results.append(
                            {
                                "filename": filename,
                                "success": False,
                                "error": str(e),
                            }
                        )

            logger.info(f"Processed {len(results)} files")
            return jsonify({"results": results}), 200

        except Exception as e:
            logger.error(f"Upload error: {e}")
            return jsonify({"error": str(e)}), 500

    def _is_valid_audio_file(self, filename: str) -> bool:
        if not filename:
            return False
        ext = filename.rsplit(".", 1)[1].lower() if "." in filename else ""
        return ext in self.config.SUPPORTED_FORMATS

    def run(self):
        logger.info(f"🎵 Starting Genre Classifier on {self.config.HOST}:{self.config.PORT}")
        self.app.run(
            host=self.config.HOST,
            port=self.config.PORT,
            debug=self.config.DEBUG,
            use_reloader=False,
        )


def create_app(config=None):
    if config is None:
        config = Config()
    return MusicGenreClassifierApp(config)

def cleanup_uploaded_files(folder):
    if os.path.exists(folder):
        logger.info(f"Cleaning up uploaded files in {folder}....")
        shutil.rmtree(folder)


if __name__ == "__main__":
    app = create_app()
    atexit.register(cleanup_uploaded_files, app.config.UPLOAD_FOLDER)
    app.run()
    
