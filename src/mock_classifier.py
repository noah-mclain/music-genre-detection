import random
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MockGenreClassifier:
    def __init__(
        self,
        model_path=None,
        genre_mapping=None,
        sample_rate=22050,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        duration=30.0,
        segment_length=130,
        device="cpu",
    ):
        self.genre_mapping = genre_mapping or {
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
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        self.segment_length = segment_length
        self.device = device

        logger.info("✅ Mock Classifier Loaded (Testing Mode)")
        logger.info(f"   Genres: {list(self.genre_mapping.values())}")
        logger.info(f"   Note: Returning RANDOM predictions for UI testing")

    def predict(self, file_path):
        try:
            # Pick a random genre as the top prediction
            pred_genre_idx = random.randint(0, len(self.genre_mapping) - 1)
            pred_genre = self.genre_mapping[pred_genre_idx]

            # Generate realistic confidence (60-99%)
            confidence = random.uniform(60, 99)

            # Generate all genre predictions that sum to ~100%
            predictions = {}
            scores = np.random.dirichlet(np.ones(len(self.genre_mapping))) * 100

            # Assign scores to all genres
            for idx, genre in self.genre_mapping.items():
                predictions[genre] = float(scores[idx])

            # Ensure predicted genre has the highest confidence
            predictions[pred_genre] = confidence

            logger.debug(
                f"Mock prediction for {file_path}: {pred_genre} ({confidence:.1f}%)"
            )

            return pred_genre, confidence, predictions

        except Exception as e:
            logger.error(f"Error in mock prediction: {e}")
            # Return fallback prediction
            avg_score = 100.0 / len(self.genre_mapping)
            return (
                list(self.genre_mapping.values())[0],
                50.0,
                {g: avg_score for g in self.genre_mapping.values()},
            )

    def predict_batch(self, file_paths):
        predictions = []

        for file_path in file_paths:
            try:
                genre, confidence, preds = self.predict(file_path)

                # Extract filename from path
                filename = file_path.split("/")[-1] if "/" in file_path else file_path

                predictions.append(
                    {
                        "filename": filename,
                        "success": True,
                        "genre": genre,
                        "confidence": float(confidence),
                        "predictions": preds,
                    }
                )

            except Exception as e:
                filename = file_path.split("/")[-1] if "/" in file_path else file_path
                logger.error(f"Error processing {file_path}: {e}")

                predictions.append(
                    {"filename": filename, "success": False, "error": str(e)}
                )

        return predictions


if __name__ == "__main__":
    print("Testing MockGenreClassifier...")

    # Create instance with all parameters
    classifier = MockGenreClassifier(
        model_path="dummy.pth",
        genre_mapping={
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
        },
        sample_rate=22050,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        duration=30.0,
        segment_length=130,
        device="cpu",
    )

    # Test single prediction
    print("\nTest 1: Single prediction")
    genre, confidence, predictions = classifier.predict("test.mp3")
    print(f"  Genre: {genre}")
    print(f"  Confidence: {confidence:.1f}%")
    print(f"  Top 3 predictions:")
    top_3 = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]
    for g, c in top_3:
        print(f"    - {g}: {c:.1f}%")

    # Test batch prediction
    print("\nTest 2: Batch prediction")
    files = ["song1.mp3", "song2.wav", "song3.flac"]
    results = classifier.predict_batch(files)
    for result in results:
        print(
            f"  {result['filename']}: {result['genre']} ({result['confidence']:.1f}%)"
        )

    print("\n✅ Mock classifier working correctly!")
    print("Now run: export USE_MOCK=true && python app.py")
