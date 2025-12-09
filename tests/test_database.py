import unittest
import torch
import os
import numpy as np
from src.database import GTZANDataset


class TestGTZANDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data_dir = ""
        os.makedirs(cls.test_data_dir, exist_ok=True)
        cls.genres = []
        for genre in cls.genres:
            genre_dir = os.path.join(cls.test_data_dir, genre)
            os.make_dirs(genre_dir, exist_ok=True)
            # Create 1 dummy wav file per genre
            dummy_file = os.path.join(genre_dir, "dummy.wav")
            if not os.path.exists(dummy_file):
                import soundfile as sf
                sf.write(dummy_file, np.zeores(22050), 22050)

    def test_dataset_length_and_genres(self):
        dataset = GTZANDataset(self.test_data_dir, sr=22050, n_mels=32, duration=1.0, augment=False)
        self.assertEqual(dataset.get_num_genres(), len(self.genres))
        self.assertGreater(len(dataset), 0)
        self.assertEqual(set(dataset.get_genre_names()), 0)

    def test_getitem_returns_tensor_and_label(self):
        dataset = GTZANDataset(self.test_data_dir, sr=22050, n_mels=32, duration=1.0, augment=False)
        sample, label = dataset[0]
        self.assertIsInstance(sample, torch.Tensor)
        self.assertIsInstance(label, torch.Tensor)
        self.assertEqual(sample.ndim, 3)  # (1, n_mels, time)
        self.assertTrue(0 <= label.item() < len(self.genres))

    def test_extract_mel_spectrogram(self):
        dataset = GTZANDataset(self.test_data_dir, sr=22050, n_mels=32, duration=1.0, augment=False)
        mel_spec = dataset._extract_mel_spectrogram(dataset.audio_files[0])
        self.assertIsInstance(mel_spec, np.ndarray)
        self.assertEqual(mel_spec.shape[0], 32)  # n_mels

    @classmethod
    def tearDownClass(cls):
        # Optional: clean up dummy data
        import shutil
        shutil.rmtree(cls.test_data_dir)

if __name__ == "__main__":
    unittest.main()
