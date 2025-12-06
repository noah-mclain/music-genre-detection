import librosa
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def audio_generator(data_path, genres, batch_size=32, test_size=0.2, random_state=42):
    all_files = []
    all_labels = []
    
    # Collect all file paths and labels
    for genre in genres:
        genre_dir = os.path.join(data_path, genre)
        for file in os.listdir(genre_dir):
            if file.endswith(('.wav', '.mp3')):
                all_files.append(os.path.join(genre_dir, file))
                all_labels.append(genre)
    
    # Encode labels once
    le = LabelEncoder()
    y_encoded = le.fit_transform(all_labels)
    
    # Split file paths and labels into train and validation sets
    train_files, val_files, train_labels, val_labels = train_test_split(
        all_files, y_encoded, test_size=test_size, stratify=y_encoded, random_state=random_state
    )
    
    def generator(files, labels, batch_size):
        while True:
            indices = np.arange(len(files))
            np.random.shuffle(indices)
            for start in range(0, len(files), batch_size):
                batch_indices = indices[start:start + batch_size]
                X_batch, y_batch = [], []
                for i in batch_indices:
                    try:
                        y_audio, sr = librosa.load(files[i], duration=30)
                        mfccs = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=20)
                        mfccs_mean = np.mean(mfccs.T, axis=0)
                        
                        X_batch.append(mfccs_mean)
                        y_batch.append(labels[i])
                    except:
                        continue  # Skip any file that crashes
                if X_batch and y_batch:
                    yield np.array(X_batch), np.array(y_batch)
    
    return generator(train_files, train_labels, batch_size), generator(val_files, val_labels, batch_size), le
