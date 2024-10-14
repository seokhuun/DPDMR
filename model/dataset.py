# dataset.py
import os
import torch
import librosa
from torch.utils.data import Dataset

class AudioMNISTDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_length=44):
        self.root_dir = root_dir
        self.transform = transform
        self.max_length = max_length
        self.data = []
        self.labels = []
        self.label_map = {}
        self._load_data()

    def _load_data(self):
        label_count = 0
        for folder in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.wav'):
                        file_path = os.path.join(folder_path, file_name)
                        label = int(file_name.split('_')[0])
                        if label not in self.label_map:
                            self.label_map[label] = label_count
                            label_count += 1
                        self.data.append(file_path)
                        self.labels.append(self.label_map[label])

    def extract_features(self, file_path, sr=8000, n_fft=512, hop_length=256):
        audio_data, sample_rate = librosa.load(file_path, sr=sr)
        stft = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        if log_spectrogram.shape[1] > self.max_length:
            log_spectrogram = log_spectrogram[:, :self.max_length]
        else:
            log_spectrogram = np.pad(log_spectrogram, ((0, 0), (0, self.max_length - log_spectrogram.shape[1])), mode='constant')
        return torch.tensor(log_spectrogram.flatten(), dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]
        label = self.labels[idx]
        features = self.extract_features(file_path)
        return features, label
