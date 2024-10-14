import os
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 데이터셋 클래스 정의
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
                        label = int(file_name.split('_')[0])  # 파일 이름에서 라벨 추출
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

# Prototypical Network 정의
class PrototypicalNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PrototypicalNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
    
    def forward(self, x):
        return self.encoder(x)

# Triplet Loss 정의
def triplet_loss(anchor, positive, negative, margin=1.0):
    pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
    neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
    loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
    return loss.mean()

# Triplet 샘플 생성 함수
def create_triplets(x, y):
    anchors, positives, negatives = [], [], []
    for i in range(len(y)):
        anchor = x[i]
        positive_idx = (y == y[i]).nonzero(as_tuple=True)[0].tolist()
        negative_idx = (y != y[i]).nonzero(as_tuple=True)[0].tolist()
        positive_idx.remove(i)
        # 양성 또는 음성 샘플이 없는 경우 건너뛰기
        if not positive_idx or not negative_idx:
            continue
        positive = x[positive_idx[torch.randint(len(positive_idx), (1,))]]
        negative = x[negative_idx[torch.randint(len(negative_idx), (1,))]]
        anchors.append(anchor)
        positives.append(positive)
        negatives.append(negative)
    if not anchors or not positives or not negatives:
        return None, None, None
    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)

# 임베딩 추출 함수
def extract_embeddings(data_loader, model, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            output = model(data)
            embeddings.append(output.cpu().numpy())
            labels.extend(target.cpu().numpy())
    return np.concatenate(embeddings), np.array(labels)

# 시각화 함수
def plot_embedding_space(embeddings_before, embeddings_after, labels, title_before, title_after):
    # t-SNE로 차원 축소
    tsne = TSNE(n_components=2, random_state=33)

    reduced_before = tsne.fit_transform(embeddings_before)
    reduced_after = tsne.fit_transform(embeddings_after)

    # 트리플렛 손실 전 임베딩 시각화
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title(title_before)
    plt.scatter(reduced_before[:, 0], reduced_before[:, 1], c=labels, cmap='Spectral', s=50, alpha=0.65, edgecolor='black', linewidth=0.5)
    plt.colorbar()
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    # 트리플렛 손실 후 임베딩 시각화
    plt.subplot(1, 2, 2)
    plt.title(title_after)
    plt.scatter(reduced_after[:, 0], reduced_after[:, 1], c=labels, cmap='Spectral', s=50, alpha=0.65, edgecolor='black', linewidth=0.5)
    plt.colorbar()
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    plt.tight_layout()
    plt.show()

# 간단한 모델 학습 함수
def train_simple_model(model, data_loader, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            embeddings = model(x)
            loss = F.cross_entropy(embeddings, y)
            loss.backward()
            optimizer.step()
    return model

# 트리플렛 손실 모델 학습 함수
def train_triplet_model(model, data_loader, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            anchors, positives, negatives = create_triplets(x, y)
            if anchors is None:
                print("Skipping batch due to insufficient data for triplets.")
                continue
            anchor_embeddings = model(anchors)
            positive_embeddings = model(positives)
            negative_embeddings = model(negatives)
            loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
            loss.backward()
            optimizer.step()
    return model

# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

# 데이터 로드 및 데이터 로더 생성
root_dir = '../../../Documents/CGB_AI_LAB/Data/Audio_MNIST'  # AudioMNIST 데이터셋의 경로
dataset = AudioMNISTDataset(root_dir)
class_label = 0  # 예를 들어, 클래스 '0' 데이터 사용

# 충분한 샘플이 있는 특정 클래스 데이터 로더
def get_class_data_loader(dataset, class_label, batch_size=32, min_samples=10):
    class_indices = [i for i, label in enumerate(dataset.labels) if label == class_label]
    if len(class_indices) < min_samples:
        raise ValueError(f"Class {class_label} has insufficient samples: {len(class_indices)} available, {min_samples} required.")
    class_subset = Subset(dataset, class_indices)
    return DataLoader(class_subset, batch_size=batch_size, shuffle=True)

try:
    class_data_loader = get_class_data_loader(dataset, class_label)
except ValueError as e:
    print(e)
    # 데이터 증강 또는 다른 클래스를 선택하는 등의 대체 조치를 여기서 수행하세요

# 모델 초기화
input_dim = 11308  # 스펙트로그램 특징 차원
hidden_dim = 128

# 간단한 모델 학습
simple_model = PrototypicalNetwork(input_dim, hidden_dim).to(device)
simple_optimizer = optim.Adam(simple_model.parameters(), lr=0.001)

simple_model = train_simple_model(simple_model, class_data_loader, simple_optimizer)

# 트리플렛 손실 모델 학습
triplet_model = PrototypicalNetwork(input_dim, hidden_dim).to(device)
triplet_optimizer = optim.Adam(triplet_model.parameters(), lr=0.001)

triplet_model = train_triplet_model(triplet_model, class_data_loader, triplet_optimizer)

# 임베딩 추출
embeddings_simple, labels_simple = extract_embeddings(class_data_loader, simple_model, device)
embeddings_triplet, labels_triplet = extract_embeddings(class_data_loader, triplet_model, device)

# 임베딩 시각화
plot_embedding_space(
    embeddings_simple,
    embeddings_triplet,
    labels_simple,
    f'Embedding Space Without Triplet Loss (Class {class_label})',
    f'Embedding Space With Triplet Loss (Class {class_label})'
)
