import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
import librosa
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import torch.nn.functional as F

# 데이터셋 클래스 정의
class PRPDDataset(Dataset):
    def __init__(self, root_dir, max_length=44):
        self.root_dir = root_dir
        self.max_length = max_length
        self.data = []
        self.labels = []
        self.label_map = self._load_data()
    
    def _load_data(self):
        label_map = {label: idx for idx, label in enumerate(os.listdir(self.root_dir))}
        for label in os.listdir(self.root_dir):
            label_dir = os.path.join(self.root_dir, label)
            if os.path.isdir(label_dir):
                for file in os.listdir(label_dir):
                    file_path = os.path.join(label_dir, file)
                    features = self.extract_features(file_path)
                    self.data.append(features)
                    self.labels.append(label_map[label])
        return label_map

    def extract_features(self, file_path, sr=96000, n_fft=1024, hop_length=512):
        audio_data, sample_rate = librosa.load(file_path, sr=sr)
        stft = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        if log_spectrogram.shape[1] > self.max_length:
            log_spectrogram = log_spectrogram[:, :self.max_length]
        else:
            log_spectrogram = np.pad(log_spectrogram, ((0, 0), (0, self.max_length - log_spectrogram.shape[1])), mode='constant')
        # 데이터 정규화
        log_spectrogram = (log_spectrogram - log_spectrogram.mean()) / (log_spectrogram.std() + 1e-10)
        return torch.tensor(log_spectrogram.flatten(), dtype=torch.float32)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 평가 함수 정의
def evaluate(model, test_loader, num_classes, prototype_means=None):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            if prototype_means is not None:
                features = model.encoder(x)
                dists = torch.cdist(features, prototype_means)  # calculate distance to prototypes
                preds = dists.argmin(dim=1)
            else:
                outputs = model(x)
                preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    accuracy = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))

    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", cm)

    # 실제로 평가된 클래스에 따라 타겟 이름 조정
    present_classes = np.unique(all_targets + all_preds)
    class_labels = [str(i) for i in present_classes]
    class_report = classification_report(all_targets, all_preds, target_names=class_labels, zero_division=0)
    print("Classification Report:\n", class_report)
    
    return accuracy, recall, f1, cm

# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

# 데이터셋 및 데이터 로더 정의
root_dir = '../../../../../Documents/CGB_AI_LAB/Data/PRPD_augmented'  # PRPD 데이터셋의 루트 디렉토리 경로
dataset = PRPDDataset(root_dir)
class_indices = {label: [] for label in range(len(dataset.label_map))}

# 각 클래스별로 인덱스를 분리합니다.
for idx, (_, label) in enumerate(dataset):
    class_indices[label].append(idx)

# 각 클래스를 독립적인 테스크로 만듭니다.
tasks = [Subset(dataset, indices) for indices in class_indices.values()]

# PrototypicalNetwork 정의
class PrototypicalNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PrototypicalNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),  # 드롭아웃 비율
        )
        self.classifier = nn.Linear(hidden_dim, len(dataset.label_map))  # PRPD 클래스 수에 맞게 조정

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)

# Variational Autoencoder 정의
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # 평균
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # 로그-분산
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, x.size(1)))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def generate(self, z):
        return self.decode(z)

# VAE 학습 함수
def train_vae(vae, data_loader, epochs=10, lr=0.001):
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    vae.train()
    for epoch in range(epochs):
        train_loss = 0
        for x, _ in data_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(x)
            # 손실 함수
            recon_loss = F.mse_loss(recon_batch, x.view(-1, x.size(1)), reduction='sum')
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kld_loss
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss / len(data_loader.dataset)}')

# DGR 클래스 정의
class DGR:
    def __init__(self, model, vae, memory_size=2000, lr=0.0001, weight_decay=0.0001):
        self.model = model
        self.vae = vae
        self.memory_size = memory_size
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)

    def generate_replay_data(self, num_samples, num_classes):
        self.vae.eval()
        with torch.no_grad():
            z = torch.randn(num_samples * num_classes, self.vae.fc21.out_features).to(device)
            generated_data = self.vae.generate(z)
            return generated_data

    def train_task(self, task_loader, replay_data=None):
        self.model.train()
        best_loss = float('inf')
        patience, trials = 3, 0
        for epoch in range(10):
            running_loss = 0.0
            for x, y in task_loader:
                x, y = x.to(device), y.to(device)
                if replay_data is not None:
                    x_replay, y_replay = replay_data
                    x = torch.cat((x, x_replay), dim=0)
                    y = torch.cat((y, y_replay), dim=0)

                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = nn.CrossEntropyLoss()(outputs, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(task_loader)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                trials = 0
            else:
                trials += 1
                if trials >= patience:
                    print("Early stopping")
                    break

            self.scheduler.step()

    def train_and_evaluate(self, tasks):
        num_classes = len(tasks)
        for class_id, task in enumerate(tasks):
            print(f"\nTraining Class {class_id} with DGR:")
            train_loader = DataLoader(task, batch_size=32, shuffle=True)

            # 현재 태스크에 대해 VAE 학습
            print("Training VAE...")
            train_vae(self.vae, train_loader)

            # VAE에서 리플레이 데이터 생성
            replay_data = None
            if class_id > 0:
                replay_x = self.generate_replay_data(self.memory_size // class_id, class_id)
                replay_y = torch.tensor([i for i in range(class_id) for _ in range(self.memory_size // class_id)]).to(device)
                replay_data = (replay_x, replay_y)

            # 리플레이 데이터와 함께 태스크 학습
            self.train_task(train_loader, replay_data)

            # 현재까지의 모든 태스크에 대해 평가
            self.evaluate_intermediate(tasks[:class_id + 1])

    def evaluate_intermediate(self, tasks_up_to_now):
        print("\nEvaluating on tasks up to now:")
        combined_loader = DataLoader(ConcatDataset(tasks_up_to_now), batch_size=32, shuffle=False)
        evaluate(self.model, combined_loader, num_classes=len(dataset.label_map))

# 모델 초기화 및 DGR 학습
input_dim = 22572  # PRPD 데이터셋에 맞게 조정
hidden_dim = 128
latent_dim = 64  # 잠재 공간의 차원

model = PrototypicalNetwork(input_dim, hidden_dim).to(device)
vae = VAE(input_dim, latent_dim, hidden_dim).to(device)

dgr = DGR(model, vae, memory_size=200, lr=0.0001, weight_decay=0.0001)  # 학습률을 낮춤

# 클래스별 학습 및 중간 평가
dgr.train_and_evaluate(tasks)

# 전체 테스트 세트에 대한 최종 평가
print("\nEvaluating on the entire test set:")
full_test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
evaluate(model, full_test_loader, num_classes=len(dataset.label_map))
