import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
import librosa
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

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
        return torch.tensor(log_spectrogram.flatten(), dtype=torch.float32)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 평가 함수 정의
def evaluate(model, test_loader, class_means, num_classes):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            features = model.encoder(x)
            preds = predict_using_icarl(class_means, features)
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

def predict_using_icarl(class_means, features):
    # Compute distances between features and class means
    features = features.cpu()
    class_means = torch.stack(list(class_means.values()))

    # Normalize using torch's built-in normalization
    class_means = nn.functional.normalize(class_means, p=2, dim=1)
    features = nn.functional.normalize(features, p=2, dim=1)
    
    # Calculate distance between features and class means
    distances = torch.cdist(features, class_means)
    
    # Predict the class with the minimum distance
    preds = distances.argmin(dim=1)
    return preds

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
            nn.Dropout(0.4),  # 드롭아웃 비율
            nn.Linear(hidden_dim, hidden_dim),  # 추가 레이어
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, len(dataset.label_map))  # PRPD 클래스 수에 맞게 조정

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)

# iCaRL 클래스 정의
class iCaRL:
    def __init__(self, model, memory_size=400, lr=0.005, weight_decay=0.0001):
        self.model = model
        self.memory_size = memory_size
        self.exemplar_sets = []
        self.class_means = {}
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def compute_class_means(self):
        with torch.no_grad():
            for class_id, exemplars in enumerate(self.exemplar_sets):
                features = self.model.encoder(exemplars.to(device))
                class_mean = features.mean(dim=0)
                self.class_means[class_id] = class_mean.cpu()

    def select_exemplars(self, data, y, class_id, m):
        class_data = data[y == class_id]
        class_labels = y[y == class_id]
        
        # Calculate class mean
        with torch.no_grad():
            class_features = self.model.encoder(class_data.to(device))
            class_mean = class_features.mean(dim=0)

        # Calculate distance from class mean
        distances = torch.norm(class_features - class_mean, dim=1)
        exemplar_indices = torch.argsort(distances)[:m]

        # Ensure that tensors are on the CPU for correct indexing
        exemplar_indices = exemplar_indices.cpu()
        class_data = class_data.cpu()
        class_labels = class_labels.cpu()

        return class_data[exemplar_indices], class_labels[exemplar_indices]

    def update_exemplar_memory(self, train_loader, class_id):
        self.model.eval()
        data_list, labels_list = [], []
        with torch.no_grad():
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                data_list.append(x)
                labels_list.append(y)
        data = torch.cat(data_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        num_exemplars = self.memory_size // (len(self.exemplar_sets) + 1)
        exemplars, _ = self.select_exemplars(data, labels, class_id, num_exemplars)
        self.exemplar_sets.append(exemplars)

    def train_task(self, task_loader):
        self.model.train()
        for epoch in range(10):  # 필요에 따라 에포크 수 조정
            running_loss = 0.0
            for x, y in task_loader:
                x, y = x.to(device), y.to(device)
                self.optimizer.zero_grad()

                # 현재 태스크 손실 계산
                outputs = self.model(x)
                loss = nn.CrossEntropyLoss()(outputs, y)
                loss.backward()
                
                self.optimizer.step()
                running_loss += loss.item()
            
            print(f'Epoch [{epoch+1}/10], Loss: {running_loss/len(task_loader):.4f}')

    def evaluate_intermediate(self, tasks_up_to_now):
        print("\nEvaluating on tasks up to now:")
        combined_loader = DataLoader(ConcatDataset(tasks_up_to_now), batch_size=32, shuffle=False)
        evaluate(self.model, combined_loader, self.class_means, num_classes=len(dataset.label_map))

    def train_and_evaluate(self, tasks):
        for class_id, task in enumerate(tasks):
            print(f"\nTraining Class {class_id} with iCaRL:")
            train_loader = DataLoader(task, batch_size=32, shuffle=True)
            self.train_task(train_loader)
            self.update_exemplar_memory(train_loader, class_id)
            self.compute_class_means()
            self.evaluate_intermediate(tasks[:class_id + 1])

# 모델 초기화 및 iCaRL 학습
input_dim = 22572  # PRPD 데이터셋에 맞게 조정
hidden_dim = 256
model = PrototypicalNetwork(input_dim, hidden_dim).to(device)
icarl = iCaRL(model, memory_size=400, lr=0.005, weight_decay=0.0001)  # iCaRL 파라미터 설정

# 클래스별 학습 및 중간 평가
icarl.train_and_evaluate(tasks)

# 전체 테스트 세트에 대한 최종 평가
print("\nEvaluating on the entire test set:")
full_test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
evaluate(model, full_test_loader, icarl.class_means, num_classes=len(dataset.label_map))
