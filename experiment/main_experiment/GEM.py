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
def evaluate(model, test_loader, num_classes):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
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
            nn.Linear(hidden_dim, hidden_dim),  # 추가 레이어
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, len(dataset.label_map))  # PRPD 클래스 수에 맞게 조정

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)

# GEM 클래스 정의
class GEM:
    def __init__(self, model, memory_size=400, lr=0.005, weight_decay=0.0001, gamma=0.2):
        self.model = model
        self.memory_size = memory_size
        self.exemplar_sets = []
        self.exemplar_labels = []  # 각 예제에 대한 레이블을 저장할 리스트 추가
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.7)
        self.gamma = gamma  # GEM의 허용 오차

    def select_exemplars(self, data, y, class_id, m):
        class_data = data[y == class_id]
        class_labels = y[y == class_id]
        
        # 첫 번째 작업에서는 전체 메모리를 사용
        if len(self.exemplar_sets) == 0:
            num_exemplars = min(m, len(class_data))
        else:
            num_exemplars = min(m // len(self.exemplar_sets), len(class_data))
        
        if class_data.shape[0] > num_exemplars:
            indices = np.random.choice(class_data.shape[0], num_exemplars, replace=False)
        else:
            indices = np.arange(class_data.shape[0])
        return class_data[indices], class_labels[indices]  # 레이블도 함께 반환

    def update_exemplar_memory(self, train_loader, class_id):
        self.model.eval()
        data_list, labels_list = [], []
        with torch.no_grad():
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                data_list.append(x.cpu())
                labels_list.append(y.cpu())
        data = torch.cat(data_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        exemplars, exemplar_labels = self.select_exemplars(data, labels, class_id, self.memory_size)
        self.exemplar_sets.append(exemplars)
        self.exemplar_labels.append(exemplar_labels)  # 레이블 저장

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

                if self.exemplar_sets:
                    # 메모리에 있는 각 태스크에 대해 경사 제한
                    for exemplars, exemplar_labels in zip(self.exemplar_sets, self.exemplar_labels):
                        exemplar_outputs = self.model(exemplars.to(device))
                        exemplar_loss = nn.CrossEntropyLoss()(exemplar_outputs, exemplar_labels.to(device))

                        # 이전 경사를 저장
                        grads_old = [p.grad.clone() for p in self.model.parameters()]

                        # 이전 손실에 대한 경사 계산
                        self.model.zero_grad()
                        exemplar_loss.backward()
                        grads_new = [p.grad.clone() for p in self.model.parameters()]

                        # 경사 제한 적용
                        for p, grad_old, grad_new in zip(self.model.parameters(), grads_old, grads_new):
                            if p.grad is not None:
                                dot_product = (grad_old * grad_new).sum()
                                if dot_product < 0:
                                    proj_coeff = dot_product / (grad_new.norm() ** 2 + 1e-9)
                                    p.grad = grad_old - proj_coeff * grad_new
                                    p.grad = p.grad.clamp_(-self.gamma, self.gamma)
                
                self.optimizer.step()
                running_loss += loss.item()
            
            # 학습률 스케줄러 업데이트
            self.scheduler.step()
            print(f'Epoch [{epoch+1}/10], Loss: {running_loss/len(task_loader):.4f}')

    def evaluate_intermediate(self, tasks_up_to_now):
        print("\nEvaluating on tasks up to now:")
        combined_loader = DataLoader(ConcatDataset(tasks_up_to_now), batch_size=32, shuffle=False)
        evaluate(self.model, combined_loader, num_classes=len(dataset.label_map))

    def train_and_evaluate(self, tasks):
        for class_id, task in enumerate(tasks):
            print(f"\nTraining Class {class_id} with GEM:")
            train_loader = DataLoader(task, batch_size=32, shuffle=True)
            self.train_task(train_loader)
            self.update_exemplar_memory(train_loader, class_id)
            self.evaluate_intermediate(tasks[:class_id + 1])

# 모델 초기화 및 GEM 학습
input_dim = 22572  # PRPD 데이터셋에 맞게 조정
hidden_dim = 256
model = PrototypicalNetwork(input_dim, hidden_dim).to(device)
gem = GEM(model, memory_size=400, lr=0.005, weight_decay=0.0001, gamma=0.2)  # GEM 파라미터 설정

# 클래스별 학습 및 중간 평가
gem.train_and_evaluate(tasks)

# 전체 테스트 세트에 대한 최종 평가
print("\nEvaluating on the entire test set:")
full_test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
evaluate(model, full_test_loader, num_classes=len(dataset.label_map))
