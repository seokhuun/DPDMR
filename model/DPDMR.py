import os
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, Subset, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score, confusion_matrix, precision_score, recall_score, f1_score, 
    roc_auc_score, log_loss, matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score, 
    jaccard_score, fowlkes_mallows_score, hamming_loss, zero_one_loss, classification_report,
    roc_curve, auc, precision_recall_curve
)
import seaborn as sns
import logging

# 로깅 설정
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
logger = logging.getLogger()

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

# Classifier 정의
class Classifier(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        return self.classifier(x)

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
        if not positive_idx or not negative_idx:
            continue
        positive = x[positive_idx[torch.randint(len(positive_idx), (1,))]]
        negative = x[negative_idx[torch.randint(len(negative_idx), (1,))]]
        anchors.append(anchor)
        positives.append(positive)
        negatives.append(negative)
    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)

# MetaLearner 정의
class MetaLearner:
    def __init__(self, proto_net, classifier, inner_lr=0.01, outer_lr=0.001, weight_decay=1e-4, device='cpu'):
        self.proto_net = proto_net
        self.classifier = classifier
        self.inner_optimizer = optim.SGD(self.classifier.parameters(), lr=inner_lr, weight_decay=weight_decay)
        self.outer_optimizer = optim.Adam(
            list(self.proto_net.parameters()) + list(self.classifier.parameters()), lr=outer_lr, weight_decay=weight_decay
        )
        self.device = device
    
    def inner_update(self, support_set):
        self.classifier.train()
        self.proto_net.eval()
        support_set = (support_set[0].to(self.device), support_set[1].to(self.device))
        classification_loss = self.compute_loss(support_set)
        prototypical_loss = self.compute_prototypical_loss(support_set)
        triplet_loss_value = self.compute_triplet_loss(support_set)
        loss = classification_loss + 0.5 * prototypical_loss + 0.5 * triplet_loss_value  # 가중치 조정
        self.inner_optimizer.zero_grad()
        loss.backward()
        self.inner_optimizer.step()
        return loss.item()
    
    def compute_loss(self, data):
        x, y = data
        x, y = x.to(self.device), y.to(self.device)
        embeddings = self.proto_net(x)
        logits = self.classifier(embeddings)
        loss = F.cross_entropy(logits, y)
        return loss
    
    def compute_prototypical_loss(self, support_set):
        x, y = support_set
        x, y = x.to(self.device), y.to(self.device)
        unique_labels = y.unique()
        if len(unique_labels) == 0:
            return torch.tensor(0.0, requires_grad=True, device=self.device)
        prototypes = []
        for class_label in unique_labels:
            class_indices = (y == class_label).nonzero(as_tuple=True)[0]
            if len(class_indices) > 0:
                class_prototypes = self.proto_net(x[class_indices]).mean(0)
                prototypes.append(class_prototypes)
        if len(prototypes) == 0:
            return torch.tensor(0.0, requires_grad=True, device=self.device)
        prototypes = torch.stack(prototypes)
        prototypical_loss = 0
        for idx in range(x.size(0)):
            distances = torch.cdist(self.proto_net(x[idx].unsqueeze(0)), prototypes)
            target = torch.where(unique_labels == y[idx])[0].item()
            prototypical_loss += F.cross_entropy(-distances, torch.tensor([target], device=self.device))
        return prototypical_loss
    
    def compute_triplet_loss(self, data):
        x, y = data
        x, y = x.to(self.device), y.to(self.device)
        anchors, positives, negatives = create_triplets(x, y)
        anchor_embeddings = self.proto_net(anchors)
        positive_embeddings = self.proto_net(positives)
        negative_embeddings = self.proto_net(negatives)
        return triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
    
    def outer_update(self, query_set, memory):
        self.proto_net.train()
        self.classifier.train()
        query_set = (query_set[0].to(self.device), query_set[1].to(self.device))
        classification_loss = self.compute_loss(query_set)
        prototypical_loss = self.compute_prototypical_loss(query_set)
        triplet_loss_value = self.compute_triplet_loss(query_set)
        query_loss = classification_loss + prototypical_loss + triplet_loss_value
        if memory is not None:
            memory = (memory[0].to(self.device), memory[1].to(self.device))
            memory_classification_loss = self.compute_loss(memory)
            memory_prototypical_loss = self.compute_prototypical_loss(memory)
            memory_triplet_loss_value = self.compute_triplet_loss(memory)
            memory_loss = memory_classification_loss + memory_prototypical_loss + memory_triplet_loss_value
            total_loss = query_loss + memory_loss
        else:
            total_loss = query_loss
        self.outer_optimizer.zero_grad()
        total_loss.backward()
        self.outer_optimizer.step()
        return total_loss.item()

# DynamicMemoryReplay 클래스 정의
class DynamicMemoryReplay:
    def __init__(self, memory_size=1000, nk=10):
        self.memory = []
        self.memory_size = memory_size
        self.nk = nk

    def update_memory(self, proto_net, data, n_clusters):
        x, y = data
        x, y = x.to(proto_net.encoder[0].weight.device), y.to(proto_net.encoder[0].weight.device)
        x_hidden = proto_net(x).cpu().detach().numpy()
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=33, n_init=10).fit(x_hidden)
        neigh = NearestNeighbors(n_neighbors=self.nk)
        neigh.fit(x_hidden)
        
        memory_data_buffer = []
        memory_label_buffer = []
        
        for center in kmeans.cluster_centers_:
            _, neighbors = neigh.kneighbors([center], n_neighbors=self.nk, return_distance=True)
            for neighbor_idx in neighbors[0]:
                memory_data_buffer.append(x[neighbor_idx].unsqueeze(0))
                memory_label_buffer.append(y[neighbor_idx].unsqueeze(0))
        
        new_memory_data = torch.cat(memory_data_buffer).to(proto_net.encoder[0].weight.device)
        new_memory_labels = torch.cat(memory_label_buffer).to(proto_net.encoder[0].weight.device)

        if len(self.memory) < self.memory_size:
            self.memory.append((new_memory_data, new_memory_labels))
        else:
            self.memory.pop(0)
            self.memory.append((new_memory_data, new_memory_labels))

    def sample_memory(self):
        if len(self.memory) == 0:
            return None
        x_memory, y_memory = zip(*self.memory)
        return torch.cat(x_memory), torch.cat(y_memory)

# 시각화 함수 정의
def visualize_memory(memory_data, memory_labels, title):
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto')
    memory_data_np = memory_data.cpu().detach().numpy()
    memory_labels_np = memory_labels.cpu().detach().numpy()
    X_embedded = tsne.fit_transform(memory_data_np)
    
    plt.figure(figsize=(10, 8))
    plt.title(title)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=memory_labels_np, cmap='Spectral', s=50, alpha=0.6)
    plt.colorbar()
    plt.show()

# 성능 지표 시각화 함수 정의
def visualize_metrics(metrics_dict, title):
    for metric_name, metric_values in metrics_dict.items():
        plt.figure()
        plt.plot(range(len(metric_values)), metric_values, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} Over Epochs')
        plt.grid(True)
        plt.show()

# 성능 지표 기록 함수 수정
def log_performance_metrics(epoch, y_true, y_pred, y_pred_proba, loss, classes, metrics_dict):
    y_pred = np.array(y_pred).flatten()  # Flatten the prediction array
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
    logloss = log_loss(y_true, y_pred_proba)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    jaccard = jaccard_score(y_true, y_pred, average='weighted', zero_division=0)
    fmi = fowlkes_mallows_score(y_true, y_pred)
    hamming = hamming_loss(y_true, y_pred)
    zero_one = zero_one_loss(y_true, y_pred)

    metrics = {
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC': auc,
        'Log Loss': logloss,
        'MCC': mcc,
        'Cohen\'s Kappa': kappa,
        'Balanced Accuracy': balanced_acc,
        'Jaccard Index': jaccard,
        'FMI': fmi,
        'Hamming Loss': hamming,
        'Zero-One Loss': zero_one
    }

    for metric_name, metric_value in metrics.items():
        if metric_name not in metrics_dict:
            metrics_dict[metric_name] = []
        metrics_dict[metric_name].append(metric_value)

    logger.info(f'Epoch {epoch}: Loss: {loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, '
                f'F1-Score: {f1:.4f}, AUC: {auc:.4f}, Log Loss: {logloss:.4f}, MCC: {mcc:.4f}, '
                f'Cohen\'s Kappa: {kappa:.4f}, Balanced Accuracy: {balanced_acc:.4f}, '
                f'Jaccard Index: {jaccard:.4f}, FMI: {fmi:.4f}, Hamming Loss: {hamming:.4f}, '
                f'Zero-One Loss: {zero_one:.4f}')

# 혼동 행렬 시각화 함수 정의
def log_confusion_matrix(cm, classes):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()

# ROC 및 PR 곡선 시각화 함수 정의
def plot_roc_pr_curves(all_targets, all_pred_proba, classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    precision = dict()
    recall = dict()
    pr_auc = dict()
    n_classes = len(classes)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(np.array(all_targets) == i, all_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        precision[i], recall[i], _ = precision_recall_curve(np.array(all_targets) == i, all_pred_proba[:, i])
        pr_auc[i] = auc(recall[i], precision[i])

    # Plot ROC curves
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # Plot PR curves
    plt.figure()
    for i in range(n_classes):
        plt.plot(recall[i], precision[i], lw=2, label=f'PR curve of class {classes[i]} (area = {pr_auc[i]:0.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")
    plt.show()

# 테스트 함수 정의 (정확도와 리콜을 출력하도록 수정)
def test(meta_learner, test_loader, epoch, classes, metrics_dict):
    meta_learner.proto_net.eval()
    meta_learner.classifier.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_targets = []
    all_predictions = []
    all_pred_proba = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(meta_learner.device), y.to(meta_learner.device)
            embeddings = meta_learner.proto_net(x)
            logits = meta_learner.classifier(embeddings)
            loss = F.cross_entropy(logits, y).item()
            total_loss += loss * x.size(0)
            predictions = torch.argmax(logits, dim=1)
            pred_proba = torch.softmax(logits, dim=1)
            total_correct += (predictions == y).sum().item()
            total_samples += y.size(0)
            all_targets.extend(y.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_pred_proba.extend(pred_proba.cpu().numpy())

    accuracy = total_correct / total_samples
    avg_loss = total_loss / total_samples
    cm = confusion_matrix(all_targets, all_predictions)
    
    # 클래스별 성능 보고서 출력
    class_report = classification_report(all_targets, all_predictions, target_names=[str(c) for c in classes])
    print("Classification Report:\n", class_report)

    all_pred_proba = np.array(all_pred_proba)
    all_pred_proba = all_pred_proba / all_pred_proba.sum(axis=1, keepdims=True)

    log_performance_metrics(epoch, all_targets, all_predictions, all_pred_proba, avg_loss, classes, metrics_dict)
    log_confusion_matrix(cm, classes)

    precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)

    print(f'Test set: Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}')
    return avg_loss, accuracy, recall, precision

# 전체 테스트 세트에서 모델 평가 (정확도와 리콜을 출력하도록 수정)
def evaluate(meta_learner, test_loader, class_labels, epochs, metrics_dict):
    meta_learner.proto_net.eval()
    test_loss = 0
    correct = 0
    all_targets = []
    all_predictions = []
    all_pred_proba = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).long()
            embeddings = meta_learner.proto_net(data)
            output = meta_learner.classifier(embeddings)
            test_loss += F.cross_entropy(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(pred.cpu().numpy().flatten())  # Flatten the predictions
            all_pred_proba.extend(torch.softmax(output, dim=1).cpu().numpy())

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
    precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}')

    # 성능 지표 기록
    cm = confusion_matrix(all_targets, all_predictions)

    # Ensure y_pred_proba sums to 1 for each sample
    all_pred_proba = np.array(all_pred_proba)
    all_pred_proba = all_pred_proba / all_pred_proba.sum(axis=1, keepdims=True)

    log_performance_metrics(epochs, all_targets, all_predictions, all_pred_proba, test_loss, class_labels, metrics_dict)
    log_confusion_matrix(cm, class_labels)
    plot_roc_pr_curves(all_targets, all_pred_proba, class_labels)

    memory_sample = memory_replay.sample_memory()
    if memory_sample:
        visualize_memory(memory_sample[0], memory_sample[1], "Final memory space after training")

# 학습 및 테스트 정확도 시각화 함수 정의
def plot_accuracies(train_accuracies, test_accuracies):
    plt.figure()
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

# 학습 함수 정의
def train(meta_learner, data_loader, memory_replay, test_loader, epochs=10, replay_frequency=50, support_size=16, n_clusters=5, class_labels=None):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    test_recalls = []
    test_precisions = []
    metrics_dict = {}  # 이 부분을 추가하여 metrics_dict 변수를 초기화합니다.
    
    for epoch in range(epochs):
        total_correct = 0
        total_samples = 0
        for i, batch in enumerate(data_loader):
            support_set, query_set = split_support_query(batch, support_size)
            inner_loss = meta_learner.inner_update(support_set)
            if i % replay_frequency == 0:
                memory = memory_replay.sample_memory()
                if memory is not None:
                    memory = (memory[0].to(meta_learner.device), memory[1].to(meta_learner.device))
                total_loss = meta_learner.outer_update(query_set, memory)
            else:
                total_loss = meta_learner.outer_update(query_set, None)
            with torch.no_grad():
                memory_replay.update_memory(meta_learner.proto_net, support_set, n_clusters)
            
            # Calculate training accuracy
            embeddings = meta_learner.proto_net(query_set[0].to(meta_learner.device))
            logits = meta_learner.classifier(embeddings)
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == query_set[1].to(meta_learner.device)).sum().item()
            total_samples += query_set[1].size(0)
            
            if (i + 1) % 100 == 0:
                logger.info(f'Epoch {epoch+1}/{epochs}, Step {i+1}, Loss: {total_loss:.4f}')
        
        train_accuracy = total_correct / total_samples
        train_accuracies.append(train_accuracy)
        
        test_loss, test_accuracy, test_recall, test_precision = test(meta_learner, test_loader, epoch, class_labels, metrics_dict)
        train_losses.append(total_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        test_recalls.append(test_recall)
        test_precisions.append(test_precision)
        logger.info(f'Epoch {epoch+1}/{epochs}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    # 학습 곡선 시각화
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    plot_accuracies(train_accuracies, test_accuracies)
    visualize_metrics(metrics_dict, 'Performance Metrics Over Epochs')

# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

# 모델 초기화
input_dim = 11308  # 스펙트로그램 특징 차원 (1025 frequency bins, 44 frames)
hidden_dim = 128
num_classes = 10  # AudioMNIST 데이터셋의 클래스 수

prototypical_net = PrototypicalNetwork(input_dim, hidden_dim).to(device)
classifier = Classifier(hidden_dim, num_classes).to(device)
meta_learner = MetaLearner(prototypical_net, classifier, device=device)

# 데이터셋 및 데이터 로더 정의
root_dir = '../../../Documents/CGB_AI_LAB/Data/Audio_MNIST'  # AudioMNIST 데이터셋의 루트 디렉토리 경로
dataset = AudioMNISTDataset(root_dir)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 학습 실행
n_classes = 10
memory_size = 50000
samples_per_class = 459
batch_size = 32
epochs = 10

memory_replay = DynamicMemoryReplay(memory_size=memory_size, nk=5)
class_labels = list(dataset.label_map.keys())
class_labels = [str(label) for label in class_labels]  # Convert class labels to strings

metrics_dict = {}  # Initialize metrics_dict here

for class_num in range(n_classes):
    print(f'\n=== Class {class_num} ===')
    
    class_indices = (torch.tensor(train_dataset.dataset.labels) == class_num).nonzero(as_tuple=True)[0][:samples_per_class]
    class_subset = Subset(train_dataset.dataset, class_indices)
    class_loader = DataLoader(class_subset, batch_size=batch_size, shuffle=True)
    
    X_class, y_class = next(iter(class_loader))
    X_class = X_class.to(device)
    y_class = y_class.to(device).long()  # Ensure target is of type long
    
    # 이전 클래스 성능 평가
    if class_num > 0:
        meta_learner.proto_net.eval()
        with torch.no_grad():
            previous_accuracy = []
            for prev_class_num in range(class_num):
                prev_class_indices = (torch.tensor(train_dataset.dataset.labels) == prev_class_num).nonzero(as_tuple=True)[0][:samples_per_class]
                prev_class_subset = Subset(train_dataset.dataset, prev_class_indices)
                prev_class_loader = DataLoader(prev_class_subset, batch_size=batch_size, shuffle=True)
                
                correct = 0
                total = 0
                for data, target in prev_class_loader:
                    data, target = data.to(device), target.to(device)
                    embeddings = meta_learner.proto_net(data)
                    output = meta_learner.classifier(embeddings)
                    pred = output.argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += target.size(0)
                
                prev_accuracy = correct / total
                previous_accuracy.append(prev_accuracy)
                print(f'Accuracy on class {prev_class_num}: {prev_accuracy:.4f}')
    
    X_class_flat = X_class.view(X_class.shape[0], -1).cpu().detach().numpy()
    tsne = TSNE(n_components=2, random_state=33)
    X_class_tsne = tsne.fit_transform(X_class_flat)

    silhouette_scores = []
    n_clusters_range = range(2, min(len(X_class_flat), 11))
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=33, n_init=10).fit(X_class_tsne)
        score = silhouette_score(X_class_tsne, kmeans.labels_)
        silhouette_scores.append(score)
    optimal_n_clusters = np.argmax(silhouette_scores) + 2
    kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=33).fit(X_class_tsne)

    print("Searching for optimal n of clusters...")
    print(f"Optimal n of clusters for class {class_num}: {optimal_n_clusters} (score: {np.max(silhouette_scores):.4f})")
    print(f"Set: n_clusters: {optimal_n_clusters}, n_neighbors: {int(memory_size/n_classes/optimal_n_clusters)} (Memory allocated for class {class_num}: {optimal_n_clusters}*{int(memory_size/n_classes/optimal_n_clusters)}={memory_size})")

    plt.figure(figsize=(10, 8))
    plt.title(f"Input space for class: {class_num} with KMeans Centers")
    plt.scatter(X_class_tsne[:, 0], X_class_tsne[:, 1], c=kmeans.labels_, cmap='viridis', s=50, alpha=0.5)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=50, alpha=0.6, edgecolor='black', marker='^')
    plt.colorbar()
    plt.show()

    n_neighbors = min(int((memory_size/n_classes)/optimal_n_clusters), len(X_class_tsne))
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(X_class_tsne)
    memory_data_buffer = []
    memory_label_buffer = []
    for center in kmeans.cluster_centers_:
        _, neighbors = neigh.kneighbors([center], n_neighbors=n_neighbors, return_distance=True)
        for neighbor_idx in neighbors[0]:
            memory_data_buffer.append(X_class[neighbor_idx].unsqueeze(0))
            memory_label_buffer.append(y_class[neighbor_idx].unsqueeze(0))

    memory_sample = memory_replay.sample_memory()
    if memory_sample:
        memory_data = torch.cat((memory_sample[0], torch.cat(memory_data_buffer).to(device)), dim=0)
        memory_labels = torch.cat((memory_sample[1], torch.cat(memory_label_buffer).to(device)), dim=0)
    else:
        memory_data = torch.cat(memory_data_buffer).to(device)
        memory_labels = torch.cat(memory_label_buffer).to(device)
    
    if len(memory_data) > memory_size:
        memory_data = memory_data[-memory_size:]
        memory_labels = memory_labels[-memory_size:]

    memory_replay.memory = [(memory_data, memory_labels)]

    visualize_memory(memory_data, memory_labels, f"Memory space in episode: {class_num}")

    combined_data = torch.cat((X_class, memory_data))
    combined_labels = torch.cat((y_class, memory_labels))
    combined_dataset = TensorDataset(combined_data, combined_labels)
    combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    
    meta_learner.proto_net.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(combined_loader):
            data, target = data.to(device), target.to(device).long()
            meta_learner.inner_optimizer.zero_grad()
            embeddings = meta_learner.proto_net(data)
            output = meta_learner.classifier(embeddings)
            loss = F.cross_entropy(output, target)
            loss.backward()
            meta_learner.inner_optimizer.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}')

    if class_num > 0:
        meta_learner.proto_net.eval()
        with torch.no_grad():
            memory = memory_replay.sample_memory()
            if memory:
                output_previous = meta_learner.classifier(meta_learner.proto_net(memory[0]))
                loss_previous = F.cross_entropy(output_previous, memory[1]).item()
                accuracy_previous = (output_previous.argmax(dim=1) == memory[1]).float().mean().item()
                print(f'Performance on previous classes after class {class_num}: Loss: {loss_previous:.4f}, Accuracy: {accuracy_previous:.4f}')

    visualize_memory(memory_data, memory_labels, f"Memory space in episode: {class_num}")

    # 분리된 테스트 데이터셋을 사용한 중간 평가
    test_loss, test_accuracy, test_recall, test_precision = test(meta_learner, test_loader, class_num, class_labels, metrics_dict)
    print(f'Intermediate Test set: Average loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, Recall: {test_recall:.4f}, Precision: {test_precision:.4f}')

# 전체 테스트 세트에서 모델 평가
evaluate(meta_learner, test_loader, class_labels, epochs, metrics_dict)
