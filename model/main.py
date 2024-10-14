# main.py
import torch
from torch.utils.data import DataLoader, random_split
from dataset import AudioMNISTDataset
from models import PrototypicalNetwork, Classifier
from meta_learner import MetaLearner
from dynamic_memory_replay import DynamicMemoryReplay
from visualization import visualize_memory, log_confusion_matrix
from sklearn.metrics import classification_report

# 모델 초기화, 데이터 로더 및 학습 루프 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

input_dim = 11308
hidden_dim = 128
num_classes = 10

prototypical_net = PrototypicalNetwork(input_dim, hidden_dim).to(device)
classifier = Classifier(hidden_dim, num_classes).to(device)
meta_learner = MetaLearner(prototypical_net, classifier, device=device)

root_dir = '../../../Documents/CGB_AI_LAB/Data/Audio_MNIST'
dataset = AudioMNISTDataset(root_dir)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 학습 실행
memory_size = 50000
epochs = 10
memory_replay = DynamicMemoryReplay(memory_size=memory_size, nk=5)
class_labels = list(dataset.label_map.keys())
class_labels = [str(label) for label in class_labels]

# 학습 및 평가 수행
train(meta_learner, train_loader, memory_replay, test_loader, epochs, class_labels)
