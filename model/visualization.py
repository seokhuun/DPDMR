# visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report

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

def log_confusion_matrix(cm, classes):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()
