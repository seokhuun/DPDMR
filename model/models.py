# models.py
import torch
import torch.nn as nn

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

class Classifier(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        return self.classifier(x)
