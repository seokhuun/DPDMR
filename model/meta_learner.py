# meta_learner.py
import torch
import torch.nn.functional as F
import torch.optim as optim
from losses import triplet_loss
from dataset import create_triplets

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

    # 다른 메서드들은 원래대로 유지
