# dynamic_memory_replay.py
import torch
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

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
