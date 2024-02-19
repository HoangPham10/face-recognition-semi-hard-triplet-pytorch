import torch 
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler
import numpy as np
from tqdm.auto import tqdm

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """
    def __init__(self, dataset, n_classes, n_samples):
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = self.n_classes * self.n_samples
        self.dataset = dataset
    
        self.labels = []
        for i in tqdm(range(dataset.__len__())):
            _, label = self.dataset.__getitem__(i)
            self.labels.append(int(label))
        self.labels = torch.LongTensor(self.labels)
        self.labels_set = list(set(self.labels.numpy()))
        self.labels2indices = {label: np.where(self.labels.numpy() == label)[0] for label in self.labels_set}
        
        # Shuffle the labels
        for label in self.labels_set:
            np.random.shuffle(self.labels2indices[label])
        
        self.n_visited_samples = {label: 0 for label in self.labels_set}
        self.count = 0

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)

            indices = []
            
            for class_ in classes:
                indices.extend(self.labels2indices[class_][
                            self.n_visited_samples[class_]:self.n_visited_samples[class_] + self.n_samples])
                 
                self.n_visited_samples[class_] += self.n_samples
             
                if self.n_visited_samples[class_] + self.n_samples > len(self.labels2indices[class_]):
                    np.random.shuffle(self.labels2indices[class_])
                    self.n_visited_samples[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples



     
    def __len__(self):
        return len(self.dataset) // self.batch_size
    

def semi_triplet_hard_mining(embeddings, labels, margin = 1):
    embeddings = embeddings.detach().cpu()
    num = embeddings.shape[0]

    dist = torch.sum(embeddings ** 2, dim=1) + torch.sum(embeddings ** 2, dim=1).view(-1, 1) - 2 * embeddings.matmul(embeddings.T)
    dist = F.relu(dist).sqrt()
    dist = dist.numpy()


    hori = labels.expand((num, num))
    verti = labels.view(-1, 1).expand((num, num))
    mask = (hori == verti).numpy().astype(np.int) # Same label = 1, different = 0
    anchor = []
    posi_index = []
    nega_index = []
    for i in range(dist.shape[0]): 
        for j in range(dist.shape[0]):
          if mask[i,j] == 0.: 
            continue 
          if i == j: 
            continue 
          dp = dist[i,j]
          for k in range(dist.shape[0]):
            if mask[i,k] == 1: 
              continue 
            dn = dist[i,k]
            loss = dp - dn + margin 
            if loss > 0: 
              anchor.append(i)
              posi_index.append(j)
              nega_index.append(k)

    anchor = np.asarray(anchor)
    posi_index = np.asarray(posi_index)
    nega_index = np.asarray(nega_index) 
    batch_semi = np.vstack([anchor, posi_index, nega_index]).T
    return batch_semi
