import torch 
from torch import nn
from torch.nn import functional as F
import torchvision

class MobilenetV2Embedding(nn.Module):
    def __init__(self, out_features=192):
        super(MobilenetV2Embedding, self).__init__()
        net = torchvision.models.mobilenet_v2(pretrained=True)
        net.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1280, out_features=out_features)
        )
        self.feature_extractor = net
    
    def forward(self, x):
        out = self.feature_extractor(x)
        out = F.normalize(out, p=2, dim=1)
        return out
    
    

class Resnet50Embedding(nn.Module):
    def __init__(self, out_features=192):
        super(Resnet50Embedding, self).__init__()
        net = torchvision.models.resnet50(pretrained=True)
        net.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1000, out_features=out_features)
        )
        self.feature_extractor = net
    
    def forward(self, x):
        out = self.feature_extractor(x)
        out = F.normalize(out, p=2, dim=1)
        return out
