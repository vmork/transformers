from jaxtyping import Float 
import torch
import torch.nn as nn
from torch import Tensor

from transformer.vision.base_model import VisionModel

class _VGGBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, n_layers=2, ker_size=3, padding=1):
        super().__init__()
        self.convs = nn.Sequential()
        for i in range(n_layers):
            c_in = c_in if i == 0 else c_out
            self.convs.append(nn.Conv2d(c_in, c_out, ker_size, padding=padding))
            self.convs.append(nn.BatchNorm2d(c_out))
            self.convs.append(nn.ReLU(inplace=True))
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x: Float[Tensor, "B 3 32 32"]):
        return self.pool(self.convs(x))
    
class VGG16(VisionModel):
    """
    From [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556)
    """
    def __init__(self, n_classes: int, p_dropout=0.75):
        super().__init__(n_classes)
        self.blocks = nn.Sequential(
            _VGGBlock(3, 64, 2),    # 32,32 -> 16,16
            _VGGBlock(64, 128, 2),  # 16,16 -> 8,8
            _VGGBlock(128, 256, 3), # 8,8   -> 4,4
            _VGGBlock(256, 512, 3), # 4,4   -> 2,2
            _VGGBlock(512, 512, 3), # 2,2   -> 1,1 
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p_dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p_dropout),
            nn.Linear(4096, n_classes)
        )
    
    def forward(self, x: Float[Tensor, "B 3 32 32"]):
        x = self.blocks(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x