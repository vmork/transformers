from jaxtyping import Float 
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers.vision.base_model import VisionModel

class _ResNetBase(VisionModel):
    """
    Base structure for all models from the [ResNet paper](https://arxiv.org/pdf/1512.03385)
    """
    def __init__(self, n_classes: int, final_dim: int, blocks: nn.Module):
        super().__init__(n_classes)
        self.blocks = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
            blocks,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(final_dim, n_classes)
            
    def forward(self, x: Float[Tensor, "B 3 32 32"]):
        x = self.blocks(x)
        x = x.flatten(1)
        x = self.head(x)
        return x


class _PlainBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, n_layers: int, first_stride: int):
        super().__init__()
        self.convs = nn.Sequential()
        for i in range(n_layers):
            c_in = c_in if i == 0 else c_out 
            stride = first_stride if i == 0 else 1
            self.convs.append(nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(c_out), 
                nn.ReLU(inplace=True)
            ))
            
    def forward(self, x: Float[Tensor, "B 3 32 32"]):
        return self.convs(x)

class PlainNet34(_ResNetBase):
    """ResNet34 without residual connections"""
    def __init__(self, n_classes: int):
        super().__init__(n_classes, final_dim=512, blocks=nn.Sequential(
            _PlainBlock(64, 64, n_layers=6, first_stride=1),
            _PlainBlock(64, 128, n_layers=8, first_stride=2),
            _PlainBlock(128, 256, n_layers=12, first_stride=2),
            _PlainBlock(256, 512, n_layers=6, first_stride=2)))


class _ResidualBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, n_layers: int, first_stride: int):
        super().__init__() 
        self.shortcut_proj = nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, stride=first_stride, padding=0, bias=False),
            nn.BatchNorm2d(c_out)
        )
        
        self.blocks = nn.ModuleList([])
        for i in range(n_layers):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, stride=(first_stride if i == 0 else 1), padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(c_out)
            ))
            c_in = c_out    

    def forward(self, x: Float[Tensor, "B 3 32 32"]):
        for i, block in enumerate(self.blocks):
            identity = self.shortcut_proj(x) if i == 0 else x
            x = F.relu(identity + block(x))
        return x

class ResNet18(_ResNetBase):
    def __init__(self, n_classes: int):
        super().__init__(n_classes, final_dim=512, blocks=nn.Sequential(
            _ResidualBlock(64, 64, n_layers=2, first_stride=1),
            _ResidualBlock(64, 128, n_layers=2, first_stride=2),
            _ResidualBlock(128, 256, n_layers=2, first_stride=2),
            _ResidualBlock(256, 512, n_layers=2, first_stride=2),
        ))

class ResNet34(_ResNetBase):
    def __init__(self, n_classes: int):
        super().__init__(n_classes, final_dim=512, blocks=nn.Sequential(
            _ResidualBlock(64, 64, n_layers=3, first_stride=1),
            _ResidualBlock(64, 128, n_layers=4, first_stride=2),
            _ResidualBlock(128, 256, n_layers=6, first_stride=2),
            _ResidualBlock(256, 512, n_layers=3, first_stride=2),
        ))


class _BottleNeckResidualBlock(nn.Module):
    def __init__(self, c_in: int, c_mid: int, c_out: int, n_layers: int, first_stride: int):
        super().__init__()
        self.shortcut_proj = nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, stride=first_stride, padding=0, bias=False),
            nn.BatchNorm2d(c_out)
        )
            
        self.blocks = nn.ModuleList([])
        for i in range(n_layers):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(c_in, c_mid,  1, stride=(first_stride if i == 0 else 1), padding=0, bias=False),
                nn.BatchNorm2d(c_mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_mid, c_mid, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(c_mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_mid, c_out, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(c_out)
            ))
            c_in = c_out    

    def forward(self, x: Float[Tensor, "B 3 32 32"]):
        for i, block in enumerate(self.blocks):
            identity = self.shortcut_proj(x) if i == 0 else x
            x = F.relu(identity + block(x))
        return x

class ResNet50(_ResNetBase):
    def __init__(self, n_classes: int):
        super().__init__(n_classes, final_dim=2048, blocks=nn.Sequential(
            _BottleNeckResidualBlock(64, 64, 256, n_layers=3, first_stride=1),
            _BottleNeckResidualBlock(256, 128, 512, n_layers=4, first_stride=2),
            _BottleNeckResidualBlock(512, 256, 1024, n_layers=6, first_stride=2),
            _BottleNeckResidualBlock(1024, 512, 2048, n_layers=3, first_stride=2),
        ))

class ResNet101(_ResNetBase):
    def __init__(self, n_classes: int):
        super().__init__(n_classes, final_dim=2048, blocks=nn.Sequential(
            _BottleNeckResidualBlock(64, 64, 256, n_layers=3, first_stride=1),
            _BottleNeckResidualBlock(256, 128, 512, n_layers=4, first_stride=2),
            _BottleNeckResidualBlock(512, 256, 1024, n_layers=23, first_stride=2),
            _BottleNeckResidualBlock(1024, 512, 2048, n_layers=3, first_stride=2),
        ))
        
class ResNet152(_ResNetBase):
    def __init__(self, n_classes: int):
        super().__init__(n_classes, final_dim=2048, blocks=nn.Sequential(
            _BottleNeckResidualBlock(64, 64, 256, n_layers=3, first_stride=1),
            _BottleNeckResidualBlock(256, 128, 512, n_layers=8, first_stride=2),
            _BottleNeckResidualBlock(512, 256, 1024, n_layers=36, first_stride=2),
            _BottleNeckResidualBlock(1024, 512, 2048, n_layers=3, first_stride=2),
        ))