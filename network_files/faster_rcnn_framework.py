import torch
from torch import nn




class FasterRCNNBase(nn.Module):
    def __init__(self,backbone,rpn,roi_heads,transforms):
        super().__init__()
        self.transform=transforms
        self.backbone=backbone
        self.rpn=rpn
        self.roi_heads=roi_heads

