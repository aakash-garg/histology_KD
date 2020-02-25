import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet18

class PatchNet(nn.Module):
    def __init__(self, model='resnet50', pretrained=True):
        super(PatchNet, self).__init__()

        assert model in ['resnet18', 'resnet50']
        if model is 'resnet50':
            self.model = resnet50(pretrained=pretrained)
            self.model.avgpool = nn.AdaptiveAvgPool2d(1)
            self.model.fc = nn.Linear(2048, 4)
        if model is 'resnet18':
            self.model = resnet18(pretrained=pretrained)
            self.model.avgpool = nn.AdaptiveAvgPool2d(1)
            self.model.fc = nn.Linear(512, 4)

        for param in self.model.parameters():
            param.require_grad = True

    def forward(self, x):
        output = self.model(x)
        output = F.log_softmax(output, dim=1)
        # output = F.softmax(output, dim=1)
        return output
