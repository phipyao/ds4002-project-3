import torch
import torch.nn as nn
import torch.nn.functional as F

GRID_DIMS = (18, 11) # dimensions

class BinaryClimbCNN(nn.Module):

    def __init__(self, n_classes, n_channels=8):
        super(BinaryClimbCNN, self).__init__()
        self.n_classes = n_classes

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, 3, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(2),

            nn.Conv2d(64, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.fc = nn.Linear(64, n_classes)
        
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = torch.unsqueeze(x, 1)       # (N, 1, 18, 11)
        logits = self.network(x)        # (N, n_classes, 1, 1)
        logits = torch.squeeze(logits)       # (N, n_classes)
        return logits