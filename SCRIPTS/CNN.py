import torch
import torch.nn as nn
import torch.nn.functional as F

GRID_DIMS = (18, 11) # dimensions

class BinaryClimbCNN(nn.Module):

    def __init__(self, n_classes, n_channels=8):
        super(BinaryClimbCNN, self).__init__()
        self.n_classes = n_classes

        self.cnn = nn.Sequential(
            nn.Conv2d(1, n_channels, (3,3), padding=1, bias=True),               # (N, n_channels, 18, 11)
            nn.BatchNorm2d(n_channels),                                          # (N, n_channels, 18, 11)
            nn.ReLU(),                                                           # (N, n_channels, 18, 11)
            nn.MaxPool2d(2, padding=1),                                          # (N, n_channels, 10, 6)
            
            nn.Conv2d(n_channels, n_channels*2, (3,3), padding=1, bias=True),    # (N, n_channels*2, 10, 6)
            nn.BatchNorm2d(n_channels*2),                                        # (N, n_channels*2, 10, 6)
            nn.ReLU(),                                                           # (N, n_channels*2, 10, 6)
            nn.MaxPool2d(2),                                                     # (N, n_channels*2, 5, 3)
            
            nn.Conv2d(n_channels*2, n_channels*4, (3,3), padding=1, bias=True),  # (N, n_channels*4, 5, 3)
            nn.BatchNorm2d(n_channels*4),                                        # (N, n_channels*4, 5, 3)
            nn.ReLU(),                                                           # (N, n_channels*4, 5, 3)
            nn.MaxPool2d(2, padding=1),                                          # (N, n_channels*4, 3, 2)
            
            nn.Conv2d(n_channels*4, n_channels*8, (3,3), padding=1, bias=True),  # (N, n_channels*8, 3, 2)
            nn.BatchNorm2d(n_channels*8),                                        # (N, n_channels*8, 3, 2)
            nn.ReLU(),                                                           # (N, n_channels*8, 3, 2)
            nn.AvgPool2d((3, 2)),                                                # (N, n_channels*8, 1, 1)
            
            nn.Conv2d(n_channels*8, n_channels*8, (1,1), bias=True),             # (N, n_channels*8, 1, 1)
            nn.BatchNorm2d(n_channels*8),                                        # (N, n_channels*8, 1, 1)
            nn.ReLU()                                                           # (N, n_channels*8, 1, 1)
            )
        
        self.fc = nn.Linear(64, n_classes)
        
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = torch.unsqueeze(x, 1)  # Add a channel dimension: (N, 1, 18, 11)
        x = self.cnn(x)            # Pass through the CNN layers
        x = torch.flatten(x, 1)    # Flatten the output for the fully connected layer
        logits = self.fc(x)        # Pass through the fully connected layer
        return logits