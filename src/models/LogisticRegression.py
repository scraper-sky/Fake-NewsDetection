import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(30, 1)
        
    def forward(self, x):
        x1 = self.fc(x)
        return torch.sigmoid(x1)