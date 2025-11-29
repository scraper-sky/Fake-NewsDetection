import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=30522, embedding_dim=64)
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=128)
        self.fc = nn.Linear(16, 1)
        
    def forward(self, x):
        x1 = self.embedding(x).permute(0, 2, 1)
        x2 = self.conv1(x1)
        x3 = self.pool1(x2)
        x4 = self.conv2(x3)
        x5 = self.pool2(x4)
        x6 = x5.flatten(start_dim=1)
        x7 = self.fc(x6)
        return torch.sigmoid(x7)