import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=30522, embedding_dim=64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x1 = self.embedding(x)
        outputs, _ = self.lstm(x1)
        max_pooled, _ = torch.max(outputs, dim=1)
        x2 = max_pooled
        x3 = self.fc(x2)
        return torch.sigmoid(x3)
    