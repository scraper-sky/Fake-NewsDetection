import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=30522, embedding_dim=256)
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x1 = self.embedding(x)
        outputs, hiddens = self.lstm(x1)
        final_output = outputs[:, -1, :]
        x2 = final_output
        x3 = self.fc(x2)
        return torch.sigmoid(x3)
    