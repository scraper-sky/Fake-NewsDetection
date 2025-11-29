import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=30522, embedding_dim=64)
        self.rnn = nn.RNN(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x1 = self.embedding(x)
        outputs, hiddens = self.rnn(x1)
        final_output = outputs[:, -1, :]
        x2 = final_output
        x3 = self.fc(x2)
        return torch.sigmoid(x3)
    