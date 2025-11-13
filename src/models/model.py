import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

class NewsPredictor(nn.module):
    def __init__(self, dict_size, category_size):
        super(NewsPredictor, self).__init__()
        self.title_embedding = nn.embedding(dict_size, 128)
        self.text_embedding = nn.embedding(dict_size, 128)
        self.subject_embedding = nn.embedding(category_size, 8)

        self.title_conv = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=0)
        self.text_conv = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=0)
        self.subject_conv = nn.Conv1d(8, 64, kernel_size=3, stride=1, padding=0)

        self.fc1 = nn.Linear(192, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, title_id, text_id, subject_id):
        title_embedding = self.title_embedding(title_id)
        text_embedding = self.text_embedding(text_id)
        subject_embedding = self.subject_embedding(subject_id)

        title_embedding = title_embedding.permute(0, 2, 1)
        text_embedding = text_embedding.permute(0, 2, 1)
        subject_embedding = subject_embedding.permute(0, 2, 1)

        title_conv = self.title_conv(title_embedding)
        text_conv = self.text_conv(text_embedding)
        subject_conv = self.subject_conv(subject_embedding)

        title_pooled = torch.max(title_conv, dim=2).values
        text_pooled = torch.max(text_conv, dim=2).values
        subject_pooled = torch.max(subject_conv, dim=2).values

        x = torch.cat([title_pooled, text_pooled, subject_pooled], dim=1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x