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
        pass 