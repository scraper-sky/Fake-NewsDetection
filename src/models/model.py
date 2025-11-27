import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

class NewsPredictor(nn.Module):
    def __init__(self, dict_size, hidden_dim=128, num_layers=2, dropout=0.5):
        super(NewsPredictor, self).__init__()
        
        self.title_embedding = nn.Embedding(dict_size, 128)
        self.text_embedding = nn.Embedding(dict_size, 128)
        
        self.title_lstm = nn.LSTM(128, hidden_dim, num_layers, 
                                   batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.text_lstm = nn.LSTM(128, hidden_dim, num_layers, 
                                  batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        
        self.dropout = nn.Dropout(dropout)
        
        combined_dim = hidden_dim * 2 * 2
        self.fc1 = nn.Linear(combined_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.01)
    
    def forward(self, title_id, text_id):
        title_emb = self.title_embedding(title_id)
        text_emb = self.text_embedding(text_id)
        
        title_lstm_out, (title_hidden, _) = self.title_lstm(title_emb)
        text_lstm_out, (text_hidden, _) = self.text_lstm(text_emb)
        
        title_final = torch.cat([title_hidden[-2], title_hidden[-1]], dim=1)
        text_final = torch.cat([text_hidden[-2], text_hidden[-1]], dim=1)
        
        combined = torch.cat([title_final, text_final], dim=1)
        combined = self.dropout(combined)
        x = self.fc1(combined)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        
        return x