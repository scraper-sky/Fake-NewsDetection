import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

class NewsPredictor(nn.Module):
    def __init__(self, dict_size, category_size, hidden_dim=128, num_layers=2, dropout=0.3):
        super(NewsPredictor, self).__init__()
        
        # Embedding layers
        self.title_embedding = nn.Embedding(dict_size, 128)
        self.text_embedding = nn.Embedding(dict_size, 128)
        self.subject_embedding = nn.Embedding(category_size, 16)
        
        # LSTM layers for sequential processing
        self.title_lstm = nn.LSTM(128, hidden_dim, num_layers, 
                                   batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.text_lstm = nn.LSTM(128, hidden_dim, num_layers, 
                                  batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        
        # Subject processing (simple embedding since it's categorical)
        self.subject_fc = nn.Linear(16, 32)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Final classification layers
        # hidden_dim * 2 because bidirectional, + 32 for subject
        combined_dim = hidden_dim * 2 * 2 + 32  # title + text + subject
        self.fc1 = nn.Linear(combined_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights to prevent collapse"""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.01)
    
    def forward(self, title_id, text_id, subject_id):
        # Embeddings
        title_emb = self.title_embedding(title_id)  # [batch, seq_len, 128]
        text_emb = self.text_embedding(text_id)     # [batch, seq_len, 128]
        subject_emb = self.subject_embedding(subject_id.squeeze(1))  # [batch, 16]
        
        # LSTM processing
        title_lstm_out, (title_hidden, _) = self.title_lstm(title_emb)
        text_lstm_out, (text_hidden, _) = self.text_lstm(text_emb)
        
        # Get final hidden states (bidirectional: forward + backward)
        # title_hidden shape: [num_layers * 2, batch, hidden_dim]
        title_final = torch.cat([title_hidden[-2], title_hidden[-1]], dim=1)  # [batch, hidden_dim*2]
        text_final = torch.cat([text_hidden[-2], text_hidden[-1]], dim=1)      # [batch, hidden_dim*2]
        
        # Subject processing
        subject_features = self.subject_fc(subject_emb)  # [batch, 32]
        subject_features = self.dropout(subject_features)
        
        # Combine features
        combined = torch.cat([title_final, text_final, subject_features], dim=1)
        combined = self.dropout(combined)
        
        # Classification
        x = self.fc1(combined)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        
        return x