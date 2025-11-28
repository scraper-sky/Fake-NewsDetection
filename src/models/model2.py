import torch
import torch.nn as nn
import torch.nn.functional as f


class NewsPredictorCNN(nn.Module):
    def __init__(self, dict_size, embedding_dim=128, conv_filters=64, dropout=0.5):
        super().__init__()
        self.title_embedding = nn.Embedding(dict_size, embedding_dim)
        self.text_embedding = nn.Embedding(dict_size, embedding_dim)

        self.title_conv = nn.Conv1d(embedding_dim, conv_filters, kernel_size=3)
        self.text_conv = nn.Conv1d(embedding_dim, conv_filters, kernel_size=3)

        self.dropout = nn.Dropout(dropout)

        combined_dim = conv_filters * 2
        self.fc1 = nn.Linear(combined_dim, 64)
        self.fc2 = nn.Linear(64, 1)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.01)

    def _encode(self, emb_layer, conv_layer, inputs):
        emb = emb_layer(inputs)
        emb = emb.permute(0, 2, 1)
        conv_out = conv_layer(emb)
        pooled = torch.max(conv_out, dim=2).values
        return pooled

    def forward(self, title_id, text_id):
        title_features = self._encode(self.title_embedding, self.title_conv, title_id)
        text_features = self._encode(self.text_embedding, self.text_conv, text_id)

        combined = torch.cat([title_features, text_features], dim=1)
        combined = self.dropout(combined)

        x = self.fc1(combined)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x

