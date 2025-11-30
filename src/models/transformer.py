import torch
import torch.nn as nn
from transformers import DistilBertModel

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')

        # for parameter in self.encoder.parameters():
        #     parameter.requires_grad = False

        self.fc = nn.Linear(768, 1)
        
    def forward(self, input_ids, attention_mask):
        encoded_layers = self.encoder(input_ids, attention_mask=attention_mask)
        last_encoded = encoded_layers.last_hidden_state
        token = last_encoded[:, 0, :]
        x = self.fc(token)
        return torch.sigmoid(x)
