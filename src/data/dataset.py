import torch
from torch.utils.data import Dataset

class LogisticRegressionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class TransformerDataset(Dataset):
    def __init__(self, features, labels, tokenizer):
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        encoded = self.tokenizer(feature, padding='max_length', truncation=True, max_length=256, return_tensors='pt')

        input_ids = encoded['input_ids'][0]
        attention_mask = encoded['attention_mask'][0]
        tensor_label = torch.tensor(label, dtype=torch.float32)

        return input_ids, attention_mask, tensor_label

class CNNDataset(Dataset):
    def __init__(self, features, labels, tokenizer):
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        encoded = self.tokenizer(feature, padding='max_length', truncation=True, max_length=256, return_tensors='pt')

        input_ids = encoded['input_ids'][0]
        tensor_label = torch.tensor(label, dtype=torch.float32)

        return input_ids, tensor_label

class RNNDataset(Dataset):
    def __init__(self, features, labels, tokenizer):
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        encoded = self.tokenizer(feature, padding='max_length', truncation=True, max_length=256, return_tensors='pt')

        input_ids = encoded['input_ids'][0]
        tensor_label = torch.tensor(label, dtype=torch.float32)

        return input_ids, tensor_label