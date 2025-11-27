import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
from src.models.model import NewsPredictor
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

data_dir = 'src/data'
id_data_dir = os.path.join(data_dir, 'ID-data')

fake_train = pd.read_csv(os.path.join(id_data_dir, 'FakeIDTrain.csv'))
true_train = pd.read_csv(os.path.join(id_data_dir, 'TrueIDTrain.csv'))

train_data = pd.concat([fake_train, true_train], ignore_index=True)
shuffled_data = train_data.sample(frac=1, random_state=229, ignore_index=True).reset_index(drop=True)

with open(os.path.join(data_dir, 'vocab_dict.json'), 'r') as file:
    vocab_dict = json.load(file)
with open(os.path.join(data_dir, 'subject_dict.json'), 'r') as file:
    subject_dict = json.load(file)

num_epochs = 10
batch_size = 32  

class NewsDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        title_id = json.loads(row['title_id'])
        text_id = json.loads(row['text_id'])
        subject_id = row['subject_id']
        label = row['label']
        
        return {
            'title': torch.tensor(title_id, dtype=torch.long),
            'text': torch.tensor(text_id, dtype=torch.long),
            'subject': torch.tensor([subject_id], dtype=torch.long),
            'label': torch.tensor([label], dtype=torch.float32)
        }

def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    titles = torch.stack([item['title'] for item in batch])
    texts = torch.stack([item['text'] for item in batch])
    subjects = torch.stack([item['subject'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    return titles, texts, subjects, labels

def train():
    dict_size = len(vocab_dict)
    category_size = len(subject_dict)

    model = NewsPredictor(dict_size, category_size, hidden_dim=128, num_layers=2, dropout=0.3).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate for stability

    dataset = NewsDataset(shuffled_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

    model.eval()
    initial_losses = []
    with torch.no_grad():
        sample_size = min(100, len(shuffled_data))
        for i in range(0, sample_size, batch_size):
            batch_end = min(i + batch_size, sample_size)
            batch_data = [dataset[j] for j in range(i, batch_end)]
            titles, texts, subjects, labels = collate_fn(batch_data)
            
            titles = titles.to(device)
            texts = texts.to(device)
            subjects = subjects.to(device)
            labels = labels.to(device)
            
            outputs = model(titles, texts, subjects)
            loss = criterion(outputs, labels)
            initial_losses.append(loss.item())
    
    avg_initial_loss = sum(initial_losses) / len(initial_losses)
    print(f'Initial Loss (before training): {avg_initial_loss:.6f}')
    print(f'Training on {len(shuffled_data)} samples with batch size {batch_size}')
    print('-' * 50)
    
    model.train()
    for epoch in range(num_epochs):
        epoch_losses = []
        
        for batch_idx, (titles, texts, subjects, labels) in enumerate(dataloader):
            titles = titles.to(device)
            texts = texts.to(device)
            subjects = subjects.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(titles, texts, subjects)
            loss = criterion(outputs, labels)
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            # Print progress every 100 batches to verify it's working
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch: {epoch:3d} | Batch: {batch_idx + 1:4d}/{len(dataloader)} | Loss: {loss.item():.6f}')
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f'Epoch: {epoch:3d} | Average Loss: {avg_loss:.6f}')
    
    # Save model
    model_path = 'src/models/NewsPredictor.pth'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

if __name__ == '__main__':
    train()
            