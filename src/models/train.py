import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
from src.models.model import NewsPredictor
import os
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

data_dir = 'src/data'
id_data_dir = os.path.join(data_dir, 'ID-data')

fake_train = pd.read_csv(os.path.join(id_data_dir, 'FakeIDTrain.csv'))
true_train = pd.read_csv(os.path.join(id_data_dir, 'TrueIDTrain.csv'))

train_data = pd.concat([fake_train, true_train], ignore_index=True)
# Shuffle with stratification to ensure balanced split
shuffled_data = train_data.sample(frac=1, random_state=229, ignore_index=True).reset_index(drop=True)

# Stratified split to ensure balanced train/val distribution
train_split, val_split = train_test_split(
    shuffled_data, 
    test_size=0.2, 
    random_state=42, 
    stratify=shuffled_data['label']  # Ensure balanced label distribution
)
train_split = train_split.reset_index(drop=True)
val_split = val_split.reset_index(drop=True)

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

    model = NewsPredictor(dict_size, category_size, hidden_dim=128, num_layers=2, dropout=0.5).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Balanced learning rate

    train_dataset = NewsDataset(train_split)
    val_dataset = NewsDataset(val_split)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model.eval()
    initial_losses = []
    with torch.no_grad():
        sample_size = min(100, len(train_split))
        for i in range(0, sample_size, batch_size):
            batch_end = min(i + batch_size, sample_size)
            batch_data = [train_dataset[j] for j in range(i, batch_end)]
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
    print(f'Training on {len(train_split)} samples, Validation on {len(val_split)} samples')
    print(f'Batch size: {batch_size}')
    print('-' * 50)
    
    model.train()
    for epoch in range(num_epochs):
        epoch_losses = []
        all_preds = []
        all_labels = []
        
        # Training phase
        for batch_idx, (titles, texts, subjects, labels) in enumerate(train_dataloader):
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
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Print progress every 100 batches with prediction stats
            if (batch_idx + 1) % 100 == 0:
                with torch.no_grad():
                    pred_probs = outputs.detach().cpu().numpy()
                    avg_pred = pred_probs.mean()
                    std_pred = pred_probs.std()
                    print(f'Epoch: {epoch:3d} | Batch: {batch_idx + 1:4d}/{len(train_dataloader)} | Loss: {loss.item():.6f} | Avg Pred: {avg_pred:.4f} | Std: {std_pred:.4f}')
        
        # Calculate training accuracy
        train_preds = (torch.tensor(all_preds) > 0.5).float()
        train_labels = torch.tensor(all_labels)
        train_acc = (train_preds == train_labels).float().mean().item()
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        
        # Validation phase
        model.eval()
        val_losses = []
        val_preds = []
        val_labels_list = []
        with torch.no_grad():
            for titles, texts, subjects, labels in val_dataloader:
                titles = titles.to(device)
                texts = texts.to(device)
                subjects = subjects.to(device)
                labels = labels.to(device)
                
                outputs = model(titles, texts, subjects)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                val_preds.extend(outputs.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
        
        val_preds_tensor = (torch.tensor(val_preds) > 0.5).float()
        val_labels_tensor = torch.tensor(val_labels_list)
        val_acc = (val_preds_tensor == val_labels_tensor).float().mean().item()
        avg_val_loss = sum(val_losses) / len(val_losses)
        
        model.train()
        
        print(f'Epoch: {epoch:3d} | Train Loss: {avg_train_loss:.6f} | Train Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.6f} | Val Acc: {val_acc:.4f}')
    
    # Save model
    model_path = 'src/models/NewsPredictor.pth'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')
    
    # Evaluate on actual test set
    print('\n' + '=' * 50)
    print('Evaluating on Test Set')
    print('=' * 50)
    
    fake_test = pd.read_csv(os.path.join(id_data_dir, 'FakeIDTest.csv'))
    true_test = pd.read_csv(os.path.join(id_data_dir, 'TrueIDTest.csv'))
    test_data = pd.concat([fake_test, true_test], ignore_index=True)
    test_dataset = NewsDataset(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    model.eval()
    test_losses = []
    test_preds = []
    test_labels_list = []
    
    with torch.no_grad():
        for titles, texts, subjects, labels in test_dataloader:
            titles = titles.to(device)
            texts = texts.to(device)
            subjects = subjects.to(device)
            labels = labels.to(device)
            
            outputs = model(titles, texts, subjects)
            loss = criterion(outputs, labels)
            test_losses.append(loss.item())
            test_preds.extend(outputs.cpu().numpy())
            test_labels_list.extend(labels.cpu().numpy())
    
    test_preds_tensor = (torch.tensor(test_preds) > 0.5).float()
    test_labels_tensor = torch.tensor(test_labels_list)
    test_acc = (test_preds_tensor == test_labels_tensor).float().mean().item()
    avg_test_loss = sum(test_losses) / len(test_losses)
    
    print(f'Test Loss: {avg_test_loss:.6f}')
    print(f'Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)')
    print('=' * 50)

if __name__ == '__main__':
    train()
            