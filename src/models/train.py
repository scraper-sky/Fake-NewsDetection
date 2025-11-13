import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import json
from src.models.model import NewsPredictor

fake_train = pd.read_csv('src/data/ID-data/FakeIDTrain.csv')
true_train = pd.read_csv('src/data/ID-data/TrueIDTrain.csv')

train_data = pd.concat([fake_train, true_train], ignore_index=True)
shuffled_data = train_data.sample(frac=1, random_state=229, ignore_index=True)


with open('src/data/vocab_dict.json', 'r') as file:
    vocab_dict = json.load(file)
with open('src/data/subject_dict.json', 'r') as file:
    subject_dict = json.load(file)

num_epochs = 100

def train():
    dict_size = len(vocab_dict)
    category_size = len(subject_dict)

    model = NewsPredictor(dict_size, category_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        for i in range(len(shuffled_data)):
            row = shuffled_data.iloc[i]
            title_id = json.loads(row['title_id'])
            text_id = json.loads(row['text_id'])
            subject_id = json.loads(row['subject_id'])
            label = json.loads(row['label'])

            title_tensor = torch.tensor(title_id, dtype=torch.long).reshape(1, -1)
            text_tensor = torch.tensor(text_id, dtype=torch.long).reshape(1, -1)
            subject_tensor = torch.tensor(subject_id, dtype=torch.long).reshape(1, -1)
            label_tensor = torch.tensor(label, dtype=torch.float32).reshape(1, -1)

            output = model(title_tensor, text_tensor, subject_tensor).reshape(1, -1)
            loss = criterion(output, label_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch: {epoch} and Loss: {loss.item()}')
    
    torch.save(model.state_dict(), 'src/models/NewsPredictor.pth')

if __name__ == '__main__':
    train()
            