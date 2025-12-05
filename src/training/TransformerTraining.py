import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from src.data.dataset import TransformerDataset
from src.models.transformer import Transformer

GPU_status = torch.cuda.is_available()
if GPU_status:
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0001
DATA_TRUE_PATH = 'NewsDataset/true_processed.csv'
DATA_FAKE_PATH = 'NewsDataset/fake_processed.csv'
SAVE_PATH = 'NewsResults/Transformer/'
MODEL_NAME = 'transformer.pth'
TOKENIZER = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
MODEL = Transformer()
    
def get_set_split():
    true_data = pd.read_csv(DATA_TRUE_PATH)
    fake_data = pd.read_csv(DATA_FAKE_PATH)
    
    N_train = 15000
    N_dev = 2500
    N_test = 2500

    true_train = true_data.iloc[:N_train]
    true_dev = true_data.iloc[N_train:N_train+N_dev]
    true_test = true_data.iloc[N_train+N_dev:N_train+N_dev+N_test]

    fake_train = fake_data.iloc[:N_train]
    fake_dev = fake_data.iloc[N_train:N_train+N_dev]
    fake_test = fake_data.iloc[N_train+N_dev:N_train+N_dev+N_test]

    train_set = pd.concat([true_train, fake_train]).sample(frac=1, random_state=229)
    dev_set = pd.concat([true_dev, fake_dev]).sample(frac=1, random_state=229)
    test_set = pd.concat([true_test, fake_test]).sample(frac=1, random_state=229)

    return train_set, dev_set, test_set

def get_data_loaders():
    train_set, dev_set, test_set = get_set_split()
    train_dataset = TransformerDataset(train_set['combined'].tolist(), train_set['label'].tolist(), TOKENIZER)
    dev_dataset = TransformerDataset(dev_set['combined'].tolist(), dev_set['label'].tolist(), TOKENIZER)
    test_dataset = TransformerDataset(test_set['combined'].tolist(), test_set['label'].tolist(), TOKENIZER)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, dev_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    loss_total = 0
    correct = 0
    total = 0

    for input_ids, attention_mask, labels in tqdm(train_loader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask).reshape(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        predictions = (outputs >= 0.5).float()
        loss_total += loss.item() * labels.size(0)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    loss_avg = loss_total/ total
    accuracy = correct/ total
    return loss_avg, accuracy

def dev_epoch(model, dev_loader, criterion, device):
    model.eval()
    loss_total = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for input_ids, attention_mask, labels in dev_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask).reshape(-1)
            loss = criterion(outputs, labels)

            loss_total += loss.item() * labels.size(0)
            predictions = (outputs >= 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    loss_avg = loss_total/ total
    accuracy = correct/ total
    return loss_avg, accuracy

def get_label_accuracy(predictions, all_labels):
    tp_count = 0
    tn_count = 0
    true_count = 0
    p_count = 0
    n_count = 0
    total_count = 0
    for i in range(len(all_labels)):
        total_count += 1
        if all_labels[i] == 1:
            p_count += 1
            if predictions[i] == 1:
                tp_count += 1
                true_count += 1
        else:
            n_count += 1
            if predictions[i] == 0:
                tn_count += 1
                true_count += 1
    positive_accuracy = tp_count/ p_count
    negative_accuracy = tn_count/ n_count
    total_accuracy = (true_count)/ total_count
    return positive_accuracy, negative_accuracy, total_accuracy

def threshold_curve(model, dev_loader, device):
    model.eval()
    thresholds = np.linspace(0, 1, 100)
    positive_accuracy = []
    negative_accuracy = []
    total_accuracy = []

    with torch.no_grad():
        all_outputs = []
        all_labels = []
        for input_ids, attention_mask, labels in dev_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(input_ids, attention_mask).reshape(-1)
            all_outputs.append(outputs)
            all_labels.append(labels)
        
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)

    for threshold in thresholds:
        predictions = (all_outputs >= threshold).float()
        positive_item, negative_item, total_item = get_label_accuracy(predictions, all_labels)

        positive_accuracy.append(positive_item)
        negative_accuracy.append(negative_item)
        total_accuracy.append(total_item)

    return thresholds, positive_accuracy, negative_accuracy, total_accuracy

def category_count(model, test_loader, threshold, device):
    model.eval()
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(input_ids, attention_mask).reshape(-1)
            predictions = (outputs >= threshold).float()
            for i in range(len(labels)):
                if labels[i] == 1:
                    if predictions[i] == 1:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if predictions[i] == 0:
                        tn += 1
                    else:
                        fp += 1

    return tp, fp, tn, fn

def main():
    model = MODEL
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.005)
    criterion = nn.BCELoss()

    train_loader, dev_loader, test_loader = get_data_loaders()

    train_loss_list = []
    dev_loss_list = []
    train_accuracy_list = []
    dev_accuracy_list = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)

        dev_loss, dev_accuracy = dev_epoch(model, dev_loader, criterion, DEVICE)
        dev_loss_list.append(dev_loss)
        dev_accuracy_list.append(dev_accuracy)

        print('Epoch:', epoch)
        print('Train Loss:', train_loss)
        print('Train Accuracy:', train_accuracy)
        print('Dev Loss:', dev_loss)
        print('Dev Accuracy:', dev_accuracy)
    
    torch.save(model.state_dict(), SAVE_PATH + MODEL_NAME)

    epochs_list = np.linspace(1, EPOCHS, EPOCHS)

    plt.figure()
    plt.plot(epochs_list, train_loss_list, label='Train Loss')
    plt.plot(epochs_list, dev_loss_list, label='Dev Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig(SAVE_PATH + 'loss_curve.png')
    plt.close()

    plt.figure()
    plt.plot(epochs_list, train_accuracy_list, label='Train Accuracy')
    plt.plot(epochs_list, dev_accuracy_list, label='Dev Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.savefig(SAVE_PATH + 'accuracy_curve.png')
    plt.close()

    thresholds, positive_accuracy, negative_accuracy, total_accuracy = threshold_curve(model, dev_loader, DEVICE)
    plt.figure()
    plt.plot(thresholds, positive_accuracy, label='Positive Accuracy')
    plt.plot(thresholds, negative_accuracy, label='Negative Accuracy')
    plt.plot(thresholds, total_accuracy, label='Total Accuracy')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Threshold Curve')
    plt.legend()
    plt.savefig(SAVE_PATH + 'threshold_curve.png')
    plt.close()

    tp, fp, tn, fn = category_count(model, test_loader, 0.5, DEVICE)
    matrix = np.array([[tp, fn], [fp, tn]])
    plt.figure(figsize=(7, 7))
    plt.imshow(matrix, cmap='Blues')
    plt.colorbar()
    plt.xticks([0, 1], ['Predicted Positive', 'Predicted Negative'])
    plt.yticks([0, 1], ['Actual Positive', 'Actual Negative'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, matrix[i, j], ha='center', va='center', color='black')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    # title_string = 'Confusion Matrix With Best Threshold = ' + str(selected_threshold)
    title_string = 'Confusion Matrix'
    plt.title(title_string)
    plt.tight_layout()
    plt.savefig(SAVE_PATH + 'confusion_matrix.png')
    plt.close()
    
if __name__ == '__main__':
    main()