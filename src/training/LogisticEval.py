import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from src.training.LogisticTraining import get_data_loaders, category_count
from src.models.LogisticRegression import LogisticRegression    

GPU_status = torch.cuda.is_available()
if GPU_status:
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

BATCH_SIZE = 32
DATA_TRUE_PATH = 'NewsDataset/true_processed.csv'
DATA_FAKE_PATH = 'NewsDataset/fake_processed.csv'
MODEL_PATH = 'NewsResults/LogisticRegression/LogisticRegression.pth'
SAVE_PATH = 'NewsResults/LogisticRegression/'   

def evaluate_set(model, loader, criterion, device):
    model.eval()
    loss_total = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in tqdm(loader):
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features).reshape(-1)
            loss = criterion(outputs, labels)

            loss_total += loss.item() * labels.size(0)
            predictions = (outputs >= 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    loss_avg = loss_total/ total
    accuracy = correct/ total
    return loss_avg, accuracy

def main():
    model = LogisticRegression()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    criterion = nn.BCELoss()
    train_loader, dev_loader, test_loader = get_data_loaders()
    train_loss, train_accuracy = evaluate_set(model, train_loader, criterion, DEVICE)
    dev_loss, dev_accuracy = evaluate_set(model, dev_loader, criterion, DEVICE)
    test_loss, test_accuracy = evaluate_set(model, test_loader, criterion, DEVICE)
    print(f'Train Loss = {train_loss}, Train Accuracy = {train_accuracy}')
    print(f'Dev Loss = {dev_loss}, Dev Accuracy = {dev_accuracy}')
    print(f'Test Loss = {test_loss}, Test Accuracy = {test_accuracy}')

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
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(SAVE_PATH + 'confusion_matrix.png')
    plt.close()

if __name__ == '__main__':  
    main()