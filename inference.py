import sys
import os

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from load import CustomDataset
from vit import ViT
import torch
import torch.nn as nn
import pandas as pd


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def main(filename, image_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        "Using device: ",
        device,
        f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "Not using CUDA",
    )
    print()

    mean = [0.4222541138533666, 0.5447886376476403, 0.5485434299736155]
    std = [0.3002451276771523, 0.2678455320690934, 0.27645630883075295]

    testTransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    csv_test = 'jute-pest-classification/test.csv'
    img_test_dir = 'jute-pest-classification/test_images_'+str(image_size)+'_preprocessed/'

    test_dataset = CustomDataset(csv_file=csv_test, image_dir=img_test_dir, transform=testTransform)
    test_loader = DataLoader(test_dataset)

    model = ViT()
    model.load_state_dict(f"model/{filename}")
    to_device(model, device)
    model.eval()

    predictions = []
    true_labels = []
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = to_device(inputs, device), to_device(labels, device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    test_loss /= len(test_loader.dataset)

    predictions = np.array(predictions)
    pred_labels = np.argmax(predictions, axis=1)
    # print(pred_labels)
    # print(pred_labels.shape)
    true_labels = np.array(true_labels)
    # print(true_labels)
    # print(true_labels.shape)

    print(f"Test Loss: {test_loss:.6f}")

    # Calculate metrics
    test_f1 = f1_score(true_labels, pred_labels, average='macro')  # or 'micro', 'weighted'
    test_accuracy = accuracy_score(true_labels, pred_labels)
    test_precision = precision_score(true_labels, pred_labels, average='macro')  # or 'micro', 'weighted'
    test_recall = recall_score(true_labels, pred_labels, average='macro')  # or 'micro', 'weighted'
    # test_specificity = recall_score(true_labels, pred_labels.round(), pos_label=0)
    # fpr, tpr, thresholds = roc_curve(true_labels, pred_labels)
    # test_auc = auc(fpr, tpr)

    print("Test F1:", test_f1)
    print("Test Accuracy:", test_accuracy)
    print("Test Precision:", test_precision)
    print("Test Recall:", test_recall)

    df = pd.read_csv(csv_test)
    df = df.drop(columns=['filename'])
    df['class'] = predictions

    dict_file_path = 'jute-pest-classification/transformation_dict.txt'
    label_dict = {}

    with open(dict_file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split(': ')
            label_dict[int(value)] = key  # Convert value to int for matching with predictions

    decoded_labels = [label_dict[label] for label in pred_labels]

    df['class'] = [label_dict[label] for label in df['class']]

    df.to_csv('final_predictions.csv', index=False)

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))

