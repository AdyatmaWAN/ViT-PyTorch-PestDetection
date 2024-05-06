import sys
import os

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from load import CustomDatasetTest
from vit import ViT
import torch
import torch.nn as nn
import pandas as pd

global c
global image_size
global patch_size
global num_classes
global dim
global depth
global heads
global mlp_dim
global dropout
global emb_dropout
global warmup_epochs


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

    test_dataset = CustomDatasetTest(csv_file=csv_test, image_dir=img_test_dir, transform=testTransform)
    test_loader = DataLoader(test_dataset)

    model = ViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout, emb_dropout=emb_dropout, channels=c)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(f"../ViT/{filename}"))
    else:
        model.load_state_dict(torch.load(f"../ViT/{filename}", map_location=torch.device('cpu')))
    to_device(model, device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = to_device(inputs, device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())

    predictions = np.array(predictions)
    pred_labels = np.argmax(predictions, axis=1)
    print(pred_labels)

    df = pd.read_csv(csv_test)
    df = df.drop(columns=['filename'])
    # df['class'] = predictions

    dict_file_path = 'jute-pest-classification/transformation_dict.txt'
    label_dict = {}

    with open(dict_file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split(': ')
            label_dict[int(value)] = key  # Convert value to int for matching with predictions

    # decoded_labels = [label_dict[label] for label in pred_labels]

    df['class'] = [label_dict[label] for label in pred_labels]

    df.to_csv('final_predictions.csv', index=False)

if __name__ == "__main__":

    c, w, h = 3, 256, 256

    weight_decay = 0.05
    num_epochs = 1000

    # #AWS
    # image_size = 128
    # patch_size = 16
    # num_classes = 17
    # dim = 16
    # depth = 16
    # heads = 16
    # mlp_dim = 64
    # dropout = 0.1
    # emb_dropout = 0.1

    # DGX-A100-1
    #image_size = 256
    #patch_size = 16
    #num_classes = 17
    #dim = 192
    #depth = 9
    #heads = 12
    #mlp_dim = 384
    #dropout = 0.01
    #emb_dropout = 0.01
    #warmup_epochs = 10

    # DGX-V100-1
    image_size = 256
    patch_size = 16
    num_classes = 17
    # dim = 16
    dim = 192
    # depth = 16
    depth = 9
    # heads = 16
    heads = 12
    mlp_dim = 384
    # mlp_dim = 64
    # mlp_dim = 2
    dropout = 0.1
    emb_dropout = 0.1
    warmup_epochs = 10



    # # DGX-A100-2 (Grey)
    # c, w, h = 1, 256, 256
    #
    # weight_decay = 0.00005
    # num_epochs = 1000
    #
    # image_size = 256
    # patch_size = 16
    # num_classes = 17
    # # dim = 16
    # dim = 32
    # # depth = 16
    # depth = 16
    # # heads = 16
    # heads = 16
    # mlp_dim = 64
    # # mlp_dim = 64
    # # mlp_dim = 2
    # dropout = 0.01
    # emb_dropout = 0.01
    # warmup_epochs = 10

    main(sys.argv[1], int(sys.argv[2]))

