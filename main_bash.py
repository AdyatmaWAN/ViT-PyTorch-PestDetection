import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
from torchvision import transforms
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from load import CustomDataset
from vit import ViT
from lion_pytorch import Lion


global weight_decay
global num_epochs

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
global c


# Set seed for reproducibility
def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Function to move data and model to appropriate device
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# Function to train model
def train_model(train_loader, val_loader, test_loader, batch, lr, opt_name, fold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        "Using device: ",
        device,
        f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "Not using CUDA",
    )
    print()

    model = ViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout, emb_dropout=emb_dropout, channels=c)
    # model = TransformerModel(input_shape=(50, 64, 64, 1), head_size=128, num_heads=4, ff_dim=4, num_transformer_blocks=4, mlp_units=[64, 32], dropout=0, mlp_dropout=0.05)
    model = to_device(model, device)

    criterion = nn.CrossEntropyLoss()
    if opt_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif opt_name == "Lion":
        optimizer = Lion(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = getattr(optim, opt_name)(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Define the scheduler for warm-up
    def adjust_learning_rate(optimizer, epoch, lr):
        """ Adjusts the learning rate according to warm-up schedule """
        if epoch < warmup_epochs:
            lr = lr * float(epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        return lr

    # Training loop
    best_val_loss = float('inf')
    patience = 0
    for epoch in range(num_epochs):
        current_lr = adjust_learning_rate(optimizer, epoch, lr)
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - LR: {current_lr:.6f}", leave=False)
        model.train()
        train_loss = 0.0
        for inputs, labels in train_iterator:
            inputs, labels = to_device(inputs, device), to_device(labels, device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # Remove unsqueeze(1) here
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_iterator = tqdm(val_loader, desc=f"Validation", leave=False)
        for inputs, labels in val_iterator:
            inputs, labels = to_device(inputs, device), to_device(labels, device)
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Fold: {fold}, Epoch: {epoch+1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
        else:
            patience += 1
            if patience >= 15:
                print("Early stopping...")
                break

    # Save model
    folder_path = f"model"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    torch.save(model.state_dict(), f"{folder_path}/fold_{fold}_batch_{batch}_lr_{lr}_opt_{opt_name}.pt")

    # Test the model
    test_loader = test_loader
    model.eval()
    test_loss = 0.0
    predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
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

    print(f"Fold: {fold}, Test Loss: {test_loss:.6f}")

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
    # print("Test Specificity:", test_specificity)
    # print("Test AUC:", test_auc)

    test_results = pd.DataFrame({
        "batch": [batch],
        "lr": [lr],
        "Optimization": [opt_name],
        "Fold": [fold],
        "Test F1": [test_f1],
        "Test Accuracy": [test_accuracy],
        "Test Precision": [test_precision],
        "Test Recall": [test_recall],
        # "Test Specificity": [test_specificity],
        # "Test AUC": [test_auc]
    })

    # Determine Excel file path
    excel_file_path = f"test_results.xlsx"

    # Check if Excel file exists
    if os.path.isfile(excel_file_path):
        # If file exists, open it and append new data
        existing_data = pd.read_excel(excel_file_path)
        combined_data = pd.concat([existing_data, test_results], ignore_index=True)
        combined_data.to_excel(excel_file_path, index=False)
        print("Test results appended to existing Excel file:", excel_file_path)
    else:
        # If file doesn't exist, create a new Excel file and save the data
        test_results.to_excel(excel_file_path, index=False)
        print("Test results saved to new Excel file:", excel_file_path)

# Main function
def main(batch, lr, opt_name):
    # h5_file_path = 'data/fase2_interval_4ch_gray_128x128.h5'
    # X, Y = load_data(h5_file_path)

    mean = [0.4222541138533666, 0.5447886376476403, 0.5485434299736155]
    std = [0.3002451276771523, 0.2678455320690934, 0.27645630883075295]

    trainTransform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    testTransform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    print("Loading data...")
    print()

    csv_train = 'jute-pest-classification/train_'+str(image_size)+'_preprocessed_encoded.csv'
    csv_test = 'jute-pest-classification/test.csv'

    img_train_dir = 'jute-pest-classification/train_images_'+str(image_size)+'_preprocessed/'
    img_test_dir = 'jute-pest-classification/test_images_'+str(image_size)+'_preprocessed/'

    train_set = CustomDataset(csv_file=csv_train, image_dir=img_train_dir, transform=trainTransform)
    # test_set = CustomDataset(csv_file=csv_test, image_dir=img_test_dir, transform=testTransform)

    data_df = pd.read_csv(csv_train)
    labels = data_df['class'].values

    print("Data loaded")
    print("Train set size: ", len(train_set))
    # print("Test set size: ", len(test_set))
    print()

    # Use Stratified K-Fold cross-validation
    kf = StratifiedKFold(n_splits=5)
    for fold, (train_index, test_index) in enumerate(kf.split(data_df, labels)):
        # Split data indices into train, validation, and test
        train_data_df, test_data_df = data_df.iloc[train_index], data_df.iloc[test_index]
        train_indices, val_indices = train_test_split(train_index, test_size=0.1, stratify=labels[train_index])

        # print(train_data_df.shape)
        # print(test_data_df.shape)
        # print()
        # print(train_indices.shape)
        # print(val_indices.shape)

        # Create train, validation, and test datasets
        train_data = CustomDataset(csv_file=csv_train, image_dir=img_train_dir, transform=trainTransform)
        val_data = CustomDataset(csv_file=csv_train, image_dir=img_train_dir, transform=testTransform)
        test_data = CustomDataset(csv_file=csv_train, image_dir=img_train_dir, transform=testTransform)

        # Assign indices for train, validation, and test datasets
        train_data = Subset(train_data, train_indices)
        val_data = Subset(val_data, val_indices)
        test_data = Subset(test_data, test_index)
        #
        # print(len(train_data))
        # print(len(val_data))
        # print(len(test_data))

        # Create DataLoader objects
        train_loader = DataLoader(train_data, batch_size=batch, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=batch, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch, shuffle=False)

        # Print some information about the DataLoader objects
        print("Train DataLoader Info:")
        print("Number of batches in train_loader:", len(train_loader))
        print("Number of samples in the train dataset:", len(train_loader.dataset))
        print("Shape of the first batch of inputs:", next(iter(train_loader))[0].shape)
        print("Shape of the first batch of labels:", next(iter(train_loader))[1].shape)
        print()

        print("Validation DataLoader Info:")
        print("Number of batches in val_loader:", len(val_loader))
        print("Number of samples in the validation dataset:", len(val_loader.dataset))
        print("Shape of the first batch of inputs:", next(iter(val_loader))[0].shape)
        print("Shape of the first batch of labels:", next(iter(val_loader))[1].shape)
        print()

        print("Test DataLoader Info:")
        print("Number of batches in test_loader:", len(test_loader))
        print("Number of samples in the test dataset:", len(test_loader.dataset))
        print("Shape of the first batch of inputs:", next(iter(test_loader))[0].shape)
        print("Shape of the first batch of labels:", next(iter(test_loader))[1].shape)
        print()

        # X_train, y_train = X[train_index], Y[train_index]
        # X_test, y_test = X[test_index], Y[test_index]
        #
        # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.1)
        #
        # print(f"Fold: {fold}")
        #
        # # Train model
        train_model(train_loader, val_loader, test_loader, batch, lr, opt_name, fold)


if __name__ == "__main__":
    c, w, h = 3, 256, 256

    weight_decay = 0.00005
    num_epochs = 1000

    image_size = 256
    patch_size = 16
    num_classes = 17
    # dim = 16
    dim = 32
    # depth = 16
    depth = 16
    # heads = 16
    heads = 16
    mlp_dim = 64
    # mlp_dim = 64
    # mlp_dim = 2
    dropout = 0.01
    emb_dropout = 0.01
    warmup_epochs = 10

    # Set seed for reproducibility
    SEED = 1
    set_seeds(SEED)

    batch_size = int(sys.argv[1])
    learning_rate = float(sys.argv[2])
    opt_name = sys.argv[3]
    weight_decay = learning_rate/3
    # weight_decay = learning_rate/3 #weight decay based on learning rate
    main(batch_size, learning_rate, opt_name)
