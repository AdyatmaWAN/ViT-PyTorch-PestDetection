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

from load import CustomDataset
from vit import ViT

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

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# Function to train model
# def train_model(X_train, y_train, X_val, y_val, X_test, y_test, batch_size, lr, opt_name, fold):
def train_model(train_loader, val_loader, test_loader, batch_size, lr, opt_name, fold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout, emb_dropout=emb_dropout)
    # model = TransformerModel(input_shape=(50, 64, 64, 1), head_size=128, num_heads=4, ff_dim=4, num_transformer_blocks=4, mlp_units=[64, 32], dropout=0, mlp_dropout=0.05)
    model.to(device)

    criterion = nn.BCELoss()
    if opt_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = getattr(optim, opt_name)(model.parameters(), lr=lr)

    # Training loop
    best_val_loss = float('inf')
    patience = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            print("Train data type:", type(inputs), type(labels))
            print("Train data shape:", inputs.shape, labels.shape)

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
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
    torch.save(model.state_dict(), f"model/fold_{fold}_batch_{batch_size}_lr_{lr}_opt_{opt_name}.pt")

    # Test the model
    test_loader = DataLoader(test_loader, batch_size=batch_size, shuffle=False)
    model.eval()
    test_loss = 0.0
    predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            test_loss += loss.item() * inputs.size(0)
            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    test_loss /= len(test_loader.dataset)

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    print(f"Fold: {fold}, Test Loss: {test_loss:.6f}")

    # Calculate metrics
    test_f1 = f1_score(true_labels, predictions.round())
    test_accuracy = accuracy_score(true_labels, predictions.round())
    test_precision = precision_score(true_labels, predictions.round())
    test_recall = recall_score(true_labels, predictions.round())
    test_specificity = recall_score(true_labels, predictions.round(), pos_label=0)
    fpr, tpr, thresholds = roc_curve(true_labels, predictions)
    test_auc = auc(fpr, tpr)

    print("Test F1:", test_f1)
    print("Test Accuracy:", test_accuracy)
    print("Test Precision:", test_precision)
    print("Test Recall:", test_recall)
    print("Test Specificity:", test_specificity)
    print("Test AUC:", test_auc)

# Main function
def main(batch, lr, opt_name):
    # h5_file_path = 'data/fase2_interval_4ch_gray_128x128.h5'
    # X, Y = load_data(h5_file_path)

    mean = [0.4222541138533666, 0.5447886376476403, 0.5485434299736155]
    std = [0.3002451276771523, 0.2678455320690934, 0.27645630883075295]

    trainTransform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    testTransform = transforms.Compose([
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
    kf = StratifiedKFold(n_splits=10)
    for fold, (train_index, test_index) in enumerate(kf.split(data_df, labels)):
        train_data = Subset(train_set, train_index)
        test_data = Subset(train_set, test_index)

        # Further split train data into train and validation sets
        train_indices, val_indices = train_test_split(train_index, test_size=0.1, stratify=labels[train_index])

        train_data = Subset(train_set, train_indices)
        val_data = Subset(train_set, val_indices)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

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
    c, w, h = 3, 224, 224

    weight_decay = 0.001
    num_epochs = 5

    image_size = 224
    patch_size = 16
    num_classes = 17
    dim = 1024
    depth = 6
    heads = 16
    mlp_dim = 2048
    dropout = 0.1
    emb_dropout = 0.1
    # image_size = 224  # We'll resize input images to this size
    # patch_size = 16  # Size of the patches to be extract from the input images
    # num_patches = (image_size // patch_size) ** 2
    # num_heads = 4

    # Set seed for reproducibility
    SEED = 1
    set_seeds(SEED)

    batch_size = int(sys.argv[1])
    learning_rate = float(sys.argv[2])
    opt_name = sys.argv[3]
    main(batch_size, learning_rate, opt_name)
