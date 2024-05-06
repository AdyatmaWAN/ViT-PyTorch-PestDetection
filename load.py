import torch
import cv2
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.df.iloc[idx, 0])
        image = cv2.imread(img_name)
        # OpenCV reads in BGR format, convert it to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert NumPy array to PIL image and apply transformations
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        label = self.df.iloc[idx, 1]

        return image, label


class CustomDatasetTest(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # If your CSV file includes the index column as the first column in pandas, adjust the column index accordingly
        img_name = os.path.join(self.image_dir, self.df.iloc[idx, 1])  # Assuming column 1 is the filename
        image = cv2.imread(img_name)
        if image is None:
            raise FileNotFoundError(f"Image {img_name} not found.")

        # OpenCV reads in BGR format, convert it to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert NumPy array to PIL image and apply transformations
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        # Check if there is a second column for labels; if so, fetch the label
        if self.df.shape[1] > 2:  # Adjust this if your CSV has different columns
            label = self.df.iloc[idx, 2]  # Assuming column 2 is the label
            return image, label

        return image


class ApplyTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        return {'image': self.transform(image), 'label': label}
