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
        label = self.df.iloc[idx, 1]

        # Convert NumPy array to PIL image
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)

        return {'image': image, 'label': label}

class ApplyTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        return {'image': self.transform(image), 'label': label}

# # Example of usage
# # Assuming you have defined your transformations if any (e.g., normalization, resizing, etc.)
# # transform = transforms.Compose([transforms.ToTensor()])

# # Initialize your custom dataset
# dataset = CustomDataset(csv_file='/path/to/your/csv/file.csv',
#                         image_dir='/path/to/your/image/directory/',
#                         transform=None)  # Pass your transformations if any

# # Initialize DataLoader
# data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# # Now you can iterate over the data_loader to get batches of images and labels
# for batch in data_loader:
#     images, labels = batch['image'], batch['label']
#     # Your training/validation loop or any other processing here
