import os
import cv2
import pandas as pd

def load_images_and_find_max_dimensions(df, image_dir):
    df, image_dir = df, image_dir
    images = []

    # Check if 'class' column exists in the DataFrame
    if 'class' in df.columns:
        class_column_name = 'class'
    else:
        # If 'class' column is not found, assume it's the first column
        class_column_name = df.columns[0]

    for index, row in df.iterrows():
        filename = row['filename']
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)
        if image.shape[0] != 256 or image.shape[1] != 256:
            print(filename, image.shape)

        # Get the maximum pixel value
        max_pixel_value = image.max()

        # Check the range
        if max_pixel_value > 1:
            # print("Image range is 0-255 (8-bit format)")
            pass
        else:
            print(filename, max_pixel_value)

pixel = '256'

folder = 'jute-pest-classification/'

# Path to your CSV file
trainCSVFile = folder + "train.csv"
testCSVFile = folder + "test.csv"

# Path to the directory containing your images
imageTrainDir = folder + "train_images/"
imageTestDir = folder + "test_images/"

# Read the CSV file into a pandas DataFrame
train_df = pd.read_csv(trainCSVFile)
test_df = pd.read_csv(testCSVFile)

trainCSVFileOutput = folder + "train_"+pixel+"_preprocessed.csv"

imageTrainDirOutput = folder + "train_images_"+pixel+"_preprocessed/"
imageTestDirOutput = folder + "test_images_"+pixel+"_preprocessed/"

# Load images and find maximum dimensions
load_images_and_find_max_dimensions(train_df, imageTrainDirOutput)
load_images_and_find_max_dimensions(test_df, imageTestDirOutput)