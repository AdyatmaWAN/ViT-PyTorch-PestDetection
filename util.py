import multiprocessing
import threading
import pandas as pd
import numpy as np
import os
import cv2

global pixel
pixel = 128

def preprocess():
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

    trainCSVFileOutput = folder + "train_"+str(pixel)+"_preprocessed.csv"

    imageTrainDirOutput = folder + "train_images_"+str(pixel)+"_preprocessed/"
    imageTestDirOutput = folder + "test_images_"+str(pixel)+"_preprocessed/"

    # Load images and find maximum dimensions
    pool = multiprocessing.Pool()
    results_train = pool.map(load_images_and_find_max_dimensions, [(train_df, imageTrainDir), (test_df, imageTestDir)])
    pool.close()
    pool.join()

    images_train = results_train[0]
    images_test = results_train[1]

    # Calculate mean and std
    mean, std = calculate_mean_std(images_train)

    # Perform image augmentation
    pool = multiprocessing.Pool()
    augmented_images_info_train = pool.starmap(augment_images, [(images_train, imageTrainDirOutput)])
    augmented_images_info_test = pool.starmap(augment_tests_images, [(images_test, imageTestDirOutput)])
    pool.close()
    pool.join()

    # Create a new DataFrame for augmented images info
    augmented_df_train = pd.DataFrame(augmented_images_info_train[0], columns=['filename', 'class'])
    augmented_df_train.to_csv(trainCSVFileOutput, index=False)

    # Create a new DataFrame for augmented images info
    print("Preprocessing done")
    print("Mean: ", mean)
    print("Std: ", std)
    return mean, std


# Function to calculate mean and std
def calculate_mean_std(images):
    # Initialize lists to store pixel values for each channel
    channel_means = []
    channel_stds = []

    # Iterate through each channel (BGR)
    for channel in range(3):
        channel_pixels = []

        # Extract pixel values for the current channel
        for image, _, _ in images:
            channel_pixels.append(image[:, :, channel].flatten())

        # Concatenate all pixel values for the current channel
        channel_pixels = np.concatenate(channel_pixels, axis=0)

        # Calculate mean and standard deviation for the current channel
        mean = np.mean(channel_pixels) / 255.0  # Normalize mean to [0, 1]
        std = np.std(channel_pixels) / 255.0  # Normalize std to [0, 1]

        # Append mean and std to the respective lists
        channel_means.append(mean)
        channel_stds.append(std)

    return channel_means, channel_stds


# Function to load images and find the maximum dimensions
def load_images_and_find_max_dimensions(args):
    df, image_dir = args
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

        # Resize the image to pixel x pixel while maintaining aspect ratio
        height, width, _ = image.shape
        if height == width:
            resized_image = cv2.resize(image, (pixel, pixel))
        elif height > width:
            new_height = pixel
            new_width = int(width * (pixel / height))
            resized_image = cv2.resize(image, (new_width, new_height))
            # Calculate padding for width and height
            pad_height = (pixel - new_height) // 2
            pad_width = pixel - new_width
            # Add padding around the image
            resized_image = cv2.copyMakeBorder(resized_image, pad_height, pixel - new_height - pad_height, pad_width // 2, pad_width - pad_width // 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:  # width > height
            new_width = pixel
            new_height = int(height * (pixel / width))
            resized_image = cv2.resize(image, (new_width, new_height))
            # Calculate padding for width and height
            pad_width = (pixel - new_width) // 2
            pad_height = pixel - new_height
            # Add padding around the image
            resized_image = cv2.copyMakeBorder(resized_image, pad_height // 2, pad_height - pad_height // 2, pad_width, pixel - new_width - pad_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # Get the class label
        label = row[class_column_name]

        images.append((resized_image, filename, label))

    return images


# Function to perform image augmentation
def augment_images(images, output_dir):
    augmented_images_info = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define a function for thread
    def process_image(image, filename, label):
        augmented_images_info.append((filename, label))
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, image)

        # for angle in [0]:
        #     rotated_image = image #cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        #     augmented_filename = f"{filename.split('.')[0]}_rotated_{angle}.{filename.split('.')[1]}"
        #     augmented_images_info.append((augmented_filename, label))
        #     output_path = os.path.join(output_dir, augmented_filename)
        #     cv2.imwrite(output_path, rotated_image)

    # Create threads
    threads = []
    for image, filename, label in images:
        thread = threading.Thread(target=process_image, args=(image, filename, label))
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    return augmented_images_info


# Function to perform image augmentation
def augment_tests_images(images, output_dir):
    augmented_images_info = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image, filename, label in images:
        padded_image = image

        # Save augmented image
        augmented_filename = filename # Add suffix to filename
        augmented_images_info.append((augmented_filename, label))

        output_path = os.path.join(output_dir, augmented_filename)
        cv2.imwrite(output_path, padded_image)

    return augmented_images_info


def encode_the_classes():
    folder = 'jute-pest-classification/'

    # Path to your CSV file
    trainCSVFile = folder + "train_"+str(pixel)+"_preprocessed.csv"

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(trainCSVFile)

    # Encoding classes
    class_encoding = {cls: idx for idx, cls in enumerate(df['class'].unique())}
    df['class'] = df['class'].map(class_encoding)  # Replace 'class' column with encoded values

    # Save encoded DataFrame to CSV
    df.to_csv(folder+"train_"+str(pixel)+"_preprocessed_encoded.csv", index=False)

    # Save transformation dictionary
    with open(folder+'transformation_dict.txt', 'w') as f:
        for cls, idx in class_encoding.items():
            f.write(f"{cls}: {idx}\n")


def decode_the_classes():
    # Load transformation dictionary
    folder = 'jute-pest-classification/'

    # Load submission.csv
    submission_df = pd.read_csv(folder+'submission.csv')

    # Load transformation dictionary
    class_decoding = {}
    with open(folder+'transformation_dict.txt', 'r') as f:
        for line in f:
            cls, idx = line.strip().split(': ')
            class_decoding[int(idx)] = cls

    # Decode class
    submission_df['class_decoded'] = submission_df['class_encoded'].map(class_decoding)

    # Save the decoded submission
    submission_df.to_csv(folder+'decoded_submission.csv', index=False)