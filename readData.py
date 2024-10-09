import os
import cv2
import numpy as np

def load_kaggle_dataset(data_dir):
    images = []
    labels = []
    
    # Load images and their labels
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        for filename in os.listdir(folder_path):
            if filename.endswith('.png'):
                # Read image
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                # Resize image (e.g., 128x32)
                image = cv2.resize(image, (128, 32))
                images.append(image)

                # Assume the label is part of the filename (e.g., label.png)
                label = filename.split('.')[0]  # Adjust based on your dataset format
                labels.append(label)
    
    # Normalize images
    images = np.array(images) / 255.0
    labels = np.array(labels)
    
    return images, labels

# Usage
data_dir = 'path/to/handwritten_text_recognition/train/'
images, labels = load_kaggle_dataset(data_dir)