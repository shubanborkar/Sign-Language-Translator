import os
import cv2
import numpy as np
import pickle
from glob import glob
from sklearn.model_selection import train_test_split

image_x, image_y = 50, 50  # Your model input shape

def preprocess_data():
    images = []
    labels = []

    gesture_folders = sorted(os.listdir("gestures"))
    for label in gesture_folders:
        folder_path = os.path.join("gestures", label)
        if not os.path.isdir(folder_path):
            continue
        for img_file in glob(os.path.join(folder_path, "*.jpg")):
            img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (image_x, image_y))
            images.append(img)
            labels.append(int(label))  # Convert folder name (e.g., "0") to label

    images = np.array(images)
    labels = np.array(labels)

    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42)

    with open("train_images", "wb") as f:
        pickle.dump(train_images, f)
    with open("train_labels", "wb") as f:
        pickle.dump(train_labels, f)
    with open("val_images", "wb") as f:
        pickle.dump(val_images, f)
    with open("val_labels", "wb") as f:
        pickle.dump(val_labels, f)

    print("✅ Preprocessing complete. Pickle files saved!")

if __name__ == "__main__":
    preprocess_data()
