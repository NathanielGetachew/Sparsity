import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_data(data_dir, image_size=(100, 100), images_per_category=1000):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    # Define categories as key-value pairs
    CATEGORIES = {'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5}

    print(f"Targeting {images_per_category} images per category, totaling {images_per_category * len(CATEGORIES)} images.")

    # Load images from the training set
    print("Checking train folder structure and image count...")
    for category, label in CATEGORIES.items():
        category_path = os.path.join(data_dir, 'train', category)
        print(f"Looking in {category_path}...")
        if os.path.exists(category_path):
            all_images = os.listdir(category_path)
            image_count = len(all_images)
            print(f"✔️ Found {image_count} images in {category_path}")
            
            # Load up to the required images_per_category
            for filename in all_images[:images_per_category]:
                img_path = os.path.join(category_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, image_size)
                    train_images.append(img)
                    train_labels.append(label)
        else:
            print(f"❌ Category folder not found: {category_path}")

    # Load images from the test set
    print("Checking test folder structure and image count...")
    for category, label in CATEGORIES.items():
        category_path = os.path.join(data_dir, 'test', category)
        print(f"Looking in {category_path}...")
        if os.path.exists(category_path):
            all_images = os.listdir(category_path)
            image_count = len(all_images)
            print(f"✔️ Found {image_count} images in {category_path}")
            
            # Load up to the required images_per_category
            for filename in all_images[:images_per_category]:
                img_path = os.path.join(category_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, image_size)
                    test_images.append(img)
                    test_labels.append(label)
        else:
            print(f"❌ Category folder not found: {category_path}")

    # Convert to NumPy arrays
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    print(f"Loaded {len(train_images)} training images and {len(test_images)} test images.")

    # Split the training data into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, random_state=42)

    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)
