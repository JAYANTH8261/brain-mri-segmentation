# preprocess/data_preprocessing.py

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def apply_clahe(images):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return [clahe.apply(img) for img in images]

def preprocess(images, masks):
    images = apply_clahe(images)
    images = np.array(images) / 255.0
    masks = np.array(masks) / 255.0
    return train_test_split(images, masks, test_size=0.2, random_state=42)
