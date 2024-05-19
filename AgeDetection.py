import numpy as np
import pandas as pd
import cv2
import os
from glob import glob
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import  Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

data_path = "UTKFace"
all_images = glob(os.path.join(data_path, "*.jpg"))


def get_age(file_path):
    file_name = os.path.basename(file_path)
    age = int(file_name.split('_')[0])
    return age


images = []
ages = []

for path in all_images:
    img = cv2.imread(path)
    if img.size != (200, 200):
        img = cv2.resize(img, (200, 200))
    images.append(img)
    ages.append(get_age(path))

images = np.array(images)
ages = np.array(ages)

# Normalising Image Pixel Values between 0 and 1
images = images / 255

X_train, X_test, y_train, y_test = train_test_split(images, ages, test_size=0.2)

