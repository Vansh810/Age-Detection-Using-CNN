import numpy as np
import pandas as pd
import cv2
import os
from matplotlib import pyplot as plt
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
    img = cv2.resize(img, (128, 128))
    images.append(img)
    ages.append(get_age(path))

images = np.array(images)
ages = np.array(ages)

# Normalising Image Pixel Values between 0 and 1
images = images / 255

X_train, X_test, y_train, y_test = train_test_split(images, ages, test_size=0.2)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1)  # Single output for age regression
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
datagen.fit(X_train)

history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=1, validation_data=(X_test, y_test))

loss, mae = model.evaluate(X_test, y_test)
print(f'Mean Absolute Error: {mae}')

plt.plot(history.history['mae'], label='MAE (training)')
plt.plot(history.history['val_mae'], label='MAE (validation)')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()

model.save('age_detection_model.h5')
