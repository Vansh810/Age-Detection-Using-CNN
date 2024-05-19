import numpy as np
import cv2
import os
import pickle
from matplotlib import pyplot as plt
from glob import glob
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

data_path = "UTKFace"
all_images = glob(os.path.join(data_path, "*.jpg"))


def get_age(file_path):
    file_name = os.path.basename(file_path)
    age = int(file_name.split('_')[0])
    return age


def preprocess():
    try:
        images = np.load('images.npy')
        ages = np.load('ages.npy')
        print("Loaded Preprocessed files")

    except FileNotFoundError:
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
        np.save('images.npy', images)
        np.save('ages.npy', ages)

        print("Preprocessing Done")

    return train_test_split(images, ages, test_size=0.2)


def train_model(X_train, X_test, y_train, y_test):
    try:
        model = load_model('age_detection_model.h5')
        with open('training_history.pkl', 'rb') as f:
            history = pickle.load(f)
        print("Loaded saved model")

    except (OSError, FileNotFoundError):
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

        history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))
        with open('training_history.pkl', 'wb') as f:
            pickle.dump(history.history, f)

        model.save('age_detection_model.h5')
        print("Model Trained")

    return model, history


def evaluate(X_test, y_test):
    loss, mae = model.evaluate(X_test, y_test)
    print(f'Mean Absolute Error: {mae}')


def plot_history(history):
    plt.plot(history.history['mae'], label='MAE (training)')
    plt.plot(history.history['val_mae'], label='MAE (validation)')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.show()


def predict_age(image_path, model):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    predicted_age = model.predict(img)
    return predicted_age[0][0]


X_train, X_test, y_train, y_test = preprocess()
model, history = train_model(X_train, X_test, y_train, y_test)
evaluate(X_test, y_test)
plot_history(history)

test_image_path = 'test25.jpg'
predicted_age = predict_age(test_image_path, model)
print(f'Predicted Age: {predicted_age}')
