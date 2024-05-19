# Age Detection Program using Convolutional Neural Networks

This program utilizes Convolutional Neural Networks (CNNs) implemented in Keras to predict the age of individuals from facial images. It preprocesses the images, trains the model, and evaluates its performance.

## Overview


1. **Preprocessing**:
   - Reads facial images from the dataset located in the "UTKFace" directory.
   - Resizes the images to (128, 128) pixels.
   - Normalizes the pixel values between 0 and 1.
   - Saves preprocessed images and corresponding ages into NumPy arrays (`images.npy` and `ages.npy`) to avoid repeated preprocessing.

2. **Model Training**:
   - Defines a CNN model architecture using Keras Sequential API.
   - Compiles the model with Adam optimizer and mean squared error loss for age regression.
   - Applies data augmentation using `ImageDataGenerator`.
   - Trains the model on the training data with 25 epochs.
   - Saves the trained model as `age_detection_model.h5` and training history as `training_history.pkl`.

3. **Evaluation**:
   - Evaluates the trained model on the test data.
   - Computes Mean Absolute Error (MAE) as the evaluation metric.

4. **Visualization**:
   - Plots the training and validation MAE over epochs to visualize model performance.

5. **Prediction**:
   - Predicts the age of an individual from a test image specified by the user (`test_image_path`).
   - Displays the predicted age.
