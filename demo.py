import streamlit as st
import numpy as np
import cv2
from keras.models import load_model


# Load the pre-trained model
@st.cache_resource
def load_trained_model():
    return load_model('age_detection_model.h5')


# Preprocess the uploaded image
def preprocess_image(uploaded_image):
    img = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128))  # Resize to match the model input size
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


def main():
    st.title("Age Detection Demo")
    st.write("Upload a face image to predict the age.")

    # Load the trained model
    model = load_trained_model()

    # Image uploader
    uploaded_image = st.file_uploader("Upload an image...", type=['jpg', 'png', 'jpeg'])

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        img = preprocess_image(uploaded_image)

        # Predict the age
        predicted_age = model.predict(img)
        st.subheader(f"Predicted Age: {int(predicted_age[0][0])}")


if __name__ == '__main__':
    main()
