import streamlit as st
import cv2
import numpy as np
from google.cloud import storage
from PIL import Image
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing.image import ImageDataGenerator
import mlflow

from ultralytics import YOLO

bucket_name = "mlops-car-detection"
os.environ.setdefault("GCLOUD_PROJECT","mlops-cardetection")

# Initialize the storage client
storage_client = storage.Client()



def load_model(model_name, **kwargs):
    if model_name == 'ResNet':
        # Load ResNet model
        resnet_model = 1
        # Update ResNet model with new parameters if provided
        if 'new_parameter' in kwargs:
            new_parameter_value = kwargs['new_parameter']
            # Set the new parameter in the ResNet model
            resnet_model.set_new_parameter(new_parameter_value)
        return resnet_model
    elif model_name == 'YOLO':
        # Load YOLO model
        yolo_model = YOLO("yolov8n.pt")
        # Update YOLO model with new parameters if provided
        if 'new_parameter' in kwargs:
            new_parameter_value = kwargs['new_parameter']
            # Set the new parameter in the YOLO model
            yolo_model.set_new_parameter(new_parameter_value)
        return yolo_model
    else:
        raise ValueError("Invalid model selection")

# Main Streamlit app
def main():
    st.title("Car Detection")

    # Model selection
    selected_model = st.selectbox("Select Model", ["YOLO","ResNet"])

    # Parameter tuning
    if selected_model == "ResNet":
        # Add widgets for tuning ResNet parameters
        learning_rate = st.slider("Learning Rate", 0.001, 0.01, 0.001, 0.001)
        num_epochs = st.slider("Number of Epochs", 1, 20, 10, 1)
    elif selected_model == "YOLO":
    # Add widgets for tuning YOLO parameters
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
        input_size_options = [(320, 320), (416, 416), (608, 608)]  # List of tuples
        input_size = st.select_slider("Input Size", options=input_size_options)

    # Upload image file
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Load the selected model
        model = load_model(selected_model)

        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert PIL image to OpenCV format
        cv_image = np.array(image)

        # Process the image
        processed_image = model.predict(cv_image)

        # Display the processed image
        st.image(processed_image[0].orig_img, caption="Processed Image", use_column_width=True)
        st.text_area(label="results",value=processed_image[0].names)
# Run the Streamlit app
if __name__ == "__main__":
    main()
