import streamlit as st
import cv2
import numpy as np
from PIL import Image
from google.cloud import storage
import os
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn

bucket_name = "mlops-car-detection"
os.environ.setdefault("GCLOUD_PROJECT","mlops-cardetection")

# Initialize the storage client
storage_client = storage.Client()

# Function to load the selected model
def load_model(model_name, **kwargs):
    if model_name == 'ResNet':
        # Load ResNet model
        resnet_model = fasterrcnn_resnet50_fpn(pretrained=True)
        # Initialize Faster R-CNN
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

    # Sidebar for model selection and parameter tuning
    st.sidebar.title("Model Configuration")
    selected_model = st.sidebar.selectbox("Select Model", ["YOLO", "ResNet"])

    # Parameter tuning
    if selected_model == "ResNet":
        st.sidebar.header("ResNet Parameters")
        learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.01, 0.001, 0.001)
        num_epochs = st.sidebar.slider("Number of Epochs", 1, 20, 10, 1)
    elif selected_model == "YOLO":
        st.sidebar.header("YOLO Parameters")
        confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
        input_size_options = [(320, 320), (416, 416), (608, 608)]
        input_size = st.sidebar.select_slider("Input Size", options=input_size_options)

    # Main content area
    st.sidebar.markdown("---")

    # Upload image file
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Load the selected model
        model = load_model(selected_model)

        # Display the uploaded image
        st.header("Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)

        # Convert PIL image to OpenCV format
        cv_image = np.array(image)

        # Process the image
        processed_image = model.predict(cv_image)

        # Plot the results and convert to RGB
        cv_plot = processed_image[0].plot()
        cv_plot = cv2.cvtColor(cv_plot, cv2.COLOR_BGR2RGB)

        # Convert the image to bytes
        img_bytes = cv2.imencode(".png", cv_plot)[1].tobytes()

        # Display the processed image with bounding boxes
        st.header("Processed Image with Bounding Boxes")
        st.image(img_bytes, caption="Processed Image", use_column_width=True)

        # Display the results
        st.header("Time Processing")
        st.text("Preprocess")
        st.caption(value=processed_image[0].speed['preprocess'])
        # st.text("Time takes for DL to apply Trained NN to new data")
        st.text("Inference")
        st.caption(value=processed_image[0].speed['inference'])
        # st.text("Time takes for DL to apply Trained NN to new data")
        st.text("Postprocess")
        st.caption(value=processed_image[0].speed['postprocess'])

# Run the Streamlit app
if __name__ == "__main__":
    main()
