import streamlit as st
import cv2
import numpy as np
from PIL import Image
from google.cloud import storage
import os
from ultralytics import YOLO
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from logging  import getLogger
import logging
import torchvision.transforms as transforms

bucket_name = "mlops-car-detection"
os.environ.setdefault("GCLOUD_PROJECT","mlops-cardetection")

# Initialize the storage client
storage_client = storage.Client()

app_logger = getLogger()
app_logger.addHandler(logging.StreamHandler())
app_logger.setLevel(logging.INFO)


# Function to load the selected model
def load_model(model_name, **kwargs):

    if model_name == 'YOLO':
        # Load YOLO model
        yolo_model = YOLO("yolov8n.pt")
        # Update YOLO model with new parameters if provided
        # if 'new_parameter' in kwargs:
        #     new_parameter_value = kwargs['new_parameter']
        #     # Set the new parameter in the YOLO model
        #     yolo_model.set_new_parameter(new_parameter_value)
        return yolo_model
    
    elif model_name == 'ResNet':
        # Load ResNet model
        faster_rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True)
        faster_rcnn_model.eval()

        # resnet_model = faster_rcnn_model()
        # Initialize Faster R-CNN
        # Update ResNet model with new parameters if provided
        # if 'new_parameter' in kwargs:
        #     new_parameter_value = kwargs['new_parameter']
        #     # Set the new parameter in the ResNet model
        #     resnet_model.set_new_parameter(new_parameter_value)
        return faster_rcnn_model
    
    else:
        raise ValueError("Invalid model selection")

def process_plot(model,selected_model,cv_image,image):
    if selected_model == "ResNet":
        transform = transforms.Compose([transforms.ToTensor()])
        cv_image = transform(cv_image).unsqueeze(0)
        cv_plot = np.array(image)

        with torch.no_grad():
            processed_image = model(cv_image)
        boxes = processed_image[0]['boxes'].cpu().numpy()
        labels = processed_image[0]['labels'].cpu().numpy()
        scores = processed_image[0]['scores'].cpu().numpy()
        
        for box, label, score in zip(boxes, labels, scores):
            box = box.astype(int)
            color = (255, 0, 0)  # Red color
            cv2.rectangle(cv_plot, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(cv_plot, f"{label}: {score:.2f}", (box[0], box[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # Plot the results and convert to RGB
        cv_plot = cv2.cvtColor(cv_plot, cv2.COLOR_BGR2RGB)

        # processed_image = model(cv_image)
    elif selected_model == "YOLO":
        processed_image = model.predict(cv_image)
        cv_plot = processed_image[0].plot()
        cv_plot = cv2.cvtColor(cv_plot, cv2.COLOR_BGR2RGB)

        # Convert the image to bytes
    img_bytes = cv2.imencode(".png", cv_plot)[1].tobytes()
    st.image(img_bytes, caption="Processed Image", use_column_width=True)
    return processed_image

# Main Streamlit app
def main():
    st.title("Car Detection")

    # Sidebar for model selection and parameter tuning
    st.sidebar.title("Model Configuration")
    selected_model = st.sidebar.selectbox("Select Model", ["YOLO", "ResNet"])

    # # Parameter tuning
    # if selected_model == "ResNet":
    #     st.sidebar.header("ResNet Parameters")
    #     learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.01, 0.001, 0.001)
    #     num_epochs = st.sidebar.slider("Number of Epochs", 1, 20, 10, 1)
    # elif selected_model == "YOLO":
    #     st.sidebar.header("YOLO Parameters")
    #     confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
    #     input_size_options = [(320, 320), (416, 416), (608, 608)]
    #     input_size = st.sidebar.select_slider("Input Size", options=input_size_options)

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
        processed_image = process_plot(model,selected_model,cv_image,image)

        # Display the processed image with bounding boxes
        st.header("Processed Image with Bounding Boxes")
        # Display the results
        st.header("Time Processing")
        st.text("Preprocess")
        st.caption(processed_image[0].speed['preprocess'])
        # st.text("Time takes for DL to apply Trained NN to new data")
        st.text("Inference")
        st.caption(processed_image[0].speed['inference'])
        # st.text("Time takes for DL to apply Trained NN to new data")
        st.text("Postprocess")
        st.caption(processed_image[0].speed['postprocess'])

# Run the Streamlit app
if __name__ == "__main__":
    main()
