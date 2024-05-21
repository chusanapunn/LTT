import streamlit as st
import cv2
import time
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
import torchvision

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
        faster_rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True, progress=False)
        faster_rcnn_model.eval()

        # resnet_model = faster_rcnn_model()
        # Initialize Faster R-CNN
        # Update ResNet model with new parameters if provided
        # if 'new_parameter' in kwargs:
        #     new_parameter_value = kwargs['new_parameter']
        #     # Set the new parameter in the ResNet model
        #     resnet_model.set_new_parameter(new_parameter_value)
        return faster_rcnn_model
    
    
    elif model_name == 'MobileNet':
        # Load ResNet model
        mobilenet = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True, progress=False)
        mobilenet.eval()

        # resnet_model = faster_rcnn_model()
        # Initialize Faster R-CNN
        # Update ResNet model with new parameters if provided
        # if 'new_parameter' in kwargs:
        #     new_parameter_value = kwargs['new_parameter']
        #     # Set the new parameter in the ResNet model
        #     resnet_model.set_new_parameter(new_parameter_value)
        return mobilenet
    
    elif model_name == 'Single Shot Multibox Detector (SSD)':
        # Load ResNet model
        ssd = torchvision.models.detection.ssd300_vgg16(pretrained=True, progress=False)
        ssd.eval()

        # resnet_model = faster_rcnn_model()
        # Initialize Faster R-CNN
        # Update ResNet model with new parameters if provided
        # if 'new_parameter' in kwargs:
        #     new_parameter_value = kwargs['new_parameter']
        #     # Set the new parameter in the ResNet model
        #     resnet_model.set_new_parameter(new_parameter_value)
        return ssd
    
    else:
        raise ValueError("Invalid model selection")



def process_plot(model, selected_model, cv_image, image):
    start_total = time.time()

    if selected_model == "ResNet" or selected_model == "MobileNet" or selected_model == "Single Shot Multibox Detector (SSD)":
        # Preprocess the image
        start_preprocess = time.time()
        transform = transforms.Compose([transforms.ToTensor()])
        cv_image_tensor = transform(cv_image).unsqueeze(0)
        preprocess_time = time.time() - start_preprocess

        # Perform inference
        start_inference = time.time()
        with torch.no_grad():
            processed_image = model(cv_image_tensor)
        inference_time = time.time() - start_inference

        # Post-process the output
        start_postprocess = time.time()
        cv_plot = np.array(image)
        car_boxes = []
        car_labels = []
        car_scores = []

        for box, label, score in zip(processed_image[0]['boxes'], processed_image[0]['labels'], processed_image[0]['scores']):
            if (label == 3 or label == 8) and score >= 0.6:  # Label 3 corresponds to 'car' in COCO dataset
                car_boxes.append(box.cpu().numpy())
                car_labels.append(label.cpu().numpy())
                car_scores.append(score.cpu().numpy())

        for box, label, score in zip(car_boxes, car_labels, car_scores):
            box = box.astype(int)
            color = (255, 0, 0)  # Red color
            cv2.rectangle(cv_plot, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(cv_plot, f"Car: {score:.2f}", (box[0], box[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv_plot = cv2.cvtColor(cv_plot, cv2.COLOR_BGR2RGB)
        postprocess_time = time.time() - start_postprocess

    elif selected_model == "YOLO":
        # Perform YOLO inference
        start_inference = time.time()
        processed_image = model.predict(cv_image)
        inference_time = time.time() - start_inference

        # Post-process the output
        cv_plot = processed_image[0].plot()
        cv_plot = cv2.cvtColor(cv_plot, cv2.COLOR_BGR2RGB)

        preprocess_time = 0  # No preprocessing for YOLO
        postprocess_time = 0  # No postprocessing for YOLO

    # Save the processed image
    cv2.imwrite("processed_image.png", cv_plot)

    # Display processing times
    total_time = time.time() - start_total
    processing_time = {
        "preprocess": preprocess_time,
        "inference": inference_time,
        "postprocess": postprocess_time
    }
    img_bytes = cv2.imencode(".png", cv_plot)[1].tobytes()
    st.image(img_bytes, caption="Processed Image", use_column_width=True)
    return cv_plot, processing_time



# Main Streamlit app
def main():
    st.title("Car Detection")

    # Sidebar for model selection and parameter tuning
    st.sidebar.title("Model Configuration")
    selected_model = st.sidebar.selectbox("Select Model", ["YOLO", "ResNet","MobileNet","Single Shot Multibox Detector (SSD)"])

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
        processed_image, processing_time = process_plot(model, selected_model, cv_image, image)

       

        # Display the processing time
        st.header("Time Processing")
        st.text("Preprocess: {:.1f} ms".format(processing_time["preprocess"] * 1000))
        st.text("Inference: {:.1f} ms".format(processing_time["inference"] * 1000))
        st.text("Postprocess: {:.1f} ms".format(processing_time["postprocess"] * 1000))



# Run the Streamlit app
if __name__ == "__main__":
    main()
