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
from logging import getLogger
import logging
import torchvision.transforms as transforms
import torchvision
import mlflow
import seaborn as sns
import mlflow.pytorch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
bucket_name = "mlops-car-detection"
os.environ.setdefault("GCLOUD_PROJECT", "mlops-cardetection")

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
        return yolo_model
    elif model_name == 'ResNet':
        # Load ResNet model
        faster_rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True, progress=False)
        faster_rcnn_model.eval()
        return faster_rcnn_model
    elif model_name == 'MobileNet':
        # Load MobileNet model
        mobilenet = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True, progress=False)
        mobilenet.eval()
        return mobilenet
    elif model_name == 'SSD':
        # Load SSD model
        ssd = torchvision.models.detection.ssd300_vgg16(pretrained=True, progress=False)
        ssd.eval()
        return ssd
    else:
        raise ValueError("Invalid model selection")

def process_plot(model, selected_model, cv_image, image):
    start_total = time.time()

    if selected_model in ["ResNet", "MobileNet", "SSD"]:
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
    
    elif selected_model in ["MobileNet"]:
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
            if (label == 3 or label == 8) and score >= 0.4:  # Label 3 corresponds to 'car' in COCO dataset
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

    elif selected_model in ["SSD"]:
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
            if (label == 3 or label == 8) and score >= 0.2:  # Label 3 corresponds to 'car' in COCO dataset
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
        start_postprocess = time.time()
        cv_plot = processed_image[0].plot()
        cv_plot = cv2.cvtColor(cv_plot, cv2.COLOR_BGR2RGB)
        postprocess_time = time.time() - start_postprocess

        preprocess_time = 0  # YOLO does not require preprocessing for this implementation

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

def plot_metrics_from_mlflow():
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("car_detection_experiment")
    
    if experiment:
        experiment_id = experiment.experiment_id
        runs = client.search_runs(experiment_id)

        if not runs:
            st.write("No model selected yet")
            return

        preprocess_times = []
        inference_times = []
        postprocess_times = []
        model_names = []

        for run in runs:
            model_name = run.data.params.get("selected_model", "Unknown Model")
            run_id_short = run.info.run_id[:8]  # Shortened run ID
            model_names.append(f"{model_name} ({run_id_short})")
            preprocess_times.append(run.data.metrics.get("preprocess_time_ms", 0))
            inference_times.append(run.data.metrics.get("inference_time_ms", 0))
            postprocess_times.append(run.data.metrics.get("postprocess_time_ms", 0))

        # Create a DataFrame for easy plotting
        df = pd.DataFrame({
            'Model (Run ID)': model_names,
            'Preprocess Time (ms)': preprocess_times,
            'Inference Time (ms)': inference_times,
            'Postprocess Time (ms)': postprocess_times
        })
        df.set_index('Model (Run ID)', inplace=True)

        # Select a subset of the data if there are more than 5 runs
        if len(df) > 5:
            df = df.iloc[:5]

        # Set the Seaborn style
        sns.set(style="whitegrid")

        # Plot bar charts
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))

        sns.lineplot(data=df[['Preprocess Time (ms)']], ax=axes[0], palette="Blues_d", marker="o")
        axes[0].set_xlabel('Model (Run ID)')
        axes[0].set_ylabel('Preprocess Time (ms)')
        axes[0].set_title('Preprocess Time by Model and Run ID')

        sns.lineplot(data=df[['Inference Time (ms)']], ax=axes[1], palette="Greens_d", marker="o")
        axes[1].set_xlabel('Model (Run ID)')
        axes[1].set_ylabel('Inference Time (ms)')
        axes[1].set_title('Inference Time by Model and Run ID')

        sns.lineplot(data=df[['Postprocess Time (ms)']], ax=axes[2], palette="Reds_d", marker="o")
        axes[2].set_xlabel('Model (Run ID)')
        axes[2].set_ylabel('Postprocess Time (ms)')
        axes[2].set_title('Postprocess Time by Model and Run ID')

        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.write("No experiment found with the name 'car_detection_experiment'")



# Main Streamlit app
def main():
    st.title("Car Detection")

    # Sidebar for model selection and parameter tuning
    st.sidebar.title("Model Configuration")
    selected_model = st.sidebar.selectbox("Select Model", ["YOLO", "ResNet","MobileNet","SSD"])

    # Main content area
    st.sidebar.markdown("---")

    # Upload image file
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Initialize MLFlow
        mlflow.set_tracking_uri("http://127.0.0.1:8080")  # Set your MLFlow tracking URI
        mlflow.set_experiment("car_detection_experiment")  # Set the name of your MLFlow experiment

        with mlflow.start_run():
            mlflow.log_param("selected_model", selected_model)

            # Load the selected model
            model = load_model(selected_model)

            # Display the uploaded image
            st.header("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)

            # Log image artifact
            image_path = "uploaded_image.png"
            image.save(image_path)
            mlflow.log_artifact(image_path)

            # Convert PIL image to OpenCV format
            cv_image = np.array(image)

            # Process the image
            processed_image, processing_time = process_plot(model, selected_model, cv_image, image)

            # Log processed image with bounding boxes as artifact
            mlflow.log_artifact("processed_image.png")

            # Log processing times as metrics
            mlflow.log_metric("preprocess_time_ms", processing_time["preprocess"] * 1000)
            mlflow.log_metric("inference_time_ms", processing_time["inference"] * 1000)
            mlflow.log_metric("postprocess_time_ms", processing_time["postprocess"] * 1000)

          

            # Display the processing time
            st.header("Time Processing")
            st.text("Preprocess: {:.1f} ms".format(processing_time["preprocess"] * 1000))
            st.text("Inference: {:.1f} ms".format(processing_time["inference"] * 1000))
            st.text("Postprocess: {:.1f} ms".format(processing_time["postprocess"] * 1000))

    # Plot metrics from MLflow
    st.header("Metrics from MLflow")
    plot_metrics_from_mlflow()

# Run the Streamlit app
if __name__ == "__main__":
    main()
