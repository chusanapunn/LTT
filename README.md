# Setup And Update
## Create environment and Install equirement
```python
# 1st way
conda create --name lttenv --file requirements.txt
conda activate lttenv
# or
conda create -n lttenv
conda install --name lttenv --file requirements.txt
conda activate lttenv
# Main tools
# mlflow
# streamlit
# gcs
# mamba - install dvc
```

## Run test App
```python
streamlit run testApp.py
```

# Updating to git
Wokring in vscode, In source control -> 3dots (...)-> Checkout to "development" branch

## If you add any library, don't forget to;
```python
# Create requirements file
conda list --explicit > requirements.txt
# Use above .txt file to update environnment
conda env update --file requirements.txt --prune # Update environment using the above .txt file
```

# Task - Cars detection

## Find Different cars dataset, differs quality (road, showroom) - Put it on to DVC, GCS
### Dataset from kaggle
Use only "street_data" select only reasonable portion of it to work on
https://www.kaggle.com/datasets/mikhailma/house-rooms-streets-image-dataset

### GCS
https://cloud.google.com/sdk/docs/install
After installing Google cloud CLI
Authenticate yourself with gcloud *also give me your google cloud storage email for permission*
```bash
gcloud auth application-default login
```
Then run the "test_GCS_getdata.ipynb", I provide code to simple fetching the data out while setting environment variable for gcs.
### DVC
WIP -> use DVC to track GCS data version
If anyone wanna work on this --> https://dvc.org/doc/user-guide/data-management/remote-storage/google-cloud-storage 

## Different models (resnet, mobilenet, yolo etc) - implement each with mlflow
### Implement car detection model and metrics (recall, precision at k, AUC ROC etc.)

Evaluation Metrics: Precision, Recall, AUC ROC, F1 Score.

## Allow using different models in the streamlit UI, input images, output interface

##



