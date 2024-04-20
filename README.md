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
# apache-airflow[s3, postgres]
# streamlit
# dvc
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
## Find Different cars dataset, differs quality (road, showroom) - Put it on to DVC
## Different models (resnet, mobilenet, yolo etc) - implement each with mlflow
## Allow using different models in the streamlit UI, input images, output interface





