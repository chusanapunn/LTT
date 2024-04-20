# Setup And Update
## Create environment and Install equirement
```python
conda create --name lttenv --file requirements.txt
conda activate lttenv

conda env create -n lttenv -f environment.yml --force
```

## Run test App
```python
streamlit run testApp.py
```

# Updating to git
Wokring in vscode, In source control -> 3dots (...)-> Checkout to "development" branch
git pull 


## If you add any library, don't forget to;
```python
# Create requirements file
conda list --explicit > requirements.txt
# Use above .yml file to update environnment
conda env update --file requirements.txt --prune # Update environment using the above env.yml file

```

# Task - Cars detection
## Find Different cars dataset, differs quality (road, showroom) - Put it on to DVC
## Different models (resnet, mobilenet, yolo etc) - implement each with mlflow
## Allow using different models in the streamlit UI, input images, output interface





