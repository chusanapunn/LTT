# Setup And Update
## Create environment and Install equirement
```python
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
# The one add any library create env.yml file for others
conda env export > environment.yml # Create env file
# Use above .yml file to update environnment
conda env update --file environment.yml --prune # Update environment using the above env.yml file

```

# Task - Cars detection
## Find Different cars dataset, differs quality (road, showroom) - Put it on to DVC
## Different models (resnet, mobilenet, yolo etc) - implement each with mlflow
## Allow using different models in the streamlit UI, input images, output interface





