# Setup And Update
## Create environment and Install equirement
```python
conda create -n lttenv
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
conda env export > environment.yml # Create env file
conda env update --file environment.yml --prune

```
# Task
## LLM to talk with each other


