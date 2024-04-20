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
# openAI secret API key
sk-proj-5J7w4jYq8jOwGZ9SBuCoT3BlbkFJGAvJfrqcBlFhOBqPDmrk

# Task
## 0. Create proper prompt set
## 1. DVC for prompt-answer storage
## 2. MLFLOW for monitoring/evaluation
## 3. LANGCHAIN LLM to talk with each other
## 4. Streamlit UI for Front-End


