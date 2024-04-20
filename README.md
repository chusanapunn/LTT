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

## If you add any library, don't forget to;
```python
conda env export > environment.yml
```
don't forget to push it to repo



