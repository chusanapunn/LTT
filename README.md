## Create environment and Install equirement
```python
conda create -n lttenv
conda activate lttenv

conda install --yes --file requirements.txt

streamlit run testApp.py
```

## If you add any library, don't forget to;
```python
pip3 freeze > requirements.txt
```



