```python
pip3 freeze > requirements.txt
```


```python
conda create -n lttenv
conda activate lttenv

conda install --yes --file requirements.txt

streamlit run testApp.py
```
