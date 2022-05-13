# Failure_Analyzer
Data_engineering_from_Streamlit

## Streamlit App: Failure Analyzer

[Try it out!](https://share.streamlit.io/dynamichoos/Failure_Analyzer/main/streamlit_test.py)

> Execute Result_analyzer!

Based on the streamlit framework, visualization and tree analysis is implimented.

## Requirements

The app is developed in Python 3.7x

```bash
pip install -r requirements.txt
```

## How to use app

### 1. Download stock information data

- Collecting Failure Analysis list.
- The following script will store the listed stock information into `./resource` directory (TBD)
	```bash
	python prepare_data.py
	```

### 2. Run app

```bash
streamlit run streamlit_test.py
```
