# Data Science Course — IBA

Course material for the Data Science course at IBA — lectures, assignments, and projects using Python.

## Contents
- Lecture notebooks
- Lab assignments
- Capstone project

## Requirements
```
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook
```

## Topics
- Data wrangling with Pandas
- Visualization with Matplotlib/Seaborn
- Machine learning with scikit-learn

## License
Academic use only
<!-- updated: 2025-11-23-r01 -->

## Description
This repository contains the teaching materials and practical projects for the Data Science course at IBA. It includes Jupyter notebooks for lectures, lab assignments, and several Python scripts demonstrating data analysis, visualization, and simple machine‑learning applications such as stock price prediction and a mental‑health sentiment classifier.

## Features
- **Stock price analysis and prediction** using Yahoo Finance data, moving averages, Random Forest regression, and Streamlit UI (`project/testing/streamlit.py`).
- **Sentiment classification app** for mental‑health related text using TF‑IDF and Logistic Regression, also presented with Streamlit (`project/chatbot.py`).
- **Technical indicator calculations** (SMA, RSI) and time‑series preparation for modeling (`project/FinalProject.py`).
- Jupyter notebooks for course lectures and lab exercises (located in the `practice` directory).

## Installation
The required Python packages are listed in the README under *Requirements*. Install them with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter yfinance streamlit
```

## Usage
- Run the stock price prediction app:
```bash
streamlit run project/testing/streamlit.py
```
- Run the mental‑health sentiment app:
```bash
streamlit run project/chatbot.py
```
- Execute the final project script to generate plots and indicators:
```bash
python project/FinalProject.py
```

## Project Structure
```
.gitignore
LICENSE
README.md
classes/
    streamlit.py
practice/
    NoteBooks_Codes/
        example.txt
project/
    FinalProject.py
    chatbot.py
    info.txt
    testing/
        str.py
        streamlit.py
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
