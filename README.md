# DLMDSPWP01 – Programming with Python

## Project Overview

This project was developed as part of the IU International University course **DLMDSPWP01 – Programming with Python**.

The objective is to select the best fitting ideal functions for four training datasets using the Least Squares method, map test data points to the selected ideal functions according to the assignment rules, store the results in a SQLite database, and visualize the outcome using Bokeh.

---

## Technologies Used

* Python 3.11
* Pandas
* NumPy
* SQLAlchemy
* SQLite
* Bokeh
* Pytest

---

## Project Structure

```text
DLMDSPWP01/
│
├── data/
│   ├── train.csv
│   ├── ideal.csv
│   └── test.csv
│
├── output/
│   └── visualization.html
│
├── src/
│   ├── data_loader.py
│   ├── function_selector.py
│   ├── mapper.py
│   ├── database_manager.py
│   ├── visualization.py
│   └── exceptions.py
│
├── tests/
│   ├── test_function_selector.py
│   └── test_mapper.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Assignment Workflow

### Step 1 – Load Data

Training, ideal and test datasets are loaded into Pandas DataFrames.

### Step 2 – Select Ideal Functions

For each training function (y1–y4), the ideal function with the smallest Sum of Squared Errors (SSE) is selected.

Selected functions:

| Training Function | Ideal Function |
| ----------------- | -------------- |
| y1                | y13            |
| y2                | y24            |
| y3                | y36            |
| y4                | y40            |

### Step 3 – Calculate Allowed Deviations

The maximum deviation between each training function and its selected ideal function is calculated.

The assignment threshold is:

Allowed Deviation = √2 × Maximum Deviation

### Step 4 – Map Test Data

Each test point is assigned to the ideal function whose deviation is within the allowed threshold and is minimal among all candidates.

Mapped test points: 34

### Step 5 – Store Results

Data and mapping results are stored in a SQLite database using SQLAlchemy.

### Step 6 – Visualize Results

A Bokeh visualization is generated showing:

* Training functions
* Selected ideal functions
* Mapped test points

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running the Project

```bash
python main.py
```

---

## Running Tests

```bash
pytest
```

Expected result:

```text
2 passed
```

---

## Author

Mohamed Saad

IU International University

DLMDSPWP01 – Programming with Python
