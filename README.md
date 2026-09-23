# DLMDSPWP01 – Programming with Python

Python implementation for the official IU written assignment **DLMDSPWP01 – Programming with Python**.

## What the program does

The program:

1. loads and validates the four training functions and 50 ideal functions;
2. creates a SQLite database through SQLAlchemy and stores the training and ideal tables;
3. selects one ideal function for each training function by minimum sum of squared errors (SSE);
4. calculates the assignment threshold `sqrt(2) * maximum training-to-ideal deviation`;
5. reads the test CSV **line-by-line**, maps each qualifying point, and stores `x`, `y`, `delta_y`, and the selected ideal-function number;
6. creates Bokeh visualizations for the complete test data, the four selected training/ideal pairs, mapped observations, and relative mapping deviations;
7. provides unit tests for data loading/validation, least-squares selection, threshold calculation, mapping, database persistence, and visualization.

If more than one selected ideal function satisfies the mapping threshold, the qualifying function with the **smallest absolute y-deviation** is selected. This rule makes the mapping deterministic.

## Expected results for the supplied personal dataset

| Training function | Selected ideal | Minimum SSE |
|---|---:|---:|
| y1 | y13 | 34.080707581466 |
| y2 | y24 | 33.451760953116 |
| y3 | y36 | 35.572700395769 |
| y4 | y40 | 34.998874813202 |

Expected number of mapped test observations: **34**.

## Project structure

```text
DLMDSPWP01/
├── data/
│   └── raw/
│       ├── train.csv        # supplied separately by IU
│       ├── ideal.csv        # supplied separately by IU
│       └── test.csv         # supplied separately by IU
├── docs/
│   ├── class_diagram.png
│   ├── class_diagram.dot
│   ├── workflow.png
│   └── workflow.dot
├── output/
│   ├── results.db           # generated
│   └── visualization.html   # generated
├── src/
│   ├── base_processor.py
│   ├── data_loader.py
│   ├── data_validator.py
│   ├── database_manager.py
│   ├── deviation_calculator.py
│   ├── exceptions.py
│   ├── function_selector.py
│   ├── mapper.py
│   └── plot_generator.py
├── tests/
├── main.py
├── pytest.ini
└── requirements.txt
```

The personal IU dataset is intentionally not committed to the repository. Place the three supplied CSV files in `data/raw/` using the exact names shown above.

## Installation

Python 3.11 or a compatible newer Python 3 release is recommended.

```bash
python -m venv .venv
```

Activate the environment and install only the direct project dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Generated files:

- `output/results.db`
- `output/visualization.html`

The SQLite database contains exactly the three assignment tables:

- `training_data` – `x, y1, y2, y3, y4`
- `ideal_functions` – `x, y1, ..., y50`
- `mapped_results` – `x, y, delta_y, ideal_function`

## Tests

```bash
pytest -q
```

The audited test suite contains **24 tests** and uses synthetic fixtures, so it can run without publishing the personal assignment dataset.
