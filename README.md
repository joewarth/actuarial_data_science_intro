# Actuarial Data Science Presentation

## Overview

This project is an educational demonstration of insurance pure-premium modeling using the public French Motor Third-Party Liability Claims dataset. The workflow prepares policy-level data, explores pure-premium patterns, and compares three models:

- Ordinary least squares regression
- A Tweedie generalized linear model
- An XGBoost Tweedie gradient-boosting model

The primary workflow is in [modeling.ipynb](modeling.ipynb). Supporting data preparation, plotting, model-fitting, and evaluation functions are in [modeling_utils.py](modeling_utils.py).

## Setup

Create and activate a virtual environment from the project directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On macOS or Linux, activate it with:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Open [modeling.ipynb](modeling.ipynb) in VS Code or Jupyter and run the cells in order. The workflow reads cached parquet files from [data](data) when available. If the raw parquet files are unavailable, it downloads the frequency and severity datasets from OpenML.

The XGBoost tuning configuration requests a CUDA device. A CUDA-capable XGBoost installation and compatible GPU are required for that configuration. For CPU-only environments, change the XGBoost `device` setting in [modeling_utils.py](modeling_utils.py) to `cpu`.

Export the notebook to HTML with:

```powershell
python -m jupyter nbconvert --to html modeling.ipynb
```

This creates `modeling.html` in the project directory.

## Project Notes

- The project is an educational example, not a production-ready insurance pricing system.
- The modeling target is capped pure premium with individual claims capped at $1 million.
- The train/test split is policy-level and approximately matched on portfolio claim frequency.
- XGBoost tuning uses cross-validation; the held-out test data is reserved for final comparison.
- The workflow does not provide production deployment, regulatory filing support, rate certification, calibration analysis, fairness assessment, or model monitoring.
- Results depend on the package versions, random seeds, hardware, and available cached data.
- The source dataset is publicly available through OpenML datasets 41214 and 41215.
