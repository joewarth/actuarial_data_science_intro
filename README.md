# Project Overview

This repository contains an end-to-end modeling workflow organized into two Jupyter notebooks. 

This is an overly simplistic demonstration that I use for brief presentations to audiences familiar with OLS regression, but that need to be introduced to other modeling techniques. The goal of this project is to provide a brief and transparent modeling workflow that clearly documents both *why* modeling decisions were made (EDA) and *how* final models were constructed and evaluated.

## Environment Setup

This project does not commit the virtual environment (`.venv/`) to version control. Recreate the Python environment from `requirements.txt` using python -m pip install -r requirements.txt

## Notebooks

### 01_data_eda.ipynb — Data Exploration & Preparation

This notebook focuses on understanding and preparing the data prior to modeling. Typical steps include:

- Data loading and initial validation
- Exploratory data analysis (distributions, relationships, data quality checks)
- Feature inspection and preliminary transformations
- Identification of modeling considerations (e.g., sparsity, nonlinearity, outliers)

Outputs from this notebook are intended to inform modeling decisions rather than finalize them.

### 02_modeling.ipynb — Modeling & Evaluation

This notebook builds on insights from the EDA phase to develop predictive models. It typically includes:

- Model specification and training
- Hyperparameter selection
- Model evaluation and diagnostics
- Comparison of alternative modeling approaches

Model results and key performance metrics are documented directly in the notebook.

## Usage Notes

- Notebooks are designed to be run sequentially (`01_data_eda.ipynb` → `02_modeling.ipynb`).
- Assumes required Python packages are installed in the active environment.
- Paths, parameters, and settings may be adjusted for experimentation or extension.