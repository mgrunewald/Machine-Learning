# Dataset Description: Breast Cancer Diagnosis

## Overview

The Breast Cancer Diagnostic dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass, which are used to predict whether the mass is benign or malignant.

## Data Source

The data used in this analysis is inspired by the UCI Machine Learning Repository, available at [UCI Breast Cancer Wisconsin (Diagnostic) dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).

## Dataset Information

- **File Names**: breast_cancer_dev.csv, breast_cancer_test.csv
- **Purpose**: Dataset for machine learning analysis
- **Format**: Comma-separated values (CSV)
- **Variables**:
  
  - `mean concavity`: Mean severity of concave portions of the contour.
  - `concavity error`: Standard error of severity of concave portions of the contour.
  - `mean compactness`: Mean of perimeter^2 / area - 1.0.
  - `compactness error`: Standard error of perimeter^2 / area - 1.0.
  - `mean fractal dimension`: Mean of "coastline approximation" - 1.
  - `fractal dimension error`: Standard error for "coastline approximation" - 1.
  - `target`: The diagnosis of breast tissues (1 = malignant, 0 = benign). Target variable.
