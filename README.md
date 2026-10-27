# Can Historical Earnings Patterns Predict Future Earnings Surprises?

A data mining research project investigating whether a company's historical earnings performance can predict its next earnings surprise.

## Overview

This project examines whether information from a company's four preceding recorded earnings events can predict whether its next reported earnings per share (EPS) will exceed analyst expectations.

An earnings beat occurs when actual EPS exceeds estimated EPS. Outcomes equal to or below the estimate are classified as 'meet or miss'.

The study uses the publicly available [Sov.ai Earnings Surprise dataset](https://huggingface.co/datasets/sovai/earnings_surprise).

## Research Objectives

- Evaluate whether historical earnings patterns provide useful predictive information.
- Compare simple baseline predictors with machine learning classification models.
- Investigate the effects of feature selection, historical information and model complexity.
- Analyze clustering patterns, unusual observations and model performance across time.
- Develop a reproducible experimental pipeline with documented methodology and results.

## Planned Methodology

The project will investigate:

- **Classification:** Logistic Regression, Decision Tree, Random Forest, XGBoost.
- **Clustering:** K-Means, Agglomerative Clustering, DBSCAN.
- **Outlier Detection:** Isolation Forest, Local Outlier Factor,  IQR-based detection.
- **Feature Selection:** Mutual information.
- **Evaluation:** Chronological validation, baseline comparisons, classification metrics, and feature ablation experiments.

All historical input availability assumptions will be audited and documented prior to the final modeling dataset is established.

## Dataset

**Source:** Sov.ai Earnings Surprise (Hugging Face)

At present, I have identified approximately 1.66 million source snapshots and 90,314 potentially eligible prediction examples.

These numbers are preliminary and will be verified during data acquisition and preprocessing.

## Technologies

Python, pandas, NumPy, scikit-learn, XGBoost, Jupyter Notebook, Git.

## Author

Jordan Archer