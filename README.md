# zeolite-co2-adsorption-ml-predictive-modeling
Machine learning analysis of CO₂ adsorption capacity in Na-ZSM-5 zeolites using 12 regression algorithms to predict adsorption performance based on Si/Al ratio, temperature, and pressure.

## Overview

This research project evaluates 12 machine learning algorithms for predicting CO₂ adsorption capacity in zeolites. The study provides insights into the relationship between zeolite properties (Si/Al ratio), experimental conditions (temperature, pressure), and adsorption performance.

## Dataset

The dataset consists of 8 experimental samples with the following features:

| Feature | Description | Range |
|---------|-------------|-------|
| Zeolite_Type | Type of Na-ZSM zeolite | Na-ZSM-5 to Na-ZSM-200 |
| Si_Al_Ratio | Silicon to Aluminum ratio | 5.0 - 200.0 |
| Temperature | Experimental temperature | 6°C - 26°C |
| Pressure_kPa | Operating pressure | 81.05 - 90.67 kPa |
| Adsorption_Capacity | Target variable (mmol/g) | 1.64 - 4.49 mmol/g |

## Features

- **12 Regression Algorithms**: Comprehensive model comparison
- **Multiple Metrics**: R², MAE, MSE, RMSE with cross-validation
- **Research-Ready Visualizations**: Publication-quality graphs
- **Feature Importance Analysis**: Identification of key predictors
- **Statistical Correlation**: Understanding variable relationships
