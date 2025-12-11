# Customer Churn Prediction Analysis

## Overview

This project focuses on predicting customer churn using multiple linear regression analysis. The analysis examines the relationship between customer churn and key independent variables including monthly charges, bandwidth usage, and customer tenure.

## Dataset

The project uses a customer churn dataset (`churn_clean.csv`) containing customer information and churn status. The dataset includes the following key variables:

- **Churn**: Binary target variable (Yes/No) indicating whether a customer churned
- **MonthlyCharge**: Monthly charges for the customer
- **Bandwidth_GB_Year**: Annual bandwidth usage in GB
- **Tenure**: Customer tenure (duration of service)

## Project Structure

The analysis follows a structured workflow:

1. **Data Loading**: Import and initial exploration of the dataset
2. **Data Quality Assessment**: Analysis of missing values and data types
3. **Statistical Summary**: Descriptive statistics for key variables
4. **Exploratory Data Analysis**: Visual analysis of variable distributions and relationships
5. **Data Preprocessing**: Data transformation and preparation for modeling
6. **Model Development**: Multiple linear regression model construction
7. **Model Evaluation**: Assessment of model performance and diagnostics

## Key Features

### Data Preprocessing

- Conversion of churn status from categorical (Yes/No) to binary (1/0)
- Skewness analysis and log transformation applied to highly skewed variables
- Data normalization to improve model assumptions

### Exploratory Data Analysis

The analysis includes comprehensive visualizations:

- Univariate distributions for churn, monthly charges, bandwidth usage, and tenure
- Bivariate analysis comparing churn status with key predictors
- Box plots to examine relationships between churn and continuous variables

### Modeling Approach

- **Model Type**: Multiple Linear Regression
- **Predictors**: MonthlyCharge, Bandwidth_GB_Year, Tenure
- **Target Variable**: Churn (binary)
- **Train-Test Split**: 80% training, 20% testing
- **Evaluation Metrics**: R-squared, Residual Standard Error (RSE), residual plots

## Requirements

The project requires the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels

## Usage

1. Ensure the dataset file `churn_clean.csv` is in the project directory
2. Run the notebook cells sequentially
3. The prepared dataset will be saved as `churn_prepared.csv` after preprocessing

## Model Results

The model provides:

- Coefficient estimates for each predictor variable
- Statistical significance testing using p-values
- R-squared values for both training and testing datasets
- Residual analysis to assess model assumptions

## Files

- `Churn.ipynb`: Main analysis notebook
- `churn_clean.csv`: Original dataset (required input)
- `churn_prepared.csv`: Preprocessed dataset (generated output)

## Notes

- The model uses log transformations to address skewness in predictor variables
- Residual plots are generated to validate linear regression assumptions
- The analysis includes both scikit-learn and statsmodels implementations for comprehensive model evaluation

