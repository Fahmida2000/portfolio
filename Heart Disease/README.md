# Heart Disease Prediction Model

## Overview

This project focuses on predicting heart disease using machine learning techniques. The analysis employs a Decision Tree classifier to identify patients at risk of heart disease based on various clinical and demographic features. The model provides comprehensive evaluation metrics including accuracy, confusion matrices, ROC-AUC curves, and feature importance analysis.

## Dataset

The project uses a heart disease dataset (`heart.csv`) containing patient information with the following key variables:

- **Age**: Patient age in years
- **Sex**: Patient gender
- **ChestPainType**: Type of chest pain experienced
- **RestingBP**: Resting blood pressure (mm Hg)
- **Cholesterol**: Serum cholesterol level (mg/dl)
- **FastingBS**: Fasting blood sugar (binary indicator)
- **RestingECG**: Resting electrocardiographic results
- **MaxHR**: Maximum heart rate achieved
- **ExerciseAngina**: Exercise-induced angina (binary indicator)
- **Oldpeak**: ST depression induced by exercise relative to rest
- **ST_Slope**: Slope of the peak exercise ST segment
- **HeartDisease**: Target variable indicating presence of heart disease (0 = No, 1 = Yes)

## Project Structure

The analysis follows a structured workflow:

1. **Data Loading**: Import and initial exploration of the heart disease dataset
2. **Data Preprocessing**: Handling missing values and encoding categorical variables
3. **Exploratory Data Analysis**: Correlation matrix visualization
4. **Model Development**: Decision Tree classifier training
5. **Model Evaluation**: Comprehensive performance assessment
6. **Visualization**: Confusion matrices, ROC curves, and feature importance

## Key Features

### Data Preprocessing

- **Missing Value Handling**: Removal of records with missing values
- **Label Encoding**: Conversion of categorical variables to numerical format for model compatibility
- **Train-Test Split**: 80% training, 20% testing with random state for reproducibility

### Exploratory Data Analysis

- **Correlation Matrix**: Heatmap visualization showing relationships between all features
- **Descriptive Statistics**: Summary statistics for all numerical variables
- **Data Quality Assessment**: Identification of data completeness and distributions

### Modeling Approach

- **Model Type**: Decision Tree Classifier
- **Features**: All variables except the target (HeartDisease)
- **Target Variable**: HeartDisease (binary classification)
- **Train-Test Split**: 80% training, 20% testing
- **Random State**: 42 for reproducibility

## Model Results

### Performance Metrics

The model is evaluated using multiple metrics:

- **Accuracy**: Overall classification accuracy
- **Confusion Matrix**: Detailed breakdown of true positives, true negatives, false positives, and false negatives
- **Classification Report**: Precision, recall, and F1-score for each class
- **ROC-AUC**: Receiver Operating Characteristic curve and Area Under Curve metric

### Training Performance

- Perfect training accuracy (100%) indicating potential overfitting
- Complete separation of classes in training data
- High precision and recall for both classes

### Test Performance

- Test accuracy provides realistic performance estimate
- Confusion matrix shows actual prediction errors
- Classification report provides detailed per-class metrics

## Visualization

The project includes comprehensive visualizations:

1. **Correlation Heatmap**: Shows relationships between all features using a color-coded matrix
2. **Confusion Matrices**: Visual representation of prediction accuracy for both training and test sets
3. **ROC Curves**: Receiver Operating Characteristic curves for training and test data showing the trade-off between sensitivity and specificity
4. **Feature Importance**: Bar plot displaying the relative importance of each feature in the decision tree model

## Requirements

The project requires the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Usage

1. Ensure the dataset file `heart.csv` is in the project directory
2. Run the notebook cells sequentially to:
   - Load and preprocess the data
   - Visualize correlations
   - Train the Decision Tree model
   - Evaluate model performance
   - Generate visualizations

## Model Evaluation

### Metrics Provided

- **Accuracy**: Proportion of correct predictions
- **Precision**: Proportion of positive predictions that are actually positive
- **Recall**: Proportion of actual positives that are correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve, measuring the model's ability to distinguish between classes

### Feature Importance

The Decision Tree model provides feature importance scores indicating which variables are most influential in predicting heart disease. This helps identify key risk factors and can inform clinical decision-making.

## Files

- `Model.ipynb`: Main analysis notebook
- `heart.csv`: Heart disease dataset (required input)

## Key Insights

- Decision Tree achieves high performance on the heart disease prediction task
- Feature importance analysis reveals which clinical indicators are most predictive
- Correlation analysis helps understand relationships between different health metrics
- The model can potentially assist in early identification of heart disease risk

## Notes

- The Decision Tree model shows perfect training accuracy, which may indicate overfitting
- Test set performance provides a more realistic assessment of model generalization
- Feature importance scores help identify the most critical risk factors
- ROC-AUC curves provide insight into the model's discrimination ability across different thresholds
- The model can be used as a baseline for comparison with more complex algorithms
- Further model tuning and cross-validation could improve generalization performance

