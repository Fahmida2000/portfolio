# Payment Service Provider (PSP) Transaction Analysis and Recommendation System

## Overview

This project analyzes payment service provider transaction data to predict transaction success rates and recommend optimal PSPs for different transaction scenarios. The analysis includes comprehensive data exploration, feature engineering, and machine learning model development to improve payment processing efficiency.

## Dataset

The project uses transaction data from January and February 2019 (`PSP_Jan_Feb_2019.xlsx`). The dataset contains the following key variables:

- **tmsp**: Timestamp of the transaction
- **country**: Country where the transaction originated
- **amount**: Transaction amount
- **success**: Binary indicator (0 = failed, 1 = successful)
- **PSP**: Payment Service Provider used for the transaction
- **3D_secured**: Binary indicator for 3D Secure authentication
- **card**: Card type used (Visa, Master, Diners)

## Project Structure

The analysis follows a structured workflow:

1. **Data Loading**: Import and initial exploration of transaction data
2. **Data Quality Assessment**: Comprehensive analysis of transaction distributions and success rates
3. **Exploratory Data Analysis**: Visual analysis of transaction patterns across PSPs, countries, and card types
4. **Feature Engineering**: Creation of time-based and categorical features for improved model performance
5. **Baseline Model Development**: Logistic Regression model as a baseline
6. **Advanced Modeling**: Random Forest classifier with SMOTE for handling class imbalance
7. **Model Evaluation**: Performance assessment using confusion matrices and classification reports
8. **Feature Importance Analysis**: Identification of key factors influencing transaction success
9. **GUI Application**: Interactive interface for PSP recommendations

## Key Features

### Data Preprocessing

- Extraction of temporal features from timestamps (hour, day of week, part of day)
- Categorization of transaction amounts into low, medium, and high ranges
- Encoding of categorical variables for machine learning compatibility
- Standardization of numerical features

### Exploratory Data Analysis

The analysis includes comprehensive visualizations:

- Distribution of transactions across different PSPs
- Success rate analysis by PSP, country, and card type
- Relationship between transaction amount and success rate
- Impact of 3D Secure authentication on transaction success
- Temporal patterns in transaction processing

### Feature Engineering

- **Hour**: Extracted hour of day from timestamp
- **Day of Week**: Day of week (0-6) from timestamp
- **Part of Day**: Categorized into Night, Morning, Afternoon, Evening
- **Amount Category**: Binned into Low, Medium, High ranges

### Modeling Approach

#### Baseline Model
- **Model Type**: Logistic Regression
- **Features**: Country, amount, 3D_secured, card type, temporal features
- **Train-Test Split**: 80% training, 20% testing
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score

#### Advanced Model
- **Model Type**: Random Forest Classifier
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Features**: Enhanced feature set with engineered temporal variables
- **Evaluation**: Comprehensive performance metrics and confusion matrix analysis

## Model Results

### Feature Importance

The Random Forest model identified the following key factors (in order of importance):

1. **Amount**: Transaction amount is the most significant predictor
2. **Hour**: Time of day significantly impacts success rates
3. **Day of Week**: Day of the week shows notable influence
4. **Country**: Geographic location affects transaction outcomes
5. **Card Type**: Card type contributes to success prediction
6. **Part of Day**: Time period categorization provides additional insights
7. **3D_secured**: Security authentication method impact
8. **Amount Category**: Categorized amount ranges

### Performance Metrics

The models provide insights into transaction success prediction with evaluation through:

- Classification accuracy
- Precision and recall for both success and failure classes
- F1-scores for balanced performance assessment
- Confusion matrices for detailed error analysis

## Requirements

The project requires the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn (for SMOTE)
- tkinter (for GUI application)

## Usage

1. Ensure the dataset file `PSP_Jan_Feb_2019.xlsx` is in the project directory
2. Run the notebook cells sequentially to perform the complete analysis
3. The GUI application can be launched to interact with the recommendation system

## GUI Application

The project includes a graphical user interface built with tkinter that allows users to:

- Input transaction details (amount, country, card type)
- Receive PSP recommendations based on the trained model
- Interact with the recommendation system in a user-friendly manner

## Files

- `Credit Card.ipynb`: Main analysis notebook
- `PSP_Jan_Feb_2019.xlsx`: Transaction dataset (required input)

## Key Insights

- Transaction amount and timing are the most critical factors in predicting success
- Different PSPs show varying success rates across different transaction characteristics
- Temporal patterns significantly influence transaction outcomes
- Class imbalance in the dataset requires specialized handling techniques like SMOTE
- Feature engineering, particularly temporal features, substantially improves model performance

## Notes

- The dataset contains class imbalance, with successful transactions being less frequent than failures
- SMOTE technique is employed to address the imbalanced nature of the target variable
- The Random Forest model provides feature importance rankings to understand key success factors
- The GUI application serves as a practical tool for real-world PSP selection based on transaction characteristics

