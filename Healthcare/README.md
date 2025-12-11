# Healthcare ICU Patient Outcome Prediction

## Overview

This project focuses on predicting patient outcomes in Intensive Care Unit (ICU) settings, specifically targeting mortality prediction. The analysis integrates multiple healthcare data sources including patient demographics, medical history, treatments, diagnoses, vital signs, and laboratory results to build predictive models for clinical decision support.

## Dataset

The project utilizes comprehensive ICU patient data from multiple sources:

- **diagnosis.csv.gz**: ICU diagnoses with ICD-9 codes and timestamps
- **patient.csv.gz**: Patient demographics and discharge information
- **pastHistory.csv.gz**: Past medical history records
- **treatment.csv.gz**: Treatment records during ICU stay
- **vitalPeriodic.csv.gz**: Periodic vital sign measurements
- **vitalAperiodic.csv.gz**: Aperiodic vital sign measurements
- **lab.csv.gz**: Laboratory test results

## Project Structure

The analysis follows a comprehensive workflow:

1. **Data Loading**: Import and integration of multiple healthcare datasets
2. **Patient Selection**: Filtering and balancing of positive (death) and negative (survival) cases
3. **Feature Engineering**: Creation of temporal and clinical features
4. **Data Preprocessing**: Handling missing values, standardization, and encoding
5. **Model Development**: Multiple machine learning approaches
6. **Model Evaluation**: Performance assessment across different feature sets

## Key Features

### Data Preprocessing

- **Time Windowing**: Implementation of baseline and prediction windows for temporal analysis
- **Patient Filtering**: Age-based filtering (patients over 15 years) and minimum stay requirements
- **Class Balancing**: Balanced sampling of positive and negative cases
- **Missing Value Handling**: Comprehensive analysis and imputation strategies

### Feature Engineering

#### Sequence Features (Feature Set A)
- **Past Medical History**: Historical diagnoses converted to ICD-9 codes
- **Treatment Sequences**: Treatment records chronologically ordered
- **Diagnosis Sequences**: ICU diagnoses with ICD-9 codes
- **Tokenization**: Conversion of sequences to numerical tokens with vocabulary mapping
- **Padding**: Sequence padding to fixed length for model compatibility

#### Clinical Features (Feature Set B)
- **Vital Signs**: Heart rate, respiration, oxygen saturation (SaO2), blood pressure
- **Laboratory Results**: Glucose, electrolytes (sodium, potassium, chloride), blood counts (Hgb, Hct, WBC, platelets), kidney function (creatinine, BUN), liver function, blood gases
- **Demographics**: Age, gender, ethnicity, admission weight, unit type, admission source
- **Aggregated Metrics**: Min, max, mean values across time windows
- **Derived Features**: Ratios and changes from baseline (e.g., BUN/creatinine ratio, PaO2/FiO2 ratio)

### Temporal Analysis

- **Baseline Window**: Initial 0-360 minutes for establishing baseline measurements
- **Prediction Window**: 240 minutes before event for capturing predictive patterns
- **Event Offset**: Time-based alignment of all features relative to outcome events

## Modeling Approach

The project implements and compares multiple machine learning algorithms:

### Logistic Regression
- Baseline linear model for binary classification
- Evaluated on both feature sets

### Decision Tree Classifier
- Non-parametric tree-based model
- Provides interpretable decision rules

### Random Forest Classifier
- Ensemble method combining multiple decision trees
- Handles non-linear relationships and feature interactions

### Neural Network (Deep Learning)
- Multi-layer perceptron with TensorFlow/Keras
- Architecture: 8 hidden layers (256-128-64-32-16-8-4-2 neurons)
- Activation: ReLU for hidden layers, sigmoid for output
- Training: Early stopping and model checkpointing
- Loss function: Sparse categorical crossentropy

### Feature Sets Comparison

- **Feature Set A**: Sequence-based features (tokenized diagnosis/treatment sequences)
- **Feature Set B**: Clinical features (vitals, labs, demographics)

## Model Results

Models are evaluated using stratified train/validation/test splits:

- **Training Set**: 80% of data
- **Validation Set**: 10% of data (from training split)
- **Test Set**: 20% of data

Performance metrics include:
- Accuracy
- Precision
- Recall
- F1-score
- Classification reports for both classes

### Key Findings

- Feature Set B (clinical features) generally outperforms Feature Set A (sequence features)
- Random Forest with Feature Set B achieves best overall performance
- Neural networks show potential but require careful tuning for small datasets
- Class imbalance challenges require appropriate handling strategies

## Requirements

The project requires the following Python libraries:

- pandas
- numpy
- scikit-learn
- tensorflow (or tensorflow-gpu)
- matplotlib
- seaborn
- tqdm
- pickle (for model serialization)

## Usage

1. Ensure all dataset files are in the project directory:
   - diagnosis.csv.gz
   - patient.csv.gz
   - pastHistory.csv.gz
   - treatment.csv.gz
   - vitalPeriodic.csv.gz
   - vitalAperiodic.csv.gz
   - lab.csv.gz

2. Configure parameters in the notebook:
   - Diagnosis target (default: 'death')
   - Time windows and offsets
   - Sample sizes and ratios
   - Sequence length and vocabulary size

3. Run the notebook cells sequentially to:
   - Load and preprocess data
   - Engineer features
   - Train and evaluate models
   - Generate predictions

## Key Parameters

- **dx_offset**: Minimum time offset for diagnosis events (60 minutes)
- **pos_examples**: Number of positive examples (10,000 default, adjusted based on availability)
- **neg_examples**: Number of negative examples (balanced with positive)
- **seq_length**: Maximum sequence length for tokenization (100)
- **vocab_size**: Vocabulary size for sequence features (1000)
- **window**: Prediction window [240, 0] minutes before event
- **baseline**: Baseline window [0, 360] minutes from admission

## Files

- `HEALTHCARE.ipynb`: Main analysis notebook
- `word_to_ID.pkl`: Vocabulary mapping for sequence features
- `full_pipeline.pkl`: Preprocessing pipeline for feature transformation
- `model_checkpoint.h5`: Saved neural network model (if trained)
- `model_results.csv`: Model performance metrics

## Data Quality

The project includes comprehensive data quality checks:

- Missing value analysis for all features
- Completeness assessment across datasets
- Handling of incomplete records
- Data validation and cleaning procedures

## Clinical Features

### Vital Signs
- Heart rate (min, max, mean, change from baseline)
- Respiration rate (min, max, mean, change from baseline)
- Oxygen saturation (SaO2 minimum)
- Blood pressure measurements (systolic, diastolic, mean)

### Laboratory Tests
- Metabolic: Glucose, electrolytes (Na, K, Cl), calcium, bicarbonate
- Hematology: Hemoglobin, hematocrit, WBC, RBC, platelets
- Renal: Creatinine, BUN, BUN/creatinine ratio
- Liver: Albumin, total bilirubin, AST, ALT
- Blood gases: PaO2, PaCO2, pH, HCO3
- Coagulation: PT-INR
- Other: Lactate, anion gap, FiO2

## Notes

- The dataset contains class imbalance with significantly more survivors than deaths
- Small sample sizes require careful model selection and validation
- Missing data is prevalent in some features (especially specialized measurements)
- Feature engineering plays a critical role in model performance
- Temporal alignment of features is essential for accurate predictions
- The project demonstrates the integration of heterogeneous healthcare data sources
- Models are designed to support clinical decision-making in ICU settings

