# Heart Disease Prediction Using Logistic Regression

This project implements a logistic regression model using the Framingham Heart Study dataset to predict the 10-year risk of coronary heart disease (CHD). The dataset contains over 4,000 records with 15 attributes and is widely used for cardiovascular research.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Dependencies](#dependencies)

## Overview
The goal of this project is to build a Machine-learning model using logistic regression to predict whether a patient has a 10-year risk of developing CHD. The model is evaluated based on accuracy, precision, recall, and other metrics.

## Dataset
The dataset used in this project comes from the Framingham Heart Study. It includes the following attributes:

- `Sex_male`: Gender of the patient (1 = male, 0 = female)
- `age`: Age of the patient
- `currentSmoker`: Smoking status (1 = yes, 0 = no)
- `cigsPerDay`: Average number of cigarettes smoked per day
- `BPMeds`: Whether the patient is on blood pressure medication
- `prevalentStroke`: History of stroke (1 = yes, 0 = no)
- `prevalentHyp`: Hypertension (1 = yes, 0 = no)
- `diabetes`: Diabetes status (1 = yes, 0 = no)
- `totChol`: Total cholesterol level
- `sysBP`: Systolic blood pressure
- `diaBP`: Diastolic blood pressure
- `BMI`: Body mass index
- `heartRate`: Heart rate
- `glucose`: Glucose level
- `TenYearCHD`: Target variable (1 = CHD in 10 years, 0 = no CHD in 10 years)

### Handling Missing Values
Missing values were removed from the dataset to ensure a clean input for the model.

## Project Workflow
1. **Data Preparation**
   - Load the dataset.
   - Drop unnecessary columns (e.g., `education`).
   - Handle missing values.
   - Normalize features for better model performance.

2. **Exploratory Data Analysis**
   - Visualize the distribution of patients affected by CHD.
   - Count the number of patients affected by CHD.

3. **Model Training**
   - Split the dataset into training and testing sets (70:30 ratio).
   - Train a logistic regression model using `sklearn`.

4. **Model Evaluation**
   - Evaluate the model's accuracy.
   - Generate a confusion matrix and classification report.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Bhukya7/Heart-Disease-Prediction-Using-Logistic-Regression.git
   cd heart-disease-prediction
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Open the project in a Jupyter Notebook or Google Colab.
2. Run the script to preprocess the data, train the model, and evaluate its performance.
3. Customize the code to experiment with different features or machine learning models.

## Results
- **Accuracy**: 84.90%
- **Confusion Matrix**:
  ```
             Predicted:0  Predicted:1
  Actual:0          951          11
  Actual:1          161          14
  ```
- **Classification Report**:
  ```
              precision    recall  f1-score   support

           0       0.85      0.99      0.92       951
           1       0.61      0.08      0.14       175

    accuracy                           0.85      1126
   macro avg       0.73      0.54      0.53      1126
weighted avg       0.82      0.85      0.80      1126
  ```

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels

Install dependencies using the following command:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```
