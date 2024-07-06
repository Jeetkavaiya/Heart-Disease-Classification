# Heart-Disease-Classification
<br/>
This is a complete Heart Disease Classification Project using Machine learning.
<br/>
1️⃣ Data Collection & Preprocessing: The original data came from the Cleavland data from the UCI Machine Learning Repository. 
https://archive.ics.uci.edu/ml/datasets/heart+Disease
<br/>
2️⃣ Exploratory Data Analysis (EDA): Investigated correlations and patterns within the dataset to guide feature selection and model design.
<br/>
3️⃣ Feature Engineering: Engineered relevant features like age, cp levels, and exercise habits to enhance model performance.
<br/>
4️⃣ Model Selection: Explored various algorithms including Logistic Regression, Random Forest, and K-Nearest Neighbours classifier to find the best fit for the data.
<br/>
5️⃣ Hyperparameter Tuning: Optimized model parameters using techniques like Grid Search to maximize accuracy and robustness.
<br/>
6️⃣ Evaluation & Validation: Assessed models using metrics such as ROC curve and AUC score, Confusion matrix, Classification report, Precision, Recall, and F1-score to ensure reliability and effectiveness.


# Heart Disease Classification Project

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction
This project focuses on predicting the presence of heart disease in patients using various machine learning algorithms. The primary goal is to explore different models and select the one that provides the best performance in terms of accuracy, precision, recall, and F1-score.

## Dataset
The dataset used for this project is the [Heart Disease UCI dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease). The original data came from the Cleavland data from the UCI Machine Learning Repository. It consists of several features that include patient medical attributes and demographic information.

## Installation
To run this project, you'll need to have Python installed. Follow the steps below to set up the environment:

1. Clone the repository:
   ```sh
   git clone https://github.com/Jeetkavaiya/Heart-Disease-Classification.git
   cd heart-disease-classification

2. Install the required dependencies:
   pip install -r requirements.txt

## Model Training
The train_model.py script preprocesses the data, splits it into training and testing sets, and trains multiple machine learning models. The following models are implemented:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

## Evaluation
Model performance is evaluated based on the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## Results
Detailed results of the models are provided in the results directory. This includes confusion matrices, classification reports, and ROC curves for each model.

## Acknowledgments
This project uses the following resources:

UCI Machine Learning Repository
<br/>
Various Python libraries including scikit-learn, pandas, numpy, and matplotlib
