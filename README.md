# Heart-Disease-Classification

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction
This project focuses on predicting the presence of heart disease in patients using various machine learning algorithms. The primary goal is to explore different models and select the one that provides the best performance in terms of accuracy, precision, recall, and F1 score.

## Dataset
The dataset used for this project is the [Heart Disease UCI dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease). The original data came from the Cleavland data from the UCI Machine Learning Repository. It consists of several features that include patient medical attributes and demographic information.

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
- ROC-AUC Curve
- Confusion matrix
- Classification report

## Results
This includes confusion matrices, classification reports, and ROC curves for each model.

Achieved Overall Accuracy score of 90%

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/Jeetkavaiya/Heart-Disease-Classification/blob/main/LICENSE) file for details.

## Acknowledgments
This project uses the following resources:

UCI Machine Learning Repository
<br/>
Various Python libraries including scikit-learn, pandas, numpy, and matplotlib
