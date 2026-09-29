# Homework 3: Logistic Regression Classification

## Overview

In this homework, I built binary logistic regression classifiers for the Diabetes and Breast Cancer datasets. I used an 80/20 training and testing split, standardized the input features, and tracked the loss and classification accuracy over 100 training iterations.

For the cancer dataset, I trained one model without a weight penalty and another model with an L2 weight penalty.

## Problem 1: Diabetes Classification

The goal of the first problem was to predict whether a patient had a positive or negative diabetes outcome.

### Input Features

- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`

The target variable was `Outcome`, where `0` represents a negative outcome and `1` represents a positive outcome.

### Preprocessing

I completed the following steps:

1. Replaced impossible zero measurements in glucose, blood pressure, skin thickness, insulin, and BMI with missing values.
2. Split the data into 80% training and 20% testing using stratification.
3. Filled missing values with medians calculated from the training data.
4. Standardized the input features using the training-set mean and standard deviation.
5. Trained an `SGDClassifier` with logistic loss, no penalty, and a learning rate of `0.01`.

### Diabetes Results

| Metric | Result |
|---|---:|
| Accuracy | 70.13% |
| Precision | 58.70% |
| Recall | 50.00% |
| F1 Score | 54.00% |

The diabetes confusion matrix was:

|  | Predicted Negative | Predicted Positive |
|---|---:|---:|
| Actual Negative | 81 | 19 |
| Actual Positive | 27 | 27 |

The model correctly classified 81 negative cases and 27 positive cases. However, the 50% recall shows that it missed half of the positive diabetes cases.

## Problem 2: Cancer Classification

The goal of the second problem was to classify breast cancer as benign or malignant using all 30 medical input features.

### Preprocessing

I completed the following steps:

1. Removed the patient ID and empty column.
2. Encoded malignant cancer as `1` and benign cancer as `0`.
3. Used a stratified 80/20 training and testing split.
4. Standardized all 30 input features.
5. Trained one logistic regression model without a penalty.
6. Repeated the training with an L2 penalty using `alpha=0.01`.

### Cancer Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Without penalty | 97.37% | 97.56% | 95.24% | 96.39% |
| With L2 penalty | 98.25% | 100.00% | 95.24% | 97.56% |

The L2 model produced the best overall result. It improved the accuracy and precision while keeping the same malignant recall. It also removed the false positive produced by the model without a penalty.

## Training Plots

For each classifier, I plotted:

- Training loss and testing loss over iterations
- Training accuracy and testing accuracy over iterations
- Final confusion matrix

These plots made it easier to see how each model learned and whether the testing performance remained stable.

## Files

- `diabetes.csv` - diabetes dataset
- `Cancer.csv` - breast cancer dataset
- `HomeWork_3_Commented.ipynb` - complete commented notebook
- `HomeWork_3_Commented.pdf` - PDF copy of the code and outputs
- `Homework_3_Writeup.docx` - written report

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## How to Run

1. Download or clone this repository.
2. Place both CSV datasets in the dataset folder used by the notebook.
3. Update the dataset paths if necessary.
4. Open the notebook in Google Colab or Jupyter Notebook.
5. Run all cells from top to bottom.

## Conclusion

The diabetes classifier provided a reasonable baseline, but its recall showed that it missed many positive cases. The cancer classifiers performed much better after all 30 inputs were standardized. Adding the L2 weight penalty gave the best cancer result by slightly increasing accuracy and removing a false positive.

## Author

Kanda Mateta
