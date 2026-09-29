# Homework 2: Housing Price Prediction

## Overview

In this homework, I built linear regression models to predict house prices using the US Housing dataset. I implemented gradient descent from scratch instead of using a built-in machine-learning regression model.

The data was divided into 80% for training and 20% for testing. I also compared different learning rates and preprocessing methods to determine which settings produced the best results.

## Dataset

The dataset contains 545 houses. The target variable is `price`.

For the first model, I used these five input features:

- `area`
- `bedrooms`
- `bathrooms`
- `stories`
- `parking`

I also trained an expanded model using 11 input features, including the numerical housing information and encoded yes/no amenities.

## Preprocessing

I completed the following steps before training:

1. Loaded and inspected the housing dataset.
2. Confirmed that the dataset did not contain missing values.
3. Converted the categorical yes/no columns into numerical values.
4. Split the data into 80% training and 20% testing using `random_state=42`.
5. Standardized the inputs and target using values learned from the training data.
6. Initialized all model parameters to zero.

## Gradient Descent

I wrote the cost function and gradient descent algorithm from scratch. Each model was trained for 1,000 iterations.

The learning rates I tested were:

- `0.1`
- `0.05`
- `0.025`
- `0.01`

The training and validation losses were recorded during every iteration and plotted on the same graph. A learning rate of `0.1` gave the best stable result among the tested values.

## Results

| Model | Training Loss | Validation Loss |
|---|---:|---:|
| Five input features | 0.2189 | 0.3718 |
| Expanded 11-feature model | 0.1609 | 0.2920 |

The expanded model produced the lower training and validation losses. This shows that the additional housing features provided useful information for predicting price.

## Main Observations

- Standardization helped gradient descent converge more consistently.
- Larger stable learning rates reached a low loss faster.
- The expanded model performed better than the five-feature model.
- The validation loss remained higher than the training loss, which is expected because the model did not train on the test data.

## Files

- `Housing.csv` - housing dataset
- `HomeWork_2_Commented.ipynb` - complete commented notebook
- `HomeWork_2_Commented.pdf` - PDF copy of the code and outputs
- `Homework_2_Writeup.docx` - written report

## Requirements

```bash
pip install numpy pandas matplotlib scikit-learn
```

## How to Run

1. Download or clone this repository.
2. Place `Housing.csv` in the dataset folder used by the notebook.
3. Update the dataset path if necessary.
4. Open the notebook in Google Colab or Jupyter Notebook.
5. Run the cells from top to bottom.

## Conclusion

This homework helped me understand how linear regression and gradient descent work internally. Instead of only calling a completed regression model, I calculated the predictions, errors, costs, and parameter updates directly. The expanded model gave the best final result because it used more information about each house.

## Author

Kanda Mateta
