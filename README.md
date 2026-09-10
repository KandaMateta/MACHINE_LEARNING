# Linear Regression with Gradient Descent

## Overview

This homework implements linear regression using the gradient descent algorithm.

The assignment is divided into two parts:

- Part 1: Linear regression using X1, X2, and X3 separately.
- Part 2: Multiple linear regression using X1, X2, and X3 together.

Different learning rates were tested to observe their effects on convergence and final loss.

---

## Part 1: Individual Linear Regression

Three models were trained independently:

### X1 Model

Y_hat = 5.7185 - 1.9568X1

Final Cost:

0.9906

### X2 Model

Y_hat = 0.7199 + 0.5639X2

### X3 Model

Y_hat = 2.7805 - 0.4845X3

Different learning rates were tested:

- 0.1
- 0.05
- 0.025
- 0.01

The experiments showed that larger learning rates generally converged faster.
A learning rate of 0.1 reached the stable loss in fewer iterations,
while 0.01 required more iterations.

---

## Part 2: Multiple Linear Regression

All three explanatory variables were used together.

The final model was:

Y_hat = 5.1912 - 1.9863X1 + 0.5529X2 - 0.2476X3

Best Final Cost:

0.73927

### Predictions

For (X1, X2, X3) = (1, 1, 1):

Y_hat = 3.5102

For (X1, X2, X3) = (2, 0, 4):

Y_hat = 0.2282

For (X1, X2, X3) = (3, 2, 1):

Y_hat = 0.0903

---

## Source Code

The complete Jupyter/Google Colab notebook in this repository contains:

- Python source code
- Gradient descent implementation
- Cost function
- Training results
- Learning-rate experiments
- Regression plots
- Loss vs. iteration plots
- Multiple linear regression
- Predictions
