# Homework 6: Fully Connected Neural Networks

This homework uses fully connected neural networks (FCNNs) for a housing-price regression problem and a breast-cancer classification problem. I compared the neural networks with traditional machine-learning models using the same 80% training and 20% testing splits.

## Problems

### Problem 1: Housing Price Prediction

The goal was to predict housing prices using all available input features from the US Housing dataset.

The numerical features were standardized, while the categorical features were one-hot encoded. This produced 13 processed input features. I used a separate validation portion of the training data to select the architecture and stopping epoch. After model selection, the FCNN was retrained using the complete training set.

The selected network had:

- 13 input neurons
- Two hidden layers with 32 and 16 neurons
- ReLU activation
- 10% dropout after each hidden layer
- Adam optimizer
- Learning rate of 0.001
- Weight decay of 0.001
- Batch size of 32
- 59 final training epochs

I considered adding more hidden layers, but the larger network was already overfitting. Since the dataset contains only 545 observations, adding more layers would increase the number of parameters and could make the validation performance worse.

### Housing Results

| Model | MSE | RMSE | R² |
|---|---:|---:|---:|
| FCNN | 1,667,220,766,720.00 | $1,291,209.03 | 0.670156 |
| Linear Regression | 1,754,318,687,330.67 | $1,324,506.96 | 0.652924 |
| SVR | 1,904,828,428,945.08 | $1,380,155.22 | 0.623147 |

The FCNN produced the best housing results. It had the lowest MSE and RMSE and the highest R² score. However, its improvement over Linear Regression was modest, which shows that a more complex model does not always create a large improvement on a small tabular dataset.

### Problem 2: Breast Cancer Classification

The goal was to classify tumors as malignant or benign using all 30 input features from the Breast Cancer dataset.

I removed the ID and empty columns, encoded benign as 0 and malignant as 1, and standardized all input features. The split was stratified so that the training and testing sets kept similar class proportions.

The selected network had:

- 30 input neurons
- Two hidden layers with 32 and 16 neurons
- ReLU activation
- 20% dropout after each hidden layer
- Adam optimizer
- Learning rate of 0.001
- Weight decay of 0.001
- Batch size of 32
- Binary cross-entropy loss with logits
- 61 final training epochs

### Cancer Classification Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| FCNN | 0.9737 | 1.0000 | 0.9286 | 0.9630 |
| Logistic Regression | 0.9649 | 0.9750 | 0.9286 | 0.9512 |
| SVM | 0.9737 | 1.0000 | 0.9286 | 0.9630 |

The FCNN and the tuned RBF SVM produced the same test results. Both models correctly classified all 72 benign cases and 39 of the 42 malignant cases. The FCNN slightly outperformed Logistic Regression, but it did not outperform the SVM.

The FCNN confusion matrix contained:

- 72 true negatives
- 39 true positives
- 0 false positives
- 3 false negatives

## Main Libraries

- Python
- NumPy
- pandas
- Matplotlib
- Seaborn
- PyTorch
- Scikit-learn

## Repository Files

- `Homework_6_Commented.ipynb` — complete notebook with short code comments, results, and plots
- `Homework_6_Report.pdf` — final written report
- `Homework_6_Report.docx` — editable report
- `Homework_6_Commented_Code.pdf` — PDF copy of the complete code and outputs
- `README.md` — project summary and instructions

## Running the Notebook

1. Download or clone this repository.
2. Open `Homework_6_Commented.ipynb` in Google Colab or Jupyter Notebook.
3. Upload the Housing and Breast Cancer CSV files.
4. Update the dataset file paths if necessary.
5. Run the notebook cells from top to bottom.

If a required library is missing, install it before running the notebook. For example:

```bash
pip install numpy pandas matplotlib seaborn torch scikit-learn
```

## Conclusion

The FCNN performed best for housing-price prediction, although its advantage over Linear Regression was small. For cancer classification, the FCNN tied with the SVM and slightly outperformed Logistic Regression. Overall, the results show that neural networks can work well on tabular data, but traditional machine-learning models can remain very competitive when the datasets are small.

## Author

Kanda Mateta
