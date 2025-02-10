# Credit-Card-Fraud-Detection

## Overview
This project aims to detect fraudulent credit card transactions using machine learning techniques. We use Python and the `LogisticRegression` model for classification, applying the train-test split method to evaluate performance.

## Dataset
The dataset used for this project contains anonymized credit card transactions, where each entry is labeled as fraudulent (1) or non-fraudulent (0). It includes various transaction attributes useful for classification.

## Technologies Used
- **Python**: Primary programming language
- **Pandas & NumPy**: Data manipulation and preprocessing
- **Scikit-Learn**: Machine learning library for model building
- **Logistic Regression**: Supervised learning model for classification
- **Train-Test Split**: Splitting dataset for training and evaluation

## Installation
Ensure you have Python installed, then install dependencies using:
```bash
pip install pandas numpy scikit-learn
```

## Implementation Steps
1. **Load Dataset**: Read the credit card transaction data.
2. **Preprocess Data**: Handle missing values, normalize features, and balance the dataset if needed.
3. **Split Data**: Divide the dataset into training and testing sets using `train_test_split`.
4. **Train Model**: Use `LogisticRegression` to fit the model on training data.
5. **Evaluate Model**: Test the model on the test set and measure accuracy, precision, recall, and F1-score.

## Usage
Run the following script to train and test the model:
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("creditcard.csv")

# Split features and labels
X = df.drop("Class", axis=1)
y = df["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## Results
The model is evaluated based on accuracy, precision, recall, and F1-score to assess its performance in detecting fraudulent transactions.

## Contribution
Feel free to fork this repository, make improvements, and submit a pull request.

## License
This project is licensed under the MIT License.

