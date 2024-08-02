import pytest
import pickle
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def test_model():
    with open('regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    california_housing = fetch_california_housing()
    X, y = california_housing.data, california_housing.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    assert mse < 1.0  # Example condition
