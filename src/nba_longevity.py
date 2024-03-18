from abc import ABC
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler


class NBALongevity(ABC):
    """
    An abstract base class (ABC) for building and evaluating a Logistic Regression model
    to predict NBA player longevity. It provides functionalities for data preprocessing,
    model training, prediction, evaluation, saving, and loading trained models and scaler.

    This class is designed to be subclassed for specific implementations of
    NBA longevity prediction models. Subclasses can override methods as needed.

    Args:
        C (float, optional): Regularization parameter for LogisticRegression. Defaults to 0.1.
        penalty (str, optional): Penalty type for LogisticRegression.
                                 Defaults to 'l2' (L2 regularization).
        random_state (int, optional): Seed for random number generation in model training.
                                     Defaults to 42 for reproducibility.
    """

    def __init__(self, C: float = 0.1, penalty: str = 'l2', random_state: int = 42):
        """
        Initializes the NBALongevity model with specified parameters.

        Args:
          C (float, optional): Regularization parameter for LogisticRegression. Defaults to 0.1.
          penalty (str, optional): Penalty type for LogisticRegression.
                                   Defaults to 'l2' (L2 regularization).
          random_state (int, optional): Seed for random number generation in model training.
                                       Defaults to 42 for reproducibility.
        """
        self.model = LogisticRegression(C=C, penalty=penalty, random_state=random_state)
        self.scaler = StandardScaler()
        self.name_features = None  # Stores feature names (optional)
        self.name_target = None  # Stores target variable name (optional)
        self.is_fitted = False  # Flag to indicate if model is fitted

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the Logistic Regression model on the provided training data (X) and target variable (y).

        Args:
          X (pd.DataFrame): The training data features.
          y (pd.Series): The training data target variable (e.g., indicating long vs. short career).
        """
        self.name_features = X.columns.to_numpy()
        self.name_target = y.name
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts the class labels (5 years career longevity) for new data points (X).

        Args:
          X (pd.DataFrame): The data for which to predict career longevity.

        Returns:
          np.ndarray: The predicted class labels (0 or 1).
        """

        if not self.is_fitted:
            raise ValueError("Model not trained yet. Please call 'fit' first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts the probability of each class (5 years career longevity) for new data points (X).

        Args:
          X (pd.DataFrame): The data for which to predict career longevity probabilities.

        Returns:
          np.ndarray: The predicted probabilities of each class (probability of long vs. short career).
        """
        if not self.is_fitted:
            raise ValueError("Model not trained yet. Please call 'fit' first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, metrics: List[str] = ["precision"]) -> dict:
        """
        Evaluates the model performance on the provided testing data (X_test, y_test)
        using the specified metrics.

        Args:
          X_test (pd.DataFrame): The testing data features.
          y_test (pd.Series): The testing data target variable (e.g., indicating long vs. short career).
          metrics (List[str], optional): A list of metric names to calculate. Defaults to ["precision"].

        Returns:
          dict: A dictionary containing the calculated metric scores.

        Raises:
          ValueError: If the model is not fitted yet (call `fit` first).
        """

        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Please call 'fit' first.")

        X_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_scaled)

        results = {}
        for metric in metrics:
            if metric == 'precision':
                results['precision'] = precision_score(y_test, y_pred)
            elif metric == 'recall':
                results['recall'] = recall_score(y_test, y_pred)
            elif metric == 'f1':
                results['f1'] = f1_score(y_test, y_pred)
            elif metric == 'accuracy':
                results['accuracy'] = accuracy_score(y_test, y_pred)
            else:
                raise ValueError(f"Invalid metric: {metric}")

        return results

    def load_models(self, model_path: str, scaler_path: str):
        """
        Loads a previously trained LogisticRegression model and StandardScaler from specified paths.

        Args:
          model_path (str): The path to the saved LogisticRegression model file.
          scaler_path (str): The path to the saved StandardScaler model file.

        Raises:
          ValueError: If the model or scaler is not fitted yet (likely due to a missing file).
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.is_fitted = True

    def save_models(self, model_path: str, scaler_path: str):
        """
        Saves the trained LogisticRegression model and StandardScaler to specified paths.

        Args:
          model_path (str): The path to save the LogisticRegression model file.
          scaler_path (str): The path to save the StandardScaler model file.

        Raises:
          ValueError: If the model or scaler is not fitted yet (call `fit` first).
        """
        if not self.is_fitted:
            raise ValueError("Model or scaler not fitted yet. Please call 'fit' first.")
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
