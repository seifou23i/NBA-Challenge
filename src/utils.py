import numpy as np
from sklearn.metrics import recall_score, precision_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold


def evaluate_classifiers(classifiers, X, y, n_splits=3, random_state=42):
    """
    Evaluates multiple classifiers using K-Fold cross-validation.

    Args:
        classifiers (dict): A dictionary of classifiers to evaluate.
        X (numpy.ndarray): The features.
        y (numpy.ndarray): The target labels.
        n_splits (int, optional): Number of cross-validation folds. Defaults to 3.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        None
    """

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for name, classifier in classifiers.items():
        print(f"******* Training {name} *******")

        precisions = []
        recalls = []
        confusion_mat = np.zeros((2, 2))

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            precision = precision_score(y_test, y_pred)
            precisions.append(precision)
            recall = recall_score(y_test, y_pred)
            recalls.append(recall)
            confusion_mat += confusion_matrix(y_test, y_pred)

        avg_precisions = sum(precisions) / len(precisions)
        avg_recalls = sum(recalls) / len(recalls)

        print(f"Average precision of {name}: {avg_precisions:.4f}")
        print(f"Average recall of {name}: {avg_recalls:.4f}")
        print(f"Confusion matrix of {name}: \n {confusion_mat}")


def tune_and_evaluate_models(models, X, y, scoring='precision', cv=3, n_jobs=-1):
    """
    Performs hyperparameter tuning using GridSearchCV and evaluates the best model
    for each model in the provided dictionary.

    Args:
        models (dict): A dictionary containing machine learning models and their hyperparameter grids.
        X (numpy.ndarray): The features.
        y (numpy.ndarray): The target labels.
        scoring (str, optional): The scoring metric for GridSearchCV. Defaults to 'precision'.
        cv (int, optional): Number of cross-validation folds. Defaults to 3.
        n_jobs (int, optional): Number of CPU cores to use in parallel. Defaults to -1 (all cores).

    Returns:
        None
    """

    for name, (model, params) in models.items():
        print(f"************Tuning hyperparameters for {name} ************")

        best_params, test_precision = tune_and_evaluate_model(model, params, X, y, scoring, cv, n_jobs)

        print(f"Best Hyperparameters: {best_params}")
        print(f"Test Precision: {test_precision:.4f}")


def tune_and_evaluate_model(model, params, X, y, scoring='precision', cv=3, n_jobs=-1):
    """
    Hyperparameter tuning using GridSearchCV and evaluation of the best model.

    Args:
        model (estimator): The machine learning model to tune.
        params (dict): A dictionary of hyperparameters and their search grids.
        X (numpy.ndarray): The features.
        y (numpy.ndarray): The target labels.
        scoring (str, optional): The scoring metric for GridSearchCV. Defaults to 'precision'.
        cv (int, optional): Number of cross-validation folds. Defaults to 3.
        n_jobs (int, optional): Number of CPU cores to use in parallel. Defaults to -1 (all cores).

    Returns:
        tuple: A tuple containing the best hyperparameters and the best model's test precision.
    """

    # Same logic as previous tune_and_evaluate_model function

    grid_search = GridSearchCV(model, params, scoring=scoring, cv=cv, n_jobs=n_jobs)
    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    return best_params, best_model.score(X, y)


def replace_outliers_with_bounds(data):
    """
    Replaces outliers in a NumPy array with the calculated lower and upper bounds.

    Args:
        data (numpy.ndarray): The input NumPy array.

    Returns:
        numpy.ndarray: The modified array with outliers replaced by bounds.

    Raises:
        ValueError: If the input data is not a 2D NumPy array.
    """

    if data.ndim != 2:
        raise ValueError("Input data must be a 2D NumPy array.")

    # Calculate quartiles and IQR efficiently
    Q1, Q3 = np.percentile(data, [25, 75], axis=0)
    IQR = Q3 - Q1

    # Calculate bounds in a single operation
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Replace outliers efficiently using np.clip
    data = np.clip(data, lower_bound, upper_bound)

    return data
