import numpy as np
import pandas as pd


def detect_problem_type(y):
    """
    Detect whether the problem is classification or regression based on the target variable.
    """
    # Convert to numpy array if it's a pandas series
    if isinstance(y, pd.Series):
        y = y.values

    y_series = pd.Series(y)
    n_samples = len(y_series)
    n_unique = y_series.nunique(dropna=True)

    # Non-numeric target -> classification
    if not pd.api.types.is_numeric_dtype(y_series):
        problem_type = "classification"
    else:
        unique_ratio = n_unique / max(n_samples, 1)

        # Numeric target with very high cardinality is almost always regression.
        if unique_ratio >= 0.5:
            problem_type = "regression"
        # Binary/very-low-cardinality numeric targets are classification.
        elif n_unique <= 2:
            problem_type = "classification"
        elif n_unique <= 20 and unique_ratio <= 0.1:
            problem_type = "classification"
        else:
            # Prefer regression for continuous/high-cardinality numeric targets.
            problem_type = "regression"

    print(f"Detected Problem Type: {problem_type} (unique values: {n_unique}, total samples: {n_samples})")
    return problem_type