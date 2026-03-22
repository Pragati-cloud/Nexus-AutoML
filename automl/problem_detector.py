import numpy as np
import pandas as pd


def detect_problem_type(y):
    """
    Detect whether the problem is classification or regression based on the target variable.
    """
    # Convert to numpy array if it's a pandas series
    if isinstance(y, pd.Series):
        y = y.values

    unique_values = np.unique(y)
    n_unique = len(unique_values)
    n_samples = len(y)

    # Check if target is numeric
    try:
        y_numeric = pd.to_numeric(y, errors='coerce')
        is_numeric = not pd.isna(y_numeric).any()
    except:
        is_numeric = False

    if not is_numeric:
        # If not numeric, it's classification
        problem_type = "classification"
    else:
        # For numeric data, check various heuristics
        if n_unique == 2:
            # Binary case - could be classification or regression
            # Check if values are 0/1 or other binary
            if set(unique_values) == {0, 1} or set(unique_values) == {0.0, 1.0}:
                problem_type = "classification"
            else:
                # Other binary values - likely classification
                problem_type = "classification"
        elif n_unique <= 10 and n_samples > 50:
            # Few unique values relative to sample size - likely classification
            problem_type = "classification"
        elif n_unique / n_samples < 0.1:
            # Less than 10% unique values - likely classification
            problem_type = "classification"
        else:
            # Many unique values - likely regression
            problem_type = "regression"

    print(f"Detected Problem Type: {problem_type} (unique values: {n_unique}, total samples: {n_samples})")
    return problem_type