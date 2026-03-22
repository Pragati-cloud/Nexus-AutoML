import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from automl.data_analyzer import analyze_dataset
from automl.data_cleaner import clean_data
from automl.feature_engineering import preprocess_features
from automl.problem_detector import detect_problem_type
from automl.feature_selector import select_features
from automl.model_trainer import train_models
from automl.model_selector import select_best_model
from automl.report_generator import generate_report


def run_automl_pipeline(df, target_column):

    analyze_dataset(df, target_column)

    X, y = clean_data(df, target_column)

    # detect problem type using raw y values
    problem_type = detect_problem_type(y)

    # for classification, ensure labels are consecutive integers starting at 0
    if problem_type == "classification":
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

    X_processed = preprocess_features(X)

    X_selected = select_features(X_processed, y, problem_type)

    results = train_models(X_selected, y, problem_type)

    best_model, best_score = select_best_model(results)

    report = generate_report(df, problem_type, results, best_model, best_score)

    return {
        "results": results,
        "best_model": best_model,
        "best_score": best_score,
        "report": report
    }