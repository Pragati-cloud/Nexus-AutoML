from sklearn.preprocessing import LabelEncoder
from automl.data_analyzer import analyze_dataset
from automl.data_cleaner import clean_data
from automl.feature_engineering import preprocess_features
from automl.problem_detector import detect_problem_type
from automl.feature_selector import select_features
from automl.model_trainer import train_models
from automl.model_selector import select_best_model
from automl.report_generator import generate_report

from automl.cache import hash_dataset, load_cache, save_cache


def run_automl_pipeline(df, target_column):

    dataset_hash = hash_dataset(df)

    analyze_dataset(df, target_column)
    if df[target_column].isnull().sum() > 0:
        print(f"⚠️ Dropping {df[target_column].isnull().sum()} rows with missing target")
        df = df.dropna(subset=[target_column])

    X, y = clean_data(df, target_column)

    # ✅ STEP 1: detect problem type FIRST
    problem_type = detect_problem_type(y)

    # 🔥 Safety override
    if len(set(y)) / len(y) > 0.5:
        print("⚠️ Overriding to regression due to high cardinality")
        problem_type = "regression"

    # ✅ STEP 2: create cache key AFTER problem_type
    target_key = f"{dataset_hash}_{target_column}_{problem_type}"

    # 🔥 Check full cache
    cached = load_cache(target_key)
    if cached is not None:
        print("⚡ Loaded full result from cache")
        return cached

    # Encode target if classification
    if problem_type == "classification":
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

    # 🔥 Dataset-level cache
    dataset_cached = load_cache(dataset_hash)

    if dataset_cached is not None:
        print("⚡ Loaded processed dataset from cache")
        X_selected = dataset_cached
    else:
        X_processed = preprocess_features(X)
        X_selected = select_features(X_processed, y, problem_type)
        save_cache(dataset_hash, X_selected)

    results = train_models(X_selected, y, problem_type)

    best_model, best_score = select_best_model(results)

    report = generate_report(df, problem_type, results, best_model, best_score)

    output = {
        "results": results,
        "best_model": best_model,
        "best_score": best_score,
        "report": report
    }

    # 💾 Save cache
    save_cache(target_key, output)

    return output
