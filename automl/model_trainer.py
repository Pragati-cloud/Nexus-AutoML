from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

from automl.hyperparameter_tuner import tune_random_forest, tune_xgboost

try:
    from xgboost import XGBClassifier, XGBRegressor
    xgb_available = True
except Exception:
    xgb_available = False

def train_fast_models(X, y, problem_type):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if problem_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=50),
            "SVM": SVC(),
        }
        if xgb_available:
            models["XGBoost"] = XGBClassifier(n_estimators=50, eval_metric="mlogloss")
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=50),
            "SVR": SVR(),
        }
        if xgb_available:
            models["XGBoost"] = XGBRegressor(n_estimators=50)

    results = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            if problem_type == "classification":
                score = accuracy_score(y_test, preds)
            else:
                score = mean_squared_error(y_test, preds)

            results[name] = (model, score)
        except Exception as e:
            print(name, "failed:", e)

    return results


def get_top_models(results, problem_type, top_k=2):
    reverse = True if problem_type == "classification" else False

    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1][1],
        reverse=reverse
    )

    return [name for name, _ in sorted_models[:top_k]]


def train_tuned_models(X, y, problem_type, top_models):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {}

    if "Random Forest" in top_models:
        rf_params = tune_random_forest(X, y, problem_type)
        if problem_type == "classification":
            models["Random Forest"] = RandomForestClassifier(**rf_params)
        else:
            models["Random Forest"] = RandomForestRegressor(**rf_params)

    if "XGBoost" in top_models and xgb_available:
        xgb_params = tune_xgboost(X, y, problem_type)
        if problem_type == "classification":
            models["XGBoost"] = XGBClassifier(**xgb_params, eval_metric="mlogloss")
        else:
            models["XGBoost"] = XGBRegressor(**xgb_params)

    results = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            if problem_type == "classification":
                score = accuracy_score(y_test, preds)
            else:
                score = mean_squared_error(y_test, preds)

            results[name] = (model, score)
        except Exception as e:
            print(name, "failed:", e)

    return results


def train_models(X, y, problem_type):
    fast_results = train_fast_models(X, y, problem_type)
    if not fast_results:
        return {}

    top_models = get_top_models(fast_results, problem_type, top_k=2)
    tuned_results = train_tuned_models(X, y, problem_type, top_models)

    final_results = dict(fast_results)
    final_results.update(tuned_results)
    return final_results