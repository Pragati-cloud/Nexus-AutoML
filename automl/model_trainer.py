from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier, XGBRegressor

# LightGBM
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    lgbm_available = True
except:
    lgbm_available = False

from automl.hyperparameter_tuner import tune_random_forest, tune_xgboost


# -------------------------
# 🚀 Stage 1 (Fast Models)
# -------------------------
def train_fast_models(X, y, problem_type):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if problem_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=500),
            "Naive Bayes": GaussianNB(),
            "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=8),
            "XGBoost": XGBClassifier(n_estimators=50, max_depth=6, eval_metric="logloss")
        }

        if lgbm_available:
            models["LightGBM"] = LGBMClassifier(n_estimators=50)

    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=8),
            "XGBoost": XGBRegressor(n_estimators=50, max_depth=6)
        }

        if lgbm_available:
            models["LightGBM"] = LGBMRegressor(n_estimators=50)

    results = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            score = accuracy_score(y_test, preds) if problem_type == "classification" else mean_squared_error(y_test, preds)

            results[name] = (model, score)
            del preds

        except Exception as e:
            print(name, "failed:", e)

    return results


# -------------------------
# 🎯 Select Top Models
# -------------------------
def get_top_models(results, problem_type):

    reverse = True if problem_type == "classification" else False

    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1][1],
        reverse=reverse
    )

    return [name for name, _ in sorted_models[:2]]


# -------------------------
# ⚡ Stage 2 (Tuning)
# -------------------------
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

    if "XGBoost" in top_models:
        xgb_params = tune_xgboost(X, y, problem_type)
        if problem_type == "classification":
            models["XGBoost"] = XGBClassifier(**xgb_params, eval_metric="logloss")
        else:
            models["XGBoost"] = XGBRegressor(**xgb_params)

    results = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            score = accuracy_score(y_test, preds) if problem_type == "classification" else mean_squared_error(y_test, preds)

            results[name] = (model, score)
            del preds

        except Exception as e:
            print(name, "failed:", e)

    return results


# -------------------------
# 🧠 FINAL TRAIN FUNCTION
# -------------------------
def train_models(X, y, problem_type):

    # Stage 1
    fast_results = train_fast_models(X, y, problem_type)

    # Select top models
    top_models = get_top_models(fast_results, problem_type)
    print("Top Models:", top_models)

    # Stage 2
    tuned_results = train_tuned_models(X, y, problem_type, top_models)

    # Merge results
    final_results = {**fast_results, **tuned_results}

    return final_results
