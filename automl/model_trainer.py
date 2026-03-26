from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from xgboost import XGBClassifier, XGBRegressor

# LightGBM (optional)
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    lgbm_available = True
except:
    lgbm_available = False

from automl.hyperparameter_tuner import tune_random_forest, tune_xgboost


def train_models(X, y, problem_type):

    n_rows = len(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ⚡ Only tune on smaller datasets
    if n_rows < 20000:
        rf_params = tune_random_forest(X, y, problem_type)
        xgb_params = tune_xgboost(X, y, problem_type)
    else:
        rf_params = {"n_estimators": 20, "max_depth": 5}
        xgb_params = {"n_estimators": 20, "max_depth": 4}

    # -------------------------
    # 📉 MODEL SELECTION LOGIC
    # -------------------------
    if problem_type == "classification":

        models = {
            "Logistic Regression": LogisticRegression(max_iter=500),
            "Naive Bayes": GaussianNB(),
            "Random Forest": RandomForestClassifier(**rf_params),
            "XGBoost": XGBClassifier(**xgb_params, use_label_encoder=False, eval_metric="logloss")
        }

        # ❌ Skip heavy models for large datasets
        if n_rows < 15000:
            from sklearn.svm import SVC
            from sklearn.neighbors import KNeighborsClassifier

            models["SVM"] = SVC()
            models["KNN"] = KNeighborsClassifier()
            models["Gradient Boosting"] = GradientBoostingClassifier()

        if lgbm_available:
            models["LightGBM"] = LGBMClassifier(n_estimators=20)

    else:

        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "Random Forest": RandomForestRegressor(**rf_params),
            "XGBoost": XGBRegressor(**xgb_params)
        }

        if n_rows < 15000:
            from sklearn.svm import SVR
            models["SVR"] = SVR()
            models["Gradient Boosting"] = GradientBoostingRegressor()

        if lgbm_available:
            models["LightGBM"] = LGBMRegressor(n_estimators=20)

    results = {}

    # -------------------------
    # 🚀 TRAIN MODELS
    # -------------------------
    for name, model in models.items():

        try:
            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            if problem_type == "classification":
                score = accuracy_score(y_test, preds)
            else:
                score = mean_squared_error(y_test, preds)

            results[name] = (model, score)

            print(name, "Score:", score)

            # 🧹 Free prediction memory
            del preds

        except Exception as e:
            print(name, "failed:", e)

    return results
