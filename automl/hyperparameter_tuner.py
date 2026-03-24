import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np

try:
    from xgboost import XGBClassifier, XGBRegressor
    xgb_available = True
except Exception:
    xgb_available = False

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Fast defaults to keep API response time practical.
TUNING_TRIALS = 3
CV_FOLDS = 2
MAX_TUNING_ROWS = 3000


def _sample_for_tuning(X, y, max_rows=MAX_TUNING_ROWS, random_state=42):
    if len(y) <= max_rows:
        return X, y

    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(y), size=max_rows, replace=False)
    return X[idx], y[idx]


def tune_random_forest(X, y, problem_type):
    X_tune, y_tune = _sample_for_tuning(X, y)

    def objective(trial):

        if problem_type == "classification":

            model = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                max_depth=trial.suggest_int("max_depth", 3, 20),
                n_jobs=-1
            )

            score = cross_val_score(model, X_tune, y_tune, cv=CV_FOLDS, scoring="accuracy", n_jobs=-1).mean()

        else:

            model = RandomForestRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                max_depth=trial.suggest_int("max_depth", 3, 20),
                n_jobs=-1
            )

            score = -cross_val_score(
                model, X_tune, y_tune, cv=CV_FOLDS, scoring="neg_mean_squared_error", n_jobs=-1
            ).mean()

        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=TUNING_TRIALS, show_progress_bar=False)

    return study.best_params


def tune_xgboost(X, y, problem_type):
    if not xgb_available:
        return {}

    X_tune, y_tune = _sample_for_tuning(X, y)

    def objective(trial):

        if problem_type == "classification":

            model = XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3)
            )

            score = cross_val_score(model, X_tune, y_tune, cv=CV_FOLDS, scoring="accuracy").mean()

        else:

            model = XGBRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3)
            )

            score = -cross_val_score(
                model, X_tune, y_tune, cv=CV_FOLDS, scoring="neg_mean_squared_error"
            ).mean()

        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=TUNING_TRIALS, show_progress_bar=False)

    return study.best_params