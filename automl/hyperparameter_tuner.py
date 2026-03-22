import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor


def tune_random_forest(X, y, problem_type):

    def objective(trial):

        if problem_type == "classification":

            model = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                max_depth=trial.suggest_int("max_depth", 3, 20)
            )

            score = cross_val_score(model, X, y, cv=3, scoring="accuracy").mean()

        else:

            model = RandomForestRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                max_depth=trial.suggest_int("max_depth", 3, 20)
            )

            score = -cross_val_score(model, X, y, cv=3,
                                     scoring="neg_mean_squared_error").mean()

        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5, show_progress_bar=False)

    return study.best_params


def tune_xgboost(X, y, problem_type):

    def objective(trial):

        if problem_type == "classification":

            model = XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3)
            )

            score = cross_val_score(model, X, y, cv=3, scoring="accuracy").mean()

        else:

            model = XGBRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3)
            )

            score = -cross_val_score(model, X, y, cv=3,
                                     scoring="neg_mean_squared_error").mean()

        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5, show_progress_bar=False)

    return study.best_params