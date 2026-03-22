from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso

from automl.hyperparameter_tuner import tune_random_forest, tune_xgboost

# LightGBM
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    lgbm_available = True
except:
    lgbm_available = False


def train_models(X, y, problem_type):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Hyperparameter tuning
    rf_params = tune_random_forest(X, y, problem_type)
    xgb_params = tune_xgboost(X, y, problem_type)

    if problem_type == "classification":

        models = {
           
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(**rf_params),
            "SVM": SVC(),
            "KNN": KNeighborsClassifier(),                
            "Naive Bayes": GaussianNB(),                  
            "Gradient Boosting": GradientBoostingClassifier(),  
            "XGBoost": XGBClassifier(**xgb_params, use_label_encoder=False, eval_metric="logloss")
        }

        if lgbm_available:
            models["LightGBM"] = LGBMClassifier()         

    else:

        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(),                            
            "Lasso": Lasso(),                            
            "Random Forest": RandomForestRegressor(**rf_params),
            "SVR": SVR(),
            "Gradient Boosting": GradientBoostingRegressor(),  
            "XGBoost": XGBRegressor(**xgb_params, eval_metric="rmse")
        }

        if lgbm_available:
            models["LightGBM"] = LGBMRegressor()          

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

            print(name, "Score:", score)

        except Exception as e:
            print(name, "failed:", e)

    return results