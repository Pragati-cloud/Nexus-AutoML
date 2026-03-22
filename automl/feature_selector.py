def select_features(X, y, problem_type, k=5):

    from sklearn.feature_selection import SelectKBest, f_classif, f_regression

    k = min(k, X.shape[1])

    if problem_type == "classification":
        selector = SelectKBest(score_func=f_classif, k=k)
    else:
        selector = SelectKBest(score_func=f_regression, k=k)

    X_new = selector.fit_transform(X, y)

    print("Selected Top", k, "Features")

    return X_new