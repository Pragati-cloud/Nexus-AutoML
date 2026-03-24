from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


def clean_data(df, target_column):
    # Remove rows where target is missing; supervised training cannot use NaN labels.
    df = df.dropna(subset=[target_column]).copy()
    if df.empty:
        raise ValueError("All rows were removed because target column contains only missing values.")

    # split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # numeric and categorical feature columns
    num_cols = X.select_dtypes(include="number").columns
    cat_cols = X.select_dtypes(include="object").columns

    # imputers
    num_imputer = SimpleImputer(strategy="mean")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    if len(num_cols) > 0:
        X[num_cols] = num_imputer.fit_transform(X[num_cols])

    if len(cat_cols) > 0:
        X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

    # encode target if it is categorical
    if y.dtype == "object":
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

    return X, y