import pandas as pd

def analyze_dataset(df, target_column):

    info = {}

    info["rows"] = df.shape[0]
    info["columns"] = df.shape[1]

    features = df.drop(target_column, axis=1)

    numeric_features = features.select_dtypes(include="number").columns.tolist()
    categorical_features = features.select_dtypes(include="object").columns.tolist()

    info["numeric_features"] = numeric_features
    info["categorical_features"] = categorical_features
    info["missing_values"] = df.isnull().sum().sum()

    print("Dataset Analysis")
    print(info)

    return info