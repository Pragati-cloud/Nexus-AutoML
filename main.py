import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder

from automl.data_analyzer import analyze_dataset
from automl.data_cleaner import clean_data
from automl.feature_engineering import preprocess_features
from automl.problem_detector import detect_problem_type
from automl.feature_selector import select_features
from automl.model_trainer import train_models
from automl.model_selector import select_best_model


st.title("AutoML Model Finder")

st.write("Upload a dataset and automatically find the best ML model.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    target_column = st.selectbox("Select Target Column", df.columns)

    if st.button("Run AutoML"):

        # Analyze dataset
        analyze_dataset(df, target_column)

        # Clean data
        X, y = clean_data(df, target_column)

        # Encode target if needed
        if isinstance(y.iloc[0], str):
            encoder = LabelEncoder()
            y = encoder.fit_transform(y)

        # Detect problem
        problem_type = detect_problem_type(y)

        # Preprocess
        X_processed = preprocess_features(X)

        # Feature selection
        X_selected = select_features(X_processed, y, problem_type)

        # Train models
        results = train_models(X_selected, y, problem_type)

        # -----------------
        # MODEL COMPARISON
        # -----------------
        model_names = []
        scores = []

        for name, (model, score) in results.items():
            model_names.append(name)
            scores.append(score)

        chart_df = pd.DataFrame({
            "Model": model_names,
            "Score": scores
        })

        st.subheader("Model Comparison")
        st.bar_chart(chart_df.set_index("Model"))
        # -----------------
        # BEST MODEL
        # -----------------
        best_model, best_score = select_best_model(results)

        st.success("AutoML Completed")

        st.write("Best Model:", best_model)
        st.write("Score:", round(best_score, 3))

        # -----------------
        # DOWNLOAD MODEL
        # -----------------
        best_model_obj = results[best_model][0]

        joblib.dump(best_model_obj, "best_model.pkl")

        with open("best_model.pkl", "rb") as f:
            st.download_button(
                label="Download Best Model",
                data=f,
                file_name="best_model.pkl"
            )