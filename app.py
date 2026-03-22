import streamlit as st
import pandas as pd
import pickle
from automl.engine import run_automl_pipeline

st.title("AutoML Model Finder")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    target_column = st.selectbox("Select Target Column", df.columns)

    if st.button("Run AutoML"):

        output = run_automl_pipeline(df, target_column)

        results = output["results"]
        best_model = output["best_model"]
        best_score = output["best_score"]
        report = output["report"]

        st.success("AutoML Completed")

        st.write("Best Model:", best_model)
        st.write("Score:", best_score)

        st.subheader("ML Report")
        st.text(report)

        results_df = pd.DataFrame(
            [(name, score) for name,(model,score) in results.items()],
            columns=["Model","Score"]
        )

        st.subheader("Model Comparison")
        st.bar_chart(results_df.set_index("Model"))

        model_bytes = pickle.dumps(results[best_model][0])

        st.download_button(
            "Download Model",
            model_bytes,
            file_name="best_model.pkl"
        )