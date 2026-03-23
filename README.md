<div align="center">

# ⚡ Nexus AutoML

**Machine Learning, Automated.**

Nexus AutoML is a production-grade, custom-built AutoML system designed to eliminate the complexity of building machine learning pipelines from scratch. Upload any tabular dataset, select a target column, and the system automatically handles data cleaning, feature engineering, model training, hyperparameter tuning, and performance comparison — delivering a ready-to-use model in minutes.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-latest-orange?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

</div>

---

## 📌 Overview

Most AutoML tools are black boxes. Nexus AutoML is designed to be **transparent, extensible, and developer-friendly**. The system exposes a clean REST API, supports a custom HTML/CSS/JS frontend, and provides detailed per-model performance breakdowns alongside a downloadable report — giving you full visibility into how your best model was selected.

**Who is this for?**
- Data scientists who want rapid model prototyping without boilerplate
- Developers building data-driven applications that need an embedded ML backend
- Students and researchers exploring how AutoML pipelines are architected

---

## 🚀 Features

| Feature | Description |
|---|---|
| 📊 Automatic Dataset Analysis | Detects column types, missing values, cardinality, and data distributions |
| 🧹 Data Cleaning & Preprocessing | Handles missing values, encodes categoricals, and scales numerical features |
| 🔍 Problem Type Detection | Automatically identifies Classification vs. Regression tasks |
| 🧠 Multi-Model Training | Trains 8 algorithms simultaneously and ranks them by performance |
| ⚙️ Hyperparameter Tuning | Uses **Optuna** for intelligent, trial-based hyperparameter optimization |
| 📈 Model Comparison Dashboard | Side-by-side performance metrics with visual progress bars |
| 🏆 Automatic Best Model Selection | Selects and serializes the highest-performing model |
| 📄 Auto-Generated ML Report | Produces a structured, downloadable text report for every run |
| 📦 Model Export | Download the trained model as a `.pkl` file for deployment |
| 🌐 FastAPI Backend | Clean REST API with interactive Swagger UI at `/docs` |
| 🎨 Custom Web UI | Standalone HTML/CSS/JS frontend with drag-and-drop file upload |

---

## 🧠 Supported Algorithms

### Classification

| Algorithm | Notes |
|---|---|
| Logistic Regression | Baseline linear model |
| Random Forest | Ensemble of decision trees |
| Support Vector Machine (SVM) | Effective for high-dimensional data |
| K-Nearest Neighbors (KNN) | Distance-based non-parametric model |
| Naive Bayes | Fast probabilistic classifier |
| Gradient Boosting | Sequential ensemble method |
| XGBoost | Optimized distributed gradient boosting |
| LightGBM | High-performance leaf-wise boosting |

### Regression

| Algorithm | Notes |
|---|---|
| Linear Regression | Baseline linear model |
| Ridge Regression | L2-regularized linear model |
| Lasso Regression | L1-regularized, performs feature selection |
| Random Forest Regressor | Ensemble of decision trees |
| SVR | Support Vector Regression |
| Gradient Boosting Regressor | Sequential ensemble method |
| XGBoost Regressor | Optimized distributed gradient boosting |
| LightGBM Regressor | High-performance leaf-wise boosting |

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| API Framework | FastAPI |
| ML Library | Scikit-learn |
| Gradient Boosting | XGBoost, LightGBM |
| Hyperparameter Tuning | Optuna |
| Data Processing | Pandas, NumPy |
| Frontend | HTML5 / CSS3 / Vanilla JS |
| Serialization | Pickle (`.pkl`) |

---

## 📂 Project Structure

```
nexus-automl/
│
├── automl/                        # Core AutoML engine
│   ├── engine.py                  # Orchestrates the full AutoML pipeline
│   ├── data_cleaner.py            # Missing value imputation, type inference
│   ├── feature_engineering.py     # Encoding, scaling, feature creation
│   ├── model_trainer.py           # Trains all candidate models
│   ├── model_selector.py          # Evaluates and ranks trained models
│   ├── hyperparameter_tuner.py    # Optuna-based tuning for the best model
│   └── report_generator.py        # Generates the downloadable ML report
│
├── api.py                         # FastAPI application and route definitions
├── app.py                         # Streamlit UI (alternative interface)
├── automl-ui/                      # Custom HTML/CSS/JS web interface
│   ├── index.html
│   ├── Style.css
│   └── script.js
│
├── requirements.txt               # Python dependencies
└── README.md
```

---

## ▶️ Getting Started

### Prerequisites

- Python 3.9 or higher
- `pip` package manager

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Nexus-Automl.git
cd nexus-automl
```

### 2. Create a Virtual Environment

```bash
# Create environment
python -m venv .venv

# Activate — Windows
.venv\Scripts\activate

# Activate — macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI Server

```bash
uvicorn api:app --reload
```

The server will start at `http://127.0.0.1:8000`.

Visit the interactive API documentation at:

```
http://127.0.0.1:8000/docs
```

### 5. (Optional) Launch the Streamlit UI

```bash
streamlit run app.py
```

### 6. (Optional) Use the Custom HTML Frontend

Open `automl-ui/index.html` in your browser directly, or serve it with any static file server. Make sure the FastAPI server is running on port `8000`.

---

## 📡 API Reference

### `POST /automl`

Submits a dataset and target column for automated model training and evaluation.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | `File` | ✅ | CSV file containing the dataset |
| `target_column` | `string` | ✅ | Name of the column to predict |

**Example using `curl`:**

```bash
curl -X POST "http://127.0.0.1:8000/automl" \
  -F "file=@your_dataset.csv" \
  -F "target_column=target"
```

**Example Response — `200 OK`:**

```json
{
  "best_model": "Random Forest",
  "best_score": 0.9423,
  "models": {
    "Logistic Regression": 0.8971,
    "Random Forest": 0.9423,
    "SVM": 0.9105,
    "XGBoost": 0.9388,
    "LightGBM": 0.9401
  },
  "report": "AutoML Report\n==============\nDataset: 1500 rows × 12 columns\n..."
}
```

**Error Response — `400 Bad Request`:**

```json
{
  "error": "Target column 'price' not found in dataset."
}
```

---

## 📊 Supported Data Format

| Requirement | Detail |
|---|---|
| File format | CSV (`.csv`) |
| Data type | Tabular / structured data only |
| Target column | Must be a single column present in the file |
| Minimum rows | Recommended 100+ rows for reliable model training |
| Encoding | UTF-8 |

---

## ⚠️ Current Limitations

The following use cases are outside the current scope of Nexus AutoML:

- ❌ **Image and audio data** — computer vision and audio ML not supported
- ❌ **Natural language processing** — no text vectorization or transformer models
- ❌ **Time-series forecasting** — no temporal sequence modeling (ARIMA, LSTM, etc.)
- ❌ **Deep learning** — no neural network architectures
- ❌ **Streaming / real-time data** — batch processing only

---

## 🗺️ Roadmap

Planned improvements for future releases:

- [ ] **SHAP integration** — automatic feature importance explanations
- [ ] **Smarter model selection** — meta-learning-based algorithm recommendation
- [ ] **Deep learning support** — optional PyTorch/TensorFlow model candidates
- [ ] **Time-series module** — dedicated pipeline for forecasting tasks
- [ ] **Model versioning** — track and compare runs across experiments
- [ ] **Enhanced UI dashboard** — richer charts, confusion matrix, ROC curves
- [ ] **One-click deployment** — export model as a ready-to-serve FastAPI endpoint

---

## 👩‍💻 Author

**Pragati Mishra**
Full Stack Developer · AI/ML Enthusiast

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

If you find this project useful, please consider giving it a ⭐ on GitHub — it helps others discover the project.

</div>
