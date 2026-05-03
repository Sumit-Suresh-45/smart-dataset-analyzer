# 📊 Smart Dataset Analyzer (Insightify AI)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Data%20Analysis-Pandas-150458.svg)](https://pandas.pydata.org/)

**Smart Dataset Analyzer** is an end-to-end automated data science web application that transforms raw datasets into actionable insights and predictive models without requiring a single line of code. 

Designed to act as your personal "Senior Data Scientist," the system accepts any CSV or Excel file and instantly performs intelligent data cleaning, generates interactive visualizations, extracts natural language insights, and trains machine learning models. 

---

## 🎯 Core Features

### 1. Universal Data Ingestion
- **Formats Supported**: Instantly upload `.csv` or `.xlsx` files.
- **Auto-Profiling**: The moment data is uploaded, the system generates a comprehensive overview including dataset shape, column names, missing value counts, and data type breakdowns.

### 2. Intelligent Data Cleaning
- **Automatic Imputation**: Handles missing values smartly (e.g., mean/median for numerical data, mode for categorical).
- **Type Conversion**: Automatically parses datetime columns and correctly casts numeric vs. string data.
- **Deduplication**: Identifies and removes duplicate rows to ensure dataset integrity.
- **Categorical Encoding**: Converts categorical labels into machine-readable formats for modeling.

### 3. Automated Exploratory Data Analysis (EDA)
- **Statistical Summaries**: Instantly calculates mean, median, standard deviation, min, and max for all numeric features.
- **Interactive Visualizations**: 
  - *Correlation Heatmaps* to identify feature relationships.
  - *Histograms & Box Plots* for distribution analysis and outlier detection.
  - *Count Plots & Bar Charts* for categorical variable comparisons.

### 4. Auto-ML & Predictive Modeling
- **Target Selection**: Users simply choose a target variable from a dropdown.
- **Smart Task Detection**: The system automatically determines if the problem requires **Classification** (categorical target) or **Regression** (numerical target).
- **Model Training**: Splits data into training/testing sets and applies algorithms like Random Forest, Logistic Regression, or Linear Regression.
- **Evaluation Metrics**: Displays real-time Accuracy scores, R² scores, and visualizes Feature Importance to explain *how* the model is making decisions.

### 5. Dynamic Filtering & UI
- **Sidebar Controls**: Slice and dice your data using interactive sidebar filters (select columns, set numeric ranges, or filter by specific categories).
- **Modular Dashboard**: Navigate seamlessly between Home, Data Overview, Cleaning, EDA, Visualizations, and Predictions.

---

## 🏗️ Project Architecture

The application follows a clean, modular architecture to ensure scalability and maintainability:

```text
auto_analyzer/
│
├── app.py                     # Main Streamlit application entry point
├── requirements.txt           # Project dependencies
│
└── modules/                   # Core logic separated by concern
    ├── loader.py              # Handles file uploading and parsing
    ├── cleaner.py             # Data preprocessing and imputation
    ├── analyzer.py            # Statistical analysis and EDA logic
    ├── visualization.py       # Matplotlib & Seaborn chart generation
    ├── predictor.py           # Auto-ML pipeline and model evaluation
    └── insights.py            # Natural language insight generation
