# Build an Automatic Dataset Analysis and Prediction System

Act as a Senior Data Scientist and Python Developer.

Build a system that can accept ANY dataset (CSV/Excel) and automatically:

- Analyze dataset structure
- Clean data automatically
- Generate insights
- Create visualizations
- Detect column types
- Build prediction model
- Show dashboard UI

The system must work dynamically for any dataset.

---

# 🎯 Core Features

## 1. Upload Any Dataset
User can upload:

- CSV file
- Excel file

After upload automatically:

- Show dataset preview
- Show shape (rows, columns)
- Show column names
- Show data types
- Show missing values

---

## 2. Automatic Column Detection

System should automatically detect:

Numerical Columns  
Categorical Columns  
Date Columns  
Target Column (optional selection)

Use:

- dtype detection
- unique value count
- datetime parsing

---

## 3. Automatic Data Cleaning

Perform:

- Remove duplicates
- Handle missing values
- Convert date columns
- Encode categorical variables
- Remove outliers (optional)

---

## 4. Automatic Insights Generation

System should generate:

### Statistical Summary
- mean
- median
- std
- min
- max

### Correlation Analysis
- Heatmap
- Correlation matrix

### Distribution Analysis
- Histogram
- Box plot

### Categorical Analysis
- Count plots
- Bar charts

---

## 5. Automatic Visualization Dashboard

Generate:

- Numeric column distributions
- Category comparisons
- Correlation heatmap
- Time series trend (if date column exists)
- Top categories
- Feature importance (if prediction enabled)

---

## 6. Automatic Prediction Model

User selects:

Target Column

System automatically:

- Detects regression or classification
- Splits train/test
- Trains model

If numeric target:
→ Linear Regression / Random Forest Regressor

If categorical target:
→ Logistic Regression / Random Forest Classifier

Show:

- Accuracy / R2 score
- Prediction chart
- Feature importance

---

## 7. Interactive Filters

Add sidebar filters:

- Select column
- Select category
- Select numeric range
- Date filter

---

## 8. UI Sections

Dashboard must include:

Home  
Dataset Overview  
Data Cleaning  
Exploratory Analysis  
Visualizations  
Prediction  
Insights Summary  

---

## 9. Technologies

Python  
Pandas  
NumPy  
Matplotlib  
Seaborn  
Scikit-learn  
Streamlit  

Optional:

Plotly  
AutoML  
SHAP  

---

## 10. Extra Smart Features

Add:

- Auto chart suggestions
- Best target suggestion
- Feature importance chart
- Download cleaned dataset
- Download predictions
- Natural language insights

Example:

"Sales increase in December"
"Product A has highest revenue"
"Column Age strongly impacts salary"

---

## 11. Folder Structure

auto_analyzer/
│
├── app.py
├── modules/
│   ├── loader.py
│   ├── cleaner.py
│   ├── analyzer.py
│   ├── visualization.py
│   ├── predictor.py
│   └── insights.py
│
└── requirements.txt

---

## 12. Expected Output

Generate:

- Fully working Streamlit app
- Automatic dataset detection
- Automatic insights
- Prediction model
- Interactive UI
- Clean modular code

Now generate the full system step-by-step.
Start with dataset upload module.