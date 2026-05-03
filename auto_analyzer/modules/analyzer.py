import pandas as pd
import numpy as np

def get_basic_info(df):
    mem_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
    missing_sum = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    
    missing_pct = (missing_sum / total_cells) * 100 if total_cells > 0 else 0
    duplicate_rows = df.duplicated().sum()
    
    # Calculate simple Data Quality Score roughly:
    # 100 - (missing% + (duplicates% / 2))
    completeness = 100 - missing_pct
    uniqueness = 100 - ((duplicate_rows / df.shape[0]) * 100 if df.shape[0] > 0 else 0)
    
    # Calculate Consistency score dynamically
    def check_mixed_types(col):
        if col.empty: return False
        return col.dropna().apply(lambda x: type(x).__name__).nunique() > 1
        
    inconsistent_cols = sum([1 for col in df.columns if check_mixed_types(df[col])])
    consistency = 100 - ((inconsistent_cols / df.shape[1]) * 100 if df.shape[1] > 0 else 0)
    
    dq_score = (completeness * 0.5) + (uniqueness * 0.3) + (consistency * 0.2)
    info = {
        "Shape": df.shape,
        "Total Rows": df.shape[0],
        "Total Columns": df.shape[1],
        "Missing Values": missing_sum,
        "Missing Percentage": round(missing_pct, 2),
        "Duplicated Rows": duplicate_rows,
        "Memory Usage (MB)": round(mem_usage, 2),
        "Data Quality Score": round(dq_score, 1)
    }
    return info

def get_column_types(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols and c not in date_cols]
    
    return {
        "numeric": numeric_cols,
        "categorical": cat_cols,
        "date": date_cols
    }

def get_data_types_df(df):
    types_df = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.astype(str),
        'Missing': df.isnull().sum().values,
        'Unique': df.nunique().values
    })
    return types_df
