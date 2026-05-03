import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def clean_data(df, options):
    df_clean = df.copy()
    
    # 1. Smart Drop Useless Columns (IDs, constants, high nulls)
    if options.get('smart_drop', False):
        cols_to_drop = []
        for col in df_clean.columns:
            # Constant columns
            if df_clean[col].nunique() <= 1:
                cols_to_drop.append(col)
                continue
            # Unnamed/Index columns
            if 'unnamed: 0' in str(col).lower():
                cols_to_drop.append(col)
                continue
            # High Null columns (> 75%)
            if (df_clean[col].isnull().sum() / len(df_clean)) > 0.75:
                cols_to_drop.append(col)
                continue
            # ID Columns (High cardinality categorical)
            if df_clean[col].dtype == 'object' and df_clean[col].nunique() == len(df_clean) and len(df_clean) > 50:
                cols_to_drop.append(col)
                
        df_clean = df_clean.drop(columns=list(set(cols_to_drop)))
    
    # 2. Date Parsing
    if options.get('parse_dates', False):
        cat_cols = df_clean.select_dtypes(exclude=np.number).columns
        for col in cat_cols:
            if df_clean[col].astype(str).str.match(r'^\d{4}-\d{2}-\d{2}').any() or \
               df_clean[col].astype(str).str.match(r'^\d{2}/\d{2}/\d{4}').any():
                try:
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                except:
                    pass
    
    if options.get('drop_duplicates', False):
        df_clean = df_clean.drop_duplicates()
        
    if options.get('remove_nulls', False):
        df_clean = df_clean.dropna()
        
    if options.get('fill_mean', False):
        numeric_cols = df_clean.select_dtypes(include=np.number).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        
    if options.get('fill_median', False):
        numeric_cols = df_clean.select_dtypes(include=np.number).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        

    if options.get('remove_outliers', False):
        numeric_cols = df_clean.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            lower_bound = df_clean[col].quantile(0.01)
            upper_bound = df_clean[col].quantile(0.99)
            df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
            
    return df_clean
