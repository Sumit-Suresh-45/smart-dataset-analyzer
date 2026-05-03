import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def train_auto_model(df, target_col):
    if target_col not in df.columns:
        return {"error": "Target column not found"}
        
    # We must drop rows where the target value itself is missing
    df_clean = df.dropna(subset=[target_col]).copy()
    
    if df_clean.empty:
        return {"error": "All rows are missing the target value. Cannot train."}
    
    # Drop columns that are entirely null as they break imputers
    df_clean = df_clean.dropna(axis=1, how='all')
    
    if target_col not in df_clean.columns:
        return {"error": "Target column is entirely null. Cannot train."}
        
    # Drop datetime columns
    date_cols = [col for col in df_clean.select_dtypes(include=['datetime', 'datetime64', 'datetimetz']).columns if col != target_col]
    df_clean = df_clean.drop(columns=date_cols)
    
    # Identify column types
    num_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df_clean.select_dtypes(exclude=np.number).columns.tolist()
    
    # Drop high-cardinality categorical cols
    for col in cat_cols[:]:
        if col != target_col and df_clean[col].nunique() > 50:
            cat_cols.remove(col)
            df_clean = df_clean.drop(columns=[col])
            
    target_was_num = target_col in num_cols
    
    if target_col in num_cols:
        num_cols.remove(target_col)
    else:
        cat_cols.remove(target_col)
        df_clean[target_col] = df_clean[target_col].astype(str)
        
    # Label encoding for target if categorical
    target_le = None
    if target_col in cat_cols or df_clean[target_col].dtype == 'object':
        target_le = LabelEncoder()
        df_clean[target_col] = target_le.fit_transform(df_clean[target_col])
        
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    unique_vals = len(np.unique(y))
    is_classification = False
    
    if not target_was_num or unique_vals < 20: 
        is_classification = True
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocessing Pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ])
    
    models = {}
    if is_classification:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomizedSearchCV(RandomForestClassifier(random_state=42), {'n_estimators':[50, 100], 'max_depth':[None, 10, 20]}, cv=3, n_iter=3, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42)
        }
        score_func = accuracy_score
        score_metric = "Accuracy"
        model_type = "classification"
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomizedSearchCV(RandomForestRegressor(random_state=42), {'n_estimators':[50, 100], 'max_depth':[None, 10, 20]}, cv=3, n_iter=3, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, random_state=42)
        }
        score_func = r2_score
        score_metric = "R2 Score"
        model_type = "regression"
        
    results = []
    best_score = -float('inf')
    best_model_name = ""
    best_pipeline = None
    best_preds = None
    
    for name, model in models.items():
        try:
            clf = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', model)])
            
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            score = score_func(y_test, preds)
            results.append({"Model": name, "Score": round(score, 4)})
            
            if score > best_score:
                best_score = score
                best_model_name = name
                best_pipeline = clf
                best_preds = preds
        except Exception as e:
            results.append({"Model": name, "Score": "Error"})
            print(f"Error training {name}: {e}")
            
    if not best_model_name:
        return {"error": "All models failed to train automatically. This usually means the data format is incompatible after cleaning. Try dropping more missing values or excluding dates!"}
            
    comparison_df = pd.DataFrame(results).sort_values(by="Score", ascending=False, na_position='last')
        
    # Feature Importance extraction
    feat_imp_df = None
    try:
        if best_pipeline:
            # Extract final estimator
            final_estimator = best_pipeline.named_steps['classifier']
            if hasattr(final_estimator, 'best_estimator_'):
                final_estimator = final_estimator.best_estimator_
                
            # Get feature names from preprocessor
            ohe = best_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
            cat_feature_names = ohe.get_feature_names_out(cat_cols)
            all_features = num_cols + list(cat_feature_names)
            
            if hasattr(final_estimator, 'feature_importances_'):
                importances = final_estimator.feature_importances_
                feat_imp_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})
                feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(10)
            elif hasattr(final_estimator, 'coef_'):
                importances = np.abs(final_estimator.coef_[0] if len(final_estimator.coef_.shape)>1 else final_estimator.coef_)
                feat_imp_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})
                feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(10)
    except Exception as e:
        print(f"Feature importance error: {e}")
            
    return {
        "model_type": model_type,
        "score_metric": score_metric,
        "best_model_name": best_model_name,
        "best_score": best_score,
        "comparison_table": comparison_df,
        "feature_importance": feat_imp_df,
        "y_test": y_test.values if hasattr(y_test, 'values') else y_test,
        "y_pred": best_preds,
        "best_model": best_pipeline, # Now saving the full pipeline
        "features": X.columns.tolist(),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "target_le": target_le
    }

def manual_predict(input_data_dict, predictor_info):
    try:
        df_input = pd.DataFrame([input_data_dict])
        
        # Ensure correct column order
        X_input = df_input[predictor_info["features"]]
        
        # Best model is now the full pipeline
        pred = predictor_info["best_model"].predict(X_input)
        
        if predictor_info["target_le"] is not None:
            pred = predictor_info["target_le"].inverse_transform(pred.astype(int))
            
        return pred[0]
    except Exception as e:
        return f"Prediction Error: {e}"
