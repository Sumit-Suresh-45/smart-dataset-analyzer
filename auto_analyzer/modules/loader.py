import streamlit as st
import pandas as pd

def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload CSV or Excel.")
                return None
                
            # Automatically drop index columns like 'Unnamed: 0' globally at load time
            df = df.loc[:, ~df.columns.str.contains('^Unnamed', case=False, na=False)]
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    return None
