import streamlit as st
import pandas as pd
import io

from modules.loader import load_data
from modules.analyzer import get_basic_info, get_column_types, get_data_types_df
from modules.cleaner import clean_data
from modules.visualization import plot_histogram, plot_box, plot_count, plot_pie, plot_correlation, plot_feature_importance, plot_predictions
from modules.predictor import train_auto_model, manual_predict
from modules.insights import generate_insights, parse_nlp_query
from modules.reporter import generate_business_report

st.set_page_config(page_title="Insightify AI", layout="wide", page_icon="🧠")

st.title("🧠 Insightify AI")
st.markdown("### Universal Dataset Analysis, Prediction & AI Insight System")

# Implement Caching for heavy tasks
@st.cache_data
def cached_load_data(uploaded_file):
    # Need to read the file into bytes so we can hash it reliably if it's an uploaded file object.
    # However, Streamlit's cache_data automatically hashes UploadedFile objects securely.
    return load_data(uploaded_file)

@st.cache_data
def cached_train_auto_model(df, target_col):
    return train_auto_model(df, target_col)

if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None
if 'clean_df' not in st.session_state:
    st.session_state.clean_df = None
if 'predictor_info' not in st.session_state:
    st.session_state.predictor_info = None

# Progress visualization in sidebar
st.sidebar.title("🧭 System Status")
st.sidebar.markdown("---")
if st.session_state.raw_df is not None: st.sidebar.success("✅ Dataset Loaded")
if st.session_state.clean_df is not None: st.sidebar.success("✅ Data Cleaned")
if st.session_state.predictor_info is not None: st.sidebar.success("✅ Model Trained")

# Main Navigation using Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. 📂 Upload & Overview", 
    "2. 🧹 Clean", 
    "3. 📊 Explore & Viz", 
    "4. 🤖 ML Forecast", 
    "5. 💡 AI Insights", 
    "6. 📄 Report & Download"
])

# ─── TAB 1: UPLOAD & OVERVIEW ──────────────────────────────────────────────────
with tab1:
    st.header("Upload your Dataset")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        df = cached_load_data(uploaded_file)
        if df is not None:
            st.session_state.raw_df = df
            if st.session_state.clean_df is None:
                 st.session_state.clean_df = df.copy()
            st.success("Dataset loaded successfully!")
            
    if st.session_state.raw_df is not None:
        df = st.session_state.raw_df
        st.write("### Dataset Preview:")
        st.dataframe(df.head(10))
        
        st.markdown("---")
        st.header("Dataset Overview & Quality")
        info = get_basic_info(df)
        
        st.metric("Data Quality Score", f"{info['Data Quality Score']}%")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", info["Total Rows"])
        col2.metric("Total Columns", info["Total Columns"])
        col3.metric("Memory Usage (MB)", f"{info['Memory Usage (MB)']} MB")
        col4.metric("Missing Vals %", f"{info['Missing Percentage']}%")
        
        st.write("### Complete Features")
        types_df = get_data_types_df(df)
        st.dataframe(types_df, use_container_width=True)

# ─── TAB 2: CLEANING ───────────────────────────────────────────────────────────
with tab2:
    st.header("Smart Data Preprocessing")
    if st.session_state.raw_df is not None:
        
        with st.expander("Advanced Preprocessing Engine", expanded=True):
            col_c1, col_c2 = st.columns(2)
            
            with col_c1:
                smart_drop = st.checkbox("Auto-Remove Useless Columns (IDs, >75% null, constants)", value=True)
                parse_dates = st.checkbox("Attempt Date Parsing", value=True)
                drop_dups = st.checkbox("Drop Duplicate Rows")
                remove_nulls = st.checkbox("Drop Rows with Any Nulls")
                
            with col_c2:
                fill_mean = st.checkbox("Fill Numeric Nulls with Mean")
                fill_median = st.checkbox("Fill Numeric Nulls with Median", value=True)
                remove_outliers = st.checkbox("Winsorize Extreme Outliers (1st/99th %tile)", value=True)
                
        if st.button("Apply Suggested Fixes"):
            options = {
                'smart_drop': smart_drop,
                'parse_dates': parse_dates,
                'drop_duplicates': drop_dups,
                'remove_nulls': remove_nulls,
                'fill_mean': fill_mean,
                'fill_median': fill_median,
                'remove_outliers': remove_outliers
            }
            cleaned = clean_data(st.session_state.raw_df, options)
            st.session_state.clean_df = cleaned
            st.success("Data Preprocessing Completed Successfully!")
            
        st.write("### Cleaned Output Snapshot:")
        st.dataframe(st.session_state.clean_df.head(10))
    else:
        st.warning("Please upload a dataset first.")

# ─── TAB 3: EXPLORE & VIZ ──────────────────────────────────────────────────────
with tab3:
    st.header("Exploratory Analysis & Visualizations")
    if st.session_state.clean_df is not None:
        df = st.session_state.clean_df
        types = get_column_types(df)
        
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Univariate", "Bivariate", "Correlation & Filters"])
        
        with viz_tab1:
            st.subheader("Single Variable Analysis")
            if types['numeric']:
                num_col = st.selectbox("Select Numeric Column", types['numeric'])
                st.write(df[num_col].describe())
                
                exclude_outliers = st.checkbox("Exclude Extreme Outliers from Charts (View Only)")
                plot_df = df.copy()
                if exclude_outliers:
                    Q1 = plot_df[num_col].quantile(0.25)
                    Q3 = plot_df[num_col].quantile(0.75)
                    IQR = Q3 - Q1
                    plot_df = plot_df[(plot_df[num_col] >= Q1 - 1.5 * IQR) & (plot_df[num_col] <= Q3 + 1.5 * IQR)]
                    
                st.plotly_chart(plot_histogram(plot_df, num_col), use_container_width=True)
                st.plotly_chart(plot_box(plot_df, num_col), use_container_width=True)
            if types['categorical']:
                cat_col = st.selectbox("Select Categorical Column", types['categorical'])
                st.plotly_chart(plot_count(df, cat_col), use_container_width=True)
                st.plotly_chart(plot_pie(df, cat_col), use_container_width=True)
                
        with viz_tab2:
            st.subheader("Cross-Variable Trends")
            num_cols = types['numeric']
            cat_cols = types['categorical']
            
            c1, c2 = st.columns(2)
            max_num = len(num_cols)
            x_ax = c1.selectbox("X-Axis (Numeric)", num_cols) if max_num > 0 else None
            y_ax = c2.selectbox("Y-Axis (Numeric)", num_cols, index=min(1, max_num-1)) if max_num > 0 else None
            
            if x_ax and y_ax:
                import plotly.express as px
                color_col = cat_cols[0] if cat_cols else None
                try:
                    fig = px.scatter(df, x=x_ax, y=y_ax, color=color_col, template='plotly_dark')
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not render scatter: {e}")
                    
        with viz_tab3:
            st.subheader("Data Filters & Correlation")
            if types['categorical']:
                filter_col = st.selectbox("Filter Field", ["None"] + types['categorical'])
                if filter_col != "None":
                    val = st.selectbox("Value", df[filter_col].unique())
                    st.dataframe(df[df[filter_col] == val].head(50))
                    
            st.markdown("---")
            st.write("### Correlation Heatmap")
            corr_fig = plot_correlation(df)
            if corr_fig:
                st.plotly_chart(corr_fig, use_container_width=True)
    else:
        st.warning("Please upload a dataset first.")

# ─── TAB 4: ML FORECAST ────────────────────────────────────────────────────────
with tab4:
    st.header("Auto ML Training & Prediction Engine")
    if st.session_state.clean_df is not None:
        df = st.session_state.clean_df
        target_col = st.selectbox("Select Target Column to Predict", df.columns.tolist())
        
        if st.button("Start Auto ML Training"):
            with st.spinner('Training Robust Pipelines (Tuning Hyperparameters)... This may take a minute.'):
                res = cached_train_auto_model(df, target_col)
                if "error" in res:
                    st.error(res["error"])
                else:
                    st.session_state.predictor_info = res
                    st.success(f"Best Model Found: {res['best_model_name']}!")
                    
        if st.session_state.predictor_info is not None:
            pi = st.session_state.predictor_info
            
            st.write(f"### Best Model: **{pi['best_model_name']}**")
            st.write(f"**{pi['score_metric']}:** {pi['best_score']:.4f}")
            
            st.write("#### Comparison Matrix")
            st.dataframe(pi['comparison_table'], use_container_width=True)
            
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                if pi.get('feature_importance') is not None:
                    st.plotly_chart(plot_feature_importance(pi['feature_importance']), use_container_width=True)
                else:
                    st.info("Feature importance not available for this model type.")
            with col_chart2:
                st.plotly_chart(plot_predictions(pi['y_test'], pi['y_pred'], pi['model_type']), use_container_width=True)
                
            st.markdown("---")
            st.subheader("Manual Inference Sandbox")
            st.write("Input values to test the model dynamically:")
            
            features = pi['features']
            input_dict = {}
            # Dynamically create input fields
            num_cols_for_input = pi["num_cols"]
            cat_cols_for_input = pi["cat_cols"]
            
            input_cols = st.columns(3)
            for i, f in enumerate(features):
                with input_cols[i % 3]:
                    if f in num_cols_for_input:
                        val = df[f].median() if pd.api.types.is_numeric_dtype(df[f]) else 0
                        input_dict[f] = st.number_input(f, value=float(val))
                    elif f in cat_cols_for_input:
                        known_classes = df[f].astype(str).unique().tolist()
                        input_dict[f] = st.selectbox(f, known_classes)
            
            if st.button("Predict"):
                prediction = manual_predict(input_dict, pi)
                st.success(f"**Predicted Result for {target_col}:** {prediction}")
    else:
        st.warning("Please upload a dataset first.")

# ─── TAB 5: AI INSIGHTS ────────────────────────────────────────────────────────
with tab5:
    st.header("🧠 AI Auto Insights & Semantic NLP Queries")
    if st.session_state.clean_df is not None:
        df = st.session_state.clean_df
        
        st.subheader("Ask Insightify AI (Fuzzy Engine Active)")
        q = st.text_input("Type a natural language query (e.g., 'show top products by sales', 'distribution of age'):")
        if q:
            try:
                fig, msg = parse_nlp_query(df, q)
                st.info(f"{msg}")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error parsing NLP query: {e}")
                
        st.markdown("---")
        st.subheader("Automated Business & Data Insights")
        insights = generate_insights(df)
        for i, ins in enumerate(insights):
            st.markdown(f"> **{i+1}.** {ins}")
    else:
        st.warning("Please upload a dataset first.")

# ─── TAB 6: REPORT & DOWNLOAD ──────────────────────────────────────────────────
with tab6:
    st.header("📄 Auto Business Report & Download Center")
    if st.session_state.clean_df is not None:
        if st.button("Generate One-Click Report"):
            with st.spinner("Compiling Insights..."):
                info = get_basic_info(st.session_state.raw_df)
                insights = generate_insights(st.session_state.clean_df)
                report_md = generate_business_report(st.session_state.clean_df, info, insights, st.session_state.predictor_info)
                st.session_state.last_report = report_md
                st.success("Report Generated!")
            
        if 'last_report' in st.session_state:
            with st.expander("View Full Report", expanded=True):
                st.markdown(st.session_state.last_report)
                
            col_d1, col_d2, col_d3 = st.columns(3)
            
            with col_d1:
                csv = st.session_state.clean_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Cleaned Dataset (CSV)",
                    data=csv,
                    file_name="InsightifyAI_Cleaned.csv",
                    mime="text/csv"
                )
            
            with col_d2:
                st.download_button(
                    label="📥 Download Business Report (TXT)",
                    data=st.session_state.last_report.encode('utf-8'),
                    file_name="InsightifyAI_Report.txt",
                    mime="text/plain"
                )
                
            with col_d3:
                if st.session_state.predictor_info is not None:
                    pi = st.session_state.predictor_info
                    if pi['y_pred'] is not None:
                        preds_df = pd.DataFrame({'Actual': pi['y_test'], 'Predicted': pi['y_pred']})
                        preds_csv = preds_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Test Predictions (CSV)",
                            data=preds_csv,
                            file_name="InsightifyAI_Predictions.csv",
                            mime="text/csv"
                        )
    else:
        st.warning("Please upload a dataset first.")
