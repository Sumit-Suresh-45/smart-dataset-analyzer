import numpy as np
import pandas as pd
import plotly.express as px
from thefuzz import process

def generate_insights(df):
    insights = []
    
    # Missing values
    missing = df.isnull().sum()
    cols_with_missing = missing[missing > 0].index.tolist()
    if cols_with_missing:
        insights.append(f"Missing values detected in: {', '.join(cols_with_missing)}.")
        
    num_df = df.select_dtypes(include=np.number)
    cat_df = df.select_dtypes(exclude=np.number)
    
    # Business Insight: Largest numeric sum (e.g. Sales)
    if not num_df.empty:
        sums = num_df.sum().sort_values(ascending=False)
        top_col = sums.index[0]
        insights.append(f"'{top_col}' has the highest overall aggregate sum ({sums.iloc[0]:,.2f}).")
        
    # Correlation
    if not num_df.empty and num_df.shape[1] > 1:
        corr = num_df.corr().abs()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        corr_masked = corr.mask(mask)
        if not corr_masked.isna().all().all():
            max_corr_val = corr_masked.max().max()
            if int(max_corr_val) != 1 and max_corr_val > 0.5:
                col1, col2 = corr_masked.stack().idxmax()
                insights.append(f"Strong correlation ({max_corr_val:.2f}) between '{col1}' and '{col2}'.")
                
    # Categorical domination
    for col in cat_df.columns:
        if df[col].nunique() < len(df)/2: # Ensure it's somewhat categorical
            top_cat = df[col].value_counts().head(1)
            if not top_cat.empty:
                cat_name = top_cat.index[0]
                cat_count = top_cat.values[0]
                pc = (cat_count / len(df)) * 100
                if pc > 50:
                    insights.append(f"Category '{cat_name}' dominates '{col}' column ({pc:.1f}% of data).")
                    break 
                elif pc > 20:
                    insights.append(f"'{cat_name}' is the top performing/most frequent '{col}'.")
                    break
                    
    if not insights:
        insights.append("Dataset seems very uniform with no obvious dominant correlations or categories.")
        
    return insights

def parse_nlp_query(df, query):
    """
    Rule-based NLP chart generator supporting natural language patterns.
    Understands prepositions: over, by, across, vs, per, for
    """
    q = query.lower()
    # Strip stopwords / prepositions for matching purposes
    stopwords = ['of', 'over', 'by', 'across', 'per', 'for', 'the', 'in', 'show', 'me', 'vs', 'and', 'between', 'compare', 'region', 'with']
    
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    
    # Helper: fuzzy column match (partial name inside query) using thefuzz
    def find_cols_in_query(col_list):
        if not col_list: return []
        matched = []
        words = q.split()
        
        # Check individual query words against columns
        for word in words:
            if word in stopwords or len(word) < 3: continue
            res = process.extractOne(word, col_list)
            if res and res[1] >= 80:
                if res[0] not in matched:
                    matched.append(res[0])
                    
        # Check column names against entire query words for multi-word columns
        for c in col_list:
            if c not in matched:
                res = process.extractOne(c.lower(), words)
                if res and res[1] >= 85:
                    matched.append(c)
        return matched
    
    found_num = find_cols_in_query(num_cols)
    found_cat = find_cols_in_query(cat_cols)
    
    # ─── 1. "Top" queries ─────────────────────────────────────────────────────
    if 'top' in q or 'best' in q or 'highest' in q or 'most' in q:
        if found_cat:
            c_col = found_cat[0]
            if found_num:
                n_col = found_num[0]
                agg = df.groupby(c_col)[n_col].sum().reset_index().sort_values(by=n_col, ascending=False).head(10)
                fig = px.bar(agg, x=c_col, y=n_col, title=f"Top 10 {c_col} by {n_col}",
                             template='plotly_dark', color=n_col, color_continuous_scale='Blues')
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                return fig, f"Here are the top {c_col} ranked by {n_col}."
            else:
                val_counts = df[c_col].value_counts().reset_index().head(10)
                val_counts.columns = [c_col, 'Count']
                fig = px.bar(val_counts, x=c_col, y='Count', title=f"Top 10 {c_col} by Frequency",
                             template='plotly_dark', color='Count', color_continuous_scale='Blues')
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                return fig, f"Showing top {c_col} by frequency."
        # No cat, but numeric → just sort numeric
        elif found_num:
            n_col = found_num[0]
            top_df = df[[n_col]].dropna().sort_values(by=n_col, ascending=False).head(10).reset_index()
            fig = px.bar(top_df, x='index', y=n_col, title=f"Top 10 rows by {n_col}", template='plotly_dark')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            return fig, f"Showing top 10 records sorted by {n_col}."

    # ─── 2. Scatter / Correlation ──────────────────────────────────────────────
    if 'correlation' in q or 'scatter' in q or 'vs' in q or 'versus' in q or 'relationship' in q:
        if len(found_num) >= 2:
            color_arg = found_cat[0] if found_cat else None
            fig = px.scatter(df, x=found_num[0], y=found_num[1], color=color_arg,
                             title=f"{found_num[0]} vs {found_num[1]}",
                             template='plotly_dark', opacity=0.7)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            return fig, f"Scatter plot of {found_num[0]} vs {found_num[1]}."
        elif len(found_num) == 1 and found_cat:
            fig = px.box(df, x=found_cat[0], y=found_num[0],
                         title=f"{found_num[0]} by {found_cat[0]}",
                         template='plotly_dark', color=found_cat[0])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            return fig, f"Box plot of {found_num[0]} across {found_cat[0]}."

    # ─── 3. Distribution / Histogram ──────────────────────────────────────────
    if 'distribution' in q or 'histogram' in q or 'spread' in q:
        if found_num:
            if found_cat:
                fig = px.histogram(df, x=found_num[0], color=found_cat[0],
                                   title=f"Distribution of {found_num[0]} by {found_cat[0]}",
                                   template='plotly_dark', barmode='overlay', opacity=0.75)
                msg = f"Distribution of {found_num[0]} broken down by {found_cat[0]}."
            else:
                fig = px.histogram(df, x=found_num[0], title=f"Distribution of {found_num[0]}",
                                   template='plotly_dark')
                msg = f"Distribution of {found_num[0]}."
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            return fig, msg

    # ─── 4. Trend / Time series ────────────────────────────────────────────────
    if 'trend' in q or 'over time' in q or 'monthly' in q or 'daily' in q:
        date_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
        if not date_cols:
            # Try finding a col that might be date-like
            date_cols = [c for c in cat_cols if 'date' in c.lower() or 'month' in c.lower() or 'year' in c.lower() or 'time' in c.lower()]
        if date_cols and found_num:
            agg = df.groupby(date_cols[0])[found_num[0]].sum().reset_index()
            fig = px.line(agg, x=date_cols[0], y=found_num[0],
                          title=f"Trend of {found_num[0]} over {date_cols[0]}",
                          template='plotly_dark')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            return fig, f"Trend of {found_num[0]} over time."

    # ─── 5. Group-by bars ("sales by city", "revenue per product") ─────────────
    if found_num and found_cat:
        agg = df.groupby(found_cat[0])[found_num[0]].sum().reset_index().sort_values(by=found_num[0], ascending=False).head(15)
        fig = px.bar(agg, x=found_cat[0], y=found_num[0],
                     title=f"{found_num[0]} by {found_cat[0]}",
                     template='plotly_dark', color=found_num[0], color_continuous_scale='Blues')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return fig, f"Bar chart of {found_num[0]} grouped by {found_cat[0]}."

    # ─── 6. Fallback single-column detection ───────────────────────────────────
    if found_cat:
        c_col = found_cat[0]
        val_counts = df[c_col].value_counts().reset_index().head(15)
        val_counts.columns = [c_col, 'Count']
        fig = px.bar(val_counts, x=c_col, y='Count', title=f"Frequency of {c_col}",
                     template='plotly_dark', color='Count', color_continuous_scale='Blues')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return fig, f"Frequency chart for {c_col}."
    if found_num:
        n_col = found_num[0]
        fig = px.histogram(df, x=n_col, title=f"Distribution of {n_col}", template='plotly_dark')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return fig, f"Distribution of {n_col}."

    return None, ("❗ I couldn't understand that query. Try examples like:\n"
                  "- **show top products by sales**\n"
                  "- **distribution of sales by region**\n"
                  "- **sales vs quantity**\n"
                  "- **trend of revenue**")

