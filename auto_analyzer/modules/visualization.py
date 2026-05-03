import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_histogram(df, column):
    fig = px.histogram(df, x=column, title=f'Distribution of {column}', marginal='box', template='plotly_dark')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def plot_box(df, column):
    fig = px.box(df, y=column, title=f'Box Plot of {column}', template='plotly_dark')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def plot_count(df, column):
    fig = px.histogram(df, x=column, title=f'Count of {column}', template='plotly_dark')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def plot_pie(df, column):
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, 'count']
    fig = px.pie(counts, values='count', names=column, title=f'Proportion of {column}', template='plotly_dark')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def plot_correlation(df):
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        return None
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap", template='plotly_dark')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def plot_feature_importance(importance_df):
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title='Feature Importance', template='plotly_dark')
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def plot_predictions(y_test, y_pred, model_type):
    if model_type == 'regression':
        fig = px.scatter(x=y_test, y=y_pred, labels={'x':'Actual', 'y':'Predicted'}, title='Actual vs Predicted', template='plotly_dark')
        # Add equality line
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color="Red", dash="dash"))
    else:
        # For classification, we can plot a confusion matrix 
        # But we would need sklearn.metrics.confusion_matrix, let's keep it simple with a bar chart of predictions
        import pandas as pd
        res = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        acc_counts = (res['Actual'] == res['Predicted']).value_counts().reset_index()
        acc_counts.columns = ['Correct', 'Count']
        acc_counts['Correct'] = acc_counts['Correct'].map({True: 'Correct', False: 'Incorrect'})
        fig = px.bar(acc_counts, x='Correct', y='Count', color='Correct', title='Prediction Accuracy Count', template='plotly_dark')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig
