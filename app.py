# -------- BLOCK 10: Dash App (Polished UI with SHAP Explainability) --------
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import os
import shap
import base64
import matplotlib.pyplot as plt
from io import BytesIO
import dash_bootstrap_components as dbc

# Load model
with open("finalized_model.sav", "rb") as f:
    risk_model = pickle.load(f)

# Synthetic returns (replace with real data)
symbols = ["GOOGL", "FB", "GS", "MS", "GE", "MSFT"]
dates = pd.date_range(end=datetime.today(), periods=252)
np.random.seed(42)
returns_df = pd.DataFrame(np.random.randn(252, len(symbols)) * 0.01,
                          index=dates, columns=symbols)

# SHAP Explainer (Tree-based)
explainer = shap.TreeExplainer(risk_model)
sample_data = pd.DataFrame(
    np.array([[30, 2, 1, 1, 2, 2, 2, 2]]),
    columns=["age", "edcl", "married", "kids", "occat1", "income", "risk", "networth"]
)
shap_values = explainer.shap_values(sample_data)
plt.figure()
shap.summary_plot(shap_values, sample_data, plot_type="bar", show=False)
buf = BytesIO()
plt.savefig(buf, format="png", bbox_inches="tight")
buf.seek(0)
encoded_image = base64.b64encode(buf.read()).decode('utf-8')
plt.close()

# Mean-variance optimizer
def mean_variance_weights(returns, risk_tolerance):
    mu = returns.mean()
    Sigma = returns.cov()
    inv_sigma = np.linalg.pinv(Sigma.values)
    w = inv_sigma.dot(mu) * risk_tolerance
    w = np.maximum(w, 0)
    return w / w.sum()

# Build app
dash_app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
server = dash_app.server

dash_app.layout = dbc.Container([
    html.H1("Robo Advisor Dashboard", style={'textAlign': 'center'}),
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Dashboard', value='tab-1'),
        dcc.Tab(label='Model Info', value='tab-2')
    ]),
    html.Div(id='tabs-content')
], fluid=True)

@dash_app.callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return dbc.Row([
            dbc.Col([
                html.H4("Step 1: Enter Investor Characteristics"),
                html.Label("Age:"),
                dcc.Slider(18, 65, 5, value=30, marks=None, tooltip={"placement": "bottom"}, id='age'),
                html.Label("NetWorth:"),
                dcc.Slider(0, 5, 1, value=2, marks={0: '-$1M', 1: '0', 2: '$500K', 3: '$1M', 4: '$2M', 5: '$5M'}, id='networth'),
                html.Label("Income:"),
                dcc.Slider(0, 5, 1, value=2, marks={0: '-$1M', 1: '0', 2: '$500K', 3: '$1M', 4: '$2M', 5: '$5M'}, id='income'),
                html.Label("Education Level (scale of 4):"),
                dcc.Slider(1, 4, 1, value=2, marks=None, tooltip={"placement": "bottom"}, id='edcl'),
                html.Label("Married:"),
                dcc.Slider(1, 2, 1, value=1, marks={1: 'No', 2: 'Yes'}, id='married'),
                html.Label("Kids:"),
                dcc.Slider(0, 7, 1, value=1, marks=None, tooltip={"placement": "bottom"}, id='kids'),
                html.Label("Occupation:"),
                dcc.Slider(1, 4, 1, value=2, marks=None, tooltip={"placement": "bottom"}, id='occat1'),
                html.Label("Willingness to take Risk:"),
                dcc.Slider(1, 4, 1, value=2, marks=None, tooltip={"placement": "bottom"}, id='risk'),
                html.Br(),
                html.Button("CALCULATE RISK TOLERANCE", id='btn_rt', style={'backgroundColor': '#2a9df4', 'color': 'white'})
            ], width=4),
            dbc.Col([
                html.H4("Step 2: Asset Allocation and Portfolio Performance"),
                html.Div(id='output_rt'),
                html.Label("Select Assets:"),
                dcc.Dropdown(id='assets', options=[{'label': s, 'value': s} for s in symbols],
                             value=symbols[:4], multi=True),
                html.Br(),
                html.Button("SUBMIT", id='btn_alloc', style={'backgroundColor': '#2a9df4', 'color': 'white'}),
                dcc.Graph(id='alloc_graph'),
                dcc.Graph(id='perf_graph')
            ], width=8)
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H4("Model Explainability with SHAP"),
            html.P("The model uses a Random Forest to predict risk tolerance. Below is the feature importance visualized using SHAP (SHapley values)."),
            html.Img(src=f"data:image/png;base64,{encoded_image}", style={"width": "100%", "height": "auto"})
        ])

@dash_app.callback(
    Output('output_rt', 'children'),
    Input('btn_rt', 'n_clicks'),
    [Input('age', 'value'), Input('edcl', 'value'), Input('married', 'value'),
     Input('kids', 'value'), Input('occat1', 'value'), Input('income', 'value'),
     Input('risk', 'value'), Input('networth', 'value')]
)
def predict_rt(n, age, edcl, married, kids, occat1, income, risk, networth):
    if not n:
        return ''
    feat = np.array([[age, edcl, married, kids, occat1, income, risk, networth]])
    rt = risk_model.predict(feat)[0] * 25
    return f"Risk Tolerance Score: {rt:.2f}"

@dash_app.callback(
    [Output('alloc_graph', 'figure'), Output('perf_graph', 'figure')],
    Input('btn_alloc', 'n_clicks'),
    [Input('assets', 'value'), Input('output_rt', 'children')]
)
def allocate_plot(n, assets, rt_text):
    if not n or not assets or not rt_text:
        return {}, {}
    try:
        rt = float(rt_text.split(":")[-1])
    except:
        return {}, {}
    sub = returns_df[assets]
    w = mean_variance_weights(sub, rt)
    alloc_fig = go.Figure(data=[go.Bar(x=assets, y=w, marker=dict(color='red'))])
    alloc_fig.update_layout(title='Asset Allocation - Mean-Variance', yaxis_title='Weight')
    perf = (sub * w).sum(axis=1)
    cum = (1 + perf).cumprod() * 100
    perf_fig = go.Figure(data=[go.Scatter(x=cum.index, y=cum.values, line=dict(color='red'))])
    perf_fig.update_layout(title='Portfolio Value of $100 Investment', xaxis_title='Date', yaxis_title='Value')
    return alloc_fig, perf_fig

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    dash_app.run_server(host="0.0.0.0", port=port, debug=True)
