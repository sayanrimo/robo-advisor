# -------- BLOCK 10: Dash App (Polished UI) --------
# Save the following content to app.py and run with `python app.py`

import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import os

# Load model
with open("finalized_model.sav", "rb") as f:
    risk_model = pickle.load(f)

# Synthetic returns (replace with real data)
symbols = ["GOOGL", "FB", "GS", "MS", "GE", "MSFT"]
dates = pd.date_range(end=datetime.today(), periods=252)
np.random.seed(42)
returns_df = pd.DataFrame(np.random.randn(252, len(symbols)) * 0.01,
                          index=dates, columns=symbols)

# Mean-variance optimizer
def mean_variance_weights(returns, risk_tolerance):
    mu = returns.mean()
    Sigma = returns.cov()
    inv_sigma = np.linalg.pinv(Sigma.values)
    w = inv_sigma.dot(mu) * risk_tolerance
    w = np.maximum(w, 0)
    return w / w.sum()

# Build app
dash_app = dash.Dash(__name__)
server = dash_app.server

dash_app.layout = html.Div([
    html.H1("Robo Advisor Dashboard", style={'textAlign': 'center'}),

    html.Div([
        html.Div([
            html.H3("Step 1 : Enter Investor Characteristics", style={'backgroundColor': '#e6e6e6'}),
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

            html.Button("CALCULATE RISK TOLERANCE", id='btn_rt', style={'marginTop': '10px', 'backgroundColor': '#555', 'color': 'white'})
        ], className='three columns', style={'padding': '20px'}),

        html.Div([
            html.H3("Step 2 : Asset Allocation and portfolio performance", style={'backgroundColor': 'black', 'color': 'white'}),
            html.Label("Risk Tolerance (scale of 100) :"),
            html.Div(id='output_rt', style={'marginBottom': '10px'}),

            html.Label("Select the assets for the portfolio:"),
            dcc.Dropdown(id='assets', options=[{'label': s, 'value': s} for s in symbols],
                         value=symbols[:4], multi=True),

            html.Button("SUBMIT", id='btn_alloc', style={'marginTop': '10px', 'backgroundColor': '#555', 'color': 'white'}),

            html.Div([
                dcc.Graph(id='alloc_graph'),
                dcc.Graph(id='perf_graph')
            ], style={'marginTop': '20px'})
        ], className='nine columns', style={'padding': '20px'})
    ], className='row')
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
    rt = risk_model.predict(feat)[0] * 25  # Scale to 100
    return f"{rt:.2f}"

@dash_app.callback(
    [Output('alloc_graph', 'figure'), Output('perf_graph', 'figure')],
    Input('btn_alloc', 'n_clicks'),
    [Input('assets', 'value'), Input('output_rt', 'children')]
)
def allocate_plot(n, assets, rt_text):
    if not n or not assets or not rt_text:
        return {}, {}
    rt = float(rt_text)
    sub = returns_df[assets]
    w = mean_variance_weights(sub, rt)
    alloc_fig = go.Figure(data=[go.Bar(x=assets, y=w, marker=dict(color='red'))])
    alloc_fig.update_layout(title='Asset allocation - Mean-Variance Allocation', yaxis_title='Weight')
    perf = (sub * w).sum(axis=1)
    cum = (1 + perf).cumprod() * 100
    perf_fig = go.Figure(data=[go.Scatter(x=cum.index, y=cum.values, line=dict(color='red'))])
    perf_fig.update_layout(title='Portfolio value of $100 investment', xaxis_title='Date', yaxis_title='Value')
    return alloc_fig, perf_fig

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    dash_app.run(host="0.0.0.0", port=port, debug=True)


# Overwrite app.py
with open("app.py", "w") as f:
    f.write(app_code)

print(" app.py polished and updated for deployment.")
