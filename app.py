import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# Load model
with open("finalized_model.sav", "rb") as f:
    risk_model = pickle.load(f)

# Synthetic returns (replace with real data)
symbols = ["AAPL","MSFT","GOOGL","AMZN","TSLA"]
dates = pd.date_range(end=datetime.today(), periods=252)
np.random.seed(42)
returns_df = pd.DataFrame(np.random.randn(252, len(symbols))*.01,
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
    html.H1("Robo-Advisor Dashboard"),
    html.Div([
        html.Label("Age (1-6)"), dcc.Input(id='age', type='number', value=3),
        html.Label("Education (1-4)"), dcc.Input(id='edcl', type='number', value=2),
        html.Label("Married (1-2)"), dcc.Input(id='married', type='number', value=1),
        html.Label("Kids (#)"), dcc.Input(id='kids', type='number', value=1),
        html.Label("Occupation (1-4)"), dcc.Input(id='occat1', type='number', value=2),
        html.Label("Income (1-5)"), dcc.Input(id='income', type='number', value=3),
        html.Label("Willingness to Risk (1-4)"), dcc.Input(id='risk', type='number', value=2),
        html.Label("Net Worth (1-5)"), dcc.Input(id='networth', type='number', value=3),
        html.Button('Calculate RT', id='btn_rt'),
        html.Div(id='output_rt')
    ], style={'padding':'20px','border':'1px solid #ccc'}),
    html.Div([
        html.Label("Select Assets"),
        dcc.Dropdown(id='assets', options=[{'label':s,'value':s} for s in symbols], value=symbols[:3], multi=True),
        html.Label("Initial Capital"), dcc.Input(id='capital', type='number', value=10000),
        html.Button('Allocate & Plot', id='btn_alloc'),
        dcc.Graph(id='alloc_graph'),
        dcc.Graph(id='perf_graph')
    ], style={'padding':'20px','border':'1px solid #ccc','marginTop':'20px'})
])

@dash_app.callback(
    Output('output_rt','children'),
    Input('btn_rt','n_clicks'),
    [Input('age','value'), Input('edcl','value'), Input('married','value'),
     Input('kids','value'), Input('occat1','value'), Input('income','value'),
     Input('risk','value'), Input('networth','value')]
)
def predict_rt(n, age, edcl, married, kids, occat1, income, risk, networth):
    if not n:
        return ''
    feat = np.array([[age, edcl, married, kids, occat1, income, risk, networth]])
    rt = risk_model.predict(feat)[0]
    return f"Predicted Risk Tolerance: {rt:.3f}"

@dash_app.callback(
    [Output('alloc_graph','figure'), Output('perf_graph','figure')],
    Input('btn_alloc','n_clicks'),
    [Input('assets','value'), Input('capital','value'), Input('output_rt','children')]
)
def allocate_plot(n, assets, cap, rt_text):
    if not n or not assets or not rt_text:
        return {}, {}
    rt = float(rt_text.split()[-1])
    sub = returns_df[assets]
    w = mean_variance_weights(sub, rt)
    alloc_fig = go.Figure(data=[go.Bar(x=assets, y=w)])
    alloc_fig.update_layout(title='Portfolio Allocation', yaxis_title='Weight')
    perf = (sub * w).sum(axis=1)
    cum = (1+perf).cumprod() * cap
    perf_fig = go.Figure(data=[go.Scatter(x=cum.index, y=cum.values)])
    perf_fig.update_layout(title='Portfolio Value Over Time', xaxis_title='Date', yaxis_title='Value')
    return alloc_fig, perf_fig

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    dash_app.run(host="0.0.0.0", port=port, debug=True)


'''
# Write app.py file
txt = open('app.py','w')
txt.write(app_code)
txt.close()
print("Pipeline and app.py created successfully!")
'''