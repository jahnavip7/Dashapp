
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import os
from datetime import datetime

models_dir = "model_data/"

if not os.path.exists(models_dir):
    raise FileNotFoundError(f"Directory not found: {models_dir}")

model_files = [f for f in os.listdir(models_dir) if f.endswith(".csv")]
if not model_files:
    raise ValueError(f"No CSV files found in {models_dir}. Please add CSV files.")

weekly_data = {}
daily_data = {}

for file in model_files:
    model_name = file.replace(".csv", "")
    df = pd.read_csv(os.path.join(models_dir, file))
    df["dates"] = pd.to_datetime(df["dates"], format='%m/%d/%y')
    df = df.sort_values("dates")

    weekly_data[model_name] = df

    full_date_range = pd.date_range(start=df["dates"].min(), end=df["dates"].max(), freq="D")
    df_full = pd.DataFrame({"dates": full_date_range})
    df_merged = pd.merge(df_full, df, on="dates", how="left")
    if "ground_truth" in df.columns:
        df_merged["ground_truth"] = df_merged["ground_truth"].interpolate(method="linear")
    if "predictions" in df.columns:
        df_merged["predictions"] = df_merged["predictions"].interpolate(method="linear")
    elif "predicted values" in df.columns:
        df_merged["predicted values"] = df_merged["predicted values"].interpolate(method="linear")
    daily_data[model_name] = df_merged

weekly_date_list = pd.date_range(
    start=min(df["dates"].min() for df in weekly_data.values()),
    end=max(df["dates"].max() for df in weekly_data.values()),
    freq="W"
)

daily_date_list = pd.date_range(
    start=min(df["dates"].min() for df in daily_data.values()),
    end=max(df["dates"].max() for df in daily_data.values()),
    freq="D"
)


app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Time-Series Forecast Dashboard", style={'textAlign': 'center', 'marginBottom': 30}),

    dcc.RadioItems(
        id="view-toggle",
        options=[{"label": "Weekly Forecast", "value": "weekly"},
                 {"label": "Daily Forecast", "value": "daily"}],
        value="weekly",
        labelStyle={'display': 'inline-block', 'marginRight': '20px'},
        style={"marginBottom": "20px", "marginLeft": "40px"}
    ),

    html.Div([
        html.Div([
            html.H3("Select Models:"),
            dcc.Checklist(
                id="model-selection",
                options=[{"label": model.replace("2results_v14_", "").replace("results-csv_", "").replace("result-csv_", ""),
                          "value": model} for model in model_files],
                value=[],
                inline=False,
                style={'fontSize': '16px', 'lineHeight': '2'}
            )
        ], style={"width": "20%", "padding": "20px", "backgroundColor": "#f8f9fa",
                  "borderRadius": "10px", "marginRight": "20px"}),

        html.Div([
            dcc.Graph(id="time-series-graph", style={'height': '600px'}),
            html.Div([
                html.Label("Adjust Date Range"),
                dcc.RangeSlider(id="date-range-slider")
            ], style={'marginTop': '20px', 'padding': '20px'})
        ], style={"width": "75%"})
    ], style={"display": "flex", "margin": "20px"})
])

@app.callback(
    [Output("date-range-slider", "min"),
     Output("date-range-slider", "max"),
     Output("date-range-slider", "value"),
     Output("date-range-slider", "marks")],
    Input("view-toggle", "value")
)
def update_slider(view_type):
    date_list = weekly_date_list if view_type == "weekly" else daily_date_list
    min_val = 0
    max_val = len(date_list) - 1
    value = [min_val, max_val]
    marks = {
        min_val: {'label': date_list[0].strftime('%Y-%m-%d')},
        max_val: {'label': date_list[-1].strftime('%Y-%m-%d')}
    }
    return min_val, max_val, value, marks

@app.callback(
    Output("time-series-graph", "figure"),
    [Input("model-selection", "value"),
     Input("date-range-slider", "value"),
     Input("view-toggle", "value")]
)
def update_graph(selected_models, slider_range, view_type):
    try:
        data_source = weekly_data if view_type == "weekly" else daily_data
        date_list = weekly_date_list if view_type == "weekly" else daily_date_list

        if not slider_range or len(slider_range) < 2:
            return go.Figure()

        start_date = date_list[slider_range[0]]
        end_date = date_list[slider_range[1]]

        fig = go.Figure()
        ground_truth_plotted = False
        all_values = []

        for model in selected_models:
            df = data_source[model.replace(".csv", "")]
            df_filtered = df[(df["dates"] >= start_date) & (df["dates"] <= end_date)]

            if not ground_truth_plotted and "ground_truth" in df_filtered.columns:
                all_values.extend(df_filtered["ground_truth"].dropna().tolist())
                fig.add_trace(go.Scatter(
                    x=df_filtered["dates"],
                    y=df_filtered["ground_truth"],
                    mode="lines",
                    name="Actual Values",
                    line=dict(color='black', width=2, shape='spline')
                ))
                ground_truth_plotted = True

            pred_col = "predictions" if "predictions" in df_filtered.columns else (
                "predicted values" if "predicted values" in df_filtered.columns else None)

            if pred_col:
                display_name = model.replace("2results_v14_", "").replace("results-csv_", "").replace("result-csv_", "")
                all_values.extend(df_filtered[pred_col].dropna().tolist())
                fig.add_trace(go.Scatter(
                    x=df_filtered["dates"],
                    y=df_filtered[pred_col],
                    mode="lines",
                    name=f"{display_name}",
                    line=dict(dash='dash', width=2, shape='spline')
                ))

        if all_values:
            y_min = 0
            y_max = ((max(all_values) // 2000) + 1) * 2000
            fig.update_layout(
                title={
                    'text': f"Smooth Time-Series Data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=20)
                },
                xaxis_title="Date",
                yaxis_title="Value",
                template="plotly_white",
                hovermode="x unified",
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    font=dict(size=12)
                ),
                margin=dict(l=50, r=50, t=80, b=50),
                showlegend=True,
                plot_bgcolor='white',
                yaxis=dict(range=[y_min, y_max], tickmode="linear", dtick=2000)
            )
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

        return fig

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        fig = go.Figure()
        fig.update_layout(title={'text': f"Error loading data: {str(e)}", 'x': 0.5, 'xanchor': 'center'})
        return fig

if __name__ == "__main__":
    app.run(debug=True, port=8051)



