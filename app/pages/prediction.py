import dash
from dash import html, dcc, Input, Output, State, callback
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("code/los_model")  # adjust path
# Load real profit per day mapping
profit_map = joblib.load("code/profit_map.pkl")

# Example rare code sets and encoders (replace with actual from training data)
rare_drg_set = {999, 888}  # example
rare_mdc_set = {77, 88}  # example
global_los_mean = 5.0  # fallback for missing encoding

# Dummy DRG/MDC encoders (replace with your actual logic)
def compute_oof_avg(code, mapping):
    return mapping.get(code, global_los_mean)

def compute_smooth_enc(code):
    return np.log1p(code % 100)  # placeholder logic

def estimate_profit_per_day(admission_type):
    feature_to_label = {
        "Admission_Emergency": "Emergency",
        "Admission_Elective": "Elective",
        "Admission_Urgent": "Urgent"
    }
    if not isinstance(admission_type, list):
        admission_type = [admission_type]
    mapped = [feature_to_label.get(a) for a in admission_type if a in feature_to_label]
    profits = [profit_map.get(adm, 0) for adm in mapped]
    return np.mean(profits) if profits else 0

def get_charge_and_cost_per_day(admission_type):
    feature_to_label = {
        "Admission_Emergency": "Emergency",
        "Admission_Elective": "Elective",
        "Admission_Urgent": "Urgent"
    }
    if not isinstance(admission_type, list):
        admission_type = [admission_type]
    mapped = [feature_to_label.get(a) for a in admission_type if a in feature_to_label]
    charges = [profit_map.get(adm, 0) + 100 for adm in mapped]  # placeholder
    costs = [100 for adm in mapped]  # placeholder
    return np.mean(charges), np.mean(costs)

def get_recommendations(los, charge_total):
    tips = []

    if los < 3:
        tips.extend([
            "Prepare for early discharge: Arrange home care or outpatient follow-up to ensure smooth transition.",
            "Optimize resource allocation: Prioritize beds and nursing care for higher-need patients.",
            "Monitor closely for readmission risk: Short stays may risk premature discharge; ensure appropriate monitoring."
        ])
    elif 3 <= los <= 7:
        tips.extend([
            "Standard care pathway: Follow established protocols, with periodic assessment for any deviations.",
            "Plan for potential complications: Allocate resources anticipating medium-level complexity.",
            "Engage multidisciplinary teams: Include rehab, social work, and discharge planners early."
        ])
    else:
        tips.extend([
            "Initiate intensive case management: Focus on complex care coordination and multidisciplinary approach.",
            "Consider transfer to specialized care units: e.g., rehabilitation or long-term care.",
            "Review for potential complications: Early detection and management to reduce further prolongation.",
            "Financial counseling: Inform patients/families about possible cost implications."
        ])

    if charge_total > 10000:
        tips.extend([
            "Evaluate treatment plans: Look for cost-effective alternatives without compromising quality.",
            "Leverage insurance or financial aid options: To reduce patient burden.",
            "Optimize resource use: Reduce unnecessary tests or procedures."
        ])

    return tips


disease_options = [
    {'label': 'Mental Health', 'value': 'Disease_Mental Health'},
    {'label': 'Respiratory Disorders', 'value': 'Disease_Respiratory Disorders'},
    {'label': 'Infectious Diseases', 'value': 'Disease_Infectious Diseases'},
    {'label': 'Neurological Disorders', 'value': 'Disease_Neurological Disorders'},
    {'label': 'Chronic Conditions', 'value': 'Disease_Chronic Conditions'},
    {'label': 'Musculoskeletal Disorders', 'value': 'Disease_Musculoskeletal Disorders'},
    {'label': 'General Symptoms', 'value': 'Disease_General Symptoms'},
    {'label': 'Cardiovascular Diseases', 'value': 'Disease_Cardiovascular Diseases'}
]

race_options = [
    {'label': 'Black / African American', 'value': 'Black/African American'},
    {'label': 'White', 'value': 'White'},
    {'label': 'Other Race', 'value': 'Other Race'}
]

admission_options = [
    {'label': 'Emergency', 'value': 'Admission_Emergency'},
    {'label': 'Elective', 'value': 'Admission_Elective'},
    {'label': 'Urgent', 'value': 'Admission_Urgent'}
]

dash.register_page(__name__, path="/prediction", name="Prediction")

layout = html.Div([
    html.Div([
        html.H2("Hospital Stay & Profit Estimator", style={
            'textAlign': 'center', 'marginBottom': '30px', 'color': '#333', 'fontWeight': '600'
        }),

        html.Div([
            html.Div([
                html.Label("APR DRG Code", style={'marginBottom': '5px', 'fontWeight': '500'}),
                dcc.Input(id='drg_code', type='number', value=100, step=1,
                          style={'width': '100%', 'padding': '8px', 'marginBottom': '15px'}),

                html.Label("APR MDC Code", style={'marginBottom': '5px', 'fontWeight': '500'}),
                dcc.Input(id='mdc_code', type='number', value=20, step=1,
                          style={'width': '100%', 'padding': '8px', 'marginBottom': '15px'}),

                html.Label("APR Severity of Illness", style={'marginBottom': '5px', 'fontWeight': '500'}),
                dcc.Dropdown(id='apr_severity', options=[
                    {'label': 'Low', 'value': 1},
                    {'label': 'Moderate', 'value': 2},
                    {'label': 'Major', 'value': 3},
                    {'label': 'Extreme', 'value': 4}
                ], value=2, style={'marginBottom': '15px'}),

                html.Label("APR Risk of Mortality", style={'marginBottom': '5px', 'fontWeight': '500'}),
                dcc.Dropdown(id='apr_risk', options=[
                    {'label': 'Low', 'value': 1},
                    {'label': 'Moderate', 'value': 2},
                    {'label': 'Major', 'value': 3},
                    {'label': 'Extreme', 'value': 4}
                ], value=2, style={'marginBottom': '15px'}),

                html.Label("Age Group", style={'marginBottom': '5px', 'fontWeight': '500'}),
                dcc.Dropdown(id='age_group', options=[
                    {'label': '0 to 17', 'value': 1},
                    {'label': '18 to 29', 'value': 2},
                    {'label': '30 to 49', 'value': 3},
                    {'label': '50 to 69', 'value': 4},
                    {'label': '70 or Older', 'value': 5}
                ], value=2, style={'marginBottom': '15px'}),

                html.Label("Gender", style={'marginBottom': '5px', 'fontWeight': '500'}),
                dcc.Dropdown(id='gender', options=[
                    {'label': 'Female', 'value': 0},
                    {'label': 'Male', 'value': 1},
                    {'label': 'Unknown', 'value': 2}
                ], value=1, style={'marginBottom': '15px'})
            ], style={'width': '48%', 'display': 'inline-block', 'paddingRight': '2%'}),

            html.Div([
                html.Label("Admission Type", style={'marginBottom': '5px', 'fontWeight': '500'}),
                dcc.Dropdown(id='admission', options=admission_options, multi=False, style={'marginBottom': '15px'}),

                html.Label("Race", style={'marginBottom': '5px', 'fontWeight': '500'}),
                dcc.Dropdown(id='race', options=race_options, multi=False, style={'marginBottom': '15px'}),

                html.Label("Diseases", style={'marginBottom': '5px', 'fontWeight': '500'}),
                dcc.Dropdown(id='diseases', options=disease_options, multi=True, style={'marginBottom': '15px'})
            ], style={'width': '48%', 'display': 'inline-block', 'paddingLeft': '2%'})
        ], style={
            'display': 'flex', 'justifyContent': 'space-between', 'gap': '20px',
            'flexWrap': 'wrap', 'marginBottom': '30px'
        }),

        html.Div([
            html.Button("Predict", id='predict-btn', n_clicks=0, style={
                'backgroundColor': '#007bff', 'color': 'white', 'border': 'none',
                'padding': '12px 30px', 'borderRadius': '6px', 'fontSize': '16px', 'cursor': 'pointer'
            }),
            html.Div(id='output', style={'marginTop': '25px', 'fontSize': '20px', 'color': '#333'})
        ], style={'textAlign': 'center'})
    ], style={
        'maxWidth': '1000px', 'margin': '40px auto', 'padding': '30px', 'borderRadius': '12px',
        'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.08)', 'backgroundColor': 'white', 'fontFamily': 'Inter, sans-serif'
    })
])

@callback(
    Output('output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('drg_code', 'value'),
    State('mdc_code', 'value'),
    State('apr_severity', 'value'),
    State('apr_risk', 'value'),
    State('age_group', 'value'),
    State('gender', 'value'),
    State('admission', 'value'),
    State('race', 'value'),
    State('diseases', 'value')
)
def predict(n_clicks, drg_code, mdc_code, apr_severity, apr_risk, age_group, gender, admission, race, diseases):
    if n_clicks == 0:
        return ""

    data = {
        'apr_drg_code_oof_avg': compute_oof_avg(drg_code, {}),
        'apr_mdc_code_oof_avg': compute_oof_avg(mdc_code, {}),
        'apr_severity_of_illness': apr_severity,
        'apr_risk_of_mortality': apr_risk,
        'age_group': age_group,
        'gender': gender,
        'Admission_Emergency': int('Admission_Emergency' in admission),
        'Admission_Elective': int('Admission_Elective' in admission),
        'Admission_Urgent': int('Admission_Urgent' in admission),
        'Disease_Mental Health': int('Disease_Mental Health' in diseases),
        'Disease_Respiratory Disorders': int('Disease_Respiratory Disorders' in diseases),
        'Disease_Infectious Diseases': int('Disease_Infectious Diseases' in diseases),
        'Disease_Neurological Disorders': int('Disease_Neurological Disorders' in diseases),
        'Disease_Chronic Conditions': int('Disease_Chronic Conditions' in diseases),
        'Disease_Musculoskeletal Disorders': int('Disease_Musculoskeletal Disorders' in diseases),
        'Disease_General Symptoms': int('Disease_General Symptoms' in diseases),
        'Disease_Cardiovascular Diseases': int('Disease_Cardiovascular Diseases' in diseases),
        'Black/African American': int('Black/African American' in race),
        'White': int('White' in race),
        'Other Race': int('Other Race' in race),
        'apr_drg_code_enc': compute_smooth_enc(drg_code),
        'chronic_load_score': sum([
            int('Disease_Cardiovascular Diseases' in diseases),
            int('Disease_Chronic Conditions' in diseases),
            int('Disease_Mental Health' in diseases),
            int('Disease_Respiratory Disorders' in diseases),
            int('Disease_Musculoskeletal Disorders' in diseases),
            int('Disease_Neurological Disorders' in diseases)
        ]),
        'is_rare_drg': int(drg_code in rare_drg_set),
        'is_rare_mdc': int(mdc_code in rare_mdc_set),
        'disease_burden': len(diseases)
    }

    X = pd.DataFrame([data])
    los = model.predict(X)[0]
    profit_per_day = estimate_profit_per_day(admission)
    expected_profit = los * profit_per_day
    charge_per_day, cost_per_day = get_charge_and_cost_per_day(admission)
    total_charge = los * charge_per_day
    total_cost = los * cost_per_day
    tips = get_recommendations(los, total_charge)

    return [
        html.Div(f"Predicted Length of Stay: {round(los, 2)} days"),
        html.Div(f"Estimated Charge: ${round(total_charge, 2)}"),
        html.Div(f"Estimated Cost: ${round(total_cost, 2)}"),
        html.Div(f"Estimated Profit: ${round(expected_profit, 2)}"),
        html.Br(),
        html.Div("Recommendations:", style={'fontWeight': 'bold', 'marginTop': '20px'}),
        html.Ul([html.Li(tip) for tip in tips])
    ]
