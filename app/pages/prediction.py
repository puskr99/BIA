import dash
from dash import html, dcc, Input, Output, State, callback
import pandas as pd
import numpy as np
import joblib

# Load model and data
model = joblib.load("code/los_model")
profit_map = joblib.load("code/profit_map.pkl")
drg_oof_map = joblib.load("code/drg_oof_map.pkl")
mdc_oof_map = joblib.load("code/mdc_oof_map.pkl")
feature_importances = pd.read_csv("code/feat_imp.csv") 

# Example rare code sets and encoders (replace with actual from training)
rare_drg_set = {999, 888}  # Example
rare_mdc_set = {77, 88}  # Example
global_los_mean = 6.24  # Fallback for missing encoding

# Model error metrics (from cross-validation)
MODEL_MAE = 0.86 # Mean Absolute Error
SHORT_STAY_MAE = 1.15  # MAE for short stays (≤2 days)
LONG_STAY_MAE = 2.5

# Dummy DRG/MDC encoders (replace with actual logic)
def compute_oof_avg(code, mapping):
    return mapping.get(code, global_los_mean)

def compute_smooth_enc(code):
    return np.log1p(code % 100)  # Placeholder logic

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
    charges = [profit_map.get(adm, 0) + 100 for adm in mapped]  # Placeholder
    costs = [100 for adm in mapped]  # Placeholder
    return np.mean(charges), np.mean(costs)

def get_recommendations(los, charge_total, drg_code, mdc_code, apr_severity, apr_risk, admission, diseases, age_group, gender, race):
    """
    Generate data-driven, patient-specific recommendations based on LOS, financials, and patient features.
    
    Args:
        los (float): Predicted length of stay.
        charge_total (float): Total estimated charges.
        drg_code (int): APR DRG code.
        m Chinhdc_code (int): APR MDC code.
        apr_severity (int): Severity of illness (1-4).
        apr_risk (int): Risk of mortality (1-4).
        admission (str): Admission type (Emergency, Elective, Urgent).
        diseases (list): List of disease categories.
        age_group (int): Age group (1-5).
        gender (int): Gender (0: Female, 1: Male, 2: Unknown).
        race (str): Race category.
    
    Returns:
        list: Tailored clinical, operational, and financial recommendations.
    """
    tips = []
    feature_weights = dict(zip(feature_importances['feature'], feature_importances['importance']))
    
    # Short stays (≤3 days, adjusted based on SHORT_STAY_MAE ~1.3)
    if los < 3:
        tips.append(f"High readmission risk detected (MAE: {SHORT_STAY_MAE:.2f} days). Schedule outpatient follow-up within 48 hours to monitor recovery.")
        if "Admission_Emergency" in admission:
            tips.append("Emergency admission: Prioritize rapid diagnostics (e.g., labs, imaging) to confirm discharge readiness.")
        if apr_severity <= 2:
            tips.append("Low-moderate severity: Consider same-day discharge protocols with telehealth follow-up to optimize bed turnover.")
        else:
            tips.append(f"Severity {apr_severity}: Monitor for complications despite short predicted LOS, as model uncertainty is higher (MAE: {SHORT_STAY_MAE:.2f}).")
        if len(diseases) > 2:
            tips.append(f"Multiple diseases ({len(diseases)}): Coordinate with primary care for post-discharge management of comorbidities.")
    
    # Medium stays (3-7 days)
    elif 3 <= los <= 7:
        tips.append(f"Moderate LOS (predicted: {los:.2f} days, MAE: {MODEL_MAE:.2f}). Follow standardized care pathways for {', '.join(diseases) or 'general care'}.")
        if apr_severity >= 3:
            tips.append(f"High severity ({apr_severity}): Engage specialists (e.g., cardiology for {diseases}) to address potential complications.")
        if "Disease_Chronic Conditions" in diseases:
            tips.append("Chronic conditions detected: Implement chronic disease management protocols (e.g., medication reconciliation, patient education).")
        tips.append(f"Allocate multidisciplinary team (e.g., nursing, rehab) to streamline care and prepare for discharge by day {round(los)}.")

    # Long stays (≥7 days)
    else:
        tips.append(f"Long LOS (predicted: {los:.2f} days, MAE: {LONG_STAY_MAE:.2f}). Initiate intensive case management to coordinate complex care.")
        if apr_risk >= 3:
            tips.append(f"High mortality risk ({apr_risk}): Consult palliative care or ethics team to discuss care goals.")
        if drg_code in rare_drg_set or mdc_code in rare_mdc_set:
            tips.append(f"Rare DRG/MDC ({drg_code}/{mdc_code}): Refer to specialized care unit (e.g., tertiary center) due to model uncertainty.")
        if age_group >= 4:  # 50+ years
            tips.append("Older age group: Assess for frailty and involve physical therapy to prevent functional decline.")
        tips.append("Review daily for complications (e.g., infections, organ failure) to avoid further LOS prolongation.")

    # Financial recommendations
    if charge_total > 700:
        tips.append(f"High charges (${charge_total:,.2f}): Review treatment plan for cost-effective alternatives (e.g., generic medications, outpatient diagnostics).")
        if "Admission_Elective" in admission:
            tips.append("Elective admission: Explore pre-negotiated insurance bundles to reduce patient financial burden.")
    if estimate_profit_per_day(admission) < 50:
        tips.append("Low profit margin: Optimize resource use (e.g., reduce unnecessary imaging) to improve financial outcomes.")

    # Feature importance-based recommendations
    top_features = sorted(feature_weights, key=feature_weights.get, reverse=True)[:3]
    if 'apr_severity_of_illness' in top_features:
        tips.append(f"High importance of severity ({feature_weights['apr_severity_of_illness']:.2f}): Escalate care intensity for severity {apr_severity}.")
    if 'chronic_load_score' in top_features and len(diseases) > 1:
        tips.append(f"High chronic disease burden ({len(diseases)} diseases): Develop long-term management plan with specialists.")

    # Demographic-specific recommendations
    if race in ['Black/African American', 'Other Race']:
        tips.append("Consider social determinants (e.g., access to care, transportation) to support discharge planning.")
    if gender == 0 and 'Disease_Cardiovascular Diseases' in diseases:
        tips.append("Female with cardiovascular disease: Ensure gender-specific risk factors (e.g., atypical symptoms) are addressed.")

    return tips

# Dash app setup (unchanged from your code)
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

                
            ], style={'width': '48%', 'display': 'inline-block', 'paddingRight': '2%'}),

            html.Div([

                html.Label("Gender", style={'marginBottom': '5px', 'fontWeight': '500'}),
                dcc.Dropdown(id='gender', options=[
                    {'label': 'Female', 'value': 0},
                    {'label': 'Male', 'value': 1},
                    {'label': 'Others', 'value': 2}
                ], value=1, style={'marginBottom': '15px'}),
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

    # Ensure inputs are lists for consistency
    if not isinstance(diseases, list):
        diseases = [diseases] if diseases else []
    if not isinstance(admission, list):
        admission = [admission] if admission else []
    if not isinstance(race, list):
        race = [race] if race else []

    data = {
        'apr_drg_code_oof_avg': compute_oof_avg(drg_code, drg_oof_map),
        'apr_mdc_code_oof_avg': compute_oof_avg(mdc_code, mdc_oof_map),
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
    tips = get_recommendations(los, total_charge, drg_code, mdc_code, apr_severity, apr_risk, admission, diseases, age_group, gender, race)

    return [
        html.Div([
            html.H3("Patient Outcome Summary", style={
                'marginBottom': '10px',
                'fontWeight': 'bold',
                'fontSize': '22px'
            }),
            html.Hr(style={'marginTop': '0', 'marginBottom': '20px', 'borderTop': '2px solid #ccc'}),

            html.Table([
                html.Tr([
                    html.Th("Metric", style={'textAlign': 'center', 'paddingRight': '20px'}),
                    html.Th("Value", style={'textAlign': 'center'})
                ]),
                html.Tr([
                    html.Td("Predicted Length of Stay:", style={'textAlign': 'center'}), 
                    html.Td(f"{round(los, 2)} days ± {MODEL_MAE} days", style={'textAlign': 'center', 'paddingRight': '20px'})
                ]),
                html.Tr([
                    html.Td("Estimated Total Charge:", style={'textAlign': 'center'}), 
                    html.Td(f"${round(total_charge, 2):,}", style={'textAlign': 'center', 'paddingRight': '20px'})
                ]),
                html.Tr([
                    html.Td("Estimated Total Cost:", style={'textAlign': 'center'}), 
                    html.Td(f"${round(total_cost, 2):,}", style={'textAlign': 'center', 'paddingRight': '20px'})
                ]),
                html.Tr([
                    html.Td("Estimated Profit:", style={'textAlign': 'center'}), 
                    html.Td(f"${round(expected_profit, 2):,}", style={'fontWeight': 'bold', 'color': 'green', 'textAlign': 'center', 'paddingRight': '20px'})
                ])
            ], style={
                'width': '100%',
                'borderCollapse': 'collapse',
                'marginBottom': '30px',
                'fontSize': '16px'
            }),

            html.H3("Recommendations", style={
                'marginBottom': '10px',
                'fontWeight': 'bold',
                'fontSize': '22px'
            }),
            html.Hr(style={'marginTop': '0', 'marginBottom': '20px', 'borderTop': '2px solid #ccc'}),

            html.Ol([
                html.Li(tip, style={'marginBottom': '10px', 'textAlign': 'left', 'lineHeight': '1.6'}) for tip in tips
            ])
        ], style={
            'padding': '25px',
            'border': '1px solid #ccc',
            'borderRadius': '10px',
            'backgroundColor': '#fcfcfc',
            'boxShadow': '0px 2px 5px rgba(0, 0, 0, 0.05)',
            'fontFamily': 'Arial, sans-serif'
        })
    ]