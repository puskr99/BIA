from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import dash
from dash import dash_table


dash.register_page(__name__, path="/", name="Home")

# Load data
import dash
import pandas as pd
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, callback
import plotly.express as px

# Load and clean data safely
try:
    data = pd.read_csv("data/bia.csv", low_memory=False)

    # Clean length_of_stay
    data['length_of_stay'] = (
        data['length_of_stay']
        .astype(str)
        .str.split()
        .str[0]
        .str.extract(r'(\d+)')[0]
        .astype('Int64')
    )
    data = data[data['length_of_stay'] <= 30]

    # Clean and convert charges and costs
    data['total_charges'] = (
        data['total_charges']
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    data['total_costs'] = (
        data['total_costs']
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    # Convert to numeric (after string cleaning)
    data['total_charges'] = pd.to_numeric(data['total_charges'], errors='coerce')
    data['total_costs'] = pd.to_numeric(data['total_costs'], errors='coerce')

    # Build summary table
    summary_df = (
        data.groupby("ccsr_diagnosis_description")
            .agg(
                avg_los=('length_of_stay', 'mean'),
                avg_charges=('total_charges', 'mean'),
                avg_costs=('total_costs', 'mean'),
                patient_count=('length_of_stay', 'count')
            )
            .sort_values(by='patient_count', ascending=False)
            .reset_index()
    )

    # Round and rename
    summary_df['avg_los'] = summary_df['avg_los'].round(1)
    summary_df['avg_charges'] = summary_df['avg_charges'].round(2)
    summary_df['avg_costs'] = summary_df['avg_costs'].round(2)
    summary_df.columns = [
        "Diagnosis",
        "Average Length of Stay (days)",
        "Average Charges (USD)",
        "Average Costs (USD)",
        "Patient Count"
    ]

except Exception as e:
    print(f"⚠️ Error in data preprocessing: {e}")
    data = pd.DataFrame()


# layout = dbc.Container([
#     html.H2("Analytics Dashboard"),
#     html.Hr(),

#     # KPI Cards
#     dbc.Row([
#         dbc.Col(
#             dbc.Card([
#                 dbc.CardHeader("Total Patients"),
#                 dbc.CardBody(html.H4(f"{len(data):,}", className="card-title"))
#             ] ),
#             width=4
#         ),
#         dbc.Col(
#             dbc.Card([
#                 dbc.CardHeader("Average LOS (Days)"),
#                 dbc.CardBody(html.H4(f"{data['length_of_stay'].mean():.1f}", className="card-title"))
#             ]),
#             width=4
#         ),
#         dbc.Col(
#             dbc.Card([
#                 dbc.CardHeader("Total Charges (USD)"),
#                 dbc.CardBody(html.H4(f"${data['total_charges'].sum():,.0f}", className="card-title"))
#             ]),
#             width=4
#         )
#     ], className="mb-4"),

#     # Chart Cards
#     dbc.Row([
#         dbc.Col(
#             dbc.Card([
#                 dbc.CardHeader("Average LOS by Severity"),
#                 dbc.CardBody(dcc.Graph(id='los-by-severity'))
#             ]), width=6
#         ),
#         dbc.Col(
#             dbc.Card([
#                 dbc.CardHeader("Total Charges by Admission Type"),
#                 dbc.CardBody(dcc.Graph(id='charges-by-admission'))
#             ]), width=6
#         )
#     ]),
#     dbc.Row([
#         dbc.Col(
#             dbc.Card([
#                 dbc.CardHeader("LOS & Charges Summary by Diagnosis"),
#                 dbc.CardBody(
#                     dash.dash_table.DataTable(
#                         columns=[{"name": i, "id": i} for i in summary_df.columns],
#                         data=summary_df.to_dict("records"),
#                         page_size=10,
#                         style_table={"overflowX": "auto"},
#                         style_cell={"textAlign": "left", "padding": "5px"},
#                         style_header={"backgroundColor": "lightgrey", "fontWeight": "bold"}
#                     )
#                 )
#             ]), width=12
#         )
#     ]),

#     dbc.Row([
#         dbc.Col(
#             dbc.Card([
#                 dbc.CardHeader("Patients by County"),
#                 dbc.CardBody(dcc.Graph(id='patients-by-county'))
#             ]), width=6
#         ),
#         dbc.Col(
#             dbc.Card([
#                 dbc.CardHeader("Patients by Severity"),
#                 dbc.CardBody(dcc.Graph(id='patients-by-severity'))
#             ]), width=6
#         )
#     ]),
#     dbc.Row([
#         dbc.Col(
#             dbc.Card([
#                 dbc.CardHeader("Age Group Distribution"),
#                 dbc.CardBody(dcc.Graph(id='age-distribution'))
#             ]), width=6
#         ),
#         dbc.Col(
#             dbc.Card([
#                 dbc.CardHeader("Avg LOS by Year"),
#                 dbc.CardBody(dcc.Graph(id='los-distribution'))
#             ]), width=6
#         )
#     ]),


# ], fluid=True)




# # Callbacks for analytics charts
# @callback(Output('los-by-severity', 'figure'), Input('los-by-severity', 'id'))
# def update_los_by_severity(_):
#     df = data.groupby("apr_severity_of_illness")['length_of_stay'].mean().reset_index()
#     return px.bar(df, x='apr_severity_of_illness', y='length_of_stay',color='apr_severity_of_illness', title='Avg LOS by Severity')

# @callback(Output('charges-by-admission', 'figure'), Input('charges-by-admission', 'id'))
# def update_charges_by_admission(_):
#     df = data.groupby("type_of_admission")['total_charges'].sum().reset_index()
#     return px.bar(df, x='type_of_admission', y='total_charges',color='type_of_admission', title='Total Charges by Admission Type')

# @callback(Output('patients-by-county', 'figure'), Input('patients-by-county', 'id'))
# def update_patients_by_county(_):
#     df = data['hospital_county'].value_counts().reset_index()
#     df.columns = ['hospital_county', 'count']
#     return px.bar(df, x='hospital_county', y='count', title='Patients by County')   

# @callback(Output('patients-by-severity', 'figure'), Input('patients-by-severity', 'id'))
# def update_patients_by_severity(_):
#     df = data['apr_severity_of_illness'].value_counts().reset_index()
#     df.columns = ['Severity', 'Count']
#     return px.bar(df, x='Severity', y='Count',color='Severity', title='Patient Count by Illness Severity')


# @callback(Output('age-distribution', 'figure'), Input('age-distribution', 'id'))
# def update_age_distribution(_):
#     df = data['age_group'].value_counts().reset_index()
#     df.columns = ['age_group', 'count']  # rename for clarity
#     return px.pie(df, names='age_group', values='count', title='Age Group Distribution')

# @callback(Output('los-distribution', 'figure'), Input('los-distribution', 'id'))
# def update_los_distribution(_):
#     return px.histogram(
#         data,
#         x='length_of_stay',
#         nbins=30,
#         title='Distribution of Length of Stay',
#         labels={'length_of_stay': 'Length of Stay (Days)'}
#     )


layout = html.Div([
    # html.H3("Power BI Dashboard", className="mb-4"),

    html.Iframe(
        src="https://app.powerbi.com/view?r=eyJrIjoiYTJkOGNlMjgtNTkyYi00MDQ5LWI1ODMtMTI3YjlkNDY5Y2ZmIiwidCI6Ijk5ZWViMDA5LWU3YTItNDdiNi05ZGVkLTAyOGNkY2MzMDBlNiIsImMiOjEwfQ%3D%3D" ,  # ← Replace with your actual Power BI link
        width="100%",
        height="650px",
        style={"border": "none"}
    )
])
