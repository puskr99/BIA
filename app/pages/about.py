import dash
from dash import html

dash.register_page(__name__, path="/about")

layout = html.Div([
    # Section 1: Introduction
    html.Section([
        html.H2("About This App", className="text-center mb-4"),
        html.P(
            "This Decision Support System (DSS) is designed to help hospital administrators estimate "
            "the expected Length of Stay (LOS) for patients based on clinical and demographic inputs. "
            "By integrating predictive modeling and real hospital data, the tool offers actionable insights "
            "such as anticipated costs, charges, and profit, along with care recommendations. "
            "It aims to optimize resource allocation and improve patient outcomes.",
            style={'maxWidth': '800px', 'margin': '0 auto 50px', 'fontSize': '18px', 'lineHeight': '1.6'}
        ),
    ], className="section p-5"),

    # Section 2: Team
    html.Section([
        html.H3("Meet the Team", className="text-center mb-4"),
        html.Div([
            html.Div([
                html.Img(src="assets/girl.jpg", className="img-fluid rounded-circle shadow-lg", style={"width": "150px", "height": "150px", "margin-bottom": "10px"}),
                html.P("Cassandra Chang", className="h5 text-center mb-2"),
                html.P("Data Scientist", className="text-muted text-center"),
            ], className="col-md-6 d-flex flex-column align-items-center mb-4"),
            html.Div([
                html.Img(src="assets/boy.jpg", className="img-fluid rounded-circle shadow-lg", style={"width": "150px", "height": "150px", "margin-bottom": "10px"}),
                html.P("Puskar Adhikari", className="h5 text-center mb-2"),
                html.P("Data Scientist", className="text-muted text-center"),
            ], className="col-md-6 d-flex flex-column align-items-center mb-4"),
            
            html.Div([
                html.Img(src="assets/girl.jpg", className="img-fluid rounded-circle shadow-lg", style={"width": "150px", "height": "150px", "margin-bottom": "10px"}),
                html.P("Shreeyukta Pradhanang", className="h5 text-center mb-2"),
                html.P("Data Analyst & Web Developer", className="text-muted text-center"),
            ], className="col-md-6 d-flex flex-column align-items-center mb-4"),
        ], className="row justify-content-center")
    ], className="section p-5 bg-light"),

    html.Section([
        html.H3("Find the Dataset and GitHub Repository", className="text-center mb-4"),
        html.Div([
            html.A(
                "View the Dataset",
                href="https://healthdata.gov/dataset/Hospital-Inpatient-Discharges-SPARCS-De-Identified/tpup-hey9/about_data", 
                target="_blank", 
                className="btn btn-outline-primary me-2" 
            ),
            html.A(
                "Visit GitHub Repository",
                href="https://github.com/puskr99/BIA",  
                target="_blank",
                className="btn btn-outline-secondary"  
            ),
        ], className="text-center")
    ], className="section p-5")
    
], className="container")