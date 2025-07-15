import dash
from dash import html, dcc
from dash_iconify import DashIconify
import dash_bootstrap_components as dbc
from components.navbar import register_callbacks

CONTENT_STYLE = {
    "marginLeft": "18rem",
    "marginRight": "2rem",
    "padding": "2rem 1rem",
}

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "backgroundColor": "#2d3a46",
    "borderRight": "1px solid #333"
}

sidebar = html.Div([
    html.H1("Inpatient Care", className="titlebar mb-4"),

    html.H6("Analysis", className="subsection fw-bold"),
    html.Hr(style={"borderColor": "#e0e0e0"}),

    dbc.Nav([
        dbc.NavLink(
            [DashIconify(icon="mdi:home", width=20, className="me-2"), "Overview"],
            href="/", active="exact"
        )
    ], vertical=True, pills=True, className="mb-4"),

    html.H6("Model", className="subsection fw-bold"),
    html.Hr(style={"borderColor": "#e0e0e0"}),
    dbc.Nav([
        dbc.NavLink(
            [DashIconify(icon="mdi:sparkles", width=20, className="me-2"), "Prediction"],
            href="/prediction", active="exact"
        ),
    ], vertical=True, pills=True, className="mb-4"),

    html.H6("Other", className="subsection fw-bold"),
    html.Hr(style={"borderColor": "#e0e0e0"}),
    dbc.Nav([
        dbc.NavLink(
            [DashIconify(icon="mdi:information", width=20, className="me-2"), "About"],
            href="/about",active="exact"
        )
    ], vertical=True, pills=True, className="mb-4"),
    

], style=SIDEBAR_STYLE)


app = dash.Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

register_callbacks(app)

app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
     html.Div(dash.page_container, style=CONTENT_STYLE)
])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

