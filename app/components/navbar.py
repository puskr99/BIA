import dash
import dash_bootstrap_components as dbc
from dash import Input, Output

def register_callbacks(app):
    @app.callback(
        Output("navbar-collapse", "is_open"),
        [Input("navbar-toggler", "n_clicks")],
        [dash.State("navbar-collapse", "is_open")],
    )
    def toggle_navbar(n, is_open):
        if n:
            return not is_open
        return is_open
