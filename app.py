import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from pages import Home, overview, trends, comparison, similar_anime

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Navbar
navbar = dbc.Navbar(
    dbc.Container(
        [
            html.H2("AniLens",style={'margin-right':'300px'}),
            html.Br(),
            dbc.NavLink("Home", href="/", active='exact'),
            dbc.NavLink("Overview", href="/overview", active='exact'),
            dbc.NavLink("Trends", href="/trends", active='exact'),
            dbc.NavLink("Comparison", href="/comparison", active='exact'),
            dbc.NavLink("Similar Anime", href="/similar-anime", active='exact')
        ]
    ),
    color="primary", dark=True,
)

# Main Layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content')
])


# Callback to update navbar and page content
@callback(
    Output('page-content', 'children'),
    Output('url', 'pathname'),  # Update for navbar highlighting
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/':
        return Home.layout, '/'
    elif pathname == '/overview':
        return overview.layout, '/overview'
    elif pathname == '/trends':
        return trends.layout, '/trends'
    elif pathname == '/comparison':
        return comparison.layout, '/comparison'
    elif pathname == '/similar-anime':
        return similar_anime.layout, '/similar-anime'
    else:
        return "404 - Page not found"


if __name__ == '__main__':
    app.run_server(debug=True)
