import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mpld3    
import plotly.express as px
from plotly.tools import mpl_to_plotly
import sys
import os
import base64
import io
from io import BytesIO

# Add locally cloned Lux source code to path, and import Lux from there
sys.path.insert(0, os.path.abspath("./lux"))
import lux

# Disable widget attachment
lux.config.default_display = "none"  

# Use non-interactive backend
matplotlib.use('Agg')  

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Global variable to store Lux recommendations
recommendations = {}

# Global variable to store recommendation options
rec_options = []

# Set dashboard layout
app.layout = dbc.Container([
    html.H1("Visual Data Engineering", className="text-center my-4"),

    # File upload component
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ', html.A('Select a CSV File')
        ]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
        },
    ),

    # Placeholder for uploaded data and visualisations
    html.Div(id='output-data-upload', className="my-4"),

    # Button to trigger Lux recommendations
    html.Div([
        dbc.Button("Show Recommendations", id="show-recs", color="primary", className="mt-2"),
        dcc.Dropdown(placeholder="Select a recommendation option", id='rec-dropdown'),
        html.Div(id='rec-output-container'),
        html.Div(id="lux-output", className="mt-4")
    ])
])

# Global variable to store the uploaded DataFrame
uploaded_df = None

# Function to parse uploaded data
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    if filename.endswith('.csv'):
        return pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    return None

# Callback to handle file upload
@app.callback(
    Output(component_id='output-data-upload', component_property='children'),
    Input(component_id='upload-data', component_property='contents'),
    State(component_id='upload-data', component_property='filename')
)
def update_output(contents, filename):
    global uploaded_df
    if contents is not None:
        # Parse uploaded contents
        uploaded_df = parse_contents(contents, filename)
        if uploaded_df is not None:
            # Enable Lux for the uploaded DataFrame
            uploaded_df = pd.DataFrame(uploaded_df)
            return html.Div([
                html.H5(f"Uploaded File: {filename}"),
                dbc.Table.from_dataframe(uploaded_df.head(), striped=True, bordered=True, hover=True)
            ])
        return html.Div("Unsupported file type.")
    return html.Div("No file uploaded.")

# Callback to display Lux recommendations
@app.callback(
    [Output(component_id='lux-output', component_property='children'),
    Output(component_id='rec-dropdown', component_property='options')],
    [Input(component_id='show-recs', component_property='n_clicks'),
    Input(component_id='rec-dropdown', component_property='value')],
    prevent_initial_call=True
)
def show_recommendations(n_clicks, drop_value):
    global uploaded_df
    global recommendations
    global rec_options
    if n_clicks and uploaded_df is not None:
        # Generate recommendations and store the resulting dictionary explicitly
        recommendations = uploaded_df.recommendation
        if rec_options == []:
            rec_options = [{'label': key, 'value': key} for key in recommendations]  # .keys()
        # print("recommendations: ", recommendations)
        if recommendations:
            # Ensure drop_value is valid; default to the first option if not
            if drop_value not in recommendations:
                drop_value = rec_options[0]['value']

            # Access specified recommendation group
            selected_recommendations = recommendations[drop_value]
            print("drop_value: ", drop_value)
            print("selected_recommendations: ", selected_recommendations)

            if selected_recommendations:
                graph_components = []

                for vis in selected_recommendations:
                    
                    # Initialise variables that will be specified in the fig_code 
                    fig, ax = plt.subplots()

                    # Render the visualisation using Lux
                    fig_code = vis.to_matplotlib()
                    exec(fig_code)

                    # Capture the current Matplotlib figure
                    fig = plt.gcf()
                    plt.draw()

                    fig.savefig("debug_figure.png")

                    # Fix incompatible properties in the Matplotlib figure
                    # Loop through axes and update any properties (like bargap) that may have invalid types
                    for ax in fig.axes:
                        for artist in ax.get_children():
                            artist.bargap = 0.8

                    # Convert the Matplotlib figure to Plotly
                    plotly_fig = mpl_to_plotly(fig)

                    # Append the figure as a Dash Graph component
                    graph_components.append(
                        dcc.Graph(figure=plotly_fig, style={'flex': '1 0 30%', 'margin': '5px'})
                    )


                    # # Convert the Matplotlib figure to HTML
                    # html_matplotlib = mpld3.fig_to_html(fig)

                # Return all Graph components inside a flexbox container
                return html.Div(
                    children=graph_components,
                    style={
                        'display': 'flex',
                        'flexWrap': 'wrap',
                        'justifyContent': 'space-around',
                        'margin': '5px'
                    }
                ), rec_options
    return html.Div("No recommendations available. Upload data first."), []

# Callback to choose which recommendations to display
@app.callback(
    Output(component_id='rec-output-container', component_property='children'),
    Input(component_id='rec-dropdown', component_property='value')
)
def update_rec_option(value):
    if value is None:
        return ""
    return f'Showing {value} recommendations'

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
