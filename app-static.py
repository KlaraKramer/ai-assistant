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
import IPython.display as display
from PIL import Image
import re

# Add locally cloned Lux source code to path, and import Lux from there
sys.path.insert(0, os.path.abspath("./lux"))
import lux

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

    html.Div([
        # Button to trigger Lux recommendations
        dbc.Button("Show Recommendations", id="show-recs", color="primary", className="mt-2"),
        # Dropdown to choose recommendation option (initially hidden)
        dcc.Dropdown(
            placeholder="Select a recommendation option", 
            id='rec-dropdown',
            style={'display': 'none'}  # Initially hidden
        ),
        html.Div(id='rec-output-container', style={'display': 'none'}),
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
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        # Convert all column names to lowercase and replace spaces with underscores
        df.rename(columns=lambda x: x.lower().replace(" ", "_").replace("-", "_"), inplace=True)
        # Remove all special characters from column names
        df.rename(columns=lambda x: x.lower().replace(":", "").replace("$", "").replace("(", "").replace(")", ""), inplace=True)
        return df
    return None

def create_styled_matplotlib_figure(fig):
    """Apply Plotly-like styling to an existing Matplotlib figure."""
    
    # Set the figure background to white (to match Plotly)
    fig.patch.set_facecolor("white")

    # Get the main axis
    ax = fig.axes[0] if fig.axes else fig.add_subplot(111)  # Ensure there is an axis
    ax.set_facecolor("#E5ECF6")  # Light blue background only for the plotting area

    # Update bar colours if applicable
    for patch in ax.patches:
        patch.set_facecolor("#4C59C2")

        # Make bars slimmer
        if isinstance(patch, plt.Rectangle):  # Ensure it's a bar
            if patch.get_width() > patch.get_height():  # Horizontal bars
                patch.set_height(patch.get_height() * 0.3)  # Reduce thickness
            else:  # Vertical bars
                patch.set_width(patch.get_width() * 0.3)

    # Style the labels
    ax.set_xlabel(ax.get_xlabel(), fontsize=10, labelpad=12, 
                  bbox=dict(facecolor="white", edgecolor="none"))
    ax.set_ylabel(ax.get_ylabel(), fontsize=10, labelpad=12, 
                  bbox=dict(facecolor="white", edgecolor="none"))

    # Style the tick labels
    ax.tick_params(axis='both', labelsize=7)

    # Remove unnecessary spines (top & right)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#AAB8C2")  # Light gray for a cleaner look
    ax.spines["left"].set_color("#AAB8C2")

    return fig


def fig_to_base64(fig):
    """Convert a Matplotlib figure to a base64-encoded PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded_img = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return f"data:image/png;base64,{encoded_img}"

def fix_lux_code(lux_code):
    """
    This function takes the Lux-generated code as input, identifies
    and fixes any syntax issues related to the `ax.barh()` function.
    Specifically, it replaces the incorrectly formatted bar data with
    proper `.values` access to the Pandas Series.
    """
    
    # Pattern to identify the `ax.barh()` function call
    # barh_pattern = r"(ax\.barh\().*?(dtype: object,.*?dtype: int64.*?)\)"
    pattern = r"(ax\.barh\()(.*?dtype:.*?dtype:.*?)\)"
    
    # Replacement that ensures bars and measurements are properly formatted
    replacement = r"ax.barh(bars.values, measurements.values, align='center')"

    # Apply the substitution to the code to fix the barh() call
    fixed_code = re.sub(pattern, replacement, lux_code, flags=re.DOTALL)
    
    return fixed_code

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
     Output(component_id='rec-dropdown', component_property='style'),
     Output(component_id='rec-output-container', component_property='style'),
     Output(component_id='rec-dropdown', component_property='options')],
    [Input(component_id='show-recs', component_property='n_clicks'),
     Input(component_id='rec-dropdown', component_property='value')],
    [State(component_id='upload-data', component_property='filename')],
    #  State(component_id='upload-data', component_property='contents')],
    prevent_initial_call=True
)
def show_recommendations(n_clicks, drop_value, filename):
    global uploaded_df
    global recommendations
    global rec_options
    if n_clicks:
        # Make the dropdown visible when the button is clicked
        dropdown_style = {'display': 'block'}
    else:
        # Keep the dropdown hidden by default
        dropdown_style = {'display': 'none'}

    if n_clicks and uploaded_df is not None:
        # Generate recommendations and store the resulting dictionary explicitly
        recommendations = uploaded_df.recommendation
        if rec_options == []:
            rec_options = [{'label': key, 'value': key} for key in recommendations]
        if recommendations:
            # Ensure drop_value is valid; default to the first option if not
            if drop_value not in recommendations:
                drop_value = rec_options[0]['value']

            # Access specified recommendation group
            selected_recommendations = recommendations[drop_value]

            if selected_recommendations:
                graph_components = []

                for vis in selected_recommendations:
                    
                    # Initialise variables that will be specified in the fig_code 
                    fig, ax = plt.subplots()

                    # Render the visualisation using Lux
                    fig_code = vis.to_matplotlib()
                    fixed_fig_code = fix_lux_code(fig_code)
                    exec(fixed_fig_code)

                    # Capture the current Matplotlib figure
                    fig = plt.gcf()
                    plt.draw()

                    # Try to convert Matplotlib figure to Plotly
                    try:
                        plotly_fig = mpl_to_plotly(fig)

                        # plotly_fig.update_layout(width=1000, height=600)

                        # Append the figure as a Dash Graph component
                        graph_components.append(
                            dcc.Graph(figure=plotly_fig, style={'flex': '1 0 30%', 'margin': '5px'})
                        )
                    except ValueError as e:
                        error_message = str(e)
                        print(f"Error during mpl_to_plotly conversion: {error_message}")

                        # If an error occurs, display the static Matplotlib image instead
                        print("Falling back to displaying a static image.")

                        # Create the styled Matplotlib figure
                        fallback_fig = create_styled_matplotlib_figure(fig)

                        # Convert Matplotlib figure to base64 image
                        img_src = fig_to_base64(fallback_fig)

                        # Append the image as an Img component
                        graph_components.append(
                            html.Img(src=img_src, style={'flex': '1 0 30%', 'margin': '5px'})
                        )

                # Return all Graph components inside a flexbox container
                return html.Div(
                    children=graph_components,
                    style={
                        'display': 'flex',
                        'flexWrap': 'wrap',
                        'justifyContent': 'space-around',
                        'margin': '5px'
                    }
                ), dropdown_style, dropdown_style, rec_options
    return html.Div("No recommendations available. Upload data first."), dropdown_style, dropdown_style, []

# Callback to choose which recommendations to display
@app.callback(
    Output(component_id='rec-output-container', component_property='children'),
    Input(component_id='rec-dropdown', component_property='value')
)
def update_rec_option(value):
    if value is None:
        value = ""
    return f'Showing {value} recommendations'

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
