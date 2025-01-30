import dash
from dash import dcc, html, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import Set1
import mpld3    
from plotly.tools import mpl_to_plotly
import sys
import os
import numpy as np

from helper_functions import *

# Add locally cloned Lux source code to path, and import Lux from there
sys.path.insert(0, os.path.abspath("./lux"))
import lux

# Use non-interactive backend
matplotlib.use('Agg')  

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# # Global variable to store Lux recommendations
# recommendations = {}

# Global variable to store recommendation options
rec_options = []

# Global variable to store the n_clicks_list for figures
figure_clicks = []

# Global variable to store the uploaded DataFrame
uploaded_df = None

# Global variable to store selected columns from clicking on a figure
selected_columns = ()

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
        dbc.Button(
            "Show Recommendations", 
            id="show-recs", 
            color="primary", 
            className="mt-2"
        ),
        # Dropdown to choose recommendation option (initially hidden)
        dcc.Dropdown(
            placeholder="Select a recommendation option", 
            id='rec-dropdown',
            style={'display': 'none'}  # Initially hidden
        ),
        html.Div(id='rec-output-container', style={'display': 'none'}),
        html.Div(id='lux-output', className='mt-4'),
        html.Div(id='vis-selection-output', style={'display': 'none'}),
        # html.Div(id='enhance-button-container'),
        dbc.Button(
            "Enhance",
            id='enhance-button',
            className="btn btn-success",
            style={'display': 'none', 'margin-left': '10px'}
        ),
        html.Div(id='enhanced-output', className='mt-4')
    ])
])

# Callback to handle both the file upload and button click
@app.callback(
    [Output(component_id='output-data-upload', component_property='children'),
     Output(component_id='rec-dropdown', component_property='style'),
     Output(component_id='rec-output-container', component_property='style'),
     Output(component_id='lux-output', component_property='children'),
     Output(component_id='rec-dropdown', component_property='options')],
    [Input(component_id='upload-data', component_property='contents'),
     Input(component_id='show-recs', component_property='n_clicks'),
     Input(component_id='rec-dropdown', component_property='value')],
    [State(component_id='upload-data', component_property='filename')],
    prevent_initial_call=True
)
def update_ui(contents, n_clicks, drop_value, filename):
    global uploaded_df
    # global recommendations
    global rec_options

    # If no data has been uploaded yet
    if contents is None:
        return html.Div("Unsupported file type."), {'display': 'none'}, {'display': 'none'}, html.Div("No recommendations available. Upload data first."), []

    # Handle file upload (uploading data)
    if contents:
        # Parse uploaded contents
        uploaded_df = parse_contents(contents, filename)
        if uploaded_df is not None:
            # Enable Lux for the uploaded DataFrame
            uploaded_df = pd.DataFrame(uploaded_df)
            # Generate recommendations and store the resulting dictionary explicitly
            recommendations = uploaded_df.recommendation
            # Store the recommendation options (e.g., Occurrence, Correlation, Temporal)
            rec_options = [{'label': key, 'value': key} for key in recommendations]
            # Return the output for file upload
            return_file_upload = (
                html.Div([
                    html.H5(f"Uploaded File: {filename}"),
                    dbc.Table.from_dataframe(uploaded_df.head(), striped=True, bordered=True, hover=True)
                ]),
                {'display': 'none'},  # Hide dropdown
                {'display': 'none'},  # Hide recommendations container
                [],
                rec_options
            )

    # Handle recommendation displaying
    if n_clicks and uploaded_df is not None:
        if recommendations:
            # Ensure drop_value is valid; default to the first option if not
            if drop_value not in recommendations:
                drop_value = rec_options[0]['value']

            # Access specified recommendation group
            selected_recommendations = recommendations[drop_value]

            if selected_recommendations:
                graph_components = []

                for i, vis in enumerate(selected_recommendations):

                    # Get the relevant column names
                    selected_cols = extract_vis_columns(vis)
                    
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

                        # Append the graph as a Dash Graph component and wrap it in Div to track clicks
                        graph_components.append(
                            html.Div(
                                children=[
                                    dcc.Graph(
                                        id={'type': 'dynamic-graph', 'index': i},
                                        figure=plotly_fig,
                                        style={'flex': '1 0 30%', 'margin': '5px'}
                                    )
                                ],
                                id={'type': 'graph-container', 'index': i, 'columns': str(selected_cols)},  # Store columns in ID
                                style={'cursor': 'pointer'},  # Indicate clickability
                                n_clicks=0  # Track clicks
                            )
                        )
                    except ValueError as e:
                        # error_message = str(e)
                        # If an error occurs, display the static Matplotlib image instead
                        print("Error during mpl_to_plotly conversion, falling back to displaying a static image.")

                        # Create the styled Matplotlib figure
                        fallback_fig = create_styled_matplotlib_figure(fig)

                        # Convert Matplotlib figure to base64 image
                        img_src = fig_to_base64(fallback_fig)

                        # Append the image as an Img component and wrap it in Div to track clicks
                        graph_components.append(
                            html.Div(
                                children=[
                                    html.Img(
                                        id={'type': 'image', 'index': i}, 
                                        src=img_src,
                                        style={'flex': '1 0 27%', 'margin': '5px'}
                                    )
                                ],
                                id={'type': 'graph-container', 'index': i, 'columns': str(selected_cols)},  # Store columns ID
                                style={'cursor': 'pointer'},  # Indicate clickability
                                n_clicks=0  # Track clicks
                            )
                        )

                # Return all Graph components inside a flexbox container
                return (
                    html.Div([
                        html.H5(f"Uploaded File: {filename}"),
                        dbc.Table.from_dataframe(uploaded_df.head(), striped=True, bordered=True, hover=True)
                    ]),
                    {'display': 'block'},  # Show dropdown
                    {'display': 'block'},  # Show recommendations container
                    html.Div(
                        children=graph_components,
                        style={
                            'display': 'flex',
                            'flexWrap': 'wrap',
                            'justifyContent': 'space-around',
                            'margin': '5px'
                        }
                    ),
                    rec_options
                )
    else:
        return return_file_upload      

# Callback to choose which recommendations to display
@app.callback(
    Output(component_id='rec-output-container', component_property='children'),
    Input(component_id='rec-dropdown', component_property='value')
)
def update_rec_option(value):
    if value is None:
        value = ""
    return f'Showing {value} recommendations'

# Callback to handle graph clicks
@app.callback(
    [Output(component_id='vis-selection-output', component_property='children'),
     Output(component_id='vis-selection-output', component_property='style'),
     Output(component_id='enhance-button', component_property='style')],
    [Input(component_id={'type': 'graph-container', 'index': ALL, 'columns': ALL}, component_property='n_clicks')],
    [State(component_id={'type': 'graph-container', 'index': ALL, 'columns': ALL}, component_property='id')],
    prevent_initial_call=True
)
def handle_graph_click(n_clicks_list, component_ids):
    global figure_clicks
    global selected_columns
    # Find which graph was clicked by finding the difference between the global figure_clicks list and the new n_clicks_list
    if len(figure_clicks) == len(n_clicks_list):
        arr1 = np.array(figure_clicks)
        arr2 = np.array(n_clicks_list)
        clicked_index = int(np.where(arr1 != arr2)[0][-1])

        # Extract the selected columns, and display them to the console and the dashboard user
        selected_columns = component_ids[clicked_index]['columns']
        print("Selected Columns:", selected_columns)
        # Reset figure_clicks to prepare for the identification of the next click to be added to n_clicks_list
        figure_clicks = n_clicks_list
        return f"Selected Graph Columns: {selected_columns}", {'display': 'block'}, {'display': 'block'}     

    # Reset figure_clicks to prepare for the identification of the next click to be added to n_clicks_list
    figure_clicks = n_clicks_list
    return dash.no_update, {'display': 'none'}, dash.no_update

# Callback to handle 'Enhance' button clicks
@app.callback(
    Output(component_id='enhanced-output', component_property='children'),
    Input(component_id='enhance-button', component_property='n_clicks')
)
def handle_enhance_click(n_clicks):
    global selected_columns
    global uploaded_df

    if n_clicks and uploaded_df is not None:
        # Extract the selected columns from the stored selected_columns string
        selected_cols_clean = selected_columns.replace('(', '').replace(')', '').replace("'", "").replace(',', '')
        selected_column_tuple = tuple(selected_cols_clean.split())
        # Specify intent based on the selected columns
        uploaded_df.intent = [selected_column_tuple[0], selected_column_tuple[1]]
        # Generate new recommendations and store the resulting dictionary
        recommendations = uploaded_df.recommendation
        print(recommendations)
        graph_components = []
        if recommendations:
            for selected_recommendations in recommendations.values():
                for i, vis in enumerate(selected_recommendations):

                    # # Get the relevant column names
                    # selected_cols = extract_vis_columns(vis)
                    
                    # Initialise variables that will be specified in the fig_code 
                    fig, ax = plt.subplots()

                    # Render the visualisation using Lux
                    try:
                        fig_code = vis.to_matplotlib()
                    except ValueError as e:
                        print(f"Error in vis.to_matplotlib(): {e}. Handling manually.")
                        # # Manually create the scatter plot if vis.to_matplotlib() fails
                        # scatter = ax.scatter(x_pts, y_pts, c=colors, cmap=Set1, alpha=0.5)

                        # # Add a colorbar manually
                        # cbar = plt.colorbar(scatter, ax=ax)
                        # cbar.set_label('Previous Accidents')
                    fixed_fig_code = fix_lux_code(fig_code)
                    exec(fixed_fig_code)

                    # Capture the current Matplotlib figure
                    fig = plt.gcf()
                    plt.draw()

                    # Try to convert Matplotlib figure to Plotly
                    try:
                        plotly_fig = mpl_to_plotly(fig)

                        # plotly_fig.update_layout(width=1000, height=600)

                        # Append the graph as a Dash Graph component
                        graph_components.append(
                            dcc.Graph(
                                id={'type': 'dynamic-graph', 'index': i},
                                figure=plotly_fig,
                                style={'flex': '1 0 30%', 'margin': '5px'}
                            )
                        )
                    except ValueError as e:
                        # error_message = str(e)
                        # If an error occurs, display the static Matplotlib image instead
                        print("Error during mpl_to_plotly conversion, falling back to displaying a static image.")

                        # Create the styled Matplotlib figure
                        fallback_fig = create_styled_matplotlib_figure(fig)

                        # Convert Matplotlib figure to base64 image
                        img_src = fig_to_base64(fallback_fig)

                        # Append the image as an Img component
                        graph_components.append(
                            html.Img(
                                id={'type': 'image', 'index': i}, 
                                src=img_src,
                                style={'flex': '1 0 27%', 'margin': '5px'}
                            )
                        )

                # Return all Graph components inside a flexbox container
                return (
                    html.Div(
                        children=graph_components,
                        style={
                            'display': 'flex',
                            'flexWrap': 'wrap',
                            'justifyContent': 'space-around',
                            'margin': '5px'
                        }
                    )
                )
    else:
        return dash.no_update



# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
