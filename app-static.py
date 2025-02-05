import dash
from dash import dcc, html, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import Set1
from mpl_toolkits.axes_grid1 import make_axes_locatable
import altair as alt
import mpld3    
from plotly.tools import mpl_to_plotly
import sys
import os
import numpy as np

from helper_functions import *
from outlier_isolation_forest import *
from classes.vis import Vis
from classes.graph_component import Graph_component

# Add locally cloned Lux source code to path, and import Lux from there
sys.path.insert(0, os.path.abspath('./lux'))
import lux

# Use non-interactive backend
matplotlib.use('Agg')  

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# # Global variable to store recommendation options
# rec_options = []

# Global variable to store the n_clicks_list for figures
figure_clicks = []

# Global variable to store the uploaded DataFrame
uploaded_df = None

# Global variable to store selected figure id
selected_id = None

# Global variable to store selected columns from clicking on a figure
selected_columns = ()

# Global variable to store Lux Vis objects and the indices corresponding to the figures they are displayed in
vis_objects = {}

# The following style items were adapted from https://github.com/Coding-with-Adam/Dash-by-Plotly/blob/master/Bootstrap/Side-Bar/side_bar.py 
# styling the progress bar
PROGRESS_BAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '12rem',
    'padding': '2rem 1rem',
    'background-color': '#333333',
}

# padding for the main dashboard
DASHBOARD_STYLE = {
    'margin-left': '8rem'
}

progress_bar = html.Div(
    [
        html.H2('Progress', className='display-6', style={'color': 'white'}),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink('Data loading', href='#output-data-upload', active='exact', id='progress-load', style={'background-color': 'red', 'color': 'white'}),
                dbc.NavLink('Duplicate removal', href='#lux-output', active='exact', id='progress-duplicate', style={'background-color': 'red', 'color': 'white'}),
                dbc.NavLink('Outlier handling', href='#outlier', active='exact', id='progress-outlier', style={'background-color': 'red', 'color': 'white'}),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=PROGRESS_BAR_STYLE,
)

dashboard = html.Div(id='dashboard', children=[
    dbc.Container([
        html.H1('Visual Data Engineering', className='text-center my-4'),

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
        html.Div(id='output-data-upload', className='my-4'),

        html.Div([
            # # Button to trigger Lux recommendations
            # dbc.Button(
            #     'Show Recommendations', 
            #     id='show-recs', 
            #     color='primary', 
            #     className='mt-2'
            # ),
            # # Dropdown to choose recommendation option (initially hidden)
            # dcc.Dropdown(
            #     placeholder='Select a recommendation option', 
            #     id='rec-dropdown',
            #     style={'display': 'none'}  # Initially hidden
            # ),
            # html.Div(id='rec-output-container', style={'display': 'none'}),
            html.Div(id='lux-output', className='mt-4'),
            html.Div(id='vis-selection-output'), # , style={'display': 'none'}
            dbc.Button(
                'Enhance',
                id='enhance-button',
                className='btn btn-success',
                style={'display': 'none', 'margin-left': '10px'}
            ),
            html.Div(id='enhanced-output', className='mt-4')
        ])
    ]) # style={'justify': 'center', 'align': 'center'})
], style=DASHBOARD_STYLE)

# Set dashboard layout
app.layout = dbc.Container([
    dcc.Location(id='url'),
    progress_bar,
    dashboard,
])

# Callback to handle the file upload
@app.callback(
    [Output(component_id='output-data-upload', component_property='children'),
     Output(component_id='lux-output', component_property='children')],
    [Input(component_id='upload-data', component_property='contents')],
    [State(component_id='upload-data', component_property='filename')],
    prevent_initial_call=True
)
def update_ui(contents, filename):
    global uploaded_df
    # global rec_options
    global vis_objects

    # If no data has been uploaded yet
    if contents is None:
        return html.Div('Unsupported file type.'), html.Div('No recommendations available. Upload data first.')

    # Handle file upload (uploading data)
    else:
        # Parse uploaded contents
        uploaded_df = parse_contents(contents, filename)
        if uploaded_df is not None:
            # Enable Lux for the uploaded DataFrame
            uploaded_df = pd.DataFrame(uploaded_df)
            graph_components = []

            # Display the first recommended visualisation
            vis1 = Vis(len(vis_objects), uploaded_df)
            # Populate vis_objects dictionary for referring back to the visualisations
            vis_objects[vis1.id] = vis1.lux_vis
            # Append the graph, wrapped in a Div to track clicks, to graph_components
            graph1 = Graph_component(vis1)
            if graph1.div is not None:
                graph_components.append(graph1.div)
            else:
                print("No recommendations available. Please upload data first.")

                    
            ### TO-DO: Add second visualisation here ###


            # Return all Graph components inside a flexbox container
            return (
                html.Div([
                    html.H5(f'Uploaded File: {filename}'),
                    dbc.Table.from_dataframe(uploaded_df.head(), striped=True, bordered=True, hover=True)
                ]),
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

# # Callback to choose which recommendations to display
# @app.callback(
#     Output(component_id='rec-output-container', component_property='children'),
#     Input(component_id='rec-dropdown', component_property='value')
# )
# def update_rec_option(value):
#     if value is None:
#         value = ''
#     return f'Showing {value} recommendations'

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
    global selected_id
    # Find which graph was clicked by finding the difference between the global figure_clicks list and the new n_clicks_list
    if len(figure_clicks) == len(n_clicks_list):
        arr1 = np.array(figure_clicks)
        arr2 = np.array(n_clicks_list)
        clicked_index = int(np.where(arr1 != arr2)[0][-1])

        # Extract the selected columns, and display them to the console and the dashboard user
        selected_columns = component_ids[clicked_index]['columns']
        selected_id = component_ids[clicked_index]['index']
        print('Selected Columns:', selected_columns)
        # Reset figure_clicks to prepare for the identification of the next click to be added to n_clicks_list
        figure_clicks = n_clicks_list
        return f'Selected Graph Columns: {selected_columns}', {'display': 'block'}, {'display': 'block'}     

    # Reset figure_clicks to prepare for the identification of the next click to be added to n_clicks_list
    figure_clicks = n_clicks_list
    return dash.no_update, {'display': 'none'}, dash.no_update

# Callback to handle 'Enhance' button clicks
@app.callback(
    Output(component_id='enhanced-output', component_property='children'),
    Input(component_id='enhance-button', component_property='n_clicks')
)
def handle_enhance_click(n_clicks):
    # global selected_columns
    global uploaded_df
    global vis_objects
    global selected_id

    if n_clicks and uploaded_df is not None:
        # Extract the selected visualisation from the stored vis_objects and specify Lux intent
        vis = vis_objects[selected_id]
        uploaded_df.intent = vis
        # Generate new recommendations and store the resulting dictionary
        recommendations = uploaded_df.recommendation
        graph_components = []
        if recommendations:
            for selected_recommendations in recommendations.values():
                # Populate vis_objects dictionary for referring back to the visualisations
                i = len(vis_objects)
                vis = selected_recommendations[0]
                
                # Initialise variables that will be specified in the fig_code 
                fig, ax = plt.subplots()
                tab20c = plt.get_cmap('tab20c')

                # Render the visualisation using Lux
                try:
                    fig_code = vis.to_matplotlib()
                except ValueError:
                    print('Error in to_matplotlib()')
                    fig_code = ''
                fixed_fig_code = fix_lux_code(fig_code)
                exec(fixed_fig_code)

                # Capture the current Matplotlib figure
                fig = plt.gcf()
                plt.draw()

                ax.legend(loc='upper right', bbox_to_anchor=(1, 1))

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
                except ValueError:
                    # If an error occurs, display the static Matplotlib image instead
                    print('Error during mpl_to_plotly conversion, falling back to displaying a static image.')

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
