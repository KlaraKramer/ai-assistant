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
from duplicate_detection import *
from classes.vis import Vis
from classes.graph_component import Graph_component

# Add locally cloned Lux source code to path, and import Lux from there
sys.path.insert(0, os.path.abspath('./lux'))
import lux

# Use non-interactive backend
matplotlib.use('Agg')  

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Global variables to keep track of progress
stage = 'data-loading'
step = 0

# Global variables to store the original uploaded DataFrame and the current state of it
uploaded_df = None
current_df = None

# Global variable to store the name of the file currently being used
file_name = None

# Global variable to store Vis objects, including the indices corresponding to the figures they are displayed in
vis_objects = []

# Global variable to store components of the various sections of the pipeline
dups_count = 0
outlier_count = 0

# Global variable to store the n_clicks_list for figures
figure_clicks = []

# Global variable to store selected figure id
selected_id = None

# Global variable to store selected columns from clicking on a figure
selected_columns = ()

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
                dbc.NavLink('Duplicate removal', href='#duplicate-output', active='exact', id='progress-duplicate', style={'background-color': 'red', 'color': 'white'}),
                dbc.NavLink('Outlier handling', href='#outlier-output', active='exact', id='progress-outlier', style={'background-color': 'red', 'color': 'white'}),
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

        # Placeholder for uploaded data and initial visualisations
        html.Div(
            id='output-data-upload', 
            className='my-4'
        ),

        dbc.Button(
            'Start Data Engineering Process',
            id='start-button',
            className='btn btn-success',
            style={'display': 'none'}
        ),

        html.Div([
            html.Div(
                id='duplicate-output', 
                className='mt-4',
                children=[]
            ),
            html.Div(
                id='duplicate-output-1', 
                className='mt-4',
                children=[]
            ),
            # html.Div(id='vis-selection-output'),
            # dbc.Button(
            #     'Enhance',
            #     id='enhance-button',
            #     className='btn btn-success',
            #     style={'display': 'none'}
            # ),
            # html.Div(id='enhanced-output', className='mt-4'),
            dbc.Button(
                'Finish Duplicate Removal',
                id='duplicate-end-btn',
                className='btn btn-success',
                style={'display': 'none'}
            ),
            html.Div(
                id='outlier-output', 
                className='mt-4',
                children=[]
            ),
            html.Div(
                id='outlier-output-1', 
                className='mt-4',
                children=[]
            ),
            dbc.Button(
                'Finish Outlier Handling',
                id='outlier-end-btn',
                className='btn btn-success',
                style={'display': 'none'}
            ),
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
    Output(component_id='start-button', component_property='style')],
    [Input(component_id='upload-data', component_property='contents')],
    [State(component_id='upload-data', component_property='filename')],
    prevent_initial_call=True
)
def update_ui(contents, filename):
    global uploaded_df
    global current_df
    global stage
    global step
    global vis_objects
    global file_name

    # If no data has been uploaded yet
    if contents is None:
        return html.Div('Unsupported file type.'), {'display': 'block'}#, html.Div('No recommendations available. Upload data first.')

    # Handle file upload (uploading data)
    else:
        # Parse uploaded contents
        uploaded_df = parse_contents(contents, filename)
        file_name = filename
        if uploaded_df is not None:
            step += 1
            # Enable Lux for the uploaded DataFrame
            uploaded_df = pd.DataFrame(uploaded_df)
            current_df = uploaded_df
            if 'unnamed_0' in current_df.columns:
                current_df = current_df.drop('unnamed_0', axis=1)
            graph_components = []
            # Reset global variable to empty visualisation storage
            vis_objects = []

            ## Machine View ##
            # Display a parallel coordinates plot
            vis1 = Vis(len(vis_objects), current_df, machine_view=True)
            # Populate vis_objects list for referring back to the visualisations
            vis_objects.append(vis1)
            # Append the graph, wrapped in a Div to track clicks, to graph_components
            graph1 = Graph_component(vis1)
            if graph1.div is not None:
                graph_components.append(graph1.div)
            else:
                print("No recommendations available. Please upload data first.")

            ## Human View ##
            # Display the first recommended visualisation
            vis2 = Vis(len(vis_objects), current_df)
            # Populate vis_objects list for referring back to the visualisations
            vis_objects.append(vis2)
            # Append the graph, wrapped in a Div to track clicks, to graph_components
            graph2 = Graph_component(vis2)
            if graph2.div is not None:
                graph_components.append(graph2.div)
            else:
                print("No recommendations available. Please upload data first.")

            # Return all components
            graph_div = show_side_by_side(graph_components)
            return (
                html.Div([
                    html.H5(f'Uploaded File: {filename}'),
                    dbc.Table.from_dataframe(current_df.head(), striped=True, bordered=True, hover=True),
                    graph_div
                ]), 
                {'display': 'block'}
            )


# Callback to handle the first render within the 'duplicate-removal' stage
@app.callback(
    [Output(component_id='duplicate-output', component_property='children')],
    [Input(component_id='start-button', component_property='n_clicks')],
    prevent_initial_call=True
)
def render_duplicates(n_clicks):
    global current_df
    global stage
    global step
    global vis_objects
    global dups_count

    selected_option = ''
    graph_list = []
    # First render
    if n_clicks > 0 and current_df is not None:
        stage = 'duplicate-removal'
        step += 1
        # Access the last visualisation rendered on the right (human view)
        human_previous = vis_objects[-1]
        
        ## Machine View ##
        # Display a parallel coordinates plot
        vis1 = Vis(len(vis_objects), current_df, machine_view=True)
        # Populate vis_objects list for referring back to the visualisations
        vis_objects.append(vis1)
        # Append the graph, wrapped in a Div to track clicks, to graph_list
        graph1 = Graph_component(vis1)
        if graph1.div is not None:
            graph_list.append(graph1.div)
        else:
            print('No recommendations available. Please upload data first.')

        ## Human View ##
        # Detect and visualise duplicates
        current_df, dups_count = detect_duplicates(current_df)
        right_df = current_df
        right_df.intent = [human_previous.columns]
        # Display the second visualisation
        vis2 = Vis(len(vis_objects), right_df)
        # Populate vis_objects list for referring back to the visualisations
        vis_objects.append(vis2)
        # Append the graph, wrapped in a Div to track clicks, to graph_list
        graph2 = Graph_component(vis2)
        if graph2.div is not None:
            graph_list.append(graph2.div)
        else:
            print('No recommendations available. Please upload data first.')
            
        # Return all components
        graph_div = show_side_by_side(graph_list)
        new_div = html.Div(children=[
                html.P(f'{dups_count} duplicated rows were detected'),
                graph_div,
                dcc.Dropdown(
                    placeholder='Select an action to take', 
                    id={'type': 'duplicate-removal', 'index': step},
                    options={'highlight': 'Highlight duplicated rows', 'delete': 'Delete duplicates'}
                ),
                html.P(f'{selected_option}')
            ])
        return [new_div]


# Callback to handle updates within the 'duplicate-removal' stage
@app.callback(
    [Output(component_id='duplicate-output-1', component_property='children')],
    [Input(component_id={'type': 'duplicate-removal', 'index': ALL}, component_property='value')],
    [State(component_id='start-button', component_property='n_clicks')],
    prevent_initial_call=True
)
def update_duplicates(drop_value, n_clicks):
    global current_df
    global stage
    global step
    global vis_objects
    global dups_count

    selected_option = ''
    graph_list = []
    if n_clicks > 0 and current_df is not None:
        step += 1
        # Access the last visualisation rendered on the left (second-to-last in vis_objects)
        left_previous = vis_objects[-2]

        if 'highlight' == drop_value[-1]:
            selected_option = 'Highlight duplicated rows'
            # Detect and show duplicates
            highlight_df, highlight_count = detect_duplicates(current_df, keep=False)
            highlight_df = highlight_df[highlight_df.duplicate != False]
            highlight_df = highlight_df.sort_values(by=[highlight_df.columns[2], highlight_df.columns[3], highlight_df.columns[4]])
            new_div = html.Div(children=[
                html.P(f'Selected action: {selected_option}'),
                html.P(f'{dups_count} duplicated rows were detected'),
                dbc.Table.from_dataframe(highlight_df, striped=True, bordered=True, hover=True),
                dcc.Dropdown(
                    placeholder='Select an action to take', 
                    id={'type': 'duplicate-removal', 'index': step},
                    options={'highlight': 'Highlight duplicated rows', 'delete': 'Delete duplicates'}
                )
            ])
            return [new_div]
        
        elif 'delete' == drop_value[-1]:
            selected_option = 'Delete duplicates'
            current_df = current_df[current_df.duplicate != True]
            left_df = current_df
            left_df.intent = left_previous.columns
            # Display the first recommended visualisation
            vis1 = Vis(len(vis_objects), left_df)
            # Populate vis_objects list for referring back to the visualisations
            vis_objects.append(vis1)
            # Append the graph, wrapped in a Div to track clicks, to graph_list
            graph1 = Graph_component(vis1)
            if graph1.div is not None:
                graph_list.append(graph1.div)
            else:
                print('No recommendations available. Please upload data first.')

            # Detect and visualise duplicates
            current_df, dups_count = detect_duplicates(current_df)
            right_df = current_df

            right_df.intent = ['duplicate']
            # Display the second visualisation
            vis2 = Vis(len(vis_objects), right_df)
            # Populate vis_objects list for referring back to the visualisations
            vis_objects.append(vis2)
            # Append the graph, wrapped in a Div to track clicks, to graph_list
            graph2 = Graph_component(vis2)
            if graph2.div is not None:
                graph_list.append(graph2.div)
            else:
                print('No recommendations available. Please upload data first.')
            # Return all components
            graph_div = show_side_by_side(graph_list)
            new_div = html.Div(children=[
                html.P(f'Selected action: {selected_option}'),
                html.P(f'{dups_count} duplicated rows were detected'),
                graph_div,
                dcc.Dropdown(
                    placeholder='Select an action to take', 
                    id={'type': 'duplicate-removal', 'index': step},
                    options={'highlight': 'Highlight duplicated rows', 'delete': 'Delete duplicates'}
                ),
                html.P(f'{selected_option}')
            ])
            return [new_div]
        else:
            return dash.no_update
    else:
        return dash.no_update


# Callback to handle the first render within the 'outlier-handling' stage
@app.callback(
    [Output(component_id='outlier-output', component_property='children')],
    [Input(component_id='duplicate-end-btn', component_property='n_clicks')],
    prevent_initial_call=True
)
def render_outliers(n_clicks):
    global current_df
    global stage
    global step
    global vis_objects
    global outlier_count

    selected_option = ''
    graph_list = []
    if n_clicks > 0 and current_df is not None:
        stage = 'outlier-handling'
        step += 1
        # Access the last visualisation rendered on the left (second-to-last in vis_objects)
        left_previous = vis_objects[-2]
        # First render
        left_df = current_df
        left_df.intent = left_previous.columns
        # Display the first recommended visualisation
        vis1 = Vis(len(vis_objects), left_df)              

        # Populate vis_objects list for referring back to the visualisations
        vis_objects.append(vis1)
        # Append the graph, wrapped in a Div to track clicks, to graph_list
        graph1 = Graph_component(vis1)
        if graph1.div is not None:
            graph_list.append(graph1.div)
        else:
            print('No recommendations available. Please upload data first.')

        # Detect and visualise outliers
        outlier_df, outlier_count = train_isolation_forest(current_df, intent=left_previous.columns)

        # Display the second visualisation
        vis2 = Vis(len(vis_objects), outlier_df)
        # Populate vis_objects list for referring back to the visualisations
        vis_objects.append(vis2)
        # Append the graph, wrapped in a Div to track clicks, to graph_list
        graph2 = Graph_component(vis2)
        if graph2.div is not None:
            graph_list.append(graph2.div)
        else:
            print('No recommendations available. Please upload data first.')
        # Return all components
        graph_div = show_side_by_side(graph_list)
        new_div = html.Div(children=[
                html.P(f'{outlier_count} outlier values were detected'),
                graph_div,
                dcc.Dropdown(
                    placeholder='Select an action to take', 
                    id={'type': 'outlier-handling', 'index': step},
                    options={'more': 'Find more outliers', 'less': 'Find less outliers', 'accept': 'Remove the highlighted outliers'}
                ),
                html.P(f'{selected_option}')
            ])
        return [new_div]


# # Callback to handle graph clicks
# @app.callback(
#     [Output(component_id='vis-selection-output', component_property='children'),
#      Output(component_id='vis-selection-output', component_property='style'),
#      Output(component_id='enhance-button', component_property='style')],
#     [Input(component_id={'type': 'graph-container', 'index': ALL, 'columns': ALL}, component_property='n_clicks')],
#     [State(component_id={'type': 'graph-container', 'index': ALL, 'columns': ALL}, component_property='id')],
#     prevent_initial_call=True
# )
# def handle_graph_click(n_clicks_list, component_ids):
#     global figure_clicks
#     global selected_columns
#     global selected_id
#     # Find which graph was clicked by finding the difference between the global figure_clicks list and the new n_clicks_list
#     if len(figure_clicks) == len(n_clicks_list):
#         arr1 = np.array(figure_clicks)
#         arr2 = np.array(n_clicks_list)
#         clicked_index = int(np.where(arr1 != arr2)[0][-1])

#         # Extract the selected columns, and display them to the console and the dashboard user
#         selected_columns = component_ids[clicked_index]['columns']
#         selected_id = component_ids[clicked_index]['index']
#         print('Selected Columns:', selected_columns)
#         # Reset figure_clicks to prepare for the identification of the next click to be added to n_clicks_list
#         figure_clicks = n_clicks_list
#         return f'Selected Graph Columns: {selected_columns}', {'display': 'block'}, {'display': 'block'}     

#     # Reset figure_clicks to prepare for the identification of the next click to be added to n_clicks_list
#     figure_clicks = n_clicks_list
#     return dash.no_update, {'display': 'none'}, dash.no_update

# # Callback to handle 'Enhance' button clicks
# @app.callback(
#     Output(component_id='enhanced-output', component_property='children'),
#     Input(component_id='enhance-button', component_property='n_clicks')
# )
# def handle_enhance_click(n_clicks):
#     # global selected_columns
#     global uploaded_df
#     global vis_objects
#     global selected_id

#     if n_clicks and uploaded_df is not None:
#         # Extract the selected visualisation from the stored vis_objects and specify Lux intent
#         vis = vis_objects[selected_id]
#         uploaded_df.intent = vis
#         graph_components = []

#         # Display the first recommended visualisation
#         vis1 = Vis(len(vis_objects), uploaded_df)
#         # Populate vis_objects dictionary for referring back to the visualisations
#         vis_objects[vis1.id] = vis1.lux_vis
#         # Append the graph, wrapped in a Div to track clicks, to graph_components
#         graph1 = Graph_component(vis1)
#         if graph1.div is not None:
#             graph_components.append(graph1.div)
#         else:
#             print("No recommendations available. Please upload data first.")


#         ### TO-DO: Add second visualisation here ###


#         # Return all Graph components inside a flexbox container
#         return (
#             html.Div(
#                 children=graph_components,
#                 style={
#                     'display': 'flex',
#                     'flexWrap': 'wrap',
#                     'justifyContent': 'space-around',
#                     'margin': '5px'
#                 }
#             )
#         )

#     else:
#         return dash.no_update

# Callback to update progress
@app.callback(
    [Output(component_id='progress-load', component_property='style'),
     Output(component_id='progress-duplicate', component_property='style'),
     Output(component_id='progress-outlier', component_property='style'),
     Output(component_id='duplicate-end-btn', component_property='style'),
     Output(component_id='outlier-end-btn', component_property='style')],
    [Input(component_id='upload-data', component_property='contents'),
     Input(component_id='duplicate-end-btn', component_property='n_clicks'),
     Input(component_id='outlier-end-btn', component_property='n_clicks')],
    prevent_initial_call=True
)
def update_progress(contents, click_dup, click_out):
    ctx = dash.callback_context
    # Default colours and display values
    load_colour, dup_colour, out_colour = 'red', 'red', 'red'
    dup_style, out_style = {'display': 'block'}, {'display': 'none'}

    # If buttons are clicked, change the respective progress bars
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'duplicate-end-btn' in changed_id:
        load_colour = 'green'
        dup_colour = 'green'
        out_colour = 'red'
        dup_style = {'display': 'block'}
        out_style = {'display': 'block'}
    if 'outlier-end-btn' in changed_id:
        load_colour = 'green'
        dup_colour = 'green'
        out_colour = 'green'
        dup_style = {'display': 'block'}
        out_style = {'display': 'block'}
    
    # If a new file is uploaded, reset dup_colour and out_colour to 'red'
    if ctx.triggered and 'upload-data' in ctx.triggered[0]['prop_id']:
        load_colour = 'green'
        dup_colour = 'red'
        out_colour = 'red'
        dup_style = {'display': 'block'}
        out_style = {'display': 'none'}
    
    return (
        {'background-color': load_colour, 'color': 'white'},
        {'background-color': dup_colour, 'color': 'white'},
        {'background-color': out_colour, 'color': 'white'},
        dup_style,
        out_style
    )


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
