###################################################################################################
### This is the main executable app file of the Visual Data Wizard application                  ###
###                                                                                             ###
### It specifies the application’s layout, generates and displays buttons and dropdown          ###
### menus for user interaction, and calls the individual data-cleaning functions where required ###
### Sections:                                                                                   ###
### 1. Imports                                                                                  ###
### 2. Global variable and function definitions                                                 ###
### 3. Definition of the dashboard's layout and UI components                                   ###
### 4. Main application logic: Event-driven callback functions                                  ###
### 5. Callback functions for styling                                                           ###
### 6. Callback functions for triggering the downloads                                          ###
### 7. Functionality for running the app                                                        ###
###                                                                                             ###
### The callback functions to handle event-driven UI updates form the largest part of this file ###
### and the application's logic in general                                                      ###
###################################################################################################


###############
### Imports ###

# External imports
import dash
from dash import dcc, html, Input, Output, State, ALL, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import matplotlib
from matplotlib.cm import Set1
import altair as alt
import mpld3
from sklearn import set_config
from flask import Flask, send_file
import sys
import os
import numpy as np
matplotlib.use('Agg')  

# Internal imports
from helper_functions import *
from backend_magic.outlier_isolation_forest import *
from backend_magic.duplicate_detection import *
from backend_magic.missing_value_detection import *
from classes.vis import Vis
from classes.graph_component import Graph_component
# Add locally cloned Lux source code to path, and import Lux from there
sys.path.insert(0, os.path.abspath('./lux'))
import lux


################################################
### Global variable and function definitions ###

# Initialise Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Global variables to keep track of progress
stage = 'data-loading'
step = 0
action_log = []
download_completion = [0, 0]
# Default colours and display values
load_colour, miss_colour, dup_colour, out_colour, down_colour = 'red', 'red', 'red', 'red', 'red'
missing_style, dup_style, out_style, info_style, download_style, down_info_style, completion_style = {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

# Global variables to store the original uploaded DataFrame, the current state of it, and the previous saved state
uploaded_df = None
current_df = None
previous_df = None

# Global variable to store the name of the file currently being used
file_name = None

# Global variable to store Vis objects, including the indices corresponding to the figures they are displayed in
vis_objects = []

# Global variables to store components of the various sections of the pipeline
dups_count = 0
missing_count = 0
outlier_count = 0
outlier_contamination_history = []

# Display a parallel coordinates plot
def render_machine_view(vis_objects, df, graph_components):
    vis1 = Vis(len(vis_objects), df, machine_view=True)
    # Populate vis_objects list for referring back to the visualisations
    vis_objects.append(vis1)
    # Append the graph, wrapped in a Div to track clicks, to graph_components
    graph1 = Graph_component(vis1)
    if graph1.div is not None:
        graph_components.append(graph1.div)
    return vis_objects, graph_components

# Create an entry in the action log
def log(message, type):
    global action_log
    entry = ''
    if type == 'system':
        entry = 'SYSTEM NOTE: ' + message
    elif type == 'user':
        entry = 'USER ACTION: ' + message
    action_log.append(entry)


##############################################################
### Definition of the dashboard's layout and UI components ###

# The following style items were adapted from https://github.com/Coding-with-Adam/Dash-by-Plotly/blob/master/Bootstrap/Side-Bar/side_bar.py 
# Styling the progress bar
PROGRESS_BAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '12rem',
    'padding': '2rem 1rem',
    'background-color': '#333333',
}

# Padding for the main dashboard
DASHBOARD_STYLE = {
    'margin-left': '8rem'
}

# Define layout of sidebar
progress_bar = html.Div(
    [
        html.H2('Progress', className='display-6', style={'color': 'white'}),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink('Data loading', href='#output-data-upload', active='exact', id='progress-load', style={'background-color': 'red', 'color': 'white'}),
                dbc.NavLink('Missing value handling', href='#missing-output', active='exact', id='progress-missing', style={'background-color': 'red', 'color': 'white'}),
                dbc.NavLink('Duplicate removal', href='#duplicate-output', active='exact', id='progress-duplicate', style={'background-color': 'red', 'color': 'white'}),
                dbc.NavLink('Outlier handling', href='#outlier-output', active='exact', id='progress-outlier', style={'background-color': 'red', 'color': 'white'}),
                dbc.NavLink('Downloading', href='#download-header', active='exact', id='progress-download', style={'background-color': 'red', 'color': 'white'}),
            ],
            vertical=True,
            pills=True,
        ),
        html.Hr(),
        html.Div(
            children=[
                html.P('Congratulations on finishing the data cleaning.', style={'font': 'bold', 'color': 'white'}),
                html.P('Please reload the page to upload a new dataset.', style={'font': 'bold', 'color': 'white'})
            ],
            id='completion-message',
            style={'display': 'none'}
        )
    ],
    style=PROGRESS_BAR_STYLE,
)

# Define layout of main dashboard
dashboard = html.Div(id='dashboard', children=[
    dbc.Container([
        html.H1('Visual Data Wizard', className='text-center my-4'),

        # File upload
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
        # Preloaded dataset selection
        html.P('Or choose one of the preloaded datasets from the menu below:'),
        dcc.Dropdown(
            placeholder='Select a preloaded dataset', 
            id='dataset-selection',
            options={
                'corrupted_iris.csv': 'Iris Flower',
                'corrupted_energy.csv': 'Energy Consumption',
                'corrupted_car.csv': 'Car Insurance'
            }
        ),
        html.Br(),

        # Placeholder for uploaded data and initial visualisations
        html.Div(
            id='output-data-upload', 
            className='my-4'
        ),

        # Process start button
        dbc.Button(
            'Start Data Engineering Process',
            id='start-button',
            className='btn btn-success',
            style={'display': 'none'}
        ),

        # Beginning of the data engineering pipeline
        html.Div([
            # Beginning of the missing value handling stage
            html.Div(
                id='missing-output', 
                className='mt-4',
                children=[]
            ),
            html.Div(
                id='missing-output-1', 
                className='mt-4',
                children=[]
            ),
            dbc.Button(
                'Finish Missing Value Handling',
                id='missing-end-btn',
                className='btn btn-success',
                style={'display': 'none'}
            ),
            # Beginning of the duplicate detection stage
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
            dbc.Button(
                'Finish Duplicate Removal',
                id='duplicate-end-btn',
                className='btn btn-success',
                style={'display': 'none'}
            ),
            # Beginning of the outlier handling stage
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
            html.Div(
                id='outlier-output-2', 
                className='mt-4',
                children=[]
            ),
            html.Div(
                id='outlier-output-3', 
                className='mt-4',
                children=[]
            ),
            # Process completion button
            dbc.Button(
                'Finish Outlier Handling',
                id='outlier-end-btn',
                className='btn btn-success',
                style={'display': 'none'}
            ),
            # Process completion message
            html.P('You have completed the data engineering process. Please now use the below buttons to download the cleaned data and the action log.', 
                   id='out-end-info', 
                   style={'display': 'none'}
            )
        ]),

        html.Br(),
        html.Br(),
        html.H5('Download Section', id='download-header', style={'display': 'none'}),
        # The below message is only for evaluation purposes
        html.P(
            'Please use these buttons only at the end of the process, or if your time runs out.', 
            id='download-info',
            style={'display': 'none'}
        ),
        # Download buttons and outputs
        dbc.Button(
            'Download Cleaned Data',
            id='csv-btn',
            className='btn btn-success',
            style={'display': 'none'}
        ),
        dcc.Download(id='download-dataframe-csv'),
        html.Br(),
        dbc.Button(
            'Download Action Log',
            id='download-btn',
            className='btn btn-success',
            style={'display': 'none'}
        ),
        dcc.Download(id='download-log'),
        html.Br()
    ])
], style=DASHBOARD_STYLE)

# Set dashboard layout
app.layout = dbc.Container([
    dcc.Location(id='url'),
    progress_bar,
    dashboard,
])


#####################################################################################
### Main application logic:                                                       ###
### Event-driven callback functions that constitute the data engineering pipeline ###

# Callback to handle the file upload
@app.callback(
    [Output(component_id='output-data-upload', component_property='children'),
     Output(component_id='start-button', component_property='style')],
    [Input(component_id='upload-data', component_property='contents'),
     Input(component_id='dataset-selection', component_property='value')],
    [State(component_id='upload-data', component_property='filename')],
    prevent_initial_call=True
)
def update_ui(contents, selected_dataset, filename):
    global uploaded_df
    global current_df
    global previous_df
    global stage
    global step
    global vis_objects
    global file_name
    global dups_count
    global outlier_count
    global action_log

    if contents is None:
        # Handle selection of preloaded datasets
        if selected_dataset and len(selected_dataset) > 0:
            file_name = selected_dataset
            filename = selected_dataset
            uploaded_df = prepare_contents(file_name)
        else:
            # If no data has been uploaded yet
            log('Unsupported file type', 'system')
            return html.Div('Unsupported file type.'), {'display': 'block'}
    # Handle file upload (uploading data)
    else:
        # Parse uploaded contents
        uploaded_df = parse_contents(contents, filename)
        file_name = filename
    stage = 'data-loading'
    step = 0
    if uploaded_df is not None:
        step += 1
        # Enable Lux for the uploaded DataFrame
        uploaded_df = pd.DataFrame(uploaded_df)
        if 'unnamed_0' in uploaded_df.columns:
            uploaded_df = uploaded_df.drop('unnamed_0', axis=1)
        current_df = uploaded_df.copy()
        previous_df = uploaded_df.copy()
        graph_components = []
        # Reset global variables
        vis_objects = []
        dups_count = 0
        outlier_count = 0
        action_log = []

        # Display a parallel coordinates plot
        vis_objects, graph_components = render_machine_view(vis_objects, uploaded_df, graph_components)

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

        log('Data uploaded', 'system')
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
    else:
        return dash.no_update


# Callback to handle the first render within the 'missing-value-handling' stage
@app.callback(
    [Output(component_id='missing-output', component_property='children')],
    [Input(component_id='start-button', component_property='n_clicks')],
    prevent_initial_call=True,
    running=[(Output(component_id='missing-end-btn', component_property='disabled'), True, False)]
)
def render_missing_values(n_clicks):
    global current_df
    global stage
    global step
    global vis_objects
    global missing_count
    global file_name

    if n_clicks > 0 and current_df is not None:
        stage = 'missing-value-handling'
        step += 1
        file_name = determine_filename(file_name)

        # Call the backend function for missing value detection
        missing_df, missing_count = detect_missing_values(current_df)

        if isinstance(missing_df, pd.Series):
            missing_df = missing_df.to_frame()

        message = str(missing_count) + ' missing values were detected'
        log(message, 'system')

        # Add new div to the UI
        if missing_count == 0:
            new_div = html.Div(children=[
                html.P(message, style={'color': 'green'}),
            ])
        else:
            new_div = html.Div(children=[
                html.P(message, style={'color': 'red'}),
                dbc.Table.from_dataframe(missing_df, striped=True, bordered=True, hover=True),
                dcc.Dropdown(
                    placeholder='Select an action to take', 
                    id={'type': 'missing-value-removal', 'index': step},
                    options={
                        'highlight': 'Show rows with missing values', 
                        'delete': 'Delete rows with missing values', 
                        'impute-simple': 'Impute missing values using the univariate mean', 
                        'impute-KNN': 'Impute missing values using the k nearest neighbours'
                    }
                )
            ])
        return [new_div]
    

# Callback to handle updates within the 'missing-value-handling' stage
@app.callback(
    [Output(component_id='missing-output-1', component_property='children')],
    [Input(component_id={'type': 'missing-value-removal', 'index': ALL}, component_property='value')],
    [State(component_id='start-button', component_property='n_clicks')],
    prevent_initial_call=True,
    running=[(Output(component_id='missing-end-btn', component_property='disabled'), True, False),
             (Output(component_id={'type': 'missing-value-removal', 'index': ALL}, component_property='disabled'), True, False)]
)
def update_missing_values(drop_value, n_clicks):
    global current_df
    global previous_df
    global stage
    global step
    global vis_objects
    global missing_count

    selected_option = ''
    graph_list = []
    if drop_value[-1] is None:
        return dash.no_update
    if n_clicks > 0 and current_df is not None:
        step += 1

        if 'highlight' == drop_value[-1]:
            selected_option = 'Show rows with missing values'
            # Detect and show rows with missing values
            highlight_df = current_df[current_df.isnull().any(axis=1)]
            new_div = html.Div(children=[
                html.P(f'Selected action: {selected_option}'),
                html.P(f'{missing_count} missing values were detected', style={'color': 'red'}),
                dbc.Table.from_dataframe(highlight_df, striped=True, bordered=True, hover=True),
                dcc.Dropdown(
                    placeholder='Select an action to take', 
                    id={'type': 'missing-value-removal', 'index': step},
                    options={
                        'delete': 'Delete rows with missing values', 
                        'impute-simple': 'Impute missing values using the univariate mean', 
                        'impute-KNN': 'Impute missing values using the k nearest neighbours'
                    }
                )
            ])
            return [new_div]
        elif 'impute-simple' == drop_value[-1]:
            # Use the backend univariate mean imputer
            selected_option = 'Impute missing values using the univariate mean'
            current_df = impute_missing_values(current_df)
        elif 'impute-KNN' == drop_value[-1]:
            # Use the backend KNN imputer
            selected_option = 'Impute missing values using the k nearest neighbours'
            current_df = impute_missing_values(current_df, 'KNN')
        elif 'delete' == drop_value[-1]:
            # Remove missing values
            selected_option = 'Delete rows with missing values'
            current_df = current_df[current_df.notnull().all(axis=1)]
        elif 'undo' == drop_value[-1]:
            # Revert dataframe back to its previous state
            selected_option = 'Undo the last step'
            current_df = previous_df.copy()
        else:
            return dash.no_update

        missing_df, missing_count = detect_missing_values(current_df)
        # Display a parallel coordinates plot
        vis_objects, graph_list = render_machine_view(vis_objects, current_df, graph_list)

        # Display the first recommended visualisation
        vis2 = Vis(len(vis_objects), current_df) #, num_rec=1
        # Populate vis_objects list for referring back to the visualisations
        vis_objects.append(vis2)
        # Append the graph, wrapped in a Div to track clicks, to graph_list
        graph2 = Graph_component(vis2)
        if graph2.div is not None:
            graph_list.append(graph2.div)
        else:
            print('No recommendations available. Please upload data first.')
        log(selected_option, 'user')
        message = str(missing_count) + ' missing values were detected'
        log(message, 'system')
        # Return all components
        graph_div = show_side_by_side(graph_list)
        if 'undo' == drop_value[-1]:
            new_div = html.Div(children=[
                html.P(f'Selected action: {selected_option}'),
                html.P(message, style={'color': 'red'}),
                graph_div,
                dcc.Dropdown(
                    placeholder='Select an action to take', 
                    id={'type': 'missing-value-removal', 'index': step},
                    options={
                        'highlight': 'Show rows with missing values', 
                        'delete': 'Delete rows with missing values', 
                        'impute-simple': 'Impute missing values using the univariate mean', 
                        'impute-KNN': 'Impute missing values using the k nearest neighbours'
                    }
                )
            ])
        else:
            new_div = html.Div(children=[
                html.P(f'Selected action: {selected_option}'),
                html.P(message, style={'color': 'green'}),
                graph_div,
                dcc.Dropdown(
                    placeholder='Select an action to take', 
                    id={'type': 'missing-value-removal', 'index': step},
                    options={
                        'undo': 'Undo the last step'
                    }
                ),
                html.Br()
            ])
        return [new_div]
        
    else:
        return dash.no_update


# Callback to handle the first render within the 'duplicate-removal' stage
@app.callback(
    [Output(component_id='duplicate-output', component_property='children')],
    [Input(component_id='missing-end-btn', component_property='n_clicks')],
    prevent_initial_call=True,
    running=[(Output(component_id='duplicate-end-btn', component_property='disabled'), True, False)]
)
def render_duplicates(n_clicks):
    global current_df
    global previous_df
    global stage
    global step
    global vis_objects
    global dups_count

    graph_list = []
    # First render
    if n_clicks > 0 and current_df is not None:
        log('Finish Missing Value Handling', 'user')
        stage = 'duplicate-removal'
        step += 1
        previous_df = current_df.copy()
        # Access the last visualisation rendered on the right for intent specification
        human_previous = vis_objects[-1]
        
        # Display a parallel coordinates plot
        vis_objects, graph_list = render_machine_view(vis_objects, current_df, graph_list)

        # Detect and visualise duplicates
        current_df, dups_count = detect_duplicates(current_df)
        right_df = current_df.copy()
        right_df.intent = extract_intent(human_previous.columns)
        # Display the second visualisation
        vis2 = Vis(len(vis_objects), right_df, enhance='duplicate')
        # Catch the missing value error if applicable:
        if vis2.missing_value_flag:
            message = 'ERROR: Visualisations cannot be displayed due to missing values in the data. Please revisit the "Missing Value Handling" step above, and click the "Finish Missing Value Handling" button when done.'
            log(message, 'system')
            new_div = html.Div(children=[
                html.P(message, style={'color': 'red'})
            ])
            return [new_div]
        # Populate vis_objects list for referring back to the visualisations
        vis_objects.append(vis2)
        # Append the graph, wrapped in a Div to track clicks, to graph_list
        graph2 = Graph_component(vis2)
        if graph2.div is not None:
            graph_list.append(graph2.div)
        else:
            print('No recommendations available. Please upload data first.')

        # Determine colour of system message
        if dups_count == 0:
            show_dropdown = {'display': 'none'}
            text_col = {'color': 'green'}
        else:
            show_dropdown = {'display': 'block'}
            text_col = {'color': 'red'}
        message = str(dups_count) + ' duplicated rows were detected'
        log(message, 'system')
        # Return all components
        graph_div = show_side_by_side(graph_list)
        new_div = html.Div(children=[
            html.P(message, style=text_col),
            graph_div,
            dcc.Dropdown(
                placeholder='Select an action to take', 
                id={'type': 'duplicate-removal', 'index': step},
                options={'highlight': 'Show duplicated rows', 'delete': 'Delete duplicates', 'keep': 'Keep all duplicates'},
                style=show_dropdown
            ),
            html.Br()
        ])
        return [new_div]


# Callback to handle updates within the 'duplicate-removal' stage
@app.callback(
    [Output(component_id='duplicate-output-1', component_property='children')],
    [Input(component_id={'type': 'duplicate-removal', 'index': ALL}, component_property='value')],
    [State(component_id='missing-end-btn', component_property='n_clicks')],
    prevent_initial_call=True,
    running=[(Output(component_id='duplicate-end-btn', component_property='disabled'), True, False),
             (Output(component_id={'type': 'duplicate-removal', 'index': ALL}, component_property='disabled'), True, False)]
)
def update_duplicates(drop_value, n_clicks):
    global current_df
    global previous_df
    global stage
    global step
    global vis_objects
    global dups_count

    selected_option = ''
    graph_list = []
    if n_clicks > 0 and current_df is not None:
        step += 1
        # Access the last visualisation rendered on the right (human view)
        human_previous = vis_objects[-1]

        if 'highlight' == drop_value[-1]:
            selected_option = 'Highlight duplicated rows'
            # Detect and show duplicates
            highlight_df, highlight_count = detect_duplicates(current_df, keep=False)
            highlight_df = highlight_df[highlight_df.duplicate != False]
            highlight_df = highlight_df.sort_values(by=[highlight_df.columns[2], highlight_df.columns[3], highlight_df.columns[4]])
            log(selected_option, 'user')
            message = str(dups_count) + ' duplicated rows were detected'
            log(message, 'system')
            # Add a new div to the UI
            new_div = html.Div(children=[
                html.P(f'Selected action: {selected_option}'),
                html.P(message, style={'color': 'red'}),
                dbc.Table.from_dataframe(highlight_df, striped=True, bordered=True, hover=True),
                dcc.Dropdown(
                    placeholder='Select an action to take', 
                    id={'type': 'duplicate-removal', 'index': step},
                    options={'highlight': 'Show duplicated rows', 'delete': 'Delete duplicates', 'keep': 'Keep all duplicates'}
                ),
                html.Br()
            ])
            return [new_div]
        
        elif 'undo' == drop_value[-1]:
            # Revert the dataframe back to its previous state
            selected_option = 'Undo the last step'
            current_df = previous_df.copy()
        elif 'delete' == drop_value[-1]:
            # Remove duplicated rows
            selected_option = 'Delete duplicates'
            current_df = current_df[current_df.duplicate != True]
        else:
            return dash.no_update
         
        # Display a parallel coordinates plot
        vis_objects, graph_list = render_machine_view(vis_objects, current_df, graph_list)

        # Detect and visualise duplicates
        current_df, dups_count = detect_duplicates(current_df)
        right_df = current_df.copy()
        right_df.intent = extract_intent(human_previous.columns)
        # Display the second visualisation
        vis2 = Vis(len(vis_objects), right_df)
        # Catch the missing value error if applicable:
        if vis2.missing_value_flag:
            # Display the second visualisation (second recommendation - num_rec=1 - rather than the first as usual)
            temp_vis = Vis(len(vis_objects), current_df, num_rec=1, temporary=True)
            current_df.intent = extract_intent(temp_vis.columns)
            vis2 = Vis(len(vis_objects), current_df, enhance='duplicate')
        # Populate vis_objects list for referring back to the visualisations
        vis_objects.append(vis2)
        # Append the graph, wrapped in a Div to track clicks, to graph_list
        graph2 = Graph_component(vis2)
        if graph2.div is not None:
            graph_list.append(graph2.div)
        else:
            print('No recommendations available. Please upload data first.')

        # Add entries to the action log
        log(selected_option, 'user')
        message = str(dups_count) + ' duplicated rows were detected'
        log(message, 'system')

        # Return all components
        graph_div = show_side_by_side(graph_list)
        if dups_count == 0:
            new_div = html.Div(children=[
                html.P(f'Selected action: {selected_option}'),
                html.P(message, style={'color': 'green'}),
                graph_div,
                dcc.Dropdown(
                    placeholder='Select an action to take', 
                    id={'type': 'duplicate-removal', 'index': step},
                    options={'undo': 'Undo the last step'}
                ),
                html.Br()
            ])
        else:
            new_div = html.Div(children=[
                html.P(f'Selected action: {selected_option}'),
                html.P(message, style={'color': 'red'}),
                graph_div,
                dcc.Dropdown(
                    placeholder='Select an action to take', 
                    id={'type': 'duplicate-removal', 'index': step},
                    options={'highlight': 'Show duplicated rows', 'delete': 'Delete duplicates', 'keep': 'Keep all duplicates'}
                ),
                html.Br()
            ])
        return [new_div]
        
    else:
        return dash.no_update
         

# Callback to handle the first render within the 'outlier-handling' stage
@app.callback(
    [Output(component_id='outlier-output', component_property='children'),
     Output(component_id='duplicate-end-btn', component_property='n_clicks')],
    [Input(component_id='duplicate-end-btn', component_property='n_clicks'),
     Input(component_id={'type': 'duplicate-removal', 'index': ALL}, component_property='value')],
    prevent_initial_call=True,
    running=[(Output(component_id='outlier-end-btn', component_property='disabled'), True, False)]
)
def render_outliers(n_clicks, drop_value):
    global current_df
    global previous_df
    global stage
    global step
    global vis_objects
    global outlier_count
    global outlier_contamination_history

    graph_list = []
    # First render
    if n_clicks is None:
        if drop_value and not None in drop_value and len(drop_value) > 0:
            if 'keep' != drop_value[-1]:
                return dash.no_update
        else:
            return dash.no_update
    log('Finish Duplicate Removal', 'user')
    stage = 'outlier-handling'
    step += 1
    previous_df = current_df.copy()
    # Access the last visualisation rendered on the right for intent specification
    human_previous = vis_objects[-1]
    
    # Display a parallel coordinates plot
    vis_objects, graph_list = render_machine_view(vis_objects, current_df, graph_list)

    # Detect and visualise outliers
    outlier_contamination = determine_contamination(outlier_contamination_history, True)
    outlier_contamination_history.append(outlier_contamination)
    intent = extract_intent(human_previous.columns)
    current_df, outlier_count = train_isolation_forest(current_df, contamination=outlier_contamination, intent=intent)
    outlier_df = current_df.copy()
    outlier_df.intent = intent
    # Display the second visualisation
    vis2 = Vis(len(vis_objects), outlier_df, enhance='outlier')
    # Catch the missing value error if applicable:
    if vis2.missing_value_flag or vis2.output_type == 'img':
        # Display the second visualisation (second recommendation - num_rec=1 - rather than the first as usual)
        temp_vis = Vis(len(vis_objects), current_df, num_rec=1, temporary=True)
        current_df.intent = extract_intent(temp_vis.columns)
        vis2 = Vis(len(vis_objects), current_df, enhance='outlier')
    # Populate vis_objects list for referring back to the visualisations
    vis_objects.append(vis2)
    # Append the graph, wrapped in a Div to track clicks, to graph_list
    graph2 = Graph_component(vis2)
    if graph2.div is not None:
        graph_list.append(graph2.div)

    # Add an entry to the action log
    message = str(outlier_count) + ' outlier values were detected'
    log(message, 'system')
    # Return all components
    graph_div = show_side_by_side(graph_list)
    new_div = html.Div(children=[
        html.P(message, style={'color': 'red'}),
        graph_div,
        dcc.Dropdown(
            placeholder='Select an action to take', 
            id={'type': 'outlier-handling', 'index': step},
            options={
                'more': 'Find more outliers', 
                'less': 'Find less outliers', 
                'accept-0': 'Remove the detected outliers', 
                'keep-0': 'Keep all outliers'
            }
        ),
        html.Br()
    ])
    return [new_div], 1


# Callback to handle updates within the 'outlier-handling' stage
@app.callback(
    [Output(component_id='outlier-output-1', component_property='children')],
    [Input(component_id={'type': 'outlier-handling', 'index': ALL}, component_property='value')],
    [State(component_id='duplicate-end-btn', component_property='n_clicks')],
    prevent_initial_call=True,
    running=[(Output(component_id='outlier-end-btn', component_property='disabled'), True, False),
             (Output(component_id={'type': 'outlier-handling', 'index': ALL}, component_property='disabled'), True, False)]
)
def update_outliers(drop_value, n_clicks):
    global current_df
    global previous_df
    global stage
    global step
    global vis_objects
    global outlier_count
    global outlier_contamination_history

    selected_option = ''
    graph_list = []
    # Specify dropdown options
    options={
        'more': 'Find more outliers', 
        'less': 'Find less outliers', 
        'accept': 'Remove the detected outliers', 
        'keep': 'Keep all outliers',
    }

    if n_clicks is None or None in drop_value or stage != 'outlier-handling':    
        return dash.no_update
    else:
        if n_clicks > 0 and current_df is not None:
            step += 1
            # Access the last visualisation rendered on the right for intent specification
            human_previous = vis_objects[-1]

            if 'next' == drop_value[-1] or 'accept' == drop_value[-1]:
                stage = 'outlier-handling-2'  
                return dash.no_update
            elif 'accept-0' == drop_value[-1]:
                # Just got sent here from render_outliers
                options['keep'] = 'Keep remaining outliers'
                options['undo'] = 'Undo the last step'
                selected_option = 'Remove the detected outliers'
                current_df = current_df[current_df.outlier != True]
                
                # Display a parallel coordinates plot
                vis_objects, graph_list = render_machine_view(vis_objects, current_df, graph_list)

                # Detect and visualise outliers
                outlier_contamination = outlier_contamination_history[-1]
                outlier_contamination_history.append(outlier_contamination)
                intent = extract_intent(human_previous.columns)
                current_df, outlier_count = train_isolation_forest(current_df, contamination=outlier_contamination, intent=intent)
                outlier_df = current_df.copy()
                outlier_df.intent = intent
                # Display the second visualisation
                vis2 = Vis(len(vis_objects), outlier_df, enhance='outlier')
                # Catch the missing value error if applicable:
                if vis2.missing_value_flag or vis2.output_type == 'img':
                    # Display the second visualisation (second recommendation - num_rec=1 - rather than the first as usual)
                    temp_vis = Vis(len(vis_objects), current_df, num_rec=1, temporary=True)
                    current_df.intent = extract_intent(temp_vis.columns)
                    vis2 = Vis(len(vis_objects), current_df, enhance='outlier')
                # Populate vis_objects list for referring back to the visualisations
                vis_objects.append(vis2)
                # Append the graph, wrapped in a Div to track clicks, to graph_list
                graph2 = Graph_component(vis2)
                if graph2.div is not None:
                    graph_list.append(graph2.div)
                else:
                    print('No recommendations available. Please upload data first.')    
            elif 'undo' == drop_value[-1]:
                # Revert the dataframe back to its previous state
                selected_option = 'Undo the last step'
                current_df = previous_df.copy()
                if 'undo' in options:
                    # Remove undo from the dropdown options
                    rv = options.pop('undo')
                outlier_contamination = outlier_contamination_history[-1]
            elif 'more' == drop_value[-1]:
                previous_df = current_df.copy()
                selected_option = 'Find more outliers'
                # Increase contamination parameter to find more outliers
                outlier_contamination = determine_contamination(outlier_contamination_history, True)
                outlier_contamination_history.append(outlier_contamination)
            elif 'less' in drop_value[-1]:
                previous_df = current_df.copy()
                selected_option = 'Find less outliers'
                # Decrease contamination parameter to find more outliers
                outlier_contamination = determine_contamination(outlier_contamination_history, False)
                outlier_contamination_history.append(outlier_contamination)
            else:
                return dash.no_update
                
            # Display a parallel coordinates plot
            vis_objects, graph_list = render_machine_view(vis_objects, current_df, graph_list)
            
            intent = extract_intent(human_previous.columns)
            current_df, outlier_count = train_isolation_forest(current_df, contamination=outlier_contamination, intent=intent)
            outlier_df = current_df.copy()
            outlier_df.intent = intent
            # Display the second visualisation
            vis2 = Vis(len(vis_objects), outlier_df, enhance='outlier')
            # Catch the missing value error if applicable:
            if vis2.missing_value_flag or vis2.output_type == 'img':
                # Display the second visualisation (second recommendation - num_rec=1 - rather than the first as usual)
                temp_vis = Vis(len(vis_objects), current_df, num_rec=1, temporary=True)
                current_df.intent = extract_intent(temp_vis.columns)
                vis2 = Vis(len(vis_objects), current_df, enhance='outlier')
            # Populate vis_objects list for referring back to the visualisations
            vis_objects.append(vis2)
            # Append the graph, wrapped in a Div to track clicks, to graph_list
            graph2 = Graph_component(vis2)
            if graph2.div is not None:
                graph_list.append(graph2.div)
            else:
                print('No recommendations available. Please upload data first.')

            # Add entries to the action log
            if drop_value[-1] != 'accept':
                log(selected_option, 'user')
                message = str(outlier_count) + ' outlier values were detected'
                log(message, 'system')
            # Return all components
            graph_div = show_side_by_side(graph_list)
            new_div = html.Div(children=[
                html.P(f'Selected action: {selected_option}'),
                html.P(message, style={'color': 'red'}),
                graph_div,
                dcc.Dropdown(
                    placeholder='Select an action to take', 
                    id={'type': 'outlier-handling', 'index': step},
                    options=options
                ),
                html.Br()
            ])
            return [new_div]
        else:
            return dash.no_update


# Callback to handle updates within the 'outlier-handling' stage #2
@app.callback(
    [Output(component_id='outlier-output-2', component_property='children')],
    [Input(component_id={'type': 'outlier-handling', 'index': ALL}, component_property='value')],
    [State(component_id='duplicate-end-btn', component_property='n_clicks')],
    prevent_initial_call=True,
    running=[(Output(component_id='outlier-end-btn', component_property='disabled'), True, False),
             (Output(component_id={'type': 'outlier-handling', 'index': ALL}, component_property='disabled'), True, False)]
)
def update_outliers_2(drop_value, n_clicks):
    global current_df
    global previous_df
    global stage
    global step
    global vis_objects
    global outlier_count
    global outlier_contamination_history

    selected_option = ''
    graph_list = []
    # Specify dropdown options
    options={
        'more-2': 'Find more outliers', 
        'less-2': 'Find less outliers', 
        'remove': 'Remove the detected outliers', 
        'keep': 'Keep remaining outliers'
    }

    if n_clicks is None or None in drop_value or len(drop_value) < 2 or stage != 'outlier-handling-2':    
        return dash.no_update
    else:
        if n_clicks > 0 and current_df is not None:
            stage = 'outlier-handling-2'
            step += 1
            # Access the last visualisation rendered on the right for intent specification
            human_previous = vis_objects[-1]

            if 'accept' == drop_value[-1]:
                # Just got sent here from update_outliers
                options['undo-2'] = 'Undo the last step'
                selected_option = 'Remove the detected outliers'
                current_df = current_df[current_df.outlier != True]
                
                # Display a parallel coordinates plot
                vis_objects, graph_list = render_machine_view(vis_objects, current_df, graph_list)

                # Detect and visualise outliers
                outlier_contamination = outlier_contamination_history[-1]
                outlier_contamination_history.append(outlier_contamination)
                intent = extract_intent(human_previous.columns)
                current_df, outlier_count = train_isolation_forest(current_df, contamination=outlier_contamination, intent=intent)
                outlier_df = current_df.copy()
                outlier_df.intent = intent
                # Display the second visualisation
                vis2 = Vis(len(vis_objects), outlier_df, enhance='outlier')
                # Catch the missing value error if applicable:
                if vis2.missing_value_flag or vis2.output_type == 'img':
                    # Display the second visualisation (second recommendation - num_rec=1 - rather than the first as usual)
                    temp_vis = Vis(len(vis_objects), current_df, num_rec=1, temporary=True)
                    current_df.intent = extract_intent(temp_vis.columns)
                    vis2 = Vis(len(vis_objects), current_df, enhance='outlier')
                # Populate vis_objects list for referring back to the visualisations
                vis_objects.append(vis2)
                # Append the graph, wrapped in a Div to track clicks, to graph_list
                graph2 = Graph_component(vis2)
                if graph2.div is not None:
                    graph_list.append(graph2.div)
                else:
                    print('No recommendations available. Please upload data first.')
            else:
                # Access the last visualisation rendered on the right for intent specification
                human_previous = vis_objects[-1]
                if 'next' == drop_value[-1]:
                    previous_df = current_df.copy()
                    # Just got sent here from update_outliers_1
                    selected_option = 'Show remaining outliers'

                    # Display a parallel coordinates plot
                    vis_objects, graph_list = render_machine_view(vis_objects, current_df, graph_list)
                    
                    # Detect and visualise outliers
                    outlier_contamination = outlier_contamination_history[-1]
                    outlier_contamination_history.append(outlier_contamination)
                    current_df, outlier_count = train_isolation_forest(current_df, contamination=outlier_contamination)
                    current_df = lux.LuxDataFrame(current_df)

                    # Display the second visualisation (second recommendation - num_rec=1 - rather than the first as usual)
                    temp_vis = Vis(len(vis_objects), current_df, num_rec=1, temporary=True)
                    current_df.intent = extract_intent(temp_vis.columns)
                    vis2 = Vis(len(vis_objects), current_df, enhance='outlier')
                    # Catch the missing value error if applicable:
                    if vis2.missing_value_flag or vis2.output_type == 'img':
                        # Display the second visualisation (second recommendation - num_rec=1 - rather than the first as usual)
                        temp_vis = Vis(len(vis_objects), current_df, num_rec=1, temporary=True)
                        current_df.intent = extract_intent(temp_vis.columns)
                        vis2 = Vis(len(vis_objects), current_df, enhance='outlier')
                    # Populate vis_objects list for referring back to the visualisations
                    vis_objects.append(vis2)
                    # Append the graph, wrapped in a Div to track clicks, to graph_list
                    graph2 = Graph_component(vis2)
                    if graph2.div is not None:
                        graph_list.append(graph2.div)
                    else:
                        print('No recommendations available. Please upload data first.')

                elif 'undo-2' == drop_value[-1]:
                    # Revert the dataframe back to its previous state
                    selected_option = 'Undo the last step'
                    current_df = previous_df.copy()
                    if 'undo-2' in options:
                        # Remove undo from the dropdown options
                        rv = options.pop('undo-2')
                    outlier_contamination = outlier_contamination_history[-1]
                elif 'more-2' == drop_value[-1]:
                    previous_df = current_df.copy()
                    selected_option = 'Find more outliers'
                    # Increase contamination parameter to find more outliers
                    outlier_contamination = determine_contamination(outlier_contamination_history, True)
                    outlier_contamination_history.append(outlier_contamination)
                elif 'less-2' in drop_value[-1]:
                    previous_df = current_df.copy()
                    selected_option = 'Find less outliers'
                    # Decrease contamination parameter to find more outliers
                    outlier_contamination = determine_contamination(outlier_contamination_history, False)
                    outlier_contamination_history.append(outlier_contamination)
                elif 'remove' == drop_value[-1]:
                    # Remove the selected outliers (handled by the next callback function and in a new UI section)
                    stage = 'outlier-handling-3'
                    return dash.no_update
                else:
                    return dash.no_update
                    
                # Display a parallel coordinates plot
                vis_objects, graph_list = render_machine_view(vis_objects, current_df, graph_list)
                
                intent = extract_intent(human_previous.columns)
                current_df, outlier_count = train_isolation_forest(current_df, contamination=outlier_contamination, intent=intent)
                outlier_df = current_df.copy()
                outlier_df.intent = intent
                # Display the second visualisation, catching any AttributeError that occurs
                try:
                    vis2 = Vis(len(vis_objects), outlier_df, enhance='outlier')
                    # Catch the missing value error if applicable:
                    if vis2.missing_value_flag or vis2.output_type == 'img':
                        # Display the second visualisation (second recommendation - num_rec=1 - rather than the first as usual)
                        temp_vis = Vis(len(vis_objects), current_df, num_rec=1, temporary=True)
                        current_df.intent = extract_intent(temp_vis.columns)
                        vis2 = Vis(len(vis_objects), current_df, enhance='outlier')
                except AttributeError as e:
                    print(e)
                    # Display the second visualisation (second recommendation - num_rec=1 - rather than the first as usual)
                    temp_vis = Vis(len(vis_objects), current_df, num_rec=1, temporary=True)
                    current_df.intent = extract_intent(temp_vis.columns)
                    vis2 = Vis(len(vis_objects), current_df, enhance='outlier')
                # Populate vis_objects list for referring back to the visualisations
                vis_objects.append(vis2)
                # Append the graph, wrapped in a Div to track clicks, to graph_list
                graph2 = Graph_component(vis2)
                if graph2.div is not None:
                    graph_list.append(graph2.div)
                else:
                    print('No recommendations available. Please upload data first.')

            # Add entries to the action log
            if drop_value[-1] != 'remove':
                log(selected_option, 'user')
                message = str(outlier_count) + ' new potential outlier values were detected. If no more outliers should be removed, please click on "Finish Outlier Handling" below.'
                log(message, 'system')
            # Return all components
            graph_div = show_side_by_side(graph_list)
            new_div = html.Div(children=[
                html.P(f'Selected action: {selected_option}'),
                html.P(message, style={'color': 'green'}),
                graph_div,
                dcc.Dropdown(
                    placeholder='Select an action to take', 
                    id={'type': 'outlier-handling', 'index': step},
                    options=options
                )
            ])
            return [new_div]
        else:
            return dash.no_update


# Callback to handle updates within the 'outlier-handling' stage #3 (final UI section for this stage)
@app.callback(
    [Output(component_id='outlier-output-3', component_property='children')],
    [Input(component_id={'type': 'outlier-handling', 'index': ALL}, component_property='value')],
    [State(component_id='duplicate-end-btn', component_property='n_clicks')],
    prevent_initial_call=True,
    running=[(Output(component_id='outlier-end-btn', component_property='disabled'), True, False),
             (Output(component_id={'type': 'outlier-handling', 'index': ALL}, component_property='disabled'), True, False)]
)
def update_outliers_3(drop_value, n_clicks):
    global current_df
    global previous_df
    global stage
    global step
    global vis_objects
    global outlier_count
    global outlier_contamination_history

    selected_option = ''
    graph_list = []
    # Specify the dropdown options
    options={
        'more-3': 'Find more outliers', 
        'less-3': 'Find less outliers', 
        'remove-3': 'Remove the detected outliers', 
        'keep': 'Keep remaining outliers'
    }

    if n_clicks is None or None in drop_value or len(drop_value) < 2  or stage != 'outlier-handling-3':    
        return dash.no_update
    else:
        if n_clicks > 0 and current_df is not None:
            stage = 'outlier-handling-3'
            step += 1
            # Access the last visualisation rendered on the right for intent specification
            human_previous = vis_objects[-1]

            if 'remove' == drop_value[-1] or 'remove-3' == drop_value[-1]:
                # Just got sent here from update_outliers_2, or it is the final removal
                options['undo-3'] = 'Undo the last step'
                selected_option = 'Remove the detected outliers'
                current_df = current_df[current_df.outlier != True]
                
                # Display a parallel coordinates plot
                vis_objects, graph_list = render_machine_view(vis_objects, current_df, graph_list)

                # Detect and visualise outliers
                outlier_contamination = outlier_contamination_history[-1]
                outlier_contamination_history.append(outlier_contamination)
                intent = extract_intent(human_previous.columns)
                current_df, outlier_count = train_isolation_forest(current_df, contamination=outlier_contamination, intent=intent)
                outlier_df = current_df.copy()
                outlier_df.intent = intent
                # Display the second visualisation
                vis2 = Vis(len(vis_objects), outlier_df, enhance='outlier')
                # Catch the missing value error if applicable:
                if vis2.missing_value_flag or vis2.output_type == 'img':
                    # Display the second visualisation (second recommendation - num_rec=1 - rather than the first as usual)
                    temp_vis = Vis(len(vis_objects), current_df, num_rec=1, temporary=True)
                    current_df.intent = extract_intent(temp_vis.columns)
                    vis2 = Vis(len(vis_objects), current_df, enhance='outlier')
                # Populate vis_objects list for referring back to the visualisations
                vis_objects.append(vis2)
                # Append the graph, wrapped in a Div to track clicks, to graph_list
                graph2 = Graph_component(vis2)
                if graph2.div is not None:
                    graph_list.append(graph2.div)
                else:
                    print('No recommendations available. Please upload data first.')
            else:
                # Access the last visualisation rendered on the right for intent specification
                human_previous = vis_objects[-1]
                if 'undo-3' == drop_value[-1]:
                    # Revert the dataframe back to its previous state
                    selected_option = 'Undo the last step'
                    current_df = previous_df.copy()
                    if 'undo-3' in options:
                        # Remove undo from the dropdown options
                        rv = options.pop('undo-3')
                    outlier_contamination = outlier_contamination_history[-1]
                elif 'more-3' == drop_value[-1]:
                    previous_df = current_df.copy()
                    selected_option = 'Find more outliers'
                    # Increase contamination parameter to find more outliers
                    outlier_contamination = determine_contamination(outlier_contamination_history, True)
                    outlier_contamination_history.append(outlier_contamination)
                elif 'less-3' in drop_value[-1]:
                    previous_df = current_df.copy()
                    selected_option = 'Find less outliers'
                    # Decrease contamination parameter to find more outliers
                    outlier_contamination = determine_contamination(outlier_contamination_history, False)
                    outlier_contamination_history.append(outlier_contamination)
                else:
                    return dash.no_update
                
                # Display a parallel coordinates plot
                vis_objects, graph_list = render_machine_view(vis_objects, current_df, graph_list)
                
                intent = extract_intent(human_previous.columns)
                current_df, outlier_count = train_isolation_forest(current_df, contamination=outlier_contamination, intent=intent)
                outlier_df = current_df.copy()
                outlier_df.intent = intent
                # Display the second visualisation, catching any AttributeError that occurs
                try:
                    vis2 = Vis(len(vis_objects), outlier_df, enhance='outlier')
                    # Catch the missing value error if applicable:
                    if vis2.missing_value_flag or vis2.output_type == 'img':
                        # Display the second visualisation (second recommendation - num_rec=1 - rather than the first as usual)
                        temp_vis = Vis(len(vis_objects), current_df, num_rec=1, temporary=True)
                        current_df.intent = extract_intent(temp_vis.columns)
                        vis2 = Vis(len(vis_objects), current_df, enhance='outlier')
                except AttributeError as e:
                    print(e)
                    # Display the second visualisation (second recommendation - num_rec=1 - rather than the first as usual)
                    temp_vis = Vis(len(vis_objects), current_df, num_rec=1, temporary=True)
                    current_df.intent = extract_intent(temp_vis.columns)
                    vis2 = Vis(len(vis_objects), current_df, enhance='outlier')
                # Populate vis_objects list for referring back to the visualisations
                vis_objects.append(vis2)
                # Append the graph, wrapped in a Div to track clicks, to graph_list
                graph2 = Graph_component(vis2)
                if graph2.div is not None:
                    graph_list.append(graph2.div)
                else:
                    print('No recommendations available. Please upload data first.')

            # Add entries to the action log
            log(selected_option, 'user')
            message = str(outlier_count) + ' new potential outlier values were detected. If no more outliers should be removed, please click on "Finish Outlier Handling" below.'
            log(message, 'system')
            # Return all components
            graph_div = show_side_by_side(graph_list)
            new_div = html.Div(children=[
                html.P(f'Selected action: {selected_option}'),
                html.P(message, style={'color': 'green'}),
                graph_div,
                dcc.Dropdown(
                    placeholder='Select an action to take', 
                    id={'type': 'outlier-handling', 'index': step},
                    options=options
                ),
                html.Br()
            ])
            return [new_div]
        else:
            return dash.no_update


#################################################################
### Callback functions that determine various styling aspects ###

# Callback to handle disabling the stage-end buttons
@app.callback(
    [Output(component_id='start-button', component_property='disabled'),
     Output(component_id='missing-end-btn', component_property='disabled'),
     Output(component_id='duplicate-end-btn', component_property='disabled'),
     Output(component_id='outlier-end-btn', component_property='disabled')],
    [Input(component_id='start-button', component_property='n_clicks'),
     Input(component_id='missing-end-btn', component_property='n_clicks'),
     Input(component_id='duplicate-end-btn', component_property='n_clicks'),
     Input(component_id='outlier-end-btn', component_property='n_clicks'),
     Input(component_id={'type': 'duplicate-removal', 'index': ALL}, component_property='value'),
     Input(component_id={'type': 'outlier-handling', 'index': ALL}, component_property='value')],
    prevent_initial_call=True
)
def indicate_process_end(start_n_clicks, miss_n_clicks, dup_n_clicks, out_n_clicks, drop_dup, drop_out):
    # If the outlier-end button is clicked, the end of the process is reached and the buttons should all be disabled
    if out_n_clicks:
        return True, True, True, True
    elif not None in drop_out and len(drop_out) > 0:
        if 'keep-0' == drop_out[-1] or 'keep' == drop_out[-1]:
            return True, True, True, True
        else:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    # If the duplicate-end button is clicked, the first 3 stage-end buttons should be disabled
    elif dup_n_clicks:
        return True, True, True, False
    elif not None in drop_dup and len(drop_dup) > 0:
        if 'keep' == drop_dup[-1]:
            return True, True, True, False
        else:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    # If the missing-end button is clicked, the first 2 stage-end buttons should be disabled
    elif miss_n_clicks:
        return True, True, False, False
    # Else only the first stage-end button should be disabled
    elif start_n_clicks:
        return True, False, False, False
    else:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update


# Callback to update progress
@app.callback(
    [Output(component_id='progress-load', component_property='style'),
     Output(component_id='progress-missing', component_property='style'),
     Output(component_id='progress-duplicate', component_property='style'),
     Output(component_id='progress-outlier', component_property='style'),
     Output(component_id='progress-download', component_property='style'),
     Output(component_id='missing-end-btn', component_property='style'),
     Output(component_id='duplicate-end-btn', component_property='style'),
     Output(component_id='outlier-end-btn', component_property='style'),
     Output(component_id='out-end-info', component_property='style'),
     Output(component_id='csv-btn', component_property='style'),
     Output(component_id='download-header', component_property='style'),
     Output(component_id='download-info', component_property='style'),
     Output(component_id='download-btn', component_property='style'),
     Output(component_id='completion-message', component_property='style')],
    [Input(component_id='upload-data', component_property='contents'),
     Input(component_id='dataset-selection', component_property='value'),
     Input(component_id='start-button', component_property='n_clicks'),
     Input(component_id='missing-end-btn', component_property='n_clicks'),
     Input(component_id='duplicate-end-btn', component_property='n_clicks'),
     Input(component_id='outlier-end-btn', component_property='n_clicks'),
     Input(component_id='csv-btn', component_property='n_clicks'),
     Input(component_id='download-btn', component_property='n_clicks'),
     Input(component_id={'type': 'duplicate-removal', 'index': ALL}, component_property='value'),
     Input(component_id={'type': 'outlier-handling', 'index': ALL}, component_property='value')],
    prevent_initial_call=True
)
def update_progress(contents, selected_dataset, click_start, click_miss, click_dup, click_out, click_down, click_down_dash, drop_dup, drop_out):
    global download_completion
    global load_colour
    global miss_colour
    global dup_colour
    global out_colour
    global down_colour
    global missing_style
    global dup_style
    global out_style
    global info_style
    global download_style
    global down_info_style
    global completion_style

    ctx = dash.callback_context

    # If buttons are clicked, change the respective progress bars
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    
    load_colour, miss_colour, dup_colour, out_colour, down_colour, missing_style, dup_style, out_style, info_style, download_style, down_info_style, completion_style, download_completion, log_msg = style_progress(ctx, changed_id, click_out, download_completion, drop_dup, drop_out, load_colour, miss_colour, dup_colour, out_colour, down_colour, missing_style, dup_style, out_style, info_style, download_style, down_info_style, completion_style)

    if log_msg[0] != '':
        log(log_msg[0], log_msg[1])
    
    return (
        {'background-color': load_colour, 'color': 'white'},  # progress-load
        {'background-color': miss_colour, 'color': 'white'},  # progress-missing
        {'background-color': dup_colour, 'color': 'white'},   # progress-duplicate
        {'background-color': out_colour, 'color': 'white'},   # progress-outlier
        {'background-color': down_colour, 'color': 'white'},  # progress-download
        missing_style,    # missing-end-btn
        dup_style,        # duplicate-end-btn
        out_style,        # outlier-end-btn
        info_style,       # out-end-info
        download_style,   # csv-btn
        download_style,   # download-header
        down_info_style,  # download-info
        download_style,   # download-btn
        completion_style  # completion-message
    )


#################################################################################
### Callback functions for the downloading of the cleaned data and action log ###

# Callback to download cleaned dataset into a csv file
@app.callback(
    Output(component_id='download-dataframe-csv', component_property='data'),
    Input(component_id='csv-btn', component_property='n_clicks'),
    prevent_initial_call=True,
)
def func(n_clicks):
    global file_name
    return dcc.send_data_frame(downloadable_data(current_df).to_csv, file_name)


# Callback to download log into a txt file
@app.callback(
    Output(component_id='download-log', component_property='data'),
    Input(component_id='download-btn', component_property='n_clicks'),
    prevent_initial_call=True,
)
def save_list_to_file(n_clicks):
    global file_name
    global action_log
    # Create a suitable filename
    filename = file_name[:-4]
    filename = filename + '_log.txt'
    # Convert the list into a file-like object
    file_content = "\n".join(action_log)  # Each item on a new line
    file_obj = io.StringIO(file_content)
    return dict(content=file_obj.getvalue(), filename=filename)


#########################################
### Functionality for running the app ###

# Expose the Flask server
server = app.server

# Run the Dash app
if __name__ == '__main__':
    app.run(debug=True)
