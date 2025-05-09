######################################################
### This class is responsible for the rendering of ###
### interactive graphs on the GUI                  ###
######################################################

from dash import html, dcc

class Graph_component:

    def __init__(self, vis):
        self.div = None

        if vis.output_type == 'plotly':
            # Create a Dash Graph component and wrap it in Div to track clicks
            self.div = html.Div(
                children=[
                    dcc.Graph(
                        id={'type': 'dynamic-graph', 'index': vis.id},
                        figure=vis.figure,
                        style={'width': '45%', 'boxSizing': 'border-box'} # 'padding': '5px', 
                    )
                ],
                id={'type': 'graph-container', 'index': vis.id, 'columns': str(vis.columns)},  # Store columns in ID
                style={'cursor': 'pointer'},  # Indicate clickability
                n_clicks=0  # Track clicks
            )
            
        # If fallback figure is provided instead of graph, display a warning message
        elif vis.output_type == 'img':
            self.div = html.Div(
                children=[
                    html.P(
                        'There is currently no recommended visualisation available. Please choose an option.'
                    )
                ],
                id={'type': 'graph-container', 'index': vis.id, 'columns': str(vis.columns)},  # Store columns ID
            )
        