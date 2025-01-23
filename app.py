import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import base64
import io
# from luxwidget import render_widget

# Add locally cloned Lux source code to path, and import Lux from there
sys.path.insert(0, os.path.abspath("./lux"))
import lux

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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

    # Placeholder for uploaded data and visualizations
    html.Div(id='output-data-upload', className="my-4"),

    # Button to trigger Lux recommendations
    html.Div([
        dbc.Button("Show Recommendations", id="show-recs", color="primary", className="mt-2"),
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
            # Set Lux as default display
            uploaded_df.default_display = "lux"
            return html.Div([
                html.H5(f"Uploaded File: {filename}"),
                dbc.Table.from_dataframe(uploaded_df.head(), striped=True, bordered=True, hover=True)
            ])
        return html.Div("Unsupported file type.")
    return html.Div("No file uploaded.")

# Callback to display Lux recommendations
@app.callback(
    Output(component_id='lux-output', component_property='children'),
    Input(component_id='show-recs', component_property='n_clicks')
)
# def show_recommendations(n_clicks):
#     global uploaded_df
#     if n_clicks and uploaded_df is not None:
#         uploaded_df.save_as_html("lux_vis.html")
#         with open("lux_vis.html", "r") as f:
#             lux_html = f.read()
#         return html.Iframe(srcDoc=lux_html, style={"width": "100%", "height": "500px", "border": "none"})
#         # # Generate Lux widget output
#         # widget_html = render_widget(uploaded_df)
#         # return html.Iframe(srcDoc=widget_html, style={"width": "100%", "height": "500px", "border": "none"})
#         # # Automatically generate recommendations in Lux
#         # lux_html = uploaded_df.to_html()  # Convert Lux recommendations to HTML
#         # return html.Iframe(srcDoc=lux_html, style={"width": "100%", "height": "500px", "border": "none"})
#     # return html.Div("No recommendations available. Upload data first.")

# def show_recommendations(n_clicks):
#     global uploaded_df
#     if n_clicks and uploaded_df is not None:
#         # Save Lux visualizations to a static HTML file
#         uploaded_df.save_as_html("lux_vis.html")
#         with open("lux_vis.html", "r") as f:
#             lux_html = f.read()
#         # Embed the saved HTML file in an iframe
#         return html.Iframe(srcDoc=lux_html, style={"width": "100%", "height": "500px", "border": "none"})
#     return html.Div("No recommendations available. Upload data first.")

def show_recommendations(n_clicks):
    global uploaded_df
    if n_clicks and uploaded_df is not None:
        # Export Lux visualizations to Matplotlib figures
        exported_vis = uploaded_df.exported
        fig, ax = plt.subplots(figsize=(10, 5))
        if exported_vis and len(exported_vis) > 0:
            exported_vis[0].plot(ax=ax)  # Plot the first visualization as an example
            plt.savefig("lux_vis.png")
            encoded_image = base64.b64encode(open("lux_vis.png", "rb").read()).decode("ascii")
            return html.Img(src=f"data:image/png;base64,{encoded_image}")
    return html.Div("No recommendations available. Upload data first.")

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
