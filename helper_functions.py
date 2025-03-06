import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import re
import base64
import io
from io import BytesIO
import IPython.display as display
from PIL import Image
import plotly.express as px
from dash import dcc, html

# Add locally cloned Lux source code to path, and import Lux from there
sys.path.insert(0, os.path.abspath('./lux'))
import lux
from lux.vis.Vis import Vis

# Function to parse uploaded data
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    if filename.endswith('.csv'):
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        # Convert all column names to lowercase and replace spaces with underscores
        df.rename(columns=lambda x: x.lower().replace(' ', '_').replace('-', '_'), inplace=True)
        # Remove all special characters from column names
        df.rename(columns=lambda x: x.lower().replace(':', '').replace('$', '').replace('(', '').replace(')', ''), inplace=True)
        # Detect and convert any datetime columns
        df = parse_datetime_cols(df)

        return df
    return None

# Function to prepare preloaded data
def prepare_contents(filename):
    access_string = 'assets/' + filename
    df = pd.read_csv(access_string)
    # Convert all column names to lowercase and replace spaces with underscores
    df.rename(columns=lambda x: x.lower().replace(' ', '_').replace('-', '_'), inplace=True)
    # Remove all special characters from column names
    df.rename(columns=lambda x: x.lower().replace(':', '').replace('$', '').replace('(', '').replace(')', ''), inplace=True)
    # Detect and convert any datetime columns
    df = parse_datetime_cols(df)
    return df

def parse_datetime_cols(df):
    # Make a copy of the data to retain original
    data_original = df.copy()
    # Convert LuxDataFrame to Pandas DataFrame if necessary
    if not isinstance(data_original, pd.DataFrame):
        data_original = pd.DataFrame(data_original)
    # Find the object columns
    object_cols = data_original.select_dtypes(include=['object']).columns

    # Check if column is datetime column
    for col in object_cols:
        try:
            parsed_col = pd.to_datetime(df[col].str.strip(), errors='coerce')
            timestamp_ratio = parsed_col.notna().mean()  # Proportion of successfully converted values
            if timestamp_ratio > 0.9:  # If most values convert successfully, treat it as a timestamp
                data_original[col] = parsed_col
        except Exception as e:
            print('EXCEPTION: ', e)
            # Treat it as a non-datetime column
            data_original[col] = data_original[col]
    return data_original

def create_styled_matplotlib_figure(fig):
    # Apply Plotly-like styling to an existing Matplotlib figure
    # Set the figure background to white (to match Plotly)
    fig.patch.set_facecolor('white')

    # Get the main axis
    ax = fig.axes[0] if fig.axes else fig.add_subplot(111)  # Ensure there is an axis
    ax.set_facecolor('#E5ECF6')  # Light blue background only for the plotting area

    # Update bar colours if applicable
    for patch in ax.patches:
        patch.set_facecolor('#4C59C2')

        # Make bars slimmer
        if isinstance(patch, plt.Rectangle):  # Ensure it's a bar
            if patch.get_width() > patch.get_height():  # Horizontal bars
                patch.set_height(patch.get_height() * 0.3)  # Reduce thickness
            else:  # Vertical bars
                patch.set_width(patch.get_width() * 0.3)

    # Style the labels
    ax.set_xlabel(ax.get_xlabel(), fontsize=10, labelpad=12, 
                  bbox=dict(facecolor='white', edgecolor='none'))
    ax.set_ylabel(ax.get_ylabel(), fontsize=10, labelpad=12, 
                  bbox=dict(facecolor='white', edgecolor='none'))

    # Style the tick labels
    ax.tick_params(axis='both', labelsize=7)

    # Remove unnecessary spines (top & right)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#AAB8C2')  # Light gray for a cleaner look
    ax.spines['left'].set_color('#AAB8C2')
    return fig

def fig_to_base64(fig):
    # Convert a Matplotlib figure to a base64-encoded PNG
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded_img = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return f'data:image/png;base64,{encoded_img}'

def fix_lux_code(lux_code):
    # Pattern to identify the ax.barh() function call
    pattern = r'(ax\.barh\()(.*?dtype:.*?dtype:.*?)\)'
    # Replacement that ensures bars and measurements are properly formatted
    replacement = r'ax.barh(bars.values, measurements.values, align="center")'
    # Apply the substitution to the code to fix the barh() call
    fixed_code = re.sub(pattern, replacement, lux_code, flags=re.DOTALL)
    return fixed_code

def extract_vis_columns(visualisation):
    extracted_columns = []
    # Convert Vis object to string and extract x and y column names
    vis_str = str(visualisation)
    match = re.search(r'x: ([^,]+), y: ([^)]+)', vis_str)
    if match:
        x_col, y_col = match.groups()
        extracted_columns = [x_col.strip(), y_col.strip()]
    return extracted_columns

def parse_vis_string(vis_str):
    # Extract x and y axis
    x_match = re.search(r'x:\s*([\w_]+)', vis_str)
    y_match = re.search(r'y:\s*([\w_]+)', vis_str)
    mark_match = re.search(r'mark:\s*([\w_]+)', vis_str)
    score_match = re.search(r'score:\s*([\d.]+)', vis_str)

    if not x_match or not y_match:
        raise ValueError('Invalid Vis string format')

    x_attr = x_match.group(1)
    y_attr = y_match.group(1)
    mark = mark_match.group(1) if mark_match else None
    score = float(score_match.group(1)) if score_match else None

    # Create Vis object (without 'mark' argument)
    vis = Vis([{'x': x_attr}, {'y': y_attr}])

    # Set mark if available
    if mark:
        vis.mark = mark

    # Assign score if available
    if score is not None:
        vis.score = score
    return vis

def show_side_by_side(components):
    return html.Div(
        children=components,
        style={
            'display': 'flex',
            'justifyContent': 'space-between',
            'width': '90%'
        }
    )

def extract_intent(lst):
    if not lst or len(lst) < 2:
        # Return an empty list if input is invalid
        return []  
    
    first_item = lst[0]
    # Split at the first comma and take the first part
    second_item = lst[1].split(',')[0]  
    return [first_item, second_item]

def determine_contamination(cont_history, more):
    contamination = 0.15
    if len(cont_history) > 0:
        if more is True:
            contamination = cont_history[-1] * 1.4
            if contamination > 0.5:
                contamination = 0.5
        else:
            contamination = cont_history[-1] * 0.6
            if contamination <= 0.0:
                contamination = 0.01
    return round(contamination, 4)

def downloadable_data(df):
    if 'duplicate' in df.columns:
        df = df.drop(columns=['duplicate'])
    if 'outlier' in df.columns:
        df = df.drop(columns=['outlier'])
    return df

def determine_filename(og_filename):
    og_filename = og_filename[:-4]
    if 'corrupted' in og_filename:
        og_filename = og_filename.replace('corrupted', 'clean')
    else:
        og_filename = og_filename + '_clean'
    og_filename = og_filename + '.csv'
    return og_filename

def update_colours(code_str):
    # Regular expression to find the colourmap specification block
    pattern = re.compile(r"(cmap=Set1)", re.DOTALL)
    # Replacement colourmap specification
    new_colors = "cmap='RdYlGn_r'"
    # Replace the found colourmap block with the new one
    updated_code = re.sub(pattern, new_colors, code_str)
    return updated_code

def style_progress(ctx, changed_id, click_out, download_completion, drop_dup, drop_out, load_colour, miss_colour, dup_colour, out_colour, down_colour, missing_style, dup_style, out_style, info_style, download_style, down_info_style, completion_style):
    log_msg = ['', '']
    # If a new file is uploaded, reset colours to 'red' and display styles to 'none'
    if (ctx.triggered and 'upload-data' in ctx.triggered[0]['prop_id']) or ('dataset-selection' in changed_id):
        load_colour = 'green'
        miss_colour = 'red'
        dup_colour = 'red'
        out_colour = 'red'
        down_colour = 'red'
        missing_style = {'display': 'none'}
        dup_style = {'display': 'none'}
        out_style = {'display': 'none'}
        info_style = {'display': 'none'}
        down_info_style = {'display': 'none'}
        download_style = {'display': 'none'}
    
    if 'start-button' in changed_id:
        load_colour = 'green'
        missing_style = {'display': 'block'}
        # The below is only for evaluation purposes
        download_style = {'display': 'block'}
        down_info_style = {'display': 'block'}
        # dirtiness = determine_dirtiness()
    elif 'missing-end-btn' in changed_id:
        miss_colour = 'green'
        missing_style = {'display': 'block'}
        dup_style = {'display': 'block'}
    elif 'duplicate-end-btn' in changed_id:
        dup_colour = 'green'
        out_style = {'display': 'block'}
    elif 'outlier-end-btn' in changed_id or click_out == 10:
        out_colour = 'green'
        download_style = {'display': 'block'}
        log_msg[0] = 'Finish Outlier Handling'
        log_msg[1] = 'user'
        # The below is only for evaluation purposes
        down_info_style = {'display': 'none'}
        info_style = {'display': 'block'}
    elif 'csv-btn' in changed_id:
        download_completion[0] = 1
    elif 'download-btn' in changed_id:
        download_completion[1] = 1
    
    if download_completion[0] == 1 and download_completion[1] == 1:
        down_colour = 'green'
        completion_style = {'display': 'block'}
    
    if None not in drop_dup and len(drop_dup) >= 1:
        if 'keep' == drop_dup[-1]:
            log_msg[0] = 'Keep all duplicates'
            log_msg[1] = 'user'
            dup_colour = 'green'
            out_style = {'display': 'block'}
    if None not in drop_out and len(drop_out) >= 1:
        if 'keep-0' == drop_out[-1]:
            log_msg[0] = 'Keep all outliers'
            log_msg[1] = 'user'
            out_colour = 'green'
            download_style = {'display': 'block'}
            # The below is only for evaluation purposes
            down_info_style = {'display': 'none'}
            info_style = {'display': 'block'}
            log_msg[0] = 'Finish Outlier Handling'
            log_msg[1] = 'user'
        elif 'keep' == drop_out[-1]:
            log_msg[0] = 'Keep remaining outliers'
            log_msg[1] = 'user'
            out_colour = 'green'
            download_style = {'display': 'block'}
            # The below is only for evaluation purposes
            down_info_style = {'display': 'none'}
            info_style = {'display': 'block'}
            log_msg[0] = 'Finish Outlier Handling'
            log_msg[1] = 'user'
    
    return load_colour, miss_colour, dup_colour, out_colour, down_colour, missing_style, dup_style, out_style, info_style, download_style, down_info_style, completion_style, log_msg
