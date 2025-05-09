#######################################################################
### This class generates and converts visualisations, including the ###
### parallel coordinates plots and Lux recommended plots            ###
#######################################################################

from plotly.tools import mpl_to_plotly
from matplotlib.cm import Set1
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas import NaT


from helper_functions import *

class Vis:

    def __init__(self, id, df, rec_group=0, num_rec=0, machine_view=False, enhance=None, temporary=False):
        self.id = id
        self.columns = None
        self.output_type = None
        self.lux_vis = None
        self.figure = None
        self.machine_view = machine_view
        self.enhance = enhance
        self.missing_value_flag = False

        if machine_view:
            # Display a parallel coordinates plot
            fig = px.parallel_coordinates(df)
            # Specify layout size
            fig.update_layout(
                autosize=True,
                height=400,  
                width=600  
            )
            # Set attributes
            self.figure = fig
            self.output_type = 'plotly'
            self.columns = list(df.columns)

        else:
            # Generate recommendations and store the resulting dictionary explicitly
            if 'id' in df.columns:
                df = df.drop('id', axis=1)
            df = self.infer_column_types(df)
            try:
                recommendations = df.recommendation
                if recommendations:
                    # Store the recommendation options (e.g., Occurrence, Correlation, Temporal)
                    rec_options = [key for key in recommendations]
                    if 'Enhance' in rec_options:
                        self.rec_type = 'Enhance'
                    elif 'Correlation' in rec_options:
                        self.rec_type = 'Correlation'
                    else:
                        if len(rec_options) > rec_group:
                            # Access chosen recommendation group
                            self.rec_type = rec_options[rec_group]
                        else:
                            # Access first recommendation group
                            self.rec_type = rec_options[0]
                    self.selected_recommendations = recommendations[self.rec_type]

                    # Plot figure
                    if self.selected_recommendations:
                        # If a specific attribute is chosen for plot enhancement, ensure it is used
                        if self.rec_type == 'Enhance' and self.enhance is not None:
                            # Search self.selected_recommendations for the color attribute stored in self.enhance
                            for rec in self.selected_recommendations:
                                # Get the color column
                                color_column = rec.get_attr_by_channel('color')
                                # Get the column name
                                if len(color_column) > 0:
                                    color_attribute = color_column[0].attribute
                                    if color_attribute == self.enhance:
                                        self.lux_vis = rec
                                        break                        
                        elif len(self.selected_recommendations) > num_rec:
                            self.lux_vis = self.selected_recommendations[num_rec]
                        else:
                            self.lux_vis = self.selected_recommendations[0]
                        # Get the relevant column names
                        self.columns = extract_vis_columns(self.lux_vis)

                        if not temporary:                    
                            # Initialise variables that will be specified in the fig_code 
                            fig, ax = plt.subplots()
                            tab20c = plt.get_cmap('tab20c')
                            # Render the visualisation using Lux
                            try:
                                # The below print is very useful for debugging
                                # print("**********self.lux_vis: ", self.lux_vis, "****************")
                                fig_code = self.lux_vis.to_matplotlib()
                            # Catch errors if applicable
                            except (ValueError, AttributeError) as e:
                                print('Error in to_matplotlib()')
                                fig_code = ''
                                self.missing_value_flag = True
                            fixed_fig_code = fix_lux_code(fig_code)
                            # Use easily visible colours
                            fixed_fig_code = update_colours(fixed_fig_code)
                            try:
                                exec(fixed_fig_code)
                            except ValueError as e:
                                print(e)
                                self.missing_value_flag = True

                            # Capture the current Matplotlib figure
                            fig = plt.gcf()
                            if fig is None:
                                pass
                            plt.draw()

                            # Adjust layout to prevent legend cutoff
                            plt.tight_layout()
                            # Manually adjust legend if needed
                            fig.subplots_adjust(right=0.8)

                            # Try to convert Matplotlib figure to Plotly
                            try:
                                fig = mpl_to_plotly(fig)
                                # Specify layout size
                                fig.update_layout(
                                    autosize=True,
                                    height=400,  
                                    width=600  
                                )
                                self.figure = fig
                                self.output_type = 'plotly'
                            except Exception as e:
                                # If an error occurs, create a static Matplotlib image instead
                                # In the current implementation, this is not displayed but instead a warning message is shown
                                fallback_fig = create_styled_matplotlib_figure(fig)
                                # Convert Matplotlib figure to base64 image
                                self.figure = fig_to_base64(fallback_fig)
                                self.output_type = 'img'   
            except IndexError as e:
                print('IndexError: ', e)
                self.missing_value_flag = True
            

    def infer_column_types(self, df):
        # Check if column is datetime column
        for col in df.columns:
            try:
                parsed_col = pd.to_datetime(col, errors='coerce')
                timestamp_ratio = parsed_col.notna().mean()  # Proportion of successfully converted values
                if timestamp_ratio > 0.9:  # If most values convert successfully, treat it as a timestamp
                    # Convert timestamp column to integer
                    df[col] = parsed_col
            except Exception:
                continue
        return df
