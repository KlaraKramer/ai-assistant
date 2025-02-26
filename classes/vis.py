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
                autosize=True,  # Allow auto-sizing within Dash
                height=400,  
                width=600  
            )
            self.figure = fig
            self.output_type = 'plotly'
            self.columns = list(df.columns)

        else:
            # Generate recommendations and store the resulting dictionary explicitly
            if 'id' in df.columns:
                df = df.drop('id', axis=1)
            df = self.infer_column_types(df)
            recommendations = df.recommendation
            if recommendations:
                # Store the recommendation options (e.g., Occurrence, Correlation, Temporal)
                rec_options = [key for key in recommendations]
                # print('****************************rec_options: ', rec_options, '**************************')
                if 'Correlation' in rec_options:
                    self.rec_type = 'Correlation'
                elif 'Enhance' in rec_options:
                    self.rec_type = 'Enhance'
                else:
                    if len(rec_options) > rec_group:
                        # Access chosen recommendation group
                        self.rec_type = rec_options[rec_group]
                    else:
                        # Access first recommendation group
                        self.rec_type = rec_options[0]
                self.selected_recommendations = recommendations[self.rec_type]
                # print('****************************selected_recs: ', self.selected_recommendations, '*******************')

                # Plot figure
                if self.selected_recommendations:
                    if self.rec_type == 'Enhance' and self.enhance is not None:
                        # Search self.selected_recommendations for the color attribute stored in self.enhance
                        for rec in self.selected_recommendations:
                            # Get the color column
                            color_column = rec.get_attr_by_channel('color')
                            # Get the column name
                            if len(color_column) > 0:
                                color_attribute = color_column[0].attribute
                                # print('**************color attribute: ', color_attribute, '****************')
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
                            print("**********self.lux_vis: ", self.lux_vis, "****************")
                            fig_code = self.lux_vis.to_matplotlib()
                        except ValueError:
                            print('Error in to_matplotlib()')
                            fig_code = ''
                        fixed_fig_code = fix_lux_code(fig_code)
                        try:
                            exec(fixed_fig_code)
                        except ValueError as e:
                            print(e)
                            self.missing_value_flag = True

                        # Capture the current Matplotlib figure
                        fig = plt.gcf()
                        if fig is None:
                            print('~~~~~~~~~~~~ FIG IS NONE ~~~~~~~~~~~~~~')
                            pass ########### Return useful error here
                        plt.draw()

                        # Adjust layout to prevent legend cutoff
                        plt.tight_layout()  # Auto-adjust layout to fit everything
                        # Manually adjust legend if needed
                        fig.subplots_adjust(right=0.8)  # Make space on the right for the legend

                        if not fig.axes:
                            print("~~~~~~~~~~~~~~ Matplotlib figure has no axes. Nothing to convert. ~~~~~~~~~~~~~~")
                            print("fig\n", fig, "\n")
                        if not any(ax.has_data() for ax in fig.axes):
                            print("~~~~~~~~~~~~~~ Matplotlib figure has no plotted data. Cannot convert. ~~~~~~~~~~")
                            print("fig\n", fig, "\n")

                        # for ax in fig.axes:
                        #     if not ax.has_data():
                        #         ax.plot([0], [0], alpha=0)  # Invisible dummy point


                        # Try to convert Matplotlib figure to Plotly
                        try:
                            fig = mpl_to_plotly(fig)
                            # Specify layout size
                            fig.update_layout(
                                autosize=True,  # Allow auto-sizing within Dash
                                height=400,  
                                width=600  
                            )
                            self.figure = fig
                            self.output_type = 'plotly'
                        except Exception as e: #ValueError
                            # If an error occurs, display the static Matplotlib image instead
                            print('Error during mpl_to_plotly conversion, falling back to displaying a static image:\n', e, '\n********************************')
                            # Create the styled Matplotlib figure
                            fallback_fig = create_styled_matplotlib_figure(fig)
                            # Convert Matplotlib figure to base64 image
                            self.figure = fig_to_base64(fallback_fig)
                            self.output_type = 'img'   

    def infer_column_types(self, df):
        # Check if column is datetime column
        for col in df.columns:
            try:
                parsed_col = pd.to_datetime(col, errors='coerce') # , infer_datetime_format=True
                timestamp_ratio = parsed_col.notna().mean()  # Proportion of successfully converted values
                if timestamp_ratio > 0.9:  # If most values convert successfully, treat it as a timestamp
                    # Convert timestamp column to integer
                    df[col] = parsed_col
            except Exception:
                continue
        return df
