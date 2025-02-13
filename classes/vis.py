from plotly.tools import mpl_to_plotly
from matplotlib.cm import Set1
from mpl_toolkits.axes_grid1 import make_axes_locatable

from helper_functions import *

class Vis:

    def __init__(self, id, df, rec_group=0, num_rec=0, machine_view=False, enhance=None):
        self.id = id
        self.columns = None
        self.output_type = None
        self.lux_vis = None
        self.figure = None
        self.machine_view = machine_view
        self.enhance = enhance

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
            recommendations = df.recommendation
            if recommendations:
                # Store the recommendation options (e.g., Occurrence, Correlation, Temporal)
                rec_options = [key for key in recommendations]
                print('****************************rec_options: ', rec_options, '**************************')
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
                print('****************************selected_recs: ', self.selected_recommendations, '*******************')

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
                    
                    # Initialise variables that will be specified in the fig_code 
                    fig, ax = plt.subplots()
                    tab20c = plt.get_cmap('tab20c')
                    # Render the visualisation using Lux
                    try:
                        fig_code = self.lux_vis.to_matplotlib()
                    except ValueError:
                        print('Error in to_matplotlib()')
                        fig_code = ''
                    fixed_fig_code = fix_lux_code(fig_code)
                    exec(fixed_fig_code)

                    # Capture the current Matplotlib figure
                    fig = plt.gcf()
                    plt.draw()

                    # Adjust layout to prevent legend cutoff
                    plt.tight_layout()  # Auto-adjust layout to fit everything
                    # Manually adjust legend if needed
                    fig.subplots_adjust(right=0.8)  # Make space on the right for the legend

                    # ax.legend(loc='upper right', bbox_to_anchor=(1, 1))

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
                    except ValueError:
                        # If an error occurs, display the static Matplotlib image instead
                        print('Error during mpl_to_plotly conversion, falling back to displaying a static image.')
                        # Create the styled Matplotlib figure
                        fallback_fig = create_styled_matplotlib_figure(fig)
                        # Convert Matplotlib figure to base64 image
                        self.figure = fig_to_base64(fallback_fig)
                        self.output_type = 'img'        
