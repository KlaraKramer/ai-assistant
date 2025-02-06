from plotly.tools import mpl_to_plotly
from matplotlib.cm import Set1
from mpl_toolkits.axes_grid1 import make_axes_locatable

from helper_functions import *

class Vis:

    def __init__(self, id, df, rec_group=0, num_rec=0):
        self.id = id
        self.columns = None
        self.output_type = None
        self.lux_vis = None
        self.figure = None

        # Generate recommendations and store the resulting dictionary explicitly
        recommendations = df.recommendation
        if recommendations:
            # Store the recommendation options (e.g., Occurrence, Correlation, Temporal)
            rec_options = [key for key in recommendations]
            # Access first recommendation group
            if len(rec_options) > rec_group:
                self.rec_type = rec_options[rec_group]
            else:
                self.rec_type = rec_options[0]
            self.selected_recommendations = recommendations[self.rec_type]

            # Plot figure
            if self.selected_recommendations:
                if len(self.selected_recommendations) > num_rec:
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

                # ax.legend(loc='upper right', bbox_to_anchor=(1, 1))

                # Try to convert Matplotlib figure to Plotly
                try:
                    self.figure = mpl_to_plotly(fig)
                    self.output_type = 'plotly'
                except ValueError:
                    # If an error occurs, display the static Matplotlib image instead
                    print('Error during mpl_to_plotly conversion, falling back to displaying a static image.')
                    # Create the styled Matplotlib figure
                    fallback_fig = create_styled_matplotlib_figure(fig)
                    # Convert Matplotlib figure to base64 image
                    self.figure = fig_to_base64(fallback_fig)
                    self.output_type = 'img'        
