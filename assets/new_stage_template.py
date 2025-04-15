##################################################
### This is the template for adding new stages ###
### to the data cleaning pipeline              ###
##################################################


# Callback to handle the first render within the 'NEW-STAGE' stage
@app.callback(
    [Output(component_id='NEW-STAGE-output', component_property='children')],
    [Input(component_id='PREVIOUS-STAGE-end-btn', component_property='n_clicks')],
    prevent_initial_call=True,
    running=[(Output(component_id='NEW-STAGE-btn', component_property='disabled'), True, False)]
)
def render_NEW_STAGE(n_clicks):
    global current_df
    global previous_df
    global stage
    global step
    global vis_objects

    graph_list = []
    # First render
    if n_clicks > 0 and current_df is not None:
        log('Finish PREVIOUS-STAGE', 'user')
        stage = 'NEW-STAGE'
        step += 1
        previous_df = current_df.copy()
        # Access the last visualisation rendered on the right (human view)
        human_previous = vis_objects[-1]
        
        # Display the parallel coordinates plot
        vis_objects, graph_list = render_machine_view(vis_objects, current_df, graph_list)

        # CALL BACKEND FUNCTION
        current_df, NEW_STAGE_count = NEW_STAGE_function(current_df)
        right_df = current_df.copy()
        right_df.intent = extract_intent(human_previous.columns)
        # Display the scatterplot
        vis2 = Vis(len(vis_objects), right_df, enhance='NEW_STAGE_interest')
        # Catch the missing value error if applicable:
        if vis2.missing_value_flag:
            # Display the second visualisation (second recommendation - num_rec=1 - rather than the first as usual)
            temp_vis = Vis(len(vis_objects), current_df, num_rec=1, temporary=True)
            current_df.intent = extract_intent(temp_vis.columns)
            vis2 = Vis(len(vis_objects), current_df, enhance='NEW_STAGE_interest')
        # Populate vis_objects list for referring back to the visualisations
        vis_objects.append(vis2)
        # Append the graph, wrapped in a Div to track clicks, to graph_list
        graph2 = Graph_component(vis2)
        if graph2.div is not None:
            graph_list.append(graph2.div)
        else:
            print('No recommendations available. Please upload data first.')

        # Add to the action log
        message = 'LOGIC FOR DISPLAYING INFORMATION ABOUT DISCOVERIES IN THE NEW-STAGE'
        log(message, 'system')
        # Return all components
        graph_div = show_side_by_side(graph_list)
        new_div = html.Div(children=[
            html.P(message, style=text_col),
            graph_div,
            # Render the action dropdown
            dcc.Dropdown(
                placeholder='Select an action to take', 
                id={'type': 'NEW-STAGE', 'index': step},
                options={'option1': 'NEW-STAGE relevant options'}
            ),
            html.Br()
        ])
        return [new_div]
