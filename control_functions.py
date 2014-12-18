import numpy as np
import networkx as nx

def hub_control(graph, balls, step, control_interval=10, control_fraction=0.1):
    '''
    Exercises control by periodically targeting hubs (high-degree nodes) and setting all their balls to colour 0.
    :param graph: The graph on which dynamics are being simulated.
    :param balls: The current distribution of balls at nodes.
    :param step: The current time step.
    :param control_interval: The interval in time steps at which control is to be exercised.
    :param control_fraction: The fraction of nodes to be targeted.
    '''
    num_nodes = graph.number_of_nodes()
    degree = np.asarray(graph.degree().values())
	
    # Get indices of highest-degree nodes (hubs)
    num_hubs = int(num_nodes*control_fraction)
    top_hubs = degree.argsort()[-num_hubs:][::-1]

    # Array to store control changes
    controls = []	

    # Check if anything is to be done at this step
    if step%control_interval==0:
	for hub in range(num_hubs):
		# Change the colour of all type 1 balls to type 0
		controls.append((top_hubs[hub], 0, balls[top_hubs[hub], 1]))
		controls.append((top_hubs[hub], 1, -balls[top_hubs[hub], 1]))

    return controls



def tom_control(graph, balls, step, control_interval=10, control_fraction=0.1):


	influence = [0]*graph.number_of_nodes

	for node in graph.number_of_nodes():
		neighbors_array = graph.neighbors(node)

		for neighbors in neighbors_array:
			if balls(neighbors, 0) < balls(neighbors, 1)
				influence[node] + = 1


   	num_influence = int(num_nodes*control_fraction)
    	top_influence_nodes = influence.argsort()[-num_influence:][::-1]
	
	
	
    if step % control_interval==0:
	for i in range(num_influence):
		# Add balls of colour 0
		controls.append((top_influence_nodes[i], 0, balls[top_influence_nodes[i], amount]))

    return controls



	


