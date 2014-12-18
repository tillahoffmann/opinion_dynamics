import numpy as np
import networkx as nx
from random import randint

def hub_control(graph, balls, step, control_interval=100, control_fraction=0.1, control_balls=10):
    '''
    Exercises control by periodically targeting hubs (high-degree nodes) and sending them a fixed number of balls of colour 0.
    :param graph: The graph on which dynamics are being simulated.
    :param balls: The current distribution of balls at nodes.
    :param step: The current time step.
    :param control_interval: The interval in time steps at which control is to be exercised.
    :param control_fraction: The fraction of nodes to be targeted.
    :param control_balls: The total number of balls available to the controller at each iteration.
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
	balls_per_node = float(control_balls)/num_hubs
	for hub in range(num_hubs):
		# Send some colour 0 balls to this hub
		controls.append((top_hubs[hub], 0, balls_per_node))
		
    return controls
	
def random_control(graph, balls, step, control_interval=100, control_fraction=0.1, control_balls=10):
    '''
    Exercises control by periodically targeting a certain fraction of the nodes at random and sending them a fixed number of balls of colour 0.
    :param graph: The graph on which dynamics are being simulated.
    :param balls: The current distribution of balls at nodes.
    :param step: The current time step.
    :param control_interval: The interval in time steps at which control is to be exercised.
    :param control_fraction: The fraction of nodes to be targeted.
    :param control_balls: The total number of balls available to the controller at each iteration.
    '''
    num_nodes = graph.number_of_nodes()
    num_targets = int(num_nodes*control_fraction)

    # Array to store control changes
    controls = []	

    # Check if anything is to be done at this step
    if step%control_interval==0:
	balls_per_node = float(control_balls)/num_targets
	for node in range(num_targets):
		# Send some colour 0 balls to an arbitrarily chosen node
		controls.append((int(num_nodes*node/num_targets), 0, balls_per_node))
		
    return controls
	
def broadcast_control(graph, balls, step, control_interval=100, control_balls=10):
    '''
    Exercises control by periodically sending all nodes a fixed number of balls of colour 0.
    :param graph: The graph on which dynamics are being simulated.
    :param balls: The current distribution of balls at nodes.
    :param step: The current time step.
    :param control_interval: The interval in time steps at which control is to be exercised.
    :param control_balls: The total number of balls available to the controller at each iteration.
    '''
    num_nodes = graph.number_of_nodes()
    
    # Array to store control changes
    controls = []	

    # Check if anything is to be done at this step
    if step%control_interval==0:
	balls_per_node = float(control_balls)/num_nodes
	for node in range(num_nodes):
		# Send some colour 0 balls to this node
		controls.append((node, 0, balls_per_node))
		
    return controls
	
