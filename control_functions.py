import numpy as np
import networkx as nx
from random import randint

def hub_control(graph, balls, step, burn_in=5000, burn_out=10000, control_interval=1000, control_fraction=0.1, control_balls_fraction=1):
    """
    Exercises control by periodically targeting hubs (high-degree nodes) and sending them a fixed number of balls of colour 0.
    :param graph: The graph on which dynamics are being simulated.
    :param balls: The current distribution of balls at nodes.
    :param step: The current time step.
    :param burn_in: The number of time steps to wait for before exercising any control.
    :param burn_out: The number of time steps after which to stop exercising control.
    :param control_interval: The interval in time steps at which control is to be exercised.
    :param control_fraction: The fraction of nodes to be targeted.
    :param control_balls_fraction: The total number of balls available to the controller at each iteration, as a fraction of the total number of nodes in the graph.
    """	
    # Array to store control changes
    controls = []	
 
    # Check if anything is to be done at this step
    if step >= burn_in and step < burn_out and step % control_interval == 0:

    	num_nodes = graph.number_of_nodes()
    	degree = np.asarray(graph.out_degree().values())

    	# Get indices of highest-degree nodes (hubs)
    	num_hubs = int(num_nodes*control_fraction)
    	top_hubs = degree.argsort()[-num_hubs:][::-1]

        balls_per_node = float(num_nodes*control_balls_fraction) / num_hubs
	for hub in range(num_hubs):
            # Send some colour 0 balls to this hub
            controls.append((top_hubs[hub], 0, balls_per_node))

    return controls


def degree_control(graph, balls, step, burn_in=5000, burn_out=10000, control_interval=1000, control_fraction=0.1, control_balls_fraction=1):
    """
    Exercises control by periodically targeting a certain fraction of nodes (with higher-degree nodes more likely to be chosen) and sending them a fixed number of balls of colour 0.
    :param graph: The graph on which dynamics are being simulated.
    :param balls: The current distribution of balls at nodes.
    :param step: The current time step.
    :param burn_in: The number of time steps to wait for before exercising any control.
    :param burn_out: The number of time steps after which to stop exercising control.
    :param control_interval: The interval in time steps at which control is to be exercised.
    :param control_fraction: The fraction of nodes to be targeted.
    :param control_balls_fraction: The total number of balls available to the controller at each iteration, as a fraction of the total number of nodes in the graph.
    """	
    # Array to store control changes
    controls = []	
 
    # Check if anything is to be done at this step
    if step >= burn_in and step < burn_out and step % control_interval == 0:

    	num_nodes = graph.number_of_nodes()
    	cumulative_degree = np.cumsum(np.asarray(graph.out_degree().values()))
	
    	# Get indices of nodes to be targeted
    	num_targets = int(num_nodes*control_fraction)
	
        balls_per_node = float(num_nodes*control_balls_fraction) / num_targets
	for node in range(num_targets):
	    # Select index of node to target	
	    rand = randint(1,cumulative_degree[num_nodes-1])
	    node = 0
	    while cumulative_degree[node] < rand:
		node = node + 1

            # Send some colour 0 balls to this node
            controls.append((node, 0, balls_per_node))

    return controls


def random_control(graph, balls, step, burn_in=5000, burn_out=10000, control_interval=1000, control_fraction=0.1, control_balls_fraction=1):
    """
    Exercises control by periodically targeting a certain fraction of the nodes at random and sending them a fixed number of balls of colour 0.
    :param graph: The graph on which dynamics are being simulated.
    :param balls: The current distribution of balls at nodes.
    :param step: The current time step.
    :param burn_in: The number of time steps to wait for before exercising any control.
    :param burn_out: The number of time steps after which to stop exercising control.
    :param control_interval: The interval in time steps at which control is to be exercised.
    :param control_fraction: The fraction of nodes to be targeted.
    :param control_balls_fraction: The total number of balls available to the controller at each iteration, as a fraction of the total number of nodes in the graph.
    """
    # Array to store control changes
    controls = []	

    # Check if anything is to be done at this step
    if step >= burn_in and step < burn_out and step % control_interval == 0:

	num_nodes = graph.number_of_nodes()
    	num_targets = int(num_nodes*control_fraction)

        balls_per_node = float(num_nodes*control_balls_fraction)/num_targets
        for node in range(num_targets):
            # Send some colour 0 balls to an arbitrarily chosen node
            controls.append((int(num_nodes*node/num_targets), 0, balls_per_node))

    return controls


def tom_control(graph, balls, step, burn_in=5000, burn_out=10000, control_interval=1000, control_fraction=0.1, control_balls_fraction=1):
    """
        Exercises control by periodically targeting a certain fraction of the nodes at random and sending them a fixed number of balls of colour 0.
        :param graph: The graph on which dynamics are being simulated.
        :param balls: The current distribution of balls at nodes.
        :param step: The current time step.
    	:param burn_in: The number of time steps to wait for before exercising any control.
	:param burn_out: The number of time steps after which to stop exercising control.
        :param control_interval: The interval in time steps at which control is to be exercised.
        :param control_fraction: The fraction of nodes to be targeted.
        :param control_balls_fraction: The total number of balls available to the controller at each iteration, as a fraction of the total number of nodes in the graph.
    """
    # Array to store control changes
    controls = []

    if step >= burn_in and step < burn_out and step % control_interval == 0:
	influence = np.asarray([0] * graph.number_of_nodes())
    	for node in graph.nodes():
        	neighbors_array = graph.neighbors(node)

        for neighbors in neighbors_array:
            if balls[neighbors, 0] < balls[neighbors, 1]:
                influence[node] += 1

	num_nodes = graph.number_of_nodes()
    	num_influence = int(num_nodes * control_fraction)
    	top_influence_nodes = influence.argsort()[-num_influence:][::-1]

        balls_per_node = float(num_nodes*control_balls_fraction) / num_influence

        for i in range(num_influence):
            # Add balls of colour 0
            controls.append((top_influence_nodes[i], 0, balls_per_node))

    return controls


def broadcast_control(graph, balls, step, burn_in=5000, burn_out=10000, control_interval=1000, control_balls_fraction=1):
    """
    Exercises control by periodically sending all nodes a fixed number of balls of colour 0.
    :param graph: The graph on which dynamics are being simulated.
    :param balls: The current distribution of balls at nodes.
    :param step: The current time step.
    :param burn_in: The number of time steps to wait for before exercising any control.
    :param burn_out: The number of time steps after which to stop exercising control.
    :param control_interval: The interval in time steps at which control is to be exercised.
    :param control_balls_fraction: The total number of balls available to the controller at each iteration, as a fraction of the total number of nodes in the graph.
    """
    # Array to store control changes
    controls = []	

    # Check if anything is to be done at this step
    if step >= burn_in and step < burn_out and step % control_interval == 0: 

	num_nodes = graph.number_of_nodes()
    
        balls_per_node = float(num_nodes*control_balls_fraction) / num_nodes

        for node in range(num_nodes):
            # Send some colour 0 balls to this node
            controls.append((node, 0, balls_per_node))

    return controls


def young_control(graph, balls, step, burn_in=5000, burn_out=10000, control_interval=1000, control_fraction=0.1, control_balls_fraction=1):
    """
    Exercises control by periodically targeting nodes with few balls and sending them a fixed number of balls of colour 0.
    :param graph: The graph on which dynamics are being simulated.
    :param balls: The current distribution of balls at nodes.
    :param step: The current time step.
    :param burn_in: The number of time steps to wait for before exercising any control.
    :param burn_out: The number of time steps after which to stop exercising control.
    :param control_interval: The interval in time steps at which control is to be exercised.
    :param control_fraction: The fraction of nodes to be targeted.
    :param control_balls_fraction: The total number of balls available to the controller at each iteration, as a fraction of the total number of nodes in the graph.
    """	
    # Array to store control changes
    controls = []	
 
    # Check if anything is to be done at this step
    if step >= burn_in and step < burn_out and step % control_interval == 0:

    	num_nodes = graph.number_of_nodes()
	for i in range(num_nodes):
    		experience[i] = balls[i,0] + balls[i,1]

    	# Get indices of youngest nodes
    	num_targets = int(num_nodes*control_fraction)
    	youngest = experience.argsort()[:num_targets][::1]

        balls_per_node = float(num_nodes*control_balls_fraction) / num_targets
	for node in range(num_targets):
            # Send some colour 0 balls to this hub
            controls.append((youngest[node], 0, balls_per_node))

    return controls
