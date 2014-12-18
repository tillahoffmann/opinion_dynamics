import numpy as np
import networkx as nx
from control_functions import *
from edge_simulation import *

def _main():
    # Import plotting library
    import matplotlib.pyplot as plt
    # Fix a seed for reproducibility
    seed = None
    if seed is not None:
        np.random.seed(seed)

    # Define a number of nodes and simulation steps
    num_steps = 10000
    num_nodes = 100

    # Define a number of simulations to be run
    num_sim = 100

    # Define variables to store final mean beliefs over all simulations
    urn_avg = []
    ball_avg = []

    for sim in range(num_sim):

       	# Set up the initial ball configuration
       	balls = np.ones((num_nodes, 2))
       	# Set up a network
       	graph = nx.erdos_renyi_graph(num_nodes, 5 / float(num_nodes), seed)
       	graph = graph.to_directed()

        steps = simulate(graph, balls, num_steps)


    	# Evaluate the mean belief urn-weighted
        mean_belief = evaluate_statistic(balls, steps, statistic_mean_belief_urn_weighted)
	urn_avg.append(mean_belief[num_steps-1])
        # Visualise the mean belief
        #plt.plot(mean_belief, label='urn-weighted')

        # Evaluate the mean belief ball-weighted
        mean_belief = evaluate_statistic(balls, steps, statistic_mean_belief_ball_weighted)
	ball_avg.append(mean_belief[num_steps-1])
        # Visualise the mean belief
	'''
        plt.plot(mean_belief, label='ball-weighted')

        plt.xlabel('Time step')
        plt.ylabel('Urn-weighted mean belief')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
	'''

    print "Urn-weighted mean: {}; std. dev.: {}".format(np.mean(urn_avg), np.std(urn_avg))
    print "Ball-weighted mean: {}; std. dev.: {}".format(np.mean(ball_avg), np.std(ball_avg))

if __name__=='__main__':
    _main()
