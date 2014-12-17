__author__ = 'tillhoffmann'

import numpy as np
from cedge_simulation import simulate as csimulate
from cedge_simulation import evaluate_statistic as cevaluate_statistic
from cedge_simulation import seed_rng
from datetime import datetime
from edge_simulation import evaluate_statistic, statistic_mean_belief_ball_weighted, \
    statistic_mean_belief_urn_weighted, simulate
import networkx as nx

# Import plotting library
import matplotlib.pyplot as plt
from scipy import stats
# Fix a seed for reproducibility
seed = 42
if seed is not None:
    np.random.seed(seed)
    seed_rng(seed)

graph = 'erdos'
vis = 'steady'
num_steps = 5000
num_nodes = 100
num_runs = 10000

if graph == 'erdos':
    # Define a number of nodes and simulation steps
    num_nodes = 100

    # Set up a network
    graph = nx.erdos_renyi_graph(num_nodes, 5 / float(num_nodes), seed)
    graph = graph.to_directed()
elif graph == 'pair':
    # Create a pair graph
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 0)])
    num_nodes=2

# Initialise
balls = np.ones((num_nodes, 2))


if vis == 'trajectory':
    # Simulate one trajectory
    steps = simulate(graph, balls, num_steps)
    # Evaluate the mean belief urn-weighted
    mean_belief = evaluate_statistic(balls, steps, statistic_mean_belief_urn_weighted)
    # Visualise the mean belief
    plt.plot(mean_belief, label='urn-weighted')

    # Evaluate the mean belief ball-weighted
    mean_belief = evaluate_statistic(balls, steps, statistic_mean_belief_ball_weighted)
    # Visualise the mean belief
    plt.plot(mean_belief, label='ball-weighted')

    plt.xlabel('Time step')
    plt.ylabel('Urn-weighted mean belief')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
elif vis == 'steady':
    values = []
    # Simulate a number of trajectories
    for run in range(num_runs):
        # Simulate one trajectory
        dt = datetime.now()
        final_balls = csimulate(graph, balls, num_steps, output='last')
        if run % 1000 == 0:
            print run, (datetime.now() - dt).total_seconds()
        # Evaluate the mean belief urn-weighted
        values.append(statistic_mean_belief_ball_weighted(final_balls))

    # Plot a histogram
    plt.hist(values, color='k', histtype='stepfilled',
             alpha=.1, normed=True, range=(0, 1), bins=20)

    #Plot a KDE
    x = np.linspace(0, 1)
    kde = stats.gaussian_kde(values)
    y = kde(x)
    plt.plot(x, y)

    if graph=='pair':
        #Plot a beta distribution
        a, b = np.mean(balls, axis=0) + 1
        y = stats.beta(a, b).pdf(x)
        plt.plot(x, y)

    plt.xlabel('Steady-state belief')
    plt.ylabel('Probability')
    plt.tight_layout()
    plt.show()