__author__ = 'tillhoffmann'

import numpy as np
from cedge_simulation import *
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats

# Fix a seed for reproducibility
seed = 42
if seed is not None:
    np.random.seed(seed)
    seed_rng(seed)

# If profile is `True`, no visualisation will be performed
profile = True
graph = 'pair'
vis = 'steady'
num_steps = 5000
num_nodes = 100
num_runs = 10

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

# Do some comparisons
if profile:
    from edge_simulation import simulate as psimulate
    from edge_simulation import evaluate_statistic as pevaluate_statistic
    from edge_simulation import statistic_mean_belief_ball_weighted as pstatistic_mean_belief_ball_weighted

    sim_times = []
    for _ in range(num_runs):
        # Measure run times for normal run
        dt = datetime.now()
        psimulate(graph, balls, num_steps)
        time = (datetime.now() - dt).total_seconds()
        # Measure run times for Cython run
        dt = datetime.now()
        steps = simulate(graph, balls, num_steps)
        ctime = (datetime.now() - dt).total_seconds()
        sim_times.append((time, ctime))

    sim_times = np.asarray(sim_times)

    print "simulate: {} +- {} s".format(np.mean(sim_times[:,0]), np.std(sim_times[:,0]))
    print "csimulate: {} +- {} s".format(np.mean(sim_times[:,1]), np.std(sim_times[:,1]))
    ratio = sim_times[:, 0] / sim_times[:, 1]
    print "ratio: {} +- {}".format(np.mean(ratio), np.std(ratio))

    eval_times = []
    for _ in range(num_runs):
        # Measure run times for normal run
        dt = datetime.now()
        pevaluate_statistic(balls, steps, pstatistic_mean_belief_ball_weighted)
        time = (datetime.now() - dt).total_seconds()
        # Measure run times for Cython run
        dt = datetime.now()
        evaluate_statistic(balls, steps, statistic_mean_belief_ball_weighted)
        ctime = (datetime.now() - dt).total_seconds()
        eval_times.append((time, ctime))

    eval_times = np.asarray(eval_times)

    print "simulate: {} +- {} s".format(np.mean(eval_times[:,0]), np.std(eval_times[:,0]))
    print "csimulate: {} +- {} s".format(np.mean(eval_times[:,1]), np.std(eval_times[:,1]))
    ratio = eval_times[:, 0] / eval_times[:, 1]
    print "ratio: {} +- {}".format(np.mean(ratio), np.std(ratio))

    exit()


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
        final_balls = simulate(graph, balls, num_steps, output='last')
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