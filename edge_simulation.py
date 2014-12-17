__author__ = 'tillhoffmann'

import numpy as np
import networkx as nx

def simulate(graph, initial_balls, num_steps, control=None, **kwargs):
    """
    Generates a step sequence `S[t, :]` of prior updates. Each row of `S` represents one time step. At time `t`, node
    `S[t,0]` transfers a ball of color `S[t,2]` to node `S[t,1]`.
    :param graph: The graph to simulate dynamics on.
    :param initial_balls: A 2D-array representing the initial ball configuration. The element `initial_balls[i,c]`
    represents the number of balls of color `c` that node `i` holds.
    :param num_steps: The number of steps to simulate the dynamics for.
    :param control: A function that implements a control strategy.
    :param kwargs: Extra keyword arguments passed to the control function.
    :return: A step sequence.
    """
    edges = graph.edges()
    num_edges = len(edges)
    balls = np.array(initial_balls)

    steps = []

    for step in range(num_steps):
        #Apply a control strategy if supplied
        if control is not None:
            #Get the controls
            controls = control(graph, balls, step, **kwargs)
            #Apply the controls
            for node, ball, number in controls:
                balls[node, ball] += number
                steps.append((None, node, ball, number))

        # Select an edge
        u, v = edges[np.random.randint(num_edges)]
        # Compute the probability to draw a ball from the transmitter
        probability_u = float(balls[u, 0]) / np.sum(balls[u])
        # Draw a ball
        ball_u = int(probability_u < np.random.uniform())
        # Update the balls
        balls[v, ball_u] += 1
        # Add to the steps
        steps.append((u, v, ball_u, 1))

    return np.array(steps)


def evaluate_statistic(initial_balls, steps, statistic):
    """
    Evaluates the specified statistic as a function of the time.
    :param balls: A 2D-array representing the initial ball configuration. The element `initial_balls[i,c]` represents
    the number of balls of color `c` that node `i` holds.
    :param steps: The step sequence generated by `simulate`.
    :param statistic: A function that takes a ball configuration as input and computes a statistic.
    :return: A `np.array` of statistics evaluated as a function of time.
    """
    balls = np.array(initial_balls)
    statistics = []

    for _, r, ball, number in steps:
        balls[r, ball] += number
        statistics.append(statistic(balls))

    return np.asarray(statistics)


def statistic_mean_belief_urn_weighted(balls):
    """
    Computes the mean urn-weighted belief.
    :param balls: A 2D-array representing a ball configuration. The element `balls[i,c]` represents
    the number of balls of color `c` that node `i` holds.
    :return: The mean urn-weighted belief.
    """
    # Compute the mean belief of each urn
    belief = balls[:, 0] / np.sum(balls, axis=1)
    return np.mean(belief)


def statistic_mean_belief_ball_weighted(balls):
    """
    Computes the fraction of balls of color `0`.
    :param balls: A 2D-array representing a ball configuration. The element `balls[i,c]` represents
    the number of balls of color `c` that node `i` holds.
    :return: The fraction of balls of color `0`.
    """
    # Compute the fraction of balls of a given colour
    return np.sum(balls[:, 0]) / np.sum(balls)


def _main():
    # Import plotting library
    import matplotlib.pyplot as plt
    from scipy import stats
    # Fix a seed for reproducibility
    seed = None
    if seed is not None:
        np.random.seed(seed)

    graph = 'pair'
    vis = 'steady'
    num_steps = 5000
    num_nodes = 100
    num_runs = 100

    if graph == 'erdos':
        # Define a number of nodes and simulation steps
        num_nodes = 100

        # Set up the initial ball configuration
        balls = np.ones((num_nodes, 2))
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


    if vis == 'mean_belief':
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
            steps = simulate(graph, balls, num_steps)
            # Evaluate the mean belief urn-weighted
            mean_belief = evaluate_statistic(balls, steps, statistic_mean_belief_urn_weighted)
            values.append(mean_belief[-1])
            print run
        # Plot a histogram
        x = np.linspace(0, 1)
        kde = stats.gaussian_kde(values)
        y = kde(x)
        plt.plot(x, y)
        plt.xlabel('Steady-state belief')
        plt.ylabel('Probability')
        plt.tight_layout()
        plt.show()


if __name__=='__main__':
    _main()
