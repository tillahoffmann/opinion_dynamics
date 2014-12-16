from simulation import *
import matplotlib.pyplot as plt

def main_simulated(stationarity):

    # Define a number of nodes and simulation steps
    num_nodes = 100
    num_steps = 50000
    seed=42
    np.random.seed(seed)
    concentration = 1

    # Create a graph
    graph = nx.erdos_renyi_graph(num_nodes, 5 / float(num_nodes), seed)
    # Remove any isolated nodes and relabel the nodes
    graph = remove_isolates(graph)
    # Obtain the number of remaining nodes and initialise the alpha and beta vectors
    num_nodes = graph.number_of_nodes()
    alpha = concentration * np.ones(num_nodes)
    beta = concentration * np.ones(num_nodes)

    # Run the simulation
    return simulate(graph, alpha, beta, num_steps, stationarity=stationarity)

def plot_stats(name, data):
    alphas, betas = data
    stats = SummaryStats(alphas, betas)
    stats.collect_stats()

    # Compute the fraction of `alpha` balls in the population and visualise
    probability = np.mean(alphas / (alphas + betas), axis=1)
    plt.figure()
    plt.plot(probability)
    plt.xlabel('Step number')
    plt.ylabel('Population probability')
    plt.tight_layout()

    # Plot the number of blue and red balls as a function of time
    plt.figure()
    plt.plot(np.sum(alphas, axis=1), color='b')
    plt.plot(np.sum(betas, axis=1), color='r')
    plt.xlabel('Step number')
    plt.ylabel('Number of balls')
    plt.tight_layout()

    plt.figure()
    plt.plot(stats.stats["mean_entropy_per_urn"])
    plt.xlabel('Step number')
    plt.ylabel('Mean entropy')

    plt.show()

plot_stats('moran', main_simulated('moran'))
plot_stats('polya', main_simulated('polya'))