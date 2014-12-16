from simulation import *
import matplotlib.pyplot as plt

# Define a number of nodes and simulation steps
num_nodes = 100
num_steps = 100000
np.random.seed(42)
concentration = 3

# Create a graph
graph = nx.erdos_renyi_graph(num_nodes, 5 / float(num_nodes))
# Remove any isolated nodes and relabel the nodes
graph = remove_isolates(graph)
# Obtain the number of remaining nodes and initialise the alpha and beta vectors
num_nodes = graph.number_of_nodes()
alpha = concentration * np.ones(num_nodes)
beta = concentration * np.ones(num_nodes)

# Run the simulation
alphas, betas = simulate(graph, alpha, beta, num_steps, stationarity='moran')

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

plt.show()