__author__ = 'tillhoffmann'

import numpy as np
import networkx as nx


def remove_isolates(graph, inplace=False, relabel=True):
    """
    Removes isolated nodes from the graph.
    :param graph: The graph to remove isolated nodes from.
    :param inplace: Removes the nodes from the specified graph if `True` and creates a copy otherwise.
    :param relabel: Relabels the nodes with zero-based integers if `True`.
    :return:
    """
    # Copy the graph if not `inplace`
    if not inplace:
        graph = graph.copy()
    # Remove the isolated nodes
    graph.remove_nodes_from(nx.isolates(graph))
    # Relabel the graph if desired
    if relabel:
        graph = nx.convert_node_labels_to_integers(graph)
    return graph


class NodeSelector:
    """
    A class that specifies how nodes are selected to participate in information exchange.
    """

    def __init__(self, method='uniform'):
        """
        Initialises a `NodeSelector` instance.
        :param method: Selects nodes uniformly at random if 'uniform', selects nodes proportional to their degree
        if `degree`, selects nodes proportional to the number of balls in their urn if 'confidence'.
        """
        self.method = method

    def __call__(self, alpha, beta, graph, nodes=None):
        # If no nodes are specified, select from all the nodes
        if nodes is None:
            nodes = graph.nodes()

        # Select a node given the prescription
        if self.method == 'uniform':
            return np.random.choice(nodes)
        elif self.method == 'degree':
            weights = graph.degree().values()
        elif self.method == 'confidence':
            weights = alpha + beta
        else:
            raise ValueError("Node selection method '{}' is invalid.".format(self.method))

        weights = np.asarray(weights) / float(np.sum(weights))
        return np.random.choice(nodes, p=weights)


class SummaryStats:
    """
    A class to collect some summary metrics
    """
    def __init__(self, alphas, betas):
        self.alphas = alphas
        self.betas = betas
        self.prob_alpha = self.alphas / (self.alphas + self.betas)
        self.stats = {}

    def collect_stats(self):
        """Collect the statisics for one run, summarising over nodes"""

        self.stats["mean_prob_alpha_per_urn"] = np.mean(self.prob_alpha, axis=1)
        self.stats["mean_delta_prob_alpha_per_urn"] \
            = self._mean_delta_prob_alpha_per_urn()

        #ball mean_prob_alpha_for_balls
        #ball mean_delta_prob_alpha_for_balls
        #mean_entropy_per_urn -- Till

        ## Measures of distribution of confidence: sum(alpha+beta) - 2
        #correlation of confidence with sum(neighbours)
        #std of sum(alpha+beta) of urns
        #entropy of sum(alpha+beta) of urns
        #gini distribution of sum(alpha+beta)

    def _mean_delta_prob_alpha_per_urn(self):
        """How delta alpha changes"""
        delta_nodes = []
        for node in np.rollaxis(self.prob_alpha, 1):
            delta_nodes.append(np.ediff1d(node))
        return np.mean(np.array(delta_nodes), axis=0)


class PriorUpdater:
    """
    A class that specifies how the prior of a node is updated when it receives a ball.
    """

    def __init__(self, method='normal', weight=1):
        """
        Initialises a `PriorUpdateer` instance.
        :param method: Currently only 'normal' is implemented.
        """
        self.method = method
        self.weight = weight

    def __call__(self, alpha, beta, node, ball):
        if self.method == 'normal':
            if ball == 0:
                alpha[node] += self.weight
            else:
                beta[node] += self.weight
        else:
            raise ValueError("Prior update method '{}' is invalid.".format(self.method))


class Stationarity:
    """
    A class that specifies how the model attains a stationary distribuion.
    """

    def __init__(self, method='polya', weight=1):
        """
        Initialises a `Stationarity` instance.
        :param method: Lets the number of balls in the system increase steadily if 'polya' and keeps the number of
        balls constant if 'moran' (cf. Moran, Patrick Alfred Pierce (1962). The Statistical Processes of Evolutionary
        Theory. Oxford: Clarendon Press.).
        """
        self.method = method
        self.weight = weight

    def __call__(self, alpha, beta, node, ball):
        if self.method == 'polya':
            # Nothing to do
            pass
        elif self.method == 'moran':
            # Compute the probability of selecting a ball of either colour...
            a = alpha[node]
            b = beta[node]
            probability = a / float(a + b)
            # ... and delete it
            if probability < np.random.uniform():
                beta[node] -= self.weight
            else:
                alpha[node] -= self.weight


def simulate(graph, alpha, beta, num_steps, node_selector='uniform', prior_updater='normal', stationarity=None,
             neighbor_selector=None):
    """
    Simulates the evolution of the balls in the urns of each node in a network. IMPORTANT: The order of keyword
    arguments is not guaranteed to remain the same. Always set keyword arguments with keyword syntax.
    :param graph: The interaction topology.
    :param alpha: The initial number of balls of type 0.
    :param beta: The initial number of balls of type 1.
    :param num_steps: The number of simulation steps to run.
    :param node_selector: How to select nodes to transmit information.
    :param prior_updater: How to update the belief of nodes.
    :param neighbor_selector: How to select neighbors.
    :return: Two matrices `alpha[t,i]` and `beta[t,i]`, where `t` refers to the simulation step and `i` to the node.
    """
    # Ensure there are no isolated nodes
    degree = np.asarray(graph.degree().values())
    assert np.all(degree > 0), "The graph must not contain isolated nodes. You can remove isolated nodes by " \
                               "calling `remove_isolates`."

    # Ensure the nodes are properly labelled
    nodes = np.asarray(graph.nodes())
    assert np.all(nodes == np.arange(len(nodes))), "The nodes must be labelled with a zero-based index. You can " \
                                                   "relabel nodes by calling `remove_isolates` or `nx.convert_node_labels_to_integers`."

    # Copy the initial parameters as numpy arrays
    alpha = np.array(alpha)
    beta = np.array(beta)

    # Ensure we have the right dimensions for the parameters
    num_nodes = graph.number_of_nodes()
    assert num_nodes == len(alpha) and num_nodes == len(beta), "The `alpha` and `beta` vectors must have exactly " \
                                                               "the same number of elements as there are nodes in the network."

    # Initialise the node_selector...
    if type(node_selector) is str:
        # ...based on a method string
        node_selector = NodeSelector(node_selector)
    elif callable(node_selector):
        # ...all good--the selector is callable
        pass
    else:
        raise ValueError("'{}' is not a valid node selector.".format(node_selector))

    # Initialise the neighbor_selector
    if neighbor_selector is None:
        # Use the same method as the node selector unless specified otherwise
        neighbor_selector = node_selector
    elif type(neighbor_selector) is str:
        neighbor_selector = NodeSelector(neighbor_selector)
    elif callable(neighbor_selector):
        # ...all good--the selector is callable
        pass
    else:
        raise ValueError("'{}' is not a valid neighbor selector.".format(neighbor_selector))

    # Initialise the prior_updater
    if type(prior_updater) is str:
        prior_updater = PriorUpdater(prior_updater)
    elif callable(prior_updater):
        # ...all good--the updater is callable
        pass
    else:
        raise ValueError("'{}' is not a valid prior updater.".format(neighbor_selector))

    # Initialise the stationarity
    if type(stationarity) is str:
        stationarity = Stationarity(stationarity)
    elif callable(stationarity) or stationarity is None:
        # ...all good--the stationarity is callable or we aren't doing anything
        pass
    else:
        raise ValueError("'{}' is not a valid stationarity specification".format(stationarity))

    # Initialise the matrices containing the hyperparameters
    # as a function of time for each node
    alphas = [alpha.copy()]
    betas = [beta.copy()]

    for step in range(num_steps):
        # Select a node as the transmitter of information
        node = node_selector(alpha, beta, graph)
        # Obtain the neighbors of the node
        neighbors = graph.neighbors(node)
        neighbor = neighbor_selector(alpha[neighbors], beta[neighbors], graph, neighbors)
        # Compute the probability of an alpha ball
        probability = alpha[node] / (alpha[node] + beta[node])
        # Obtain a ball from the urn
        ball = probability < np.random.uniform()
        # Update the prior of the neighbor
        if stationarity is not None:
            stationarity(alpha, beta, neighbor, ball)
        prior_updater(alpha, beta, neighbor, ball)

        # Add the results
        alphas.append(alpha.copy())
        betas.append(beta.copy())

    return np.array(alphas), np.array(betas)


def _main():
    # Import plotting library
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

    summary_stats = SummaryStats(alphas, betas)
    summary_stats.collect_stats()

    # Compute the fraction of `alpha` balls in the population and visualise
    probability = summary_stats.stats["mean_prob_alpha_per_urn"]
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


if __name__ == '__main__':
    _main()