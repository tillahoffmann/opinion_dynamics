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


class PriorUpdater:
    """
    A class that specifies how the prior of a node is updated when it receives a ball.
    """
    def __init__(self, method='normal'):
        self.method = method

    def __call__(self, alpha, beta, node, ball):
        if self.method == 'normal':
            if ball == 0:
                alpha[node] += 1
            else:
                beta[node] += 1
        else:
            raise ValueError("Prior update method '{}' is invalid.".format(self.method))


def simulate(graph, alpha, beta, num_steps, node_selector='uniform', prior_updater='normal', neighbor_selector=None):
    """
    Simulates the evolution of the balls in the urns of each node in a network.
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
    assert np.all(degree > 0), "The graph must not contain isolated nodes. You can remove isolated nodes by "\
    "calling `remove_isolates`."

    # Ensure the nodes are properly labelled
    nodes = np.asarray(graph.nodes())
    assert np.all(nodes == np.arange(len(nodes))), "The nodes must be labelled with a zero-based index. You can "\
    "relabel nodes by calling `remove_isolates` or `nx.convert_node_labels_to_integers`."

    # Copy the initial parameters as numpy arrays
    alpha = np.array(alpha)
    beta = np.array(beta)

    # Ensure we have the right dimensions for the parameters
    num_nodes = graph.number_of_nodes()
    assert num_nodes == len(alpha) and num_nodes == len(beta), "The `alpha` and `beta` vectors must have exactly "\
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
        #Use the same method as the node selector unless specified otherwise
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

    # Initialise the matrices containing the hyperparameters
    # as a function of time for each node
    alphas = [alpha]
    betas = [beta]

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
    num_steps = 1000

    # Create a graph
    graph = nx.erdos_renyi_graph(num_nodes, 5 / float(num_nodes))
    # Remove any isolated nodes and relabel the nodes
    graph = remove_isolates(graph)
    # Obtain the number of remaining nodes and initialise the alpha and beta vectors
    num_nodes = graph.number_of_nodes()
    alpha = np.ones(num_nodes)
    beta = np.ones(num_nodes)

    # Run the simulation
    alphas, betas = simulate(graph, alpha, beta, num_steps)

    # Compute the fraction of `alpha` balls in the population and visualise
    probability = np.mean(alphas / (alphas + betas), axis=1)
    plt.plot(probability)
    plt.xlabel('Step number')
    plt.ylabel('Population probability')
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    _main()