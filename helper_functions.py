__author__ = 'tillhoffmann'

import numpy as np
import networkx as nx
from scipy.special import gammaln, polygamma
import matplotlib.pyplot as plt


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


def GraphType(num_nodes, str, p=0.05, m=3):
    """
    :param num_nodes: the number of nodes of the graph (if that option is available)
    :param str: the type of graph that is used. We have
                'erdos'         an erdos renyi graph
                'powerlaw'      a graph with powerlaw degree distribution
                'enron'         a social network graph loaded from
                                http://snap.stanford.edu/data/email-Enron.html. (36692 nodes)
                'karateclub'    some karate club graph
                'women'         women social network
    :return: the graph
    """
    if str == 'erdos':
        graph = nx.erdos_renyi_graph(num_nodes, p)
    elif str == 'powerlaw':
        graph = nx.powerlaw_cluster_graph(num_nodes, m, p)
    elif str == 'enron':
        graph = nx.Graph()
        edges = np.loadtxt('Enron.txt',skiprows=4)
        graph.add_edges_from(edges)
    elif str == 'karateclub':
        graph = nx.karate_club_graph()
    elif str == 'women':
        graph = nx.davis_southern_women_graph()
    elif str == 'pair':
        graph = nx.DiGraph()
        graph.add_edge(0,1)
        graph.add_edge(1,0)
    elif str == 'star':
        graph = nx.star_graph(num_nodes)
    elif str == 'cycle':
        graph = nx.cycle_graph(num_nodes)
    elif str == 'config':
        max_degree = int(num_nodes/5)
        #Create some degrees
        degrees = np.asarray(np.round(np.exp(np.log(max_degree) * np.random.uniform(size=num_nodes))), np.int)
        #Ensure the total number of degrees is even
        if sum(degrees) % 2 != 0:
            degrees[np.random.randint(num_nodes)] += 2 * np.random.randint(2) - 1
        #Create a graph and apply the configuration model
        graph = nx.Graph()
        graph = nx.configuration_model(degrees, graph)
        graph = graph.to_directed()

    return graph