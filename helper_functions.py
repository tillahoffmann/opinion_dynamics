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


def GraphType(num_nodes, str):
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
        graph = nx.erdos_renyi_graph(num_nodes, 5 / float(num_nodes))
    elif str == 'powerlaw':
        graph = nx.powerlaw_cluster_graph(num_nodes, 3,5 / float(num_nodes))
    elif str == 'enron':
        graph = nx.Graph()
        edges = np.loadtxt('Enron.txt',skiprows=4)
        graph.add_edges_from(edges)
    elif str == 'karateclub':
        graph = nx.karate_club_graph()
    elif str == 'women':
        graph = nx.davis_southern_women_graph()

    return graph