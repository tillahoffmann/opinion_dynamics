__author__ = 'ThomasOuldridge'

import numpy as np
import networkx as nx
from edge_simulation import *
try:
    import matplotlib.pyplot as plt
except:
    raise


""" 
pos=nx.spring_layout(G)
colors="blue"
nx.draw(G,pos,node_color='#A0CBE2',edge_color=colors,width=4,edge_cmap=plt.cm.Blues,with_labels=False)
plt.savefig("edge_colormap.png") # save as png
plt.show() # display
"""

def statistic_belief_f0(balls):
	belief = balls[0,0] / (balls[0,0]+balls[0,1])
    	return (belief)

def statistic_belief_f1(balls):
	belief = balls[1,0] / (balls[1,0]+balls[1,1])
    	return (belief)

def statistic_num_f0(balls):
	num = (balls[0,0]+balls[0,1])
    	return (num)

def statistic_num_f1(balls):
	num = (balls[1,0]+balls[1,1])
    	return (num)

np.random.seed(70)
num_steps = 100000
num_nodes=2
G = nx.star_graph(num_nodes-1)
G = G.to_directed()
balls = np.ones((num_nodes, 2))

steps = simulate(G, balls, num_steps)
belief_f0 = evaluate_statistic(balls, steps, statistic_belief_f0)
belief_f1 = evaluate_statistic(balls, steps, statistic_belief_f1)

num_f0 = evaluate_statistic(balls, steps, statistic_num_f0)
num_f1 = evaluate_statistic(balls, steps, statistic_num_f1)

plt.plot(num_f0,num_f1)   
plt.xlabel('num f0')
plt.ylabel('num f1')
plt.tight_layout()
plt.show()

plt.plot(belief_f0)
plt.plot(belief_f1)
plt.plot(belief_f1+belief_f0)
plt.xlabel('Time step')
plt.ylabel('Belief of f0, f1')
plt.tight_layout()
plt.show()



