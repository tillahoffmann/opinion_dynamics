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

def _main():

	"set up a graph"
	np.random.seed(50)
	num_steps = 10
	num_nodes=2
	G = nx.star_graph(num_nodes-1)
	G=G.to_directed()
	balls = np.ones((num_nodes, 2))	

	"I set the balls of one type to 0." 	
	balls[0,0]=0
	balls[1,0]=0

	print[balls]

	steps = simulate(G, balls, num_steps)


	"I ask for the belief and the total number of balls in each urn as a function of time."
	belief_f0 = evaluate_statistic(balls, steps, statistic_belief_f0)
	belief_f1 = evaluate_statistic(balls, steps, statistic_belief_f1)

	num_f0 = evaluate_statistic(balls, steps, statistic_num_f0)
	num_f1 = evaluate_statistic(balls, steps, statistic_num_f1)

	"output results - as you can see,balls of the type that should be absent are geerated."
	print(steps)
	print(num_f0)
	print(num_f1)

	print(belief_f0)
	print(belief_f1)

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


if __name__=='__main__':
    _main()
