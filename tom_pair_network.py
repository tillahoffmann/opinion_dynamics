# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 13:24:42 2014

@author: tmm13
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

N = 5**3 # number of timesteps
nRuns = 10000

X1 = np.zeros((N,2))
X2 = np.zeros((N,2))

# initialise distro
X1[0] = [1, 1]
X2[0] = [1, 1]

def chooseBall(alpha,beta):
    p = float(alpha)/float(alpha+beta)
    choice = np.random.binomial(1, p)
    #print p, choice
    return choice

endPts = np.zeros((nRuns,2))

for n in range(1, nRuns):
    print n
    for t in range(1,N):            
        node = np.random.binomial(1, 0.5)
        
        if node == 0: # select a ball from first node
            ball = chooseBall(X1[t-1][0], X1[t-1][1])
            if ball == 0: # add a type b ball to node 2
                X1[t] = X1[t-1]
                X2[t] = X2[t-1] + [0, 1]
                
            elif ball == 1: # add a type a ball to node 2
                X1[t] = X1[t-1]
                X2[t] = X2[t-1] + [1, 0]
                
        elif node == 1:
            ball = chooseBall(X2[t-1][0], X2[t-1][1]) # select a ball from the second node
            if ball == 0: # add a type b ball to node 1
                X1[t] = X1[t-1]  + [0, 1]
                X2[t] = X2[t-1]
                
            elif ball == 1: # add a type a ball to node 1
                X1[t] = X1[t-1] + [1, 0]
                X2[t] = X2[t-1]
    
    # generate belief time series           
    belief1 = X1[:,0]/(X1[:,0] + X1[:,1])
    belief2 = X2[:,0]/(X2[:,0] + X2[:,1])
    
    # store endpoints    
    endPts[n,0] = belief1[-1]
    endPts[n,1] = belief2[-1]

sns.kdeplot(endPts[:,0], c="b")
sns.kdeplot(endPts[:,1], c="g")

x = np.linspace(0,1)
y = stats.beta(2,2).pdf(x)
plt.plot(x,y)

"""         
fig = plt.figure()
ax = fig.add_subplot(111)

dummy = np.linspace(0, N/4, N)
ax.plot(dummy, ls = "--", c = "k")
ax.plot(X1[:, 0], c = "b", label = "X1 alpha")
ax.plot(X1[:, 1], c = "b", ls="--", label = "X1 beta")
ax.plot(X2[:, 0], c = "g", label = "X2 alpha")
ax.plot(X2[:, 1], c = "g", ls="--", label = "X2 beta")

ax.plot(belief1, c="b", label = "Belief 1")
ax.plot(belief2, c="g", label = "Belief 2")

ax.axis([0, N, 0, 1])
ax.legend(loc=2)
ax.set_xlabel("Timestep")
ax.set_ylabel("Number of balls")
"""

        
            
        

        