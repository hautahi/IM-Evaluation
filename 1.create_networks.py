"""
- Creates the network csv files used in the evaluation
- Designed to be called from the command line with keyword arguments described below in the CommandLine() class. 
- Output is a csv file for each network
"""

import random, argparse, os
import pandas as pd
from igraph import *
import networkx as nx

# ---------------
# Define functions to create graphs
# ---------------

def gen_ER_random(nodes, p, name):

    # Generate Graph
    G = Graph.Erdos_Renyi(n=nodes,p=p,directed=True)

    # Transform into dataframe of edges
    source, target = [], []
    for edge in G.es:
        source.append(edge.source)
        target.append(edge.target)
    d = pd.DataFrame({'source': source,'target': target})

    # Save to file
    d.to_csv('./network_data/ER/' + name +'.csv', index=False)

def gen_WS(nodes, K, beta, name):

    # Generate Graph
    G = Graph.Watts_Strogatz(1,nodes,K,beta)

    # Transform into dataframe of edges
    source, target = [], []
    for edge in G.es:
        source.append(edge.source)
        target.append(edge.target)
    d = pd.DataFrame({'source': source,'target': target})

    # Save to file
    d.to_csv('./network_data/WS/' + name +'.csv', index=False)

def gen_SF(nodes, gam, name):

    # Generate Graph
    s = nx.utils.powerlaw_sequence(nodes, gam)
    s = [6*i for i in s]
    G = nx.expected_degree_graph(s, selfloops=False)

    # Transform into dataframe of edges
    source, target = [], []
    for edge in G.edges():
        source.append(edge[0])
        target.append(edge[1])
    d = pd.DataFrame({'source': source,'target': target})

    # Save to file
    d.to_csv('./network_data/SF/' + name +'.csv', index=False)

# ---------------
# Define command line argument class
# ---------------

class CommandLine:
    def __init__(self):
        parser = argparse.ArgumentParser(description = "The keyword arguments are:")
        parser.add_argument("-n", help = "Integer size of network (default=100)", required = False, default = 100)
        parser.add_argument("-vers", help = "Integer number of graphs of each type (default=10)", required = False, default = 10)

        argument = parser.parse_args()

        if argument.n:
            print("You have used '-n' with argument: {0}".format(argument.n))
        if argument.vers:
            print("You have used '-vers' with argument: {0}".format(argument.vers))

        self.n, self.vers = argument.n, argument.vers

# ---------------
# Generate Network Files
# ---------------

if __name__ == '__main__':
    
    # Read in keyword arguments 
    app = CommandLine()
    n, vers = int(app.n), int(app.vers)

    # Create folders if necessary
    if not os.path.exists("./network_data"):
        os.mkdir("./network_data")
    if not os.path.exists("./network_data/ER"):
        os.mkdir("./network_data/ER")
    if not os.path.exists("./network_data/WS"):
        os.mkdir("./network_data/WS")
    if not os.path.exists("./network_data/SF"):
        os.mkdir("./network_data/SF")

    # Generate Erdos-Renyi Network
    for p in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        for j in range(vers):

            random.seed(p * j)
        
            gen_ER_random(n, p, 'nodes' + str(n) + '_p' + str(p) + '_v' + str(j+1))

    # Generate Watts-Strogatz Networks
    for beta in [0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        for j in range(vers):

            random.seed(beta * j)
        
            gen_WS(n, 8, beta, 'nodes' + str(n) + '_beta' + str(beta) + '_v' + str(j+1))

    # Create SF Networks
    for gamma in [1.5,2,2.25,2.5,3,3.5,4]:
        for j in range(vers):

            random.seed(gamma * j)
        
            gen_SF(n, gamma, 'nodes' + str(n) + '_gamma' + str(gamma) + '_v' + str(j+1))

