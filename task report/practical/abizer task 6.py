# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 10:45:04 2020

@author: abizer
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import time
# genarating a random adjacency matrix for a undirected and unweighted graph
def a_matrix(x):
    ''' generates a adjacancy matrix of size x'''
    g=np.zeros([x+1, x+1])
    g[:, 0]= np.arange(x+1)
    g[0, :]=np.arange(x+1)
    for i in range(1,x+1):
        for j in range(1,x+1):
            if i !=j:
                g[i, j]=g[j, i]=np.random.randint(0,10)
            if sum(sum(g[1:, 1:]))>=9000:
                break
        if sum(sum(g[1:,1:]))>=9000:
            break
    return g
adj_mat=a_matrix(100)
print(adj_mat)

#generate adjacency list
def convert(a): 
    adjList = defaultdict(list) 
    for i in range(1,len(a)): 
        for j in range(1,len(a[i])): 
                       if a[i][j]== 1: 
                           adjList[i].append(j) 
    return adjList 
adj_list = convert(adj_mat)
for i in adj_list: 
    print(i, end =".array") 
    for j in adj_list[i]: 
        print("  {},".format(j), end ="") 
    print()



# generating graph from the adjacancy matrix
def graph(m):
    G=nx.Graph()
    for i in range(1,len(m)):
        for j in range(1, len(m)):
            if adj_mat[i][j]==1:
                G.add_edge(i, j)
                G.add_edge(j,i)
    return G



# returns the connected components.
def connected_components_list(graph):
    visited = []
    connected_components = []
    for node in graph.nodes:
        if node not in visited:
            cc = [] #connected component
            visited, cc = dfs(graph, node, visited, cc)
            connected_components.append(cc)
    return connected_components

# the dfs algorith to check the visited vertices
def dfs(graph, start, visited, path):
    if start in visited:
        return visited, path
    visited.append(start)
    path.append(start)
    for node in graph.neighbors(start):
        visited, path = dfs(graph, node, visited, path)
    return visited, path
Graph = nx.from_numpy_matrix(np.array(adj_mat[1:, 1:]))
# prints the number of connected components and connections
connections=connected_components_list(Graph)
print(len(connections))
for i in connections:
    print(i)


        
        
class Graph():
    def __init__(self):
        """
        self.edges is a dict of all possible next nodes
        e.g. {'X': ['A', 'B', 'C', 'E'], ...}
        self.weights has all the weights between two nodes,
        with the two nodes as a tuple as the key
        e.g. {('X', 'A'): 7, ('X', 'B'): 2, ...}
        """
        self.edges = defaultdict(list)
        self.weights = {}
    
    def add_edge(self, from_node, to_node, weight):
        # Note: assumes edges are bi-directional
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.weights[(from_node, to_node)] = weight
        self.weights[(to_node, from_node)] = weight
graph = Graph()
for i in range (1,101):
    for j in range(1,101):
        z=adj_mat[i][j]
        if z !=0:
            graph.add_edge(i, j, z)
def dijsktra(graph, initial, end):
    # shortest paths is a dict of nodes
    # whose value is a tuple of (previous node, weight)
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()
    
    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            weight = graph.weights[(current_node, next_node)] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)
        
        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return "Route Not Possible"
        # next node is the destination with the lowest weight
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])
    
    # Work back through destinations in shortest path
    path = []
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        current_node = next_node
    # Reverse path
    path = path[::-1]
    return path




def time1(a,b):
    x=[]
    for i in range(10):
        start_time=time.time()
        y=dijsktra(graph, a,b )
        x.append(time.time()-start_time)
    x=sum(x)/10
    print(y)
    print(x)
