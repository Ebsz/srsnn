#!/usr/bin/env python
#
# Visualize a network

import sys
import json

import matplotlib.pyplot as plt
import numpy as np
#import networkx as nx

#from force_graph import GraphWindow

from parse import parse_array1, parse_array2


#def parse_network(dat):

class Network():
    def __init__(self, data):
        print(data.keys())

        self.n = data['n']

        self.env = (data['env']['inputs'], data['env']['outputs'])

        self.network_cm = parse_array2(data['network_cm'])
        self.network_w = parse_array2(data['network_w'])

        self.input_cm = parse_array2(data['input_cm'])
        self.input_w = parse_array2(data['input_w'])

        #n = network['neurons']
        #self.neurons

        #self.input = (input_cm, input_w)

        #network = d[0]
        #print(network.keys())

        #self.neurons =
        #print(f"n: {n}")

        ##neurons = parse_array(network['neurons'])
        #print(neurons.keys())

    def __repr__(self):
        return f"N(n: {self.n}, env: {self.env})"

def args():
    if len(sys.argv) < 2:
        print("usage: [NETWORK.json]\n\n")
        assert(1+1 != 2)

    return sys.argv[1:]

def parse_graph(m):
    (rows, cols) = m.shape

    edges = []
    for i in range(rows):
        for j in range(cols):

            if m[i,j] == 1:
                edges.append((j,i))

    return edges

def run():
    files = args()

    d = []
    for path in files:
        with open(path) as f:
            d.append(json.load(f))

    assert(len(d) == 1) # Expect a single network

    network = Network(d[0])

    print(network.network_cm.shape)


    edges = parse_graph(network.network_cm)
    print(f"number of edges: {len(edges)}")

    print(network.input_cm.shape)


    #print(f"{edges}")

    g = nx.DiGraph()
    g.add_edges_from(edges)

    p = nx.spring_layout(g, k=0.5)

    nx.draw(g, p, node_color='gray', node_size=100, arrowsize=20)
    #nx.draw(g, p)
    #plt.show()
    plt.savefig("graph.png")

    #w = GraphWindow(N, edges) # , precompute=True
    #w.loop()


    #fun()

    #neurons = parse_array2(network['neurons'])
    #pretty_print(neurons)


if __name__ == "__main__":
    run()
