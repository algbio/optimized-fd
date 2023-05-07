#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
import networkx as nx
import gurobipy as gp
import subprocess
from gurobipy import GRB
from collections import deque
from bisect import bisect
from copy import deepcopy


def read_input_graphs(graph_file):
    graph = nx.MultiDiGraph()
    flow = dict()

    with open(graph_file, 'r') as file:
        raw_graphs = file.read().split('#')[1:]
        for g in raw_graphs:
            lines = g.split('\n')[1:]
            if not lines[-1]:
                lines = lines[:-1]
            graph.add_nodes_from(range(int(lines[0])))
            for e in lines[1:]:
                parts = e.split()
                key = graph.add_edge(int(parts[0]), int(parts[1]), f=float(parts[2]))
                flow[(int(parts[0]), int(parts[1], key))] = float(parts[2])

    return graph, flow

# Main class for Minimum flow decomposition
class Mfd:

    """
    Initialisation for minimum flow decomposition of a graph.
    Variable k represents the number of paths for the solution.
    Solve using mfd_algorithm method, initially k = -1.
    """
    def __init__(self, graph, flow):
        # Graph data
        self.G = graph
        self.flow = flow
        self.max_flow_value = self.find_max_flow_value()
        self.sources = self.find_sources()
        self.sinks = self.find_sinks()

        # Mfd data
        self.solved = False
        self.k = -1
        self.paths = []
        self.weights = []
        self.model = gp.Model('MFD')

        # greedy solution
        self.solved_greedy = False
        self.k_greedy = -1
        self.paths_greedy = []
        self.weights_greedy = []

    """
    Returns the maximum flow value of self.graph.
    """
    def find_max_flow_value(self):
        return max(self.flow)
    
    """
    Returns all sources of the graph (i.e., all nodes with no ingoing edges).
    """
    def find_sources(self):
        return [v for v in self.G.nodes() if self.G.in_degree(v) == 0]

    """
    Returns all sinks of the graph (i.e., all nodes with no outgoing edges).
    """
    def find_sinks(self):
        return [v for v in self.G.nodes() if self.G.out_degree(v) == 0]

    """
    Main algorithm to compute the minimum flow decomposition with ILP.
    """
    def mfd_algorithm(self, safe_paths=[]):
        # Note when changing the bounds, make sure not to break any of the following code
        lower_bound = len(safe_paths)
        upper_bound = len(paths_greedy) if solved_greedy else len(self.G.edges)

        if lower_bound == upper_bound:
            self.solved = True
            self.k = lower_bound
            
            # Make sure this is always correct whenever the bounds get changed
            self.paths = paths_greedy if solved_greedy else safe_paths

            return

        L, R = lower_bound, upper_bound
        # invariant: L no, R yes
        while L + 1 < R:
            i = (L + R) // 2
            if self.solve(safe_paths, i):
                R = i
            else:
                L = i
        return data

    """
    Solves the ILP given by build_ilp_model.
    """
    def solve(self, paths, k):
        self.build_ilp_model(paths, k)
        self.model.optimize()

        if model.status == GRB.OPTIMAL:
            I = [(u, v, i, j) for (u, v, i) in self.G.edges(keys=True) for j in range(k)]
            self.solved = True
            self.k = k
            self.paths = [list() for _ in range(k)]
            self.weights = [0] * len(range(k))
            
            for i in range(k):
                self.weights[i] = round(model.getVarByName(f'w[{i}]').x)
            for (u, v, i, j) in T:
                if round(model.getVarByName(f'x[{u},{v},{i},{j}]').x) == 1:
                    self.paths[j].append((u, v, i))
            for j in range(k):
                self.paths[j] = sorted(self.paths[j])
            return True
        return False
        
    """
    Build gurobi ILP model.
    """
    def build_ilp_model(self, paths, size):
        # create index sets
        T = [(u, v, i, k) for (u, v, i) in self.G.edges(keys=True) for k in range(size)]
        SC = list(range(size))

        self.model.setParam('LogToConsole', 0)
        self.model.setParam('Threads', threads)

        # Create variables
        x = self.model.addVars(T, vtype=GRB.BINARY, name='x') # path indicators
        w = self.model.addVars(SC, vtype=GRB.INTEGER, name='w', lb=0) # weights of paths
        z = self.model.addVars(T, vtype=GRB.CONTINUOUS, name='z', lb=0) # weight of path per edge

        # flow conservation
        for k in range(size):
            for v in self.G.nodes:
                if v in self.sources:
                    self.model.addConstr(sum(x[v, w, i, k] for _, w, i in self.G.out_edges(v, keys=True)) == 1)
                if v in self.sinks:
                    self.model.addConstr(sum(x[u, v, i, k] for u, _, i in self.G.in_edges(v, keys=True)) == 1)
                if v not in selfsources and v not in self.sinks:
                    self.model.addConstr(sum(x[v, w, i, k] for _, w, i in self.G.out_edges(v, keys=True)) - sum(x[u, v, i, k] for u, _, i in self.G.in_edges(v, keys=True)) == 0)

        # flow balance
        for (u, v, i, f) in self.G.edges(keys=True, data='f'):
            self.model.addConstr(f == sum(z[u, v, i, k] for k in range(size)))

        # linearization
        for (u, v, i) in self.G.edges(keys=True):
            for k in range(size):
                self.model.addConstr(z[u, v, i, k] <= max_flow_value * x[u, v, i, k])
                self.model.addConstr(w[k] - (1 - x[u, v, i, k]) * max_flow_value <= z[u, v, i, k])
                self.model.addConstr(z[u, v, i, k] <= w[k])

        # pre-defined constraints
        if paths != 0:
            for (k, path) in enumerate(paths):
                for (u, v, i) in path:
                    self.model.addConstr(x[u, v, i, k] == 1)
                    for k_ in range(len(paths)):
                        if k_ == k:
                            continue
                        self.model.addConstr(x[u, v, i, k_] == 0)

        return x, w, z

    """
    Polynomial time flow decomposition.
    """
    def decompose_flow_heuristic(self):

        if solved_greedy:
            return False
        
        self.weights_greedy = []
        self.paths_greedy = []

        remaining_flow = sum(self.G['f'][s] for s in sources])
        edges = list(G.edges(keys=True))

        for e in edges:
            G.edges[e]['remaining_flow'] = graph.edges[e]['f']

        def extract_highest_weight_path():
            highest_weight_of_any_path = {v: remaining_flow for v in self.sinks} 

    		for v in list(reversed(list(nx.topological_sort(G)))):
        		if v not in self.sinks:
            		highest_weight_of_any_path[v] = max([min(G.edges[v,u,k]['remaining_flow'], highest_weight_of_any_path[u]) for (v,u,k) in G.out_edges(v, keys=True)])

    		largest_flow = max([v for v in [highest_weight_of_any_path[s] for s in self.sources]])

    		path = []
    		for s in self.sources:
        		first_edges = [e for e in G.out_edges(s, keys=True) if G.edges[e]['remaining_flow'] >= largest_flow]
        		if first_edges:
            		path.append(first_edges[0])
            		while [e for e in G.out_edges(path[-1][1], keys=True) if highest_weight_of_any_path[e[1]] >= largest_flow and G.edges[e]['remaining_flow'] >= largest_flow]:
                		path.append([e for e in G.out_edges(path[-1][1], keys=True) if highest_weight_of_any_path[e[1]] >= largest_flow and G.edges[e]['remaining_flow'] >= largest_flow][0])
            		break
            
            assert len(path) > 0
    		return path, highest_flow

        while remaining_flow > 0:

            path, weight = extract_weighted_path() if not greedy_min else extract_highest_weight_path()
            self.weights_greedy.append(weight)
            self.paths_greedy.append(path)

            for e in path:
                G.edges[e]['remaining_flow'] -= weight
            remaining_flow -= weight

        return True
    

# TODO: Monday:
# y-to-v reduction
# symmetry
# greedy decomp       DONE
# mwa                 DONE
# remove s-t paths

# Install gurobi
# Run first tests
# Fernando

def edge_mwa_safe_paths(mfd, longest_safe_path_of_edge, max_safe_paths):
    graph = mfd.G
    greedy_paths = mfd.paths_greedy

    N = graph.number_of_nodes()
    M = graph.number_of_edges()
    orig_N = N
    orig_M = M

    # add node between every edge
    E = {u: list() for u in graph.nodes}
    newnode_to_edge = dict()
    edge_to_newnode = dict()
    for (u,v,i) in graph.edges:
        x = N
        E[u].append(x)
        E[x] = [v]
        newnode_to_edge[x] = (u,v,i)
        edge_to_newnode[u,v,i] = x
        N += 1
        M += 1
    assert M == 2*orig_M


    """
        Call C++ code to calculate maximum weight antichain
        Program mwa calls naive_minflow_reduction -> naive_minflow_solve -> maxantichain_from_minflow
    """
    mwa_input = "{} {}\n".format(N, M)
    for u in E:
        for v in E[u]:
            mwa_input += "{} {}\n".format(u, v)
    for v in range(orig_N):
        mwa_input += "{} {}\n".format(v, 0)
    for (u,v,i) in longest_safe_path_of_edge:
        x = edge_to_newnode[u,v,i]
        mwa_input += "{} {}\n".format(x, longest_safe_path_of_edge[u,v,i][0])

    res = subprocess.run(["./mwa"], input=mwa_input, text=True, capture_output=True)
    mwa = list(map(int, res.stdout.split(' ')))
    mwa = list(map(lambda x: x-1, mwa))
    for x in mwa:
        assert x >= orig_N

    mwa_safe_paths = []
    for x in mwa:
        u,v,i = newnode_to_edge[x]
        # Find corresponding longest safe path going through the edge (u,v,i)
        ip, iw = longest_safe_path_of_edge[u,v,i][1]
        l, r = max_safe_paths[ip][iw]
        mwa_safe_paths.append(greedy_paths[ip][l:r+1])

    return mwa_safe_paths

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='''
        Computes maximal safe paths for Minimum Flow Decomposition.
        This script uses the Gurobi ILP solver.
        ''',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-stats', '--output-stats', action='store_true', help='Output stats to file <output>.stats')
    parser.add_argument('-wt', '--weighttype', type=str, default='int+',
                        help='Type of path weights (default int+):\n   int+ (positive non-zero ints), \n   float+ (positive non-zero floats).')
    parser.add_argument('-t', '--threads', type=int, default=0,
                        help='Number of threads to use for the Gurobi solver; use 0 for all threads (default 0).')
    parser.add_argument('-seqt', '--sequential-threshold', type=int, default=0,
                        help='A repeated exponential search is performed to find the minimum flow decomposition, this parameter specifies the universe size at which a sequencial search is performed instead; use 0 to only perform sequential search (default 0).')

    parser.add_argument('-ilptb', '--ilp-time-budget', type=float, help='Maximum time (in seconds) that the ilp solver is allowed to take when computing safe paths for one graph')

    parser.add_argument('-uef', '--use-excess-flow', action='store_true', help='Use excess flow of a path to save ILP calls')
    parser.add_argument('-ugtd', '--use-group-top-down', action='store_true',
                        help='Use top down group testing')
    parser.add_argument('-ugbu', '--use-group-bottom-up', action='store_true',
                        help='Use bottom up group testing')
    parser.add_argument('-uy2v', '--use-y-to-v', action='store_true', help='Use Y to V contraction of the input graphs')

    parser.add_argument('-s', '--strategy', type=str, help='Strategy for extension and reduction of two-finger algorithm {scan, bin_search, exp_search, rep_exp_search}')
    parser.add_argument('-es', '--extension-strategy', type=str, help='Strategy for extension of two-finger algorithm {scan, bin_search, exp_search, rep_exp_search}')
    parser.add_argument('-rs', '--reduction-strategy', type=str, help='Strategy for reduction of two-finger algorithm {scan, bin_search, exp_search, rep_exp_search}')
    parser.add_argument('-ess', '--extension-strategy-small', type=str, help='Strategy for extension of two-finger algorithm {scan, bin_search, exp_search, rep_exp_search} when the search space is small (specified by threshold)')
    parser.add_argument('-esl', '--extension-strategy-large', type=str, help='Strategy for extension of two-finger algorithm {scan, bin_search, exp_search, rep_exp_search} when the search space is large (specified by threshold)')
    parser.add_argument('-rss', '--reduction-strategy-small', type=str, help='Strategy for reduction of two-finger algorithm {scan, bin_search, exp_search, rep_exp_search} when the search space is small (specified by threshold)')
    parser.add_argument('-rsl', '--reduction-strategy-large', type=str, help='Strategy for reduction of two-finger algorithm {scan, bin_search, exp_search, rep_exp_search} when the search space is large (specified by threshold)')
    parser.add_argument('-st', '--strategy-threshold', type=int, help='Search space threshold to switch from --<>-strategy-small to --<>-strategy-large')
    parser.add_argument('-est', '--extension-strategy-threshold', type=int, help='Search space threshold to switch from --extension-strategy-small to --extension-strategy-large')
    parser.add_argument('-rst', '--reduction-strategy-threshold', type=int, help='Search space threshold to switch from --reduction-strategy-small to --reduction-strategy-large')
    parser.add_argument('-safe', '--safepaths', type=str, help='Safe File filename')


    requiredNamed = parser.add_argument_group('required arguments')
    requiredNamed.add_argument('-i', '--input', type=str, help='Input filename', required=True)
    requiredNamed.add_argument('-o', '--output', type=str, help='Output filename', required=True)

    args = parser.parse_args()

    threads = args.threads
    if threads == 0:
        threads = os.cpu_count()
    print(f'INFO: Using {threads} threads for the Gurobi solver')

    
    mfd = Mfd(*read_input_graphs(args.input))

