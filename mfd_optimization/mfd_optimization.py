#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
import io
import networkx as nx
import gurobipy as gp
import subprocess
from gurobipy import GRB
from collections import deque
from bisect import bisect
from copy import deepcopy

# Main class for Minimum flow decomposition
class Mfd:

    """
    Initialisation for minimum flow decomposition of a graph.
    Variable k represents the number of paths for the solution.
    Solve using mfd_algorithm method, initially k = -1.
    Graph should be directed and acyclic, and contain a flow,
    with flow conservation on every node that is not a source
    or a sink.
    """
    def __init__(self, graph, number_of_contracted_paths):
        # Graph data
        self.G = graph
        self.max_flow_value = self.find_max_flow_value()
        self.sources = self.find_sources()
        self.sinks = self.find_sinks()
        self.total_flow = self.find_total_flow()
        self.max_outdeg = self.find_max_out_degree()
        self.max_indeg = self.find_max_in_degree()
        self.number_of_contracted_paths = number_of_contracted_paths

        # Mfd data
        self.solved = False
        self.k = 0
        self.paths = []
        self.weights = []
        self.model_status = None
        self.runtime = 0

        # greedy solution
        self.solved_greedy = False
        self.k_greedy = 0
        self.paths_greedy = []
        self.weights_greedy = []
        self.safe_paths = []

    """
    Returns the maximum flow value of self.graph.
    """
    def find_max_flow_value(self):
        flow = 0
        for e in self.G.edges:
            flow = max(flow, self.G.edges[e]['f'])
        return flow

    """
    Returns the total flow of the graph. Must be called
    after find_sources().
    """
    def find_total_flow(self):
        assert len(self.sources) > 0

        flow = 0
        for s in self.sources:
            flow += self.out_flow(s)
        return flow

    """
    Returns the maximum out degree of all nodes in the graph.
    """
    def find_max_out_degree(self):
        return max(self.G.out_degree(v) for v in self.G.nodes)

    """
    Returns the maximum in degree of all nodes in the graph.
    """
    def find_max_in_degree(self):
        return max(self.G.in_degree(v) for v in self.G.nodes)
    
    """
    Returns the sum of flow going out of a node v.
    """
    def out_flow(self, v):
        flow = 0
        for v,u,i in self.G.out_edges(v, keys=True):
            flow += self.G.edges[v,u,i]['f']
        return flow

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
    def mfd_algorithm(self, safe_paths=[], time_budget=float('inf')):
        # Note when changing the bounds, make sure not to break any of the following code
        lower_bound = max(self.max_outdeg, self.max_indeg, len(safe_paths))
        upper_bound = self.k_greedy if self.solved_greedy else self.G.number_of_edges() - self.G.number_of_nodes() + 2

        #if lower_bound == upper_bound:
        #   self.solved = True
        #   self.k = lower_bound
        #   
        #   # Make sure this is always correct whenever the bounds get changed
        #   self.paths = self.paths_greedy if self.solved_greedy else safe_paths

        #   return True

        #L, R = lower_bound-1, upper_bound
        ## invariant: L no, R yes
        #while L + 1 < R:
        #   i = (L + R) // 2
        #   if self.solve(safe_paths, i, time_budget):
        #       R = i
        #   else:
        #       L = i

        #if R == upper_bound:
        #   if not self.solve(safe_paths, R, time_budget):
        #       # Something went wrong
        #       return False

        #return True

        for i in range(lower_bound,upper_bound+1):
            self.model_status = self.solve(safe_paths, i, time_budget)
            if self.model_status == GRB.OPTIMAL or self.model_status == GRB.TIME_LIMIT:
                return True
        return False

    """
    Solves the ILP given by the basic model.
    """
    def solve(self, safe_paths, k, time_budget):
        model, _, _, _ = self.ilp_basic_model(safe_paths, k, time_budget)
        model.optimize()
        self.runtime += model.runtime

        if model.status == GRB.OPTIMAL:
            I = [(u, v, i, j) for (u, v, i) in self.G.edges(keys=True) for j in range(k)]
            self.solved = True
            self.k = k
            self.paths = [list() for _ in range(k)]
            self.weights = [0] * len(range(k))
            
            for i in range(k):
                self.weights[i] = round(model.getVarByName(f'w[{i}]').x)
            for (u, v, i, j) in I:
                if round(model.getVarByName(f'x[{u},{v},{i},{j}]').x) == 1:
                    self.paths[j].append((u, v, i))
            for j in range(k):
                self.paths[j] = sorted(self.paths[j])

        return model.status
    
    """
    Solves the ILP with a given set of path weights.
    """
    def solve_given_weights(self, path_weights, time_budget=float('inf')):
        model, _ = self.ilp_model_paths(path_weights, time_budget)
        model.optimize()

        if model.status == GRB.OPTIMAL:
            self.solved = True
            self.k = model.getObjective().getValue()
            self.runtime = model.runtime

            # TODO: Reconstruct paths from the model

            return model.getObjective().getValue()
        if model.status == GRB.TIME_LIMIT:
            return -2
        if model.status == GRB.INFEASIBLE:
            return -1
    

    """
    Build basic gurobi ILP model.
    """
    def ilp_basic_model(self, safe_paths, size, time_budget):
        # create index sets
        T = [(u, v, i, k) for (u, v, i) in self.G.edges(keys=True) for k in range(size)]
        SC = list(range(size))

        model = gp.Model('MFD')
        model.setParam('LogToConsole', 0)
        model.setParam('Threads', threads)
        model.setParam('Timelimit', time_budget)

        # Create variables
        x = model.addVars(T, vtype=GRB.BINARY, name='x') # path indicators
        w = model.addVars(SC, vtype=GRB.INTEGER, name='w', lb=0) # weights of paths
        z = model.addVars(T, vtype=GRB.CONTINUOUS, name='z', lb=0) # weight of path per edge

        # flow conservation
        for k in range(size):
            for v in self.G.nodes:
                if v in self.sources:
                    model.addConstr(sum(x[v, w, i, k] for _, w, i in self.G.out_edges(v, keys=True)) == 1)
                if v in self.sinks:
                    model.addConstr(sum(x[u, v, i, k] for u, _, i in self.G.in_edges(v, keys=True)) == 1)
                if v not in self.sources and v not in self.sinks:
                    model.addConstr(sum(x[v, w, i, k] for _, w, i in self.G.out_edges(v, keys=True)) - sum(x[u, v, i, k] for u, _, i in self.G.in_edges(v, keys=True)) == 0)

        # flow balance
        for (u, v, i, f) in self.G.edges(keys=True, data='f'):
            model.addConstr(f == sum(z[u, v, i, k] for k in range(size)))

        # linearization
        for (u, v, i) in self.G.edges(keys=True):
            for k in range(size):
                model.addConstr(z[u, v, i, k] <= self.max_flow_value * x[u, v, i, k])
                model.addConstr(w[k] - (1 - x[u, v, i, k]) * self.max_flow_value <= z[u, v, i, k])
                model.addConstr(z[u, v, i, k] <= w[k])

        # pre-defined constraints
        for (k, (idx, l, r)) in enumerate(safe_paths):
            for u, v, i in self.paths_greedy[idx][l:r]:
                model.addConstr(x[u, v, i, k] == 1)

        # remove some symmetry
        for k in range(len(safe_paths), size-1):
            model.addConstr(w[k] <= w[k+1])

        return model, x, w, z

    """
    Build gurobi ILP model using only a given set of path weights.
    This gives an optimal solutions if the weights appear in the optimal
    solution.
    """
    def ilp_model_paths(self, weight_set, time_budget):
        size = len(weight_set)

        # create index set
        T = [(u, v, i, k) for (u, v, i) in self.G.edges(keys=True) for k in range(size)]

        model = gp.Model('MFD_weights')
        model.setParam('LogToConsole', 0)
        model.setParam('Threads', threads)
        model.setParam('Timelimit', time_budget)

        # Create variables
        x = model.addVars(T, vtype=GRB.INTEGER, name='x', lb=0) # path indicators

        # Objective function: minimise number of paths
        model.setObjective(sum(x[u, v, i, k] for k in range(size) for s in self.sources for (u, v, i) in self.G.out_edges(s, keys=True)), GRB.MINIMIZE)
        
        # flow balance
        for (u, v, i, f) in self.G.edges(keys=True, data='f'):
            model.addConstr(f == sum(weight_set[k] * x[u, v, i, k] for k in range(size)))

        # flow conservation
        for k in range(size):
            for v in self.G.nodes:
                if v not in self.sources and v not in self.sinks:
                    model.addConstr(sum(x[v, w, i, k] for _, w, i in self.G.out_edges(v, keys=True)) - sum(x[u, v, i, k] for u, _, i in self.G.in_edges(v, keys=True)) == 0)

        return model, x

    """
    Polynomial time flow decomposition.
    If the flow has already been decomposed heuristically,
    the function does not do anything and returns False.
    """
    def decompose_flow_heuristic(self):

        if self.solved_greedy:
            return False
        
        self.weights_greedy = []
        self.paths_greedy = []

        remaining_flow = self.find_total_flow()
        edges = list(self.G.edges(keys=True))

        for e in edges:
            self.G.edges[e]['remaining_flow'] = self.G.edges[e]['f']

        def extract_highest_weight_path():
            highest_weight_of_any_path = {v: remaining_flow for v in self.sinks} 

            for v in list(reversed(list(nx.topological_sort(self.G)))):
                if v not in self.sinks:
                    highest_weight_of_any_path[v] = max([min(self.G.edges[v,u,k]['remaining_flow'], highest_weight_of_any_path[u]) for (v,u,k) in self.G.out_edges(v, keys=True)])

            largest_flow = max([highest_weight_of_any_path[s] for s in self.sources])

            path = []
            for s in self.sources:
                first_edges = [e for e in self.G.out_edges(s, keys=True) if highest_weight_of_any_path[e[1]] >= largest_flow and self.G.edges[e]['remaining_flow'] >= largest_flow]
                if first_edges:
                    path.append(first_edges[0])
                    while [e for e in self.G.out_edges(path[-1][1], keys=True) if highest_weight_of_any_path[e[1]] >= largest_flow and self.G.edges[e]['remaining_flow'] >= largest_flow]:
                        path.append([e for e in self.G.out_edges(path[-1][1], keys=True) if highest_weight_of_any_path[e[1]] >= largest_flow and self.G.edges[e]['remaining_flow'] >= largest_flow][0])
                    break
            
            assert len(path) > 0
            return path, largest_flow

        while remaining_flow > 0:

            path, weight = extract_highest_weight_path()
            self.weights_greedy.append(weight)
            self.paths_greedy.append(path)

            for e in path:
                self.G.edges[e]['remaining_flow'] -= weight
            remaining_flow -= weight

        assert remaining_flow == 0

        self.solved_greedy = True
        self.k_greedy = len(self.paths_greedy)
        return True
    
    """
    Find all safe paths of the heuristic poly-time flow decomposition.
    Can only be called after decompose_flow_heuristic() (returns False
    otherwise). Returns False if safe paths have already been calculated.
    """
    def find_safe_paths(self):

        if not self.solved_greedy:
            return False
        if len(self.safe_paths) > 0:
            return False
        assert len(self.paths_greedy) > 0

        for path in self.paths_greedy:
            if len(path) <= 1:
                self.safe_paths.append([(0, len(path))])
                continue

            self.safe_paths.append([])
            L, R = 0, 2
            excess_flow = self.G.edges[path[0]]['f'] + self.G.edges[path[1]]['f'] - self.out_flow(path[1][0])

            while L < len(path) - 1:

                while excess_flow > 0:
                    if R - L >= 2:
                        # add pair (L, R)
                        while self.safe_paths[-1] and (self.safe_paths[-1][-1][0] >= L and self.safe_paths[-1][-1][1] <= R):
                            self.safe_paths[-1].pop()
                        self.safe_paths[-1].append((L, R))
                    if R == len(path):
                        break
                    R += 1
                    excess_flow += self.G.edges[path[R-1]]['f'] - self.out_flow(path[R-1][0])

                assert R <= len(path)
                assert L < len(path)

                if R == len(path) and excess_flow > 0:
                    break

                while excess_flow <= 0:
                    if L >= R-1:
                        break
                    L += 1
                    excess_flow -= self.G.edges[path[L-1]]['f'] - self.out_flow(path[L][0])

        return True

# Calculates a maximum weight edge antichain, giving every edge
# the weight = longest safe path crossing this edge.
def edge_mwa_safe_paths(mfd, longest_safe_path_of_edge):
    graph = mfd.G
    greedy_paths = mfd.paths_greedy

    N = graph.number_of_nodes()
    M = graph.number_of_edges()
    orig_N = N
    orig_M = M

    # Normalise node indices to [0..N)
    oldnode_to_newnode = dict()
    newnode_to_oldnode = dict()
    for (x, v) in enumerate(graph.nodes()):
        oldnode_to_newnode[v] = x
        newnode_to_oldnode[x] = v

    # add node between every edge
    E = {oldnode_to_newnode[u]: list() for u in graph.nodes}
    newnode_to_edge = dict()
    edge_to_newnode = dict()
    for (u,v,i) in graph.edges:
        u = oldnode_to_newnode[u]
        v = oldnode_to_newnode[v]

        x = N
        E[u].append(x)
        E[x] = [v]
        newnode_to_edge[x] = (u,v,i)
        edge_to_newnode[u,v,i] = x
        N += 1
        M += 1
    assert M == 2*orig_M
    assert N == orig_M + orig_N

    # Call C++ code to calculate maximum weight antichain
    # Program mwa calls naive_minflow_reduction -> naive_minflow_solve -> maxantichain_from_minflow
    mwa_input = io.StringIO()
    mwa_input.write("{} {}\n".format(N, M))
    for u in E:
        for v in E[u]:
            mwa_input.write("{} {}\n".format(u, v))

    for v in graph.nodes():
        mwa_input.write("{} {}\n".format(oldnode_to_newnode[v], 0))
    for (u,v,i) in longest_safe_path_of_edge:
        x = edge_to_newnode[oldnode_to_newnode[u],oldnode_to_newnode[v],i]
        mwa_input.write("{} {}\n".format(x, longest_safe_path_of_edge[u,v,i][0]))

    res = subprocess.run(["./mwa"], input=mwa_input.getvalue(), text=True, capture_output=True)
    if res.stdout != '':
        mwa = list(map(int, res.stdout.split(' ')))
        mwa = list(map(lambda x: x-1, mwa))
        for x in mwa:
            assert x >= orig_N
    else:
        mwa = []
    
    assert orig_M == 0 or len(mwa) > 0

    mwa_safe_paths = []
    for x in mwa:
        u,v,i = newnode_to_edge[x]
        # Find corresponding longest safe path going through the edge (u,v,i)
        u = newnode_to_oldnode[u]
        v = newnode_to_oldnode[v]
        ip, l, r = longest_safe_path_of_edge[u,v,i][1]
        assert ip != -1
        mwa_safe_paths.append((ip, l, r))

    return mwa_safe_paths

def read_input_graphs(graph_file):
    graphs = []

    with open(graph_file, 'r') as file:
        raw_graphs = file.read().split('#')[1:]
        for g in raw_graphs:
            graph = nx.MultiDiGraph()
            flow = dict()

            lines = g.split('\n')[1:]
            if not lines[-1]:
                lines = lines[:-1]
            graph.add_nodes_from(range(int(lines[0])))
            for e in lines[1:]:
                parts = e.split()
                key = graph.add_edge(int(parts[0]), int(parts[1]), f=float(parts[2]))
            graphs.append(graph)

    return graphs

def read_truth(truth_file):
    solutions = []

    with open(truth_file, 'r') as file:
        raw_solutions = file.read().split('#')[1:]
        for sol in raw_solutions:
            k = sol.split('\n')
            solutions.append(len(k)-2)

    return solutions

# Contract the graph: Remove vertices of in/out degree 1.
def y_to_v(ngraph):

    def get_out_tree(ngraph, v, processed, contracted):

        cont = {e: contracted[e] for e in contracted}

        # Get root of out_tree
        root = v
        while ngraph.in_degree(root) == 1:
            root = next(ngraph.predecessors(root))

        leaf_edges = list()
        edges = deque(ngraph.out_edges(root, keys=True))

        while edges:
            u, v, i = edges.popleft()
            processed[u] = True
            cont[u, v, i] = True
            if ngraph.in_degree(v) > 1 or ngraph.out_degree(v) == 0:
                leaf_edges.append((u, v, i))
            else:
                for e in ngraph.out_edges(v, keys=True):
                    edges.append(e)

        # Filter out uncompressible edges
        for e in [(u, v, i) for (u, v, i) in leaf_edges if u == root]:
            cont[e] = contracted[e]
        leaf_edges = [(u, v, i) for (u, v, i) in leaf_edges if u != root]

        return root, leaf_edges, processed, cont

    def get_in_tree(ngraph, v, processed, contracted):

        cont = {e: contracted[e] for e in contracted}

        # Get root of in_tree
        root = v
        while ngraph.out_degree(root) == 1:
            root = next(ngraph.successors(root))

        leaf_edges = list()
        edges = deque(ngraph.in_edges(root, keys=True))

        while edges:
            u, v, i = edges.popleft()
            processed[v] = True
            cont[u, v, i] = True
            if ngraph.out_degree(u) > 1 or ngraph.in_degree(u) == 0:
                leaf_edges.append((u, v, i))
            else:
                for e in ngraph.in_edges(u, keys=True):
                    edges.append(e)

        # Filter out uncompressible edges
        for e in [(u, v, i) for (u, v, i) in leaf_edges if v == root]:
            cont[e] = contracted[e]
        leaf_edges = [(u, v, i) for (u, v, i) in leaf_edges if v != root]

        return root, leaf_edges, processed, cont


    def get_out_trees(ngraph):

        out_trees = dict()
        contracted = {e: False for e in ngraph.edges(keys=True)}
        processed = {v: ngraph.in_degree(v) != 1 for v in ngraph.nodes}
        for v in ngraph.nodes:
            if not processed[v]:
                root, leaf_edges, processed, contracted = get_out_tree(ngraph, v, processed, contracted)
                if leaf_edges:
                    out_trees[root] = leaf_edges
        return out_trees, contracted


    def get_in_trees(ngraph):

        in_trees = dict()
        contracted = {e: False for e in ngraph.edges(keys=True)}
        processed = {v: ngraph.out_degree(v) != 1 for v in ngraph.nodes}
        for v in ngraph.nodes:
            if not processed[v]:
                root, leaf_edges, processed, contracted = get_in_tree(ngraph, v, processed, contracted)
                if leaf_edges:
                    in_trees[root] = leaf_edges
        return in_trees, contracted

    def get_expanded_path(path, graph, original_graph, out_contraction_graph):

        out_contraction_path = list()

        for u, v, i in path:
            root = v
            v, i = graph.edges[u, v, i]['succ']

            expanded_edge = list()
            while v != root:
                expanded_edge.append((u, v, i))
                u, v, i = list(out_contraction_graph.out_edges(v, keys=True))[0]

            expanded_edge.append((u, v, i))
            out_contraction_path += list(expanded_edge)

        original_path = list()

        for u, v, i in out_contraction_path:
            root = u
            u = out_contraction_graph.edges[u, v, i]['pred']
            expanded_edge = list()
            while u != root:
                expanded_edge.append((u, v, i))
                u, v, i = list(original_graph.in_edges(u, keys=True))[0]
            expanded_edge.append((u, v, i))
            original_path += list(reversed(expanded_edge))

        return original_path

    out_trees, contracted = get_out_trees(ngraph)

    mngraph_out_contraction = nx.MultiDiGraph()
    for u, v, i in ngraph.edges(keys=True):
        if not contracted[u, v, i]:
            mngraph_out_contraction.add_edge(u, v, f=ngraph.edges[u, v, i]['f'], pred=u)

    for root, leaf_edges in out_trees.items():
        for u, v, i in leaf_edges:
            mngraph_out_contraction.add_edge(root, v, f=ngraph.edges[u, v, i]['f'], pred=u)

    in_trees, contracted = get_in_trees(mngraph_out_contraction)

    mngraph_in_contraction = nx.MultiDiGraph()
    for u, v, i in mngraph_out_contraction.edges(keys=True):
        if not contracted[u, v, i]:
            mngraph_in_contraction.add_edge(u, v, f=mngraph_out_contraction.edges[u, v, i]['f'], succ=(v, i))

    for root, leaf_edges in in_trees.items():
        for u, v, i in leaf_edges:
            mngraph_in_contraction.add_edge(u, root, f=mngraph_out_contraction.edges[u, v, i]['f'], succ=(v, i))

    # Remove trivial_paths found as edges from source to sink in the contraction
    trivial_paths = list()
    edges = list(mngraph_in_contraction.edges(keys=True))
    for u, v, i in edges:
        if mngraph_in_contraction.in_degree(u) == 0 and mngraph_in_contraction.out_degree(v) == 0:
            trivial_paths.append(get_expanded_path([(u, v, i)], mngraph_in_contraction, ngraph, mngraph_out_contraction))
            mngraph_in_contraction.remove_edge(u, v, i)

    return mngraph_out_contraction, mngraph_in_contraction, trivial_paths

def results_to_file(file, num_paths, times):
    with open(file, 'w') as f:
        for i in range(len(num_paths)):
            f.write("{} {}\n".format(num_paths[i], times[i]))

trivial_occur = 0
# Main pipeline function
def pipeline(graph):
    global trivial_occur
    mngraph_out_contraction, mngraph_in_contraction, trivial_paths = y_to_v(graph)
    mfd = Mfd(mngraph_in_contraction, number_of_contracted_paths=len(trivial_paths))
    if mngraph_in_contraction.number_of_edges() == 0:
        print("Trivially decomposable to {} paths.".format(len(trivial_paths)))
        trivial_occur += 1
        return mfd

    assert mfd.decompose_flow_heuristic()
    assert mfd.find_safe_paths()

    # Calculate safe paths antichain
    longest_safe_path_of_edge = dict()
    for u,v,i in mfd.G.edges:
        longest_safe_path_of_edge[u,v,i] = (0, (-1, -1))
    for ip, path in enumerate(mfd.paths_greedy):
        for j in range(len(path)):
            u,v,i = path[j]
            if longest_safe_path_of_edge[u,v,i][0] < 1:
                longest_safe_path_of_edge[u,v,i] = (1, (ip, j, j+1))
        for iw, window in enumerate(mfd.safe_paths[ip]):
            for j in range(window[0], window[1]):
                u,v,i = path[j]
                if longest_safe_path_of_edge[u,v,i][0] < window[1] - window[0] + 1:
                    longest_safe_path_of_edge[u,v,i] = (window[1] - window[0] + 1, (ip, window[0], window[1]))  

    #safe_antichain = []
    safe_antichain = edge_mwa_safe_paths(mfd, longest_safe_path_of_edge)

    assert mfd.mfd_algorithm(safe_paths=safe_antichain, time_budget=2*60)

    if mfd.model_status == GRB.OPTIMAL:
        print("Calculated minimum flow decomposition of size {}.".format(mfd.k + len(trivial_paths)))
        print("Runtime:", mfd.runtime)
        return mfd

    assert mfd.model_status == GRB.TIME_LIMIT
    print("ILP time limit.")
    return mfd

# Approximation pipeline
def approx_pipeline(graph):
    mngraph_out_contraction, mngraph_in_contraction, trivial_paths = y_to_v(graph)
    if mngraph_in_contraction.number_of_edges() == 0:
        print("Trivially decomposable to {} paths.".format(len(trivial_paths)))
        return len(trivial_paths)

    mfd = Mfd(mngraph_in_contraction)

    path_weights = []
    W = 1
    while W <= mfd.max_flow_value:
        path_weights.append(W)
        W *= 4

    return mfd.solve_given_weights(path_weights, time_budget=2*60)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='''
        Computes Minimum Flow Decomposition.
        This script uses the Gurobi ILP solver.
        ''',
        formatter_class=argparse.RawTextHelpFormatter
    )
    #parser.add_argument('-stats', '--output-stats', action='store_true', help='Output stats to file <output>.stats')
    #parser.add_argument('-wt', '--weighttype', type=str, default='int+',
    #                   help='Type of path weights (default int+):\n   int+ (positive non-zero ints), \n   float+ (positive non-zero floats).')
    #parser.add_argument('-t', '--threads', type=int, default=0,
    #                   help='Number of threads to use for the Gurobi solver; use 0 for all threads (default 0).')
    #parser.add_argument('-seqt', '--sequential-threshold', type=int, default=0,
    #                   help='A repeated exponential search is performed to find the minimum flow decomposition, this parameter specifies the universe size at which a sequencial search is performed instead; use 0 to only perform sequential search (default 0).')

    #parser.add_argument('-ilptb', '--ilp-time-budget', type=float, help='Maximum time (in seconds) that the ilp solver is allowed to take when computing safe paths for one graph')

    #parser.add_argument('-uef', '--use-excess-flow', action='store_true', help='Use excess flow of a path to save ILP calls')
    #parser.add_argument('-ugtd', '--use-group-top-down', action='store_true',
    #                   help='Use top down group testing')
    #parser.add_argument('-ugbu', '--use-group-bottom-up', action='store_true',
    #                   help='Use bottom up group testing')
    #parser.add_argument('-uy2v', '--use-y-to-v', action='store_true', help='Use Y to V contraction of the input graphs')

    #parser.add_argument('-s', '--strategy', type=str, help='Strategy for extension and reduction of two-finger algorithm {scan, bin_search, exp_search, rep_exp_search}')
    #parser.add_argument('-es', '--extension-strategy', type=str, help='Strategy for extension of two-finger algorithm {scan, bin_search, exp_search, rep_exp_search}')
    #parser.add_argument('-rs', '--reduction-strategy', type=str, help='Strategy for reduction of two-finger algorithm {scan, bin_search, exp_search, rep_exp_search}')
    #parser.add_argument('-ess', '--extension-strategy-small', type=str, help='Strategy for extension of two-finger algorithm {scan, bin_search, exp_search, rep_exp_search} when the search space is small (specified by threshold)')
    #parser.add_argument('-esl', '--extension-strategy-large', type=str, help='Strategy for extension of two-finger algorithm {scan, bin_search, exp_search, rep_exp_search} when the search space is large (specified by threshold)')
    #parser.add_argument('-rss', '--reduction-strategy-small', type=str, help='Strategy for reduction of two-finger algorithm {scan, bin_search, exp_search, rep_exp_search} when the search space is small (specified by threshold)')
    #parser.add_argument('-rsl', '--reduction-strategy-large', type=str, help='Strategy for reduction of two-finger algorithm {scan, bin_search, exp_search, rep_exp_search} when the search space is large (specified by threshold)')
    #parser.add_argument('-st', '--strategy-threshold', type=int, help='Search space threshold to switch from --<>-strategy-small to --<>-strategy-large')
    #parser.add_argument('-est', '--extension-strategy-threshold', type=int, help='Search space threshold to switch from --extension-strategy-small to --extension-strategy-large')
    #parser.add_argument('-rst', '--reduction-strategy-threshold', type=int, help='Search space threshold to switch from --reduction-strategy-small to --reduction-strategy-large')
    #parser.add_argument('-safe', '--safepaths', type=str, help='Safe File filename')


    requiredNamed = parser.add_argument_group('required arguments')
    requiredNamed.add_argument('-i', '--input', type=str, help='Input filename', required=True)
    requiredNamed.add_argument('-o', '--output', type=str, help='Output filename', required=True)

    args = parser.parse_args()

    threads = os.cpu_count()
    print(f'INFO: Using {threads} threads for the Gurobi solver')

    #truths = read_truth(args.input.split('.')[0] + '.truth')
    graphs = read_input_graphs(args.input)
    timelimit = []

    number_of_paths = []
    runtimes = []
    for i, graph in enumerate(graphs):
        print("Running graph {}:".format(i))
        mfd = pipeline(graph)
        num_paths = 0 if mfd.model_status == GRB.TIME_LIMIT else mfd.k + mfd.number_of_contracted_paths
        number_of_paths.append(num_paths)
        runtimes.append(mfd.runtime)

    output_file = ''.join([args.output,'.time'])
    results_to_file(output_file, number_of_paths, runtimes)
    print("Timelimited:", timelimit)
    print("Trivially decomposed:", trivial_occur)

    #for i, graph in enumerate(graphs):
    #   print("Running graph {}:".format(i))
    #   result = approx_pipeline(graph)
    #   assert result != -1
    #   if result < -1:
    #       timelimit.append(i)
    #       print("Timelimit ILP")
    #   elif result != truths[i]:
    #       not_truth.append(i)
    #   if result >= 0:
    #       print("Decomposed using {} paths.".format(result))

