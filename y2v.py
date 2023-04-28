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


class TimeoutILP(Exception):
    pass


def get_edge(raw_edge):

    parts = raw_edge.split()
    return int(parts[0]), int(parts[1]), float(parts[2])


def get_graph(raw_graph):

    graph = {
        'n': 0,
        'edges': list()
    }

    try:
        lines = raw_graph.split('\n')[1:]
        if not lines[-1]:
            lines = lines[:-1]
        graph['n'], graph['edges'] = int(lines[0]), [get_edge(raw_e) for raw_e in lines[1:]]

    finally:
        return graph


def read_input_graphs(graph_file):

    graphs_raw = open(graph_file, 'r').read().split('#')[1:]
    return [get_graph(raw_g) for raw_g in graphs_raw]

def read_input(graph_file):

    return read_input_graphs(graph_file)

def get_solution(model, data, size):

    data['weights'], data['solution'] = list(), list()

    if model.status == GRB.OPTIMAL:
        graph = data['graph']
        T = [(u, v, i, k) for (u, v, i) in graph.edges(keys=True) for k in range(size)]

        w_sol = [0] * len(range(size))
        paths = [list() for _ in range(size)]
        for k in range(size):
            w_sol[k] = round(model.getVarByName(f'w[{k}]').x)
        for (u, v, i, k) in T:
            if round(model.getVarByName(f'x[{u},{v},{i},{k}]').x) == 1:
                paths[k].append((u, v, i))
        for k in range(len(paths)):
            paths[k] = sorted(paths[k])

        data['weights'], data['solution'] = w_sol, paths

    return data

def get_flow(e, mfd):

    return mfd['graph'].edges[e]['flow']

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


def get_y_to_v_contraction(ngraph):

    out_trees, contracted = get_out_trees(ngraph)

    mngraph_out_contraction = nx.MultiDiGraph()
    for u, v, i in ngraph.edges(keys=True):
        if not contracted[u, v, i]:
            mngraph_out_contraction.add_edge(u, v, flow=ngraph.edges[u, v, i]['flow'], pred=u)

    for root, leaf_edges in out_trees.items():
        for u, v, i in leaf_edges:
            mngraph_out_contraction.add_edge(root, v, flow=ngraph.edges[u, v, i]['flow'], pred=u)

    in_trees, contracted = get_in_trees(mngraph_out_contraction)

    mngraph_in_contraction = nx.MultiDiGraph()
    for u, v, i in mngraph_out_contraction.edges(keys=True):
        if not contracted[u, v, i]:
            mngraph_in_contraction.add_edge(u, v, flow=mngraph_out_contraction.edges[u, v, i]['flow'], succ=(v, i))

    for root, leaf_edges in in_trees.items():
        for u, v, i in leaf_edges:
            mngraph_in_contraction.add_edge(u, root, flow=mngraph_out_contraction.edges[u, v, i]['flow'], succ=(v, i))

    # Remove trivial_paths found as edges from source to sink in the contraction
    trivial_paths = list()
    edges = list(mngraph_in_contraction.edges(keys=True))
    for u, v, i in edges:
        if mngraph_in_contraction.in_degree(u) == 0 and mngraph_in_contraction.out_degree(v) == 0:
            trivial_paths.append(get_expanded_path([(u, v, i)], mngraph_in_contraction, ngraph, mngraph_out_contraction))
            mngraph_in_contraction.remove_edge(u, v, i)

    return mngraph_out_contraction, mngraph_in_contraction, trivial_paths


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


def compute_y2v(graph):

    # creation of NetworkX Graph
    ngraph = nx.MultiDiGraph()
    ngraph.add_weighted_edges_from(graph['edges'], weight='flow')
    
    original = ngraph
    out_contraction, ngraph, trivial_paths = get_y_to_v_contraction(ngraph)
    mapping = (original, out_contraction)

    # definition of data
    return {
        'graph': ngraph,
        'mapping': mapping, 
        'trivial_paths': trivial_paths
    }

def outputY2VGraphs(graph,output_file,minimum):
    edges = graph.edges(keys=True,data='flow')
    if (len(graph.nodes())) >= minimum:
        print(len(graph.nodes()),file=output_file)
        for (u,v,i,f) in edges:
            print(u,v,i,f,file=output_file)

    

def y2vselection(graphs, minimum, output_file):

    output = open(output_file, 'w+')
    
    for g, graph in enumerate(graphs):
        
        if not graph['edges']:
            continue

        mfd = compute_y2v(graph)

        if len(mfd['graph'].nodes()) >= minimum:
            output.write(f'# graph {g}\n')
            outputY2VGraphs(mfd['graph'],output,minimum)
    
    output.close()


def build_base_ilp_model(data, paths, size, relaxed = False):


    graph = data['graph']
    max_flow_value = data['max_flow_value']
    sources = data['sources']
    sinks = data['sinks']

    # create extra sets
    T = [(u, v, i, k) for (u, v, i) in graph.edges(keys=True) for k in range(size)]
    SC = list(range(size))

    # Create a new model
    model = gp.Model('MFD')
    model.setParam('LogToConsole', 0)
    model.setParam('Threads', threads)


    # Create variables
    if relaxed:
        x = model.addVars(T, vtype=GRB.INTEGER, name='x')
    else:
        x = model.addVars(T, vtype=GRB.BINARY, name='x')
    w = model.addVars(SC, vtype=GRB.INTEGER, name='w', lb=0)
    z = model.addVars(T, vtype=GRB.CONTINUOUS, name='z', lb=0)

    # flow conservation
    for k in range(size):
        for v in graph.nodes:
            if relaxed:
                if v in sources:
                    model.addConstr(sum(x[v, w, i, k] for _, w, i in graph.out_edges(v, keys=True)) <= 1)
            else:
                if v in sources:
                    model.addConstr(sum(x[v, w, i, k] for _, w, i in graph.out_edges(v, keys=True)) == 1)
                if v in sinks:
                    model.addConstr(sum(x[u, v, i, k] for u, _, i in graph.in_edges(v, keys=True)) == 1)

            if v not in sources and v not in sinks:
                model.addConstr(sum(x[v, w, i, k] for _, w, i in graph.out_edges(v, keys=True)) - sum(x[u, v, i, k] for u, _, i in graph.in_edges(v, keys=True)) == 0)


    # flow balance
    for (u, v, i, f) in graph.edges(keys=True, data='flow'):
        model.addConstr(f == sum(z[u, v, i, k] for k in range(size)))

    # linearization
    for (u, v, i) in graph.edges(keys=True):
        for k in range(size):
            model.addConstr(z[u, v, i, k] <= max_flow_value * x[u, v, i, k])
            model.addConstr(w[k] - (1 - x[u, v, i, k]) * max_flow_value <= z[u, v, i, k])
            model.addConstr(z[u, v, i, k] <= w[k])

    # pre-defined constraints
    if paths != 0:
        k = 0
        for path in paths:
            for (u,v,i) in path:
                model.addConstr(x[u,v,i,k] == 1)
                for k_ in range(size):
                    if k_ == k:
                        continue
                    model.addConstr(x[u,v,i,k_] == 0)
            k = k + 1
    
    # objective function
    if relaxed:
        model.setObjective(sum(x[v, w, i, k] for v, w, i in graph.out_edges(keys=True) if v in sources for k in range(size)), GRB.MINIMIZE)

    return model, x, w, z


def get_solution(model, data, size):

    data['weights'], data['solution'] = list(), list()

    if model.status == GRB.OPTIMAL:
        graph = data['graph']
        T = [(u, v, i, k) for (u, v, i) in graph.edges(keys=True) for k in range(size)]

        w_sol = [0] * len(range(size))
        paths = [list() for _ in range(size)]
        for k in range(size):
            w_sol[k] = round(model.getVarByName(f'w[{k}]').x)
        for (u, v, i, k) in T:
            if round(model.getVarByName(f'x[{u},{v},{i},{k}]').x) == 1:
                paths[k].append((u, v, i))
            if model.getVarByName(f'x[{u},{v},{i},{k}]').x > 1:
                paths = []
                wsol = []
                data['weights'], data['solution'] = w_sol, paths
                return data
        for k in range(len(paths)):
            paths[k] = sorted(paths[k])

        data['weights'], data['solution'] = w_sol, paths

    return data
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='''
        ''',
        formatter_class=argparse.RawTextHelpFormatter
    )

    requiredNamed = parser.add_argument_group('required arguments')
    requiredNamed.add_argument('-i', '--input', type=str, help='Input filename', required=True)
    requiredNamed.add_argument('-o', '--output', type=str, help='Output filename', required=True)
    requiredNamed.add_argument('-m', '--minimum', type=int, help='Minimum graph size after contraction', required=True)

    args = parser.parse_args()

    
    y2vselection(read_input(args.input),args.minimum,args.output)
