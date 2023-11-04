#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse

def get_edge(raw_edge):

	parts = raw_edge.split()
	return int(parts[0]), int(parts[1]), float(parts[2])

def get_subpath(raw_path):
	
	path = raw_path.split()
	return path


def get_graph(raw_graph):

	graph = {
		'n': 0,
		'edges': list(),
		'subpaths': list()
	}

	try:
		lines = raw_graph.split('\n')[1:]
		if not lines[-1]:
			lines = lines[:-1]
		SP = 0
		while lines[SP] != "subpaths":
			SP += 1
		graph['n'], graph['edges'] = int(lines[0]), [get_edge(raw_e) for raw_e in lines[1:SP]]
		graph['subpaths'] = [get_subpath(raw_p) for raw_p in lines[SP+1:]]

	finally:
		return graph


def read_input_graphs(graph_file):

	graphs_raw = open(graph_file, 'r').read().split('#')[1:]
	return [get_graph(raw_g) for raw_g in graphs_raw]


def read_input(graph_file):

	return read_input_graphs(graph_file)


def output_graph(graph,output):

	output.write(str(graph['n']))
	output.write("\n")
	for edges in graph['edges']:
		output.write(" ".join([str(edges[0]),str(edges[1]),str(edges[2])]))
		output.write("\n")
	output.write("subpaths\n")
	for path in graph['subpaths']:
		output.write(" ".join(path))
		output.write("\n")

	return 0


def separateGraphs(graphs,inputName):

	#output_file = "".join([inputName.split(".")[0],"_heavy.graph"])

	#output = open(output_file, 'w+')

	for g, graph in enumerate(graphs):
		output_file = inputName + "_split/" + "".join([str(g),"_.graph"])
		output = open(output_file, 'w+')
		output.write(f'# graph number = {g} name = {g}\n')
		output_graph(graph,output)
		output.close()
	
	return 0

if __name__ == '__main__':

	parser = argparse.ArgumentParser(
		description='''
		Computes maximal safe paths for Minimum Flow Decomposition.
		This script uses the Gurobi ILP solver.
		''',
		formatter_class=argparse.RawTextHelpFormatter
	)

	requiredNamed = parser.add_argument_group('required arguments')
	requiredNamed.add_argument('-i', '--input', type=str, help='Input filename', required=True)

	args = parser.parse_args()

	separateGraphs(read_input(args.input),args.input)
