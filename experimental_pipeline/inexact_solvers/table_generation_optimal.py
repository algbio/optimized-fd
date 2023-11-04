import os
import sys
import argparse
import networkx as nx
from collections import deque
from bisect import bisect
from copy import deepcopy

def calculateCriteria(graphs,criteria,K,Runtime):
	
	for i, graph in enumerate(graphs):
		for g in graph:

			k = g[0]
			runtime = g[1]

			if k <= criteria[0]:
				K[i][0] += 1
				Runtime[i][0] += runtime

			if k > criteria[0] and k <= criteria[1]:
				K[i][1] += 1
				Runtime[i][1] += runtime

			if k > criteria[1] and k <= criteria[2]:
				K[i][2] += 1
				Runtime[i][2] += runtime

			if k > criteria[2] and k <= criteria[3]:
				K[i][3] += 1
				Runtime[i][3] += runtime

			if k > criteria[3]:
				K[i][4] += 1
				Runtime[i][4] += runtime

	return K,Runtime

"""
Print LaTeX table.
criteria: List containing upper bounds of solution ranges
K: list containing the number of solved instances per solver per criteria
runtime: total runtime of solver per criteria range
outputfile: outputfile
"""
def outputTable(criteria,K,runtime,outputfile):

	with open(outputfile,'w') as output:
		avg_runtime = [[0 for i in range(len(criteria)+1)] for j in range(len(runtime))]

		pcriteria = ["1"] + [str(c) for c in criteria] + ["max"]

		for k in range(len(pcriteria)-1):
			print("&", "-".join([pcriteria[k], pcriteria[k+1]]), end=' ', file=output)
			for j in range(len(runtime)):
				if K[j][k] != 0:
					avg_runtime[j][k] = runtime[j][k]/K[j][k]
				print("& ", "{:.2f}".format(avg_runtime[j][k]), "& ", "{:.2f}".format(runtime[j][k]), "& ", str(K[j][k]), end=' ', file=output)
			print("\\\\", file=output)
		

def  generate_simple_tables(graphs,output):

	criteria = [5,10,15,20]
	totalGraphsPerCriteria = [[0 for i in range(len(criteria)+1)] for _ in graphs]
	totalRuntimePerCreteria= [[0 for i in range(len(criteria)+1)] for _ in graphs]
	totalGraphsPerCriteria,totalRuntimePerCreteria = calculateCriteria(graphs,criteria,totalGraphsPerCriteria,totalRuntimePerCreteria)

	outputTable(criteria,totalGraphsPerCriteria,totalRuntimePerCreteria,output)

def read_file_input(inputs):
	
	files = inputs.split(",")
	ret = []
	for inputfile in files:
		graphs_raw = open(inputfile, 'r').read().split('\n')[0:-1]
		ret.append([getSolutionData(raw_g) for raw_g in graphs_raw])
	return ret

def getSolutionData(graph):
   return [getOptimalK(graph),getRuntime(graph)]

def getOptimalK(graph):
	return round(float(graph.split(" ")[0]))

def getRuntime(graph):
	return round(float(graph.split(" ")[1]),2)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(
		description='''
		Compare the flow decomposition output from two different formulations.
		''',
		formatter_class=argparse.RawTextHelpFormatter
	)
	
 
	requiredNamed = parser.add_argument_group('required arguments')
	requiredNamed.add_argument('-i', '--input', type=str, help='Input filenames, separated by ","', required=True)
	requiredNamed.add_argument('-o', '--output',type=str, help='Output filename', required=True)
	args = parser.parse_args()

	generate_simple_tables(read_file_input(args.input),args.output)

# add table evaluation procedure
