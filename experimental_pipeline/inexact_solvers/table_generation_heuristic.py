import os
import sys
import argparse
import networkx as nx
from collections import deque
from bisect import bisect
from copy import deepcopy

def calculateCriteria(graphs,criteria,optimal,K,Runtime,Paths):
	
	for i, graph in enumerate(graphs):
		o = 0
		for j, g in enumerate(graph):

			if g[2].split('.')[1] != optimal[o][1].split('.')[1]:
				continue
			runtime = g[1]
			hk = g[0]
			k = optimal[o][0]
			print(g)
			print(optimal[o])
			assert hk >= k
			o += 1

			if k <= criteria[0]:
				K[i][0] += 1
				Runtime[i][0] += runtime
				Paths[i][0] += hk - k

			if k > criteria[0] and k <= criteria[1]:
				K[i][1] += 1
				Runtime[i][1] += runtime
				Paths[i][1] += hk - k

			if k > criteria[1] and k <= criteria[2]:
				K[i][2] += 1
				Runtime[i][2] += runtime
				Paths[i][2] += hk - k

			if k > criteria[2] and k <= criteria[3]:
				K[i][3] += 1
				Runtime[i][3] += runtime
				Paths[i][3] += hk - k

			if k > criteria[3]:
				K[i][4] += 1
				Runtime[i][4] += runtime
				Paths[i][4] += hk - k

	return K,Runtime,Paths

"""
Print LaTeX table.
criteria: List containing upper bounds of solution ranges
K: list containing the number of solved instances per solver per criteria
runtime: total runtime of solver per criteria range
outputfile: outputfile
"""
def outputTable(criteria,K,runtime,paths,outputfile):

	with open(outputfile,'w') as output:
		avg_runtime = [[0 for i in range(len(criteria)+1)] for j in range(len(runtime))]
		avg_paths = [[0 for i in range(len(criteria)+1)] for j in range(len(runtime))]

		pcriteria = ["1"] + [str(c) for c in criteria] + ["max"]

		for k in range(len(pcriteria)-1):
			print("&", "-".join([pcriteria[k], pcriteria[k+1]]), end=' ', file=output)
			for j in range(len(runtime)):
				if K[j][k] != 0:
					avg_runtime[j][k] = runtime[j][k]/K[j][k]
					avg_paths[j][k] = float(paths[j][k])/float(K[j][k])
				print("& ", "{:.2f}".format(avg_runtime[j][k]), "& ", "{:.2f}".format(runtime[j][k]), "& ", "{} ({:.2f})".format(paths[j][k], avg_paths[j][k]), end=' ', file=output)
			print("\\\\", file=output)
		

def  generate_simple_tables(graphs,output,optimal):

	criteria = [5,10,15,20]
	totalGraphsPerCriteria = [[0 for i in range(len(criteria)+1)] for _ in graphs]
	totalRuntimePerCreteria= [[0 for i in range(len(criteria)+1)] for _ in graphs]
	totalPathsPerCriteria = [[0 for i in range(len(criteria)+1)] for _ in graphs]
	totalGraphsPerCriteria,totalRuntimePerCreteria,totalPathsPerCriteria = calculateCriteria(graphs,criteria,optimal,totalGraphsPerCriteria,totalRuntimePerCreteria,totalPathsPerCriteria)

	outputTable(criteria,totalGraphsPerCriteria,totalRuntimePerCreteria,totalPathsPerCriteria,output)

def read_file_input(inputs):
	
	files = inputs.split(",")
	ret = []
	for inputfile in files:
		graphs_raw = open(inputfile, 'r').read().split('\n')[0:-1]
		ret.append([getSolutionData(raw_g) for raw_g in graphs_raw])
	return ret

def read_opt_sol(inp):
	
	ret = []
	raw = open(inp, 'r').read().split('\n')[0:-1]
	for raw_g in raw:
		#ret.append((getOptimalK(raw_g), getOptimalK(raw_g)))
		ret.append((getOptimalK(raw_g), getFile(raw_g)))
	return ret

def getSolutionData(graph):
   return [getOptimalK(graph),getRuntime(graph),getFile(graph)]

def getOptimalK(graph):
	return round(float(graph.split(" ")[0]))

def getRuntime(graph):
	return round(float(graph.split(" ")[1]),2)

def getFile(graph):
	return graph.split(" ")[2]

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
	requiredNamed.add_argument('-u', '--optimal',type=str, help='Input filename with optimal solution', required=True)
	args = parser.parse_args()

	generate_simple_tables(read_file_input(args.input),args.output,read_opt_sol(args.optimal))

# add table evaluation procedure
