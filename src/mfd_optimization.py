#!/usr/bin/env python
# coding: utf-8

import ast
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
	def __init__(self, graph, number_of_contracted_paths=0, heuristic_paths=[], heuristic_weights=[]):
		# Graph data
		self.G = graph
		self.max_flow_value = self.find_max_flow_value()
		self.min_flow_value = self.find_min_flow_value()
		self.max_lower_flow_value = self.find_max_lower_flow_value()
		self.sources = self.find_sources()
		self.sinks = self.find_sinks()
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
		self.opt_is_greedy = False

		# Heuristic solution
		self.solved_heuristic = len(heuristic_paths) > 0
		self.heuristic_k = len(heuristic_paths)
		self.heuristic_paths = heuristic_paths
		self.heuristic_weights = heuristic_weights
		assert len(heuristic_paths) == len(heuristic_weights)
		self.safe_paths = []
		self.safe_paths_amount = 0

	"""
	Returns the maximum flow value.
	"""
	def find_max_flow_value(self):
		flow = 0
		for e in self.G.edges(keys=True):
			flow = max(flow, self.G.edges[e]['f'][1])
		return flow
	
	"""
	Returns the minimum flow value.
	Must be run after find_max_flow_value().
	"""
	def find_min_flow_value(self):
		flow = self.max_flow_value
		for e in self.G.edges(keys=True):
			flow = min(flow, self.G.edges[e]['f'][0])
		return flow

	"""
	Returns the maximum lower bound flow value.
	In case on exact flows, this is equal to
	the find_max_flow_value().
	"""
	def find_max_lower_flow_value(self):
		flow = 0
		for e in self.G.edges(keys=True):
			flow = max(flow, self.G.edges[e]['f'][0])
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
	Returns the sum of flow going out of a node v, using
	the lower bound for bound = 0 and the upper bound
	for bound = 1 in case of an inexact flow.
	"""
	def out_flow(self, v, bound=0):
		flow = 0
		for v,u,i in self.G.out_edges(v, keys=True):
			flow += self.G.edges[v,u,i]['f'][bound]
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
	def mfd_algorithm(self, safe_paths=[], path_constraints=[], time_budget=float('inf')):
		if self.min_flow_value == 0:
			lower_bound = max(1, len(safe_paths))
		else:
			lower_bound = max(self.max_outdeg, self.max_indeg, len(safe_paths))
		upper_bound = self.heuristic_k if self.solved_heuristic else self.G.number_of_edges() - self.G.number_of_nodes() + 2 + len(path_constraints)

		if lower_bound == upper_bound and self.solved_heuristic:
			self.opt_is_greedy = True
			self.k = self.heuristic_k
			self.weights = self.heuristic_weights
			self.paths = self.heuristic_paths
			return True
			

		for i in range(lower_bound,upper_bound+1):
			self.model_status = self.solve(safe_paths, path_constraints, i, time_budget)
			if self.model_status == GRB.OPTIMAL or self.model_status == GRB.TIME_LIMIT:
				return True
		return False

	"""
	Solves the ILP given by the basic model.
	"""
	def solve(self, safe_paths, path_constraints, k, time_budget):
		model, _, _, _ = self.ilp_basic_model(safe_paths, path_constraints, k, time_budget)
		model.optimize()
		self.runtime += model.runtime

		if model.status == GRB.OPTIMAL:
			I = [(u, v, i, j) for (u, v, i) in self.G.edges(keys=True) for j in range(k)]
			self.solved = True
			#self.k = model.getObjective().getValue()
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
		model, _ = self.ilp_model_weighted(path_weights, time_budget)
		model.optimize()
		self.runtime += model.runtime
		self.model_status = model.status

		if model.status == GRB.OPTIMAL:
			self.solved = True
			self.k = model.getObjective().getValue()
			self.runtime = model.runtime

			x = {}
			for (u, v, i) in self.G.edges(keys=True):
				for j, w in enumerate(path_weights):
					x[(u, v, i, w)] = round(model.getVarByName(f'x[{u},{v},{i},{j}]').x)

			def recover_path(u, v, i, w):
				assert x[(u, v, i, w)] > 0
				x[(u, v, i, w)] -= 1
				self.paths[-1].append((u, v, i))

				while self.G.out_degree(v) > 0:
					for (u_, v_, i_) in self.G.edges(v, keys=True):
						if x[(u_, v_, i_, w)] > 0:
							u, v, i = u_, v_, i_
							break
					x[(u, v, i, w)] -= 1
					self.paths[-1].append((u, v, i))


			for w in path_weights:
				for s in self.sources:
					for (s, s_out, i) in self.G.edges(s, keys=True):
						while x[(s, s_out, i, w)] > 0:
							self.weights.append(w)
							self.paths.append([])
							recover_path(s, s_out, i, w)

		return model.status
	

	"""
	Build basic gurobi ILP model.
	"""
	def ilp_basic_model(self, safe_paths, path_constraints, size, time_budget):
		# create index sets
		T = [(u, v, i, k) for (u, v, i) in self.G.edges(keys=True) for k in range(size)]
		SC = list(range(size))
		PC = [(k, j) for k in range(size) for j in range(len(path_constraints))]

		model = gp.Model('MFD')
		model.setParam('LogToConsole', 0)
		model.setParam('Threads', threads)
		model.setParam('Timelimit', time_budget)

		# Create variables
		x = model.addVars(T, vtype=GRB.BINARY, name='x') # path indicators
		w = model.addVars(SC, vtype=GRB.INTEGER, name='w', lb=0) # weights of paths
		z = model.addVars(T, vtype=GRB.CONTINUOUS, name='z', lb=0) # weight of path per edge
		r = model.addVars(PC, vtype=GRB.BINARY, name='r') # path constraint indicators

		# Objective function: minimise number of paths
		#model.setObjective(sum(x[u, v, i, k] for k in range(size) for s in self.sources for (u, v, i) in self.G.out_edges(s, keys=True)), GRB.MINIMIZE)

		# flow conservation
		for k in range(size):
			model.addConstr(sum(x[v, w, i, k] for v in self.sources for _, w, i in self.G.out_edges(v, keys=True)) == 1)
			#model.addConstr(sum(x[u, v, i, k] for v in self.sinks for u, _, i in self.G.in_edges(v, keys=True)) == 1)
			for v in self.G.nodes:
				if v not in self.sources and v not in self.sinks:
					model.addConstr(sum(x[v, w, i, k] for _, w, i in self.G.out_edges(v, keys=True)) - sum(x[u, v, i, k] for u, _, i in self.G.in_edges(v, keys=True)) == 0)

		# flow balance
		for (u, v, i, f) in self.G.edges(keys=True, data='f'):
			fuv = sum(z[u, v, i, k] for k in range(size))
			model.addConstr(f[0] <= fuv)
			model.addConstr(f[1] >= fuv)

		# linearization
		for (u, v, i) in self.G.edges(keys=True):
			for k in range(size):
				model.addConstr(z[u, v, i, k] <= self.max_flow_value * x[u, v, i, k])
				model.addConstr(w[k] - (1 - x[u, v, i, k]) * self.max_flow_value <= z[u, v, i, k])
				model.addConstr(z[u, v, i, k] <= w[k])

		# pre-defined path constraints
		for j, p in enumerate(path_constraints):
			model.addConstr(sum(r[k, j] for k in range(size)) >= 1)
			#model.addConstr(sum(w[k] * r[k, j] for k in range(size)) >= p[2])
			for k in range(size):
				model.addConstr(sum(x[u, v, i, k] for u, v, i in p[0]) >= p[1] * r[k, j])

		# pre-defined safe constraints
		for k, (idx, l, r) in enumerate(safe_paths):
			for u, v, i in self.heuristic_paths[idx][l:r]:
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
	def ilp_model_weighted(self, weight_set, time_budget):
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
			fuv = sum(weight_set[k] * x[u, v, i, k] for k in range(size))
			model.addConstr(f[0] <= fuv)
			model.addConstr(f[1] >= fuv)

		# flow conservation
		for k in range(size):
			for v in self.G.nodes:
				if v not in self.sources and v not in self.sinks:
					model.addConstr(sum(x[v, w, i, k] for _, w, i in self.G.out_edges(v, keys=True)) - sum(x[u, v, i, k] for u, _, i in self.G.in_edges(v, keys=True)) == 0)

		return model, x

	
	"""
	Find all safe paths of the heuristic poly-time flow decomposition.
	Can only be called if a heuristic flow has been read (returns False
	otherwise). Returns False if safe paths have already been calculated.
	"""
	def find_safe_paths(self):

		if not self.solved_heuristic:
			return False
		if len(self.safe_paths) > 0:
			return False
		assert len(self.heuristic_paths) > 0

		my_safe_paths_amount = 0

		for path in self.heuristic_paths:
			if len(path) <= 1:
				self.safe_paths.append([(0, len(path))])
				my_safe_paths_amount += 1
				continue

			self.safe_paths.append([])
			L, R = 0, 2
			excess_flow = self.G.edges[path[0]]['f'][0] + self.G.edges[path[1]]['f'][1] - self.out_flow(path[1][0], 1)

			while L < len(path) - 1:

				while excess_flow > 0:
					if R - L >= 2:
						# add pair (L, R)
						while self.safe_paths[-1] and (self.safe_paths[-1][-1][0] >= L and self.safe_paths[-1][-1][1] <= R):
							self.safe_paths[-1].pop()
							my_safe_paths_amount -= 1
						self.safe_paths[-1].append((L, R))
						my_safe_paths_amount += 1
					if R == len(path):
						break
					R += 1
					excess_flow += self.G.edges[path[R-1]]['f'][1] - self.out_flow(path[R-1][0], 1)

				assert R <= len(path)
				assert L < len(path)

				if R == len(path) and excess_flow > 0:
					break

				while excess_flow <= 0:
					if L >= R-1:
						# (lower bound) flow on edge is 0
						if R < len(path):
							R += 1
							excess_flow += self.G.edges[path[R-1]]['f'][1] - self.out_flow(path[R-1][0], 1)
						break
					L += 1
					excess_flow -= self.G.edges[path[L-1]]['f'][0] - self.out_flow(path[L][0], 1)
					excess_flow += self.G.edges[path[L]]['f'][0] - self.G.edges[path[L]]['f'][1]

		print("My safe paths:", my_safe_paths_amount)
		self.safe_paths_amount += my_safe_paths_amount

		return True

# Calculates the maximum weight independent path set,
# where the weight of a path is its length
def mlips(mfd):
	# Check if two safe paths are equal (they might live in different heuristic paths)
	def are_equal(x, y):
		if x[2] - x[1] != y[2] - y[1]:
			return False

		for i in range(x[2] - x[1]):
			if mfd.heuristic_paths[x[0]][x[1] + i] != mfd.heuristic_paths[y[0]][y[1] + i]:
				return False

		return True

	# Find all unique safe paths
	length_one_safe_paths = {(u,v,i): True for (u,v,i) in mfd.G.edges(keys=True)}
	safe_paths = []
	for ip, path in enumerate(mfd.heuristic_paths):
		for iw, window in enumerate(mfd.safe_paths[ip]):
			for j in range(window[0], window[1]):
				length_one_safe_paths[mfd.heuristic_paths[ip][j]] = False
			is_unique = TrueA
			for (l, (jp, L, R)) in safe_paths:
				is_unique = is_unique and not are_equal((ip, window[0], window[1]), (jp, L, R))
			if is_unique:
				safe_paths.append((window[1] - window[0], (ip, window[0], window[1])))
		for j, (u,v,i) in enumerate(path):
			if length_one_safe_paths[u,v,i]:
				safe_paths.append((1, (ip, j, j+1)))
	
	# Check whether two paths are independent (subpaths do not occur)
	def independent_paths(p1, p2):
		ip1, l1, r1 = p1
		ip2, l2, r2 = p2

		e1l = mfd.heuristic_paths[ip1][l1]
		e1r = mfd.heuristic_paths[ip1][r1-1]
		e2l = mfd.heuristic_paths[ip2][l2]
		e2r = mfd.heuristic_paths[ip2][r2-1]

		if e2l in nx.edge_dfs(mfd.G, e1r[1]):
			# One comes after the other
			return False

		# The suffix of the left is the prefix of the right
		found_first = -1
		for i in range(l1, r1):
			if found_first > -1:
				if mfd.heuristic_paths[ip1][i] != mfd.heuristic_paths[ip2][i - found_first + l2]:
					return True
			else:
				if mfd.heuristic_paths[ip1][i] == mfd.heuristic_paths[ip2][l2]:
					found_first = i
		return found_first == -1
		

	transitive_graph = nx.DiGraph()
	transitive_graph.add_nodes_from(safe_paths)
	for (ip1, l1, r1) in safe_paths:
		for (ip2, l2, r2) in safe_paths:
			if (ip1, l1, r1) != (ip2, l2, r2) and not independent_paths((ip1, l1, r1), (ip2, l2, r2)):
				transitive_graph.add_edge((ip1, l1, r1), (ip2, l2, r2))
	transitive_graph.add_nodes_from(['s', 't'])
	for v in transitive_graph.nodes():
		if v == 's' or v == 't':
			continue
		if transitive_graph.in_degree(v) == 0:
			transitive_graph.add_edge('s', v)
		if transitive_graph.out_degree(v) == 0:
			transitive_graph.add_edge(v, 't')

	# Normalise node indices to [0..n)
	oldnode_to_newnode = dict()
	newnode_to_oldnode = dict()
	for (x, v) in enumerate(transitive_graph.nodes()):
		oldnode_to_newnode[v] = x
		newnode_to_oldnode[x] = v

	# Call C++ code to calculate maximum weight antichain
	# Program mwa calls naive_minflow_reduction -> naive_minflow_solve -> maxantichain_from_minflow
	mwa_input = io.StringIO()
	mwa_input.write("{} {}\n".format(transitive_graph.number_of_nodes(), transitive_graph.number_of_edges()))
	for (u, v) in transitive_graph.edges():
		mwa_input.write("{} {}\n".format(oldnode_to_newnode[u], oldnode_to_newnode[v]))

	mwa_input.write("{} {}\n".format(oldnode_to_newnode['s'], 0))
	mwa_input.write("{} {}\n".format(oldnode_to_newnode['t'], 0))
	for (ip, l, r) in safe_paths:
		mwa_input.write("{} {}\n".format(oldnode_to_newnode[(ip, l, r)], r - l))

	res = subprocess.run(["./mwa"], input=mwa_input.getvalue(), text=True, capture_output=True)
	if res.stdout != '':
		mwa = list(map(int, res.stdout.split(' ')))
		mwa = list(map(lambda x: x-1, mwa))
	else:
		mwa = []
	
	assert mfd.G.number_of_edges() == 0 or len(mwa) > 0

	mli_safe_paths = []
	for x in mwa:
		# Find corresponding safe path going through the edge (u,v,i)
		assert ip != -1
		ip, l, r = newnode_to_oldnode[x]
		mli_safe_paths.append((ip, l, r))
	
	return mli_safe_paths 

def edge_mwa_safe_paths(mfd, longest_safe_path_of_edge):
	graph = mfd.G
	greedy_paths = mfd.heuristic_paths

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

# Contract the graph: Remove vertices of in/out degree 1.
def y_to_v(ngraph, path_constraints=[], heuristic_paths=[], heuristic_weights=[]):

	def get_out_tree(ngraph, v, processed, contracted, my_leaves, my_root):

		cont = {e: contracted[e] for e in contracted}
		mr = {e: my_root[e] for e in my_root}
		m = ngraph.number_of_edges()

		def is_root(v):
			if ngraph.in_degree(v) != 1:
				return True

			# Check if inexact flow needs to be considered (can not compress in that case)
			_, _, _, (iL, iR) = list(ngraph.in_edges(v, keys=True, data='f'))[0]
			oL, oR = 0, 0
			for _, _, _, (l, r) in ngraph.out_edges(v, keys=True, data='f'):
				oL += l
				oR += r
			if iL > oL or iR < oR:
				return True

			return False

		def is_leaf(v):
			if ngraph.in_degree(v) > 1 or ngraph.out_degree(v) == 0:
				return True

			# Check if inexact flow needs to be considered (can not compress in that case)
			_, _, _, (iL, iR) = list(ngraph.in_edges(v, keys=True, data='f'))[0]
			oL, oR = 0, 0
			for _, _, _, (l, r) in ngraph.out_edges(v, keys=True, data='f'):
				oL += l
				oR += r
			if iL > oL or iR < oR:
				return True

			return False

		# Get root of out_tree
		root = v
		while not is_root(root):
			root = next(ngraph.predecessors(root))

		leaf_edges = list()
		edges = deque(ngraph.out_edges(root, keys=True))
		to_process_intervals = dict()

		while edges:
			u, v, i = edges.pop()
			processed[u] = True
			cont[u, v, i] = True
			mr[u, v, i] = root
			if u in to_process_intervals.keys():
				to_process_intervals[u] += 1
			else:
				to_process_intervals[u] = 1
			if is_leaf(v):
				leaf_edges.append((u, v, i))
			else:
				for u_,v_,i_ in ngraph.out_edges(v, keys=True):
					edges.append((u_, v_, i_))

		# Filter out uncompressible edges
		for (u, v, i) in [(u, v, i) for (u, v, i) in leaf_edges if u == root]:
			cont[u, v, i] = contracted[u, v, i]
			mr[u, v, i] = my_root[u, v, i]
		leaf_edges = [(u, v, i) for (u, v, i) in leaf_edges if u != root]

		# Calculate leaf intervals for every node
		q = deque()
		visited_queue = set()
		for j, (u, v, i) in enumerate(leaf_edges):
			my_leaves[(u,v,i)] = (j, j)
			to_process_intervals[u] -= 1
			if u != root and to_process_intervals[u] == 0:
				q.append(u)
		while q:
			u = q.pop()
			if u in visited_queue:
				continue
			visited_queue.add(u)
			l, r = m+1, -1
			for e in ngraph.out_edges(u, keys=True):
				l = min(l, my_leaves[e][0])
				r = max(r, my_leaves[e][1])
			for e in ngraph.in_edges(u, keys=True):
				my_leaves[e] = (l, r)
				v, u, i = e
				to_process_intervals[v] -= 1
				if v != root and to_process_intervals[v] == 0:
					q.append(v)


		return root, leaf_edges, processed, cont, mr

	def get_in_tree(ngraph, v, processed, contracted, my_leaves, my_root):

		cont = {e: contracted[e] for e in contracted}
		mr = {e: my_root[e] for e in my_root}
		m = ngraph.number_of_edges()

		def is_root(v):
			if ngraph.out_degree(v) != 1:
				return True

			# Check if inexact flow needs to be considered (can not compress in that case)
			_, _, _, (iL, iR) = list(ngraph.out_edges(v, keys=True, data='f'))[0]
			oL, oR = 0, 0
			for _, _, _, (l, r) in ngraph.in_edges(v, keys=True, data='f'):
				oL += l
				oR += r
			if iL > oL or iR < oR:
				return True

			return False

		def is_leaf(v):
			if ngraph.out_degree(v) > 1 or ngraph.in_degree(v) == 0:
				return True

			# Check if inexact flow needs to be considered (can not compress in that case)
			_, _, _, (iL, iR) = list(ngraph.out_edges(v, keys=True, data='f'))[0]
			oL, oR = 0, 0
			for _, _, _, (l, r) in ngraph.in_edges(v, keys=True, data='f'):
				oL += l
				oR += r
			if iL > oL or iR < oR:
				return True

			return False

		# Get root of in_tree
		root = v
		while not is_root(root):
			root = next(ngraph.successors(root))

		leaf_edges = set()
		edges = deque(ngraph.in_edges(root, keys=True))
		to_process_intervals = dict()

		while edges:
			u, v, i = edges.pop()
			processed[v] = True
			cont[u, v, i] = True
			mr[u, v, i] = root
			if v in to_process_intervals.keys():
				to_process_intervals[v] += 1
			else:
				to_process_intervals[v] = 1
			if is_leaf(u):
				leaf_edges.add((u, v, i))
			else:
				for u_,v_,i_ in ngraph.in_edges(u, keys=True):
					edges.append((u_, v_, i_))

		# Filter out uncompressible edges
		for (u, v, i) in [(u, v, i) for (u, v, i) in leaf_edges if v == root]:
			cont[u, v, i] = contracted[u, v, i]
			mr[u, v, i] = my_root[u, v, i]
		leaf_edges = {(u, v, i) for (u, v, i) in leaf_edges if v != root}

		# Calculate leaf intervals for every node
		q = deque()
		visited_queue = set()
		for j, (u, v, i) in enumerate(leaf_edges):
			my_leaves[(u,v,i)] = (j, j)
			to_process_intervals[v] -= 1
			if v != root and to_process_intervals[v] == 0:
				q.append(v)
		while q:
			u = q.pop()
			if u in visited_queue:
				continue
			visited_queue.add(u)
			l, r = m+1, -1
			for e in ngraph.in_edges(u, keys=True):
				l = min(l, my_leaves[e][0])
				r = max(r, my_leaves[e][1])
			for e in ngraph.out_edges(u, keys=True):
				my_leaves[e] = (l, r)
				u, v, i = e
				to_process_intervals[v] -= 1
				if v != root and to_process_intervals[v] == 0:
					q.append(v)

		return root, leaf_edges, processed, cont, mr


	def get_out_trees(ngraph):

		out_trees = dict()
		contracted = {e: False for e in ngraph.edges(keys=True)}
		my_root = {e: e[0] for e in ngraph.edges(keys=True)}
		processed = {v: ngraph.in_degree(v) != 1 for v in ngraph.nodes}
		my_leaves = dict()
		for v in ngraph.nodes:
			if not processed[v]:
				root, leaf_edges, processed, contracted, my_root = get_out_tree(ngraph, v, processed, contracted, my_leaves, my_root)
				if leaf_edges:
					out_trees[root] = leaf_edges
		return out_trees, contracted, my_leaves, my_root


	def get_in_trees(ngraph):

		in_trees = dict()
		contracted = {e: False for e in ngraph.edges(keys=True)}
		my_root = {e: e[0] for e in ngraph.edges(keys=True)}
		processed = {v: ngraph.out_degree(v) != 1 for v in ngraph.nodes}
		my_leaves = dict()
		for v in ngraph.nodes:
			if not processed[v]:
				root, leaf_edges, processed, contracted, my_root = get_in_tree(ngraph, v, processed, contracted, my_leaves, my_root)
				if leaf_edges:
					in_trees[root] = leaf_edges
		return in_trees, contracted, my_leaves, my_root

	out_trees, contracted_out, my_leaves_out, my_root_out = get_out_trees(ngraph)

	mngraph_out_contraction = nx.MultiDiGraph()
	for u, v, i, f in ngraph.edges(keys=True, data='f'):
		if not contracted_out[u, v, i]:
			mngraph_out_contraction.add_edge(u, v, f=f, pred=u)

	new_out_edges = dict()
	for root, leaf_edges in out_trees.items():
		edges = list()
		for u, v, i in leaf_edges:
			j = mngraph_out_contraction.number_of_edges(root, v)
			mngraph_out_contraction.add_edge(root, v, f=ngraph.edges[u, v, i]['f'], pred=u)
			edges.append((root, v, j))
		new_out_edges[root] = edges

	in_trees, contracted_in, my_leaves_in, my_root_in = get_in_trees(mngraph_out_contraction)

	mngraph_in_contraction = nx.MultiDiGraph()
	for u, v, i, f in mngraph_out_contraction.edges(keys=True, data='f'):
		if not contracted_in[u, v, i]:
			mngraph_in_contraction.add_edge(u, v, f=f, succ=(v, i))

	new_in_edges = dict()
	for root, leaf_edges in in_trees.items():
		edges = list()
		for u, v, i in leaf_edges:
			j = mngraph_in_contraction.number_of_edges(u, root)
			mngraph_in_contraction.add_edge(u, root, f=mngraph_out_contraction.edges[u, v, i]['f'], succ=(v, i))
			edges.append((u, root, j))
		new_in_edges[root] = edges

	# Remove trivial_paths found as edges from source to sink in the contraction
	trivial_paths = list()
	edges = list(mngraph_in_contraction.edges(keys=True, data='f'))
	for u, v, i, f in edges:
		if mngraph_in_contraction.in_degree(u) == 0 and mngraph_in_contraction.out_degree(v) == 0:
			if f[0] > 0:
				trivial_paths.append((get_expanded_path([(u, v, i)], mngraph_in_contraction, ngraph, mngraph_out_contraction), f[0]))
			mngraph_in_contraction.remove_edge(u, v, i)

	# Contract given paths
	def contract_paths(paths, weights=[]):
		if not weights:
			weights = [0] * len(paths)
		sol = []
		nw = []
		for pi, path in enumerate(paths):
			contracted_path_length = 0
			cpath = []
			j = 0
			while j < len(path):
				u, v = path[j]
				contracted_path_length += 1
				if not contracted_out[u, v, 0] and not contracted_in[u, v, 0]:
					cpath.append((u, v, 0))
				else:
					if contracted_out[u, v, 0]:
						# In case of an out tree, there might be multiple leaves that we can reach
						root = my_root_out[(u, v, 0)]
						k = j
						while k + 1 < len(path) and (u,v,0) not in out_trees[root]:
							k += 1
							u, v = path[k]
						l, r = my_leaves_out[(u, v, 0)]
						for x in range(l, r+1):
							u1, v1, i1 = new_out_edges[my_root_out[(u, v, 0)]][x]
							if contracted_in[u1, v1, i1]:
								li, ri = my_leaves_in[(u1, v1, i1)]
								for y in range(li, ri+1):
									u2, v2, i2 = new_in_edges[my_root_in[(u1, v1, i1)]][y]
									assert v2 == my_root_in[(u1, v1, i1)]
									cpath.append((u2, v2, i2))
								if l == r:
									while k + 1 < len(path) and v != v2:
										k += 1
										u, v = path[k]
								else:
									assert k + 1 == len(path)
							else:
								cpath.append(new_out_edges[my_root_out[(u, v, 0)]][x])
						j = k
					else:
						# In case of an in tree, we must reach the root
						l, r = my_leaves_in[(u, v, 0)]
						root = my_root_in[(u, v, 0)]
						for x in range(l, r+1):
							u1, v1, i1 = new_in_edges[root][x]
							assert v1 == root
							cpath.append((u1, v1, i1))
						k = j
						while k + 1 < len(path) and v != root:
							k += 1
							u, v = path[k]
						j = k
				j += 1
			if mngraph_in_contraction.in_degree(cpath[0][0]) != 0 or mngraph_in_contraction.out_degree(cpath[0][1]) != 0:
				# Subpath constraint is not part of a trivial path
				sol.append((cpath, contracted_path_length))
				nw.append(weights[pi])
		return sol, nw

	contracted_path_constraints, _ = contract_paths(path_constraints)

	contracted_heuristic_paths_with_length, contracted_heuristic_weights = contract_paths(heuristic_paths, heuristic_weights)
	contracted_heuristic_paths = [path for (path, ln) in contracted_heuristic_paths_with_length]

	return mngraph_out_contraction, mngraph_in_contraction, trivial_paths, contracted_path_constraints, contracted_heuristic_paths, contracted_heuristic_weights

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
			expanded_edge.append((u, v))
			u, v = list(original_graph.in_edges(u))[0]
		expanded_edge.append((u, v))
		original_path += list(reversed(expanded_edge))

	return original_path

def time_to_file(file, num_paths, times):
	with open(file, 'w') as f:
		for i in range(len(num_paths)):
			f.write("{} {}\n".format(num_paths[i], times[i]))

def results_to_file(file, paths, weights):
	with open(file, 'w') as f:
		for weight, path in zip(weights, paths):
			f.write("{}: {}\n".format(weight, path))

trivial_occur = 0
lb_eq_ub = 0
N, M = 0, 0
SAFE_PATHS = 0
x_set_to_one = 0
x_total = 0
num_of_edges_orig = 0
num_of_edges_contracted = 0
sol_length = 0
# Main pipeline function
def pipeline(graph, mngraph_in_contraction, trivial_paths, contracted_path_constraints=[], contracted_heuristic_weights=None, contracted_heuristic_paths=None, is_inexact=False):
	global trivial_occur
	global lb_eq_ub
	global N
	global M
	global SAFE_PATHS
	global x_set_to_one
	global x_total
	global num_of_edges_orig
	global num_of_edges_contracted
	global sol_length
	mfd = Mfd(mngraph_in_contraction, number_of_contracted_paths=len(trivial_paths), heuristic_paths=contracted_heuristic_paths, heuristic_weights=contracted_heuristic_weights)
	N += mngraph_in_contraction.number_of_nodes()
	M += mngraph_in_contraction.number_of_edges()
	if mngraph_in_contraction.number_of_edges() == 0:
		print("Trivially decomposable to {} paths.".format(len(trivial_paths)))
		trivial_occur += 1
		return mfd
	if mfd.max_lower_flow_value == 0:
		print("0 flow is feasible, no decomposition necessary.")
		trivial_occur += 1
		return mfd
	
	can_fail = is_inexact or len(contracted_path_constraints) > 0
	
	if contracted_heuristic_paths:
		assert mfd.find_safe_paths()
		SAFE_PATHS += mfd.safe_paths_amount

		# Calculate safe paths antichain
		longest_safe_path_of_edge = dict()
		for u,v,i in mfd.G.edges(keys=True):
			longest_safe_path_of_edge[u,v,i] = (0, (-1, -1, -1))
		for ip, path in enumerate(mfd.heuristic_paths):
			for j in range(len(path)):
				u,v,i = path[j]
				if longest_safe_path_of_edge[u,v,i][0] < 1 and mfd.G.edges[u,v,i]['f'][0] > 0:
					longest_safe_path_of_edge[u,v,i] = (1, (ip, j, j+1))
			for iw, window in enumerate(mfd.safe_paths[ip]):
				for j in range(window[0], window[1]):
					u,v,i = path[j]
					if longest_safe_path_of_edge[u,v,i][0] < window[1] - window[0]:
						longest_safe_path_of_edge[u,v,i] = (window[1] - window[0], (ip, window[0], window[1]))  

		safe_antichain = edge_mwa_safe_paths(mfd, longest_safe_path_of_edge)
	else:
		safe_antichain = []


	found_sol_or_time_limit = mfd.mfd_algorithm(safe_paths=safe_antichain, path_constraints=contracted_path_constraints, time_budget=30*60)
	assert found_sol_or_time_limit or can_fail

	if mfd.opt_is_greedy:
		lb_eq_ub += 1

	if mfd.model_status == GRB.OPTIMAL or mfd.opt_is_greedy:
		print("Calculated minimum flow decomposition of size {}.".format(mfd.k + len(trivial_paths)))
		print("Runtime:", mfd.runtime)

		x_set_to_one += sum(r - l for (idx, l, r) in safe_antichain)
		x_total += mfd.k * mngraph_in_contraction.number_of_edges()
		sol_length += sum(len(p) for p in mfd.paths)

		num_of_edges_orig += graph.number_of_edges()
		num_of_edges_contracted += mngraph_in_contraction.number_of_edges()
		return mfd

	assert mfd.model_status == GRB.TIME_LIMIT or can_fail
	if mfd.model_status == GRB.TIME_LIMIT:
		print("ILP time limit.")
	else:
		print("Instance not feasible.")
		print("Instance minimum flow: {}".format(mfd.min_flow_value))
	return mfd

# Approximation pipeline
def approx_pipeline(graph, mngraph_in_contraction, trivial_paths):
	global trivial_occur
	mfd = Mfd(mngraph_in_contraction, number_of_contracted_paths=len(trivial_paths))
	if mngraph_in_contraction.number_of_edges() == 0:
		print("Trivially decomposable to {} paths.".format(len(trivial_paths)))
		trivial_occur += 1
		return mfd

	path_weights = set()
	W = 1
	while W <= mfd.max_flow_value:
		path_weights.add(W)
		W *= 2
	for u,v,i,f in mngraph_in_contraction.edges(keys=True, data='f'):
		if f[0] == f[1]:
			path_weights.add(f[0])
	path_weights = list(path_weights)

	status = mfd.solve_given_weights(path_weights, time_budget=30*60)
	if status == GRB.TIME_LIMIT:
		print("ILP Approx time limit.")
		return mfd

	if status == GRB.OPTIMAL:
		print("Calculated approximated flow decomposition of size {}.".format(mfd.k + len(trivial_paths)))
		print("Runtime:", mfd.runtime)
	
	return mfd
	
def read_input_graphs(graph_file, exact_flow):
	graphs = []
	subpath_constraints = []

	with open(graph_file, 'r') as file:
		raw_graphs = file.read().split('#')[1:]
		for g in raw_graphs:
			graph = nx.MultiDiGraph()
			subpaths = list()
			flow = dict()

			lines = g.split('\n')[1:]
			if not lines[-1]:
				lines = lines[:-1]
			graph.add_nodes_from(range(int(lines[0])))
			sp = False	
			for e in lines[1:]:
				if e == "subpaths":
					if not exact_flow:
						print("Warning: Subpath constraints found, but will be ignored (can not be used with inexact flow).", file=sys.stderr)
						break
					sp = True
					continue
				parts = e.split()
				if sp:
					subpath = list()
					for i in range(len(parts)-1):
						# Experimental data looks like this
						if parts[i+1] == '1.0':
							continue
						subpath.append((int(parts[i]), int(parts[i+1])))
					subpaths.append(subpath)
				else:
					if exact_flow:
						key = graph.add_edge(int(parts[0]), int(parts[1]), f=(float(parts[2]), float(parts[2])))
					else:
						key = graph.add_edge(int(parts[0]), int(parts[1]), f=(float(parts[2]), float(parts[3])))
			graphs.append((graph, subpaths))

	return graphs

def read_heuristic_solutions(heuristic_file):
	solutions = []

	with open(heuristic_file, 'r') as file:
		raw_solutions = file.read().split('#')[1:]
		for sol in raw_solutions:
			weights, paths = [], []
			sol_lst = sol.split('\n')
			for weighted_path in sol_lst[1:-1]:
				weight, path = weighted_path.split(':')
				weights.append(int(weight))
				paths.append(ast.literal_eval(path[1:]))
			solutions.append((weights, paths))
			
	return solutions

def read_truth(truth_file):
	solutions = []

	with open(truth_file, 'r') as file:
		raw_solutions = file.read().split('#')[1:]
		for sol in raw_solutions:
			k = sol.split('\n')
			solutions.append(len(k)-2)

	return solutions

def check_solution(graph, paths, weights):
	flow_from_paths = {}
	for (u, v) in graph.edges():
		flow_from_paths[u, v] = 0

	for weight, path in zip(weights, paths):
		for e in path:
			flow_from_paths[e] += weight

	for (u, v, f) in graph.edges(data='f'):
		if flow_from_paths[(u, v)] < f[0] or flow_from_paths[(u, v)] > f[1]:
			return False

	return True

if __name__ == '__main__':

	parser = argparse.ArgumentParser(
		description='''
		Computes Minimum Flow Decomposition.
		This script uses the Gurobi ILP solver.
		''',
		formatter_class=argparse.RawTextHelpFormatter
	)

	requiredNamed = parser.add_argument_group('required arguments')
	requiredNamed.add_argument('-i', '--input', type=str, help='Input filename', required=True)
	requiredNamed.add_argument('-o', '--output', type=str, help='Output filename', required=True)

	parser.add_argument('--heuristic', type=str, help='Heuristic solution (filename)', required=False)

	parser.add_argument('--exact', help='Exact flow decomposition (edge weights form a flow)', action='store_true')
	parser.add_argument('--inexact', help='Inexact flow decomposition (edge weights are intervals)', dest='exact', action='store_false')
	parser.set_defaults(exact=True)

	parser.add_argument('--approx', help='Use approximation', action='store_true')
	parser.add_argument('--no-approx', help='Do not use approximation (default)', dest='approx', action='store_false')
	parser.set_defaults(approx=False)

	parser.add_argument('--verbose', help='Use verbose output', action='store_true')
	parser.add_argument('--no-verbose', help='Do not use verbose output (default)', dest='verbose', action='store_false')
	parser.set_defaults(verbose=False)

	args = parser.parse_args()

	threads = os.cpu_count()
	print(f'INFO: Using {threads} threads for the Gurobi solver')

	#truths = read_truth(args.input.split('.')[0] + '.truth')
	graphs = read_input_graphs(args.input, args.exact)
	heuristic_solutions = read_heuristic_solutions(args.heuristic) if args.heuristic else []

	timelimit = []
	number_of_paths = []
	paths = []
	weights = []
	runtimes = []
	for i, (graph, subpaths) in enumerate(graphs):
		print("Iteration:", i)
		hs = heuristic_solutions[i] if args.heuristic else None
		if args.approx and len(subpaths) > 0:
			print("Note: subpath constraints are not supported with --approx option.")

		heuristic_weights, heuristic_paths = hs if hs  else ([], [])
		mngraph_out_contraction, mngraph_in_contraction, trivial_paths, contracted_path_constraints, contracted_heuristic_paths, contracted_heuristic_weights = y_to_v(graph, path_constraints=subpaths, heuristic_paths=heuristic_paths, heuristic_weights=heuristic_weights)
		
		mfd = approx_pipeline(graph, mngraph_in_contraction, trivial_paths) if args.approx else pipeline(graph, mngraph_in_contraction, trivial_paths, contracted_path_constraints=contracted_path_constraints, contracted_heuristic_weights=contracted_heuristic_weights, contracted_heuristic_paths=contracted_heuristic_paths, is_inexact=not args.exact)
		num_paths = 0 if mfd.model_status == GRB.TIME_LIMIT else mfd.k + mfd.number_of_contracted_paths
		number_of_paths.append(num_paths)
		runtimes.append(mfd.runtime)
		paths.append([get_expanded_path(path, mngraph_in_contraction, graph, mngraph_out_contraction) for path in mfd.paths])
		weights.append(mfd.weights)
		for (path, weight) in trivial_paths:
			paths[-1].append(path)
			weights[-1].append(round(weight))

		if check_solution(graph, paths[-1], weights[-1]):
			print("Solution of {} paths is a feasible flow decomposition.".format(num_paths))
		else:
			print("Error: solution of {} paths is NOT a feasible flow decomposition!.".format(num_paths))

	time_file = ''.join([args.output,'.time'])
	output_file = args.output
	time_to_file(time_file, number_of_paths, runtimes)
	results_to_file(output_file, paths, weights)
	print("Number of timelimited graphs:", timelimit)
	if args.verbose:
		print("Number of trivially decomposed graphs:", trivial_occur)
		print("Lower bound equals upper bound:", lb_eq_ub)
		print("Path variables set to one by safety: {}/{}".format(x_set_to_one, x_total))
		print("Number of edges in the MFD solution:", sol_length)
		print("Total sum of n and m:", N, M)
		print("Number of safe paths:", SAFE_PATHS)
		print("Number of edges in the original graph:", num_of_edges_orig)
		print("Number of edges in the contracted graph:", num_of_edges_contracted)
