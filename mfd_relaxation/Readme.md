# Relaxed MFD via Integer Linear Programming

This extension is created by relaxing one of the conditions in the basic ILP formulation for MFD. Considering the edge identifier variable $x_{uvi}$ is not longer binary, there is not guarantee that a feasible solution can be found, but at least a lower bound can be obtained.

# Pre-requisites

All the formulations available require the Gurobi solver to solve its model.  
Download the solver from [www.gurobi.com](www.gurobi.com), activate the (academic) license as instructed, and then install the Python API with:

```
pip3 install gurobipy
```

Also, it requires a few extra Python libraries:

  - itertools
  - more_itertools
  - math
  - os 
  - networkx 

# Run

To run the formulation, use 'python' to execute the 'mfd_relaxed.py' file.

As an example you can try:

`python ./mfd_relaxation/mfd_relaxed.py -i ./example_inputs/example.graph -safe ./example_inputs/example.paths -o ./example_inputs/results.path`

## Input

- The input is a file containing a sequence of (directed) acyclic flow graphs separated by lines starting with `#`.
- The first line of each flow graph contains the number of vertices of the graph, after this every flow edge from vertex
`u` to  `v` carrying `f` flow is represented in a separated line in the format `u v f`.
- Vertices must be integers following a topological order of the graph.
- An example of such a format can be found in `./example_inputs/example.graph`.


## Output

- The output is a file containing a sequence of paths separated by lines starting with `#` (one per flow
graph in the input).
- Each line contains the weight associated and the content of a path corresponding sequence of vertices.
- An example of such a format can be found in `./example_inputs/example.out`.

## Parameters

- `-i <path to input file>`. Mandatory.
- `-o <path to locate output>`. Mandatory.

Additional Parameters (optional)
- `-stats` Output stats to file <output>.stats
- `-t <n>` Use n threads for the Gurobi solver; use 0 for all threads (default 0).
- `-ilptb <n>` Maximum time (in seconds) that the ilp solver is allowed to take when computing safe paths for one flow graph.
If the solver takes more than n seconds, then safe for (all) flow decompositions is reported instead.
