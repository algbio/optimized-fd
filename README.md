# optimized-fd

optimized-fd is a tool for minimum flow decompositions (mfd) using integer linear programming by implementing several optimization to reduce their size (number of variables/constrains and feasible region dimension).

# Installation

- Install [miniconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
- Clone our this repository and `cd` to the corresponding folder
- `conda env create -f conda_environment.yml`

# Run

To run the project activate the conda environment you created during installation (`conda activate mfd-safety`) and use
`python` to execute the `fd_optimized.py` file.

As an example you can try:

`python ./src/mfd_safety.py -i ./example_inputs/example.graph -o ./example_inputs/example.safe`

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
- An example of such a format can be found in `./example_inputs/example.safe`.

## Parameters

- `-i <path to input file>`. Mandatory.
- `-o <path to locate output>`. Mandatory.
- `-stats` Output stats to file <output>.stats
- `-t <n>` Use n threads for the Gurobi solver; use 0 for all threads (default 0).
- `-ilptb <n>` Maximum time (in seconds) that the ilp solver is allowed to take when computing safe paths for one flow graph.
If the solver takes more than n seconds, then safe for (all) flow decompositions is reported instead.
- `-uef` Uses excess flow to save ILP calls.
- `-uy2v` Use Y2V contraction on the flow graphs to reduce the ILP size.
- `-s/es/rs/ess/esl/rss/rsl {scan, bin_search, exp_search, rep_exp_search}` When running the two-finger algorithm applied
the specified strategy to extend/reduce the current safe interval.
- `-st/est/rst <n>` When running the two-finger algorithm run the `small strategy` when the search space is less than n
and the `large strategy` otherwise.
- `-ugtd/-ugbu` Run a group testing algorithm (top down or bottom up) instead of two-finger.
- 
