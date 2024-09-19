#!/bin/bash

# Check if the directory argument is provided
if [ $# -eq 0 ]; then
	echo "Please provide a directory as a command line argument."
	exit 1
fi

directory=$1
output_k_and_time=$2

# Check if the directory exists
if [ ! -d "$directory" ]; then
	echo "Directory '$directory' does not exist."
	exit 1
fi

# Check if output file exists and erase its content
if [ -e "$output_k_and_time" ]; then
	echo "Erasing content of file $output_k_and_time."
	> $output_k_and_time
fi

# Check if the Python script exists
python_script="../../src/mfd_optimization.py"
if [ ! -f "$python_script" ]; then
	echo "Python script '$python_script' does not exist."
	exit 1
fi

total_runtime=0
num_runs=0
lw_eq_up=0
x_vars_set_to_one=0
x_vars_set_to_zero=0  
x_vars_total=0
num_of_edges_orig=0
num_of_edges_contracted=0

# Loop through each file in the directory
for file in "$directory"/*; do
	if [ -f "$file" ] && [ "${file##*.}" = "graph" ]; then
		# Create output file name with the same name but different extension
		output_file="${file%.graph}.out"
		orig_heuristic_file="${file%.graph}.catfish.out"
		heuristic_file="${file%.graph}.catfish_for_opt.out"

		./sol_from_catfish_to_optimized_ILP_readable.sh $orig_heuristic_file $heuristic_file

		# Run the Python script with the file as input and extract the runtime
		run_stdout=$( { time python3 "$python_script" -i "$file" -o "$output_file" --heuristic "$heuristic_file" --verbose; } 2>&1)
		runtime=$(echo "$run_stdout" | grep "real" | awk '{print $2}')

		lw_eq_up_case=$(echo "$run_stdout" | grep "Lower bound equals upper bound:" | awk '{print $6}')
		lw_eq_up=$(echo "$lw_eq_up + $lw_eq_up_case" | bc)

		x_vars_ratio=$(echo "$run_stdout" | grep "Path variables set to one by safety:" | awk '{print $8}')
		x_vars_set_to_one_case="${x_vars_ratio%/*}"
		x_vars_total_case="${x_vars_ratio#*/}"
		x_vars_set_to_one=$(echo "$x_vars_set_to_one + $x_vars_set_to_one_case" | bc)
		x_vars_total=$(echo "$x_vars_total + $x_vars_total_case" | bc)

		x_vars_zero_ratio=$(echo "$run_stdout" | grep "Path variables set to zero by safety:" | awk '{print $8}')
		x_vars_set_to_zero_case="${x_vars_zero_ratio%/*}"
		x_vars_set_to_zero=$(echo "$x_vars_set_to_zero + $x_vars_set_to_zero_case" | bc)

		num_of_edges_orig_case=$(echo "$run_stdout" | grep "Number of edges in the original graph:" | awk '{print $8}')
		num_of_edges_orig=$(echo "$num_of_edges_orig + $num_of_edges_orig_case" | bc)

		num_of_edges_contracted_case=$(echo "$run_stdout" | grep "Number of edges in the contracted graph:" | awk '{print $8}')
		num_of_edges_contracted=$(echo "$num_of_edges_contracted + $num_of_edges_contracted_case" | bc)

		echo "$run_stdout"

		# Extract the minutes and seconds
		minutes=$(echo "$runtime" | awk -F'm' '{print $1}')
		seconds=$(echo "$runtime" | awk -F'm' '{print $2}' | awk -F's' '{print $1}')

		# Calculate runtime in seconds and increase the total runtime
		runtime_seconds=$(echo "$minutes * 60 + $seconds" | bc)
		total_runtime=$(echo "$total_runtime + $runtime_seconds" | bc)

		echo "Run $num_runs finished in $minutes minutes and $seconds seconds ($runtime_seconds s) on file $file"

		# Increment the number of runs
		num_runs=$((num_runs + 1))

		# Extract the returned number of paths in output_file
		k=$(cat "${output_file}.time" | awk '{print $1}')

		if [ $k = "0" ]; then
			echo "$file"
			continue
		fi

		# Store results in output file of the script
		echo "$k $runtime_seconds $file" >> $output_k_and_time
	fi
done

# Calculate the average runtime
average_runtime=$(echo "scale=2; $total_runtime / $num_runs" | bc)

echo "Average runtime: $average_runtime seconds"
echo "Lower bound was equal to upper bound for $lw_eq_up out of $num_runs runs."
echo "Overall, $x_vars_set_to_one path indicator variables were set to one, out of totally $x_vars_total."
echo "Overall, $x_vars_set_to_zero path indicator variables were set to zero, out of totally $x_vars_total."  
echo "Number of edges in the original graphs: $num_of_edges_orig."
echo "Number of edges in the contracted graphs: $num_of_edges_contracted."
