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
python_script="/abga/work/andrgrig/MFD-ILP/MFD in DAGS/mfd_subpath.py"
if [ ! -f "$python_script" ]; then
    echo "Python script '$python_script' does not exist."
    exit 1
fi


total_runtime=0
num_runs=0

# Loop through each file in the directory
for file in "$directory"/*; do
	if [ -f "$file" ] && [ "${file##*.}" = "graph" ]; then
		# Create output file name with the same name but different extension
        output_file="${file%.graph}.out"

		# Create proper inputs for standard ILP subpaths
		graph_file="${file%.graph}.no_subpaths_graph"
		awk 'BEGIN { found = 0 } /subpaths/ { found = 1; exit } found == 0 { print }' $file > $graph_file
		echo "Created file $graph_file containing only the graph (without subpaths)"
		subpath_file="${file%.graph}.subpaths"
		paths=$(awk '/subpaths/ { found = 1; next } found == 1 { NF--; print }' $file)
		num_paths=$(echo "$paths" | wc -l)
		echo -e "# graph\n$num_paths\n$paths" > $subpath_file
		echo "Created file $subpath_file containing only the subpaths"

        # Run the Python script with the file as input and extract the runtime
		run_stdout=$( { time python3 "$python_script" -i "$graph_file" -o "$output_file" -s "$subpath_file"; } 2>&1)
        runtime=$(echo "$run_stdout" | grep "real" | awk '{print $2}')

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

		if [ ! -e "${output_file}" ]; then
			echo "TIMEOUT: $file"
			continue
		fi

		# Extract the returned number of paths in output_file
		k=$(cat "${output_file}" | wc -l)
		k=$(echo "$k - 1" | bc)

		echo "Sol: $k"

		if [ "$k" = "0" ]; then
			echo "UNFEASIBLE: $file"
			continue
		fi

		# Store results in output file of the script
		echo "$k $runtime_seconds $file" >> $output_k_and_time
    fi
done

# Calculate the average runtime
average_runtime=$(echo "scale=2; $total_runtime / $num_runs" | bc)

echo "Average runtime: $average_runtime seconds"
