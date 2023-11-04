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
script="/abga/work/andrgrig/catfish-0.2.1_linux_x86_64/catfish"
if [ ! -f "$script" ]; then
    echo "Program '$script' does not exist."
    exit 1
fi


total_runtime=0
num_runs=0

# Loop through each file in the directory
for file in "$directory"/*; do
	if [ -f "$file" ] && [ "${file##*.}" = "sgr" ]; then
		# Create output file name with the same name but different extension
        output_file="${file%.sgr}.catfish.out"

        # Run the Python script with the file as input and extract the runtime
		run_stdout=$( { time "$script" -i "$file" -o "$output_file"; } 2>&1)
        runtime=$(echo "$run_stdout" | grep "real" | awk '{print $2}')

		# Extract the minutes and seconds
		minutes=$(echo "$runtime" | awk -F'm' '{print $1}')
		seconds=$(echo "$runtime" | awk -F'm' '{print $2}' | awk -F's' '{print $1}')

		# Calculate runtime in seconds and increase the total runtime
		runtime_seconds=$(echo "$minutes * 60 + $seconds" | bc)
		total_runtime=$(echo "$total_runtime + $runtime_seconds" | bc)

		echo "Run $num_runs finished in $minutes minutes and $seconds seconds ($runtime_seconds s)"

		# Increment the number of runs
		num_runs=$((num_runs + 1))

		# Extract the returned number of paths in output_file
		k=$[$(cat $output_file | wc -l) - 1]

		if [ $k = 0 ]; then
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
