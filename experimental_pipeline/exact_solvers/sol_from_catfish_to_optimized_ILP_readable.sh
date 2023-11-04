#!/bin/bash

if [ $# -ne 2 ]; then
  echo "Usage: $0 input_file output_file"
  exit 1
fi

input_file="$1"
output_file="$2"

# Check if the input file exists
if [ ! -f "$input_file" ]; then
  echo "Error: Input file '$input_file' does not exist."
  exit 1
fi

# Process the input file and write the result to the output file
while IFS= read -r line; do
  if [[ "$line" == \#* ]]; then
    # Lines starting with '#' are copied without change
    echo "$line" >> "$output_file"
  else
    # Lines not starting with '#' are processed
    # Extracting values using awk and transforming them
    path=$(echo "$line" | awk -F', weight = ' '{print $2}' | awk -F',' '{print $1}')
    vertices=$(echo "$line" | awk -F', vertices = ' '{print $2}')
    vertices_array=($vertices)
    num_vertices=${#vertices_array[@]}

    if [ "$num_vertices" -ge 2 ]; then
      result="$path: ["
      for ((i = 0; i < num_vertices - 1; i++)); do
        result+="(${vertices_array[$i]}, ${vertices_array[$i + 1]})"
		if [ "$i" -lt "$((num_vertices - 2))" ]; then
			result+=", "
		else
			result+="]"
		fi
      done
    else
      result="$path: []"
    fi

    echo "$result" >> "$output_file"
  fi
done < "$input_file"

echo "Processing complete. Output written to '$output_file'."

