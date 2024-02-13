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

in_solutions=0

while IFS= read -r line; do
  if [[ $line == "# Solutions:" ]]; then
    # Lines starting with '#' are copied without change
	in_solutions=1
    echo "$line" > "$output_file"
  elif [[ $line == "# Paths, weights pass test: flow decomposition confirmed." ]]; then
	in_solutions=0
  elif ((in_solutions)); then
    # Extracting values using awk and transforming them
    path=$(echo "$line" | awk '{print $1}')
	vertices=$(echo "$line" | awk -F '[[, ]' '{ for (i = 3; i < NF; i++) printf $i OFS } { print substr($NF, 1, length($NF)-1) }')
	#echo "$vertices"
    vertices_array=($vertices)
    num_vertices=${#vertices_array[@]}

    if [ "$num_vertices" -ge 2 ]; then
      result="$path ["
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

echo "Processing complete. Content in '$output_file' was erased and output written to '$output_file'."

