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

echo "# Solutions:" > $output_file
awk 'NR > 1 { printf "%s: [", $1; for (i = 2; i < NF; i++) { printf("(%s, %s)%s", $i, $(i+1), (i+1 == NF ? "]\n" : ", ")); }}' $input_file >> $output_file

echo "Processing complete. Content in '$output_file' was erased and output written to '$output_file'."

