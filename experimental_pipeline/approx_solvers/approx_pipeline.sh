#!/bin/sh

# Check if the file argument is provided
if [ $# -eq 0 ]; then
    echo "Please provide the input file as a command line argument."
    exit 1
fi

file=$1

# Check if the directory exists
if [ ! -e "$file" ]; then
    echo "Input file '$file' does not exist."
    exit 1
fi

directory="${file}_split/"
current_dir=$( pwd )
echo "> mkdir -p $directory"
mkdir -p $directory

echo "Separating graphs.."
echo "> python3 ./separateGraphs.py -i "$file""
python3 ./separateGraphs.py -i "$file"

echo "Running approximate ILP.."
echo "> ./run_tests_given_weights_ILP.sh '$directory' '${current_dir}/tests_approx_ILP.out'"
./run_tests_given_weights_ILP.sh "$directory" "${current_dir}/tests_approx_ILP.out"

echo "Running catfish.."
echo "> ./run_tests_catfish.sh '$directory' '${current_dir}/tests_catfish.out'"
./run_tests_catfish.sh "$directory" "${current_dir}/tests_catfish.out"

echo "All runs finished."
echo "Results in files: ./tests_approx_ILP.out, ./tests_catfish.out."
