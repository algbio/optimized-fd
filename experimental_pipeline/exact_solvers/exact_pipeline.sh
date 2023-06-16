#!/bin/sh

# Check if the directory argument is provided
if [ $# -eq 0 ]; then
    echo "Please provide a directory as a command line argument."
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

echo "Running optimized ILP.."
echo "> ./run_tests_optimized.sh '$directory' '${current_dir}/tests_optimized.out'"
./run_tests_optimized.sh "$directory" "${current_dir}/tests_optimized.out"

echo "Running standard ILP.."
echo "> ./run_tests_standard_ilp.sh '$directory' '${current_dir}/test_standard.out'"
./run_tests_standard_ilp.sh "$directory" "${current_dir}/test_standard.out"

echo "Running toboggan.."
echo "> ./run_tests_toboggan.sh '$directory' '${current_dir}/tests_toboggan.out'"
./run_tests_toboggan.sh "$directory" "${current_dir}/tests_toboggan.out"

echo "All runs finished."
echo "Results in files: ./tests_optimized.out, ./tests_standard.out, ./tests_toboggan.out."
