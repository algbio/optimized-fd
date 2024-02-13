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

#echo "Running IFD solver heuristic.."
#echo "> ./run_tests_heuristic.sh '$directory' '${current_dir}/tests_heuristic.out'"
#./run_tests_heuristic.sh "$directory" "${current_dir}/tests_heuristic.out"
#
#echo "Running standard ILP.."
#echo "> ./run_tests_standard.sh '$directory' '${current_dir}/tests_standard.out'"
#./run_tests_standard.sh "$directory" "${current_dir}/tests_standard.out"

echo "Running optimized ILP.."
echo "> ./run_tests_optimized.sh '$directory' '${current_dir}/tests_optimized.out'"
./run_tests_optimized.sh "$directory" "${current_dir}/tests_optimized.out"

echo "All runs finished."
echo "Results in files: ./tests_heuristic.out, ./tests_standard.out, ./tests_optimized.out."
