#!/bin/sh

optimization=""

usage() {
    echo "Usage: $0 <input_file> [--optimization <opt_value>]"
    exit 1
}

if [ $# -eq 0 ]; then
    usage
fi

file=$1
shift

while [ "$#" -gt 0 ]; do
    case "$1" in
        --optimization)
            if [ -n "$2" ]; then
                optimization=$2
                shift 2
            else
                echo "Error: --optimization requires an argument."
                usage
            fi
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

if [ ! -e "$file" ]; then
    echo "Input file '$file' does not exist."
    exit 1
fi

directory="${file}_split/"
current_dir=$(pwd)

mkdir -p $directory

echo "Separating graphs..."
python3 ./separateGraphs.py -i "$file"

echo "Running optimized ILP..."
if [ -n "$optimization" ]; then
    ./run_tests_optimized.sh "$directory" "${current_dir}/tests_optimized.out" --optimization "$optimization"
else
    ./run_tests_optimized.sh "$directory" "${current_dir}/tests_optimized.out"
fi

# echo "Running standard ILP.."
# echo "> ./run_tests_standard_ilp.sh '$directory' '${current_dir}/tests_standard.out'"
# ./run_tests_standard_ilp.sh "$directory" "${current_dir}/tests_standard.out"

# echo "Running toboggan.."
# echo "> ./run_tests_toboggan.sh '$directory' '${current_dir}/tests_toboggan.out'"
# ./run_tests_toboggan.sh "$directory" "${current_dir}/tests_toboggan.out"

echo "All runs finished."
echo "Results in files: ./tests_optimized.out, ./tests_standard.out, ./tests_toboggan.out."