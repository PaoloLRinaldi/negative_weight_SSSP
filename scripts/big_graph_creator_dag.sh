#!/bin/bash

# Check if all three arguments are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 coef pow alg"
    exit 1
fi

# Coefficient of number of edges. E.g., the 2 in 2e7
coef=$1

# Power of number of edges. E.g., the 7 in 2e7
pow=$2

# The algorithm, e.g., bfct
alg=$3

# Check that the algorithm is one of the expected ones
case "$alg" in
    "bfct"|"gor"|"rd1"|"rd2"|"dfs")
        ;;
    *)
        echo "Unknown algorithm: $alg"
        exit 1
        ;;
esac

# Map the algorithm to the corresponding coefficient
declare -A alg_coefs=(
    ["bfct"]=5
    ["gor"]=3
    ["rd1"]=3
    ["rd2"]=5
    ["dfs"]=4
)

k_coef=${alg_coefs[$alg]}

# Compute m
m=$(($coef*10**$pow))

# Compute k
k=$(($m/$k_coef))

# Run create_graph.py
# pypy3 create_graph.py "bad_$alg" $k > ../data/graphs/big_"$alg"_"$coef"e"$pow"_sorted.txt
pypy3 create_graph.py "bad_$alg" $k > ../data/graphs/big_"$alg"_"$coef"e"$pow".txt

# # Run CreateGraph to permute the graph
# ../build/CreateGraph load_graph ../data/graphs/big_"$alg"_"$coef"e"$pow"_sorted.txt -p > ../data/graphs/big_"$alg"_"$coef"e"$pow".txt

echo "Script completed successfully"
