#!/bin/bash

# Check if all three arguments are provided
if [ $# -ne 4 ]; then
    echo "Usage: $0 coef pow alg dag_frac"
    exit 1
fi

# Coefficient of number of edges. E.g., the 2 in 2e7
coef=$1

# Power of number of edges. E.g., the 7 in 2e7
pow=$2

# The algorithm, e.g., bfct
alg=$3

# How much larger is the entire augmented graph wrt the original DAG (usually 6)
dag_frac=$4

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
k=$(($m/$k_coef/$dag_frac))

# Run create_graph.py
pypy3 create_graph.py "bad_$alg" $k > ../data/graphs/big_aug_"$alg"_"$coef"e"$pow"_bare.txt

# Compute maxw
# maxw=$((2000000 * coef))
maxw=$(($k*6))

# Run CreateGraph to sort the graph
../build/CreateGraph load_graph ../data/graphs/big_aug_"$alg"_"$coef"e"$pow"_bare.txt -a $dag_frac $maxw > ../data/graphs/big_aug_"$alg"_"$coef"e"$pow"_sorted.txt

# Run CreateGraph to permute the graph
../build/CreateGraph load_graph ../data/graphs/big_aug_"$alg"_"$coef"e"$pow"_sorted.txt -p > ../data/graphs/big_aug_"$alg"_"$coef"e"$pow".txt

echo "Script completed successfully"
