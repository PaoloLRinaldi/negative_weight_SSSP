#!/bin/bash
echo "Unzipping USA-road-d.USA.gr.gz..."
gunzip ../data/graphs/USA-road-d.USA.gr.gz data/graphs/USA-road-d.USA.gr

if [ ! -f "../data/graphs/USA-road-d.USA.txt" ]; then
    echo "Converting from dimacs..."
    pypy3 from_dimacs.py ../data/graphs/USA-road-d.USA.gr ../data/graphs/USA-road-d.USA.txt
else
    echo "../data/graphs/USA-road-d.USA.txt already exists, skipping..."
fi

echo "Generating feasible potential..."
# Solve the positive instance
../build/Main "SSSP time GOR ../data/graphs/USA-road-d.USA.txt 0 1"

if [ ! -f "../data/graphs/USA-road-d.USA_1.txt" ]; then
    echo "../data/graphs/USA-road-d.USA_1.txt does not exist, creating..."
    pypy3 create_graph.py restr_from_pot ../data/graphs/USA-road-d.USA.txt ../data/graphs/USA-road-d.USA_result0.txt 1 1 > ../data/graphs/USA-road-d.USA_1.txt
else
    echo "../data/graphs/USA-road-d.USA_1.txt already exists, skipping..."
fi

if [ ! -f "../data/graphs/USA-road-d.USA_10.txt" ]; then
    echo "../data/graphs/USA-road-d.USA_10.txt does not exist, creating..."
    pypy3 create_graph.py restr_from_pot ../data/graphs/USA-road-d.USA.txt ../data/graphs/USA-road-d.USA_result0.txt 10 1 > ../data/graphs/USA-road-d.USA_10.txt
else
    echo "../data/graphs/USA-road-d.USA_10.txt already exists, skipping..."
fi

if [ ! -f "../data/graphs/USA-road-d.USA_100.txt" ]; then
    echo "../data/graphs/USA-road-d.USA_100.txt does not exist, creating..."
    pypy3 create_graph.py restr_from_pot ../data/graphs/USA-road-d.USA.txt ../data/graphs/USA-road-d.USA_result0.txt 100 1 > ../data/graphs/USA-road-d.USA_100.txt
else
    echo "../data/graphs/USA-road-d.USA_100.txt already exists, skipping..."
fi
