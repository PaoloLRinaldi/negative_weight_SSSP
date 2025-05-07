import sys

if len(sys.argv) != 3:
    print("Usage: python3 from_dimacs.py <input_file> <output_file>")
    exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, 'r') as f:
    graph = f.read().splitlines()

# Construct a list called p_graph, where we collect all the elements in
# graph that start with 'p'. Also, remove the 'p' and the following whitespace(s) from the string

p_graph = []
for line in graph:
    line = line.strip()
    if line == '': continue
    if line[0] == 'p':
        line = line[1:].strip()
        if line.startswith('sp'):
            line = line[2:].strip()
        p_graph.append(line)

if len(p_graph) != 1:
    print("Input file does not contain exactly one 'p' line")
    exit(1)

p_graph = p_graph[0].split(' ')

n = int(p_graph[0])
m = int(p_graph[1])

# Construct a list called a_graph, where we collect all the elements in
# graph that start with 'a'. Also, remove the 'a' and the following whitespace(s) from the string

a_graph = []
for line in graph:
    line = line.strip()
    if line == '': continue
    if line[0] == 'a':
        line = line[1:].strip()

        line = line.split()

        a_graph.append((int(line[0]) - 1, int(line[1]) - 1, int(line[2])))


# Check whether the number of edges in a_graph is equal to the number of edges in p_graph
if len(a_graph) != m:
    print("Input file does not contain the correct number of 'a' lines")
    exit(1)

# Check whether the maximum node in a_graph is larger the number of nodes in p_graph
if max([max(el1, el2) for el1, el2, el3 in a_graph]) >= n or \
    min([min(el1, el2) for el1, el2, el3 in a_graph]) < 0:
    print("Input file contains an invalid edge")
    exit(1)


# Write the output file
with open(output_file, 'w') as f:
    f.write(str(n) + '\n')
    for edge in a_graph:
        f.write(str(edge[0]) + ' ' + str(edge[1]) + ' ' + str(edge[2]) + '\n')