#!/usr/bin/python3

import sys
import random
import math
from dataclasses import dataclass
import argparse
from min_mean_cycle import minAvgWeight

# classes

@dataclass
class Graph:
    number_of_nodes: int
    # edges: list[tuple[int,int,int]] # source, target, weight
    edges: list

# functions

def print_in_file_format(graph):
    print(graph.number_of_nodes)
    for edge in graph.edges:
        print("{} {} {}".format(*edge))

def read_graph_from_file(filename):
    with open(filename) as f:
        n = int(f.readline())
        edges = [tuple(int(elem) for elem in line.split()) for line in f.readlines()]
        return Graph(n, edges)

# just a dumb example
def create_complete_graph(parameters):
    n = int(parameters)

    graph = Graph(n, [])
    graph.edges = [ (i,j,1) for i in range(n) for j in range(n) if i != j ]
    return graph

def create_random_graph(n, p):
    n, p = int(n), float(p)
    graph = Graph(n, [])
    e = random.sample([(i,j) for i in range(n) for j in range(n) if i != j], math.floor(p * n * (n-1)))
    graph.edges = map(lambda a: (a[0], a[1], random.randint(-60,60)), e)
    return graph

def create_random_restricted_graph(n, p):
    '''
    Only using positive weights for a first dummy version
    '''
    n, p = int(n), float(p)
    graph = Graph(n, [])
    if n == 1: return graph
    source_edges = [(0, i, 0) for i in range(1, n)]
    e = random.sample([(i,j) for i in range(1, n) for j in range(1, n) if i != j], math.floor(p * (n - 1) * (n - 2)))
    other_edges = map(lambda a: (a[0], a[1], random.randint(1,60)), e)
    graph.edges = source_edges + list(other_edges)
    return graph

def create_random_restricted_graph2(n, p, max_weight):
    '''
    Minimum weight is -1. Minimum mean cycle is at least 1. The source is connected to all other nodes with edges of weight 0
    '''
    random.seed(1234)
    n, p, max_weight = int(n), float(p), int(max_weight)
    graph = Graph(n, [])
    if n == 1: return graph
    source_edges = [(0, i, 0) for i in range(1, n)]
    e = random.sample([(i,j) for i in range(1, n) for j in range(1, n) if i != j], math.floor(p * (n - 1) * (n - 2)))
    other_edges = map(lambda a: (a[0], a[1], random.randint(-1, max_weight + 1)), e)
    graph.edges = source_edges + list(other_edges)

    map_edges = [{} for _ in range(n)]  # head, tail -> edge index
    for i, edge in enumerate(graph.edges):
        map_edges[edge[0]][edge[1]] = i

    # print_in_file_format(graph)

    # Ensuring that miminum mean cycle has weight at least 1
    while True:
        avg_weight, min_cycle = minAvgWeight(graph.number_of_nodes, graph.edges)
        if avg_weight is None or avg_weight >= 1: return graph

        print('Found cycle with weight', avg_weight, 'Increasing weights.')

        # I increase weights, not caring for increasing more than max_weight

        # I don't want to increase the weight of the edges that start from the source
        edge_indices = [map_edges[min_cycle[i]][min_cycle[i + 1]] for i in range(len(min_cycle) - 1) if min_cycle[i] != 0]

        tot_length = len(min_cycle)
        tot_weight = round(avg_weight * tot_length)

        sample = random.choices(edge_indices, k=tot_length - tot_weight)

        for i in sample:
            # Increasing weight by 1
            graph.edges[i] = (graph.edges[i][0], graph.edges[i][1], graph.edges[i][2] + 1)

def create_random_restricted_graph3(n, p):

    # Creating a random graph with all weights = 2
    n, p = int(n), float(p)
    graph = create_random_restricted_graph4(n - 1, p)
    graph.number_of_nodes = n
    graph.edges = [(edge[0] + 1, edge[1] + 1, edge[2]) for edge in graph.edges]

    # Adding supersource
    graph.edges += [(0, i, 0) for i in range(1, n)]

    # Sorting edges
    graph.edges = sorted(graph.edges)

    return graph

def create_random_restricted_graph4(n, p):

    # Creating a random graph with all weights = 2
    n, p = int(n), float(p)
    graph = Graph(n, [])
    graph.edges = random.sample([(i,j, 2) for i in range(n) for j in range(n) if i != j], math.floor(p * (n - 1) * n))


    # Maximising number of edges with weight -1
    to_be_visited = {i for i in range(n)}
    potentials = [None] * n
    node_to_neighbors = [[] for _ in range(n)]
    for i, edge in enumerate(graph.edges):
        node_to_neighbors[edge[0]].append(i)

    while len(to_be_visited) > 0:
        # We will visit all reachable nodes starting from tmp_source
        tmp_source = random.choice(list(to_be_visited))

        visited_edge_idx = set()  # edges visited in this iteration
        visited_nodes = set()  # nodes visited in this iteration

        # BFS to set the potentials of all reachable nodes
        potentials[tmp_source] = 0
        queue = [tmp_source]

        while len(queue) > 0:
            curr_source = queue.pop(0)
            for edge_idx in node_to_neighbors[curr_source]:
                edge = graph.edges[edge_idx]
                # If I meet a node that has already been visited in previous macro-iterations,
                # set the weight to -1 and forget about that edge
                if edge[1] not in to_be_visited:
                    graph.edges[edge_idx] = (edge[0], edge[1], -1)
                    continue
                if potentials[edge[1]] is None:
                    potentials[edge[1]] = potentials[curr_source] + edge[2]
                    queue.append(edge[1])
                visited_edge_idx.add(edge_idx)
            visited_nodes.add(curr_source)

        # Fixing edges weights
        for edge_idx in visited_edge_idx:
            edge = graph.edges[edge_idx]
            graph.edges[edge_idx] = (edge[0], edge[1], edge[2] + potentials[edge[0]] - potentials[edge[1]] - 1)

        for node in visited_nodes:
            to_be_visited.remove(node)

    # Sorting edges
    graph.edges = sorted(graph.edges)

    # Check
    avg_weight, _ = minAvgWeight(graph.number_of_nodes, graph.edges)
    if avg_weight is None or avg_weight >= 1:
        # print('Minimum mean cycle >= 1')
        return graph

    raise Exception('Minimum mean cycle < 1')


def create_restricted_connected(n, fully_connected=1):  # 0: disconnected, 1: fully connected
    n = int(n)
    sqrtn = int(math.sqrt(n))
    n = sqrtn ** 2
    graph = Graph(n + 1, [])

    # We don't check whether the number of edges is correct, I write this down
    # in order to be able to generate graphs of the desired size
    if fully_connected == 1:
        expected_m = n * ((n - 1) + n * (n - 1) // 2 + n * n * (n - 1) // 2) + (n * n)
        expected_m = n * (n - 1) * (2 + (1 + n) * n // 2) + n
    else:
        expected_m = n * ((n - 1) + n * (n - 1) // 2 + 1) - 1 + (n * n)

    edges = []

    for scc in range(sqrtn):
        for i in range(sqrtn - 1):
            edges.append((scc * sqrtn + i + 1, scc * sqrtn + i + 1 + 1, -1))

        for i in range(1, sqrtn):
            for j in range(i):
                edges.append((scc * sqrtn + i + 1, scc * sqrtn + j + 1, 2 * (i - j + 1) - 1))

        if fully_connected == 1:
            # If you want a fully connected graph
            for scc2 in range(scc + 1, sqrtn):
                for i in range(sqrtn):
                    for j in range(sqrtn):
                        edges.append((scc * sqrtn + i + 1, scc2 * sqrtn + j + 1, -1))

        elif fully_connected == 0:
            # If you want to connect the SCCs with a single edge
            if scc != sqrtn - 1:
                edges.append((scc * sqrtn + (sqrtn - 1) + 1, (scc + 1) * sqrtn + 1, -1))
        else:
            raise Exception('Invalid fully_connected value')

    graph.edges = edges

    # Adding supersource
    graph.edges += [(0, i, 0) for i in range(1, n + 1)]

    graph.edges = sorted(graph.edges)

    return graph

def create_bad_bfct(k):
    k = int(k)
    n = 4 * k - 1
    expected_m = 5 * k - 3
    graph = Graph(n, [])

    edges = []

    # Main path P
    for i in range(3 * k - 3):
        edges.append((i + 1, i, -1))

    # Every 3rd vertex in P is connected to vertex 3k - 2
    for i in range(0, 3 * k - 2, 3):
        edges.append((i, 3 * k - 2, -1))

    # Last vertices
    for i in range(3 * k - 1, n):
        edges.append((3 * k - 2, i, -1))

    assert(len(edges) == expected_m)

    graph.edges = edges

    return graph

def create_bad_mbfct(k):
    k = int(k)
    n = 6 * k - 1
    expected_m = 7 * k - 3
    graph = Graph(n, [])

    edges = []

    # Main path P
    for i in range(3 * k - 3):
        edges.append((i + 1, i, -1))

    # Every 3rd vertex in P is connected to vertex 3k - 2
    for i in range(0, 3 * k - 2, 3):
        edges.append((i, 3 * k - 2, -1))

    # (Almost) last vertices
    for i in range(3 * k - 1, 4 * k - 1):
        edges.append((3 * k - 2, i, -1))
    
    # Last vertices
    for i in range(2 * k):
        tail = 4 * k - 1 + i
        head = 0 if i % 2 == 0 else 3 * k - 3
        weight = -4 * k * (i + 2)

        edges.append((tail, head, weight))

    assert(len(edges) == expected_m)

    graph.edges = edges

    return graph

def create_bad_gor(k):
    k = int(k)
    n = 2 * k + 1
    excepted_m = 3 * k - 1
    graph = Graph(n, [])

    edges = []

    edges.append((0, 1, -3 * k))
    edges.append((0, k, -1))
    edges += [(i, i + 1, 1) for i in range(1, k - 1)]
    edges += [(k, k + 1 + i, -1) for i in range(k)]
    edges += [(i - 1, k, 2 * (k - i)) for i in range(2, k + 1)]

    assert(len(edges) == excepted_m)

    graph.edges = sorted(edges)

    return graph

def create_bad_rd1(k):
    k = int(k)
    n = 2 * k
    excepted_m = 3 * k - 2
    graph = Graph(n, [])

    edges = []

    edges += [(2 * i, 2 * i + 1, -1) for i in range(k)]
    edges += [(2 * i + 1, 2 * i + 2, -1) for i in range(k - 1)]
    # edges += [(2 * i, 2 * i + 1, 0) for i in range(k)]
    # edges += [(2 * i + 1, 2 * i + 2, -2) for i in range(k - 1)]
    edges += [(2 * i, 2 * i + 2, -1) for i in range(k - 1)]

    assert(len(edges) == excepted_m)

    graph.edges = sorted(edges)

    return graph

def create_bad_rd2(k):
    k = int(k)
    n = 3 * k + 1
    excepted_m = 5 * k - 2

    graph = create_bad_rd1(k)

    new_edges = []
    new_edges += [(2 * i + 1, 2 * k, -1) for i in range(k)]
    new_edges += [(2 * k, i, -1) for i in range(2 * k + 1, n)]

    graph.number_of_nodes = n
    graph.edges += new_edges

    assert(len(graph.edges) == excepted_m)

    graph.edges = sorted(graph.edges)

    return graph

def create_bad_dfs(k):
    k = int(k)
    n = 2 * k
    excepted_m = 4 * k - 3
    graph = Graph(n, [])

    edges = []

    edges += [(i, i + 1, -1) for i in range(k - 1)]
    edges += [(i, i + 1, -1) for i in range(k, n - 1)]
    edges += [(i, i + k, -1) for i in range(k)]
    edges += [(i + k, i + 1, -1) for i in range(k - 1)]

    assert(len(edges) == excepted_m)

    graph.edges = sorted(edges)

    return graph

def create_restricted_from_potential(graph_filename, potential_filename, max_shift, frac):
    graph = read_graph_from_file(graph_filename)
    max_shift = int(max_shift)

    with open(potential_filename, 'r') as f:
        potentials = [int(elem) if elem != 'inf' else 0 for elem in f.read().split()]

    assert(len(potentials) == graph.number_of_nodes)

    # Take n random numbers in the range from 0 to m - 1
    # random.seed(0)
    random_indices = random.sample(range(0, graph.number_of_nodes), int(graph.number_of_nodes * float(frac)))
    for i in random_indices:
        potentials[i] += random.randint(0, max_shift)

    edges = [(edge[0], edge[1], edge[2] + potentials[edge[0]] - potentials[edge[1]]) for edge in graph.edges]

    graph.edges = sorted(edges)

    return graph


def permute_edges(edges, n):
    permutation = list(range(n))
    random.shuffle(permutation)

    new_edges = [(permutation[edge[0]], permutation[edge[1]], edge[2]) for edge in edges]
    return sorted(new_edges)


def create_graph(graph_type, parameters, graph_to_function):
    if graph_type not in graph_to_function:
        print("Unknown graph type: " + graph_type)
        exit(1)
    
    return graph_to_function[graph_type](*parameters)

def augment_graph(graph, p, max_weight):
    # It generates new edges, and each time it checkes if the new edge is already there
    n = graph.number_of_nodes
    m_old_edges = len(graph.edges)

    set_edges = set((edge[0], edge[1]) for edge in graph.edges)

    max_new_edges = math.floor(max(p * n * (n - 1) - m_old_edges, 0)) if p <= 1.0 else math.floor(min(p * m_old_edges, n * (n - 1)) - m_old_edges)

    new_edges = list(random.sample([(i,j, max_weight) for i in range(n) for j in range(n) if i != j and (i,j) not in set_edges], max_new_edges))

    graph.edges = graph.edges + new_edges
    
    return graph


# main
def main():
    graph_type_to_function = {
        "read_graph": read_graph_from_file,
        "complete_unit_graph": create_complete_graph,
        "random_graph": create_random_graph,
        "random_restricted_graph": create_random_restricted_graph,
        "random_restricted_graph2": create_random_restricted_graph2,
        "random_restricted_graph3": create_random_restricted_graph3,
        "random_restricted_graph4": create_random_restricted_graph4,
        "restricted_connected": create_restricted_connected,
        "bad_bfct": create_bad_bfct,
        "bad_mbfct": create_bad_mbfct,
        "bad_gor": create_bad_gor,
        "bad_rd1": create_bad_rd1,
        "bad_rd2": create_bad_rd2,
        "bad_dfs": create_bad_dfs,
        "restr_from_pot": create_restricted_from_potential
    }
    # Create the argument parser
    parser = argparse.ArgumentParser()

    # Add the required first argument with choices, which is the graph type
    parser.add_argument('graph_type', nargs=1, choices=list(graph_type_to_function.keys()), help=f'Graph type to create.')

    # Add the arguments for the graph
    parser.add_argument('graph_args', nargs='*')

    # Add the permutation argument
    parser.add_argument('-p', '--perm', action='store_true', help='Randomly permute vertices.', default=False)

    # Add the argument with two required values
    parser.add_argument('-a', '--augment', nargs=2, type=float, metavar=('FLOAT', 'INTEGER'), default=None, help='Augment a graph with new edges. Two required arguments: a float p and an integer w. The total amount of edges in the end is p * n * (n-1), and the weights are w.')

    # Parse the command-line arguments
    args = parser.parse_args()

    gen_graph = create_graph(args.graph_type[0], args.graph_args, graph_type_to_function)
    
    if args.augment:
        gen_graph = augment_graph(gen_graph, args.augment[0], args.augment[1])

    if args.perm:
        gen_graph.edges = permute_edges(gen_graph.edges, gen_graph.number_of_nodes)
    
    gen_graph.edges = sorted(gen_graph.edges, key=lambda x: (x[0], x[1]))

    print_in_file_format(gen_graph)


if __name__ == "__main__":
    main()
