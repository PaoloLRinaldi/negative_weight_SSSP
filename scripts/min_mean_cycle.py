# Python3 program to find minimum
# average weight of a cycle in
# connected and directed graph.

# a struct to represent edges
class edge:
    def __init__(self, u, w):
        self.From = u
        self.weight = w

def addedge(edges, u, v, w):
    edges[v].append(edge(u, w))

# calculates the shortest path
def shortestpath(dp, V, edges):
    
    # initializing all distances as None
    for i in range(V + 1):
        for j in range(V):
            dp[i][j] = None, [j]  # weight of the shortest path that goes from 0 to i, shortest path that goes from 0 to i

    # shortest distance From first vertex
    # to in itself consisting of 0 edges
    dp[0][0] = 0, [0]

    # filling up the dp table
    for i in range(1, V + 1):  # length of the path
        for j in range(V):  # vertex
            for k in range(len(edges[j])):  # incoming edges of vertex
                if (dp[i - 1][edges[j][k].From][0] != None):
                    curr_wt = (dp[i - 1][edges[j][k].From][0] +
                                        edges[j][k].weight)
                    
                    updated_path = dp[i - 1][edges[j][k].From][1] + [j]
                    if (dp[i][j][0] == None):
                        dp[i][j] = curr_wt, updated_path
                    else:
                        dp[i][j] = min(dp[i][j], (curr_wt, updated_path), key=lambda x: x[0])

# Returns minimum value of average
# weight of a cycle in graph.
def minAvgWeight(V, edges):
    dp = [[None] * V for i in range(V + 1)]
    # vector to store edges
    structured_edges = [[] for i in range(V)]

    for iter_edge in edges:
        addedge(structured_edges, *iter_edge)

    shortestpath(dp, V, structured_edges)

    # array to store the avg values
    avg = [None] * V

    # Compute average values for all
    # vertices using weights of
    # shortest paths store in dp.
    for i in range(V):
        if (dp[V][i][0] != None):
            for j in range(V):
                if (dp[j][i][0] != None):
                    fraction = (dp[V][i][0] - dp[j][i][0]) / (V - j)
                    avg[i] = fraction if avg[i] is None else max(avg[i], fraction)

    not_none_avg = [elem for elem in avg if elem is not None]

    if len(not_none_avg) == 0:
        return None, None

    result = min([elem for elem in avg if elem is not None])
    best_vertex = avg.index(result)

    complete_path = dp[V][best_vertex][1]
    
    # print(complete_path)

    # Finding minimum mean cycle, which is any cycle that appears in complete_path
    known = set()
    for i, elem in enumerate(complete_path):
        if elem not in known:
            known.add(elem)
        else:        
            start_cycle = elem
            iter_start = i
            break
    else:
        return None, None
    
    current_index = iter_start - 1
    # There should always be a cycle
    while complete_path[current_index] != start_cycle:
        current_index -= 1
    
    # minimum_cycle = minimum_cycle[::-1]
    minimum_cycle = complete_path[current_index : iter_start + 1]


    return result, minimum_cycle


if __name__ == "__main__":

    # Driver Code
    V = 4

    full_edges = [[0, 1, 1], [0, 2, 10], [1, 2, 3], [2, 3, 2], [3, 1, 0], [3, 0, 8]]

    print(minAvgWeight(V, full_edges))

    # This code is contributed by Pranchalk
