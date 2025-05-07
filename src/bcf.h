#pragma once

#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>
#include <optional>
#include <random>
#include <stack>
#include <tuple>
#include <unordered_set>
#include <vector>
#include <variant>

#include "graph.h"
#include "heap.h"
#include "permutations.h"
#include "gor.h"
#include "config.h"

namespace bcf {

using GraphHeap = AddressableKHeap<4, NodeID, Distance>;

// encapsulates the main algorithm
// TODO: organize this better
class SSSPAlg {
   public:
    SSSPAlg() {}

    // void cutEdges(GraphContainer& cont, int kappa, const int seed);
    void cutEdges(Graph& cont, Distance kappa, const int seed);
    // void cutEdges(Graph& cont, NodeID kappa, const int seed);
    void cutEdges2(Graph& graph, Distance kappa, int seed);
    void cutEdgesInHalf(Graph& graph);
    void cutEdges3(Graph& graph, Distance kappa, int seed);

    std::unordered_set<NodeID> getBallAround(const Graph& graph, NodeID source, Distance radius,
                                             Distances& current_distances, GraphHeap& q, Orientation orientation);
    template <Orientation orientation>
    void getBallAround2(Graph& graph, NodeID source, Distance radius,
                                 Distances& current_distances,
                                 bcf::GraphHeap& q);
    std::pair<std::vector<char>, std::vector<NodeID>> getBallAround3(const Graph& graph, NodeID source, Distance radius,
                                             Distances& current_distances, GraphHeap& q, Orientation orientation, std::vector<unsigned char>& balls_covered, unsigned char& current_ball);

    std::optional<Distances> runMainAlg(Graph& subgraph, Distance kappa, int level = 0);
    // std::optional<Distances> runMainAlg(Graph& subgraph, NodeID kappa, int level = 0);

    std::vector<NodeID> labelInLight(const Graph& graph, Distance radius, const int seed, Orientation orientation = Orientation::OUT);
};

Distances runDijkstra(const Graph& graph, NodeID source, Distance dist_bound = c::infty, Orientation orientation = Orientation::OUT);

std::optional<Distances> runLazyDijkstra(const Graph& graph, const Distances& potential, Orientation orientation = Orientation::OUT);
std::optional<Distances> runLazyDijkstra(const Graph& graph, NodeID source, Orientation orientation = Orientation::OUT);

void fixDagEdges(const Graph& graph, const std::vector<Graph>& components, Distances& potential);
void fixDagEdges2(const Graph& graph, const std::vector<Graph>& components, Distances& potential);

class RunDijkstra {
    public:
    RunDijkstra() = delete;
    RunDijkstra(NodeID number_of_nodes) : q(number_of_nodes), distances(number_of_nodes, c::infty) {}
    const Distances& operator()(const Graph& graph, NodeID source, Distance dist_bound = c::infty, Orientation orientation = Orientation::OUT);

    private:
    Distances distances;
    GraphHeap q;

};

}  // namespace bcf