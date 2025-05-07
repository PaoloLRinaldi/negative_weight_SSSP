#pragma once

#include <limits>
#include <queue>
#include <span>
#include <string>
#include <tuple>
#include <vector>
#include <cstdint>
#include <iostream>
#include "config.h"

namespace unit_tests {
void testGraph();
}

// basic types

using NodeID = std::size_t;
using EdgeID = std::size_t;
// using Distance = int;
using Distance = std::int64_t;

using Distances = std::vector<Distance>;

// Note: We can add "source" to this at any time,
// but it is more cache efficient this way.
struct Edge {
    NodeID target;
    Distance weight;
    bool active;
    Edge(NodeID t, Distance w) : target(t), weight(w), active(true) {}
};
using Edges = std::vector<Edge>;

using EdgeRange = std::span<Edge const>;

// head, tail, weight
using FullEdge = std::tuple<NodeID, NodeID, Distance>;

namespace c {
    Distance const infty = std::numeric_limits<Distance>::max();
    NodeID const no_id = std::numeric_limits<NodeID>::max();
}

enum class Orientation {
    OUT,
    IN,
};

// Graph

class Graph {
    //    private:
   private:
    NodeID number_of_nodes;

    Edges edges;
    Edges edges_rev;

   public:
   // TODO: make these private
    std::vector<EdgeID> offsets;
    std::vector<EdgeID> offsets_rev;


    // TODO: this extra info should be passed as a template or something similar
    // edges_rev[idx_in_reversed[i]] is the reversed version of edges[i]
    // edges[idx_in_original[i]] is the reversed version of edges_rev[i]
    std::vector<EdgeID> idx_in_reversed;
    std::vector<EdgeID> idx_in_original;

    std::vector<bool> is_vtx_active;
    // id in supergraph
    std::vector<NodeID> global_id;

    Graph() = default;
    Graph(NodeID n, std::vector<FullEdge>& e, const std::vector<NodeID>& new_global_id = {});
    Graph(NodeID n,
          Edges&& e,
          Edges&& e_rev,
          std::vector<EdgeID>&& offs,
          std::vector<EdgeID>&& offs_rev,
          std::vector<EdgeID>&& idx_in_rev,
          std::vector<EdgeID>&& idx_in_orig,
          std::vector<bool>&& is_active,
          std::vector<NodeID>&& global_id)
    : number_of_nodes(n),
      edges(std::move(e)),
      edges_rev(std::move(e_rev)),
      offsets(std::move(offs)),
      offsets_rev(std::move(offs_rev)),
      idx_in_reversed(std::move(idx_in_rev)),
      idx_in_original(std::move(idx_in_orig)),
      is_vtx_active(std::move(is_active)),
      global_id(std::move(global_id)) {
}

    NodeID numberOfNodes() const;
    EdgeID numberOfEdges() const;

    EdgeRange getEdgesOf(NodeID source) const;
    EdgeRange getEdgesOf(NodeID source, Orientation orientation) const;

    EdgeRange getEdges(Orientation orientation = Orientation::OUT) const;

    void print() const;
    void format_print(std::ostream& out = std::cout, bool check_active = true) const;

    void killOutEdge(EdgeID idx);
    void killInEdge(EdgeID idx);

    void restoreGraph();


    friend Graph readGraph(std::string const& filename);
    friend std::vector<Graph> decomposeIntoSCCs(const Graph& graph);
};

// graph file parser
Graph readGraph(std::string const& filename);

std::vector<NodeID> getTopologicalSort(const Graph& graph, Orientation orientation = Orientation::OUT);

std::vector<Graph> decomposeIntoSCCs(const Graph& graph);

Distances readDistancesFromFile(const std::string& filename);