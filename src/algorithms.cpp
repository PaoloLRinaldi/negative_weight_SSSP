#include "algorithms.h"

#include "defs.h"
#include "graph.h"
#include "measurement_tool.h"

#include <cstdlib>
#include <list>
#include <optional>

SSSPAlg toSSSPAlg(std::string const& alg_string) {
    if (alg_string == "NaiveBFM") {
        return SSSPAlg::NaiveBFM;
    } else if (alg_string == "Dijkstra") {
        return SSSPAlg::Dijkstra;
    } else if (alg_string == "BCF") {
        return SSSPAlg::BCF;
    } else if (alg_string == "GOR") {
        return SSSPAlg::GOR;
    } else if (alg_string == "BFCT") {
        return SSSPAlg::BFCT;
    } else if (alg_string == "LazyD") {
        return SSSPAlg::LazyD;
    } else {
        ERROR("Unknown SSSP algorithm string: " << alg_string);
    }
}

std::string to_string(SSSPAlg const& alg) {
    switch (alg) {
        case SSSPAlg::NaiveBFM:
            return "NaiveBFM";
        case SSSPAlg::Dijkstra:
            return "Dijkstra";
        case SSSPAlg::BCF:
            return "BCF";
        case SSSPAlg::GOR:
            return "GOR";
        case SSSPAlg::BFCT:
            return "BFCT";
        case SSSPAlg::LazyD:
            return "LazyD";
    }
    return "Unknown SSSP algorithm";
}

NegCycleAlg toNegCycleAlg(std::string const& alg_string) {
    if (alg_string == "NaiveBFM") {
        return NegCycleAlg::NaiveBFM;
    } else if (alg_string == "BCF") {
        return NegCycleAlg::BCF;
    } else if (alg_string == "LazyDijkstra") {
        return NegCycleAlg::LazyDijkstra;
    } else if (alg_string == "BFCT") {
        return NegCycleAlg::BFCT;
    } else {
        ERROR("Unknown NegCycle algorithm string: " << alg_string);
    }
}

namespace {

std::optional<Distances> NaiveBellmanFordMoore(Graph const& graph, NodeID source) {
    // init
    Distances distances(graph.numberOfNodes(), c::infty);
    distances[source] = 0;

    std::queue<NodeID> q;
    std::vector<bool> in_queue(graph.numberOfNodes(), false);
    std::vector<NodeID> cnt(graph.numberOfNodes(), 0);

    q.push(source);
    in_queue[source] = true;

    while (!q.empty()) {
        NodeID src = q.front();
        q.pop();
        in_queue[src] = false;
        for (const auto& edge : graph.getEdgesOf(src)) {
            auto tentative_d = distances[src] + edge.weight;
            if (tentative_d < distances[edge.target]) {
                distances[edge.target] = tentative_d;
                if (!in_queue[edge.target]) {
                    in_queue[edge.target] = true;
                    q.push(edge.target);
                    cnt[edge.target]++;
                    if (cnt[edge.target] > graph.numberOfNodes()) {
                        return {};
                    }
                }
            }
        }
    }

    return distances;

    // // relax all edges n-1 times
    // for (std::size_t round = 0; round < graph.numberOfNodes() - 1; ++round) {
    //     for (NodeID src = 0; src < graph.numberOfNodes(); ++src) {
    //         for (auto const& edge : graph.getEdgesOf(src)) {
    //             if (distances[src] == c::infty) {
    //                 continue;
    //             }
    //             auto const new_dist = distances[src] + edge.weight;
    //             distances[edge.target] = std::min(distances[edge.target], new_dist);
    //         }
    //     }
    // }

    // // check for negative cycle
    // for (NodeID src = 0; src < graph.numberOfNodes(); ++src) {
    //     for (auto const& edge : graph.getEdgesOf(src)) {
    //         if (distances[src] == c::infty) {
    //             continue;
    //         }
    //         auto const new_dist = distances[src] + edge.weight;
    //         if (new_dist < distances[edge.target]) {
    //             return {};  // returns empty optional
    //         }
    //     }
    // }

    // return distances;
}

bool NaiveBMFNegCycleDetection(Graph const& graph) {
    // init
    Distances distances(graph.numberOfNodes(), 0);

    // relax all edges n-1 times
    for (NodeID round = 0; round < graph.numberOfNodes() - 1; ++round) {
        for (NodeID src = 0; src < graph.numberOfNodes(); ++src) {
            for (auto const& edge : graph.getEdgesOf(src)) {
                auto const new_dist = distances[src] + edge.weight;
                distances[edge.target] = std::min(distances[edge.target], new_dist);
            }
        }
    }

    // check for negative cycle
    for (NodeID src = 0; src < graph.numberOfNodes(); ++src) {
        for (auto const& edge : graph.getEdgesOf(src)) {
            auto const new_dist = distances[src] + edge.weight;
            if (new_dist < distances[edge.target]) {
                return true;
            }
        }
    }

    return false;
}

}  // anonymous namespace

std::optional<Distances> BCF(Graph& graph, NodeID source) {
    // compute potential
    auto alg = bcf::SSSPAlg();
    const NodeID n = graph.numberOfNodes();

    MEASUREMENT::start(EXP::JUST_POTENTIAL);
    Distances potential(n);
    if constexpr (true) {
        // First dividing in connected components, then running the algorithm on each instance,
        // and finally fixing the potential of the graphs
        MEASUREMENT::start(EXP::GLOBAL_DECOMP);
        auto components = decomposeIntoSCCs(graph);
        MEASUREMENT::stop(EXP::GLOBAL_DECOMP);
        MEASUREMENT::start(EXP::GLOBAL_RECURS);
        for (auto& component : components) {
            MEASUREMENT::start(EXP::LOCAL_ALG);
            auto opt_component_potential = config::init_kappa == 0 ? alg.runMainAlg(component, component.numberOfNodes()) : alg.runMainAlg(component, c::infty - 5);  // -5 becasue during the labeling we add 3
            if (!opt_component_potential.has_value()) {
                MEASUREMENT::stop(EXP::JUST_POTENTIAL);
                MEASUREMENT::stop(EXP::GLOBAL_RECURS);
                MEASUREMENT::stop(EXP::LOCAL_ALG);
                return {};
            }
            auto component_potential = std::move(opt_component_potential.value());
            MEASUREMENT::stop(EXP::LOCAL_ALG);
            MEASUREMENT::start(EXP::LOCAL_POT);
            for (NodeID i = 0; i < component.numberOfNodes(); i++) {
                potential[component.global_id[i]] = component_potential[i];
            }
            MEASUREMENT::stop(EXP::LOCAL_POT);
        }
        MEASUREMENT::stop(EXP::GLOBAL_RECURS);
        MEASUREMENT::start(EXP::GLOBAL_FIX);
        bcf::fixDagEdges(graph, components, potential);
        MEASUREMENT::stop(EXP::GLOBAL_FIX);
    } else {
        auto direct_potential = alg.runMainAlg(graph, n);
        if (!direct_potential.has_value()){
            MEASUREMENT::stop(EXP::JUST_POTENTIAL);
            return {};
        }
        potential = std::move(direct_potential.value());
    }
    MEASUREMENT::stop(EXP::JUST_POTENTIAL);

    MEASUREMENT::start(EXP::DIJKSTRA);
    // run Dijkstra
    Distances distances(n, c::infty);
    distances[source] = 0;
    AddressableKHeap<4, NodeID, Distance> q(n);
    q.insert(source, 0);

    while (!q.empty()) {
        Distance dist;
        NodeID from;
        q.deleteMin(from, dist);
        if (dist > distances[from]) continue;
        for (auto const& edge : graph.getEdgesOf(from)) {
            // TODO: rewrite edges to avoid this
            auto pot_edge_w = edge.weight + potential[from] - potential[edge.target];
            assert(pot_edge_w >= 0);
            auto tentative_dist = distances[from] + pot_edge_w;
            if (tentative_dist < distances[edge.target]) {
                distances[edge.target] = tentative_dist;
                // if (q.contains(edge.target)) {
                    // q.decreaseKey(edge.target, tentative_dist);
                // } else {
                    // q.insert(edge.target, tentative_dist);
                // }
                q.insert(edge.target, tentative_dist);
            }
        }
    }
    MEASUREMENT::stop(EXP::DIJKSTRA);
    // set all nodes and edges to active
    graph.restoreGraph();
    for (NodeID i = 0; i < n; i++) {
        if (distances[i] != c::infty)
            distances[i] = distances[i] + potential[i] - potential[source];
    }
    return distances;
}

// Algorithm as described in Section 5.1 of CGGTW paper.
std::optional<Distances> BFCT(Graph const& graph, NodeID source) {
    class ShopaTree
    {
        NodeID source;
        Distances& distances;
        std::vector<NodeID> parent;

        using PreOrder = std::list<NodeID>;
        PreOrder pre_order;
        // either iterator to corresponding element in pre_order or pre_order.end()
        std::vector<PreOrder::iterator> to_pre_order;

    public:
        ShopaTree(Distances& distances, NodeID source)
            : distances(distances), source(source), parent(distances.size(), c::no_id)
            , to_pre_order(distances.size(), pre_order.end())
        {
            to_pre_order[source] = pre_order.insert(pre_order.end(), source);
        }

        bool update(NodeID src, NodeID tgt, Distance new_dist)
        {
            if (this->contains(tgt)) {
                bool found_negative_cycle = disassemble(src, tgt, distances[tgt] - new_dist);
                if (found_negative_cycle) {
                    return true;
                }
            }

            addChild(src, tgt, new_dist);
            return false;
        }

        bool contains(NodeID node_id)
        {
            return node_id == source || parent[node_id] != c::no_id;
        }

    private:
        // returns "true" iff a negative cycle was found
        bool disassemble(NodeID forbidden_id, NodeID root_id, Distance diff)
        {
            if (root_id == source) {
                return true; // found shorter path to source -> negative cycle
            }

            std::vector<NodeID> seen_stack = { root_id };

            assert(to_pre_order[root_id] != pre_order.end());
            auto const root_it = to_pre_order[root_id];

            // delete subtree at root
            auto it = std::next(root_it);
            while (it != pre_order.end()) {
                while (!seen_stack.empty() && seen_stack.back() != parent[*it]) {
                    seen_stack.pop_back();
                }

                if (seen_stack.empty()) { break; } // done with disassembly
                if (*it == forbidden_id) { return true; } // negative cycle!

                seen_stack.push_back(*it);
                distances[*it] -= diff-1; // ensures that vertices have to be scanned again

                assert(parent[*it] != c::no_id);
                parent[*it] = c::no_id;

                assert(to_pre_order[*it] != pre_order.end());
                to_pre_order[*it] = pre_order.end();
                it = pre_order.erase(it); // next element in pre_order
            }

            // delete root
            assert(parent[*root_it] != c::no_id);
            parent[*root_it] = c::no_id;
            to_pre_order[*root_it] = pre_order.end();
            pre_order.erase(root_it);

            return false;
        }

        // Important: Always first disassemble, then add new child!
        void addChild(NodeID src, NodeID tgt, Distance new_dist)
        {
            distances[tgt] = new_dist;

            assert(parent[tgt] == c::no_id);
            parent[tgt] = src;

            assert(to_pre_order[src] != pre_order.end());
            to_pre_order[tgt] = pre_order.insert(std::next(to_pre_order[src]), tgt);
        }
    };

    Distances distances(graph.numberOfNodes(), c::infty);
    std::queue<NodeID> q;
    std::vector<bool> in_queue(graph.numberOfNodes(), false);
    ShopaTree tree(distances, source);

    distances[source] = 0;
    q.push(source);
    in_queue[source] = true;

    while (!q.empty()) {
        int node_id = q.front();
        q.pop();
        in_queue[node_id] = false;

        if (!tree.contains(node_id)) {
            continue; // ignore as it was removed in subtree disassembly
        }

        for (const auto& edge : graph.getEdgesOf(node_id)) {
            auto const tentative_d = distances[node_id] + edge.weight;
            if (tentative_d < distances[edge.target]) {
                bool neg_cycle = tree.update(node_id, edge.target, tentative_d);
                if (neg_cycle) { return {}; }

                if (!in_queue[edge.target]) {
                    q.push(edge.target);
                    in_queue[edge.target] = true;
                }
            }
        }
    }

    return distances;
}

std::optional<Distances> computeSSSP(
    SSSPAlg algorithm, Graph& graph, NodeID source) {
    switch (algorithm) {
        case SSSPAlg::NaiveBFM:
            return NaiveBellmanFordMoore(graph, source);
        case SSSPAlg::Dijkstra:
            return bcf::runDijkstra(graph, source);
        case SSSPAlg::BCF:
            return BCF(graph, source);
        case SSSPAlg::GOR:
            return gor(graph, source);
        case SSSPAlg::BFCT:
            return BFCT(graph, source);
        case SSSPAlg::LazyD:
            return bcf::runLazyDijkstra(graph, source);
        default:
            ERROR("Unknown algorithm.");
    };
}

bool negCycleDetection(NegCycleAlg algorithm, Graph& graph) {
    switch (algorithm) {
        case NegCycleAlg::NaiveBFM:
            return NaiveBMFNegCycleDetection(graph);
        case NegCycleAlg::BCF: // FIXME: this does only report negative cycles reachable from node 0
            return !BCF(graph, 0).has_value();
        case NegCycleAlg::LazyDijkstra:
            return !bcf::runLazyDijkstra(graph, Distances(), Orientation::OUT).has_value();
        case NegCycleAlg::BFCT: // FIXME: this does only report negative cycles reachable from node 0
            return !BFCT(graph, 0).has_value();
        default:
            ERROR("Unknown algorithm.");
    };
}

bool isResultCorrect(Graph const& graph, Distances const& distances, NodeID source) {
    // For every incoming edge of v, we chack that dist(v) <= dist(u) + w(u, v),
    // and we check that for at least one incoming edge of v, dist(v) == dist(u) + w(u, v).
    // If the node has no incoming edges, we check that dist(v) == inf.

    const bool constexpr verbose = false;

    if (distances.size() != graph.numberOfNodes()) {
        if constexpr (verbose) {
            std::clog << "distances.size() != graph.numberOfNodes()" << std::endl;
        }
        return false;
    }
    if (distances[source] != 0) {
        if constexpr (verbose) {
            std::clog << "distances[source] != 0" << std::endl;
        }
        return false;
    }
    for (NodeID v = 0; v < graph.numberOfNodes(); v++) {
        const auto& range_incoming_edges = graph.getEdgesOf(v, Orientation::IN);
        if (range_incoming_edges.empty()) {
            if (v != source && distances[v] != c::infty) {
                if constexpr (verbose) {
                    std::clog << "v != source && distances[v] != c::infty" << std::endl;
                }
                return false;
            }
            continue;
        }

        bool one_is_equal = false;  // It becomes true if we find at least one edge with dist(v) == dist(u) + w(u, v)
        for (const auto& edge : range_incoming_edges) {
            NodeID u = edge.target;  // The orientation is inverted, that's why we use the target here
            Distance w = edge.weight;

            // Basically we want to check that dist(v) <= dist(u) + w(u, v)

            if (distances[v] == c::infty) {
                if (distances[u] == c::infty) {
                    one_is_equal = true;
                    continue;
                }

                if constexpr (verbose) {
                    std::clog << "Infinity. u: " << u << " v: " << v << " distances[u]: " << distances[u] << std::endl;
                }
                return false;
            } else if (distances[u] == c::infty) {
                // There are apparently other ways to reach v. But surely not from u.
                continue;
            }
            
            if (distances[v] > distances[u] + w) {
                if constexpr (verbose) {
                    std::clog << "Not less. u: " << u << " distances[u]: " << distances[u] << " distances[v]: " << distances[v] << " v: " << v << " w: " << w << std::endl;
                }
                return false;
            }

            if (distances[v] == distances[u] + w) {
                one_is_equal = true;
            }
        }

        if (!one_is_equal && v != source) {
            if constexpr (verbose) {
                std::clog << "Not equal. v: " << v << std::endl;
            }
            return false;
        }
    }
    return true;
}