#include "graph.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <numeric>
#include <sstream>
#include <tuple>
#include <stack>

#include "defs.h"
#include "permutations.h"

NodeID Graph::numberOfNodes() const { return number_of_nodes; }

EdgeID Graph::numberOfEdges() const { return edges.size(); }  // What about inactive edges?

void Graph::killOutEdge(EdgeID idx) {
    edges[idx].active = false;
    edges_rev[idx_in_reversed[idx]].active = false;
}

void Graph::killInEdge(EdgeID idx) {
    edges_rev[idx].active = false;
    edges[idx_in_original[idx]].active = false;
}

EdgeRange Graph::getEdges(Orientation orientation) const {
    // if (orientation)
    return orientation == Orientation::OUT ? EdgeRange(&edges[0], edges.size()) : EdgeRange(&edges_rev[0], edges_rev.size());
}

EdgeRange Graph::getEdgesOf(NodeID source, Orientation orientation) const {
    if (orientation == Orientation::OUT) {
        return EdgeRange(&edges[offsets[source]],
                         offsets[source + 1] - offsets[source]);
    } else {
        return EdgeRange(&edges_rev[offsets_rev[source]],
                         offsets_rev[source + 1] - offsets_rev[source]);
    }
}

EdgeRange Graph::getEdgesOf(NodeID source) const {
    return EdgeRange(&edges[offsets[source]],
                     offsets[source + 1] - offsets[source]);
}

void Graph::print() const {
    NodeID number_of_edges = 0;
    for (auto const& edge : edges)
        if (edge.active) number_of_edges++;

    std::cout << "n = " << number_of_nodes << ", m = " << number_of_edges << std::endl;
    for (NodeID v = 0; v < number_of_nodes; v++) {
        for (auto e : getEdgesOf(v)) {
            if (!e.active) continue;
            std::cout << v << " -> " << e.target << " (" << e.weight << ")" << std::endl;
        }
    }
}

void Graph::format_print(std::ostream& out, bool check_active) const {
    NodeID number_of_edges = 0;

    out << number_of_nodes << std::endl;
    for (NodeID v = 0; v < number_of_nodes; v++) {
        for (auto e : getEdgesOf(v)) {
            if (check_active && !e.active) continue;
            out << v << " " << e.target << " " << e.weight << std::endl;
        }
    }
}


void Graph::restoreGraph() {
    for (NodeID i = 0; i < number_of_nodes; i++) {
        is_vtx_active[i] = true;
    }
    for (EdgeID i = 0; i < edges.size(); i++) {
        edges[i].active = true;
        edges_rev[i].active = true;
    }
}

Graph::Graph(NodeID n, std::vector<FullEdge>& e, const std::vector<NodeID>& new_global_id) : number_of_nodes(n), is_vtx_active(n, true), offsets(n + 1, 0), offsets_rev(n + 1, 0), global_id(new_global_id) {

    // assemble graph in linear time (instead of n log(n))

    // Before sorting
    for (auto const& [source, target, weight] : e) {
        offsets[source]++;
        offsets_rev[target]++;
    }
    std::exclusive_scan(offsets.begin(), offsets.end(), offsets.begin(), 0);
    std::exclusive_scan(offsets_rev.begin(), offsets_rev.end(), offsets_rev.begin(), 0);

    // Sorting
    // Basically I iterate over the edges once. For each edge I check if the current edge
    // is in the right place (I know it thanks to offsets). If not I swap it and put it
    // in the right place (I know it thanks to offsets2).

    auto sort_edges = [&]<unsigned int T>(const std::vector<EdgeID>& off) {
        std::vector<EdgeID> off2 = off;
        // auto perm_edges = bcf::getIdentityPermutation<EdgeID>(e.size());
        for (NodeID i = 0; i < number_of_nodes; i++) {
            for (EdgeID arc = off[i]; arc != off[i + 1]; arc++) {
                const NodeID &current_source = std::get<T>(e[arc]);
                while (current_source != i) {  // Is the current edge in the right place?
                    EdgeID arc_new = off2[current_source]++;  // Where to put the current edge

                    std::swap(e[arc], e[arc_new]);  // The new current edge becomes the one in arc_new
                    if constexpr (T == 1) {
                        std::swap(idx_in_original[arc], idx_in_original[arc_new]);
                    }
                }
            }
            // std::sort(perm_edges.begin() + off[i], perm_edges.begin() + off[i + 1], [&e](const EdgeID& a, const EdgeID& b) { return std::get<2>(e[a]) < std::get<2>(e[b]); });
            // for (EdgeID j = off[i]; j != off[i + 1]; j++) {
            //     std::swap(e[j], e[perm_edges[j]]);
            //     if constexpr (T == 1) {
            //         std::swap(idx_in_original[j], idx_in_original[perm_edges[j]]);
            //     }
            // }
        }
    };

    sort_edges.operator()<0>(offsets);  // ugly way to call a non-type template lambda

    for (auto const& [source, target, weight] : e) {
        edges.emplace_back(target, weight);  // tgt, weight
    }

    // assemble reverse graph in linear time (instead of n log(n))

    // reversed_graph.edges[i] ~ graph.edges[idx_in_original[i]]
    idx_in_original = bcf::getIdentityPermutation<EdgeID>(edges.size());
    sort_edges.operator()<1>(offsets_rev);
    // graph.edges[i] ~ reversed_graph.edges[idx_in_reversed[i]]
    idx_in_reversed = bcf::getInversePermutation(idx_in_original);

    for (auto const& [target, source, weight] : e) {
        edges_rev.emplace_back(target, weight);  // tgt, weight
    }

}

// File format: first line is number of nodes, then all edges in form:
// <source> <target> <weight>
// Assumes nodes to have ids from 0 to n-1.
Graph readGraph(std::string const& filename) {
    // buffer file
    std::ifstream file(filename);
    if (!file) {
        ERROR("Could not open file: " << filename);
    }

    std::stringstream ss;
    ss << file.rdbuf();
    auto ignore_count = std::numeric_limits<std::streamsize>::max();

    // read node count
    NodeID number_of_nodes;
    ss >> number_of_nodes;

    // read edges
    using ParserEdge = std::tuple<NodeID, NodeID, Distance>;  // src, tgt, weight
    std::vector<ParserEdge> parser_edges;

    std::string source_str, target_str, weight_str;
    while (ss >> source_str >> target_str >> weight_str) {
        parser_edges.emplace_back(std::stoul(source_str), std::stoul(target_str),
                                  std::stoll(weight_str));
        ss.ignore(ignore_count, '\n');
    }

    return Graph(number_of_nodes, parser_edges);    
}

// computes topological sort by running DFS and recording exiting times
std::vector<NodeID> getTopologicalSort(const Graph& graph, Orientation orientation) {
    const NodeID n = graph.numberOfNodes();
    std::vector<bool> visited(n, false);
    std::vector<NodeID> toposort;
    std::stack<NodeID> stack;

    for (NodeID v = 0; v < n; v++) {
        if (!visited[v]) {
            stack.push(v);

            while (!stack.empty()) {
                NodeID current = stack.top();  // I don't pop it (yet) on purpose

                if (!visited[current]) {
                    visited[current] = true;

                    bool allNeighborsVisited = true;
                    for (auto& edge : graph.getEdgesOf(current, orientation)) {
                        if (!visited[edge.target] && edge.active) {
                            allNeighborsVisited = false;
                            stack.push(edge.target);
                            break;
                        }
                    }

                    if (!allNeighborsVisited) {
                        continue;
                    }

                    toposort.push_back(current);
                }
                stack.pop();

                // This is when I've visited all the children of v, and I want to
                // proceed with its parent p, which was already visited. I say
                // that it's not visited so that when we see it in the next iteration
                // we don't skip it but we proceed with it
                if (!stack.empty()) {
                    visited[stack.top()] = false;
                }
            }
        }
    }

    std::reverse(toposort.begin(), toposort.end());
    return toposort;
}

std::vector<Graph> decomposeIntoSCCs(const Graph& graph) {
    const NodeID n = graph.numberOfNodes();
    auto toposort = getTopologicalSort(graph, Orientation::IN);

    // traverse vtcs in the toposort to label each SCC
    NodeID num_components = 0;
    std::vector<int> traversal_order(n, -1);
    std::vector<NodeID> component_id(n, c::no_id);
    int time = 0;
    for (auto& v : toposort) {
        // if not visited yet, start DFS from v
        if (traversal_order[v] == -1) {
            std::stack<NodeID> s;
            s.push(v);
            traversal_order[v] = time++;
            while (!s.empty()) {
                auto from = s.top();
                s.pop();
                // assign the id of SCC
                component_id[from] = num_components;
                for (auto& edge : graph.getEdgesOf(from)) {
                    // NOTE: checking whether an edge is alive or not is ONLY done here and in getTopologicalSort
                    // TODO: re arrange edges so that we do not have to do this
                    if (!edge.active) continue;
                    if (traversal_order[edge.target] == -1) {
                        traversal_order[edge.target] = time++;
                        s.push(edge.target);
                    }
                }
            }
            num_components++;
        }
    }

    // order_scc[i] is the id of the i-th vertex explored while constructing the SCCs
    // vertices in the same SCC are contiguous in order_scc
    auto order_scc = bcf::getSortPermutation(traversal_order);

    // inv_order_scc is the permutation which applied to the vertices yields order_scc
    // that is, inv_order_scc[i] is the position of the i-th vertex in order_scc
    auto inv_order_scc = bcf::getInversePermutation(order_scc);

    // construct a GraphContainer for each SCC
    // for this we relabel all vertices and edges so that they start from 0
    int vtx_start = 0, vtx_end = 1;
    std::vector<Graph> components;

    std::vector<EdgeID> old_forw_edge_id_to_new_forw_edge_id(graph.edges.size(), 0);
    for (NodeID current_component = 0; current_component < num_components; current_component++) {
        while (vtx_end < n && component_id[order_scc[vtx_end]] == current_component) {
            vtx_end++;
        }

        const NodeID n_comp = vtx_end - vtx_start;
        assert(n_comp >= 1);
        std::vector<NodeID> comp_global_id(n_comp, c::no_id);

        if (config::eg_sort_scc) {
            std::vector<FullEdge> edges_current_component;

            for (NodeID i = vtx_start; i < vtx_end; i++) {
                comp_global_id[i - vtx_start] = order_scc[i];
                for (auto& edge : graph.getEdgesOf(order_scc[i])) {
                    if (!edge.active) continue;
                    if (component_id[edge.target] == current_component) {
                        NodeID comp_source = i - vtx_start;
                        NodeID comp_target = inv_order_scc[edge.target] - vtx_start;
                        // std::cout << order_scc[i] << " -> " << edge.target << "  aka  " << i - vtx_start << " -> " << inv_order_scc[edge.target] - vtx_start << std::endl;
                        edges_current_component.emplace_back(comp_source, comp_target, edge.weight);
                    }
                }
            }

            for (auto const& [source, target, weight] : edges_current_component) {
                assert(source < n && target < n);
            }

            components.emplace_back(n_comp, edges_current_component, comp_global_id);
        } else {
            Edges comp_edges, comp_edges_rev;
            std::vector<EdgeID> comp_offsets(n_comp + 1, 0), comp_offsets_rev(n_comp + 1, 0);
            std::vector<EdgeID> comp_idx_in_reversed, comp_idx_in_original;
            std::vector<bool> comp_is_vtx_active(n_comp, true);

            EdgeID n_comp_edges = 0;
            for (NodeID i = vtx_start; i < vtx_end; i++) {
                comp_global_id[i - vtx_start] = order_scc[i];
                const auto graph_offsets_end = graph.offsets[order_scc[i] + 1];
                for (EdgeID edge_id = graph.offsets[order_scc[i]]; edge_id < graph_offsets_end; edge_id++) {
                    const Edge &edge = graph.edges[edge_id];
                    if (!edge.active) continue;
                    if (component_id[edge.target] == current_component) {
                        const NodeID comp_source = i - vtx_start;
                        const NodeID comp_target = inv_order_scc[edge.target] - vtx_start;
                        // std::cout << order_scc[i] << " -> " << edge.target << "  aka  " << i - vtx_start << " -> " << inv_order_scc[edge.target] - vtx_start << std::endl;
                        comp_edges.emplace_back(comp_target, edge.weight);
                        comp_offsets[comp_source]++;
                        old_forw_edge_id_to_new_forw_edge_id[edge_id] = n_comp_edges++;
                    }
                }
            }

            comp_edges_rev.reserve(comp_edges.size());
            comp_idx_in_original.reserve(comp_edges.size());

            for (NodeID i = vtx_start; i < vtx_end; i++) {
                const auto graph_offsets_rev_end = graph.offsets_rev[order_scc[i] + 1];
                for (EdgeID edge_rev_id = graph.offsets_rev[order_scc[i]]; edge_rev_id < graph_offsets_rev_end; edge_rev_id++) {
                    const Edge &edge_rev = graph.edges_rev[edge_rev_id];
                    if (!edge_rev.active) continue;
                    if (component_id[edge_rev.target] == current_component) {
                        const NodeID comp_source_rev = i - vtx_start;
                        const NodeID comp_target_rev = inv_order_scc[edge_rev.target] - vtx_start;
                        comp_edges_rev.emplace_back(comp_target_rev, edge_rev.weight);
                        comp_offsets_rev[comp_source_rev]++;
                        comp_idx_in_original.emplace_back(old_forw_edge_id_to_new_forw_edge_id[graph.idx_in_original[edge_rev_id]]);
                    }
                }
            }

            std::exclusive_scan(comp_offsets.begin(), comp_offsets.end(), comp_offsets.begin(), 0);
            std::exclusive_scan(comp_offsets_rev.begin(), comp_offsets_rev.end(), comp_offsets_rev.begin(), 0);
            comp_idx_in_reversed = bcf::getInversePermutation(comp_idx_in_original);

            components.emplace_back(n_comp,
                                    std::move(comp_edges),
                                    std::move(comp_edges_rev),
                                    std::move(comp_offsets),
                                    std::move(comp_offsets_rev),
                                    std::move(comp_idx_in_reversed),
                                    std::move(comp_idx_in_original),
                                    std::move(comp_is_vtx_active),
                                    std::move(comp_global_id));

        }

        vtx_start = vtx_end;
        vtx_end++;
    }
    std::reverse(components.begin(), components.end());
    return components;
}



Distances readDistancesFromFile(const std::string& filename) {
    Distances distances;
    std::ifstream file(filename);
    std::string token;

    if (!file.is_open()) {
        std::cerr << "Could not open the file - '" << filename << "'" << std::endl;
        return distances; // Return an empty vector if the file can't be opened
    }

    while (file >> token) {
        if (token == "inf") {
            distances.push_back(c::infty); // Store the value 70 for "inf"
        } else {
            std::istringstream iss(token);
            Distance num;
            if (iss >> num) {
                distances.push_back(num);
            } else {
                std::cerr << "Invalid number found in the file: " << token << std::endl;
                // Handle the error or ignore the invalid number
            }
        }
    }

    file.close();
    return distances;
}