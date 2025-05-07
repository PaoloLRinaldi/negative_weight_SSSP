#include "bcf.h"

#include "defs.h"
#include <bit>
#include <cmath>
#define MEASURE
#include "measurement_tool.h"
#include <fstream>

// runs dijkstra from the source with the given radious bound on the graph G_{>= 0} (!!)
const Distances& bcf::RunDijkstra::operator()(const Graph& graph, NodeID source, Distance dist_bound, Orientation orientation) {
    // initialize distances
    distances.assign(graph.numberOfNodes(), c::infty);
    distances[source] = 0;

    // initialize priority queue
    // bcf::GraphHeap q(graph.numberOfNodes());
    q.insert(source, 0);

    while (!q.empty()) {
        Distance dist;
        NodeID from;
        q.deleteMin(from, dist);
        if (dist > distances[from]) continue;
        for (auto const& edge : graph.getEdgesOf(from, orientation)) {
            // NOTE: here we bump negative edges to 0
            auto tentative_dist = distances[from] + std::max(static_cast<Distance>(0), edge.weight);
            if (tentative_dist <= dist_bound) {
                if (tentative_dist < distances[edge.target]) {
                    distances[edge.target] = tentative_dist;
                    // if (q.contains(edge.target)) {
                    //     q.decreaseKey(edge.target, tentative_dist);
                    // } else {
                    //     q.insert(edge.target, tentative_dist);
                    // }
                    q.insert(edge.target, tentative_dist);
                }
            } else {
                // break;
            }
        }
    }

    return distances;
}

Distances bcf::runDijkstra(const Graph& graph, NodeID source, Distance dist_bound, Orientation orientation) {
    // return RunDijkstra()(graph, source, dist_bound, orientation);

    // initialize distances
    Distances distances(graph.numberOfNodes(), c::infty);
    distances[source] = 0;

    // initialize priority queue
    bcf::GraphHeap q(graph.numberOfNodes());
    q.insert(source, 0);

    while (!q.empty()) {
        Distance dist;
        NodeID from;
        q.deleteMin(from, dist);
        if (dist > distances[from]) continue;
        for (auto const& edge : graph.getEdgesOf(from, orientation)) {
            // NOTE: here we bump negative edges to 0
            auto tentative_dist = distances[from] + std::max(static_cast<Distance>(0), edge.weight);
            if (tentative_dist < distances[edge.target] && tentative_dist <= dist_bound) {
                distances[edge.target] = tentative_dist;
                // if (q.contains(edge.target)) {
                //     q.decreaseKey(edge.target, tentative_dist);
                // } else {
                //     q.insert(edge.target, tentative_dist);
                // }
                q.insert(edge.target, tentative_dist);
            }
        }
    }

    return distances;
}

// runs lazy dijkstra from a virtual source connected to every other vtx with 0-edge weights
// the weights of the graph are adjusted by this potential
std::optional<Distances> runLazyDijkstra(const Graph& graph, std::variant<NodeID, const Distances*> source_pot, Orientation orientation) {
    // initialize distances
    Distances distances(graph.numberOfNodes(), c::infty);
    const Distances *potential;
    bool using_potential = std::holds_alternative<const Distances*>(source_pot);
    std::vector<NodeID> cnt(graph.numberOfNodes(), 0);  // counts the number of updates for each node

    // initialize priority queue
    bcf::GraphHeap q(graph.numberOfNodes());

    if (using_potential) {
        potential = std::get<const Distances*>(source_pot);
        for (NodeID i = 0; i < graph.numberOfNodes(); i++) {
            distances[i] = -(*potential)[i];
            q.insert(i, distances[i]);
        }
    } else {
        NodeID source = std::get<NodeID>(source_pot);
        distances[source] = 0;
        q.insert(source, 0);
    }

    int rounds = 0;

    while (!q.empty()) {
        rounds++;
        std::vector<NodeID> bellman_phase;

        // Run dijkstra phase on edges with w(e) >= 0
        while (!q.empty()) {
            Distance dist;
            NodeID from;
            q.deleteMin(from, dist);
            // assert(dist == distances[from]);

            if (dist > distances[from]) continue;

            bellman_phase.emplace_back(from);

            for (auto const& edge : graph.getEdgesOf(from, orientation)) {
                auto pot_edge_w = edge.weight;
                if (using_potential) {
                    pot_edge_w += (*potential)[from] - (*potential)[edge.target];
                }

                // TODO: we can re arrange edges so that we get all non-negative first, and then negative
                // In this way Dijkstra and Bellman phase access contiguous parts of these arrays
                if (pot_edge_w < 0) {
                    // only killed edges can be negative
                    // assert(!potential || edge.active == false);
                    continue;
                }
                auto tentative_dist = distances[from] + pot_edge_w;
                if (tentative_dist < distances[edge.target]) {
                    distances[edge.target] = tentative_dist;
                    q.insert(edge.target, tentative_dist);
                    cnt[edge.target]++;
                    if (cnt[edge.target] > graph.numberOfNodes()) {
                        return {};
                    }

                }
            }
        }

        // Relax all negative edges from vertices explored in dijkstra phase
        for (auto const& from : bellman_phase) {
            for (auto const& edge : graph.getEdgesOf(from, orientation)) {
                auto pot_edge_w = edge.weight;
                if (using_potential) {
                    pot_edge_w += (*potential)[from] - (*potential)[edge.target];
                }
                if (pot_edge_w >= 0) continue;
                auto tentative_dist = distances[from] + pot_edge_w;
                if (tentative_dist < distances[edge.target]) {
                    distances[edge.target] = tentative_dist;
                    q.insert(edge.target, tentative_dist);
                    cnt[edge.target]++;
                    if (cnt[edge.target] > graph.numberOfNodes()) {
                        return {};
                    }
                }
            }
        }
    }
    // PRINT("rounds of lazy dijkstra: " << rounds);
    return distances;
}

std::optional<Distances> bcf::runLazyDijkstra(const Graph& graph, const Distances& potential, Orientation orientation) {
    return runLazyDijkstra(graph, std::variant<NodeID, const Distances*>(&potential), orientation);
}

std::optional<Distances> bcf::runLazyDijkstra(const Graph& graph, NodeID source, Orientation orientation) {
    return runLazyDijkstra(graph, std::variant<NodeID, const Distances*>(source), orientation);
}

void bcf::SSSPAlg::cutEdges(Graph& graph, Distance kappa, int seed) {
// void bcf::SSSPAlg::cutEdges(Graph& graph, NodeID kappa, int seed) {
    // initialize geometric distribution
    std::mt19937 gen(seed);
    const NodeID n = graph.numberOfNodes();
    const double p = std::max(0.0001, std::min(0.9999, 20.0 * log(static_cast<double>(n)) / static_cast<double>(kappa)));
    std::geometric_distribution<int> geom_dist(p);

    auto out_light = labelInLight(graph, (kappa + 4 - 1) / 4, config::ol_seed, Orientation::IN);
    auto in_light = labelInLight(graph, (kappa + 4 - 1) / 4, config::il_seed, Orientation::OUT);
    PRINT("num of out light vtcs: " << out_light.size() << " wrt radius = " << (kappa + 4 - 1) / 4);
    PRINT("num of in light vtcs: " << in_light.size() << " wrt radius = " << (kappa + 4 - 1) / 4);
    PRINT("avg sample = 1/p = " << 1. / p << "; fresh sample = " << geom_dist(gen));

    // cutting out balls
    Distances distances(n, c::infty);
    // initializing queue here instead of inside getBallAround is important (to avoid re initializing it each time)
    bcf::GraphHeap q(n);
    EdgeID num_cut = 0, neg_cut = 0;
    for (auto v : out_light) {
        if (graph.is_vtx_active[v]) {
            const Distance sampled_radius = geom_dist(gen);
            assert(sampled_radius >= 0);
            auto ball = getBallAround(graph, v, sampled_radius, distances, q, Orientation::OUT);
            for (auto w : ball) {
                // removing w from the graph
                graph.is_vtx_active[w] = false;
                // cutting boundary of the ball
                for (EdgeID i = graph.offsets[w]; i < graph.offsets[w + 1]; i++) {
                    auto const e = graph.getEdges(Orientation::OUT)[i];
                    // if the tail is not in the ball, and it has not been carved before: delete edge
                    if (!ball.contains(e.target) && graph.is_vtx_active[e.target]) {
                        num_cut++;
                        if (e.weight < 0) neg_cut++;
                        // Note that now this method is handling the deletion of the corresponding flipped edge
                        graph.killOutEdge(i);
                    }
                }
            }
        }
    }

    // cutting in balls
    q.clear();
    distances.assign(n, c::infty);
    for (auto v : in_light) {
        if (graph.is_vtx_active[v]) {
            const Distance sampled_radius = geom_dist(gen);
            assert(sampled_radius >= 0);
            auto ball = getBallAround(graph, v, sampled_radius, distances, q, Orientation::IN);
            for (auto w : ball) {
                // removing w from the graph
                graph.is_vtx_active[w] = false;
                // cutting boundary of the ball
                for (EdgeID i = graph.offsets_rev[w]; i < graph.offsets_rev[w + 1]; i++) {
                    auto const e = graph.getEdges(Orientation::IN)[i];
                    if (!ball.contains(e.target) && graph.is_vtx_active[e.target]) {
                        num_cut++;
                        if (e.weight < 0) neg_cut++;
                        graph.killInEdge(i);
                    }
                }
            }
        }
    }
    PRINT("cut " << num_cut << " many edges, out of which " << neg_cut << " are negative");
}

// This version is still to be tested. Should be faster than cutEdges.
void bcf::SSSPAlg::cutEdges2(Graph& graph, Distance kappa, int seed) {
// void bcf::SSSPAlg::cutEdges(Graph& graph, NodeID kappa, int seed) {
    // initialize geometric distribution
    std::mt19937 gen(seed);
    const NodeID n = graph.numberOfNodes();
    const double p = std::max(0.0001, std::min(0.9999, 20.0 * log(static_cast<double>(n)) / static_cast<double>(kappa)));
    std::geometric_distribution<int> geom_dist(p);

    auto out_light = labelInLight(graph, (kappa + 4 - 1) / 4, config::ol_seed, Orientation::IN);
    auto in_light = labelInLight(graph, (kappa + 4 - 1) / 4, config::il_seed, Orientation::OUT);
    PRINT("num of out light vtcs: " << out_light.size() << " wrt radius = " << (kappa + 4 - 1) / 4);
    PRINT("num of in light vtcs: " << in_light.size() << " wrt radius = " << (kappa + 4 - 1) / 4);
    PRINT("avg sample = 1/p = " << 1. / p << "; fresh sample = " << geom_dist(gen));

    // cutting out balls
    Distances distances(n, c::infty);
    // initializing queue here instead of inside getBallAround is important (to avoid re initializing it each time)
    bcf::GraphHeap q(n);
    EdgeID num_cut = 0, neg_cut = 0;
    for (auto v : out_light) {
        if (graph.is_vtx_active[v]) {
            const Distance sampled_radius = geom_dist(gen);
            assert(sampled_radius >= 0);
            getBallAround2<Orientation::OUT>(graph, v, sampled_radius, distances, q);
        }
    }

    // cutting in balls
    q.clear();
    distances.assign(n, c::infty);
    for (auto v : in_light) {
        if (graph.is_vtx_active[v]) {
            const Distance sampled_radius = geom_dist(gen);
            assert(sampled_radius >= 0);
            getBallAround2<Orientation::IN>(graph, v, sampled_radius, distances, q);
        }
    }
    PRINT("cut " << num_cut << " many edges, out of which " << neg_cut << " are negative");
}

void bcf::SSSPAlg::cutEdges3(Graph& graph, Distance kappa, int seed) {
// void bcf::SSSPAlg::cutEdges(Graph& graph, NodeID kappa, int seed) {
    // initialize geometric distribution
    std::mt19937 gen(seed);
    const NodeID n = graph.numberOfNodes();
    const double p = std::max(0.0001, std::min(0.9999, 20.0 * log(static_cast<double>(n)) / static_cast<double>(kappa)));
    std::geometric_distribution<int> geom_dist(p);

    auto out_light = labelInLight(graph, (kappa + 4 - 1) / 4, config::ol_seed, Orientation::IN);
    auto in_light = labelInLight(graph, (kappa + 4 - 1) / 4, config::il_seed, Orientation::OUT);
    PRINT("num of out light vtcs: " << out_light.size() << " wrt radius = " << (kappa + 4 - 1) / 4);
    PRINT("num of in light vtcs: " << in_light.size() << " wrt radius = " << (kappa + 4 - 1) / 4);
    PRINT("avg sample = 1/p = " << 1. / p << "; fresh sample = " << geom_dist(gen));

    // cutting out balls
    Distances distances(n, c::infty);
    std::vector<unsigned char> balls_covered(n, 0);
    // initializing queue here instead of inside getBallAround is important (to avoid re initializing it each time)
    bcf::GraphHeap q(n);
    EdgeID num_cut = 0, neg_cut = 0;
    unsigned char current_ball = 1;
    for (auto v : out_light) {
        if (graph.is_vtx_active[v]) {
            const Distance sampled_radius = geom_dist(gen);
            assert(sampled_radius >= 0);
            auto ball = getBallAround3(graph, v, sampled_radius, distances, q, Orientation::OUT, balls_covered, current_ball);
            // ball.first = bit vector, ball.second = list of nodes in the ball
            if (config::cutedges == 4) {
                for (NodeID w = 0; w < n; w++) {
                    if (!ball.first[w]) continue;
                    // removing w from the graph
                    graph.is_vtx_active[w] = false;
                    // cutting boundary of the ball
                    for (EdgeID i = graph.offsets[w]; i < graph.offsets[w + 1]; i++) {
                        auto const e = graph.getEdges(Orientation::OUT)[i];
                        // if the tail is not in the ball, and it has not been carved before: delete edge
                        if (!ball.first[e.target] && graph.is_vtx_active[e.target]) {
                            num_cut++;
                            if (e.weight < 0) neg_cut++;
                            // Note that now this method is handling the deletion of the corresponding flipped edge
                            graph.killOutEdge(i);
                        }
                    }
                }
            } else if (config::cutedges == 5) {
                for (NodeID w : ball.second) {
                    // removing w from the graph
                    graph.is_vtx_active[w] = false;
                    // cutting boundary of the ball
                    for (EdgeID i = graph.offsets[w]; i < graph.offsets[w + 1]; i++) {
                        auto const e = graph.getEdges(Orientation::OUT)[i];
                        // if the tail is not in the ball, and it has not been carved before: delete edge
                        if (balls_covered[e.target] != current_ball && graph.is_vtx_active[e.target]) {
                        // if (!ball.first[e.target] && graph.is_vtx_active[e.target]) {
                            num_cut++;
                            if (e.weight < 0) neg_cut++;
                            // Note that now this method is handling the deletion of the corresponding flipped edge
                            graph.killOutEdge(i);
                        }
                    }
                }
                for (NodeID w : ball.second) {
                    balls_covered[w] = 0;
                }
            }
        }
        // ++current_ball;
    }

    // cutting in balls
    q.clear();
    distances.assign(n, c::infty);
    for (auto v : in_light) {
        if (graph.is_vtx_active[v]) {
            const Distance sampled_radius = geom_dist(gen);
            assert(sampled_radius >= 0);
            auto ball = getBallAround3(graph, v, sampled_radius, distances, q, Orientation::IN, balls_covered, current_ball);
            // ball.first = bit vector, ball.second = list of nodes in the ball
            if (config::cutedges == 4) {
                for (NodeID w = 0; w < n; w++) {
                    if (!ball.first[w]) continue;
                    // removing w from the graph
                    graph.is_vtx_active[w] = false;
                    // cutting boundary of the ball
                    for (EdgeID i = graph.offsets_rev[w]; i < graph.offsets_rev[w + 1]; i++) {
                        auto const e = graph.getEdges(Orientation::IN)[i];
                        if (!ball.first[e.target] && graph.is_vtx_active[e.target]) {
                            num_cut++;
                            if (e.weight < 0) neg_cut++;
                            graph.killInEdge(i);
                        }
                    }
                }
            } else if (config::cutedges == 5) {
                for (NodeID w : ball.second) {
                    // removing w from the graph
                    graph.is_vtx_active[w] = false;
                    // cutting boundary of the ball
                    for (EdgeID i = graph.offsets_rev[w]; i < graph.offsets_rev[w + 1]; i++) {
                        auto const e = graph.getEdges(Orientation::IN)[i];
                        // if (!ball.first[e.target] && graph.is_vtx_active[e.target]) {
                        if (balls_covered[e.target] != current_ball && graph.is_vtx_active[e.target]) {
                            num_cut++;
                            if (e.weight < 0) neg_cut++;
                            graph.killInEdge(i);
                        }
                    }
                }
                for (NodeID w : ball.second) {
                    balls_covered[w] = 0;
                }
            }
        }
        // ++current_ball;
    }
    PRINT("cut " << num_cut << " many edges, out of which " << neg_cut << " are negative");
}

// returns a set of the vertices inside the ball B_out(source, radius)
// receives the array current_distances to avoid re-initializing at every iteration
// Orientation::OUT or ::IN to operate on the normal or reversed graph, respectively
std::unordered_set<NodeID> bcf::SSSPAlg::getBallAround(const Graph& graph, NodeID source, Distance radius,
                                                       Distances& current_distances,
                                                       bcf::GraphHeap& q, Orientation orientation) {
    current_distances[source] = 0;
    q.insert(source, 0);

    std::unordered_set<NodeID> inside_ball;
    while (!q.empty()) {
        Distance dist;
        NodeID from;
        q.deleteMin(from, dist);
        // assert(dist == current_distances[from]);
        assert(graph.is_vtx_active[from]);
        inside_ball.insert(from);
        for (auto const& edge : graph.getEdgesOf(from, orientation)) {
            // check if target was in some ball that is being carved
            if (!graph.is_vtx_active[edge.target]) continue;
            assert(edge.active);
            // NOTE: here we bump negative edges to 0
            auto tentative_dist = current_distances[from] + std::max(edge.weight, static_cast<Distance>(0));
            if (tentative_dist < current_distances[edge.target] && tentative_dist <= radius) {
                current_distances[edge.target] = tentative_dist;
                if (q.contains(edge.target)) {
                    // TODO: this can be removed, just like Dijkstra
                    q.decreaseKey(edge.target, tentative_dist);
                } else {
                    q.insert(edge.target, tentative_dist);
                }
            }
        }
    }
    return inside_ball;
}

template <Orientation orientation>
void bcf::SSSPAlg::getBallAround2(Graph& graph, NodeID source, Distance radius,
                                 Distances& current_distances,
                                 bcf::GraphHeap& q) {
    current_distances[source] = 0;
    q.insert(source, 0);

    while (!q.empty()) {
        Distance dist;
        NodeID from;
        q.deleteMin(from, dist);
        // assert(dist == current_distances[from]);
        assert(graph.is_vtx_active[from]);

        const std::vector<EdgeID> *offsets;
        if constexpr (orientation == Orientation::OUT) {
            offsets = &graph.offsets;
        } else {
            offsets = &graph.offsets_rev;
        }

        for (EdgeID edgeid = (*offsets)[from]; edgeid < (*offsets)[from + 1]; edgeid++) {
            const Edge edge = graph.getEdges(orientation)[edgeid];
            // check if target was in some ball that is being carved
            if (!graph.is_vtx_active[edge.target]) continue;
            assert(edge.active);
            // NOTE: here we bump negative edges to 0
            auto tentative_dist = current_distances[from] + std::max(edge.weight, static_cast<Distance>(0));
            if (tentative_dist <= radius) {
                if (tentative_dist < current_distances[edge.target]) {
                    current_distances[edge.target] = tentative_dist;
                    if (q.contains(edge.target)) {
                        // TODO: this can be removed, just like Dijkstra
                        q.decreaseKey(edge.target, tentative_dist);
                    } else {
                        q.insert(edge.target, tentative_dist);
                    }
                }
            }
            else {
                if constexpr (orientation == Orientation::OUT) {
                    if (graph.is_vtx_active[edge.target]) {
                        graph.killOutEdge(edgeid);
                    }
                } else {
                    if (graph.is_vtx_active[edge.target]) {
                        graph.killInEdge(edgeid);
                    }
                }
            }
        }
        graph.is_vtx_active[from] = false;
    }
}

// returns a set of the vertices inside the ball B_out(source, radius)
// receives the array current_distances to avoid re-initializing at every iteration
// Orientation::OUT or ::IN to operate on the normal or reversed graph, respectively
std::pair<std::vector<char>, std::vector<NodeID>> bcf::SSSPAlg::getBallAround3(const Graph& graph, NodeID source, Distance radius,
                                                       Distances& current_distances,
                                                       bcf::GraphHeap& q, Orientation orientation, std::vector<unsigned char>& balls_covered, unsigned char& current_ball) {
    current_distances[source] = 0;
    q.insert(source, 0);

    // std::vector<char> inside_ball(graph.numberOfNodes(), false);
    std::vector<NodeID> inside_ball_idx;
    while (!q.empty()) {
        Distance dist;
        NodeID from;
        q.deleteMin(from, dist);
        // assert(dist == current_distances[from]);
        assert(graph.is_vtx_active[from]);
        // inside_ball[from] = true;
        inside_ball_idx.push_back(from);
        balls_covered[from] = current_ball;
        for (auto const& edge : graph.getEdgesOf(from, orientation)) {
            // check if target was in some ball that is being carved
            if (!graph.is_vtx_active[edge.target]) continue;
            assert(edge.active);
            // NOTE: here we bump negative edges to 0
            auto tentative_dist = current_distances[from] + std::max(edge.weight, static_cast<Distance>(0));
            if (tentative_dist < current_distances[edge.target] && tentative_dist <= radius) {
                current_distances[edge.target] = tentative_dist;
                if (q.contains(edge.target)) {
                    // TODO: this can be removed, just like Dijkstra
                    q.decreaseKey(edge.target, tentative_dist);
                } else {
                    q.insert(edge.target, tentative_dist);
                }
            }
        }
    }

    // return {std::move(inside_ball), std::move(inside_ball_idx)};
    return {{}, std::move(inside_ball_idx)};
}

// Very slow. It's here in case we generate SCCs that are too big and we may
// need to just "cut it in half"
void bcf::SSSPAlg::cutEdgesInHalf(Graph& graph) {
    NodeID first_half = (graph.numberOfNodes() + 1) / 2;

    std::vector<EdgeID> first_half_edges, second_half_edges;

    for (NodeID v = 0; v < first_half; v++) {
        for (EdgeID e = graph.offsets[v]; e < graph.offsets[v + 1]; e++) {
            auto const edge = graph.getEdges(Orientation::OUT)[e];
            if (first_half <= edge.target) {
                second_half_edges.push_back(e);
            }
        }
    }

    for (NodeID v = first_half; v < graph.numberOfNodes(); v++) {
        for (EdgeID e = graph.offsets[v]; e < graph.offsets[v + 1]; e++) {
            auto const edge = graph.getEdges(Orientation::OUT)[e];
            if (first_half > edge.target) {
                first_half_edges.push_back(e);
            }
        }
    }

    if (first_half_edges.size() < second_half_edges.size()) {
        for (EdgeID e : first_half_edges) {
            graph.killOutEdge(e);
        }
    } else {
        for (EdgeID e : second_half_edges) {
            graph.killOutEdge(e);
        }
    }
}

std::optional<Distances> bcf::SSSPAlg::runMainAlg(Graph& graph, Distance kappa, int level) {
// std::optional<Distances> bcf::SSSPAlg::runMainAlg(Graph& graph, NodeID kappa, int level) {
    DEBUG("recursion level: " << level);
    const NodeID n = graph.numberOfNodes();

    bool small_graph_size = (static_cast<NodeID>(kappa) + n) <= 300;

    // Skip diameter approximation if the graph is already small
    if ((!(small_graph_size || level >= config::rec_limit)) && config::diam_apprx) {  // approximate kappa (true) or get kappa from the levels above (false)?
        // Computing 2-approx of the diameter
        const Distances &d_out = runDijkstra(graph, 0, c::infty, Orientation::OUT);  // choosing 0 as source, but could be any
        const Distances &d_in = runDijkstra(graph, 0, c::infty, Orientation::IN);
        // Below we compute kappa sa a 2-approximation of the diameter. Shouldn't we take the minimum between
        // the kappa that we gat as a parameter and this approximation?
        // kappa = *std::max_element(d_out.begin(), d_out.end()) + *std::max_element(d_in.begin(), d_in.end());
        kappa = std::min(*std::max_element(d_out.begin(), d_out.end()) + *std::max_element(d_in.begin(), d_in.end()), kappa);

        // Distance sumnegedges = 0;
        // for (NodeID v = 0; v < n; v++) {
        //     for (auto e : graph.getEdgesOf(v)) {
        //         // const Distance weight = std::get<2>(e);
        //         if (e.weight < 0) {
        //             sumnegedges -= e.weight;
        //         }
        //     }
        // }
        // kappa = std::min(sumnegedges, kappa);
    }

    // TODO: this should be tuned...
    if (small_graph_size || level >= config::rec_limit) {
        MEASUREMENT::start(EXP::LAST_LAZY);
        Distances p(n, 0);
        std::optional<Distances> opt_d;
        // std::cout << "n in leaf = " << n << " kappa = " << kappa << std::endl;
        if (config::use_lazy) {
            opt_d = bcf::runLazyDijkstra(graph, p);
        } else {
            // opt_d = choose the algorithm here(graph, p)
            opt_d = gor(graph, p);
        }
        // auto d = bcf::runBellman(subgraph.graph, p);
        MEASUREMENT::stop(EXP::LAST_LAZY);
        return opt_d;
    }

    if (level == 0) MEASUREMENT::start(EXP::CUT_EDGES);
    // TODO: now we have the random seed fixed, maybe test variance?
    if (config::cutedges == 1) {
        cutEdges(graph, kappa, config::cutedgesseed);
    } else if (config::cutedges == 2) {
        cutEdges2(graph, kappa, config::cutedgesseed);
    } else if (config::cutedges == 3) {
        cutEdgesInHalf(graph);
    } else if (config::cutedges == 4 || config::cutedges == 5) {
        cutEdges3(graph, kappa, config::cutedgesseed);
    }
    if (level == 0) MEASUREMENT::stop(EXP::CUT_EDGES);

    if (level == 0) MEASUREMENT::start(EXP::SCC);
    auto components = decomposeIntoSCCs(graph);
    if (level == 0) MEASUREMENT::stop(EXP::SCC);

    PRINT("recursing on " << components.size() << " subproblems");
    NodeID kappa_reduced = 0, size_reduced = 0;

    Distances potential(n);  // Default-initializing because we set them all later


    if (level == 0) MEASUREMENT::start(EXP::INN_REC);
    for (auto& part : components) {
        Distances d;
        // const NodeID n_part = part.numberOfNodes();
        const Distance n_part = static_cast<Distance>(part.numberOfNodes());
        std::optional<Distances> opt_d;
        if constexpr (true) {
            if (!config::diam_apprx) {
                // case where the size of the subgraph is reduced
                if (4 * n_part < 3 * n) {
                    size_reduced++;
                    opt_d = runMainAlg(part, std::min(n_part, kappa), level + 1);
                } else {  // kappa is reduced
                    kappa_reduced++;
                    // todo: check if that min makes sense always
                    opt_d = runMainAlg(part, std::min(n_part, kappa / 2), level + 1);
                }
            } else {
                // opt_d = runMainAlg(part, kappa, level + 1);
                opt_d = runMainAlg(part, std::min(n_part, kappa), level + 1);
            }

        } else {
            // opt_d = choose the algorithm here(part, potential_set_to_0)
            Distances p(part.numberOfNodes(), 0);
            opt_d = gor(part, p);
        }

        if (!opt_d.has_value()) {
            if (level == 0) MEASUREMENT::stop(EXP::INN_REC);  // otherwise we get an error
            return {};
        }

        d = std::move(opt_d.value());

        // assign potentials
        for (NodeID v = 0; v < part.numberOfNodes(); v++) {
            potential[part.global_id[v]] = d[v];
        }
    }
    if (level == 0) MEASUREMENT::stop(EXP::INN_REC);

    PRINT("reduced kappa = " << kappa_reduced << " reduced size = " << size_reduced);

    if (level == 0) MEASUREMENT::start(EXP::INN_FIX);
    // Fix DAG Edges
    if (config::fixdagedges == 1) {
        fixDagEdges(graph, components, potential);
    } else {
        fixDagEdges2(graph, components, potential);
    }
    if (level == 0) MEASUREMENT::stop(EXP::INN_FIX);

    // fix edges that were cut
    if (level == 0) MEASUREMENT::start(EXP::LAZY_END);
    std::optional<Distances> opt_d;
    if (config::use_lazy) {
        opt_d = bcf::runLazyDijkstra(graph, potential);
    } else {
        if (level == 0 && config::shift_filename != std::string()) {
            // Open file named ../data/graphs/shift_aug_gor_1e6.txt

            std::ofstream f;
            f.open(config::shift_filename);
            f << graph.numberOfNodes() + 1 << std::endl;

            // Print supersource
            for (NodeID i = 0; i < graph.numberOfNodes(); i++) {
                f << "0 " << i + 1 << " " << -potential[i] << std::endl;
            }

            // Print edges
            for (NodeID i = 0; i < graph.numberOfNodes(); i++) {
                for (auto const& edge : graph.getEdgesOf(i)) {
                    f << i + 1 << " " << edge.target + 1 << " " << edge.weight + potential[i] - potential[edge.target] << std::endl;
                }
            }
            std::cout << "Finished writing to file" << std::endl;
            MEASUREMENT::stop(EXP::LAZY_END);
            return {};
        }
        // opt_d = choose the algorithm here(graph, potential)
        opt_d = gor(graph, potential);
    }

    if (level == 0) MEASUREMENT::stop(EXP::LAZY_END);

    if (!opt_d.has_value()) return {};

    auto d = std::move(opt_d.value());

    MEASUREMENT::start(EXP::INN_POT);
    // Fixing distances with potential (the potential of virtual source is 0)
    for (NodeID i = 0; i < n; i++) {
        d[i] += potential[i];  // += potential[i] - potential[virtual_source];
    }
    MEASUREMENT::stop(EXP::INN_POT);
    return d;
}

// Fixes DAG Edges by modifying the given potential
void bcf::fixDagEdges(const Graph& graph, const std::vector<Graph>& components, Distances& potential) {// compute minimum edge weight
    const NodeID n = graph.numberOfNodes();
    Distance min_w = 0;
    for (NodeID v = 0; v < n; v++) {
        for (auto& e : graph.getEdgesOf(v)) {
            Distance pot_w = e.weight + potential[v] - potential[e.target];
            min_w = std::min(min_w, pot_w);
        }
    }

    static const int max_bit_dist = std::bit_width(static_cast<unsigned long int>(c::infty));
    const bool constexpr check_overflow = true;

    // id_of_scc is sorted in topological order
    // we set potential[w] += id_of_scc[w] * (min_W - 1)
    // so that edges between different SCCs have potentialized weight >= w + (min_W + 1) > 0
    for (NodeID id = 0; id < components.size(); id++) {
        auto& comp = components[id];
        for (NodeID v = 0; v < comp.numberOfNodes(); v++) {
            if constexpr (check_overflow) {
                if (std::bit_width(static_cast<unsigned long int>(id)) + std::bit_width(static_cast<unsigned long int>(std::abs(min_w - 1))) >= max_bit_dist - 1) {  // Potential overflow!!!
                    throw std::overflow_error("fixDagEdges: overflow");
                }
            }
            potential[comp.global_id[v]] += static_cast<Distance>(id) * (min_w - 1);  // casting to Distance to preserve the (negative) sign, hoping the id is not too big
        }
    }
}

// May be slightly slower, but it should reduce the chances of overflow
void bcf::fixDagEdges2(const Graph& graph, const std::vector<Graph>& components, Distances& potential) {
    int n_components = components.size();
    Distances new_potential(graph.numberOfNodes(), 0);

    // Check case with 1 component so that I can assume more than 1 later
    if (n_components == 1) {
        return;
    }

    // ToDo: sort component in topological order. Tarjan's algorithm to
    // decompose already yields ordered SCCs

    // Mapping each node to its SCC
    std::vector<NodeID> component_id(graph.numberOfNodes());
    for (NodeID i = 0; i < n_components; i++) {
        for (auto id : components[i].global_id) {
            component_id[id] = i;
        }
    }

    std::vector<Distance> min_component_weight(n_components, 0);  // If there are no incoming edges the minimum weight has to be 0
    // Since the minimum weight is -1, if one of the entries of this vector
    // is already -1 I guess a lot of work can be saved. I'm implementing the
    // dummy version
    for (NodeID v = 0; v < graph.numberOfNodes(); v++) {
        for (auto const& e : graph.getEdgesOf(v)) {
            NodeID tail = v;
            NodeID head = e.target;
            Distance weight = e.weight;
            if (component_id[tail] != component_id[head]) {  // if they don't belong to the same SCC
                min_component_weight[component_id[head]] = std::min(min_component_weight[component_id[head]], weight);
            }
        }
    }

    // Assuming more than one SCC
    Distance component_potential = 0;  // M_1
    for (NodeID c = 1; c < n_components; c++) {
        component_potential += min_component_weight[c];  // M_j
        for (auto id : components[c].global_id) {
            potential[id] += component_potential;
        }
    }
}

// returns a list L of vertices st whp |B_in(v, radius)| <= 7n/10 for all v in L
std::vector<NodeID> bcf::SSSPAlg::labelInLight(const Graph& graph, Distance radius, const int seed, Orientation orientation) {
    std::mt19937 rng(seed);
    const NodeID n = graph.numberOfNodes();
    const double eps = 0.1;
    const int k = std::max(1, static_cast<int>(5.0 * log(static_cast<double>(n) / eps)) / config::k_factor);
    // const int k = 1;
    if (!config::rand_label) {
        std::vector<int> counts(n, 0);
        DEBUG("sampling from " << k << " sources with radius " << radius);
        bcf::RunDijkstra runDijkstra(n);
        for (int i = 0; i < k; i++) {
            NodeID src = rng() % n;
            // TODO: make run Dijkstra increase the counts directly. initialize priority queue here
            const auto &distances = runDijkstra(graph, src, radius, orientation);
            // auto distances = bcf::runDijkstra(graph, src, radius, orientation);
            for (NodeID v = 0; v < n; v++) {
                if (distances[v] != c::infty) {
                    counts[v]++;
                }
            }
            // DEBUG(".");
        }
        // counts[v] * n/k is an approximation of |B_in(v, radius)| with additive error eps * n
        std::vector<NodeID> is_light;
        for (NodeID v = 0; v < n; v++) {
            if (5 * counts[v] <= 3 * k) {
                is_light.emplace_back(v);
            }
        }
        return is_light;
    } else {
        std::vector<NodeID> is_light;
        double chance_to_be_light = 0.5;
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        for (NodeID i = 0; i < n; ++i) {
            if (distribution(rng) < chance_to_be_light) {
                is_light.emplace_back(i);
            }
        }
        return is_light;
    }
}
