#pragma once

#include <optional>
#include <random>

#include "bcf.h"
#include "graph.h"
#include "gor.h"
#include "config.h"

namespace unit_tests {
void testNaiveBFM();
void testDijkstra();
void testLazyDijkstra();
void testSCC();
void testPermutations();
void testBCF();
void testGOR();
void testBFCT();
}  // namespace unit_tests

enum class SSSPAlg {
    NaiveBFM,
    Dijkstra,
    BCF,
    GOR,
    BFCT,
    LazyD
};
SSSPAlg toSSSPAlg(std::string const& alg_string);
std::string to_string(SSSPAlg const& alg);

enum class NegCycleAlg {
    NaiveBFM,
    BCF,
    LazyDijkstra,
    BFCT
};
NegCycleAlg toNegCycleAlg(std::string const& alg_string);

// Optional is empty if negative cycle was found
std::optional<Distances> computeSSSP(
    SSSPAlg algorithm, Graph& graph, NodeID source);

bool negCycleDetection(NegCycleAlg algorithm, Graph& graph);

bool isResultCorrect(Graph const& graph, Distances const& distances, NodeID source);
