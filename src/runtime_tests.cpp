
#include <iostream>
#include <set>
#include <string>

#define MEASURE
#include "algorithms.h"
#include "graph.h"
#include "measurement_tool.h"
// #include "permutations.h"

#define SIMPLE_TEST(x, y)                            \
    do {                                             \
        for (unsigned int i = 0; i < y; i++) {       \
            MEASUREMENT::start(EXP::INNER_LOOP_ALL); \
            x;                                       \
            MEASUREMENT::stop(EXP::INNER_LOOP_ALL);  \
        }                                            \
        MEASUREMENT::print();                        \
    } while (0)

namespace runtime_tests {

void testLazyDijkstra(std::string const& graph_string, int iterations) {
    auto graph = readGraph(graph_string);
    Distances pot(graph.numberOfNodes(), 0);
    SIMPLE_TEST(bcf::runLazyDijkstra(graph, pot), iterations);
}

void testBellman(std::string const& graph_string, int iterations) {
    auto graph = readGraph(graph_string);
    SIMPLE_TEST(computeSSSP(SSSPAlg::NaiveBFM, graph, 0), iterations);
}

void testBCF(std::string const& graph_string, int iterations) {
    auto graph = readGraph(graph_string);
    SIMPLE_TEST(computeSSSP(SSSPAlg::BCF, graph, 0), iterations);
}

void testGOR(std::string const& graph_string, int iterations) {
    auto graph = readGraph(graph_string);
    SIMPLE_TEST(computeSSSP(SSSPAlg::GOR, graph, 0), iterations);
}

void testAlgorithm(std::string const& algorithm, std::string const& graph_string, int iterations) {
    auto graph = readGraph(graph_string);
    auto alg = toSSSPAlg(algorithm);
}

}  // end namespace runtime_tests

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cout << "Invalid number of arguments: ./RuntimeTests <algorithm> <graph filename> <n iterations>" << std::endl;
        return 1;
    }

    if (std::string(argv[1]) == "lazy") {
        runtime_tests::testLazyDijkstra(argv[2], std::stoi(argv[3]));
    } else if (std::string(argv[1]) == "bcf") {
        runtime_tests::testBCF(argv[2], std::stoi(argv[3]));
    } else if (std::string(argv[1]) == "bellman") {
        runtime_tests::testBellman(argv[2], std::stoi(argv[3]));
    } else if (std::string(argv[1]) == "gor") {
        runtime_tests::testGOR(argv[2], std::stoi(argv[3]));
    } else {
        std::cout << "Invalid algorithm: " << argv[1] << std::endl;
        std::cout << "Valid algorithms: lazy, bcf, bellman, gor" << std::endl;
    }

    return 0;
}
