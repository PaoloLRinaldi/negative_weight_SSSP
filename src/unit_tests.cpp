#include <iostream>
#include <set>

#include "algorithms.h"
#include "graph.h"
// #include "permutations.h"

#define TEST(x)                                            \
    do {                                                   \
        if (!(x)) {                                        \
            std::cout << "\n";                             \
            std::cout << "TEST_FAILED!\n";                 \
            std::cout << "File: " << __FILE__ << "\n";     \
            std::cout << "Line: " << __LINE__ << "\n";     \
            std::cout << "Function: " << __func__ << "\n"; \
            std::cout << "Test: " << #x << "\n";           \
            std::cout << "\n";                             \
            std::cout << std::flush;                       \
            std::abort();                                  \
        }                                                  \
    } while (0)

#define SUCCESS()                                         \
    do {                                                  \
        std::cout << "Test Passed: " << __func__ << "\n"; \
        std::cout << std::flush;                          \
    } while (0)

namespace unit_tests {

void testAll() {
    testGraph();
    testNaiveBFM();
    testDijkstra();
    testLazyDijkstra();
    testSCC();
    testPermutations();
    testBCF();
    testGOR();
    testBFCT();
}

void testGraph() {
    auto graph = readGraph("../data/graphs/simple1.txt");

    TEST(graph.numberOfNodes() == 5);
    TEST(graph.numberOfEdges() == 6);

    for (auto const &edge : graph.getEdgesOf(0)) {
        TEST(edge.weight == 42);
    }


    TEST(graph.getEdgesOf(0, Orientation::IN).size() == 2);
    TEST(graph.getEdgesOf(4, Orientation::IN).size() == 1);

    // TODO: more thorough tests

    SUCCESS();
}

void testNaiveBFM() {
    auto graph = readGraph("../data/graphs/simple1.txt");
    auto result = computeSSSP(SSSPAlg::NaiveBFM, graph, 0);

    TEST(result);  // has a negative cycle

    // TODO: more thorough tests

    SUCCESS();
}

void testDijkstra() {
    {
        auto graph = readGraph("../data/graphs/positive.txt");
        auto result_bf = computeSSSP(SSSPAlg::NaiveBFM, graph, 3);
        auto result_dijkstra = computeSSSP(SSSPAlg::Dijkstra, graph, 3);
        TEST(result_bf == result_dijkstra);
    }
    {
        auto graph = readGraph("../data/graphs/simple2.txt");
        auto countReachable = [](const Distances &distances) {
            int reachable = 0;
            for (auto d : distances) {
                if (d != c::infty)
                    reachable++;
            }
            return reachable;
        };
        TEST(countReachable(bcf::runDijkstra(graph, 0)) == 7);
        TEST(countReachable(bcf::runDijkstra(graph, 0, 1)) == 4);
        TEST(countReachable(bcf::runDijkstra(graph, 0, 4)) == 6);
    }
    {
        auto graph = readGraph("../data/graphs/random2.txt");
        auto countReachable = [](const Distances &distances) {
            int reachable = 0;
            for (auto d : distances) {
                if (d != c::infty)
                    reachable++;
            }
            return reachable;
        };
        std::cout << "n: " << graph.numberOfNodes() << std::endl;

        auto distances = bcf::runDijkstra(graph, 0);
        std::cout << "num reachable from 0: " << countReachable(distances) << std::endl;

        // testing how accurate is the testing of light vertices

        // radius is chosen so that roughly half are light and half are heavy
        const int radius = 80;
        const int n = graph.numberOfNodes();

        std::cout << "radius " << radius << std::endl;
        std::cout << "|B_out(0, radius)| = " << countReachable(bcf::runDijkstra(graph, 0, radius)) << std::endl;

        // compute the true size |B_out(v, r)| for all v
        std::vector<int> size_reachable(n, 0);
        std::vector<bool> is_heavy(n, false);
        for (NodeID src = 0; src < n; src++) {
            size_reachable[src] = countReachable(bcf::runDijkstra(graph, src, radius));
            if (size_reachable[src] > n / 2) {
                is_heavy[src] = 1;
            }
        }

        const int bound_light = static_cast<int>(n * 7.0 / 10.0);
        std::cout << "If |B_out(v, r)| <= " << bound_light << " then it is light" << std::endl;
        std::cout << "If |B_out(v, r)| > " << n / 2 << " then it is heavy" << std::endl;

        auto alg = bcf::SSSPAlg();
        auto light_vtcs = alg.labelInLight(graph, radius, 3120, Orientation::IN);
        std::set<NodeID> is_light_approx(light_vtcs.begin(), light_vtcs.end());
        int heavy = 0, light = 0;
        int num_missclassified = 0;
        for (NodeID v = 0; v < graph.numberOfNodes(); v++) {
            if (is_light_approx.contains(v))
                light++;
            else
                heavy++;

            // if v is labeled as heavy, it should have |B(v, r)| > n/2
            if (!is_light_approx.contains(v) && !is_heavy[v]) {
                num_missclassified++;
                std::cout << "classified as heavy, true ball size: " << size_reachable[v] << std::endl;
            }
            // if v is labeled light, it should have |B(v, r)| < 7n/10
            if (is_light_approx.contains(v) && size_reachable[v] > bound_light) {
                num_missclassified++;
                std::cout << "classified as light, true ball size: " << size_reachable[v] << std::endl;
            }
        }
        std::cout << std::endl;
        std::cout << "Overall: classified as heavy = " << heavy << ", as light = " << light << std::endl;
        // TEST(num_missclassified <= 15);
    }
    SUCCESS();
}

void testLazyDijkstra() {
    {
        auto graph = readGraph("../data/graphs/negative2.txt");
        auto result_bf = computeSSSP(SSSPAlg::NaiveBFM, graph, 3);
        if (!result_bf) {
            std::cout << "cycle!" << std::endl;
            return;
        }
        // auto result_lazy = bcf::runLazyDijkstra(graph, 3);  // Old version with source
        Distances pot(graph.numberOfNodes(), 0);
        auto result_lazy = bcf::runLazyDijkstra(graph, pot);

        auto countReachable = [](const Distances &distances) {
            int reachable = 0;
            for (auto d : distances) {
                if (d != c::infty)
                    reachable++;
            }
            return reachable;
        };
        std::cout << "reachable: " << countReachable(result_lazy.value()) << std::endl;
        // TEST(result_bf == result_lazy);  // Temporarily disabled because now lazyDijkstra does not work with a source


        graph = readGraph("../data/graphs/simple1.txt");
        auto result = bcf::runLazyDijkstra(graph, 1, Orientation::OUT);

        TEST(!result.has_value());  // has a negative cycle


        SUCCESS();
    }
}

void testSCC() {
    auto graph = readGraph("../data/graphs/simple-scc.txt");
    auto comps = decomposeIntoSCCs(graph);
    for (auto g : comps) {
        for (int v = 0; v < g.numberOfNodes(); v++) {
            std::cout << g.global_id[v] << " ";
        }
        std::cout << std::endl;
    }
    TEST(comps.size() == 4);
    SUCCESS();
}

void testPermutations() {
    using P = std::vector<int>;
    using namespace bcf;

    {
        TEST(getIdentityPermutation<int>(1) == P{0});
        TEST(getIdentityPermutation<int>(2) == (P{0, 1}));
        TEST(getIdentityPermutation<int>(10) == (P{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
    }
    {
        P p = {0, 3, 1, 2};
        P inv_p = {0, 2, 3, 1};
        TEST(getInversePermutation(p) == inv_p);
        TEST(getInversePermutation(inv_p) == p);
    }
    {
        P nums = {10, 9, 15, 0};
        P sort_perm = {3, 1, 0, 2};
        TEST(getSortPermutation(nums) == sort_perm);
    }
    {
        P nums = {100, 50, 20, 30, 0};
        P sort_perm = {4, 2, 3, 1, 0};
        TEST(getSortPermutation(nums) == sort_perm);
    }
    SUCCESS();
}

void testBCF() {
    // testing distances
    {
        auto graph = readGraph("../data/graphs/rand-rest-simple.txt");
        std::cout << "n = " << graph.numberOfNodes() << ", m = " << graph.numberOfEdges() << std::endl;
        for (int src : {0, 1, 3, 10}) {
            auto result_bf = computeSSSP(SSSPAlg::NaiveBFM, graph, src);
            auto result_bcf = computeSSSP(SSSPAlg::BCF, graph, src);
            TEST(result_bcf == result_bf);
        }
    }

    {
        auto graph = readGraph("../data/graphs/rand-rest-1000.txt");
        std::cout << "n = " << graph.numberOfNodes() << ", m = " << graph.numberOfEdges() << std::endl;
        for (int src : {0, 10, 100, 250}) {
            auto result_bf = computeSSSP(SSSPAlg::NaiveBFM, graph, src);
            auto result_bcf = computeSSSP(SSSPAlg::BCF, graph, src);
            TEST(result_bcf == result_bf);
        }
    }

    {
        auto graph = readGraph("../data/graphs/aug_badgor1000_2.txt");
        std::cout << "n = " << graph.numberOfNodes() << ", m = " << graph.numberOfEdges() << std::endl;
        for (int src : {0, 10, 100, 250}) {
            auto result_bf = computeSSSP(SSSPAlg::NaiveBFM, graph, src);
            auto result_bcf = computeSSSP(SSSPAlg::BCF, graph, src);
            TEST(result_bcf.value() == result_bf.value());
        }
    }

    // Tests if potentialized weights are non-negative
    auto testPotentializedEdges = [](std::string graph_name) {
        std::cout << "\nTesting potentialized weights for graph: " << graph_name << std::endl;
        auto graph = readGraph(graph_name);
        std::cout << "n = " << graph.numberOfNodes() << ", m = " << graph.numberOfEdges() << std::endl;
        std::cout << "num SCCs = " << decomposeIntoSCCs(graph).size() << std::endl;
        const int n = graph.numberOfNodes();
        auto alg = bcf::SSSPAlg();
        auto potential = alg.runMainAlg(graph, n);

        for (int v = 0; v < n; v++) {
            for (auto e : graph.getEdgesOf(v)) {
                TEST(e.weight + potential.value()[v] - potential.value()[e.target] >= 0);
            }
        }
    };

    testPotentializedEdges("../data/graphs/rand-rest-100.txt");
    testPotentializedEdges("../data/graphs/rand-rest-300.txt");
    testPotentializedEdges("../data/graphs/rand-rest-500.txt");
    testPotentializedEdges("../data/graphs/rand-rest-700.txt");
    testPotentializedEdges("../data/graphs/rand-rest-hard-400.txt");

    {
        auto graph = readGraph("../data/graphs/simple1.txt");
        auto alg = bcf::SSSPAlg();
        auto result = alg.runMainAlg(graph, graph.numberOfNodes());

        TEST(!result.has_value());  // has a negative cycle
    }

    SUCCESS();
}

void testGOR() {
    auto test_and_compare = [](std::string graph_name) {
        std::cout << "Testing graph: " << graph_name << " for GOR" << std::endl;
        auto graph = readGraph(graph_name);
        for (int src : {0, 10, 99}) {
            auto result_bf = computeSSSP(SSSPAlg::NaiveBFM, graph, src);
            auto result_gor = computeSSSP(SSSPAlg::GOR, graph, src);
            TEST(result_gor == result_bf);
        }
    };

    std::vector<std::string> graphs = {
        "../data/graphs/rand-rest-100.txt",
        "../data/graphs/rand-rest-300.txt",
        "../data/graphs/rand-rest-500.txt",
        "../data/graphs/rand-rest-700.txt",
        "../data/graphs/rand-rest-1000.txt",
        "../data/graphs/rand-rest-hard-400.txt",
        "../data/graphs/badgor100.txt",
        "../data/graphs/aug_badgor100_2.txt",
    };

    for (auto graph_name : graphs) {
        test_and_compare(graph_name);
    }


        auto graph = readGraph("../data/graphs/rand-rest-1000.txt");
        std::cout << "n = " << graph.numberOfNodes() << ", m = " << graph.numberOfEdges() << std::endl;
        for (int src : {0, 10, 100, 250}) {
            auto result_bf = computeSSSP(SSSPAlg::NaiveBFM, graph, src);
            auto result_gor = computeSSSP(SSSPAlg::GOR, graph, src);
            TEST(result_gor == result_bf);
        }

        auto graph2 = readGraph("../data/graphs/rand-rest-hard-400.txt");
}

void testBFCT() {
    auto test_and_compare = [](std::string graph_name) {
        std::cout << "Testing graph: " << graph_name << " for BFCT" << std::endl;
        auto graph = readGraph(graph_name);
        for (int src : {0, 10, 99}) {
            auto result_bf = computeSSSP(SSSPAlg::NaiveBFM, graph, src);
            auto result_gor = computeSSSP(SSSPAlg::BFCT, graph, src);
            TEST(result_gor == result_bf);
        }
    };

    std::vector<std::string> graphs = {
        "../data/graphs/rand-rest-100.txt",
        "../data/graphs/rand-rest-300.txt",
        "../data/graphs/rand-rest-500.txt",
        "../data/graphs/rand-rest-700.txt",
        "../data/graphs/rand-rest-1000.txt",
        "../data/graphs/rand-rest-hard-400.txt",
        "../data/graphs/badbfct310.txt",
        "../data/graphs/aug_badbfct100.txt",
    };

    for (auto graph_name : graphs) {
        test_and_compare(graph_name);
    }


    {
        auto graph = readGraph("../data/graphs/simple-neg.txt");
        auto result = computeSSSP(SSSPAlg::BFCT, graph, 0);

        TEST(!result.has_value());  // has a negative cycle
    }

}

}  // end namespace unit_tests

int main() { unit_tests::testAll(); }
