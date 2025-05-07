#include "queries.h"

#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <filesystem>

#include "algorithms.h"
#include "defs.h"
#include "graph.h"

Queries readQueries(std::string const& filename) {
    Queries queries;

    // buffer file
    std::ifstream file(filename);
    std::string query_string;
    std::stringstream ss;
    if (!file.is_open()) {
        // ERROR("Could not open query file: " << filename);
        // std::clog << "Could not open query file: " << filename << std::endl;
        // std::clog << "Treating argument as query" << std::endl;
        ss << filename;
    } else {
        ss << file.rdbuf();
    }
    // ss << file.rdbuf();
    auto ignore_count = std::numeric_limits<std::streamsize>::max();

    // read queries
    for (std::string algo_type; ss >> algo_type;) {
        if (algo_type == "SSSP") {
            // parse
            std::string query_type, alg_string, graph_filename, source_str, iters_str;
            unsigned int iters = 0;
            ss >> query_type;
            if (query_type != "check" && query_type != "time") {
                ERROR("Unknown query type: " << query_type << ". Expected 'check' or 'time'.");
            }
            ss >> alg_string >> graph_filename >> source_str;
            if (query_type == "time") {
                ss >> iters_str;
                iters = std::stoul(iters_str);
                if (iters == 0) {
                    ERROR("Number of iterations must be greater than 0.");
                }
            }
            NodeID source = std::stoul(source_str);
            auto algorithm = toSSSPAlg(alg_string);
            // add
            queries.data.push_back(SSSPData{graph_filename, source, algorithm, iters});
            queries.graphs.try_emplace(graph_filename, Graph());
        } else if (algo_type == "NegCycle") {
            // parse
            std::string query_type, alg_string, graph_filename, iters_str;
            unsigned int iters = 0;
            ss >> query_type;
            if (query_type != "check" && query_type != "time") {
                ERROR("Unknown query type: " << query_type << ". Expected 'check' or 'time'.");
            }
            ss >> alg_string >> graph_filename;
            if (query_type == "time") {
                ss >> iters_str;
                iters = std::stoul(iters_str);
                if (iters == 0) {
                    ERROR("Number of iterations must be greater than 0.");
                }
            }
            auto algorithm = toNegCycleAlg(alg_string);
            // add
            queries.data.push_back(NegCycleData{graph_filename, algorithm, iters});
            queries.graphs.try_emplace(graph_filename, Graph());
        } else if (algo_type.starts_with("#")) {
            std::string comment;
            std::getline(ss, comment);
            ss.unget();
        } else {
            ERROR("Unknown query type: " << algo_type);
        }

        ss.ignore(ignore_count, '\n');
    }

    // read graphs
    for (auto& [graph_filename, graph] : queries.graphs) {
        graph = readGraph(graph_filename);
    }

    return queries;
}

namespace {

void print(std::optional<Distances> const& distances, std::ostream& os) {
    if (distances) {
        for (auto const& d : distances.value()) {
            os << (d == c::infty ? "inf" : std::to_string(d)) << " ";
        }
        os << std::endl;
    } else {
        os << "Contains negative cycle!" << std::endl;
    }
}

void print(bool has_negative_cycle, std::ostream& os) {
    if (has_negative_cycle) {
        os << "Graph has a negative cycle." << std::endl;
    } else {
        os << "Graph has no negative cycle." << std::endl;
    }
}

void print(ExpTime const& times, std::ostream& os) {
    os << "Average: " << times.avg << " ms. std: " << times.std << " ms. min: " << times.min << " ms. max: " << times.max << " ms." << std::endl << std::flush;
}

std::optional<Distances> runQuery(SSSPData const& data, Graph& graph) {
    return computeSSSP(data.algorithm, graph, data.source);
}

bool runQuery(NegCycleData const& data, Graph& graph) {
    return negCycleDetection(data.algorithm, graph);
}

ExpTime parseNumbers(const std::string& line) {
    std::vector<double> numbers;
    std::istringstream iss(line);
    std::string word;

    while (iss >> word) {
        try {
            size_t pos;
            double number = std::stod(word, &pos);
            if (pos == word.length()) {
                numbers.push_back(number);
            }
        } catch (const std::exception& e) {
            // Ignore non-numeric words
        }
    }

    if (numbers.size() != 4) {
        ERROR("Could not parse times: " << line);
    }

    return {numbers[0], numbers[1], numbers[2], numbers[3]};
}

std::string getFilenameWithoutExtension(const std::string& filename) {
    size_t lastDotPos = filename.find_last_of(".");
    if (lastDotPos == std::string::npos) {
        // No extension found, return the original filename
        return filename;
    } else {
        // Extract the substring before the last dot
        return filename.substr(0, lastDotPos);
    }
}


bool handleCorrectness(std::optional<Distances> const& result, const Graph& graph, SSSPData const& data, NodeID number_of_nodes) {

    // Get the name of the file with the result
    std::string filename_check = getFilenameWithoutExtension(data.graph_filename) + "_result" + std::to_string(data.source) + ".txt";
    std::ifstream ifs(filename_check);

    // This is the possible filename in case the result is wrong, for debugging purposes
    std::string wrong_filename_check = getFilenameWithoutExtension(data.graph_filename) + "_wrong" + std::to_string(data.source) + "_" + to_string(data.algorithm) + ".txt";

    bool check_file_exists = ifs.good();
    ifs.close();
    bool check_file_neg_cycle = false,
         check_file_distances = false;

    if (check_file_exists) {
        auto distances_check = readDistancesFromFile(filename_check);
        if (distances_check.size() == 1 && distances_check[0] == -1) {
            check_file_neg_cycle = true;
        } else if (distances_check.size() != graph.numberOfNodes()) {
            std::string filename_unknown = getFilenameWithoutExtension(data.graph_filename) + "_unknown.txt";
            std::filesystem::rename(filename_check.c_str(), filename_unknown.c_str());
            check_file_exists = false;
        } else {
            check_file_distances = true;
        }
    }

    bool has_neg_cycle = !result.has_value();
    bool has_result_and_correct = false, has_result_and_incorrect = false;

    if (!has_neg_cycle) {
        has_result_and_correct = isResultCorrect(graph, result.value(), data.source);
        has_result_and_incorrect = !has_result_and_correct;
    }

    // Case 1/3: The check file does not exist
    if (!check_file_exists) {
        if (has_neg_cycle || has_result_and_correct) {
            std::ofstream ofs(filename_check);
            if (has_neg_cycle) {
                std::clog << "Contains a negative cycle. Don't know if it's correct." << std::endl;
                ofs << "-1";
           } else {
                std::clog << "The result is correct (no negative cycle)." << std::endl;
                print(result, ofs);
           }
        } else {
            std::clog << "The result is incorrect." << std::endl;
            std::ofstream ofs(wrong_filename_check);
            print(result, ofs);
            return false;
        }
    // Case 2/3: The check file exists, and it says there is a negative cycle
    } else if (check_file_neg_cycle) {
        if (has_neg_cycle) {
            std::clog << "It contains a negative cycle. It is probably correct." << std::endl;
        } else if (has_result_and_correct) {
            std::clog << "The result is correct (no negative cycle)." << std::endl;
            std::clog << "The algorithm before this thought there was a negative cycle." << std::endl;
            std::filesystem::remove(filename_check.c_str());
            std::ofstream ofs(filename_check);
            print(result, ofs);
        } else {
            std::clog << "The result is incorrect." << std::endl;
            std::clog << "The algorithm returned distances that are not correct. The check file suggests that there is a negative cycle." << std::endl;
            std::ofstream ofs(wrong_filename_check);
            print(result, ofs);
            return false;
        }
    // Case 3/3: The check file exists, and it contains the correct distances
    } else if (check_file_distances) {
        if (has_result_and_correct) {
            std::clog << "The result is correct (no negative cycle)." << std::endl;
        } else if (has_result_and_incorrect || has_neg_cycle) {
            std::ofstream ofs(wrong_filename_check);
            if (has_neg_cycle) {
                std::clog << "The result is incorrect." << std::endl;
                std::clog << "It says it contains a negative cycle, but there is a feasible solution" << std::endl;
            } else {
                std::clog << "The result is incorrect." << std::endl;
                std::clog << "The algorithm returned distances that are not correct." << std::endl;
            }
            print(result, ofs);
            return false;
        }
    }

    return true;
}


ExpTime timeQuery(SSSPData const& data, Graph& graph) {
    auto progressBar = [](double progress, unsigned int barLength) {
        int pos = barLength * progress;
        std::clog << "\r[" << std::string(pos, '=') << std::string(barLength - pos, ' ') << "] " << static_cast<int>(progress * 100.0) << " %" << std::flush;
    };

    // Running the first iteration separatly to check if the result is correct
    MEASUREMENT::start(EXP::INNER_LOOP_ALL);
    auto result = computeSSSP(data.algorithm, graph, data.source);
    MEASUREMENT::stop(EXP::INNER_LOOP_ALL);

    bool proceed = handleCorrectness(result, graph, data, graph.numberOfNodes());

    if (!proceed) {
        return {-1.0, -1.0, -1.0, -1.0};
    }

    // Proceeding with the other iterations

    progressBar(static_cast<double>(1) / data.iters, 50);

    for (unsigned int i = 1; i < data.iters; i++) {
        MEASUREMENT::start(EXP::INNER_LOOP_ALL);
        computeSSSP(data.algorithm, graph, data.source);
        MEASUREMENT::stop(EXP::INNER_LOOP_ALL);
        progressBar(static_cast<double>(i + 1) / data.iters, 50);
    }
    std::clog << std::endl << std::flush;

    std::ostringstream oss;
    MEASUREMENT::print(EXP::INNER_LOOP_ALL, oss);
    MEASUREMENT::reset(EXP::INNER_LOOP_ALL);
    return parseNumbers(oss.str());
}

ExpTime timeQuery(NegCycleData const& data, Graph& graph) {
    for (unsigned int i = 0; i < data.iters; i++) {
        MEASUREMENT::start(EXP::INNER_LOOP_ALL);
        negCycleDetection(data.algorithm, graph);
        MEASUREMENT::stop(EXP::INNER_LOOP_ALL);
    }

    std::ostringstream oss;
    MEASUREMENT::print(EXP::INNER_LOOP_ALL, oss);
    MEASUREMENT::reset(EXP::INNER_LOOP_ALL);
    return parseNumbers(oss.str());
}

}  // namespace

void runQueries(Queries& queries, std::string const& filename) {
    PRINT("Running queries...");

    // print results
    std::streambuf* buf;
    std::ofstream of;

    if (!filename.empty()) {
        of.open(filename);
        buf = of.rdbuf();
    } else {
        buf = std::cout.rdbuf();
    }

    std::ostream os(buf);

    auto print_result = [&](auto const& result) {
        print(result, os);
    };

    auto run_query = [&](auto& data) {
        auto& graph = queries.graphs.at(data.graph_filename);

        if (data.iters == 0) {
            auto res = Result(runQuery(data, graph));
            std::visit(print_result, res);
            return res;
        }
        auto res = Result(timeQuery(data, graph));
        std::visit(print_result, res);
        return res;
    };

    Results results;
    unsigned int query_num = 1;
    for (auto const& query_data : queries.data) {
        std::cout << "Running query " << query_num++ << "..." << std::endl;
        results.push_back(std::visit(run_query, query_data));
        std::cout << "Done." << std::endl;
    }

    // for (int i = 1; auto const& result : results) {
    //     PRINT("Result of query " << i++ << ":");
    //     std::visit(print_result, result);
    //     PRINT("");
    // }
}
