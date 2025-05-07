#pragma once

#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#define MEASURE
#include "algorithms.h"
#include "graph.h"
#include "measurement_tool.h"

// QueryType

enum class QueryType {
    SSSP,
    NegCycle,
};

// QueryData

struct SSSPData {
    std::string graph_filename;
    NodeID source;
    SSSPAlg algorithm;
    unsigned int iters;
};

struct NegCycleData {
    std::string graph_filename;
    NegCycleAlg algorithm;
    unsigned int iters;
};

struct ExpTime {
    double avg;
    double std;
    double min;
    double max;
};

using QueryData = std::variant<SSSPData, NegCycleData>;

// Queries & Results

struct Queries {
    std::vector<QueryData> data;
    std::unordered_map<std::string, Graph> graphs;
};

using Result = std::variant<std::optional<Distances>, bool, ExpTime>;
using Results = std::vector<Result>;

// functions

// file structure is lines with <graph_file> <source>
Queries readQueries(std::string const& filename);
void runQueries(Queries& queries, std::string const& filename = std::string());
