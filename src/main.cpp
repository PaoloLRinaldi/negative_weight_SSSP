#include <string>

#include "algorithms.h"
#include "defs.h"
#include "queries.h"
#include "config.h"

void PrintHelp() {
    PRINT("Usage: ./Main [config1=value1 config2=value2 ...] <queries_file> [<output_file>]");
}

struct Args {
    std::string queries_filename;
    std::string output_filename = std::string();
};

Args parseArgs(int shift, int argc, char** argv) {
    int shift_argc = argc - shift;
    if (shift_argc != 2 && shift_argc != 3) {
        PrintHelp();
        ERROR("Wrong number of arguments.");
    }

    auto queries_filename = std::string(argv[1 + shift]);
    std::string output_filename = std::string();
    if (shift_argc == 3) {
        output_filename = std::string(argv[2 + shift]);
    }

    return {queries_filename, output_filename};
}

int main(int argc, char** argv) {
    int shift = config::setup_config(argc, argv) - 1;
    auto args = parseArgs(shift, argc, argv);
    auto queries = readQueries(args.queries_filename);
    runQueries(queries, args.output_filename);
}
