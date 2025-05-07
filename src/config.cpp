#include "config.h"
#include <string>
#include <unordered_set>
#include <cctype>
#include "defs.h"

std::pair<std::string, std::string> parseString(const std::string& input) {
    std::pair<std::string, std::string> result;
    
    // Find the position of the "=" sign
    size_t pos = input.find("=");
    
    // Check if "=" sign exists and is not at the beginning or end
    if (pos == std::string::npos || pos == 0 || pos == input.length() - 1) {
        return result;
    }
    
    // Extract the strings before and after the "=" sign
    std::string part1 = input.substr(0, pos);
    std::string part2 = input.substr(pos + 1);
    
    // Check if part1 only has alphanumeric characters and "_", and doesn't start with a digit
    if (part1.empty()) {
        return result;
    }
    char first_char = part1[0];
    if (isdigit(first_char)) {
        return result;
    }
    for (char c : part1) {
        if (!isalnum(c) && c != '_') {
            return result;
        }
    }
    
    if (part2.empty()) {
        return result;
    }
// // Check if part2 only has digits and the first character is either a "-" or a digit
//     bool first_char_valid = (part2[0] == '-') || isdigit(part2[0]);
//     if (!first_char_valid) {
//         return result;
//     }
//     for (size_t i = 1; i < part2.size(); ++i) {
//         if (!isdigit(part2[i])) {
//             return result;
//         }
//     }
    
    result.first = part1;
    result.second = part2;
    
    return result;
}

namespace config {
    int use_lazy = 1;  // 0 -> use GOR, 1 -> use LazyDijkstra
    int init_kappa = 0;  // 0 -> use number of nodes, 1 -> use infinity
    int k_factor = 1;  // by what factor to reduce the number of Dijkstra calls
    int rand_label = 0;  // 0 -> label light nodes with Dijkstra, 1 -> label light nodes randomly
    int fixdagedges = 1;  // 1 -> normal fixdagedges version, 2 -> old fixdagedges version
    int cutedges = 1;  // 1 -> normal cutedges version, 2 -> new cutedges version, 3 -> cut edges in half, 4 -> newer cutedges version
    int rec_limit = 100;  // recursion limit
    int cutedgesseed = 1234;  // cut edges seed
    int diam_apprx = 0;  // 0 -> normal, 1 -> approximate diameter
    int ol_seed = 1234;  // out light labeling seed
    int il_seed = 12134;  // in light labeling seed
    int eg_sort_scc = 0;  // 0 -> no edge sorting during SCC, 1 -> edge sorting during SCC
    std::string shift_filename = std::string();

    int setup_config(int argc, char** argv) {

        auto assign_value = [](std::string const& arg, std::string const& val) {
            if (arg == "use_lazy") {
                use_lazy = std::stoi(val);
                if (use_lazy != 0 && use_lazy != 1) {
                    ERROR("use_lazy must be 0 or 1.");
                }
            } else if (arg == "init_kappa") {
                init_kappa = std::stoi(val);
                if (init_kappa != 0 && init_kappa != 1) {
                    ERROR("init_kappa must be 0 or 1.");
                }
            } else if (arg == "k_factor") {
                k_factor = std::stoi(val);
                if (k_factor < 1) {
                    ERROR("k_factor must be >= 1.");
                }
            } else if (arg == "rand_label") {
                rand_label = std::stoi(val);
                if (rand_label != 0 && rand_label != 1) {
                    ERROR("rand_label must be 0 or 1.");
                }
            } else if (arg == "fixdagedges") {
                fixdagedges = std::stoi(val);
                if (fixdagedges != 1 && fixdagedges != 2) {
                    ERROR("fixdagedges must be 1 or 2.");
                }
            } else if (arg == "cutedges") {
                cutedges = std::stoi(val);
                if (cutedges < 1 && cutedges > 5) {
                    ERROR("cutedges must be betwee 1 and 4.");
                }
            } else if (arg == "rec_limit") {
                rec_limit = std::stoi(val);
                if (rec_limit < 0) {
                    ERROR("rec_limit must be >= 0.");
                }
            } else if (arg == "cutedgesseed") {
                cutedgesseed = std::stoi(val);
                if (cutedgesseed < 0) {
                    ERROR("cutedgesseed must be >= 0.");
                }
            } else if (arg == "diam_apprx") {
                diam_apprx = std::stoi(val);
                if (diam_apprx != 0 && diam_apprx != 1) {
                    ERROR("diam_apprx must be 0 or 1.");
                }
            } else if (arg == "ol_seed") {
                ol_seed = std::stoi(val);
                if (ol_seed < 0) {
                    ERROR("ol_seed must be >= 0.");
                }
            } else if (arg == "il_seed") {
                il_seed = std::stoi(val);
                if (il_seed < 0) {
                    ERROR("il_seed must be >= 0.");
                }
            } else if (arg == "eg_sort_scc") {
                eg_sort_scc = std::stoi(val);
                if (eg_sort_scc != 0 && eg_sort_scc != 1) {
                    ERROR("eg_sort_scc must be 0 or 1.");
                }
            } else if (arg == "shift_filename") {
                shift_filename = val;
            } else {
                ERROR("Unknown argument: " << arg);
            }
        };

        std::unordered_set<std::string> assigned;

        for (int i = 1; i < argc; i++) {
            std::pair<std::string, std::string> parsedStrings = parseString(argv[i]);

            if (parsedStrings.first.empty() || parsedStrings.second.empty()) {
                // ERROR("Invalid argument: " << argv[i]);
                return i;
            }

            if (assigned.find(parsedStrings.first) != assigned.end()) {
                ERROR("Duplicate argument: " << parsedStrings.first);
            }

            assigned.insert(parsedStrings.first);
            assign_value(parsedStrings.first, parsedStrings.second);
        }
        return argc;
    }
}

