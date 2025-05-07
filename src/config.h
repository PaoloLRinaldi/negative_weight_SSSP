#pragma once

#include <string>

namespace config {
    extern int use_lazy;      // 0 -> use GOR, 1 -> use LazyDijkstra //
    extern int init_kappa;    // 0 -> use number of nodes, 1 -> use infinity xx check on my own
    extern int k_factor;      // by what factor to reduce the number of Dijkstra calls //
    extern int rand_label;    // 0 -> label light nodes with Dijkstra, 1 -> label light nodes randomly
    extern int fixdagedges;   // 1 -> normal fixdagedges version, 2 -> old fixdagedges version
    extern int cutedges;      // 1 -> normal cutedges version, 2 -> new cutedges version, 3 -> cut edges in half, 4/5 -> newer cutedges version
    extern int rec_limit;     // recursion limit xx check on my own
    extern int cutedgesseed;  // cut edges seed
    extern int diam_apprx;    // 0 -> normal, 1 -> approximate diameter xx// check on my own
    extern int ol_seed;       // out light labeling seed
    extern int il_seed;       // in light labeling seed
    extern int eg_sort_scc;   // 0 -> no edge sorting during SCC, 1 -> edge sorting during SCC xx check on my own
    extern std::string shift_filename;  // name of shifted graph

    int setup_config(int argc, char** argv);
}
