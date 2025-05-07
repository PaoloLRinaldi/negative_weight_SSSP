// To add a new measurement, add a new enum value to MEASUREMENT::EXP. This will
// be the handle to the measurement. Then you also have to add a tuple of the
// form (<name>, <handle>, <type>). The values mean the following:
//
// <name>: This is just a string that will be used when printing.
//
// <handle>: The handle which is used to reference the measurment in the code. This
//           has to be the same that was also added to the enum.
//
// <type>: There are serveral types of measurements.
//     * TIMER: Measures time using a start and stop function. Afterwards the sum
//              and mean can be calculated.
//     * INT: Accumulating int values. The sum and mean can be recalled from that.
//     * FLOAT: Same as INT just for float values.
//     * TIMER_DATA: Like TIMER but all the measurements are collected and thus
//                   more complex stats can be created like standard deviation.
//     * INT_DATA: Like TIMER_DATA but for INT.
//     * FLOAT_DATA: Like TIMER_DATA but for FLOAT.
//     * COUNTER: A simple counter which can just be increased.

// Add new measurements here ...
enum class MEASUREMENT::EXP {
    // DummyTimer1,
    // DummyTimer2,
    // DummyTimer3,
    // DummyTimer4,
    // DummyTimer5,
    // DummyCounter1,
    // DummyCounter2,
    // DummyCounter3,
    // DummyCounter4,
    // DummyCounter5,
    INNER_LOOP,
    INNER_LOOP_ALL,
    SAMPLE_VALUE,
    SAMPLE_VALUES,
    IF_COUNTER,
    CUT_EDGES,
    SCC,
    LAZY_END,
    REC,
    JUST_POTENTIAL,
    DIJKSTRA,
    INIT,
    GLOBAL_DECOMP,
    GLOBAL_RECURS,
    GLOBAL_FIX,
    LOCAL_ALG,
    LOCAL_POT,
    INN_REC,
    INN_FIX,
    INN_POT,
    LAST_LAZY,
};

inline auto MEASUREMENT::getMeasurements() -> Measurements {
    // ... and here
    return {
        // {"dummy timer 1", EXP::DummyTimer1, TIMER},
        // {"dummy timer 2", EXP::DummyTimer2, TIMER},
        // {"dummy timer 3", EXP::DummyTimer3, TIMER},
        // {"dummy timer 4", EXP::DummyTimer4, TIMER},
        // {"dummy timer 5", EXP::DummyTimer5, TIMER},
        // {"dummy counter 1", EXP::DummyCounter1, COUNTER},
        // {"dummy counter 2", EXP::DummyCounter2, COUNTER},
        // {"dummy counter 3", EXP::DummyCounter3, COUNTER},
        // {"dummy counter 4", EXP::DummyCounter4, COUNTER},
        // {"dummy counter 5", EXP::DummyCounter5, COUNTER},
        {"stupid rec", EXP::REC, TIMER_DATA},
        {"dijkstra after potential", EXP::DIJKSTRA, TIMER_DATA},
        {"initializing container", EXP::INIT, TIMER_DATA},
        {"time to compute potential", EXP::JUST_POTENTIAL, TIMER_DATA},
        {"computing scc timer", EXP::SCC, TIMER_DATA},
        {"fixing edges at the end lazy", EXP::LAZY_END, TIMER_DATA},
        {"cut edges timer", EXP::CUT_EDGES, TIMER_DATA},
        {"inner loop timer", EXP::INNER_LOOP, TIMER},
        {"inner loop all", EXP::INNER_LOOP_ALL, TIMER_DATA},
        {"sample value", EXP::SAMPLE_VALUE, INT},
        {"sample values", EXP::SAMPLE_VALUES, INT_DATA},
        {"mod if statement counter", EXP::IF_COUNTER, COUNTER},
        {"time to compute decomposition of entire graph", EXP::GLOBAL_DECOMP, TIMER_DATA},
        {"time to compute potential of each component", EXP::GLOBAL_RECURS, TIMER_DATA},
        {"time to fix edges of whole graph", EXP::GLOBAL_FIX, TIMER_DATA},
        {"time to compute potential of a single component", EXP::LOCAL_ALG, TIMER_DATA},
        {"time to fix potential of a single component", EXP::LOCAL_POT, TIMER_DATA},
        {"time inner recursion", EXP::INN_REC, TIMER_DATA},
        {"time inner fix", EXP::INN_FIX, TIMER_DATA},
        {"time inner potential", EXP::INN_POT, TIMER_DATA},
        {"last lazy", EXP::LAST_LAZY, TIMER_DATA},
    };
}
