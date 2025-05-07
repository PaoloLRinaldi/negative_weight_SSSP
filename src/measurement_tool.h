#pragma once

// Only measure when not in debug mode
#ifdef NDEBUG
#define MEASURE
#endif

// Always use asserts as measurements slow down the code anyway
#ifdef NDEBUG
#undef NDEBUG
#include <cassert>
#define NDEBUG
#else
#include <cassert>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

//
// Base structure to make things work the way I want them to. The important
// stuff happens at the end of this file.
//

class MEASUREMENT {
   public:
    enum class EXP : int;

   private:
    enum Type {
        TIMER,
        INT,
        FLOAT,
        TIMER_DATA,
        INT_DATA,
        FLOAT_DATA,
        COUNTER,
    };

    struct Measurement {
        std::string name;
        EXP id;
        Type type;
    };
    using Measurements = std::vector<Measurement>;
    static Measurements getMeasurements();

    class MeasurementTool {
       public:
        using ID = EXP;
        using Name = std::string;

        using Time = double;
        using Int = int64_t;
        using Float = double;
        using Times = std::vector<Time>;
        using Counter = uint64_t;

        using hrc = std::chrono::high_resolution_clock;
        using time_point = hrc::time_point;
        static constexpr hrc::time_point invalid_time_point = hrc::time_point::max();
        using ns = std::chrono::nanoseconds;

        MeasurementTool(Measurements const& measurements);

        //
        // different entry types
        //

        struct AbstractEntry {
            virtual void print(std::ostream& os = std::cout) const = 0;
            virtual void reset() = 0;
        };

        struct TimeEntry : public AbstractEntry {
            Name const name;
            Time value = 0;
            hrc::time_point start = invalid_time_point;
            std::size_t number_of_values = 0;

            TimeEntry(Name const& name) : name(name) {}
            double mean() const { return (double)value / number_of_values; }
            void print(std::ostream& os = std::cout) const override {
                if (number_of_values == 0) {
                    os << name << ": NO DATA\n";
                    return;
                }
                os << std::setprecision(5)
                          << name << ": sum = " << value / 1000000. << " ms, mean = "
                          << mean() / 1000000. << " ms\n";
            }

            void reset() override {
                value = 0;
                number_of_values = 0;
                start = invalid_time_point;
            }
        };

        template <typename T>
        struct AccEntry : public AbstractEntry {
            Name const name;
            T value = 0;
            std::size_t number_of_values = 0;

            AccEntry(Name const& name) : name(name) {}
            double mean() const { return (double)value / number_of_values; }
            void print(std::ostream& os = std::cout) const override {
                if (number_of_values == 0) {
                    os << name << ": NO DATA\n";
                    return;
                }
                os << std::setprecision(5)
                          << name << ": sum = " << value << ", mean = " << mean() << "\n";
            }
            void reset() override {
                value = 0;
                number_of_values = 0;
            }
        };
        using IntEntry = AccEntry<Int>;
        using FloatEntry = AccEntry<Float>;

        struct TimeDataEntry : public AbstractEntry {
            Name const name;
            Times values;
            hrc::time_point start = invalid_time_point;

            TimeDataEntry(Name const& name) : name(name) {}
            Time mean() const {
                return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
            }
            Time stddev() const {
                Time sum = 0.;
                auto mean_value = mean();
                for (auto c : values) {
                    sum += (c - mean_value) * (c - mean_value);
                }
                return std::sqrt(sum / values.size());
            }
            // the following two functions should only be called when values is not empty
            Time min() const {
                return *std::min_element(values.begin(), values.end());
            }
            Time max() const {
                return *std::max_element(values.begin(), values.end());
            }
            void print(std::ostream& os = std::cout) const override {
                if (values.empty()) {
                    os << name << ": NO DATA\n";
                    return;
                }
                os << std::setprecision(5)
                          << name << ": mean = " << mean() / 1000000. << " ms, stddev = "
                          << stddev() / 1000000. << " ms, min = " << min() / 1000000.
                          << " ms, max = " << max() / 1000000. << " ms\n";
            }
            void reset() override {
                values.clear();
                start = invalid_time_point;
            }
        };

        template <typename T>
        struct DataEntry : public AbstractEntry {
            Name const name;
            std::vector<T> values;

            DataEntry(Name const& name) : name(name) {}
            double mean() const {
                return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
            }
            double stddev() const {
                T sum = 0.;
                auto mean_value = mean();
                for (auto c : values) {
                    sum += (c - mean_value) * (c - mean_value);
                }
                return std::sqrt(sum / values.size());
            }
            // the following two functions should only be called when values is not empty
            double min() const {
                return *std::min_element(values.begin(), values.end());
            }
            double max() const {
                return *std::max_element(values.begin(), values.end());
            }
            void print(std::ostream& os = std::cout) const override {
                if (values.empty()) {
                    os << name << ": NO DATA\n";
                    return;
                }
                os << std::setprecision(5)
                          << name << ": mean = " << mean() << ", stddev = " << stddev()
                          << ", min = " << min() << ", max = " << max() << "\n";
            }
            void reset() override { values.clear(); }
        };
        using IntDataEntry = DataEntry<Int>;
        using FloatDataEntry = DataEntry<Float>;

        struct CounterEntry : public AbstractEntry {
            Name const name;
            Counter value = 0;

            CounterEntry(Name const& name) : name(name) {}
            void print(std::ostream& os = std::cout) const override { os << name << ": " << value << "\n"; }
            void reset() override { value = 0; }
        };

        //
        // functions for adding measurement information to the entries
        //

        void start(ID id);
        void stop(ID id);
        void addInt(ID id, Int sample);
        void addFloat(ID id, Float sample);
        void inc(ID id);
        void print(std::ostream& os = std::cout);
        void print(ID id, std::ostream& os = std::cout);
        void reset();
        void reset(ID id);

       private:
        //
        // data of the measurements
        //

        std::vector<TimeEntry> time_entries;
        std::vector<IntEntry> int_entries;
        std::vector<FloatEntry> float_entries;
        std::vector<TimeDataEntry> time_data_entries;
        std::vector<IntDataEntry> int_data_entries;
        std::vector<FloatDataEntry> float_data_entries;
        std::vector<CounterEntry> counter_entries;

        std::vector<std::size_t> id_to_index;
        std::vector<Type> id_to_type;

        // helper functions
        std::size_t toIndex(ID id) const { return id_to_index[(std::size_t)id]; }
        Type toType(ID id) const { return id_to_type[(std::size_t)id]; }
        AbstractEntry& getEntry(Type type, std::size_t index) {
            switch (type) {
                case TIMER:
                    return time_entries[index];
                case INT:
                    return int_entries[index];
                case FLOAT:
                    return float_entries[index];
                case TIMER_DATA:
                    return time_data_entries[index];
                case INT_DATA:
                    return int_data_entries[index];
                case FLOAT_DATA:
                    return float_data_entries[index];
                case COUNTER:
                    return counter_entries[index];
                default:
                    std::cerr << "ERROR: unknown type\n";
                    exit(1);
            }
        }
    };

#ifdef MEASURE
    static MeasurementTool& getSingleton() {
        static MeasurementTool singleton(getMeasurements());
        return singleton;
    }
#endif

   public:
    using Name = MeasurementTool::Name;
    using ID = MeasurementTool::ID;
    using Int = MeasurementTool::Int;
    using Float = MeasurementTool::Float;

    MEASUREMENT() = delete;
    ~MEASUREMENT() = delete;

    // If you add a new function here, don't forget the preprocessor statements!

    static void start(ID id) {
#ifdef MEASURE
        getSingleton().start(id);
#endif
    }
    static void stop(ID id) {
#ifdef MEASURE
        getSingleton().stop(id);
#endif
    }
    static void addInt(ID id, Int sample) {
#ifdef MEASURE
        getSingleton().addInt(id, sample);
#endif
    }
    static void addFloat(ID id, Float sample) {
#ifdef MEASURE
        getSingleton().addFloat(id, sample);
#endif
    }
    static void inc(ID id) {
#ifdef MEASURE
        getSingleton().inc(id);
#endif
    }
    static void print(std::ostream& os = std::cout) {
#ifdef MEASURE
        getSingleton().print(os);
#endif
    }
    static void print(ID id, std::ostream& os = std::cout) {
#ifdef MEASURE
        getSingleton().print(id, os);
#endif
    }
    static void reset() {
#ifdef MEASURE
        getSingleton().reset();
#endif
    }
    static void reset(ID id) {
#ifdef MEASURE
        getSingleton().reset(id);
#endif
    }
};

// yes, this include has to happen exactly here.
#include "measurements.h"

// shorten enum class prefix
using EXP = MEASUREMENT::EXP;

//
// Implementations of measurement tool functions. Here is where the real stuff happens.
//

inline MEASUREMENT::MeasurementTool::MeasurementTool(Measurements const& measurements) {
    id_to_index.resize(measurements.size());
    id_to_type.resize(measurements.size());

    for (auto const& measurement : measurements) {
        switch (measurement.type) {
            case TIMER:
                id_to_index[(std::size_t)measurement.id] = time_entries.size();
                time_entries.emplace_back(measurement.name);
                break;
            case INT:
                id_to_index[(std::size_t)measurement.id] = int_entries.size();
                int_entries.emplace_back(measurement.name);
                break;
            case FLOAT:
                id_to_index[(std::size_t)measurement.id] = float_entries.size();
                float_entries.emplace_back(measurement.name);
                break;
            case TIMER_DATA:
                id_to_index[(std::size_t)measurement.id] = time_data_entries.size();
                time_data_entries.emplace_back(measurement.name);
                break;
            case INT_DATA:
                id_to_index[(std::size_t)measurement.id] = int_data_entries.size();
                int_data_entries.emplace_back(measurement.name);
                break;
            case FLOAT_DATA:
                id_to_index[(std::size_t)measurement.id] = float_data_entries.size();
                float_data_entries.emplace_back(measurement.name);
                break;
            case COUNTER:
                id_to_index[(std::size_t)measurement.id] = counter_entries.size();
                counter_entries.emplace_back(measurement.name);
                break;
            default:
                std::cerr << "ERROR: unknown type\n";
                exit(1);
        }

        id_to_type[(std::size_t)measurement.id] = measurement.type;
    }
}

inline void MEASUREMENT::MeasurementTool::start(ID id) {
    assert(toType(id) == TIMER || toType(id) == TIMER_DATA);
    if (toType(id) == TIMER) {
        auto& start = time_entries[toIndex(id)].start;
        assert(start == hrc::time_point(invalid_time_point));  // assert that it's not started twice
        start = hrc::now();
    } else {
        auto& start = time_data_entries[toIndex(id)].start;
        assert(start == hrc::time_point(invalid_time_point));  // assert that it's not started twice
        start = hrc::now();
    }
}

inline void MEASUREMENT::MeasurementTool::stop(ID id) {
    assert(toType(id) == TIMER || toType(id) == TIMER_DATA);
    if (toType(id) == TIMER) {
        auto& entry = time_entries[toIndex(id)];
        assert(entry.start != hrc::time_point(invalid_time_point));  // assert that it was started before
        entry.value += std::chrono::duration_cast<ns>(hrc::now() - entry.start).count();
        entry.start = invalid_time_point;
        ++entry.number_of_values;
    } else {
        auto& entry = time_data_entries[toIndex(id)];
        assert(entry.start != hrc::time_point(invalid_time_point));  // assert that it was started before
        entry.values.push_back(std::chrono::duration_cast<ns>(hrc::now() - entry.start).count());
        entry.start = invalid_time_point;
    }
}

inline void MEASUREMENT::MeasurementTool::addInt(ID id, Int sample) {
    assert(toType(id) == INT || toType(id) == INT_DATA);
    if (toType(id) == INT) {
        auto& entry = int_entries[toIndex(id)];
        entry.value += sample;
        ++entry.number_of_values;
    } else {
        auto& entry = int_data_entries[toIndex(id)];
        entry.values.push_back(sample);
    }
}

inline void MEASUREMENT::MeasurementTool::addFloat(ID id, Float sample) {
    assert(toType(id) == FLOAT || toType(id) == FLOAT_DATA);
    if (toType(id) == FLOAT) {
        auto& entry = float_entries[toIndex(id)];
        entry.value += sample;
        ++entry.number_of_values;
    } else {
        auto& entry = float_data_entries[toIndex(id)];
        entry.values.push_back(sample);
    }
}

inline void MEASUREMENT::MeasurementTool::inc(ID id) {
    assert(toType(id) == COUNTER);
    auto& entry = counter_entries[toIndex(id)];
    ++entry.value;
}

inline void MEASUREMENT::MeasurementTool::print(std::ostream& os) {
    for (auto& entry : time_entries) {
        entry.print(os);
    }
    for (auto& entry : int_entries) {
        entry.print(os);
    }
    for (auto& entry : float_entries) {
        entry.print(os);
    }
    for (auto& entry : time_data_entries) {
        entry.print(os);
    }
    for (auto& entry : int_data_entries) {
        entry.print(os);
    }
    for (auto& entry : float_data_entries) {
        entry.print(os);
    }
    for (auto& entry : counter_entries) {
        entry.print(os);
    }
}

inline void MEASUREMENT::MeasurementTool::print(ID id, std::ostream& os) {
    getEntry(toType(id), toIndex(id)).print(os);
}

inline void MEASUREMENT::MeasurementTool::reset() {
    for (auto& entry : time_entries) {
        entry.reset();
    }
    for (auto& entry : int_entries) {
        entry.reset();
    }
    for (auto& entry : float_entries) {
        entry.reset();
    }
    for (auto& entry : time_data_entries) {
        entry.reset();
    }
    for (auto& entry : int_data_entries) {
        entry.reset();
    }
    for (auto& entry : float_data_entries) {
        entry.reset();
    }
    for (auto& entry : counter_entries) {
        entry.reset();
    }
}

inline void MEASUREMENT::MeasurementTool::reset(ID id) {
    getEntry(toType(id), toIndex(id)).reset();
}
