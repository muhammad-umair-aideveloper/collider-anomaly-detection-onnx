/**
 * @file inference.cpp
 * @brief High-performance ONNX Runtime C++ single-event inference and benchmark runner.
 *
 * This program loads the exported autoencoder ONNX model, ingests collision events from CSV,
 * runs single-event inference with batch size 1 on CPU, classifies events as NORMAL or ANOMALY
 * relative to the recorded 95th percentile validation threshold, and logs error outputs.
 *
 * It supports a dedicated benchmark mode (--bench) measuring pure single-event inference
 * latency (mean, median, p99, min, max) with microsecond precision over thousands of runs.
 */

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

// Constants
constexpr size_t EXPECTED_NUM_FEATURES = 28;
constexpr size_t EXPECTED_CSV_COLUMNS = 29; // Column 0: label, Columns 1..28: features

#ifndef BUILD_CONFIG_STRING
#define BUILD_CONFIG_STRING "Unknown"
#endif

#if defined(_MSC_VER)
    #define COMPILER_INFO_STRING "MSVC " _CRT_STRINGIZE(_MSC_VER)
#elif defined(__clang__)
    #define COMPILER_INFO_STRING "Clang " __clang_version__
#elif defined(__GNUC__)
    #define COMPILER_INFO_STRING "GCC " __VERSION__
#else
    #define COMPILER_INFO_STRING "Unknown Compiler"
#endif

/**
 * @brief Represents a single collision event with label and 28 raw features.
 */
struct Event {
    int label{0};
    std::array<float, EXPECTED_NUM_FEATURES> features{};
};

/**
 * @brief CLI configuration options.
 */
struct Options {
    std::string model_path{"models/detector_model.onnx"};
    std::string csv_path{"data/sample_events.csv"};
    std::string threshold_file{"models/threshold.txt"};
    std::string out_path{"results/cpp_errors.txt"};
    std::string bench_json{"results/cpp_benchmark.json"};
    size_t num_events{100};
    bool benchmark_mode{false};
    size_t warmup_runs{100};
    size_t timed_runs{10000};
};

/**
 * @brief Latency benchmark statistical summary.
 */
struct BenchmarkStats {
    double mean_us{0.0};
    double median_us{0.0};
    double p99_us{0.0};
    double min_us{0.0};
    double max_us{0.0};
};

/**
 * @brief Print usage information.
 */
void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options]\n\n"
              << "Options:\n"
              << "  --model <path>           Path to ONNX model (default: models/detector_model.onnx)\n"
              << "  --csv <path>             Path to input event CSV (default: data/sample_events.csv)\n"
              << "  --threshold-file <path>  Path to threshold text file (default: models/threshold.txt)\n"
              << "  --n <int>                Number of events to load/evaluate (default: 100)\n"
              << "  --out <path>             Destination for error values (default: results/cpp_errors.txt)\n"
              << "  --bench                  Enable benchmark latency measurement mode\n"
              << "  --bench-json <path>      Destination for benchmark JSON report (default: results/cpp_benchmark.json)\n"
              << "  --warmup <int>           Number of untimed warmup iterations (default: 100)\n"
              << "  --runs <int>             Number of timed inference runs (default: 10000)\n"
              << "  --help, -h               Print this help message\n";
}

/**
 * @brief Parse CLI arguments into Options struct.
 */
bool parse_arguments(int argc, char* argv[], Options& opts) {
    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--model") {
            if (++i >= argc) { std::cerr << "Error: --model requires a path argument\n"; return false; }
            opts.model_path = argv[i];
        } else if (arg == "--csv") {
            if (++i >= argc) { std::cerr << "Error: --csv requires a path argument\n"; return false; }
            opts.csv_path = argv[i];
        } else if (arg == "--threshold-file") {
            if (++i >= argc) { std::cerr << "Error: --threshold-file requires a path argument\n"; return false; }
            opts.threshold_file = argv[i];
        } else if (arg == "--n") {
            if (++i >= argc) { std::cerr << "Error: --n requires an integer argument\n"; return false; }
            try {
                opts.num_events = std::stoul(argv[i]);
            } catch (const std::exception&) {
                std::cerr << "Error: Invalid integer for --n: " << argv[i] << "\n";
                return false;
            }
        } else if (arg == "--out") {
            if (++i >= argc) { std::cerr << "Error: --out requires a path argument\n"; return false; }
            opts.out_path = argv[i];
        } else if (arg == "--bench") {
            opts.benchmark_mode = true;
        } else if (arg == "--bench-json") {
            if (++i >= argc) { std::cerr << "Error: --bench-json requires a path argument\n"; return false; }
            opts.bench_json = argv[i];
        } else if (arg == "--warmup") {
            if (++i >= argc) { std::cerr << "Error: --warmup requires an integer argument\n"; return false; }
            try {
                opts.warmup_runs = std::stoul(argv[i]);
            } catch (const std::exception&) {
                std::cerr << "Error: Invalid integer for --warmup: " << argv[i] << "\n";
                return false;
            }
        } else if (arg == "--runs") {
            if (++i >= argc) { std::cerr << "Error: --runs requires an integer argument\n"; return false; }
            try {
                opts.timed_runs = std::stoul(argv[i]);
            } catch (const std::exception&) {
                std::cerr << "Error: Invalid integer for --runs: " << argv[i] << "\n";
                return false;
            }
        } else {
            std::cerr << "Error: Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return false;
        }
    }
    return true;
}

/**
 * @brief Load decision threshold from text file.
 */
float load_threshold(const std::string& path_str) {
    std::ifstream file(path_str);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open threshold file: " + path_str);
    }
    float threshold = 0.0f;
    if (!(file >> threshold)) {
        throw std::runtime_error("Failed to read float threshold from: " + path_str);
    }
    return threshold;
}

/**
 * @brief Read collision events from CSV handling \r\n, validating 29 fields per row.
 */
std::vector<Event> load_events_csv(const std::string& path_str, size_t max_events) {
    std::ifstream file(path_str);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open event CSV file: " + path_str);
    }

    std::vector<Event> events;
    events.reserve(max_events);

    std::string line;
    size_t line_number = 0;

    while (events.size() < max_events && std::getline(file, line)) {
        ++line_number;
        // Trim trailing \r if present
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }

        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }

        if (tokens.size() != EXPECTED_CSV_COLUMNS) {
            std::ostringstream oss;
            oss << "CSV parse error at line " << line_number << ": expected "
                << EXPECTED_CSV_COLUMNS << " fields, found " << tokens.size();
            throw std::runtime_error(oss.str());
        }

        Event ev;
        // Parse label
        char* end_ptr = nullptr;
        long parsed_label = std::strtol(tokens[0].c_str(), &end_ptr, 10);
        if (end_ptr == tokens[0].c_str() || *end_ptr != '\0' || (parsed_label != 0 && parsed_label != 1)) {
            std::ostringstream oss;
            oss << "CSV parse error at line " << line_number << ": invalid label '" << tokens[0] << "'";
            throw std::runtime_error(oss.str());
        }
        ev.label = static_cast<int>(parsed_label);

        // Parse 28 features
        for (size_t f = 0; f < EXPECTED_NUM_FEATURES; ++f) {
            const std::string& feat_str = tokens[f + 1];
            end_ptr = nullptr;
            float val = std::strtof(feat_str.c_str(), &end_ptr);
            if (end_ptr == feat_str.c_str() || *end_ptr != '\0') {
                std::ostringstream oss;
                oss << "CSV parse error at line " << line_number << ", feature " << f
                    << ": failed to parse float '" << feat_str << "'";
                throw std::runtime_error(oss.str());
            }
            ev.features[f] = val;
        }

        events.push_back(ev);
    }

    return events;
}

/**
 * @brief Compute statistical metrics (mean, median, p99, min, max) from latency vector.
 */
BenchmarkStats compute_benchmark_statistics(std::vector<double>& latencies) {
    BenchmarkStats stats;
    if (latencies.empty()) {
        return stats;
    }

    std::sort(latencies.begin(), latencies.end());

    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    stats.mean_us = sum / static_cast<double>(latencies.size());
    stats.min_us = latencies.front();
    stats.max_us = latencies.back();

    size_t n = latencies.size();
    if (n % 2 == 0) {
        stats.median_us = 0.5 * (latencies[n / 2 - 1] + latencies[n / 2]);
    } else {
        stats.median_us = latencies[n / 2];
    }

    // 99th percentile index
    size_t p99_idx = static_cast<size_t>(std::ceil(0.99 * static_cast<double>(n))) - 1;
    if (p99_idx >= n) {
        p99_idx = n - 1;
    }
    stats.p99_us = latencies[p99_idx];

    return stats;
}

/**
 * @brief Write benchmark metrics to JSON file.
 */
void write_benchmark_json(
    const std::string& json_path_str,
    const BenchmarkStats& stats,
    size_t warmup_runs,
    size_t timed_runs,
    size_t num_distinct_events
) {
    fs::path p(json_path_str);
    if (p.has_parent_path()) {
        fs::create_directories(p.parent_path());
    }

    std::ofstream f(p);
    if (!f.is_open()) {
        throw std::runtime_error("Could not open benchmark JSON for writing: " + json_path_str);
    }

    f << "{\n"
      << "  \"engine\": \"ONNX Runtime C++\",\n"
      << "  \"version\": \"" << Ort::GetVersionString() << "\",\n"
      << "  \"compiler\": \"" << COMPILER_INFO_STRING << "\",\n"
      << "  \"build_type\": \"" << BUILD_CONFIG_STRING << "\",\n"
      << "  \"batch_size\": 1,\n"
      << "  \"intra_op_threads\": 1,\n"
      << "  \"inter_op_threads\": 1,\n"
      << "  \"distinct_events\": " << num_distinct_events << ",\n"
      << "  \"warmup_runs\": " << warmup_runs << ",\n"
      << "  \"timed_runs\": " << timed_runs << ",\n"
      << "  \"latency_us\": {\n"
      << "    \"mean\": " << std::fixed << std::setprecision(4) << stats.mean_us << ",\n"
      << "    \"median\": " << std::fixed << std::setprecision(4) << stats.median_us << ",\n"
      << "    \"p99\": " << std::fixed << std::setprecision(4) << stats.p99_us << ",\n"
      << "    \"min\": " << std::fixed << std::setprecision(4) << stats.min_us << ",\n"
      << "    \"max\": " << std::fixed << std::setprecision(4) << stats.max_us << "\n"
      << "  }\n"
      << "}\n";
}

int main(int argc, char* argv[]) {
    try {
        Options opts;
        if (!parse_arguments(argc, argv, opts)) {
            return 1;
        }

        std::cout << "===============================================================\n"
                  << "COLLIDER ANOMALY DETECTION: C++ ONNX RUNTIME INFERENCE\n"
                  << "===============================================================\n"
                  << "Model Path:      " << opts.model_path << "\n"
                  << "Input CSV:       " << opts.csv_path << "\n"
                  << "Threshold File:  " << opts.threshold_file << "\n"
                  << "Events to Load:  " << opts.num_events << "\n"
                  << "Mode:            " << (opts.benchmark_mode ? "BENCHMARK" : "NORMAL") << "\n"
                  << "Compiler:        " << COMPILER_INFO_STRING << "\n"
                  << "Build Config:    " << BUILD_CONFIG_STRING << "\n"
                  << "ORT Version:     " << Ort::GetVersionString() << "\n"
                  << "===============================================================\n";

#ifndef NDEBUG
        if (opts.benchmark_mode) {
            std::cerr << "\n"
                      << "***************************************************************\n"
                      << "WARNING: NDEBUG is NOT defined! This is a DEBUG build.\n"
                      << "Benchmark timings in Debug mode are invalid and non-representative.\n"
                      << "Re-build in Release mode before publishing any benchmark figures.\n"
                      << "***************************************************************\n\n";
        }
#endif

        // 1. Read threshold
        float threshold = load_threshold(opts.threshold_file);
        std::cout << "Loaded Decision Threshold: " << std::setprecision(6) << std::fixed << threshold << "\n";

        // 2. Pre-load events into memory
        auto events = load_events_csv(opts.csv_path, opts.num_events);
        std::cout << "Loaded " << events.size() << " events into memory from CSV.\n\n";
        if (events.empty()) {
            std::cerr << "Error: No events loaded.\n";
            return 1;
        }

        // 3. Initialize ONNX Runtime Session
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ColliderAnomalyDetector");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetInterOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Path to model (using std::filesystem::path for native wchar_t handling on Windows)
        fs::path model_fs_path(opts.model_path);
        if (!fs::exists(model_fs_path)) {
            std::cerr << "Error: Model file does not exist: " << opts.model_path << "\n";
            return 1;
        }

        Ort::Session session(env, model_fs_path.c_str(), session_options);

        // 4. Query input/output names and validate tensor shape
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
        auto output_name_alloc = session.GetOutputNameAllocated(0, allocator);
        const char* input_name = input_name_alloc.get();
        const char* output_name = output_name_alloc.get();

        auto type_info = session.GetInputTypeInfo(0);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_shape = tensor_info.GetShape();

        if (input_shape.size() != 2 || input_shape[0] != 1 || input_shape[1] != EXPECTED_NUM_FEATURES) {
            std::cerr << "Error: Model input shape mismatch. Expected [1, 28], got [";
            for (size_t i = 0; i < input_shape.size(); ++i) {
                std::cerr << (i > 0 ? ", " : "") << input_shape[i];
            }
            std::cerr << "]\n";
            return 1;
        }

        std::cout << "ONNX Session Initialized:\n"
                  << "  Input Name:  " << input_name << " (Shape: [1, 28])\n"
                  << "  Output Name: " << output_name << " (Shape: [1])\n\n";

        // Setup memory info and tensor shapes
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::array<int64_t, 2> tensor_shape = {1, EXPECTED_NUM_FEATURES};
        const char* input_names[] = {input_name};
        const char* output_names[] = {output_name};

        // 5. Normal Mode execution
        if (!opts.benchmark_mode) {
            std::vector<float> recorded_errors;
            recorded_errors.reserve(events.size());

            size_t count_normal = 0;
            size_t count_anomaly = 0;
            size_t anomaly_sig = 0;
            size_t anomaly_bg = 0;

            std::cout << "--- Inference Results ---\n";
            for (size_t i = 0; i < events.size(); ++i) {
                const auto& ev = events[i];

                Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                    memory_info,
                    const_cast<float*>(ev.features.data()),
                    ev.features.size(),
                    tensor_shape.data(),
                    tensor_shape.size()
                );

                auto output_tensors = session.Run(
                    Ort::RunOptions{nullptr},
                    input_names,
                    &input_tensor,
                    1,
                    output_names,
                    1
                );

                float error = output_tensors[0].GetTensorMutableData<float>()[0];
                recorded_errors.push_back(error);

                bool is_anomaly = (error > threshold);
                if (is_anomaly) {
                    count_anomaly++;
                    if (ev.label == 1) anomaly_sig++;
                    else anomaly_bg++;
                } else {
                    count_normal++;
                }

                std::cout << "Event " << std::setw(3) << i
                          << " | Label: " << ev.label
                          << " | Error: " << std::setw(11) << std::setprecision(7) << std::fixed << error
                          << " | Decision: " << (is_anomaly ? "ANOMALY" : "NORMAL") << "\n";
            }

            // Summary
            std::cout << "\n===============================================================\n"
                      << "INFERENCE SUMMARY\n"
                      << "===============================================================\n"
                      << "Total Events Evaluated:  " << events.size() << "\n"
                      << "NORMAL  Decisions:       " << count_normal << "\n"
                      << "ANOMALY Decisions:       " << count_anomaly << "\n"
                      << "  - True Signal Flagged:   " << anomaly_sig << "\n"
                      << "  - False Alarm (Bg):      " << anomaly_bg << "\n"
                      << "===============================================================\n";

            // Write first 100 errors to output file with %.9g precision
            fs::path out_file_path(opts.out_path);
            if (out_file_path.has_parent_path()) {
                fs::create_directories(out_file_path.parent_path());
            }

            std::ofstream out_file(out_file_path);
            if (!out_file.is_open()) {
                std::cerr << "Error: Could not open output file for writing: " << opts.out_path << "\n";
                return 1;
            }

            size_t write_count = std::min<size_t>(recorded_errors.size(), 100);
            for (size_t i = 0; i < write_count; ++i) {
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%.9g", static_cast<double>(recorded_errors[i]));
                out_file << buf << "\n";
            }
            out_file.close();

            std::cout << "Wrote " << write_count << " errors to: " << opts.out_path << "\n";

        } else {
            // 6. Benchmark Mode execution
            std::cout << "===============================================================\n"
                      << "BENCHMARK CONFIGURATION\n"
                      << "===============================================================\n"
                      << "Warmup Iterations:   " << opts.warmup_runs << "\n"
                      << "Timed Iterations:    " << opts.timed_runs << "\n"
                      << "Distinct Input Pool: " << events.size() << " events\n"
                      << "Timing Clock:        std::chrono::steady_clock (pure Run() call)\n"
                      << "===============================================================\n";

            // Pre-create input tensors for all loaded events to isolate pure inference latency
            std::vector<Ort::Value> precreated_tensors;
            precreated_tensors.reserve(events.size());
            for (auto& ev : events) {
                precreated_tensors.push_back(Ort::Value::CreateTensor<float>(
                    memory_info,
                    ev.features.data(),
                    ev.features.size(),
                    tensor_shape.data(),
                    tensor_shape.size()
                ));
            }

            // Warmup loop (untimed)
            std::cout << "Executing " << opts.warmup_runs << " warmup iterations...\n";
            for (size_t w = 0; w < opts.warmup_runs; ++w) {
                const auto& input_tensor = precreated_tensors[w % events.size()];
                auto output = session.Run(
                    Ort::RunOptions{nullptr},
                    input_names,
                    &input_tensor,
                    1,
                    output_names,
                    1
                );
                (void)output;
            }

            // Timed loop
            std::cout << "Executing " << opts.timed_runs << " timed iterations...\n";
            std::vector<double> latencies_us;
            latencies_us.reserve(opts.timed_runs);

            for (size_t r = 0; r < opts.timed_runs; ++r) {
                const auto& input_tensor = precreated_tensors[r % events.size()];

                auto t0 = std::chrono::steady_clock::now();
                auto output = session.Run(
                    Ort::RunOptions{nullptr},
                    input_names,
                    &input_tensor,
                    1,
                    output_names,
                    1
                );
                auto t1 = std::chrono::steady_clock::now();

                double elapsed_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
                latencies_us.push_back(elapsed_us);
                (void)output;
            }

            // Compute metrics
            BenchmarkStats stats = compute_benchmark_statistics(latencies_us);

            std::cout << "\n===============================================================\n"
                      << "BENCHMARK RESULTS (LATENCY IN MICROSECONDS)\n"
                      << "===============================================================\n"
                      << "Mean Latency:    " << std::setw(8) << std::fixed << std::setprecision(2) << stats.mean_us << " us\n"
                      << "Median Latency:  " << std::setw(8) << std::fixed << std::setprecision(2) << stats.median_us << " us\n"
                      << "p99 Latency:     " << std::setw(8) << std::fixed << std::setprecision(2) << stats.p99_us << " us\n"
                      << "Min Latency:     " << std::setw(8) << std::fixed << std::setprecision(2) << stats.min_us << " us\n"
                      << "Max Latency:     " << std::setw(8) << std::fixed << std::setprecision(2) << stats.max_us << " us\n"
                      << "Throughput:      " << std::setw(8) << std::fixed << std::setprecision(1) << (1e6 / stats.mean_us) << " events/sec\n"
                      << "===============================================================\n";

            // Write JSON artifact
            write_benchmark_json(
                opts.bench_json,
                stats,
                opts.warmup_runs,
                opts.timed_runs,
                events.size()
            );
            std::cout << "Saved benchmark metrics JSON to: " << opts.bench_json << "\n";
        }

    } catch (const Ort::Exception& oe) {
        std::cerr << "ONNX Runtime Exception: " << oe.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
