#ifndef PERFORMANCE_ANALYZER_H
#define PERFORMANCE_ANALYZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Performance Analyzer Types
// ============================================================================

// Timer precision levels
typedef enum {
    TIMER_PRECISION_NANOSECOND,
    TIMER_PRECISION_MICROSECOND,
    TIMER_PRECISION_MILLISECOND,
    TIMER_PRECISION_SECOND
} timer_precision_t;

// Cache level identifiers
typedef enum {
    CACHE_L1_DATA,
    CACHE_L1_INSTRUCTION,
    CACHE_L2,
    CACHE_L3,
    CACHE_LEVEL_COUNT
} cache_level_t;

// Performance metric types
typedef enum {
    METRIC_WALL_TIME,
    METRIC_CPU_TIME,
    METRIC_CPU_CYCLES,
    METRIC_INSTRUCTIONS,
    METRIC_CACHE_MISSES,
    METRIC_CACHE_REFERENCES,
    METRIC_BRANCH_MISSES,
    METRIC_PAGE_FAULTS,
    METRIC_CONTEXT_SWITCHES,
    METRIC_MEMORY_BANDWIDTH,
    METRIC_FLOPS,
    METRIC_TYPE_COUNT
} perf_metric_type_t;

// Export formats
typedef enum {
    EXPORT_FORMAT_JSON,
    EXPORT_FORMAT_CSV,
    EXPORT_FORMAT_BINARY,
    EXPORT_FORMAT_FLAMEGRAPH
} export_format_t;

// Performance sample
typedef struct {
    uint64_t timestamp_ns;           // Nanosecond timestamp
    double value;                    // Metric value
    perf_metric_type_t metric_type;  // Type of metric
    uint32_t thread_id;              // Thread ID
    const char* label;               // Optional label
} perf_sample_t;

// Timer instance
typedef struct {
    uint64_t start_time_ns;
    uint64_t end_time_ns;
    uint64_t elapsed_ns;
    uint64_t cpu_cycles_start;
    uint64_t cpu_cycles_end;
    timer_precision_t precision;
    bool is_running;
    const char* name;
} perf_timer_t;

// Cache statistics
typedef struct {
    uint64_t references;        // Total cache references
    uint64_t misses;            // Cache misses
    uint64_t hits;              // Cache hits (references - misses)
    double hit_rate;            // Hit rate (0.0 - 1.0)
    double miss_rate;           // Miss rate (0.0 - 1.0)
    uint64_t bytes_read;        // Bytes read from cache
    uint64_t bytes_written;     // Bytes written to cache
} cache_stats_t;

// Memory bandwidth statistics
typedef struct {
    double read_bandwidth_gbps;    // Read bandwidth in GB/s
    double write_bandwidth_gbps;   // Write bandwidth in GB/s
    double total_bandwidth_gbps;   // Total bandwidth in GB/s
    uint64_t bytes_read;           // Total bytes read
    uint64_t bytes_written;        // Total bytes written
    double measurement_time_sec;   // Measurement duration
} memory_bandwidth_t;

// FLOPS statistics
typedef struct {
    double gflops;              // Billions of floating-point ops per second
    uint64_t total_flops;       // Total floating-point operations
    double measurement_time_sec; // Measurement duration
    double efficiency;          // Percentage of theoretical peak
    double theoretical_peak_gflops; // Theoretical peak for this hardware
} flops_stats_t;

// Profiling region
typedef struct {
    const char* name;
    const char* file;
    int line;
    uint64_t call_count;
    uint64_t total_time_ns;
    uint64_t min_time_ns;
    uint64_t max_time_ns;
    double avg_time_ns;
    double std_dev_ns;
    uint64_t total_cpu_cycles;
    struct {
        cache_stats_t cache[CACHE_LEVEL_COUNT];
    } cache_stats;
    uint64_t self_time_ns;       // Time excluding children
    uint64_t children_time_ns;   // Time in child regions
} profiling_region_t;

// Call stack entry for flame graph
typedef struct {
    const char* function_name;
    uint64_t sample_count;
    uint64_t self_samples;
    struct call_stack_entry* children;
    size_t num_children;
    size_t children_capacity;
} call_stack_entry_t;

// Aggregated statistics
typedef struct {
    double mean;
    double median;
    double std_dev;
    double variance;
    double min;
    double max;
    double p50;   // 50th percentile
    double p90;   // 90th percentile
    double p95;   // 95th percentile
    double p99;   // 99th percentile
    uint64_t count;
    double sum;
} aggregated_stats_t;

// Analyzer configuration
typedef struct {
    timer_precision_t default_precision;
    size_t max_samples;
    size_t max_regions;
    bool enable_cpu_cycles;
    bool enable_cache_counters;
    bool enable_memory_bandwidth;
    bool enable_flops_counting;
    bool enable_flamegraph;
    size_t sample_buffer_size;
    double sample_rate_hz;        // Sampling rate in Hz
    bool thread_safe;
} perf_analyzer_config_t;

// Opaque handle for performance analyzer
typedef struct performance_analyzer performance_analyzer_t;

// ============================================================================
// Initialization and Configuration
// ============================================================================

// Create a new performance analyzer with default settings
performance_analyzer_t* perf_analyzer_create(void);

// Create a performance analyzer with custom configuration
performance_analyzer_t* perf_analyzer_create_with_config(
    const perf_analyzer_config_t* config);

// Get default configuration
perf_analyzer_config_t perf_analyzer_default_config(void);

// Destroy performance analyzer and free resources
void perf_analyzer_destroy(performance_analyzer_t* analyzer);

// Reset all collected data
bool perf_analyzer_reset(performance_analyzer_t* analyzer);

// ============================================================================
// High-Resolution Timing
// ============================================================================

// Create a named timer
perf_timer_t perf_timer_create(const char* name, timer_precision_t precision);

// Start a timer
void perf_timer_start(perf_timer_t* timer);

// Stop a timer and return elapsed time in nanoseconds
uint64_t perf_timer_stop(perf_timer_t* timer);

// Get elapsed time without stopping
uint64_t perf_timer_elapsed_ns(const perf_timer_t* timer);

// Get current high-resolution timestamp in nanoseconds
uint64_t perf_get_timestamp_ns(void);

// Get CPU clock frequency in Hz
uint64_t perf_get_cpu_frequency(void);

// ============================================================================
// CPU Cycle Counting
// ============================================================================

// Read current CPU cycle counter
uint64_t perf_read_cpu_cycles(void);

// Start cycle counting for a region
void perf_cycles_begin(performance_analyzer_t* analyzer, const char* region);

// End cycle counting and return cycles elapsed
uint64_t perf_cycles_end(performance_analyzer_t* analyzer, const char* region);

// Get cycles per operation
double perf_get_cycles_per_op(uint64_t cycles, uint64_t operations);

// ============================================================================
// Cache Analysis
// ============================================================================

// Start cache monitoring (requires root/admin on some platforms)
bool perf_cache_monitoring_start(performance_analyzer_t* analyzer);

// Stop cache monitoring
void perf_cache_monitoring_stop(performance_analyzer_t* analyzer);

// Get cache statistics for a specific level
bool perf_get_cache_stats(performance_analyzer_t* analyzer,
                          cache_level_t level,
                          cache_stats_t* stats);

// Get all cache statistics
bool perf_get_all_cache_stats(performance_analyzer_t* analyzer,
                              cache_stats_t stats[CACHE_LEVEL_COUNT]);

// Reset cache counters
void perf_reset_cache_counters(performance_analyzer_t* analyzer);

// ============================================================================
// Memory Bandwidth Measurement
// ============================================================================

// Start memory bandwidth monitoring
bool perf_bandwidth_monitoring_start(performance_analyzer_t* analyzer);

// Stop memory bandwidth monitoring
void perf_bandwidth_monitoring_stop(performance_analyzer_t* analyzer);

// Get current memory bandwidth statistics
bool perf_get_memory_bandwidth(performance_analyzer_t* analyzer,
                               memory_bandwidth_t* bandwidth);

// Benchmark memory bandwidth with specified buffer size
bool perf_benchmark_memory_bandwidth(size_t buffer_size,
                                     memory_bandwidth_t* result);

// ============================================================================
// FLOPS Measurement
// ============================================================================

// Start FLOPS counting
bool perf_flops_counting_start(performance_analyzer_t* analyzer);

// Stop FLOPS counting
void perf_flops_counting_stop(performance_analyzer_t* analyzer);

// Record floating-point operations manually
void perf_record_flops(performance_analyzer_t* analyzer, uint64_t flops);

// Get FLOPS statistics
bool perf_get_flops_stats(performance_analyzer_t* analyzer,
                          flops_stats_t* stats);

// Get theoretical peak FLOPS for current hardware
double perf_get_theoretical_peak_gflops(void);

// ============================================================================
// Profiling Regions
// ============================================================================

// Begin a profiling region
void perf_region_begin(performance_analyzer_t* analyzer,
                       const char* name,
                       const char* file,
                       int line);

// End a profiling region
void perf_region_end(performance_analyzer_t* analyzer, const char* name);

// Get profiling region statistics
bool perf_get_region_stats(performance_analyzer_t* analyzer,
                           const char* name,
                           profiling_region_t* region);

// Get all profiling regions
bool perf_get_all_regions(performance_analyzer_t* analyzer,
                          profiling_region_t** regions,
                          size_t* count);

// Convenience macro for region profiling
#define PERF_REGION_BEGIN(analyzer, name) \
    perf_region_begin(analyzer, name, __FILE__, __LINE__)
#define PERF_REGION_END(analyzer, name) \
    perf_region_end(analyzer, name)

// ============================================================================
// Sample Collection
// ============================================================================

// Record a performance sample
void perf_record_sample(performance_analyzer_t* analyzer,
                        perf_metric_type_t type,
                        double value,
                        const char* label);

// Get samples for a specific metric type
bool perf_get_samples(performance_analyzer_t* analyzer,
                      perf_metric_type_t type,
                      perf_sample_t** samples,
                      size_t* count);

// Clear all samples
void perf_clear_samples(performance_analyzer_t* analyzer);

// ============================================================================
// Statistics and Aggregation
// ============================================================================

// Compute aggregated statistics for a metric
bool perf_compute_stats(performance_analyzer_t* analyzer,
                        perf_metric_type_t type,
                        aggregated_stats_t* stats);

// Compute statistics for a profiling region
bool perf_compute_region_stats(performance_analyzer_t* analyzer,
                               const char* region_name,
                               aggregated_stats_t* stats);

// Get percentile value
double perf_get_percentile(const double* values, size_t count, double percentile);

// ============================================================================
// Flame Graph Generation
// ============================================================================

// Start flame graph data collection
bool perf_flamegraph_start(performance_analyzer_t* analyzer);

// Stop flame graph data collection
void perf_flamegraph_stop(performance_analyzer_t* analyzer);

// Get flame graph root entry
bool perf_get_flamegraph_data(performance_analyzer_t* analyzer,
                              call_stack_entry_t** root);

// Generate flame graph SVG
bool perf_export_flamegraph_svg(performance_analyzer_t* analyzer,
                                const char* filename,
                                size_t width,
                                size_t height);

// Free flame graph data
void perf_free_flamegraph_data(call_stack_entry_t* root);

// ============================================================================
// Export Functions
// ============================================================================

// Export all profiling data to file
bool perf_export_data(performance_analyzer_t* analyzer,
                      const char* filename,
                      export_format_t format);

// Export profiling data to JSON string (caller must free)
char* perf_export_to_json(performance_analyzer_t* analyzer);

// Export profiling data to CSV string (caller must free)
char* perf_export_to_csv(performance_analyzer_t* analyzer);

// Import profiling data from file
bool perf_import_data(performance_analyzer_t* analyzer,
                      const char* filename,
                      export_format_t format);

// ============================================================================
// Utility Functions
// ============================================================================

// Get string name for metric type
const char* perf_metric_type_name(perf_metric_type_t type);

// Get string name for cache level
const char* perf_cache_level_name(cache_level_t level);

// Format duration for display
char* perf_format_duration(uint64_t nanoseconds);

// Format bytes for display (e.g., "1.5 GB")
char* perf_format_bytes(uint64_t bytes);

// Check if hardware performance counters are available
bool perf_hw_counters_available(void);

// Get last error message
const char* perf_get_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // PERFORMANCE_ANALYZER_H
