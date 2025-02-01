#ifndef ALLOCATION_ANALYZER_H
#define ALLOCATION_ANALYZER_H

#include <stdbool.h>
#include <stddef.h>
#include <time.h>

// Allocation patterns
typedef enum {
    ALLOC_PATTERN_SEQUENTIAL,    // Sequential allocation pattern
    ALLOC_PATTERN_RANDOM,        // Random allocation pattern
    ALLOC_PATTERN_CYCLIC,        // Cyclic allocation pattern
    ALLOC_PATTERN_BURST         // Burst allocation pattern
} allocation_pattern_t;

// Allocation sizes
typedef enum {
    ALLOC_SIZE_SMALL,           // Small allocations (<1KB)
    ALLOC_SIZE_MEDIUM,          // Medium allocations (1KB-1MB)
    ALLOC_SIZE_LARGE,           // Large allocations (1MB-1GB)
    ALLOC_SIZE_HUGE            // Huge allocations (>1GB)
} allocation_size_t;

// Allocation lifetimes
typedef enum {
    ALLOC_LIFE_TEMPORARY,       // Temporary allocations
    ALLOC_LIFE_SHORT,           // Short-lived allocations
    ALLOC_LIFE_MEDIUM,          // Medium-lived allocations
    ALLOC_LIFE_LONG            // Long-lived allocations
} allocation_lifetime_t;

// Analysis modes
typedef enum {
    ANALYSIS_REALTIME,          // Real-time analysis
    ANALYSIS_BATCH,             // Batch analysis
    ANALYSIS_PERIODIC,          // Periodic analysis
    ANALYSIS_TRIGGERED         // Triggered analysis
} analysis_mode_t;

// Analyzer configuration
typedef struct {
    analysis_mode_t mode;               // Analysis mode
    size_t sample_interval;             // Sampling interval
    size_t history_size;                // History size
    bool track_patterns;                // Track patterns
    bool track_lifetimes;               // Track lifetimes
    bool track_fragmentation;           // Track fragmentation
    bool enable_prediction;             // Enable prediction
} analyzer_config_t;

// Allocation record
typedef struct {
    void* address;                      // Allocation address
    size_t size;                        // Allocation size
    allocation_pattern_t pattern;       // Allocation pattern
    allocation_lifetime_t lifetime;     // Allocation lifetime
    struct timespec allocation_time;    // Allocation time
    struct timespec deallocation_time;  // Deallocation time
    void* caller_context;               // Caller context
} allocation_record_t;

// Allocation statistics
typedef struct {
    size_t total_allocations;           // Total allocations
    size_t active_allocations;          // Active allocations
    size_t peak_allocations;            // Peak allocations
    size_t total_bytes;                 // Total bytes allocated
    size_t active_bytes;                // Active bytes
    size_t peak_bytes;                  // Peak bytes
    double fragmentation_ratio;         // Fragmentation ratio
} allocation_stats_t;

// Pattern analysis
typedef struct {
    allocation_pattern_t pattern;       // Detected pattern
    double confidence;                  // Pattern confidence
    size_t frequency;                   // Pattern frequency
    size_t sequence_length;             // Pattern sequence length
    void* pattern_data;                 // Pattern-specific data
} pattern_analysis_t;

// Lifetime analysis
typedef struct {
    allocation_lifetime_t lifetime;     // Lifetime category
    double average_lifetime;            // Average lifetime
    double min_lifetime;                // Minimum lifetime
    double max_lifetime;                // Maximum lifetime
    size_t allocation_count;            // Number of allocations
} lifetime_analysis_t;

// Fragmentation analysis
typedef struct {
    double internal_fragmentation;      // Internal fragmentation
    double external_fragmentation;      // External fragmentation
    size_t free_blocks;                // Number of free blocks
    size_t largest_free_block;         // Largest free block
    double compactness_ratio;          // Memory compactness
} fragmentation_analysis_t;

// Opaque analyzer handle
typedef struct allocation_analyzer_t allocation_analyzer_t;

// Core functions
allocation_analyzer_t* create_allocation_analyzer(const analyzer_config_t* config);
void destroy_allocation_analyzer(allocation_analyzer_t* analyzer);

// Analysis functions
bool analyze_allocation_pattern(allocation_analyzer_t* analyzer,
                              pattern_analysis_t* pattern);
bool analyze_allocation_lifetime(allocation_analyzer_t* analyzer,
                               lifetime_analysis_t* lifetime);
bool analyze_fragmentation(allocation_analyzer_t* analyzer,
                          fragmentation_analysis_t* fragmentation);

// Recording functions
bool record_allocation(allocation_analyzer_t* analyzer,
                      const allocation_record_t* record);
bool record_deallocation(allocation_analyzer_t* analyzer,
                        void* address,
                        struct timespec* time);
bool update_allocation_stats(allocation_analyzer_t* analyzer,
                           allocation_stats_t* stats);

// Query functions
allocation_record_t* get_allocation_record(const allocation_analyzer_t* analyzer,
                                         void* address);
bool get_allocation_stats(const allocation_analyzer_t* analyzer,
                         allocation_stats_t* stats);
size_t get_allocation_history(const allocation_analyzer_t* analyzer,
                            allocation_record_t** records,
                            size_t max_records);

// Prediction functions
bool predict_allocation_pattern(const allocation_analyzer_t* analyzer,
                              pattern_analysis_t* prediction);
bool predict_memory_usage(const allocation_analyzer_t* analyzer,
                         size_t* predicted_bytes,
                         struct timespec* prediction_time);
bool validate_predictions(const allocation_analyzer_t* analyzer,
                         const pattern_analysis_t* prediction,
                         const pattern_analysis_t* actual);

// Utility functions
bool reset_analyzer_stats(allocation_analyzer_t* analyzer);
bool export_analyzer_data(const allocation_analyzer_t* analyzer,
                         const char* filename);
bool import_analyzer_data(allocation_analyzer_t* analyzer,
                         const char* filename);

#endif // ALLOCATION_ANALYZER_H
