#ifndef ACCESS_HISTORY_H
#define ACCESS_HISTORY_H

#include <stddef.h>
#include <stdbool.h>
#include <time.h>

// Access pattern types
typedef enum {
    ACCESS_SEQUENTIAL,      // Sequential memory access
    ACCESS_RANDOM,         // Random memory access
    ACCESS_STRIDED,        // Strided memory access
    ACCESS_BLOCKED,        // Block-based memory access
    ACCESS_HYBRID          // Mixed access patterns
} access_pattern_t;

// Memory lifetime hints
typedef enum {
    LIFETIME_SHORT,        // Short-lived allocations
    LIFETIME_MEDIUM,       // Medium-lived allocations
    LIFETIME_LONG,         // Long-lived allocations
    LIFETIME_PERSISTENT    // Persistent allocations
} memory_lifetime_t;

// Access priority levels
typedef enum {
    PRIORITY_LOW,          // Low priority access
    PRIORITY_MEDIUM,       // Medium priority access
    PRIORITY_HIGH,         // High priority access
    PRIORITY_CRITICAL      // Critical priority access
} access_priority_t;

// Access statistics
typedef struct {
    size_t total_accesses;     // Total number of memory accesses
    size_t cache_hits;         // Number of cache hits
    size_t cache_misses;       // Number of cache misses
    double hit_rate;           // Cache hit rate
    double avg_latency;        // Average access latency
    size_t page_faults;        // Number of page faults
    size_t tlb_misses;         // Number of TLB misses
} access_stats_t;

// Access record entry
typedef struct {
    void* address;             // Memory address
    size_t size;              // Access size
    access_pattern_t pattern; // Access pattern
    access_priority_t priority; // Access priority
    struct timespec timestamp; // Access timestamp
    double latency;           // Access latency
} access_record_t;

// Access history configuration
typedef struct {
    size_t history_size;       // Maximum history entries
    bool track_latency;        // Track access latency
    bool track_patterns;       // Track access patterns
    bool track_priorities;     // Track access priorities
    size_t sampling_rate;      // Sampling rate (1 in N accesses)
} access_history_config_t;

// Access hints for optimization
typedef struct {
    access_pattern_t pattern;  // Expected access pattern
    memory_lifetime_t lifetime; // Expected memory lifetime
    access_priority_t priority; // Access priority
    size_t alignment;          // Required alignment
    bool prefetch;             // Enable prefetching
    bool cache_control;        // Enable cache control
} access_hints_t;

// Opaque access history handle
typedef struct access_history_t access_history_t;

// Core functions
access_history_t* create_access_history(const access_history_config_t* config);
void destroy_access_history(access_history_t* history);

// Recording functions
bool record_access(access_history_t* history,
                  void* address,
                  size_t size,
                  const access_hints_t* hints);
bool record_access_pattern(access_history_t* history,
                          access_pattern_t pattern,
                          const access_hints_t* hints);
bool record_access_latency(access_history_t* history,
                          void* address,
                          double latency);

// Analysis functions
bool analyze_access_patterns(const access_history_t* history,
                           access_pattern_t* patterns,
                           size_t* num_patterns);
bool analyze_access_hotspots(const access_history_t* history,
                            void** hotspots,
                            size_t* num_hotspots);
bool analyze_access_conflicts(const access_history_t* history,
                            void** conflicts,
                            size_t* num_conflicts);

// Statistics functions
bool get_access_stats(const access_history_t* history,
                     access_stats_t* stats);
bool get_pattern_stats(const access_history_t* history,
                      access_pattern_t pattern,
                      access_stats_t* stats);
bool reset_access_stats(access_history_t* history);

// Optimization functions
bool optimize_access_pattern(access_history_t* history,
                           access_pattern_t pattern);
bool suggest_access_hints(const access_history_t* history,
                         void* address,
                         access_hints_t* hints);
bool prefetch_based_on_history(const access_history_t* history,
                              void* address,
                              size_t size);

// Query functions
bool get_access_record(const access_history_t* history,
                      void* address,
                      access_record_t* record);
bool get_access_pattern(const access_history_t* history,
                       void* address,
                       access_pattern_t* pattern);
bool get_access_latency(const access_history_t* history,
                       void* address,
                       double* latency);

#endif // ACCESS_HISTORY_H
