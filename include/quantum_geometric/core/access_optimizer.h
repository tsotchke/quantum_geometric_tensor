#ifndef ACCESS_OPTIMIZER_H
#define ACCESS_OPTIMIZER_H

#include <stdbool.h>
#include <stddef.h>
#include "quantum_geometric/core/access_history.h"

// Optimization strategy types
typedef enum {
    OPT_STRATEGY_CONSERVATIVE,  // Conservative optimization
    OPT_STRATEGY_BALANCED,      // Balanced optimization
    OPT_STRATEGY_AGGRESSIVE,    // Aggressive optimization
    OPT_STRATEGY_ADAPTIVE       // Adaptive optimization
} optimization_strategy_t;

// Cache optimization levels
typedef enum {
    CACHE_OPT_NONE,            // No cache optimization
    CACHE_OPT_MINIMAL,         // Minimal cache optimization
    CACHE_OPT_MODERATE,        // Moderate cache optimization
    CACHE_OPT_AGGRESSIVE       // Aggressive cache optimization
} cache_optimization_t;

// Prefetch strategies
typedef enum {
    PREFETCH_NONE,             // No prefetching
    PREFETCH_SEQUENTIAL,       // Sequential prefetching
    PREFETCH_ADAPTIVE,         // Adaptive prefetching
    PREFETCH_PREDICTIVE       // Predictive prefetching
} prefetch_strategy_t;

// Memory layout optimization
typedef enum {
    LAYOUT_NONE,               // No layout optimization
    LAYOUT_SEQUENTIAL,         // Sequential layout
    LAYOUT_BLOCKED,            // Blocked layout
    LAYOUT_CUSTOM             // Custom layout
} memory_layout_t;

// Optimizer configuration
typedef struct {
    optimization_strategy_t strategy;    // Overall optimization strategy
    cache_optimization_t cache_level;    // Cache optimization level
    prefetch_strategy_t prefetch;        // Prefetch strategy
    memory_layout_t layout;              // Memory layout strategy
    size_t cache_line_size;             // Cache line size
    size_t prefetch_distance;           // Prefetch distance
    bool enable_monitoring;             // Enable performance monitoring
    bool enable_adaptation;             // Enable adaptive optimization
} optimizer_config_t;

// Performance metrics
typedef struct {
    double throughput;                  // Memory throughput
    double bandwidth;                   // Memory bandwidth
    double latency;                     // Average latency
    double cache_hit_rate;             // Cache hit rate
    double prefetch_accuracy;          // Prefetch accuracy
    double optimization_overhead;       // Optimization overhead
} optimizer_metrics_t;

// Optimization hints
typedef struct {
    access_pattern_t pattern;          // Access pattern hint
    memory_lifetime_t lifetime;        // Memory lifetime hint
    access_priority_t priority;        // Access priority hint
    size_t block_size;                // Block size hint
    bool cache_sensitive;             // Cache sensitivity hint
    bool latency_sensitive;           // Latency sensitivity hint
} optimizer_hints_t;

// Opaque optimizer handle
typedef struct access_optimizer_t access_optimizer_t;

// Core functions
access_optimizer_t* create_access_optimizer(const optimizer_config_t* config);
void destroy_access_optimizer(access_optimizer_t* optimizer);

// Optimization functions
bool optimize_access_pattern(access_optimizer_t* optimizer,
                           void* address,
                           size_t size,
                           const optimizer_hints_t* hints);
bool optimize_memory_layout(access_optimizer_t* optimizer,
                          void* address,
                          size_t size,
                          memory_layout_t layout);
bool optimize_cache_usage(access_optimizer_t* optimizer,
                         void* address,
                         size_t size,
                         cache_optimization_t level);

// Prefetch control
bool configure_prefetch(access_optimizer_t* optimizer,
                       prefetch_strategy_t strategy,
                       size_t distance);
bool prefetch_region(access_optimizer_t* optimizer,
                    void* address,
                    size_t size);
bool cancel_prefetch(access_optimizer_t* optimizer,
                    void* address,
                    size_t size);

// Performance monitoring
bool get_optimizer_metrics(const access_optimizer_t* optimizer,
                         optimizer_metrics_t* metrics);
bool reset_optimizer_metrics(access_optimizer_t* optimizer);

// Integration with access history
bool apply_history_insights(access_optimizer_t* optimizer,
                          const access_history_t* history);
bool update_optimization_strategy(access_optimizer_t* optimizer,
                                const access_history_t* history);
bool suggest_optimization_hints(const access_optimizer_t* optimizer,
                              const access_history_t* history,
                              optimizer_hints_t* hints);

// Adaptation and tuning
bool adapt_optimization_strategy(access_optimizer_t* optimizer,
                               const optimizer_metrics_t* metrics);
bool tune_prefetch_parameters(access_optimizer_t* optimizer,
                            const optimizer_metrics_t* metrics);
bool adjust_cache_strategy(access_optimizer_t* optimizer,
                         const optimizer_metrics_t* metrics);

#endif // ACCESS_OPTIMIZER_H
