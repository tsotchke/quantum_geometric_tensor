#ifndef BALANCE_ANALYZER_H
#define BALANCE_ANALYZER_H

#include <stdbool.h>
#include <stddef.h>
#include <time.h>

// Balance metrics
typedef enum {
    BALANCE_CPU_LOAD,          // CPU load balance
    BALANCE_MEMORY_USAGE,      // Memory usage balance
    BALANCE_NETWORK_LOAD,      // Network load balance
    BALANCE_IO_LOAD           // I/O load balance
} balance_metric_t;

// Distribution strategies
typedef enum {
    DIST_STRATEGY_ROUND_ROBIN,  // Round-robin distribution
    DIST_STRATEGY_LEAST_LOADED, // Least loaded first
    DIST_STRATEGY_WEIGHTED,     // Weighted distribution
    DIST_STRATEGY_ADAPTIVE     // Adaptive distribution
} distribution_strategy_t;

// Balance thresholds
typedef enum {
    THRESHOLD_STRICT,          // Strict balance threshold
    THRESHOLD_MODERATE,        // Moderate balance threshold
    THRESHOLD_RELAXED,         // Relaxed balance threshold
    THRESHOLD_ADAPTIVE        // Adaptive threshold
} threshold_type_t;

// Resource types
typedef enum {
    RESOURCE_CPU,              // CPU resources
    RESOURCE_MEMORY,           // Memory resources
    RESOURCE_NETWORK,          // Network resources
    RESOURCE_STORAGE          // Storage resources
} resource_type_t;

// Analyzer configuration
typedef struct {
    balance_metric_t metrics[4];        // Metrics to analyze
    size_t num_metrics;                 // Number of metrics
    distribution_strategy_t strategy;    // Distribution strategy
    threshold_type_t threshold;         // Balance threshold
    size_t sample_interval;             // Sampling interval
    bool enable_prediction;             // Enable prediction
    bool enable_adaptation;             // Enable adaptation
} analyzer_config_t;

// Node statistics
typedef struct {
    size_t node_id;                     // Node identifier
    double cpu_usage;                   // CPU usage
    size_t memory_usage;                // Memory usage
    double network_bandwidth;           // Network bandwidth
    double io_throughput;               // I/O throughput
    size_t active_tasks;                // Active tasks
    double load_factor;                 // Load factor
} node_stats_t;

// Balance metrics
typedef struct {
    double cpu_imbalance;               // CPU imbalance ratio
    double memory_imbalance;            // Memory imbalance ratio
    double network_imbalance;           // Network imbalance ratio
    double io_imbalance;                // I/O imbalance ratio
    double overall_imbalance;           // Overall imbalance
} balance_metrics_t;

// Resource distribution
typedef struct {
    resource_type_t type;               // Resource type
    size_t total_amount;                // Total amount
    size_t available_amount;            // Available amount
    double utilization;                 // Utilization ratio
    size_t num_consumers;               // Number of consumers
} resource_distribution_t;

// Load prediction
typedef struct {
    double predicted_load;              // Predicted load
    struct timespec prediction_time;    // Prediction timestamp
    double confidence;                  // Prediction confidence
    void* prediction_data;              // Additional data
} load_prediction_t;

// Opaque analyzer handle
typedef struct balance_analyzer_t balance_analyzer_t;

// Core functions
balance_analyzer_t* create_balance_analyzer(const analyzer_config_t* config);
void destroy_balance_analyzer(balance_analyzer_t* analyzer);

// Analysis functions
bool analyze_system_balance(balance_analyzer_t* analyzer,
                          balance_metrics_t* metrics);
bool analyze_resource_distribution(balance_analyzer_t* analyzer,
                                 resource_distribution_t* distribution);
bool analyze_node_balance(balance_analyzer_t* analyzer,
                         size_t node_id,
                         node_stats_t* stats);

// Monitoring functions
bool update_node_stats(balance_analyzer_t* analyzer,
                      const node_stats_t* stats);
bool get_node_stats(const balance_analyzer_t* analyzer,
                   size_t node_id,
                   node_stats_t* stats);
bool get_system_metrics(const balance_analyzer_t* analyzer,
                       balance_metrics_t* metrics);

// Distribution functions
bool suggest_distribution(balance_analyzer_t* analyzer,
                        resource_type_t resource,
                        size_t* target_node);
bool validate_distribution(balance_analyzer_t* analyzer,
                         const resource_distribution_t* distribution);
bool optimize_distribution(balance_analyzer_t* analyzer,
                         distribution_strategy_t strategy);

// Prediction functions
bool predict_system_load(balance_analyzer_t* analyzer,
                        load_prediction_t* prediction);
bool predict_resource_usage(balance_analyzer_t* analyzer,
                          resource_type_t resource,
                          size_t* predicted_usage);
bool validate_predictions(balance_analyzer_t* analyzer,
                        const load_prediction_t* prediction,
                        const node_stats_t* actual);

// Adaptation functions
bool adapt_distribution_strategy(balance_analyzer_t* analyzer,
                               const balance_metrics_t* metrics);
bool adapt_thresholds(balance_analyzer_t* analyzer,
                     const balance_metrics_t* metrics);
bool adapt_sampling_rate(balance_analyzer_t* analyzer,
                        const balance_metrics_t* metrics);

// Utility functions
bool reset_analyzer_stats(balance_analyzer_t* analyzer);
bool export_balance_data(const balance_analyzer_t* analyzer,
                        const char* filename);
bool import_balance_data(balance_analyzer_t* analyzer,
                        const char* filename);

#endif // BALANCE_ANALYZER_H
