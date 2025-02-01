#ifndef DISTRIBUTION_ANALYZER_H
#define DISTRIBUTION_ANALYZER_H

#include <stdbool.h>
#include <stddef.h>
#include <time.h>

// Distribution types
typedef enum {
    DIST_WORKLOAD,          // Workload distribution
    DIST_RESOURCE,          // Resource distribution
    DIST_DATA,              // Data distribution
    DIST_QUANTUM,           // Quantum state distribution
    DIST_NETWORK           // Network distribution
} distribution_type_t;

// Analysis modes
typedef enum {
    ANALYZE_STATIC,         // Static analysis
    ANALYZE_DYNAMIC,        // Dynamic analysis
    ANALYZE_PREDICTIVE,     // Predictive analysis
    ANALYZE_ADAPTIVE       // Adaptive analysis
} analysis_mode_t;

// Distribution patterns
typedef enum {
    PATTERN_UNIFORM,        // Uniform distribution
    PATTERN_WEIGHTED,       // Weighted distribution
    PATTERN_LOCALITY,       // Locality-based distribution
    PATTERN_ADAPTIVE       // Adaptive distribution
} distribution_pattern_t;

// Node types
typedef enum {
    NODE_COMPUTE,           // Compute node
    NODE_STORAGE,           // Storage node
    NODE_QUANTUM,           // Quantum node
    NODE_HYBRID            // Hybrid node
} node_type_t;

// Analyzer configuration
typedef struct {
    analysis_mode_t mode;           // Analysis mode
    size_t window_size;             // Analysis window size
    bool enable_prediction;         // Enable prediction
    bool track_history;             // Track history
    bool enable_optimization;       // Enable optimization
    size_t sample_interval;         // Sampling interval
} analyzer_config_t;

// Node statistics
typedef struct {
    node_type_t type;               // Node type
    size_t node_id;                 // Node identifier
    double load_factor;             // Load factor
    double utilization;             // Resource utilization
    size_t active_tasks;            // Active tasks
    double performance_score;       // Performance score
} node_stats_t;

// Distribution metrics
typedef struct {
    distribution_type_t type;       // Distribution type
    distribution_pattern_t pattern; // Distribution pattern
    double balance_score;           // Balance score
    double efficiency_score;        // Efficiency score
    double locality_score;          // Locality score
    double overhead;               // Distribution overhead
} distribution_metrics_t;

// Workload characteristics
typedef struct {
    size_t total_tasks;             // Total tasks
    size_t completed_tasks;         // Completed tasks
    double average_duration;        // Average task duration
    double variance;                // Duration variance
    size_t dependencies;            // Task dependencies
    double priority_score;         // Priority score
} workload_stats_t;

// Distribution prediction
typedef struct {
    distribution_pattern_t pattern; // Predicted pattern
    double confidence;             // Prediction confidence
    struct timespec timestamp;     // Prediction timestamp
    void* prediction_data;         // Additional data
} distribution_prediction_t;

// Opaque analyzer handle
typedef struct distribution_analyzer_t distribution_analyzer_t;

// Core functions
distribution_analyzer_t* create_distribution_analyzer(const analyzer_config_t* config);
void destroy_distribution_analyzer(distribution_analyzer_t* analyzer);

// Analysis functions
bool analyze_distribution(distribution_analyzer_t* analyzer,
                         distribution_metrics_t* metrics);
bool analyze_node_distribution(distribution_analyzer_t* analyzer,
                             node_stats_t* stats,
                             size_t num_nodes);
bool analyze_workload_distribution(distribution_analyzer_t* analyzer,
                                 workload_stats_t* stats);

// Node management
bool register_node(distribution_analyzer_t* analyzer,
                  const node_stats_t* node);
bool unregister_node(distribution_analyzer_t* analyzer,
                    size_t node_id);
bool update_node_stats(distribution_analyzer_t* analyzer,
                      const node_stats_t* stats);

// Workload tracking
bool track_workload(distribution_analyzer_t* analyzer,
                   const workload_stats_t* stats);
bool update_workload_stats(distribution_analyzer_t* analyzer,
                          const workload_stats_t* stats);
bool get_workload_history(const distribution_analyzer_t* analyzer,
                         workload_stats_t* history,
                         size_t* num_entries);

// Pattern analysis
bool detect_distribution_pattern(distribution_analyzer_t* analyzer,
                               distribution_pattern_t* pattern);
bool validate_distribution_pattern(distribution_analyzer_t* analyzer,
                                 const distribution_pattern_t pattern);
bool optimize_distribution_pattern(distribution_analyzer_t* analyzer,
                                 distribution_pattern_t* pattern);

// Prediction functions
bool predict_distribution(distribution_analyzer_t* analyzer,
                         distribution_prediction_t* prediction);
bool validate_prediction(distribution_analyzer_t* analyzer,
                        const distribution_prediction_t* prediction,
                        const distribution_metrics_t* actual);
bool update_prediction_model(distribution_analyzer_t* analyzer,
                           const distribution_metrics_t* metrics);

// Quantum-specific functions
bool analyze_quantum_distribution(distribution_analyzer_t* analyzer,
                                distribution_metrics_t* metrics);
bool optimize_quantum_distribution(distribution_analyzer_t* analyzer,
                                 distribution_pattern_t* pattern);
bool validate_quantum_distribution(distribution_analyzer_t* analyzer,
                                 const distribution_metrics_t* metrics);

// Utility functions
bool export_analyzer_data(const distribution_analyzer_t* analyzer,
                         const char* filename);
bool import_analyzer_data(distribution_analyzer_t* analyzer,
                         const char* filename);
void free_distribution_metrics(distribution_metrics_t* metrics);

#endif // DISTRIBUTION_ANALYZER_H
